#!/usr/bin/env python3
"""Training script for audio-language models using Hydra configuration.

Supports two training modes:
- Transcription (ASR): +experiments=transcription
- SIFT (AZeroS-style): +experiments=omni

Usage:
    poetry run python scripts/train.py +experiments=transcription
    poetry run python scripts/train.py +experiments=omni
"""

import contextlib
import os
import random
from dataclasses import fields
from typing import Any

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
# Use soundfile for audio decoding instead of torchaudio (avoids compatibility issues)
os.environ.setdefault("HF_DATASETS_AUDIO_DECODER", "soundfile")
# Allow torch.compile to capture .item() calls (used by TRL packing internals)
os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

import hydra  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from datasets import (  # noqa: E402
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl.experimental.utils import DataCollatorForChatML  # noqa: E402

from tiny_audio.asr_config import ASRConfig  # noqa: E402
from tiny_audio.asr_modeling import ASRModel  # noqa: E402

# Prompts for ASR training (transcription task)
# Target: lowercased verbatim transcription of speech
TRANSCRIBE_PROMPTS = [
    "Transcribe: ",
]

# SIFT instructions by mode (matching AZeroS exactly)
# - sift_s / sift_ssp: No instruction (empty string)
# - sit_ssp: Fixed instruction to describe audio
SIFT_INSTRUCTIONS = {
    "sift_s": "",  # No instruction - conversational response
    "sift_ssp": "",  # No instruction - empathetic response
    "sit_ssp": "Describe all information you can hear: ",  # Fixed description instruction
}


class DatasetLoader:
    """Loads and prepares datasets for training."""

    def __init__(
        self,
        config: DictConfig,
        sift_enabled: bool = False,
        s2s_enabled: bool = False,
        tts_enabled: bool = False,
    ):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)
        self.sift_enabled = sift_enabled
        self.s2s_enabled = s2s_enabled
        self.tts_enabled = tts_enabled

    def _prepare_split(self, dataset_cfg: DictConfig, split: str):
        dataset_path = dataset_cfg.get("path")
        if not dataset_path:
            raise ValueError("Dataset path is required")

        ds = load_dataset(
            dataset_path,
            name=dataset_cfg.get("name"),
            split=split,
            cache_dir=self.cache_dir,
            num_proc=self.num_proc,
            trust_remote_code=True,
        )

        # Filter by column value if specified (e.g., single speaker training)
        filter_column = dataset_cfg.get("filter_column")
        filter_value = dataset_cfg.get("filter_value")
        if filter_column and filter_value:
            ds = ds.filter(
                lambda x: str(x[filter_column]) == str(filter_value),
                num_proc=self.num_proc,
            )

        # Apply text transform if specified (e.g., extract text from nested structures)
        text_transform = dataset_cfg.get("text_transform")
        if text_transform == "first_answer_text":
            # Extract text from SQuAD-style answers: [{"answer_start": N, "text": "..."}]
            text_column = dataset_cfg.get("text_column", "text")
            ds = ds.map(
                lambda x: {"text": x[text_column][0]["text"] if x[text_column] else ""},
                num_proc=self.num_proc,
            )
            # Override text_column since we've already extracted to "text"
            dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
            dataset_cfg["text_column"] = "text"
            dataset_cfg = OmegaConf.create(dataset_cfg)

        # Normalize column names for text (and audio if present)
        col_map = {"text": dataset_cfg.get("text_column", "text")}

        # Only map audio column if not TTS mode or if audio column exists
        audio_column = dataset_cfg.get("audio_column")
        if audio_column and audio_column in ds.column_names:
            col_map["audio"] = audio_column

        # Map codes column if specified (for S2S AR codec training)
        codes_column = dataset_cfg.get("codes_column")
        if codes_column and codes_column in ds.column_names:
            col_map["codes"] = codes_column

        # Map Stage 3 columns (chain-of-modality instruction fine-tuning)
        input_codes_column = dataset_cfg.get("input_codes_column")
        if input_codes_column and input_codes_column in ds.column_names:
            col_map["input_codes"] = input_codes_column
        output_codes_column = dataset_cfg.get("output_codes_column")
        if output_codes_column and output_codes_column in ds.column_names:
            col_map["output_codes"] = output_codes_column
        input_text_column = dataset_cfg.get("input_text_column")
        if input_text_column and input_text_column in ds.column_names:
            col_map["instruction"] = input_text_column
        input_field_column = dataset_cfg.get("input_field_column")
        if input_field_column and input_field_column in ds.column_names:
            col_map["input_field"] = input_field_column

        for target, source in col_map.items():
            if source != target and source in ds.column_names:
                if target in ds.column_names:
                    ds = ds.remove_columns([target])
                ds = ds.rename_column(source, target)

        # Remove extra columns BEFORE casting to avoid schema mismatch errors
        # (some datasets have complex column types that can't be cast)
        if self.sift_enabled:
            # SIFT training: text responses for distillation
            keep_cols = {
                "audio",
                "text",
                "duration",
                "sift_response",
                "mode",
                "task",
            }
        elif self.s2s_enabled or self.tts_enabled:
            # S2S / TTS training: text + codes only (no audio needed)
            # Stage 3 also needs input_codes, output_codes, instruction, input_field
            keep_cols = {
                "text",
                "duration",
                "task",
                "codes",
                "input_codes",
                "output_codes",
                "instruction",
                "input_field",
            }
        else:
            # Standard ASR training
            keep_cols = {"audio", "text", "duration", "task"}
        extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]

        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # Add task column if specified in config
        task = dataset_cfg.get("task", "transcribe")
        ds = ds.add_column("task", [task] * len(ds))

        # Cast audio column after removing problematic columns (skip for TTS)
        if "audio" in ds.column_names:
            return ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))
        return ds

    def _resample_to_target(self, ds: Dataset, target: int) -> Dataset:
        """Upsample or downsample dataset to target size."""
        current = len(ds)
        if current == target:
            return ds
        if current > target:
            # Downsample
            return ds.select(range(target))
        # Upsample by repeating
        repeats = (target // current) + 1
        indices = list(range(current)) * repeats
        return ds.select(indices[:target])

    def load(self) -> tuple[Dataset, Dataset]:
        train_datasets, val_datasets = [], []

        for d_cfg in tqdm(self.config.datasets, desc="Loading datasets"):
            train_splits = d_cfg.get("train_splits", [d_cfg.get("train_split", "train")])
            eval_splits = d_cfg.get("eval_splits", [d_cfg.get("eval_split", "validation")])
            target_samples = d_cfg.get("target_samples")

            for train_split in train_splits:
                ds = self._prepare_split(d_cfg, train_split)
                if target_samples:
                    ds = self._resample_to_target(ds, target_samples)
                train_datasets.append(ds)

            for eval_split in eval_splits:
                ds = self._prepare_split(d_cfg, eval_split)
                val_datasets.append(ds)

        # Concatenate and shuffle
        train_ds = (
            concatenate_datasets(train_datasets).shuffle(seed=self.seed) if train_datasets else None
        )
        val_ds = concatenate_datasets(val_datasets) if val_datasets else None

        if val_ds and self.config.get("max_eval_samples"):
            n_samples = min(len(val_ds), self.config.max_eval_samples)
            val_ds = val_ds.select(range(n_samples))

        return train_ds, val_ds


class DataCollator:
    """Collates audio and text data for training."""

    # Default conv layers for Whisper/GLM-ASR: [(pad, kernel, stride), ...]
    DEFAULT_ENCODER_CONV_LAYERS = [(1, 3, 1), (1, 3, 2)]

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        system_prompt: str = None,
        projector: Any = None,
        encoder_conv_layers: list = None,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.system_prompt = system_prompt
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or self.DEFAULT_ENCODER_CONV_LAYERS

        # Use trl's DataCollatorForChatML for label masking
        # max_length needs to accommodate audio tokens (1500 for 30s) + prompt + response
        self.text_collator = DataCollatorForChatML(
            tokenizer=tokenizer,
            max_length=2048,
        )

    def _compute_encoder_output_length(self, mel_length: int) -> int:
        """Compute encoder output length using conv layer formulas."""
        length = mel_length
        for padding, kernel_size, stride in self.encoder_conv_layers:
            length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        return length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio
        audio_arrays = []
        valid_features = []
        for f in features:
            try:
                audio = f["audio"]["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                audio_arrays.append(audio)
                valid_features.append(f)
            except Exception:
                continue
            finally:
                f["audio"] = None

        if not audio_arrays:
            raise ValueError("No valid audio samples in batch")

        audio_out = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding="longest",  # Pad to longest in batch, not fixed 30s
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Compute per-sample audio token counts (like GlmAsr)
        mel_lengths = audio_out.attention_mask.sum(dim=-1)  # Per-sample mel lengths
        audio_token_counts = []
        for mel_len in mel_lengths:
            encoder_len = self._compute_encoder_output_length(int(mel_len.item()))
            num_tokens = self.projector.get_output_length(encoder_len)
            audio_token_counts.append(num_tokens)

        # Lowercase all training texts
        processed_texts = [(f.get("text") or "").strip().lower() for f in valid_features]

        # Build messages for each sample with per-sample audio token counts
        text_features = []
        for text, num_audio_tokens in zip(processed_texts, audio_token_counts):
            audio_placeholder = "<audio>" * num_audio_tokens
            prompt = random.choice(TRANSCRIBE_PROMPTS)
            user_content = audio_placeholder + " " + prompt

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": text})

            text_features.append({"messages": messages})

        # Let trl handle tokenization, label masking, and padding
        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask

        return batch


# SIFT configuration (AZeroS-style training)
SIFT_SYSTEM_MESSAGE = ""  # No system message


class SIFTDataCollator(DataCollator):
    """Collates audio and text data for SIFT training."""

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        projector: Any = None,
        encoder_conv_layers: list = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=sample_rate,
            system_prompt=SIFT_SYSTEM_MESSAGE,
            projector=projector,
            encoder_conv_layers=encoder_conv_layers,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio
        audio_arrays = []
        valid_features = []
        for f in features:
            try:
                audio = f["audio"]["array"]
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = audio.squeeze()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                audio_arrays.append(audio)
                valid_features.append(f)
            except Exception:
                continue
            finally:
                f["audio"] = None

        if not audio_arrays:
            raise ValueError("No valid audio samples in batch")

        audio_out = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sample_rate,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Compute per-sample audio token counts
        mel_lengths = audio_out.attention_mask.sum(dim=-1)
        audio_token_counts = []
        for mel_len in mel_lengths:
            encoder_len = self._compute_encoder_output_length(int(mel_len.item()))
            num_tokens = self.projector.get_output_length(encoder_len)
            audio_token_counts.append(num_tokens)

        # Build messages for each sample based on task type
        text_features = []
        for f, num_audio_tokens in zip(valid_features, audio_token_counts):
            audio_placeholder = "<audio>" * num_audio_tokens
            task = f.get("task", "sift")

            if task == "transcribe":
                # Transcription task: use transcription prompt
                response = (f.get("text") or "").strip().lower()
                prompt = random.choice(TRANSCRIBE_PROMPTS)
                user_content = audio_placeholder + " " + prompt
            elif task == "answer":
                # Answer extraction task: empty prompt, lowercased text target
                response = (f.get("text") or "").strip().lower()
                user_content = audio_placeholder
            else:
                # SIFT task: use AZeroS-style fixed instructions based on mode
                mode = f.get("mode", "")
                response = (f.get("sift_response") or f.get("text") or "").strip()
                instruction = SIFT_INSTRUCTIONS.get(mode, "")

                # Format: <audio_tokens> {instruction} (matching AZeroS trainer.py)
                if instruction:
                    user_content = f"{audio_placeholder} {instruction}"
                else:
                    user_content = audio_placeholder

            # Build messages (skip system if empty)
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": response})

            text_features.append({"messages": messages})

        # Let trl handle tokenization, label masking, and padding
        batch = self.text_collator(text_features)
        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask

        return batch


class S2SDataCollator:
    """Data collator for standalone AudioHead training: text tokens + NeuCodec codes.

    Tokenizes text using the LLM tokenizer and prepares NeuCodec FSQ teacher-forced
    inputs/labels. The frozen LLM forward pass happens inside AudioHead.forward().
    """

    def __init__(
        self,
        tokenizer: Any,
        max_text_length: int = 256,
        text_column: str = "text",
        codes_column: str = "codes",
    ):
        from tiny_audio.audio_head import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.text_column = text_column
        self.codes_column = codes_column
        self.BOS_TOKEN = BOS_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Tokenize text WITHOUT special tokens — AudioHead prepends its own
        # chat-template prompt prefix that includes start-of-sequence tokens.
        texts = [(f.get(self.text_column) or "").strip().lower() for f in features]
        text_enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        batch = {
            "text_token_ids": text_enc.input_ids,
            "attention_mask": text_enc.attention_mask,
        }

        # Process NeuCodec FSQ codes into teacher-forced inputs and labels
        code_list = []
        code_lengths = []

        for f in features:
            codes = f.get(self.codes_column)
            if codes is None:
                raise ValueError(
                    f"No codec codes found - S2S requires '{self.codes_column}' column."
                )

            # codes is 1D [seq_len]
            codes_t = torch.tensor(codes, dtype=torch.long)
            if codes_t.dim() > 1:
                codes_t = codes_t.squeeze()
            code_list.append(codes_t)  # [seq_len]
            code_lengths.append(codes_t.shape[0])

        batch_size = len(code_list)
        max_audio_len = max(code_lengths)

        # Build teacher-forced inputs and labels
        # Input: [BOS, c0, c1, ..., cN, PAD...]  (length = max_len + 1)
        # Labels: [c0, c1, ..., cN, EOS, -100...]  (length = max_len + 1)
        padded_len = max_audio_len + 1  # +1 for BOS/EOS shift

        codec_input_ids = torch.full((batch_size, padded_len), self.PAD_TOKEN, dtype=torch.long)
        codec_labels = torch.full((batch_size, padded_len), -100, dtype=torch.long)
        codec_attention_mask = torch.zeros(batch_size, padded_len, dtype=torch.long)

        for i, codes in enumerate(code_list):
            seq_len = codes.shape[0]
            # Input: BOS at position 0, then codes
            codec_input_ids[i, 0] = self.BOS_TOKEN
            codec_input_ids[i, 1 : seq_len + 1] = codes
            # Labels: codes starting at position 0, then EOS
            codec_labels[i, :seq_len] = codes
            codec_labels[i, seq_len] = self.EOS_TOKEN
            # Attention mask: 1 for BOS + codes positions
            codec_attention_mask[i, : seq_len + 1] = 1

        batch["codec_input_ids"] = codec_input_ids
        batch["codec_labels"] = codec_labels
        batch["codec_attention_mask"] = codec_attention_mask

        return batch


class ASRTrainer(Trainer):
    """Trainer subclass for ASR models."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with proper label shifting for causal LM.

        HuggingFace Trainer's label_smoother checks MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        to decide whether to shift labels. Since ASRModel isn't in that mapping,
        it incorrectly uses shift_labels=False, causing misaligned predictions.
        This override forces shift_labels=True for correct causal LM behavior.
        """
        _ = num_items_in_batch  # Unused but required by Trainer signature

        # Pop labels if using label smoothing (matches Trainer behavior)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            # Force shift_labels=True since ASRModel is a causal LM
            loss = self.label_smoother(outputs, labels, shift_labels=True)
            # Scale loss for gradient accumulation (Trainer expects this)
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
        else:
            loss = outputs.loss

        # Fail fast with helpful error if loss is None
        if loss is None:
            # Get debug info - handle wrapped model (DDP/Accelerator)
            underlying_model = getattr(model, "module", model)
            has_audio_head = getattr(underlying_model, "audio_head", None) is not None
            has_codec_labels = "codec_labels" in inputs
            has_labels = "labels" in inputs
            raise ValueError(
                f"Model returned None loss. This usually means the forward pass didn't compute a loss. "
                f"Debug info: has_labels={has_labels}, has_audio_head={has_audio_head}, "
                f"has_codec_labels={has_codec_labels}. "
                f"Input keys: {list(inputs.keys())}"
            )

        return (loss, outputs) if return_outputs else loss


class GradientDebugCallback(TrainerCallback):
    """Debug callback to check gradients after backward pass."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """Called right before optimizer.step(), after backward."""
        if model is None:
            return
        underlying = getattr(model, "module", model)
        # Check ALL parameters, not just audio_head
        for name, param in underlying.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"GRAD DEBUG step {state.global_step}: NaN grad in {name}")
                print(f"  param requires_grad: {param.requires_grad}")
                break


class PushToHubCallback(TrainerCallback):
    """Pushes model to Hub on every save."""

    def on_save(self, args, state, control, **kwargs):
        if not (args.push_to_hub and args.hub_model_id):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        with contextlib.suppress(Exception):
            model.push_to_hub(
                repo_id=args.hub_model_id,
                commit_message=f"Training in progress - step {state.global_step}",
                private=args.hub_private_repo,
            )

        return control


def get_valid_training_args(config: dict) -> dict:
    """Filter config to only valid TrainingArguments fields."""
    valid_fields = {f.name for f in fields(TrainingArguments)}
    return {k: v for k, v in config.items() if k in valid_fields}


class PushAudioHeadCallback(TrainerCallback):
    """On every checkpoint save, push AudioHead weights + config to the Hub.

    Only uploads the small AudioHead (not the full multi-GB base model),
    so saves complete in seconds rather than hanging on large uploads.
    """

    def __init__(self, hub_model_id: str, audio_head_config_dict: dict):
        self.hub_model_id = hub_model_id
        self.audio_head_config_dict = audio_head_config_dict

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        import json
        import tempfile
        from pathlib import Path

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(self.hub_model_id, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Save AudioHead weights
            from safetensors.torch import save_file

            save_file(model.state_dict(), tmp_path / "audio_head.safetensors")

            # Save AudioHead config
            with (tmp_path / "audio_head_config.json").open("w") as f:
                json.dump(self.audio_head_config_dict, f, indent=2)

            print(f"Pushing AudioHead to {self.hub_model_id} (step {state.global_step})")
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=self.hub_model_id,
                commit_message=f"AudioHead weights - step {state.global_step}",
            )

        return control


def train_s2s(cfg: DictConfig) -> None:
    """Standalone AudioHead training: text -> frozen LLM -> projector -> neutts-nano -> speech codes.

    Architecture: frozen SmolLM3 (text->hidden states) + trainable projector MLP
    + frozen neutts-nano backbone (hidden states->speech codes).

    Text is tokenized using the LLM's tokenizer, run through the frozen LLM to get
    hidden states, then projected into neutts-nano's input space via the trainable projector.

    On each checkpoint save, pushes AudioHead projector weights to Hub (if hub_model_id is set).
    """
    from transformers import AutoTokenizer

    from tiny_audio.audio_head import AudioHead, AudioHeadConfig

    # Initialize wandb
    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project=cfg.training.get("wandb_project", "tiny-audio-s2s"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)

    tts_model_id = model_cfg.get("tts_model_id", "neuphonic/neutts-nano")
    llm_model_id = model_cfg.get("llm_model_id", "HuggingFaceTB/SmolLM3-3B")

    # Load the LLM's tokenizer for text tokenization.
    # Text is run through the frozen LLM to get hidden states for the projector.
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create AudioHead: frozen LLM + frozen neutts-nano + trainable projector
    audio_head_config = AudioHeadConfig(
        tts_model_id=tts_model_id,
        llm_model_id=llm_model_id,
        projector_hidden=model_cfg.get("projector_hidden", 1024),
        max_audio_tokens=model_cfg.get("max_audio_tokens", 500),
        temperature=model_cfg.get("temperature", 1.0),
        top_k=model_cfg.get("top_k", 50),
    )

    model = AudioHead(audio_head_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"AudioHead parameters: {total_params:,} total")
    print(f"  Trainable (projector): {trainable_params:,}")
    print(f"  Frozen (LLM+backbone): {frozen_params:,}")

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg, s2s_enabled=True).load()

    # Data collator: text tokenization (via LLM tokenizer) + NeuCodec codes
    data_collator = S2SDataCollator(
        tokenizer=tokenizer,
        max_text_length=cfg.model.get("max_text_length", 256),
    )

    # Training args
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    valid_args = get_valid_training_args(training_config)
    training_args = TrainingArguments(**valid_args)

    # Callbacks
    callbacks = [GradientDebugCallback()]
    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )

    # Push AudioHead projector weights to Hub on every checkpoint save
    hub_model_id = cfg.training.get("hub_model_id")
    if hub_model_id:
        callbacks.append(
            PushAudioHeadCallback(
                hub_model_id=hub_model_id,
                audio_head_config_dict={
                    "tts_model_id": tts_model_id,
                    "llm_model_id": llm_model_id,
                    "projector_hidden": model_cfg.get("projector_hidden", 1024),
                    "max_audio_tokens": model_cfg.get("max_audio_tokens", 500),
                    "temperature": model_cfg.get("temperature", 1.0),
                    "top_k": model_cfg.get("top_k", 50),
                },
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()


def train_tts(cfg: DictConfig) -> None:
    """LLASA-style TTS training with TRL SFTTrainer and sequence packing.

    Supports two stages (following SpeechGPT):
    - Stage 1: Speech-only LM on xcodec2 codes (no text). Learns speech structure.
    - Stage 2: Text→speech training. Loads Stage1 checkpoint for fine-tuning.

    Uses prompt-completion format with packing for efficient training.
    """
    from trl import SFTConfig, SFTTrainer

    from tiny_audio.lm import setup_tts_model

    stage = cfg.tts.get("stage", 2)

    # Initialize wandb
    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project=cfg.training.get("wandb_project", f"tiny-audio-lm-stage{stage}"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)

    llm_model_id = model_cfg.get("llm_model_id", "HuggingFaceTB/SmolLM3-3B")
    stage1_checkpoint = cfg.tts.get("stage1_checkpoint")

    model, tokenizer, _ = setup_tts_model(
        model_id=llm_model_id,
        dtype=torch.bfloat16 if cfg.training.get("bf16") else torch.float32,
        checkpoint_path=stage1_checkpoint,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TTS Stage {stage} parameters: {total_params:,} total, {trainable_params:,} trainable")
    if stage1_checkpoint:
        print(f"  Loaded Stage1 checkpoint: {stage1_checkpoint}")

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg, tts_enabled=True).load()

    # Token IDs for tokenization
    speech_start_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    speech_offset = tokenizer.convert_tokens_to_ids("<|s_0|>")
    max_seq_length = cfg.model.get("max_seq_length", 2048)

    if stage == 1:
        # Stage 1: Speech-only language modeling — no text, just speech codes
        def tokenize_lm_stage1_batch(batch):
            all_codes = batch["codes"]
            all_input_ids = []
            all_completion_mask = []

            for codes in all_codes:
                if codes is None:
                    all_input_ids.append([])
                    all_completion_mask.append([])
                    continue

                # Flatten nested codes: [[c0, c1, ...]] → [c0, c1, ...]
                if isinstance(codes[0], (list, tuple)):
                    codes = [c for sublist in codes for c in sublist]

                input_ids = [speech_start_id] + [speech_offset + c for c in codes] + [speech_end_id]
                input_ids = input_ids[:max_seq_length]
                completion_mask = [1] * len(input_ids)

                all_input_ids.append(input_ids)
                all_completion_mask.append(completion_mask)

            return {"input_ids": all_input_ids, "completion_mask": all_completion_mask}

        tokenize_fn = tokenize_lm_stage1_batch
    elif stage == 2:
        # Stage 2: Cross-modal instruction fine-tuning (SpeechGPT-style)
        # Produces TWO samples per data point:
        #   - Transcribe: <speech_start> codes <speech_end> <text_start> text <text_end>
        #   - Speak:      <text_start> text <text_end> <speech_start> codes <speech_end>
        # Loss is only on the output (completion) side.
        text_start_id = tokenizer.convert_tokens_to_ids("<|TEXT_UNDERSTANDING_START|>")
        text_end_id = tokenizer.convert_tokens_to_ids("<|TEXT_UNDERSTANDING_END|>")

        def tokenize_cross_modal_batch(batch):
            texts = [t.strip() if t else "" for t in batch["text"]]
            all_codes = batch["codes"]

            # Batch-tokenize all texts at once (much faster than one-by-one)
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]

            all_input_ids = []
            all_completion_mask = []

            for text_token_ids, codes in zip(encoded, all_codes):
                if codes is None:
                    continue

                # Flatten nested codes: [[c0, c1, ...]] → [c0, c1, ...]
                if isinstance(codes[0], (list, tuple)):
                    codes = [c for sublist in codes for c in sublist]

                speech_ids = [speech_offset + c for c in codes]

                # --- Transcribe direction: speech → text ---
                transcribe_prompt = [speech_start_id] + speech_ids + [speech_end_id, text_start_id]
                transcribe_completion = text_token_ids + [text_end_id]
                transcribe_seq = (transcribe_prompt + transcribe_completion)[:max_seq_length]
                t_prompt_len = min(len(transcribe_prompt), len(transcribe_seq))
                transcribe_mask = [0] * t_prompt_len + [1] * (len(transcribe_seq) - t_prompt_len)

                all_input_ids.append(transcribe_seq)
                all_completion_mask.append(transcribe_mask)

                # --- Speak direction: text → speech ---
                speak_prompt = [text_start_id] + text_token_ids + [text_end_id, speech_start_id]
                speak_completion = speech_ids + [speech_end_id]
                speak_seq = (speak_prompt + speak_completion)[:max_seq_length]
                s_prompt_len = min(len(speak_prompt), len(speak_seq))
                speak_mask = [0] * s_prompt_len + [1] * (len(speak_seq) - s_prompt_len)

                all_input_ids.append(speak_seq)
                all_completion_mask.append(speak_mask)

            return {"input_ids": all_input_ids, "completion_mask": all_completion_mask}

        tokenize_fn = tokenize_cross_modal_batch
    elif stage == 3:
        # Stage 3: Chain-of-modality instruction fine-tuning (SpeechGPT-style)
        # Produces FOUR samples per data point:
        #   1. Speech→Speech: spoken instruction → spoken response
        #   2. Speech→Text:   spoken instruction → text response
        #   3. Text→Speech:   text instruction → spoken response
        #   4. Text→Text:     text instruction → text response
        # Loss is only on the output (completion) side.
        text_start_id = tokenizer.convert_tokens_to_ids("<|TEXT_UNDERSTANDING_START|>")
        text_end_id = tokenizer.convert_tokens_to_ids("<|TEXT_UNDERSTANDING_END|>")

        def _flatten_codes(codes):
            """Flatten nested codes: [[c0, c1, ...]] → [c0, c1, ...]."""
            if codes and isinstance(codes[0], (list, tuple)):
                return [c for sublist in codes for c in sublist]
            return codes

        def tokenize_chain_of_modality_batch(batch):
            # Output text (response)
            output_texts = [t.strip() if t else "" for t in batch["text"]]
            # Input text: instruction + input field
            instructions = [t.strip() if t else "" for t in batch["instruction"]]
            input_fields = [
                t.strip() if t else "" for t in batch.get("input_field", [""] * len(output_texts))
            ]
            input_texts = [
                f"{inst}\n{inp}".strip() if inp else inst
                for inst, inp in zip(instructions, input_fields)
            ]

            all_input_codes = batch["input_codes"]
            all_output_codes = batch["output_codes"]

            # Batch-tokenize all texts at once
            encoded_inputs = tokenizer(input_texts, add_special_tokens=False)["input_ids"]
            encoded_outputs = tokenizer(output_texts, add_special_tokens=False)["input_ids"]

            all_input_ids = []
            all_completion_mask = []

            for input_text_ids, output_text_ids, in_codes, out_codes in zip(
                encoded_inputs, encoded_outputs, all_input_codes, all_output_codes
            ):
                if in_codes is None or out_codes is None:
                    continue

                in_codes = _flatten_codes(in_codes)
                out_codes = _flatten_codes(out_codes)
                in_speech_ids = [speech_offset + c for c in in_codes]
                out_speech_ids = [speech_offset + c for c in out_codes]

                # 1. Speech→Speech: spoken instruction → spoken response
                s2s_prompt = [speech_start_id] + in_speech_ids + [speech_end_id, speech_start_id]
                s2s_completion = out_speech_ids + [speech_end_id]
                s2s_seq = (s2s_prompt + s2s_completion)[:max_seq_length]
                s2s_plen = min(len(s2s_prompt), len(s2s_seq))
                all_input_ids.append(s2s_seq)
                all_completion_mask.append([0] * s2s_plen + [1] * (len(s2s_seq) - s2s_plen))

                # 2. Speech→Text: spoken instruction → text response
                s2t_prompt = [speech_start_id] + in_speech_ids + [speech_end_id, text_start_id]
                s2t_completion = output_text_ids + [text_end_id]
                s2t_seq = (s2t_prompt + s2t_completion)[:max_seq_length]
                s2t_plen = min(len(s2t_prompt), len(s2t_seq))
                all_input_ids.append(s2t_seq)
                all_completion_mask.append([0] * s2t_plen + [1] * (len(s2t_seq) - s2t_plen))

                # 3. Text→Speech: text instruction → spoken response
                t2s_prompt = [text_start_id] + input_text_ids + [text_end_id, speech_start_id]
                t2s_completion = out_speech_ids + [speech_end_id]
                t2s_seq = (t2s_prompt + t2s_completion)[:max_seq_length]
                t2s_plen = min(len(t2s_prompt), len(t2s_seq))
                all_input_ids.append(t2s_seq)
                all_completion_mask.append([0] * t2s_plen + [1] * (len(t2s_seq) - t2s_plen))

                # 4. Text→Text: text instruction → text response
                t2t_prompt = [text_start_id] + input_text_ids + [text_end_id, text_start_id]
                t2t_completion = output_text_ids + [text_end_id]
                t2t_seq = (t2t_prompt + t2t_completion)[:max_seq_length]
                t2t_plen = min(len(t2t_prompt), len(t2t_seq))
                all_input_ids.append(t2t_seq)
                all_completion_mask.append([0] * t2t_plen + [1] * (len(t2t_seq) - t2t_plen))

            return {"input_ids": all_input_ids, "completion_mask": all_completion_mask}

        tokenize_fn = tokenize_chain_of_modality_batch
    else:
        raise ValueError(f"Unknown TTS stage: {stage}. Supported: 1, 2, 3")

    # Compute max_steps from dataset size before converting to iterable
    num_epochs = cfg.training.get("num_train_epochs", 1)
    batch_size = cfg.training.get("per_device_train_batch_size", 8)
    grad_accum = cfg.training.get("gradient_accumulation_steps", 1)
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch = batch_size * grad_accum * num_gpus
    max_steps = (len(train_dataset) * num_epochs + effective_batch - 1) // effective_batch
    print(
        f"Dataset size: {len(train_dataset):,} samples, effective batch: {effective_batch}, max_steps: {max_steps:,}"
    )

    # Convert to IterableDataset so .map() is lazy (zero memory).
    # Tokenization happens on-the-fly during training instead of materializing a new Arrow table.
    train_columns = train_dataset.column_names
    train_dataset = train_dataset.to_iterable_dataset(num_shards=64)
    train_dataset = train_dataset.map(
        tokenize_fn, batched=True, batch_size=1000, remove_columns=train_columns
    )
    if val_dataset is not None:
        val_columns = val_dataset.column_names
        val_dataset = val_dataset.to_iterable_dataset(num_shards=16)
        val_dataset = val_dataset.map(
            tokenize_fn, batched=True, batch_size=1000, remove_columns=val_columns
        )

    # SFTConfig: extends TrainingArguments with packing + max_seq_length
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    valid_fields = {f.name for f in fields(SFTConfig)}
    valid_args = {k: v for k, v in training_config.items() if k in valid_fields}
    valid_args["packing"] = True
    valid_args["max_steps"] = max_steps
    valid_args["max_length"] = cfg.model.get("max_seq_length", 2048)

    sft_config = SFTConfig(**valid_args)

    # Callbacks
    callbacks = []
    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))

    trainer.save_model()

    # Save tokenizer alongside model (needed for inference with expanded vocab)
    tokenizer.save_pretrained(sft_config.output_dir)


def train_asr(cfg: DictConfig) -> None:
    """Standard ASR/SIFT training with full ASRModel."""
    # Check HF_TOKEN is set if pushing to hub
    if (
        cfg.training.get("push_to_hub")
        and cfg.training.get("hub_model_id")
        and not os.environ.get("HF_TOKEN")
    ):
        raise ValueError(
            "HF_TOKEN environment variable is required when push_to_hub is enabled. "
            "Set it with: export HF_TOKEN=your_token"
        )

    # Initialize wandb
    if cfg.training.get("report_to") == "wandb":
        wandb.init(
            project=cfg.training.get("wandb_project", "tiny-audio"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Create model config from hydra config
    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config_dict, dict), "model config must be a dict"
    training_model_params = [
        "label_smoothing",
        "use_specaugment",
        "num_time_masks",
        "time_mask_length",
        "num_freq_masks",
        "freq_mask_length",
        "attn_implementation",
        "freeze_projector",
    ]
    for param in training_model_params:
        if cfg.training.get(param) is not None:
            model_config_dict[param] = cfg.training[param]
    asr_config = ASRConfig(**model_config_dict)

    # Load or create model
    if cfg.model.get("pretrained_model_path"):
        model = ASRModel.from_pretrained(cfg.model.pretrained_model_path, config=asr_config)
    else:
        model = ASRModel(asr_config)

    model.generation_config.validate()
    model.config.use_cache = False

    if hub_model_id := cfg.training.get("hub_model_id"):
        model.config.pretrained_model_path = hub_model_id

    # Disable Qwen3 thinking mode
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",
        )

    sift_enabled = cfg.get("sift", {}).get("enabled", False)

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg, sift_enabled=sift_enabled).load()

    # Create data collator
    if sift_enabled:
        data_collator = SIFTDataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
        )
    else:
        data_collator = DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            system_prompt=cfg.model.system_prompt,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
        )

    # Setup callbacks
    callbacks = [GradientDebugCallback()]
    if cfg.early_stopping.patience:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping.patience,
                early_stopping_threshold=cfg.early_stopping.threshold,
            )
        )
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        callbacks.append(PushToHubCallback())

    # Configure torch.compile if specified
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(training_config, dict)
    if compile_config := training_config.pop("torch_compile_config", None):
        torch._dynamo.config.cache_size_limit = compile_config.get("cache_size_limit", 64)
        torch._dynamo.config.capture_scalar_outputs = compile_config.get(
            "capture_scalar_outputs", True
        )
        torch._inductor.config.compile_threads = compile_config.get("compile_threads", 4)

    valid_args = get_valid_training_args(training_config)
    trainer = ASRTrainer(
        model=model,
        args=TrainingArguments(**valid_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    trainer.save_model()

    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        trainer.model.push_to_hub(
            cfg.training.hub_model_id,
            commit_message="Training complete - final model",
            private=cfg.training.get("hub_private_repo", False),
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    tts_mode = cfg.get("tts", {}).get("enabled", False)
    s2s_mode = cfg.get("s2s", {}).get("enabled", False)
    if tts_mode:
        train_tts(cfg)
    elif s2s_mode:
        train_s2s(cfg)
    else:
        train_asr(cfg)


if __name__ == "__main__":
    main()
