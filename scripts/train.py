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

import hydra
import torch
import wandb
from datasets import (
    Audio,
    Dataset,
    concatenate_datasets,
    load_dataset,
)
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl.experimental.utils import DataCollatorForChatML

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

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

    def __init__(self, config: DictConfig, sift_enabled: bool = False, s2s_enabled: bool = False):
        self.config = config.data
        self.sample_rate = self.config.sample_rate
        self.cache_dir = self.config.dataset_cache_dir
        self.seed = config.training.get("seed", 42)
        self.num_proc = self.config.get("num_proc", 16)
        self.sift_enabled = sift_enabled
        self.s2s_enabled = s2s_enabled

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
        elif self.s2s_enabled:
            # S2S training: text + codes only (no audio needed)
            keep_cols = {"text", "duration", "task", "codes"}
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

    No audio processing, no encoder, no LLM. Just tokenizes text and prepares
    NeuCodec FSQ teacher-forced inputs/labels for the AudioHead transformer.
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
        # Tokenize text
        texts = [(f.get(self.text_column) or "").strip().lower() for f in features]
        text_enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
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
    """Standalone AudioHead training: text tokens -> NeuCodec codes.

    Architecture: frozen neutts-nano backbone + trainable projector MLP (~1.2M params).
    Text is tokenized using neutts-nano's own tokenizer, embedded through its frozen
    embedding table, then adapted through the trainable projector before being fed to
    the frozen backbone.

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

    # Load neutts-nano's tokenizer for text tokenization.
    # In Stage 1, text is embedded through neutts-nano's own embedding table,
    # so we must use its tokenizer to get correct token IDs.
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)

    tts_model_id = model_cfg.get("tts_model_id", "neuphonic/neutts-nano")
    tokenizer = AutoTokenizer.from_pretrained(tts_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create AudioHead: frozen neutts-nano + trainable projector
    audio_head_config = AudioHeadConfig(
        tts_model_id=tts_model_id,
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
    print(f"  Frozen (backbone):     {frozen_params:,}")

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(cfg, s2s_enabled=True).load()

    # Data collator: text tokenization (via neutts-nano tokenizer) + NeuCodec codes
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
    s2s_mode = cfg.get("s2s", {}).get("enabled", False)
    if s2s_mode:
        train_s2s(cfg)
    else:
        train_asr(cfg)


if __name__ == "__main__":
    main()
