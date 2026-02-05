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
            # SIFT training: include codes for audio head if present
            keep_cols = {
                "audio",
                "text",
                "duration",
                "sift_response",
                "mode",
                "task",
                "codes",
            }
        elif self.s2s_enabled:
            # S2S training: need codes column for audio head (AR codec generation)
            keep_cols = {"audio", "text", "duration", "task", "codes"}
        else:
            # Standard ASR training
            keep_cols = {"audio", "text", "duration", "task"}
        extra_cols = [c for c in (ds.column_names or []) if c not in keep_cols]

        if extra_cols:
            ds = ds.remove_columns(extra_cols)

        # Add task column if specified in config (use map for robustness after filter)
        task = dataset_cfg.get("task", "transcribe")
        ds = ds.map(lambda x: {"task": task}, num_proc=self.num_proc)

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
    """Collates audio, text, and optional Mimi codes for SIFT training.

    Supports joint training of projector (audio understanding) and audio head
    (speaking responses) when codes are available in the dataset.
    """

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        projector: Any = None,
        encoder_conv_layers: list = None,
        use_audio_head: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            sample_rate=sample_rate,
            system_prompt=SIFT_SYSTEM_MESSAGE,
            projector=projector,
            encoder_conv_layers=encoder_conv_layers,
        )
        self.use_audio_head = use_audio_head

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

        # Add codec targets for audio head training (discrete Mimi codes)
        if self.use_audio_head:
            code_batch = self._extract_codec_targets(valid_features)
            if code_batch is not None:
                batch["codec_targets"] = code_batch["codec_targets"]
                batch["codec_lengths"] = code_batch["codec_lengths"]

        return batch

    def _extract_codec_targets(self, features: list[dict]) -> dict | None:
        """Extract discrete Mimi codes for audio head training.

        Args:
            features: List of feature dicts with codes

        Returns:
            Dict with codec_targets [batch, 8, seq_len] and codec_lengths tensors, or None if no codes
        """
        if not features or "codes" not in features[0]:
            return None

        code_list = []
        code_lengths = []

        for f in features:
            codes = f.get("codes")
            if codes is None:
                continue

            # Extract all 8 codebooks: codes shape [8][seq_len]
            if isinstance(codes[0], list):
                # codes is [8 codebooks][seq_len]
                codes_t = torch.tensor(codes, dtype=torch.long)  # [8, seq_len]
            else:
                # Single codebook (legacy format) - expand to [1, seq_len]
                codes_t = torch.tensor(codes, dtype=torch.long).unsqueeze(0)

            code_list.append(codes_t)
            code_lengths.append(codes_t.shape[1])  # seq_len

        if not code_list:
            return None

        # Pad to max length with pad token
        max_len = max(code_lengths)
        pad_token = 2048 + 3  # vocab_size + PAD_OFFSET
        num_codebooks = code_list[0].shape[0]
        padded = torch.full((len(code_list), num_codebooks, max_len), pad_token, dtype=torch.long)
        for i, codes_t in enumerate(code_list):
            padded[i, :, : codes_t.shape[1]] = codes_t

        return {
            "codec_targets": padded,  # [batch, 8, seq_len]
            "codec_lengths": torch.tensor(code_lengths, dtype=torch.long),
        }


class S2SDataCollator:
    """Data collator for S2S training: audio -> Mimi discrete codes.

    This collator handles speech synthesis training where we predict
    discrete Mimi codec tokens autoregressively.
    Uses datasets with pre-computed codes (e.g., mazesmazes/libritts-mimi).
    """

    # Mimi vocab size per codebook
    VOCAB_SIZE = 2048

    def __init__(
        self,
        tokenizer: Any,
        feature_extractor: Any,
        sample_rate: int,
        projector: Any = None,
        encoder_conv_layers: list = None,
        system_prompt: str = None,
        text_column: str = "text",
        codes_column: str = "codes",
        use_codebook: int = 0,  # Which codebook to use (0 = semantic)
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or [(1, 3, 1), (1, 3, 2)]
        self.system_prompt = system_prompt
        self.text_column = text_column
        self.codes_column = codes_column
        self.use_codebook = use_codebook

        # Use trl's DataCollatorForChatML for label masking
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

    def _convert_to_right_padding(self, batch: dict) -> dict:
        """Convert left-padded batch to right-padded.

        DataCollatorForChatML uses left-padding which causes NaN in attention computation
        when padding positions try to attend (all -inf in attention scores -> NaN after softmax).
        This converts to right-padding where padding is at the end, avoiding the issue.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")

        batch_size, seq_len = input_ids.shape
        pad_token_id = self.tokenizer.pad_token_id

        # For each sample, find where content starts (first non-pad) and shift right
        new_input_ids = torch.full_like(input_ids, pad_token_id)
        new_attention_mask = torch.zeros_like(attention_mask)
        new_labels: torch.Tensor | None = None
        if labels is not None:
            new_labels = torch.full_like(labels, -100)

        for i in range(batch_size):
            # Find first non-padding position
            non_pad_mask = attention_mask[i] == 1
            content_length = non_pad_mask.sum().item()

            if content_length > 0:
                # Copy content to the beginning (right-padding style)
                content_ids = input_ids[i][non_pad_mask]
                content_attn = attention_mask[i][non_pad_mask]

                new_input_ids[i, :content_length] = content_ids
                new_attention_mask[i, :content_length] = content_attn

                if labels is not None and new_labels is not None:
                    content_labels = labels[i][non_pad_mask]
                    new_labels[i, :content_length] = content_labels

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention_mask
        if new_labels is not None:
            batch["labels"] = new_labels

        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Process audio (required for S2S training)
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
            raise ValueError("No valid audio samples in batch - S2S requires audio input")

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

        # Get text for each sample using configured text column
        processed_texts = [(f.get(self.text_column) or "").strip().lower() for f in valid_features]

        # Build messages for each sample
        text_features = []
        for text, num_audio_tokens in zip(processed_texts, audio_token_counts):
            audio_placeholder = "<audio>" * num_audio_tokens
            user_content = audio_placeholder + " Transcribe: "

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": text})

            text_features.append({"messages": messages})

        # Let trl handle tokenization, label masking, and padding
        batch = self.text_collator(text_features)

        # Convert left-padding to right-padding: DataCollatorForChatML uses left-padding
        # which causes NaN in attention (padding positions attend to all -inf -> NaN softmax)
        batch = self._convert_to_right_padding(batch)

        # Create assistant mask from labels before removing them
        # Labels have actual token IDs at assistant positions, -100 elsewhere
        # We use this mask to extract only assistant hidden states for audio head
        if "labels" in batch:
            batch["assistant_mask"] = batch["labels"] != -100
            # Remove labels since we don't compute LM loss (label_names=[] in config)
            del batch["labels"]

        batch["input_features"] = audio_out.input_features
        batch["audio_attention_mask"] = audio_out.attention_mask

        # Process Mimi codec codes (required for S2S AR training)
        # codes shape from dataset: [8 codebooks][seq_len]
        # We extract all 8 codebooks for Depformer training
        code_list = []
        code_lengths = []

        for f in valid_features:
            codes = f.get(self.codes_column)
            if codes is None:
                raise ValueError(
                    f"No codec codes found - S2S requires '{self.codes_column}' column. "
                    "Use scripts/generate_libritts_mimi.py to create dataset."
                )

            # Extract all 8 codebooks: codes shape [8][seq_len]
            if isinstance(codes[0], list):
                # codes is [8 codebooks][seq_len]
                codes_t = torch.tensor(codes, dtype=torch.long)  # [8, seq_len]
            else:
                # Single codebook (legacy format) - expand to [1, seq_len]
                codes_t = torch.tensor(codes, dtype=torch.long).unsqueeze(0)

            code_list.append(codes_t)
            code_lengths.append(codes_t.shape[1])  # seq_len

        # Pad to max length with pad token (vocab_size + 3)
        max_len = max(code_lengths)
        pad_token = self.VOCAB_SIZE + 3  # Matches AudioHead.pad_token_id
        num_codebooks = code_list[0].shape[0]
        padded = torch.full((len(code_list), num_codebooks, max_len), pad_token, dtype=torch.long)
        for i, codes_t in enumerate(code_list):
            padded[i, :, : codes_t.shape[1]] = codes_t

        batch["codec_targets"] = padded  # [batch, 8, seq_len]
        batch["codec_lengths"] = torch.tensor(code_lengths, dtype=torch.long)

        # Dual-path: Tokenize text for TTS (Freeze-Omni style)
        # Path 1: Text embeddings → Pre-NN → context (what to say)
        # Path 2: LLM hidden states → Prefix Bridge → KV cache (how to say it)
        tts_text_ids_list = []
        for text in processed_texts:
            # Tokenize just the text (no chat template)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tts_text_ids_list.append(torch.tensor(tokens, dtype=torch.long))

        # Pad text tokens
        max_text_len = max(len(t) for t in tts_text_ids_list)
        tts_text_ids = torch.full(
            (len(tts_text_ids_list), max_text_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        tts_text_mask = torch.zeros(len(tts_text_ids_list), max_text_len, dtype=torch.bool)
        for i, tokens in enumerate(tts_text_ids_list):
            tts_text_ids[i, : len(tokens)] = tokens
            tts_text_mask[i, : len(tokens)] = True

        batch["tts_text_ids"] = tts_text_ids
        batch["tts_text_mask"] = tts_text_mask

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
            has_codec_targets = "codec_targets" in inputs
            has_labels = "labels" in inputs
            raise ValueError(
                f"Model returned None loss. This usually means the forward pass didn't compute a loss. "
                f"Debug info: has_labels={has_labels}, has_audio_head={has_audio_head}, "
                f"has_codec_targets={has_codec_targets}. "
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Check HF_TOKEN is set if pushing to hub
    if cfg.training.get("push_to_hub") and cfg.training.get("hub_model_id"):
        import os

        if not os.environ.get("HF_TOKEN"):
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
    # Merge model config with training config for model-specific params
    model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_config_dict, dict), "model config must be a dict"
    # Add training params that affect model behavior
    training_model_params = [
        "label_smoothing",
        "projector_dropout",
        "use_specaugment",
        "num_time_masks",
        "time_mask_length",
        "num_freq_masks",
        "freq_mask_length",
        "attn_implementation",
        # LoRA params (Stage 2 fine-tuning)
        "use_lora",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
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

    # Validate generation config early to catch conflicts before training starts
    # (e.g., do_sample=False with temperature/top_p/top_k set)
    model.generation_config.validate()

    model.config.use_cache = False

    # Store hub_model_id in config so save_pretrained() can set base_model_name_or_path correctly
    if hub_model_id := cfg.training.get("hub_model_id"):
        model.config.pretrained_model_path = hub_model_id

    # Disable Qwen3 thinking mode by patching the chat template
    # This is a workaround for TRL's DataCollatorForChatML not passing enable_thinking=False
    # See: https://github.com/huggingface/trl/issues/3387
    if model.tokenizer.chat_template and "enable_thinking" in model.tokenizer.chat_template:
        # Replace the conditional check with a hardcoded False
        model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "true",  # Always disable thinking
        )

    # Check if SIFT training is enabled (AZeroS-style)
    sift_enabled = cfg.get("sift", {}).get("enabled", False)

    # Check if audio head training is enabled (jointly with projector)
    use_audio_head = cfg.model.get("use_audio_head", False)

    # Validate audio head exists when expected
    if use_audio_head and model.audio_head is None:
        raise ValueError(
            f"use_audio_head=True but model.audio_head is None. "
            f"Config use_audio_head={model.config.use_audio_head}. "
            "Check that the config is being passed correctly to the model."
        )

    # Check if S2S training mode (audio head with Mimi codes via AR decoder)
    s2s_mode = use_audio_head and cfg.get("s2s", {}).get("enabled", False)

    # Load datasets
    train_dataset, val_dataset = DatasetLoader(
        cfg, sift_enabled=sift_enabled, s2s_enabled=s2s_mode
    ).load()

    # Create data collator (S2S, SIFT, or standard ASR)
    if s2s_mode:
        # S2S training with Mimi discrete codes (AR generation)
        # DatasetLoader renames text_column to "text", so always use "text"
        text_column = "text"
        data_collator = S2SDataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
            system_prompt=cfg.model.system_prompt,
            text_column=text_column,
        )
    elif sift_enabled:
        # SIFT training with optional audio head for speaking responses
        data_collator = SIFTDataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=cfg.data.sample_rate,
            projector=model.projector,
            encoder_conv_layers=model.config.encoder_conv_layers,
            use_audio_head=use_audio_head,
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
    callbacks = [GradientDebugCallback()]  # Debug NaN gradients
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

    # Create trainer with only valid TrainingArguments
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
        # Use model's push_to_hub which properly sets base_model_name_or_path in adapter_config.json
        trainer.model.push_to_hub(
            cfg.training.hub_model_id,
            commit_message="Training complete - final model",
            private=cfg.training.get("hub_private_repo", False),
        )


if __name__ == "__main__":
    main()
