"""Processor that turns raw audio (and optional text) into model inputs."""

from typing import ClassVar, Union

import torch
import transformers
from transformers import ProcessorMixin

try:
    from .asr_config import DEFAULT_ENCODER_CONV_LAYERS, ASRConfig, compute_encoder_output_length
except ImportError:
    from asr_config import (  # type: ignore[no-redef]
        DEFAULT_ENCODER_CONV_LAYERS,
        ASRConfig,
        compute_encoder_output_length,
    )


class ASRProcessor(ProcessorMixin):
    """Processor for Whisper-based ASR models."""

    attributes: ClassVar[list[str]] = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    # Fallback only. The real value comes from `ASRConfig.audio_token`, which
    # resolves to the decoder's native placeholder where it has one (Gemma 4's
    # pretrained "<|audio|>") and to "<audio>" otherwise. Hardcoding the
    # fallback here fails silently on a native-token decoder: "<audio>" was
    # never added to that vocab, so it tokenizes into ordinary subwords and
    # the prompt ends up with zero scatter positions for N audio embeddings.
    AUDIO_TOKEN = "<audio>"
    TRANSCRIBE_PROMPT = "Transcribe the speech to text"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        projector=None,
        encoder_conv_layers: list | None = None,
        audio_token: str | None = None,
    ):
        """Initialize the ASR processor.

        Args:
            feature_extractor: Audio feature extractor (WhisperFeatureExtractor)
            tokenizer: Text tokenizer for the language model
            projector: Audio projector module (for computing output lengths)
            encoder_conv_layers: Conv layer specs [(pad, kernel, stride), ...]
            audio_token: Placeholder token scattered with audio embeddings.
                Must match `ASRConfig.audio_token` / `ASRModel.audio_token`;
                defaults to AUDIO_TOKEN.
        """
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_token = audio_token or self.AUDIO_TOKEN
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or DEFAULT_ENCODER_CONV_LAYERS

    def _compute_encoder_output_length(self, mel_length: int) -> int:
        """Compute encoder output length using conv layer formulas."""
        return compute_encoder_output_length(mel_length, self.encoder_conv_layers)

    def _render_prompt(self, num_audio_tokens: int, text: str | None) -> torch.Tensor:
        """Tokenize one chat prompt carrying exactly `num_audio_tokens` placeholders."""
        if num_audio_tokens > 0:
            user_content = self.audio_token * num_audio_tokens
            if self.TRANSCRIBE_PROMPT:
                user_content += " " + self.TRANSCRIBE_PROMPT
        else:
            user_content = self.TRANSCRIBE_PROMPT or ""

        messages = [{"role": "user", "content": user_content}]
        if text is not None:
            messages.append({"role": "assistant", "content": text})

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=(text is None),
            return_tensors="pt",
            enable_thinking=False,  # Disable Qwen3 thinking mode for ASR
        )

        # apply_chat_template returns a bare tensor or a BatchEncoding/mapping.
        ids = tokenized if isinstance(tokenized, torch.Tensor) else tokenized["input_ids"]
        return (ids[0] if ids.dim() > 1 else ids).to(torch.long)

    def _stack_prompt_rows(self, rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack per-sample prompt rows into a batch, left-padding if ragged.

        Left, not right: these feed `generate`, so padding must not sit between
        the prompt and the first generated token. Mirrors
        `ASRModel._left_pad_prompt_rows`; pad positions never carry
        `audio_token_id`, so the model's masked_scatter is unaffected.
        """
        max_len = max(row.shape[0] for row in rows)
        if all(row.shape[0] == max_len for row in rows):
            input_ids = torch.stack(rows)
            return input_ids, torch.ones_like(input_ids)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, max_len - row.shape[0] :] = row
            attention_mask[i, max_len - row.shape[0] :] = 1
        return input_ids, attention_mask

    def __call__(
        self,
        audio: Union[list, "torch.Tensor"] | None = None,
        text: str | None = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict:
        """Process audio and text inputs for inference.

        Args:
            audio: Raw audio waveform(s). A batch gets one prompt per sample.
            text: Target transcription (optional, for training - but use DataCollator instead)
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Dict with input_features, input_ids, attention_mask
        """
        result = {}
        token_counts = [0]

        # Process audio
        if audio is not None:
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=getattr(self.feature_extractor, "sampling_rate", 16000),
                return_attention_mask=True,
                return_tensors=return_tensors,
                **kwargs,
            )
            result["input_features"] = audio_inputs["input_features"]
            result["audio_attention_mask"] = audio_inputs["attention_mask"]

            if self.projector is None:
                raise ValueError(
                    "ASRProcessor needs a projector to size the audio prompt. Build it "
                    "with ASRModel.get_processor() instead of constructing it directly."
                )

            # One count per sample, from that sample's own mel length. Sizing a
            # single shared prompt from the batch max -- which this used to do --
            # returns batch-1 `input_ids` against batch-B `input_features`, and
            # gives every shorter row more `<audio>` placeholders than the
            # projector produced for it. `masked_scatter` then mis-scatters
            # silently. This is the same failure `_prepare_audio_inputs`
            # documents as fixed on the model side, and it only shows up on a
            # ragged batch, so batch-1 eval never sees it.
            mel_lengths = audio_inputs["attention_mask"].sum(dim=-1).reshape(-1)
            token_counts = [
                int(self.projector.get_output_length(self._compute_encoder_output_length(int(m))))
                for m in mel_lengths
            ]

        rows = [self._render_prompt(n, text) for n in token_counts]
        input_ids, attention_mask = self._stack_prompt_rows(rows)
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        return result


ASRProcessor.register_for_auto_class()
transformers.AutoProcessor.register(ASRConfig, ASRProcessor)
