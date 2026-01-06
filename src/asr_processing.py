from typing import Optional, Union

import torch
import transformers
from transformers import ProcessorMixin

try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]


class ASRProcessor(ProcessorMixin):
    """Processor for Whisper-based ASR models."""

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    AUDIO_TOKEN = "<audio>"
    TRANSCRIBE_PROMPT = "Transcribe: "
    # Default conv layers for Whisper/GLM-ASR: [(pad, kernel, stride), ...]
    DEFAULT_ENCODER_CONV_LAYERS = [(1, 3, 1), (1, 3, 2)]

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        projector=None,
        encoder_conv_layers: Optional[list] = None,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.AUDIO_TOKEN)
        self.projector = projector
        self.encoder_conv_layers = encoder_conv_layers or self.DEFAULT_ENCODER_CONV_LAYERS

    def _compute_encoder_output_length(self, mel_length: int) -> int:
        """Compute encoder output length using conv layer formulas."""
        length = mel_length
        for padding, kernel_size, stride in self.encoder_conv_layers:
            length = (length + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        return length

    def __call__(
        self,
        audio: Optional[Union[list, "torch.Tensor"]] = None,
        text: Optional[str] = None,
        system_prompt: Optional[str] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict:
        """Process audio and text inputs for inference.

        Args:
            audio: Raw audio waveform(s)
            text: Target transcription (optional, for training - but use DataCollator instead)
            system_prompt: Optional system prompt
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Dict with input_features, input_ids, attention_mask
        """
        result = {}

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

            # Use actual audio length (from attention mask) for token count
            real_mel_len = int(audio_inputs["attention_mask"].sum(dim=-1).max().item())
            encoder_output_len = self._compute_encoder_output_length(real_mel_len)
            num_audio_tokens = self.projector.get_output_length(encoder_output_len)
        else:
            num_audio_tokens = 0

        # Build prompt with audio token placeholders
        user_content = self.TRANSCRIBE_PROMPT
        if num_audio_tokens > 0:
            user_content += self.AUDIO_TOKEN * num_audio_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        if text is not None:
            messages.append({"role": "assistant", "content": text})

        # Tokenize
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=(text is None),
            return_tensors=return_tensors,
        )

        # Handle both tensor and BatchEncoding returns
        if isinstance(tokenized, torch.Tensor):
            input_ids = tokenized
        else:
            # BatchEncoding or dict-like object
            input_ids = tokenized.get("input_ids", tokenized.input_ids)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        result["input_ids"] = input_ids
        result["attention_mask"] = torch.ones_like(input_ids)

        return result


ASRProcessor.register_for_auto_class()
transformers.AutoProcessor.register(ASRConfig, ASRProcessor)
