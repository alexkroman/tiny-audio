"""Processor that turns raw audio (and optional text) into model inputs."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast, overload

import numpy as np
import numpy.typing as npt
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    SequenceFeatureExtractor,
)

if TYPE_CHECKING:
    from .asr_config import (
        DEFAULT_ENCODER_CONV_LAYERS,
        ASRConfig,
        ConvLayerSpec,
        compute_encoder_output_length,
    )
    from .asr_types import AudioFeatureExtractor, AudioInput, Waveform
    from .projectors import MLPAudioProjector
else:
    try:
        from .asr_config import (
            DEFAULT_ENCODER_CONV_LAYERS,
            ASRConfig,
            ConvLayerSpec,
            compute_encoder_output_length,
        )
        from .asr_types import AudioInput
    except ImportError:  # flat layout on the Hub: sibling modules, no package
        from asr_config import (
            DEFAULT_ENCODER_CONV_LAYERS,
            ASRConfig,
            ConvLayerSpec,
            compute_encoder_output_length,
        )
        from asr_types import AudioInput


# The instruction the model trained on (scripts/train_collator.py); the model
# and processor both default to it.
DEFAULT_TRANSCRIBE_PROMPT = "Transcribe the speech to text"


def render_audio_prompt(
    tokenizer: PreTrainedTokenizerBase,
    audio_token: str,
    num_audio_tokens: int,
    prompt: str | None,
    text: str | None = None,
) -> torch.Tensor:
    """Tokenize one chat prompt carrying exactly `num_audio_tokens` placeholders.

    The user turn is the placeholders, then `prompt` (if any); `text`, when
    given, is the assistant's reply, otherwise the generation prompt is added.
    """
    if num_audio_tokens > 0:
        user_content = audio_token * num_audio_tokens
        if prompt:
            user_content += " " + prompt
    else:
        user_content = prompt or ""

    messages = [{"role": "user", "content": user_content}]
    if text is not None:
        messages.append({"role": "assistant", "content": text})

    # With `tokenize=True, return_tensors="pt"` the ids come back as tensors.
    tokenized = cast(
        "torch.Tensor | Mapping[str, torch.Tensor]",
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=(text is None),
            return_tensors="pt",
            enable_thinking=False,  # Disable Qwen3 thinking mode for ASR
        ),
    )

    # apply_chat_template returns a bare tensor or a BatchEncoding/mapping.
    ids = tokenized if isinstance(tokenized, torch.Tensor) else tokenized["input_ids"]
    return (ids[0] if ids.dim() > 1 else ids).to(torch.long)


def left_pad_prompt_rows(
    rows: list[torch.Tensor], tokenizer: PreTrainedTokenizerBase
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-sample prompt rows into a left-padded batch: `(input_ids, attention_mask)`.

    Left, not right: these feed `generate`, so padding must not sit between
    the prompt and the first generated token. Pads with the tokenizer's pad
    token, falling back to eos, then 0. Pad positions never carry
    `audio_token_id`, so the model's masked_scatter is unaffected.
    """
    # transformers types special-token ids as any token value; a single id is an int.
    pad_id = cast("int | None", tokenizer.pad_token_id)
    if pad_id is None:
        pad_id = cast("int | None", tokenizer.eos_token_id) or 0
    input_ids = pad_sequence(rows, batch_first=True, padding_value=int(pad_id), padding_side="left")
    # Padded from ones rather than `input_ids != pad_id`: a real token may
    # equal `pad_id` when pad falls back to eos.
    attention_mask = pad_sequence(
        [torch.ones_like(row) for row in rows], batch_first=True, padding_side="left"
    )
    return input_ids, attention_mask


@overload
def prepend_lead_in[ScalarT: np.generic](
    audio: npt.NDArray[ScalarT], sampling_rate: int, seconds: float | None
) -> npt.NDArray[ScalarT]: ...
@overload
def prepend_lead_in[ScalarT: np.generic](
    audio: list[npt.NDArray[ScalarT]], sampling_rate: int, seconds: float | None
) -> list[npt.NDArray[ScalarT]]: ...
@overload
def prepend_lead_in(audio: AudioInput, sampling_rate: int, seconds: float | None) -> AudioInput: ...
def prepend_lead_in(audio: AudioInput, sampling_rate: int, seconds: float | None) -> AudioInput:
    """Prepend `seconds` of silence to a waveform (or each waveform in a list).

    Peoples ships fixed ~15s grid cuts rather than sentence-aligned segments,
    so a clip routinely opens mid-word and the model declines to emit the
    partial first token. Measured on 500 Peoples clips with a paired
    bootstrap: 20.51% -> 19.28% WER (delta -1.22, CI [-1.83, -0.64]) and
    utterances dropping a leading reference word fall 258/460 -> 170/460.
    CommonVoice, whose clips already start cleanly, is unaffected (+0.30,
    CI [-0.43, +1.17]).

    Inference only. Training feeds raw audio through the collator, so this is
    a test-time transform, and it recovers two thirds of the dropped onsets
    rather than all of them -- the remainder are clips whose first syllable
    was never recorded, which no amount of lead-in reconstructs.
    """
    if not seconds or seconds <= 0:
        return audio

    if isinstance(audio, (list, tuple)) and audio and not isinstance(audio[0], (int, float)):
        batch = cast("Sequence[Waveform]", audio)
        return [cast("Waveform", prepend_lead_in(a, sampling_rate, seconds)) for a in batch]

    waveform = cast("Waveform", audio)
    pad = round(sampling_rate * seconds)
    if pad <= 0:
        return waveform
    arr: npt.NDArray[Any] = np.asarray(waveform)
    padded: npt.NDArray[Any] = np.pad(arr, (pad, 0))
    return padded


class ASRProcessor(ProcessorMixin):
    """Processor for Whisper-based ASR models."""

    attributes: ClassVar[list[str]] = ["feature_extractor", "tokenizer"]
    feature_extractor: SequenceFeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    # Fallback only. The real value comes from `ASRConfig.audio_token`, which
    # resolves to the decoder's native placeholder where it has one (Gemma 4's
    # pretrained "<|audio|>") and to "<audio>" otherwise. Hardcoding the
    # fallback here fails silently on a native-token decoder: "<audio>" was
    # never added to that vocab, so it tokenizes into ordinary subwords and
    # the prompt ends up with zero scatter positions for N audio embeddings.
    AUDIO_TOKEN = "<audio>"
    TRANSCRIBE_PROMPT = DEFAULT_TRANSCRIBE_PROMPT

    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        projector: "MLPAudioProjector | None" = None,
        encoder_conv_layers: list[ConvLayerSpec] | None = None,
        audio_token: str | None = None,
        lead_in_seconds: float = 0.0,
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
        self.lead_in_seconds = float(lead_in_seconds)

    def _compute_encoder_output_length(self, mel_length: int) -> int:
        """Compute encoder output length using conv layer formulas."""
        return compute_encoder_output_length(mel_length, self.encoder_conv_layers)

    def _render_prompt(self, num_audio_tokens: int, text: str | None) -> torch.Tensor:
        """Tokenize one chat prompt carrying exactly `num_audio_tokens` placeholders."""
        return render_audio_prompt(
            self.tokenizer, self.audio_token, num_audio_tokens, self.TRANSCRIBE_PROMPT, text
        )

    def _stack_prompt_rows(self, rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack per-sample prompt rows into a batch (see `left_pad_prompt_rows`)."""
        return left_pad_prompt_rows(rows, self.tokenizer)

    def __call__(self, *args: Any, **kwargs: Any) -> BatchFeature:
        """Process audio and text inputs for inference; see `_process` for the arguments.

        `ProcessorMixin.__call__` takes `(images, text, videos, audio, ...)`; this
        processor takes audio first, so the arguments are forwarded unchanged to
        `_process`, which carries the real signature.
        """
        return BatchFeature(data=self._process(*args, **kwargs))

    def _process(
        self,
        audio: AudioInput | None = None,
        text: str | None = None,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Process audio and text inputs for inference.

        Args:
            audio: Raw audio waveform(s). A batch gets one prompt per sample.
            text: Target transcription (optional, for training - but use DataCollator instead)
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Dict with input_features, input_ids, attention_mask
        """
        result: dict[str, torch.Tensor] = {}
        token_counts = [0]

        # Process audio
        if audio is not None:
            sr = getattr(self.feature_extractor, "sampling_rate", 16000)
            padded_audio = prepend_lead_in(audio, sr, self.lead_in_seconds)
            extract = cast("AudioFeatureExtractor", self.feature_extractor)
            audio_inputs = extract(
                padded_audio,
                sampling_rate=sr,
                return_attention_mask=True,
                return_tensors=return_tensors,
                **kwargs,
            )
            result["input_features"] = audio_inputs["input_features"]
            result["audio_attention_mask"] = audio_inputs["attention_mask"]

            if self.projector is None:
                msg = (
                    "ASRProcessor needs a projector to size the audio prompt. Build it "
                    "with ASRModel.get_processor() instead of constructing it directly."
                )
                raise ValueError(msg)

            # One count per sample, from that sample's own mel length. Sizing a
            # single shared prompt from the batch max -- which this used to do --
            # returns batch-1 `input_ids` against batch-B `input_features`, and
            # gives every shorter row more `<audio>` placeholders than the
            # projector produced for it. `masked_scatter` then mis-scatters
            # silently. This is the same failure `_prepare_audio_inputs`
            # documents as fixed on the model side, and it only shows up on a
            # ragged batch, so batch-1 eval never sees it.
            mel_lengths = audio_inputs["attention_mask"].sum(dim=-1).reshape(-1).long()
            encoder_lengths = compute_encoder_output_length(mel_lengths, self.encoder_conv_layers)
            token_counts = self.projector.get_output_length(encoder_lengths).tolist()

        rows = [self._render_prompt(n, text) for n in token_counts]
        input_ids, attention_mask = self._stack_prompt_rows(rows)
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        return result


ASRProcessor.register_for_auto_class()
transformers.AutoProcessor.register(ASRConfig, ASRProcessor)
