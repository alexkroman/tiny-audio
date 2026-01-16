"""ASR pipeline for audio-to-text transcription with optional timestamps and diarization."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

try:
    from .asr_modeling import ASRModel
except ImportError:
    from asr_modeling import ASRModel  # type: ignore[no-redef]


def _get_device() -> str:
    """Get best available device for non-transformers models."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ForcedAligner:
    """Lazy-loaded forced aligner for word-level timestamps using torchaudio wav2vec2."""

    _bundle = None
    _model = None
    _labels = None
    _dictionary = None

    @classmethod
    def get_instance(cls, device: str = "cuda"):
        """Get or create the forced alignment model (singleton).

        Args:
            device: Device to run model on ("cuda" or "cpu")

        Returns:
            Tuple of (model, labels, dictionary)
        """
        if cls._model is None:
            import torchaudio

            cls._bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            cls._model = cls._bundle.get_model().to(device)
            cls._model.eval()
            cls._labels = cls._bundle.get_labels()
            cls._dictionary = {c: i for i, c in enumerate(cls._labels)}
        return cls._model, cls._labels, cls._dictionary

    @classmethod
    def align(
        cls,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 16000,
        _language: str = "eng",
        _batch_size: int = 16,
    ) -> list[dict]:
        """Align transcript to audio and return word-level timestamps.

        Args:
            audio: Audio waveform as numpy array
            text: Transcript text to align
            sample_rate: Audio sample rate (default 16000)
            _language: ISO-639-3 language code (default "eng" for English, unused)
            _batch_size: Batch size for alignment model (unused)

        Returns:
            List of dicts with 'word', 'start', 'end' keys
        """
        import torchaudio
        from torchaudio.functional import forced_align, merge_tokens

        device = _get_device()
        model, labels, dictionary = cls.get_instance(device)

        # Convert audio to tensor (copy to ensure array is writable)
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio.copy()).float()
        else:
            waveform = audio.clone().float()

        # Ensure 2D (channels, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed (wav2vec2 expects 16kHz)
        if sample_rate != cls._bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, cls._bundle.sample_rate
            )

        waveform = waveform.to(device)

        # Get emissions from model
        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu()

        # Normalize text: uppercase, keep only valid characters
        transcript = text.upper()
        # Build tokens from transcript
        tokens = []
        for char in transcript:
            if char in dictionary:
                tokens.append(dictionary[char])
            elif char == " ":
                tokens.append(dictionary.get("|", dictionary.get(" ", 0)))

        if not tokens:
            return []

        targets = torch.tensor([tokens], dtype=torch.int32)

        # Run forced alignment
        # Note: forced_align is deprecated in torchaudio 2.6+ and will be removed in 2.9 (late 2025)
        # No official replacement announced yet. See https://github.com/pytorch/audio/issues/3902
        aligned_tokens, scores = forced_align(emission.unsqueeze(0), targets, blank=0)

        # Use torchaudio's merge_tokens to get token spans (removes blanks and merges repeats)
        token_spans = merge_tokens(aligned_tokens[0], scores[0])

        # Convert frame indices to time (model stride is 320 samples at 16kHz = 20ms)
        frame_duration = 320 / cls._bundle.sample_rate

        # Group token spans into words based on pipe separator
        words = text.split()
        word_timestamps = []
        current_word_start = None
        current_word_end = None
        word_idx = 0

        for span in token_spans:
            token_char = labels[span.token]
            if token_char == "|":  # Word separator
                if current_word_start is not None and word_idx < len(words):
                    word_timestamps.append(
                        {
                            "word": words[word_idx],
                            "start": current_word_start * frame_duration,
                            "end": current_word_end * frame_duration,
                        }
                    )
                    word_idx += 1
                current_word_start = None
                current_word_end = None
            else:
                if current_word_start is None:
                    current_word_start = span.start
                current_word_end = span.end

        # Don't forget the last word
        if current_word_start is not None and word_idx < len(words):
            word_timestamps.append(
                {
                    "word": words[word_idx],
                    "start": current_word_start * frame_duration,
                    "end": current_word_end * frame_duration,
                }
            )

        return word_timestamps


try:
    from .diarization import SpeakerDiarizer
except ImportError:
    from diarization import SpeakerDiarizer  # type: ignore[no-redef]

# Re-export for backwards compatibility
__all__ = ["ForcedAligner", "SpeakerDiarizer", "ASRPipeline"]


class ASRPipeline(transformers.AutomaticSpeechRecognitionPipeline):
    """ASR Pipeline for audio-to-text transcription."""

    model: ASRModel

    def __init__(self, model: ASRModel, **kwargs):
        """Initialize ASR pipeline.

        Args:
            model: ASRModel instance for transcription
            **kwargs: Additional arguments (feature_extractor, tokenizer, device)
        """
        feature_extractor = kwargs.pop("feature_extractor", None)
        tokenizer = kwargs.pop("tokenizer", model.tokenizer)

        if feature_extractor is None:
            feature_extractor = model.get_processor().feature_extractor

        super().__init__(
            model=model, feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs
        )
        self._current_audio = None

    def _sanitize_parameters(self, **kwargs):
        """Intercept our custom parameters before parent class validates them."""
        # Remove our custom parameters so parent doesn't see them
        kwargs.pop("return_timestamps", None)
        kwargs.pop("return_speakers", None)
        kwargs.pop("num_speakers", None)
        kwargs.pop("min_speakers", None)
        kwargs.pop("max_speakers", None)
        kwargs.pop("hf_token", None)
        kwargs.pop("user_prompt", None)
        kwargs.pop("diarization_backend", None)

        return super()._sanitize_parameters(**kwargs)

    def __call__(
        self,
        inputs,
        **kwargs,
    ):
        """Transcribe audio with optional word-level timestamps and speaker diarization.

        Args:
            inputs: Audio input (file path, dict with array/sampling_rate, etc.)
            return_timestamps: If True, return word-level timestamps using forced alignment
            return_speakers: If True, return speaker labels for each word
            user_prompt: Custom transcription prompt (default: "Transcribe: ")
            num_speakers: Exact number of speakers (if known, for diarization)
            min_speakers: Minimum number of speakers (for diarization)
            max_speakers: Maximum number of speakers (for diarization)
            hf_token: HuggingFace token for pyannote models (or set HF_TOKEN env var)
            diarization_backend: Backend for diarization ("pyannote" or "local")
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            Dict with 'text' key, 'words' key if return_timestamps=True,
            and speaker labels on words if return_speakers=True
        """
        # Extract our params before super().__call__ (which will also call _sanitize_parameters)
        return_timestamps = kwargs.pop("return_timestamps", False)
        return_speakers = kwargs.pop("return_speakers", False)
        user_prompt = kwargs.pop("user_prompt", None)
        diarization_params = {
            "num_speakers": kwargs.pop("num_speakers", None),
            "min_speakers": kwargs.pop("min_speakers", None),
            "max_speakers": kwargs.pop("max_speakers", None),
            "hf_token": kwargs.pop("hf_token", None),
            "backend": kwargs.pop("diarization_backend", "pyannote"),
        }

        if return_speakers:
            return_timestamps = True

        # Set custom user prompt if provided
        original_prompt = None
        if user_prompt:
            original_prompt = self.model.TRANSCRIBE_PROMPT
            self.model.TRANSCRIBE_PROMPT = user_prompt

        # Store audio for timestamp alignment and diarization
        if return_timestamps or return_speakers:
            self._current_audio = self._extract_audio(inputs)

        # Run standard transcription
        result = super().__call__(inputs, **kwargs)

        # Add timestamps if requested
        if return_timestamps and self._current_audio is not None:
            text = result.get("text", "")
            if text:
                try:
                    words = ForcedAligner.align(
                        self._current_audio["array"],
                        text,
                        sample_rate=self._current_audio.get("sampling_rate", 16000),
                    )
                    result["words"] = words
                except Exception as e:
                    result["words"] = []
                    result["timestamp_error"] = str(e)
            else:
                result["words"] = []

        # Add speaker diarization if requested
        if return_speakers and self._current_audio is not None:
            try:
                # Run diarization
                speaker_segments = SpeakerDiarizer.diarize(
                    self._current_audio["array"],
                    sample_rate=self._current_audio.get("sampling_rate", 16000),
                    **{k: v for k, v in diarization_params.items() if v is not None},
                )
                result["speaker_segments"] = speaker_segments

                # Assign speakers to words
                if result.get("words"):
                    result["words"] = SpeakerDiarizer.assign_speakers_to_words(
                        result["words"],
                        speaker_segments,
                    )
            except Exception as e:
                result["speaker_segments"] = []
                result["diarization_error"] = str(e)

        # Clean up
        self._current_audio = None
        if original_prompt is not None:
            self.model.TRANSCRIBE_PROMPT = original_prompt

        return result

    def _extract_audio(self, inputs) -> dict | None:
        """Extract audio array from various input formats using HF utilities."""
        from transformers.pipelines.audio_utils import ffmpeg_read

        if isinstance(inputs, dict):
            if "array" in inputs:
                return {
                    "array": inputs["array"],
                    "sampling_rate": inputs.get("sampling_rate", 16000),
                }
            if "raw" in inputs:
                return {
                    "array": inputs["raw"],
                    "sampling_rate": inputs.get("sampling_rate", 16000),
                }
        elif isinstance(inputs, str):
            # File path - load audio using ffmpeg (same as HF pipeline)
            with Path(inputs).open("rb") as f:
                audio = ffmpeg_read(f.read(), sampling_rate=16000)
            return {"array": audio, "sampling_rate": 16000}
        elif isinstance(inputs, bytes):
            audio = ffmpeg_read(inputs, sampling_rate=16000)
            return {"array": audio, "sampling_rate": 16000}
        elif isinstance(inputs, np.ndarray):
            return {"array": inputs, "sampling_rate": 16000}

        return None

    def preprocess(self, inputs, **preprocess_params):
        """Preprocess audio inputs for the model.

        Args:
            inputs: Audio input (dict with array, file path, etc.)
            **preprocess_params: Additional preprocessing parameters

        Yields:
            Model input dicts with input_features and attention_mask
        """
        # Handle dict with "array" key (from datasets)
        if isinstance(inputs, dict) and "array" in inputs:
            inputs = {
                "raw": inputs["array"],
                "sampling_rate": inputs.get("sampling_rate", self.feature_extractor.sampling_rate),
            }

        for item in super().preprocess(inputs, **preprocess_params):
            if "is_last" not in item:
                item["is_last"] = True
            yield item

    def _forward(self, model_inputs, **generate_kwargs) -> dict[str, Any]:
        """Run model forward pass to generate transcription.

        Args:
            model_inputs: Dict with input_features and attention_mask
            **generate_kwargs: Generation parameters

        Returns:
            Dict with generated token IDs
        """
        # Extract audio features and is_last flag
        is_last = model_inputs.pop("is_last", True) if isinstance(model_inputs, dict) else True

        input_features = model_inputs["input_features"].to(self.model.device)
        audio_attention_mask = model_inputs["attention_mask"].to(self.model.device)

        generated_ids = self.model.generate(
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
            **generate_kwargs,
        )

        return {"tokens": generated_ids, "is_last": is_last}

    def postprocess(self, model_outputs, **kwargs) -> dict[str, str]:
        """Convert model output tokens to text.

        Args:
            model_outputs: Dict with 'tokens' key containing generated IDs
            **kwargs: Additional postprocessing parameters

        Returns:
            Dict with 'text' key containing transcription
        """
        # Handle list of outputs (from chunking)
        if isinstance(model_outputs, list):
            model_outputs = model_outputs[0] if model_outputs else {}

        tokens = model_outputs.get("tokens")
        if tokens is None:
            return super().postprocess(model_outputs, **kwargs)

        if torch.is_tensor(tokens):
            tokens = tokens.cpu()
            if tokens.dim() > 1:
                tokens = tokens[0]

        # Filter out eos tokens that the tokenizer doesn't recognize as special
        # (generation_config.eos_token_id may differ from tokenizer.eos_token_id)
        if hasattr(self, "model") and hasattr(self.model, "generation_config"):
            eos_ids = self.model.generation_config.eos_token_id
            if eos_ids is not None:
                eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
                tokens = [t for t in tokens.tolist() if t not in eos_set]

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        # Strip <think>...</think> tags (Qwen3 doesn't respect /no_think prompt)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        return {"text": text}
