from typing import Any

import numpy as np
import torch
import transformers

try:
    from .asr_modeling import ASRModel
except ImportError:
    from asr_modeling import ASRModel  # type: ignore[no-redef]


class ForcedAligner:
    """Lazy-loaded forced aligner for word-level timestamps."""

    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def get_instance(cls, device: str = "cuda"):
        if cls._model is None:
            from ctc_forced_aligner import load_alignment_model

            dtype = torch.float16 if device == "cuda" else torch.float32
            cls._model, cls._tokenizer = load_alignment_model(device, dtype=dtype)
        return cls._model, cls._tokenizer

    @classmethod
    def align(
        cls,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 16000,
        language: str = "eng",
        batch_size: int = 16,
    ) -> list[dict]:
        """Align transcript to audio and return word-level timestamps.

        Args:
            audio: Audio waveform as numpy array
            text: Transcript text to align
            sample_rate: Audio sample rate (default 16000)
            language: ISO-639-3 language code (default "eng" for English)
            batch_size: Batch size for alignment model

        Returns:
            List of dicts with 'word', 'start', 'end' keys
        """
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            postprocess_results,
            preprocess_text,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = cls.get_instance(device)

        # Convert audio to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).to(model.dtype).to(model.device)
        else:
            audio_tensor = audio.to(model.dtype).to(model.device)

        # Ensure 1D
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        # Generate emissions
        emissions, stride = generate_emissions(model, audio_tensor, batch_size=batch_size)

        # Preprocess text
        tokens_starred, text_starred = preprocess_text(text, romanize=True, language=language)

        # Get alignments
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)

        # Get spans
        spans = get_spans(tokens_starred, segments, blank_token)

        # Get word timestamps
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Convert to simple format
        return [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in word_timestamps]


class ASRPipeline(transformers.AutomaticSpeechRecognitionPipeline):
    """ASR Pipeline for audio-to-text transcription."""

    model: ASRModel

    def __init__(self, model: ASRModel, **kwargs):
        feature_extractor = kwargs.pop("feature_extractor", None)
        tokenizer = kwargs.pop("tokenizer", model.tokenizer)

        if feature_extractor is None:
            feature_extractor = model.get_processor().feature_extractor

        super().__init__(
            model=model, feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs
        )
        self._current_audio = None

    def __call__(self, inputs, return_timestamps: bool = False, **kwargs):
        """Transcribe audio with optional word-level timestamps.

        Args:
            inputs: Audio input (file path, dict with array/sampling_rate, etc.)
            return_timestamps: If True, return word-level timestamps using forced alignment
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            Dict with 'text' key, and 'words' key if return_timestamps=True
        """
        # Store audio for timestamp alignment
        if return_timestamps:
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
            self._current_audio = None

        return result

    def _extract_audio(self, inputs) -> dict | None:
        """Extract audio array from various input formats."""
        import librosa

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
            # File path - load audio
            audio, sr = librosa.load(inputs, sr=16000)
            return {"array": audio, "sampling_rate": sr}
        elif isinstance(inputs, np.ndarray):
            return {"array": inputs, "sampling_rate": 16000}

        return None

    def preprocess(self, inputs, **preprocess_params):
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
        # Extract audio features and is_last flag
        is_last = model_inputs.pop("is_last", True) if isinstance(model_inputs, dict) else True

        if isinstance(model_inputs, dict):
            input_features = model_inputs.get("input_features")
            if input_features is not None:
                input_features = input_features.to(self.model.device)
        else:
            input_features = model_inputs.to(self.model.device)

        generated_ids = self.model.generate(
            input_features=input_features,
            **generate_kwargs,
        )

        return {"tokens": generated_ids, "is_last": is_last}

    def postprocess(self, model_outputs, **kwargs) -> dict[str, str]:
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

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return {"text": text}
