from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import transformers
from truecase import get_true_case

from .asr_modeling import ASRModel


class ASRPipeline(transformers.AutomaticSpeechRecognitionPipeline):
    """ASR Pipeline for audio-to-text transcription."""

    model: ASRModel

    def __init__(self, model: ASRModel, **kwargs):
        feature_extractor = kwargs.pop("feature_extractor", None)
        tokenizer = kwargs.pop("tokenizer", model.tokenizer)

        # Get feature extractor from model's processor if not provided
        if feature_extractor is None:
            processor = model.get_processor()
            feature_extractor = processor.feature_extractor

        super().__init__(
            model=model, feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs
        )

        # Initialize text normalizer
        if hasattr(tokenizer, "normalize"):
            self.text_normalizer = tokenizer
        else:
            from transformers import WhisperTokenizer

            self.text_normalizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def __call__(self, inputs, **kwargs):
        generate_kwargs = {}
        generate_keys = [
            "max_new_tokens", "num_beams", "do_sample", "length_penalty",
            "repetition_penalty", "no_repeat_ngram_size", "early_stopping",
            "num_beam_groups", "diversity_penalty", "top_k", "temperature",
            "top_p", "user_prompt", "task", "text_input",
        ]
        for key in generate_keys:
            if key in kwargs:
                generate_kwargs[key] = kwargs.pop(key)

        # Handle text-only mode
        task = generate_kwargs.get("task")
        if task == "text" or generate_kwargs.get("text_input"):
            return self._process_text_only(generate_kwargs)

        # Handle list inputs
        if isinstance(inputs, list):
            return [self.__call__(inp, **kwargs, **generate_kwargs) for inp in inputs]

        model_inputs = self.preprocess(inputs, **kwargs)

        if isinstance(model_inputs, Iterator):
            return self._process_chunks(list(model_inputs), generate_kwargs)

        model_outputs = self._forward(model_inputs, **generate_kwargs)
        return self.postprocess(model_outputs)

    def _process_chunks(self, chunks: list, generate_kwargs: dict) -> dict[str, str]:
        """Process chunked audio and merge results."""
        all_tokens: list[int] = []

        for chunk in chunks:
            output = self._forward(chunk, **generate_kwargs)
            tokens = output.get("tokens") or output.get("generated_ids")
            if tokens is not None:
                if torch.is_tensor(tokens):
                    tokens = tokens.cpu()
                if len(tokens.shape) > 1:
                    tokens = tokens[0]
                all_tokens.extend(tokens.tolist() if torch.is_tensor(tokens) else tokens)

        text = self.tokenizer.decode(all_tokens, skip_special_tokens=True).strip()
        text = self.text_normalizer.normalize(text)
        text = get_true_case(text)

        return {"text": text}

    def preprocess(self, inputs, **preprocess_params):
        if isinstance(inputs, list):
            raise ValueError("Lists should not reach preprocess")

        preprocess_params.setdefault("chunk_length_s", 0)

        # Normalize input formats
        if isinstance(inputs, dict):
            if "bytes" in inputs:
                inputs = self._decode_audio_bytes(inputs["bytes"])
            elif "array" in inputs:
                inputs = {"raw": inputs["array"], "sampling_rate": inputs["sampling_rate"]}
        elif hasattr(inputs, "array") and hasattr(inputs, "sampling_rate"):
            inputs = {"raw": inputs.array, "sampling_rate": inputs.sampling_rate}
        elif hasattr(inputs, "__array__") and not isinstance(inputs, (dict, bytes, str)):
            inputs = {"raw": inputs, "sampling_rate": self.model.config.audio_sample_rate}
        elif torch.is_tensor(inputs):
            inputs = {"raw": inputs.cpu().numpy(), "sampling_rate": self.model.config.audio_sample_rate}

        return super().preprocess(inputs, **preprocess_params)

    def _decode_audio_bytes(self, wav_bytes: bytes) -> dict[str, Any]:
        """Decode audio bytes to array format."""
        import tempfile

        from torchcodec.decoders import AudioDecoder

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_path = f.name

        try:
            decoder = AudioDecoder(temp_path)
            audio_result = decoder.get_all_samples()
            return {
                "raw": audio_result.data.squeeze().numpy(),
                "sampling_rate": audio_result.sample_rate,
            }
        finally:
            Path(temp_path).unlink()

    def _forward(self, model_inputs, **generate_kwargs) -> dict[str, Any]:
        task: str | None = generate_kwargs.pop("task", None)

        # Task-specific defaults
        task_params: dict[str, dict[str, Any]] = {
            "transcribe": {"do_sample": False},
            "emotion": {"do_sample": True, "temperature": 0.7},
            "describe": {"do_sample": True, "temperature": 0.7},
            "continue": {"do_sample": True, "temperature": 1.0},
        }
        if task is not None and task in task_params:
            for key, value in task_params[task].items():
                generate_kwargs.setdefault(key, value)

        # Extract audio from model_inputs
        audio_inputs, is_whisper = self._extract_audio(model_inputs)
        is_last = model_inputs.pop("is_last", True) if isinstance(model_inputs, dict) else True

        # Generation defaults
        generate_kwargs.setdefault("eos_token_id", self.model.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        generate_kwargs.setdefault("max_new_tokens", self.model.config.max_new_tokens)

        # Generate
        if is_whisper:
            generated_ids = self.model.generate(
                input_features=audio_inputs,
                task=task,
                **generate_kwargs,
            )
        else:
            generated_ids = self.model.generate(
                input_values=audio_inputs,
                task=task,
                **generate_kwargs,
            )

        return {"tokens": generated_ids, "is_last": is_last}

    def _extract_audio(self, model_inputs) -> tuple[torch.Tensor, bool]:
        """Extract audio tensor from various input formats."""
        if isinstance(model_inputs, torch.Tensor):
            return model_inputs.to(self.model.device), False

        if isinstance(model_inputs, (list, tuple)) and model_inputs:
            model_inputs = model_inputs[0] if isinstance(model_inputs[0], dict) else {"input_values": model_inputs[0]}

        if isinstance(model_inputs, dict):
            model_inputs.pop("stride", None)
            if "input_features" in model_inputs:
                return model_inputs["input_features"].to(self.model.device), True
            if "input_values" in model_inputs:
                return model_inputs["input_values"].to(self.model.device), False

        raise ValueError(f"Could not extract audio from {type(model_inputs)}")

    def _process_text_only(self, generate_kwargs: dict) -> dict[str, str]:
        """Process text-only input without audio."""
        text_input = generate_kwargs.pop("text_input", None)
        if text_input is None:
            raise ValueError("text_input required for text task")

        generate_kwargs.pop("task", None)
        generated_ids = self.model.generate(task="text", text_input=text_input, **generate_kwargs)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {"text": text}

    def postprocess(self, model_outputs: dict[str, Any], return_timestamps=None, return_language=None) -> dict[str, str]:
        if isinstance(model_outputs, list):
            for output in model_outputs:
                for key, value in output.items():
                    if torch.is_tensor(value):
                        output[key] = value.cpu()
            return super().postprocess(model_outputs)

        model_outputs.pop("is_last", None)
        tokens = model_outputs.get("tokens") or model_outputs.get("generated_ids")

        if tokens is None:
            raise ValueError(f"Expected 'tokens' or 'generated_ids', got: {model_outputs.keys()}")

        if torch.is_tensor(tokens) and tokens.device.type != "cpu":
            tokens = tokens.cpu()
        if len(tokens.shape) > 1:
            tokens = tokens[0]

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        text = self.text_normalizer.normalize(text)
        text = get_true_case(text)

        return {"text": text}
