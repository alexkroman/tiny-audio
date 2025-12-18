from typing import Any

import torch
import transformers

try:
    from .asr_modeling import ASRModel
except ImportError:
    from asr_modeling import ASRModel  # type: ignore[no-redef]


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

    def preprocess(self, inputs, **preprocess_params):
        preprocess_params.setdefault("chunk_length_s", 0)

        # Handle dict with "array" key (from datasets)
        if isinstance(inputs, dict) and "array" in inputs:
            inputs = {
                "raw": inputs["array"],
                "sampling_rate": inputs.get("sampling_rate", self.feature_extractor.sampling_rate),
            }

        return super().preprocess(inputs, **preprocess_params)

    def _forward(self, model_inputs, **generate_kwargs) -> dict[str, Any]:
        # Extract audio features
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

        return {"tokens": generated_ids}

    def postprocess(self, model_outputs, **kwargs) -> dict[str, str]:
        tokens = model_outputs.get("tokens")
        if tokens is None:
            return super().postprocess(model_outputs, **kwargs)

        if torch.is_tensor(tokens):
            tokens = tokens.cpu()
            if tokens.dim() > 1:
                tokens = tokens[0]

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return {"text": text}
