from typing import Any, Dict

import torch
import transformers

try:
    from .asr_modeling import ASRModel
except ImportError:
    from asr_modeling import ASRModel  # type: ignore[no-redef]


class ASRPipeline(transformers.AutomaticSpeechRecognitionPipeline):
    model: ASRModel

    def __init__(self, model: ASRModel, **kwargs):
        feature_extractor = kwargs.pop("feature_extractor", model.feature_extractor)
        tokenizer = kwargs.pop("tokenizer", model.tokenizer)

        super().__init__(
            model=model, feature_extractor=feature_extractor, tokenizer=tokenizer, **kwargs
        )

    def __call__(self, inputs, **kwargs):
        generate_kwargs = {}
        for key in [
            "max_new_tokens",
            "num_beams",
            "temperature",
            "do_sample",
            "length_penalty",
            "repetition_penalty",
            "top_p",
            "top_k",
        ]:
            if key in kwargs:
                generate_kwargs[key] = kwargs.pop(key)
        if isinstance(inputs, list):
            results = []
            for single_input in inputs:
                result = self.__call__(single_input, **kwargs, **generate_kwargs)
                results.append(result)
            return results

        model_inputs = self.preprocess(inputs, **kwargs)

        from collections.abc import Iterator

        if isinstance(model_inputs, Iterator):
            all_outputs = []
            for chunk in model_inputs:
                chunk_output = self._forward(chunk, **generate_kwargs)
                all_outputs.append(chunk_output)

            transcriptions = []
            for output in all_outputs:
                chunk_result = self.postprocess(output)
                transcriptions.append(chunk_result.get("text", ""))

            return {"text": " ".join(transcriptions).strip()}
        model_outputs = self._forward(model_inputs, **generate_kwargs)
        return self.postprocess(model_outputs)

    def preprocess(self, inputs, **preprocess_params):
        if isinstance(inputs, list):
            raise ValueError("Lists should not reach preprocess - bug in __call__")

        # Handle different formats from datasets
        if isinstance(inputs, dict):
            if "bytes" in inputs:
                # Decode bytes to audio array using torchcodec
                import tempfile
                from torchcodec.decoders import AudioDecoder

                wav_bytes = inputs["bytes"]
                # Write to temp file for torchcodec to read
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav_bytes)
                    temp_path = f.name
                try:
                    decoder = AudioDecoder(temp_path)
                    # Get all audio samples
                    audio_result = decoder.get_all_samples()
                    audio_tensor = audio_result.data
                    sample_rate = audio_result.sample_rate
                    inputs = {"raw": audio_tensor.squeeze().numpy(), "sampling_rate": sample_rate}
                finally:
                    import os
                    os.unlink(temp_path)
            elif "array" in inputs:
                # Convert "array" key to "raw" key
                inputs = {"raw": inputs["array"], "sampling_rate": inputs["sampling_rate"]}
            # If it already has "raw" and "sampling_rate", it's good to go
        elif hasattr(inputs, "array") and hasattr(inputs, "sampling_rate"):
            # Audio object with attributes (not dict)
            inputs = {"raw": inputs.array, "sampling_rate": inputs.sampling_rate}
        elif hasattr(inputs, "__array__") and not isinstance(inputs, (dict, bytes, str)):
            inputs = {"raw": inputs, "sampling_rate": 16000}
        elif torch.is_tensor(inputs):
            inputs = {"raw": inputs.cpu().numpy(), "sampling_rate": 16000}

        return super().preprocess(inputs, **preprocess_params)

    def _forward(self, model_inputs, **generate_kwargs):
        is_last = True
        input_values = None

        if isinstance(model_inputs, torch.Tensor):
            input_values = model_inputs
        elif isinstance(model_inputs, list):
            if len(model_inputs) > 0:
                first_item = model_inputs[0]
                if isinstance(first_item, dict):
                    if "is_last" in first_item:
                        is_last = first_item.pop("is_last")
                    if "stride" in first_item:
                        first_item.pop("stride")
                    input_values = first_item.get("input_values")
                elif isinstance(first_item, torch.Tensor):
                    input_values = first_item
            else:
                raise ValueError("Received empty list in _forward")
        elif isinstance(model_inputs, dict):
            if "is_last" in model_inputs:
                is_last = model_inputs.pop("is_last")
            if "stride" in model_inputs:
                model_inputs.pop("stride")
            input_values = model_inputs.get("input_values")
        else:
            input_values = model_inputs

        if input_values is None:
            raise ValueError(f"Could not extract input_values from {type(model_inputs)}")

        if isinstance(input_values, torch.Tensor):
            input_values = input_values.to(self.model.device)
        else:
            raise ValueError(f"input_values must be a tensor, got {type(input_values)}")

        im_end_id = self.model.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)

        generated_ids = self.model.generate(
            input_values, system_prompt=self.model.config.system_prompt, **generate_kwargs
        )

        return {"tokens": generated_ids, "is_last": is_last}

    def postprocess(
        self, model_outputs: Dict[str, Any], return_timestamps=None, return_language=None
    ):
        if "is_last" in model_outputs:
            model_outputs.pop("is_last")

        tokens = model_outputs.get("tokens")
        if tokens is None:
            tokens = model_outputs.get("generated_ids")

        if tokens is None:
            raise ValueError(
                f"Expected 'tokens' or 'generated_ids' in model_outputs, got: {model_outputs.keys()}"
            )

        if len(tokens.shape) > 1:
            tokens = tokens[0]

        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.strip()

        return {"text": text}
