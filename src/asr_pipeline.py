from typing import Any, Dict

import torch
import transformers
from truecase import get_true_case

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

        # Initialize text normalizer (same as train.py)
        if hasattr(tokenizer, "normalize"):
            self.text_normalizer = tokenizer
        else:
            # Fallback to whisper-tiny tokenizer for its normalize() method only
            from transformers import WhisperTokenizer

            self.text_normalizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

    def __call__(self, inputs, **kwargs):
        generate_kwargs = {}
        for key in [
            "max_new_tokens",
            "num_beams",
            "do_sample",
            "length_penalty",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "early_stopping",
            "num_beam_groups",
            "diversity_penalty",
            "top_k",
            "temperature",
            "top_p",
            "user_prompt",
            "task",
            "text_input",
        ]:
            if key in kwargs:
                generate_kwargs[key] = kwargs.pop(key)

        # Handle text-only mode
        task = generate_kwargs.get("task")
        if task == "text" or generate_kwargs.get("text_input"):
            return self._process_text_only(generate_kwargs)

        if isinstance(inputs, list):
            results = []
            for single_input in inputs:
                result = self.__call__(single_input, **kwargs, **generate_kwargs)
                results.append(result)
            return results

        model_inputs = self.preprocess(inputs, **kwargs)

        from collections.abc import Iterator

        if isinstance(model_inputs, Iterator):
            # Convert iterator to list to process chunks
            chunks = list(model_inputs)

            all_outputs = []
            for _chunk_num, chunk in enumerate(chunks, start=1):
                chunk_output = self._forward(chunk, **generate_kwargs)
                # Move tensors to CPU before adding to outputs
                for key, value in chunk_output.items():
                    if torch.is_tensor(value):
                        chunk_output[key] = value.cpu()
                all_outputs.append(chunk_output)

            # Merge chunks and decode ourselves to ensure skip_special_tokens=True
            all_tokens: list[int] = []
            for output in all_outputs:
                tokens = output.get("tokens")
                if tokens is None:
                    tokens = output.get("generated_ids")
                if tokens is not None:
                    if torch.is_tensor(tokens):
                        tokens = tokens.cpu()
                    if len(tokens.shape) > 1:
                        tokens = tokens[0]
                    all_tokens.extend(tokens.tolist() if torch.is_tensor(tokens) else tokens)

            # Decode the merged tokens with skip_special_tokens
            text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
            text = text.strip()

            # Apply Whisper normalization (matches training)
            text = self.text_normalizer.normalize(text)

            # Apply truecasing for proper capitalization
            text = get_true_case(text)

            return {"text": text}

        model_outputs = self._forward(model_inputs, **generate_kwargs)
        return self.postprocess(model_outputs)

    def preprocess(self, inputs, **preprocess_params):
        if isinstance(inputs, list):
            raise ValueError("Lists should not reach preprocess - bug in __call__")

        # Set default chunking to 30 seconds with 5 second overlap
        preprocess_params.setdefault("chunk_length_s", 30)
        preprocess_params.setdefault("stride_length_s", (5, 5))

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
                    from pathlib import Path

                    Path(temp_path).unlink()
            elif "array" in inputs:
                # Convert "array" key to "raw" key
                inputs = {"raw": inputs["array"], "sampling_rate": inputs["sampling_rate"]}
            # If it already has "raw" and "sampling_rate", it's good to go
        elif hasattr(inputs, "array") and hasattr(inputs, "sampling_rate"):
            # Audio object with attributes (not dict)
            inputs = {"raw": inputs.array, "sampling_rate": inputs.sampling_rate}
        elif hasattr(inputs, "__array__") and not isinstance(inputs, (dict, bytes, str)):
            inputs = {"raw": inputs, "sampling_rate": self.model.config.audio_sample_rate}
        elif torch.is_tensor(inputs):
            inputs = {
                "raw": inputs.cpu().numpy(),
                "sampling_rate": self.model.config.audio_sample_rate,
            }

        return super().preprocess(inputs, **preprocess_params)

    def _forward(self, model_inputs, **generate_kwargs):
        # Extract task and set sampling parameters
        task = generate_kwargs.pop("task", None)

        # Task-specific sampling parameters
        task_params: Dict[str, Dict[str, Any]] = {
            "transcribe": {"do_sample": False},
            "emotion": {"do_sample": True, "temperature": 0.7},
            "describe": {"do_sample": True, "temperature": 0.7},
            "continue": {"do_sample": True, "temperature": 1.0},
        }

        if task in task_params:
            for key, value in task_params[task].items():
                generate_kwargs.setdefault(key, value)

        # Extract audio inputs from various formats
        is_last = True
        audio_inputs = None
        is_whisper = False  # Track if this is Whisper input

        # Normalize model_inputs to dict format
        if isinstance(model_inputs, torch.Tensor):
            audio_inputs = model_inputs
        elif isinstance(model_inputs, (list, tuple)) and model_inputs:
            model_inputs = (
                model_inputs[0]
                if isinstance(model_inputs[0], dict)
                else {"input_values": model_inputs[0]}
            )

        if isinstance(model_inputs, dict):
            # Pop metadata fields
            is_last = model_inputs.pop("is_last", True)
            model_inputs.pop("stride", None)
            # Get audio input (Whisper uses input_features, others use input_values)
            if "input_features" in model_inputs:
                audio_inputs = model_inputs["input_features"]
                is_whisper = True
            else:
                audio_inputs = model_inputs.get("input_values")

        if audio_inputs is None:
            raise ValueError(
                f"Could not extract input_values or input_features from {type(model_inputs)}"
            )

        if isinstance(audio_inputs, torch.Tensor):
            audio_inputs = audio_inputs.to(self.model.device)
        else:
            raise ValueError(f"audio inputs must be a tensor, got {type(audio_inputs)}")

        im_end_id = self.model.tokenizer.convert_tokens_to_ids("<|im_end|>")
        generate_kwargs.setdefault("eos_token_id", im_end_id)
        generate_kwargs.setdefault("max_new_tokens", self.model.config.max_new_tokens)

        # Pass the appropriate input type to generate
        if is_whisper:
            # Whisper model - use input_features
            generated_ids = self.model.generate(
                input_features=audio_inputs,
                system_prompt=self.model.config.system_prompt,
                task=task,
                **generate_kwargs,
            )
        else:
            # Wav2Vec2/HuBERT model - use input_values
            generated_ids = self.model.generate(
                input_values=audio_inputs,
                system_prompt=self.model.config.system_prompt,
                task=task,
                **generate_kwargs,
            )

        return {"tokens": generated_ids, "is_last": is_last}

    def _process_text_only(self, generate_kwargs):
        """Process text-only input without audio encoding."""
        text_input = generate_kwargs.pop("text_input", None)
        if text_input is None:
            raise ValueError("text_input is required for text task")

        # Remove task from generate_kwargs to avoid duplicate argument
        generate_kwargs.pop("task", None)

        # Generate text using the model
        generated_ids = self.model.generate(task="text", text_input=text_input, **generate_kwargs)

        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {"text": generated_text}

    def postprocess(
        self, model_outputs: Dict[str, Any], return_timestamps=None, return_language=None
    ):
        # Handle chunked outputs from iterator
        if isinstance(model_outputs, list):
            # Move all tensors to CPU before calling parent postprocess
            for output_dict in model_outputs:
                for key, value in output_dict.items():
                    if torch.is_tensor(value):
                        output_dict[key] = value.cpu()
            return super().postprocess(model_outputs)

        if "is_last" in model_outputs:
            model_outputs.pop("is_last")

        tokens = model_outputs.get("tokens")
        if tokens is None:
            tokens = model_outputs.get("generated_ids")

        if tokens is None:
            raise ValueError(
                f"Expected 'tokens' or 'generated_ids' in model_outputs, got: {model_outputs.keys()}"
            )

        # Move to CPU if on MPS or other device
        if torch.is_tensor(tokens) and tokens.device.type != "cpu":
            tokens = tokens.cpu()

        if len(tokens.shape) > 1:
            tokens = tokens[0]

        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        text = text.strip()

        # Apply Whisper normalization (matches training)
        text = self.text_normalizer.normalize(text)

        # Apply truecasing for proper capitalization
        text = get_true_case(text)

        return {"text": text}
