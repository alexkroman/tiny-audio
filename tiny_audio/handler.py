"""Custom inference handler for HuggingFace Inference Endpoints."""

from typing import Any, Dict, List, Union

import torch

try:
    # For remote execution, imports are relative
    from .asr_modeling import ASRModel
    from .asr_pipeline import ASRPipeline
except ImportError:
    # For local execution, imports are not relative
    from asr_modeling import ASRModel  # type: ignore[no-redef]
    from asr_pipeline import ASRPipeline  # type: ignore[no-redef]


class EndpointHandler:
    def __init__(self, path: str = ""):
        import os

        import nltk

        nltk.download("punkt_tab", quiet=True)

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Enable TF32 for faster matmul on Ampere+ GPUs (A100, etc.)
        # Also beneficial for T4 (Turing) which supports TensorFloat-32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use float16 for better T4 compatibility (bfloat16 not well supported on T4)
        # T4 has excellent float16 performance with tensor cores
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Prepare model kwargs for pipeline
        model_kwargs = {
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = (
                "flash_attention_2" if self._is_flash_attn_available() else "sdpa"
            )

        # Load model (this loads the model, tokenizer, and feature extractor)
        self.model = ASRModel.from_pretrained(path, **model_kwargs)

        # Instantiate custom pipeline - it will get feature_extractor and tokenizer from model
        self.pipe = ASRPipeline(
            model=self.model,
            feature_extractor=self.model.feature_extractor,
            tokenizer=self.model.tokenizer,
            device=self.device,
        )

        # Apply torch.compile if enabled (after model is loaded by pipeline)
        # Use "default" mode for T4 - better compatibility than "reduce-overhead"
        # "reduce-overhead" is better for A100+ but can be slower on older GPUs
        if torch.cuda.is_available() and os.getenv("ENABLE_TORCH_COMPILE", "1") == "1":
            compile_mode = os.getenv("TORCH_COMPILE_MODE", "default")
            self.model = torch.compile(self.model, mode=compile_mode)
            self.pipe.model = self.model

        # Warmup the model to trigger compilation and optimize kernels
        if torch.cuda.is_available():
            self._warmup()

    def _is_flash_attn_available(self):
        """Check if flash attention is available."""
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None

    def _warmup(self):
        """Warmup to trigger model compilation and allocate GPU memory."""
        try:
            # Create dummy audio (1 second at config sample rate)
            sample_rate = self.pipe.model.config.audio_sample_rate
            dummy_audio = torch.randn(sample_rate, dtype=torch.float32)

            # Run inference to trigger torch.compile and kernel optimization
            with torch.inference_mode():
                warmup_tokens = self.pipe.model.config.inference_warmup_tokens
                _ = self.pipe(
                    {"raw": dummy_audio, "sampling_rate": sample_rate},
                    max_new_tokens=warmup_tokens,
                )

            # Force CUDA synchronization to ensure kernels are compiled
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Clear cache after warmup to free memory
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warmup skipped due to: {e}")

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        inputs = data.get("inputs")
        if inputs is None:
            raise ValueError("Missing 'inputs' in request data")

        # Pass through any parameters from request, let model config provide defaults
        params = data.get("parameters", {})

        return self.pipe(inputs, **params)
