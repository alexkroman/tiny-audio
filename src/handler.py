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
        # Set environment variables for PyTorch/CUDA (must be before imports/operations)
        import os

        # Enable expandable segments to reduce fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Enable TF32 for faster matmul on A40/A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

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

        # Load model (this loads the model, tokenizer, and feature extractor from checkpoint)
        self.model = ASRModel.from_pretrained(path, **model_kwargs)

        # Use the tokenizer and feature extractor that were loaded with the model
        # This ensures they match the fine-tuned checkpoint
        feature_extractor = self.model.feature_extractor
        tokenizer = self.model.tokenizer

        # Instantiate custom pipeline
        self.pipe = ASRPipeline(
            model=self.model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            device=self.device,
        )

        # Apply torch.compile if enabled (after model is loaded by pipeline)
        # Enable by default for significant speedup (20-40%)
        if torch.cuda.is_available() and os.getenv("ENABLE_TORCH_COMPILE", "1") == "1":
            compile_mode = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
            print(f"⚡ Enabling torch.compile (mode={compile_mode})")
            self.model = torch.compile(self.model, mode=compile_mode)
            # Update the pipeline with the compiled model
            self.pipe.model = self.model

        # Warmup the model
        if torch.cuda.is_available():
            self._warmup()

    def _is_flash_attn_available(self):
        """Check if flash attention is available."""
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None

    def _warmup(self):
        """Warmup to trigger model compilation and allocate GPU memory."""
        print("Warming up model...")
        try:
            # Create dummy audio
            dummy_audio = torch.randn(16000, dtype=torch.float32)

            # The pipeline now handles GPU optimization internally
            with torch.inference_mode():
                _ = self.pipe({"raw": dummy_audio, "sampling_rate": 16000}, max_new_tokens=10)

            # Force CUDA synchronization to ensure kernels are compiled
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Clear cache after warmup to free memory
                torch.cuda.empty_cache()

            print("Model warmup complete!")
        except Exception as e:
            print(f"Warmup skipped due to: {e}")

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process audio transcription request.

        Supports both single and batch inputs for efficient concurrent processing.
        The endpoint infrastructure can batch multiple concurrent requests automatically.
        """
        inputs = data.get("inputs")
        if inputs is None:
            raise ValueError("Missing 'inputs' in request data")

        # Get generation parameters (matching SLAM-ASR paper defaults)
        params = data.get("parameters", {})
        max_new_tokens = params.get("max_new_tokens", 200)  # Longer transcripts

        # Beam search for better quality (5 beams for higher quality)
        # Use num_beams=1 for faster inference at cost of ~2-3% WER increase
        num_beams = params.get("num_beams", 5)

        do_sample = params.get("do_sample", False)

        # Length penalty encourages appropriate transcript length
        # >1.0 = prefer longer outputs, <1.0 = prefer shorter
        # Slight positive bias helps avoid truncated transcripts
        length_penalty = params.get("length_penalty", 1.0)

        # Repetition penalty to prevent loops (1.1-1.2 is good for ASR)
        repetition_penalty = params.get("repetition_penalty", 1.15)

        # Alternative: use no_repeat_ngram_size to prevent exact n-gram repetition
        no_repeat_ngram_size = params.get("no_repeat_ngram_size", 3)

        # Early stopping for beam search: stop when all beams end
        # "never" = generate full max_new_tokens (more accurate but slower)
        # True = stop when all beams reach EOS (faster)
        early_stopping = params.get("early_stopping", True)

        # Diversity penalty encourages different beams (helps with rare words)
        # 0.0 = no diversity, 0.5-1.0 = good diversity
        num_beam_groups = params.get("num_beam_groups", 1)
        diversity_penalty = params.get("diversity_penalty", 0.0)

        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 1.0)

        # The pipeline's __call__ method handles both single and batch inputs
        # as well as automatic chunking for long audio files
        return self.pipe(
            inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_p=top_p,
        )
