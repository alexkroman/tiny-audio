"""Custom inference handler for HuggingFace Inference Endpoints."""

from typing import Any, Dict, List, Union

try:
    # For remote execution, imports are relative
    from .asr_modeling import ASRModel
    from .asr_pipeline import ASRPipeline
except ImportError:
    # For local execution, imports are not relative
    from asr_modeling import ASRModel  # type: ignore[no-redef]
    from asr_pipeline import ASRPipeline  # type: ignore[no-redef]


class EndpointHandler:
    """HuggingFace Inference Endpoints handler for ASR model.

    Handles model loading, warmup, and inference requests for deployment
    on HuggingFace Inference Endpoints or similar services.
    """

    def __init__(self, path: str = ""):
        """Initialize the endpoint handler.

        Args:
            path: Path to model directory or HuggingFace model ID
        """
        import os

        import nltk

        nltk.download("punkt_tab", quiet=True)

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Prepare model kwargs - let transformers handle device placement
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
        if self._is_flash_attn_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model (this loads the model, tokenizer, and feature extractor)
        self.model = ASRModel.from_pretrained(path, **model_kwargs)

        # Get device from model for pipeline
        self.device = next(self.model.parameters()).device

        # Instantiate custom pipeline - it will get feature_extractor and tokenizer from model
        self.pipe = ASRPipeline(
            model=self.model,
            feature_extractor=self.model.feature_extractor,
            tokenizer=self.model.tokenizer,
            device=self.device,
        )

    def _is_flash_attn_available(self):
        """Check if flash attention is available."""
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None

    def __call__(self, data: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Process an inference request.

        Args:
            data: Request data containing 'inputs' (audio path/bytes) and optional 'parameters'

        Returns:
            Transcription result with 'text' key
        """
        inputs = data.get("inputs")
        if inputs is None:
            raise ValueError("Missing 'inputs' in request data")

        # Pass through any parameters from request, let model config provide defaults
        params = data.get("parameters", {})

        return self.pipe(inputs, **params)
