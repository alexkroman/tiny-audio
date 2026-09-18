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


def _best_device() -> torch.device:
    """Best available inference device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

        # `ASRModel.from_pretrained` constructs its own submodules and forwards
        # **kwargs to `ASRModel.__init__`, which discards them -- no loader
        # reads `device_map`, `torch_dtype` or `low_cpu_mem_usage`, and it sets
        # `_is_loading_from_pretrained` precisely to keep `device_map="auto"`
        # out of the sub-model loaders. Passing them here did nothing, so the
        # model stayed on CPU and a GPU Inference Endpoint silently decoded on
        # CPU. Place it explicitly instead. dtype comes from
        # `config.model_dtype` and the attention backend from
        # `config.attn_implementation`, which already downgrades FA2 when
        # flash_attn is missing.
        self.model = ASRModel.from_pretrained(path)
        self.device = _best_device()
        self.model.to(self.device)
        self.model.eval()

        self.pipe = ASRPipeline(
            model=self.model,
            feature_extractor=self.model.feature_extractor,
            tokenizer=self.model.tokenizer,
            device=self.device,
        )

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
