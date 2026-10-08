"""Custom inference handler for HuggingFace Inference Endpoints."""

import os
from typing import TYPE_CHECKING, Any

import nltk

if TYPE_CHECKING:
    from .asr_modeling import ASRModel
    from .asr_pipeline import ASRPipeline
    from .diarization import get_device as _best_device
else:
    try:
        # For remote execution, imports are relative
        from .asr_modeling import ASRModel
        from .asr_pipeline import ASRPipeline
        from .diarization import get_device as _best_device
    except ImportError:
        # For local execution, imports are not relative
        from asr_modeling import ASRModel
        from asr_pipeline import ASRPipeline
        from diarization import get_device as _best_device


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
        # Only fetch when absent: nltk.download re-checks the remote index
        # every call, i.e. a network round-trip on every endpoint start.
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
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
        # PreTrainedModel.to is functools.wraps'd, which pyright cannot bind as a method.
        self.model.to(self.device)  # pyright: ignore[reportArgumentType]
        self.model.eval()

        self.pipe = ASRPipeline(
            model=self.model,
            feature_extractor=self.model.feature_extractor,
            tokenizer=self.model.tokenizer,
            device=self.device,
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """Process an inference request.

        Args:
            data: Request data containing 'inputs' (audio path/bytes) and optional 'parameters'

        Returns:
            Transcription result with 'text' key
        """
        inputs = data.get("inputs")
        if inputs is None:
            msg = "Missing 'inputs' in request data"
            raise ValueError(msg)

        # Pass through any parameters from request, let model config provide defaults
        params = data.get("parameters", {})

        result: dict[str, Any] | list[dict[str, Any]] = self.pipe(inputs, **params)
        return result
