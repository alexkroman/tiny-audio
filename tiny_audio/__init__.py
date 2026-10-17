"""Tiny Audio: frozen speech encoder + trainable MLP projector + LLM decoder.

A minimal, hackable ASR model. The default recipe (configs/config.yaml) keeps
a Granite Speech encoder and a Qwen3.5 decoder frozen and trains the projector
plus LoRA adapters on the decoder.
"""

__version__ = "0.1.0"

# Import pipeline, processor, and diarization to register them with transformers
from . import asr_pipeline, asr_processing, diarization
from .asr_modeling import ASRModel

__all__ = ["ASRModel", "asr_pipeline", "asr_processing", "diarization"]
