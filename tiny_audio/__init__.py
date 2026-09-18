"""Tiny Audio: frozen GLM-ASR encoder + trainable MLP projector + Qwen3-0.6B decoder.

A minimal, hackable ASR model. The encoder stays frozen; the projector and
decoder train jointly (see configs/experiments/stage_1.yaml).
"""

__version__ = "0.1.0"

# Import pipeline, processor, and diarization to register them with transformers
from . import asr_pipeline, asr_processing, diarization
from .asr_modeling import ASRModel

__all__ = ["ASRModel", "asr_pipeline", "asr_processing", "diarization"]
