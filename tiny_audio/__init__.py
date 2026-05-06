"""SLAM-ASR: Frozen HuBERT + Trainable Projector + Frozen Qwen3-8B

Simple direct audio-to-text mapping without chat templates.
"""

__version__ = "0.1.0"

# Import pipeline, processor, and diarization to register them with transformers
from . import asr_pipeline, asr_processing, diarization
from .asr_modeling import ASRModel

__all__ = ["ASRModel", "asr_pipeline", "asr_processing", "diarization"]
