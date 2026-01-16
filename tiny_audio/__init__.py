"""SLAM-ASR: Frozen HuBERT + Trainable Projector + Frozen Qwen3-8B

Simple direct audio-to-text mapping without chat templates.
"""

__version__ = "0.1.0"

# Import pipeline, processor, and diarization to register them with transformers
from . import (
    asr_pipeline,  # noqa: F401
    asr_processing,  # noqa: F401
    diarization,  # noqa: F401
)
