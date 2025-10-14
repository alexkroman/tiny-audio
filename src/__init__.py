"""SLAM-ASR: Frozen HuBERT + Trainable Projector + Frozen SmolLM3-3B-Base

Simple direct audio-to-text mapping without chat templates.
"""

__version__ = "0.1.0"

# Import pipeline and processor to register them with transformers
from . import (
    asr_pipeline,  # noqa: F401
    asr_processing,  # noqa: F401
)
