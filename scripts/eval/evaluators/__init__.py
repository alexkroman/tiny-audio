"""Evaluator classes for ASR, diarization, and alignment tasks."""

from scripts.eval.audio import TextNormalizer

from .alignment import (
    AssemblyAIAlignmentEvaluator,
    BaseAlignmentEvaluator,
    DeepgramAlignmentEvaluator,
    TimestampAlignmentEvaluator,
    align_words_to_reference,
)
from .asr import (
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramEvaluator,
    EndpointEvaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
)
from .base import (
    AlignmentResult,
    DiarizationResult,
    EvalResult,
    Evaluator,
    setup_assemblyai,
)
from .diarization import (
    AssemblyAIDiarizationEvaluator,
    DeepgramDiarizationEvaluator,
    DiarizationEvaluator,
    LocalDiarizationEvaluator,
)

__all__ = [
    # Result types
    "EvalResult",
    "DiarizationResult",
    "AlignmentResult",
    # Base
    "Evaluator",
    "TextNormalizer",
    "setup_assemblyai",
    # ASR evaluators
    "LocalEvaluator",
    "LocalStreamingEvaluator",
    "EndpointEvaluator",
    "AssemblyAIEvaluator",
    "AssemblyAIStreamingEvaluator",
    "DeepgramEvaluator",
    # Diarization evaluators
    "DiarizationEvaluator",
    "AssemblyAIDiarizationEvaluator",
    "DeepgramDiarizationEvaluator",
    "LocalDiarizationEvaluator",
    # Alignment evaluators
    "BaseAlignmentEvaluator",
    "TimestampAlignmentEvaluator",
    "AssemblyAIAlignmentEvaluator",
    "DeepgramAlignmentEvaluator",
    "align_words_to_reference",
]
