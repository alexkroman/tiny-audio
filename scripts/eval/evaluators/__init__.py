"""Evaluator classes for ASR, diarization, and alignment tasks."""

from scripts.eval.audio import TextNormalizer

from .alignment import (
    AssemblyAIAlignmentEvaluator,
    BaseAlignmentEvaluator,
    TimestampAlignmentEvaluator,
    align_words_to_reference,
)
from .asr import (
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
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
from .diarization import AssemblyAIDiarizationEvaluator, DiarizationEvaluator

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
    # Diarization evaluators
    "DiarizationEvaluator",
    "AssemblyAIDiarizationEvaluator",
    # Alignment evaluators
    "BaseAlignmentEvaluator",
    "TimestampAlignmentEvaluator",
    "AssemblyAIAlignmentEvaluator",
    "align_words_to_reference",
]
