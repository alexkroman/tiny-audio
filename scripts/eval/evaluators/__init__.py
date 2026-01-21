"""Evaluator classes for ASR, diarization, and alignment tasks."""

from scripts.eval.audio import TextNormalizer

from .alignment import (
    AssemblyAIAlignmentEvaluator,
    BaseAlignmentEvaluator,
    DeepgramAlignmentEvaluator,
    ElevenLabsAlignmentEvaluator,
    TimestampAlignmentEvaluator,
    align_words_to_reference,
)
from .asr import (
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramEvaluator,
    ElevenLabsEvaluator,
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
    ElevenLabsDiarizationEvaluator,
    LocalDiarizationEvaluator,
)
from .mcq import (
    AssemblyAIMMAUEvaluator,
    MCQResult,
    MMAUEvaluator,
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
    "ElevenLabsEvaluator",
    # Diarization evaluators
    "DiarizationEvaluator",
    "AssemblyAIDiarizationEvaluator",
    "DeepgramDiarizationEvaluator",
    "ElevenLabsDiarizationEvaluator",
    "LocalDiarizationEvaluator",
    # Alignment evaluators
    "BaseAlignmentEvaluator",
    "TimestampAlignmentEvaluator",
    "AssemblyAIAlignmentEvaluator",
    "DeepgramAlignmentEvaluator",
    "ElevenLabsAlignmentEvaluator",
    "align_words_to_reference",
    # MCQ evaluators
    "MCQResult",
    "MMAUEvaluator",
    "AssemblyAIMMAUEvaluator",
]
