"""Evaluator classes for ASR evaluation."""

from scripts.eval.audio import TextNormalizer
from scripts.eval.constants import AssemblyAIModel

from .asr import (
    AppleSpeechEvaluator,
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramEvaluator,
    ElevenLabsEvaluator,
    EndpointEvaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
    SwiftSDKEvaluator,
)
from .base import (
    EvalResult,
    Evaluator,
    setup_assemblyai,
)

__all__ = [
    "AppleSpeechEvaluator",
    "AssemblyAIEvaluator",
    "AssemblyAIModel",
    "AssemblyAIStreamingEvaluator",
    "DeepgramEvaluator",
    "ElevenLabsEvaluator",
    "EndpointEvaluator",
    # Result types
    "EvalResult",
    # Base
    "Evaluator",
    # ASR evaluators
    "LocalEvaluator",
    "LocalStreamingEvaluator",
    "SwiftSDKEvaluator",
    "TextNormalizer",
    "setup_assemblyai",
]
