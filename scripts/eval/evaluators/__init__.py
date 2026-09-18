"""Evaluator classes for ASR evaluation."""

from scripts.eval.audio import TextNormalizer

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
    AssemblyAIModel,
    EvalResult,
    Evaluator,
    setup_assemblyai,
)

__all__ = [
    "AssemblyAIModel",
    # Result types
    "EvalResult",
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
    "AppleSpeechEvaluator",
    "SwiftSDKEvaluator",
]
