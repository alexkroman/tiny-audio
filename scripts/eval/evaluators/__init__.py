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
