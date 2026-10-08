"""Evaluator classes for ASR evaluation."""

from scripts.eval.audio import TextNormalizer
from scripts.eval.constants import AssemblyAIModel

from .apple_speech import AppleSpeechEvaluator
from .asr import (
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramEvaluator,
    ElevenLabsEvaluator,
    EndpointEvaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
    SmallestEvaluator,
)
from .base import (
    EvalResult,
    Evaluator,
    setup_assemblyai,
)
from .swift_sdk import SwiftSDKEvaluator

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
    "SmallestEvaluator",
    "SwiftSDKEvaluator",
    "TextNormalizer",
    "setup_assemblyai",
]
