"""Evaluator classes for ASR evaluation."""

from scripts.eval.audio import TextNormalizer
from scripts.eval.constants import AssemblyAIModel

from .apple_speech import AppleSpeechEvaluator
from .asr import (
    AssemblyAIEvaluator,
    AssemblyAIStreamingEvaluator,
    AssemblyAIStreamsEvaluator,
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

__all__ = [
    "AppleSpeechEvaluator",
    "AssemblyAIEvaluator",
    "AssemblyAIModel",
    "AssemblyAIStreamingEvaluator",
    "AssemblyAIStreamsEvaluator",
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
    "TextNormalizer",
    "setup_assemblyai",
]
