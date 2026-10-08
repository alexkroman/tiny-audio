"""Evaluator classes for ASR evaluation."""

from scripts.eval.audio import TextNormalizer
from scripts.eval.constants import AssemblyAIModel

from .asr import (
    AppleSpeechEvaluator,
    AssemblyAIEvaluator,
    AssemblyAINemotronEvaluator,
    AssemblyAIStreamingEvaluator,
    DeepgramEvaluator,
    ElevenLabsEvaluator,
    EndpointEvaluator,
    LocalEvaluator,
    LocalStreamingEvaluator,
    NemotronQwenEvaluator,
    SmallestEvaluator,
    SpeakerASREvaluator,
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
    "AssemblyAINemotronEvaluator",
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
    "NemotronQwenEvaluator",
    "SmallestEvaluator",
    "SpeakerASREvaluator",
    "SwiftSDKEvaluator",
    "TextNormalizer",
    "setup_assemblyai",
]
