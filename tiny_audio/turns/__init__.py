"""Turn-aware ASR runtime: Qwen3-ASR that transcribes AND says when the turn is over.

    from tiny_audio.turns import load_model, load_processor, transcribe

    processor = load_processor("mazesmazes/tiny-audio-turn-aware-qwen3-asr")
    model = load_model("mazesmazes/tiny-audio-turn-aware-qwen3-asr").eval()
    text, fired, margin = transcribe(model, processor, [audio_16k])[0]

Self-contained on purpose: imports only numpy, torch and transformers, never
the rest of tiny_audio, so it can be lifted into its own package unchanged.
Training and data tooling live in scripts/turn_aware/.
"""

from .model import (
    END_OF_TURN,
    LANGUAGE,
    MODEL_ID,
    SAMPLE_RATE,
    Decoded,
    load_model,
    load_processor,
    marker_margins,
    pick_device,
    set_end_of_turn_threshold,
    transcribe,
)
from .streaming import first_fire_times, stream_fire_times, trailing_is_silent

__all__ = [
    "END_OF_TURN",
    "LANGUAGE",
    "MODEL_ID",
    "SAMPLE_RATE",
    "Decoded",
    "first_fire_times",
    "load_model",
    "load_processor",
    "marker_margins",
    "pick_device",
    "set_end_of_turn_threshold",
    "stream_fire_times",
    "trailing_is_silent",
    "transcribe",
]
