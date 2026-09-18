"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock

import pytest
import torch

# Disable tokenizers parallelism to avoid fork warnings in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Mock Factories - Reusable mock components for testing
# =============================================================================


@pytest.fixture
def mock_feature_extractor():
    """Standard mock feature extractor used across tests."""
    fe = MagicMock()
    fe.sampling_rate = 16000
    fe.return_value = {
        "input_features": torch.randn(1, 80, 100),
        "attention_mask": torch.ones(1, 100),
    }
    return fe


@pytest.fixture
def mock_tokenizer():
    """Standard mock tokenizer."""
    tok = MagicMock()
    tok.convert_tokens_to_ids.return_value = 12345
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok.bos_token_id = 1
    tok.decode.return_value = "decoded text"
    tok.batch_decode.return_value = ["decoded text"]
    return tok


@pytest.fixture
def mock_projector():
    """Standard mock projector."""
    proj = MagicMock()
    proj.get_output_length.return_value = 100
    return proj


# =============================================================================
# Deepgram Mock Factories
# =============================================================================


def build_deepgram_word(word: str, start: float, end: float):
    """Factory for Deepgram word mocks."""
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    return w


def build_deepgram_transcription_response(transcript: str = "", words: list | None = None):
    """Factory for Deepgram transcription API response mocks.

    Args:
        transcript: The full transcript text
        words: List of dicts with keys: word, start, end

    Returns:
        MagicMock configured as a Deepgram response
    """
    response = MagicMock()
    alternative = MagicMock()
    alternative.transcript = transcript

    if words:
        alternative.words = [build_deepgram_word(w["word"], w["start"], w["end"]) for w in words]
    else:
        alternative.words = None if words is None else []

    channel = MagicMock()
    channel.alternatives = [alternative]
    response.results.channels = [channel]
    return response


@pytest.fixture
def deepgram_transcription_response():
    """Sample Deepgram transcription response with word timestamps."""
    return build_deepgram_transcription_response(
        transcript="hello world",
        words=[
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ],
    )


@pytest.fixture(scope="session")
def base_asr_config():
    """Session-scoped base ASR config (no LoRA) - loaded once per test session."""
    from tiny_audio.asr_config import ASRConfig

    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
    )


@pytest.fixture(scope="session")
def base_asr_model(base_asr_config):
    """Session-scoped base ASR model - loaded once per test session."""
    from tiny_audio.asr_modeling import ASRModel

    return ASRModel(base_asr_config)


@pytest.fixture(scope="session")
def lora_asr_config():
    """Session-scoped LoRA ASR config - loaded once per test session."""
    from tiny_audio.asr_config import ASRConfig

    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )


@pytest.fixture(scope="session")
def lora_asr_model(lora_asr_config):
    """Session-scoped LoRA ASR model - loaded once per test session."""
    from tiny_audio.asr_modeling import ASRModel

    return ASRModel(lora_asr_config)


# =============================================================================
# Projector Test Utilities
# =============================================================================


class MockProjectorConfig:
    """Mock config for projector initialization in tests.

    Provides all config attributes needed by projector classes with sensible defaults.
    Override any attribute by passing kwargs to __init__.
    """

    def __init__(self, **kwargs):
        # Core dimensions
        self.encoder_dim = kwargs.get("encoder_dim", 256)
        self.llm_dim = kwargs.get("llm_dim", 512)
        self.projector_hidden_dim = kwargs.get("projector_hidden_dim", 1024)
        self.projector_pool_stride = kwargs.get("projector_pool_stride", 4)


@pytest.fixture
def projector_config():
    """Factory fixture for creating projector configs with custom settings."""
    return MockProjectorConfig
