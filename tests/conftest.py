"""Pytest configuration and fixtures."""

# First: points huggingface_hub and NLTK at local fixtures, which only takes
# effect if nothing has imported either yet.
import offline_assets  # noqa: F401  # pyright: ignore[reportUnusedImport]

# isort: split

import os
from typing import Any
from unittest.mock import MagicMock, NonCallableMock

import pytest
import torch

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel

# Disable tokenizers parallelism to avoid fork warnings in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Mock Factories - Reusable mock components for testing
# =============================================================================


def stub(**attrs: object) -> NonCallableMock:
    """Attribute bag standing in for ``self`` (or a typed attribute) in a test.

    Behaves like ``SimpleNamespace(**attrs)`` -- reading an attribute that was
    not given raises AttributeError, and new attributes can be assigned -- but
    is accepted by the type checkers wherever the real class is expected, so
    an unbound method can be called against it without pretending it is an
    instance of that class.
    """
    obj = NonCallableMock(spec=sorted(attrs))
    obj.configure_mock(**attrs)
    return obj


@pytest.fixture
def mock_feature_extractor() -> MagicMock:
    """Standard mock feature extractor used across tests."""
    fe = MagicMock()
    fe.sampling_rate = 16000
    fe.return_value = {
        "input_features": torch.randn(1, 80, 100),
        "attention_mask": torch.ones(1, 100),
    }
    return fe


@pytest.fixture
def mock_tokenizer() -> MagicMock:
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
def mock_projector() -> MagicMock:
    """Standard mock projector."""
    proj = MagicMock()
    proj.get_output_length.return_value = 100
    return proj


@pytest.fixture
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip tenacity's backoff waits so retry tests stay fast.

    tenacity binds its sleep function when a ``@retry`` is declared, so the
    only seam is the ``time.sleep`` it ends up calling.
    """

    def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("tenacity.nap.time.sleep", no_sleep)


# =============================================================================
# ASRModel configs and models
# =============================================================================

# Tiny encoder + LM that offline_assets serves locally.
TINY_ASR_CONFIG_KWARGS: dict[str, Any] = {
    "audio_model_id": "openai/whisper-tiny",
    "text_model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "projector_type": "mlp",
    "model_dtype": "float32",
    "attn_implementation": "eager",
}


def make_asr_config(**overrides: Any) -> ASRConfig:
    """Tiny test ASRConfig with ``overrides`` applied on top of the shared defaults."""
    return ASRConfig(**{**TINY_ASR_CONFIG_KWARGS, **overrides})


@pytest.fixture(scope="session")
def base_asr_config() -> ASRConfig:
    """Session-scoped base ASR config (no LoRA) - loaded once per test session."""
    return make_asr_config()


@pytest.fixture(scope="session")
def base_asr_model(base_asr_config: ASRConfig) -> ASRModel:
    """Session-scoped base ASR model - loaded once per test session."""
    return ASRModel(base_asr_config)


@pytest.fixture(scope="session")
def lora_asr_config() -> ASRConfig:
    """Session-scoped LoRA ASR config - loaded once per test session."""
    return make_asr_config(
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )


@pytest.fixture(scope="session")
def lora_asr_model(lora_asr_config: ASRConfig) -> ASRModel:
    """Session-scoped LoRA ASR model - loaded once per test session."""
    return ASRModel(lora_asr_config)


# =============================================================================
# ASRModel Test Utilities - stubs for test_asr_modeling.py
# =============================================================================


def gemma_decode_loop_stub() -> Any:
    """Minimal causal LM exposing Gemma's ``prepare_inputs_for_generation`` signature."""

    class Stub:
        def prepare_inputs_for_generation(
            self,
            input_ids: object,
            inputs_embeds: object = None,
            per_layer_inputs: object = None,
            is_first_iteration: bool = False,
            **kwargs: object,
        ) -> dict[str, object]:
            return {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "per_layer_inputs": per_layer_inputs,
            }

    return Stub()


# =============================================================================
# Projector Test Utilities
# =============================================================================


class MockProjectorConfig:
    """Mock config for projector initialization in tests.

    Provides all config attributes needed by projector classes with sensible defaults.
    Override any attribute by passing kwargs to __init__.
    """

    def __init__(self, **kwargs: int) -> None:
        # Core dimensions
        self.encoder_dim = kwargs.get("encoder_dim", 256)
        self.llm_dim = kwargs.get("llm_dim", 512)
        self.projector_hidden_dim = kwargs.get("projector_hidden_dim", 1024)
        self.projector_pool_stride = kwargs.get("projector_pool_stride", 4)
