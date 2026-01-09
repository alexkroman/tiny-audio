"""Pytest configuration and fixtures."""

import os

import pytest

# Disable tokenizers parallelism to avoid fork warnings in tests
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        lora_enabled=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )


@pytest.fixture(scope="session")
def lora_asr_model(lora_asr_config):
    """Session-scoped LoRA ASR model - loaded once per test session."""
    from tiny_audio.asr_modeling import ASRModel

    return ASRModel(lora_asr_config)
