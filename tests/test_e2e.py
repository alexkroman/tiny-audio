#!/usr/bin/env python3
"""End-to-end tests for the ASR model."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_model_training_step():
    """Test a single training step without hydra."""
    from modeling import ASRModel, ASRModelConfig

    # Create small model
    config = ASRModelConfig(
        decoder_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        encoder_model_name="openai/whisper-small",
        lora_r=4,  # Very small for testing
        lora_alpha=8,
    )

    model = ASRModel(config)
    model.train()

    # Create dummy inputs
    batch_size = 1
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    input_features = torch.randn(
        batch_size, 80, 3000
    )  # 80 mel bins, 3000 frames (Whisper requirement)

    # Insert audio chunk token
    audio_chunk_id = model.audio_chunk_id
    input_ids[:, 5] = audio_chunk_id

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        input_features=input_features,
    )

    assert outputs.loss is not None
    assert outputs.loss.item() > 0
    print(f"Loss: {outputs.loss.item()}")


def test_model_generate():
    """Test generation without training."""
    from modeling import ASRModel, ASRModelConfig

    config = ASRModelConfig(
        decoder_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        encoder_model_name="openai/whisper-small",
        lora_r=4,
    )

    model = ASRModel(config)
    model.eval()

    # Test with dummy audio
    input_features = torch.randn(1, 80, 3000)  # 80 mel bins, 3000 frames (Whisper requirement)

    with torch.no_grad():
        output_ids = model.generate(
            input_features,
            max_new_tokens=5,
            min_new_tokens=2,
            do_sample=False,
        )

    assert output_ids is not None
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] >= 2  # At least min_new_tokens

    # Decode
    text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: '{text}'")


def test_transcribe_method():
    """Test the high-level transcribe method."""
    import numpy as np

    from modeling import ASRModel, ASRModelConfig

    config = ASRModelConfig(
        decoder_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        encoder_model_name="openai/whisper-small",
        lora_r=4,
    )

    model = ASRModel(config)
    model.eval()

    # Create dummy audio (2 seconds at 16kHz)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 440 Hz sine wave

    # Test transcribe with numpy array
    result = model.transcribe(audio, sampling_rate=sample_rate, max_new_tokens=10)
    assert isinstance(result, str)
    print(f"Transcribed: '{result}'")

    # Test with torch tensor
    audio_tensor = torch.from_numpy(audio).float()
    result2 = model.transcribe(audio_tensor, sampling_rate=sample_rate, max_new_tokens=10)
    assert isinstance(result2, str)
    print(f"Transcribed from tensor: '{result2}'")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
