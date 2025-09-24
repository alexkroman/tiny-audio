#!/usr/bin/env python3
"""End-to-end test using pytest that trains a model and tests transcription."""

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(scope="module")
def trained_model_path():
    # Use the mac_minimal experiment config for quick training
    cmd = [
        "uv",
        "run",
        "src/train.py",
        "+experiments=mac_minimal",
        "training.max_steps=1",  # Even fewer steps for testing
        "training.save_steps=1",
        "training.eval_steps=1",  # Make eval_steps match save_steps
        "training.logging_steps=1",
        "data.max_train_samples=1",  # Very small dataset
        "training.gradient_checkpointing=false",  # Disable for speed
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(f"Training failed with return code {result.returncode}:\n{result.stderr}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_outputs = Path(f"outputs/{today}")

    if not today_outputs.exists():
        pytest.fail(f"No outputs directory found for today: {today_outputs}")

    # Get the most recent directory
    dirs = sorted(
        [d for d in today_outputs.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not dirs:
        pytest.fail("No output directories found")

    # Check for model in the most recent directory
    model_path = None
    for model_name in ["mac_minimal_model", "mac_model"]:
        potential_path = dirs[0] / "outputs" / model_name
        if potential_path.exists():
            model_path = potential_path
            break

    if not model_path or not model_path.exists():
        pytest.fail(f"Could not find saved model at expected path: {model_path}")

    print(f"✓ Model saved to: {model_path}")
    return str(model_path)


@pytest.fixture
def loaded_model(trained_model_path):
    """Load the trained model."""
    from modeling import ASRModel

    model = ASRModel.from_pretrained(trained_model_path)
    model.eval()
    return model


def test_model_loads_successfully(loaded_model):
    """Test that the model loads without errors."""
    assert loaded_model is not None
    assert hasattr(loaded_model, "tokenizer")
    assert hasattr(loaded_model, "feature_extractor")
    assert hasattr(loaded_model, "encoder")
    assert hasattr(loaded_model, "decoder")
    assert hasattr(loaded_model, "audio_projector")


def test_tokenizer_has_special_tokens(loaded_model):
    """Test that special audio tokens are properly initialized."""
    assert hasattr(loaded_model, "audio_chunk_id")
    assert loaded_model.audio_chunk_id is not None

    vocab = loaded_model.tokenizer.get_vocab()
    assert "<|audio_chunk|>" in vocab
    assert vocab["<|audio_chunk|>"] == loaded_model.audio_chunk_id

    # Check other audio tokens
    assert hasattr(loaded_model, "audio_start_id")
    assert hasattr(loaded_model, "audio_end_id")
    assert hasattr(loaded_model, "audio_pad_id")


def test_embedding_size_matches_tokenizer(loaded_model):
    """Test that embedding size matches tokenizer vocabulary."""
    embed_layer = loaded_model.decoder.model.get_input_embeddings()
    embed_size = embed_layer.weight.shape[0]
    vocab_size = len(loaded_model.tokenizer)

    assert embed_size == vocab_size, f"Embedding size {embed_size} != vocab size {vocab_size}"


@pytest.mark.parametrize(
    "audio_file",
    [
        "demo/wav_outputs/sample5_short.wav",
        "demo/wav_outputs/sample4_counting.wav",
        "demo/wav_outputs/sample1_harvard.wav",
    ],
)
def test_pipeline_with_real_audio(loaded_model, audio_file):
    """Test transcription with real audio files using pipeline."""
    audio_path = Path(audio_file)

    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    # Create pipeline and test
    asr_pipeline = loaded_model.pipeline("automatic-speech-recognition")
    result = asr_pipeline(str(audio_path))

    # Pipeline returns a dict with 'text' key
    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)


def test_generate_method_works(loaded_model):
    """Test that the generate method works with dummy input."""
    import torch

    # Create dummy input features (batch_size=1, seq_len=3000, feature_dim=128)
    dummy_features = torch.randn(1, 3000, 128).to(loaded_model.device)

    # Should not raise exceptions
    output = loaded_model.generate(
        input_features=dummy_features, max_new_tokens=10, do_sample=False
    )

    assert output is not None
    assert output.shape[0] == 1  # Batch size
    assert len(output.shape) == 2  # Should be 2D tensor


if __name__ == "__main__":
    # Run with verbose output and show print statements
    pytest.main([__file__, "-v", "-s"])
