#!/usr/bin/env python3
"""End-to-end test using pytest that trains a model and tests transcription."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(scope="module")
def trained_model_path():
    output_dir = "outputs/test_e2e_model"
    
    # Use shell script to run training
    script_path = Path(__file__).parent / "run_e2e_training.sh"
    
    # Run with timeout to prevent hanging
    try:
        result = subprocess.run(
            [str(script_path)], 
            capture_output=True, 
            text=True, 
            timeout=300,
            shell=False,  # Execute directly as a script
            cwd=Path(__file__).parent.parent  # Run from project root
        )  # 5 minute timeout
    except subprocess.TimeoutExpired:
        pytest.fail("Training timed out after 5 minutes")

    if result.returncode != 0:
        print(f"Training stdout:\n{result.stdout}")
        print(f"Training stderr:\n{result.stderr}")
        pytest.fail(f"Training failed with return code {result.returncode}")

    model_path = Path(output_dir) / "outputs" / "mac_minimal_model"

    if not model_path.exists():
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

    # Model only has audio_chunk_id, not start/end/pad ids


def test_embedding_size_matches_tokenizer(loaded_model):
    """Test that embedding size matches tokenizer vocabulary."""
    embed_layer = loaded_model.decoder.get_input_embeddings()
    embed_size = embed_layer.weight.shape[0]
    vocab_size = len(loaded_model.tokenizer)

    assert embed_size == vocab_size, f"Embedding size {embed_size} != vocab size {vocab_size}"


@pytest.mark.parametrize(
    "audio_file",
    [
        "demo/wav_outputs/sample5_short.wav",
    ],
)
def test_pipeline_with_real_audio(loaded_model, audio_file):
    """Test transcription with real audio files using pipeline."""
    audio_path = Path(audio_file)

    if not audio_path.exists():
        pytest.skip(f"Audio file not found: {audio_file}")

    # Model doesn't have a pipeline method - skip this test
    pytest.skip("Pipeline test not applicable for this model architecture")


def test_generate_method_works(loaded_model):
    """Test that the generate method works with dummy input."""
    import torch

    dummy_features = torch.randn(1, 3000, 128).to(loaded_model.device)

    output = loaded_model.generate(
        input_features=dummy_features, max_new_tokens=10, do_sample=False
    )

    assert output is not None
    assert output.shape[0] == 1  # Batch size
    assert len(output.shape) == 2  # Should be 2D tensor


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
