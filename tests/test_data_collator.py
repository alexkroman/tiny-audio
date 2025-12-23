"""Tests for DataCollator label masking behavior."""

import numpy as np
import pytest
from transformers import AutoTokenizer, WhisperFeatureExtractor

from scripts.train import DataCollator


class MockProjector:
    """Mock projector that mimics stride-2 downsampling."""

    def get_output_length(self, input_length: int) -> int:
        return input_length // 2


@pytest.fixture
def projector():
    """Create a mock projector."""
    return MockProjector()


@pytest.fixture
def tokenizer():
    """Load the SmolLM tokenizer with <audio> token added."""
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    # Add <audio> token like ASRModel does
    existing_special = tok.additional_special_tokens or []
    if "<audio>" not in existing_special:
        tok.add_special_tokens({"additional_special_tokens": existing_special + ["<audio>"]})
    return tok


@pytest.fixture
def feature_extractor():
    """Load Whisper feature extractor."""
    return WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")


@pytest.fixture
def collator(tokenizer, feature_extractor, projector):
    """Create DataCollator instance."""
    return DataCollator(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        sample_rate=16000,
        system_prompt="You are a helpful assistant.",
        projector=projector,
    )


@pytest.fixture
def collator_no_system(tokenizer, feature_extractor, projector):
    """Create DataCollator without system prompt."""
    return DataCollator(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        sample_rate=16000,
        system_prompt=None,
        projector=projector,
    )


def create_sample(text: str, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Create a sample with dummy audio."""
    num_samples = int(duration_sec * sample_rate)
    audio_array = np.random.randn(num_samples).astype(np.float32) * 0.1
    return {
        "audio": {"array": audio_array, "sampling_rate": sample_rate},
        "text": text,
    }


class TestLabelMasking:
    """Test that label masking works correctly using trl's DataCollatorForCompletionOnlyLM."""

    def test_assistant_content_is_unmasked(self, collator, tokenizer):
        """Verify that assistant content tokens have valid labels (not -100)."""
        text = "Hello world this is a test."
        samples = [create_sample(text)]

        batch = collator(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        # Find non-masked positions (excluding padding)
        pad_id = tokenizer.pad_token_id
        unmasked_positions = [
            i for i, (label, inp) in enumerate(zip(labels, input_ids))
            if label != -100 and inp != pad_id
        ]

        assert len(unmasked_positions) > 0, "No unmasked labels found"

        # Decode the unmasked tokens
        unmasked_tokens = [input_ids[i] for i in unmasked_positions]
        unmasked_text = tokenizer.decode(unmasked_tokens, skip_special_tokens=True)

        # The transcription text should be in the unmasked portion
        assert "hello" in unmasked_text.lower(), \
            f"Transcription not found in unmasked text: {unmasked_text}"

    def test_stop_token_is_unmasked(self, collator, tokenizer):
        """Verify that <|im_end|> stop token is included in labels."""
        text = "Test transcription."
        samples = [create_sample(text)]

        batch = collator(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Find <|im_end|> tokens that are unmasked
        unmasked_im_end = [
            i for i, (label, inp) in enumerate(zip(labels, input_ids))
            if inp == im_end_id and label == im_end_id
        ]

        assert len(unmasked_im_end) > 0, \
            "No <|im_end|> token found in labels - model won't learn to stop"

    def test_system_and_user_prompts_are_masked(self, collator, tokenizer):
        """Verify that system prompt and user instruction are masked (-100)."""
        text = "Transcription content here."
        samples = [create_sample(text)]

        batch = collator(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        # Decode full input to verify structure
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        assert "Transcribe:" in full_text, "User instruction not in input"

        # Find "Transcribe" in input and verify it's masked
        transcribe_tokens = tokenizer.encode("Transcribe", add_special_tokens=False)
        for i in range(len(input_ids) - len(transcribe_tokens)):
            if input_ids[i:i+len(transcribe_tokens)] == transcribe_tokens:
                # These positions should be masked
                assert all(labels[i+j] == -100 for j in range(len(transcribe_tokens))), \
                    "User instruction should be masked"
                break

    def test_no_system_prompt(self, collator_no_system, tokenizer):
        """Verify masking works correctly without system prompt."""
        text = "No system prompt test."
        samples = [create_sample(text)]

        batch = collator_no_system(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        # Should still have unmasked content
        pad_id = tokenizer.pad_token_id
        unmasked_count = sum(
            1 for label, inp in zip(labels, input_ids)
            if label != -100 and inp != pad_id
        )
        assert unmasked_count > 0, "No unmasked labels without system prompt"

    def test_label_token_alignment(self, collator, tokenizer):
        """Verify that unmasked labels match corresponding input_ids."""
        text = "Alignment test."
        samples = [create_sample(text)]

        batch = collator(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        # For every unmasked position, label should equal input_id
        for i, (label, input_id) in enumerate(zip(labels, input_ids)):
            if label != -100:
                assert label == input_id, \
                    f"Label mismatch at position {i}: label={label}, input_id={input_id}"


class TestAudioTokens:
    """Test that audio tokens are correctly inserted to match projector output."""

    def test_audio_token_count_matches_encoder_output(self, collator, tokenizer):
        """Verify number of <audio> tokens matches expected encoder output length."""
        samples = [create_sample("Test transcription.", duration_sec=1.0)]

        batch = collator(samples)

        # Get the audio token ID
        audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")

        # Count audio tokens in input_ids
        num_audio_tokens = (batch["input_ids"] == audio_token_id).sum().item()

        # Expected: mel_len // 4 (Whisper stride-2 × projector stride-2)
        mel_len = batch["input_features"].shape[-1]
        expected_audio_tokens = mel_len // 4

        assert num_audio_tokens == expected_audio_tokens, (
            f"Audio token count mismatch: got {num_audio_tokens}, "
            f"expected {expected_audio_tokens} (mel_len={mel_len})"
        )

    def test_audio_tokens_not_just_one(self, collator, tokenizer):
        """Verify we have many audio tokens, not just a single placeholder."""
        samples = [create_sample("Test.", duration_sec=1.0)]

        batch = collator(samples)

        audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
        num_audio_tokens = (batch["input_ids"] == audio_token_id).sum().item()

        # Should have many audio tokens (Whisper outputs ~1500 for 30s, ~50 for 1s)
        assert num_audio_tokens > 10, (
            f"Expected many audio tokens, got only {num_audio_tokens}. "
            "This suggests audio embeddings would be discarded."
        )

    def test_audio_tokens_are_masked(self, collator, tokenizer):
        """Verify audio tokens are masked in labels (not trained on)."""
        samples = [create_sample("Test.", duration_sec=1.0)]

        batch = collator(samples)

        audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
        labels = batch["labels"][0]
        input_ids = batch["input_ids"][0]

        # All positions with <audio> token should have label=-100
        audio_positions = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
        for pos in audio_positions:
            assert labels[pos].item() == -100, (
                f"Audio token at position {pos} should be masked but has label {labels[pos]}"
            )


class TestBatchProcessing:
    """Test batch processing behavior."""

    def test_multiple_samples(self, collator):
        """Verify collator handles multiple samples correctly."""
        samples = [
            create_sample("First transcription."),
            create_sample("Second transcription here."),
            create_sample("Third one."),
        ]

        batch = collator(samples)

        assert batch["input_ids"].shape[0] == 3
        assert batch["labels"].shape[0] == 3
        assert batch["input_features"].shape[0] == 3

    def test_audio_features_shape(self, collator):
        """Verify audio features have correct shape."""
        samples = [create_sample("Test.", duration_sec=2.0)]

        batch = collator(samples)

        # Whisper expects (batch, n_mels, time)
        assert len(batch["input_features"].shape) == 3
        assert batch["input_features"].shape[1] == 80  # n_mels for Whisper


class TestModelIntegration:
    """Integration tests verifying audio actually influences the model."""

    @pytest.fixture
    def model(self):
        """Load a small model for testing."""
        from src.asr_config import ASRConfig
        from src.asr_modeling import ASRModel

        config = ASRConfig(
            encoder_model_name="openai/whisper-tiny",
            decoder_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            projector_type="mlp",
            model_dtype="float32",
            attn_implementation="eager",
        )
        return ASRModel(config)

    def test_different_audio_produces_different_loss(self, model):
        """Verify that different audio inputs produce different losses."""
        from scripts.train import DataCollator

        collator = DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=16000,
            system_prompt=None,
            projector=model.projector,
        )

        # Create two samples with different audio but same text
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # 440Hz sine wave
        audio1 = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        # 880Hz sine wave (different frequency)
        audio2 = (np.sin(2 * np.pi * 880 * t) * 0.5).astype(np.float32)

        sample1 = {"audio": {"array": audio1, "sampling_rate": sample_rate}, "text": "Test."}
        sample2 = {"audio": {"array": audio2, "sampling_rate": sample_rate}, "text": "Test."}

        batch1 = collator([sample1])
        batch2 = collator([sample2])

        model.eval()
        import torch
        with torch.no_grad():
            out1 = model(
                input_ids=batch1["input_ids"],
                input_features=batch1["input_features"],
                labels=batch1["labels"],
                attention_mask=batch1["attention_mask"],
            )
            out2 = model(
                input_ids=batch2["input_ids"],
                input_features=batch2["input_features"],
                labels=batch2["labels"],
                attention_mask=batch2["attention_mask"],
            )

        loss_diff = abs(out1.loss.item() - out2.loss.item())
        assert loss_diff > 0.001, (
            f"Different audio should produce different loss. "
            f"Loss1={out1.loss.item():.4f}, Loss2={out2.loss.item():.4f}, diff={loss_diff:.6f}"
        )

    def test_audio_embeddings_replace_audio_tokens(self, model):
        """Verify that <audio> token embeddings are replaced with projected audio."""
        from scripts.train import DataCollator
        import torch

        collator = DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=16000,
            system_prompt=None,
            projector=model.projector,
        )

        sample = create_sample("Test.", duration_sec=1.0)
        batch = collator([sample])

        # Get original text embeddings (before audio injection)
        original_embeds = model.language_model.get_input_embeddings()(batch["input_ids"])

        # Get the audio token positions
        audio_token_mask = batch["input_ids"] == model.audio_token_id

        # Encode audio and get what would be injected
        audio_embeds = model._encode_audio(batch["input_features"], None)

        # The audio embeddings should be different from the original <audio> token embedding
        audio_token_embed = original_embeds[audio_token_mask][0]  # First audio token's original embedding
        projected_audio_embed = audio_embeds[0]  # First projected audio embedding

        # They should be different (projected audio != text embedding of <audio> token)
        embed_diff = (audio_token_embed - projected_audio_embed).abs().mean().item()
        assert embed_diff > 0.01, (
            f"Projected audio embedding should differ from <audio> token embedding. "
            f"Mean diff={embed_diff:.6f}"
        )

    def test_all_audio_embeddings_used(self, model):
        """Verify that all projected audio embeddings are used, not just one."""
        from scripts.train import DataCollator
        import torch

        collator = DataCollator(
            tokenizer=model.tokenizer,
            feature_extractor=model.feature_extractor,
            sample_rate=16000,
            system_prompt=None,
            projector=model.projector,
        )

        sample = create_sample("Test.", duration_sec=1.0)
        batch = collator([sample])

        # Count audio tokens in input
        num_audio_tokens = (batch["input_ids"] == model.audio_token_id).sum().item()

        # Get projected audio embeddings
        audio_embeds = model._encode_audio(batch["input_features"], None)
        num_audio_embeds = audio_embeds.shape[0]

        assert num_audio_tokens == num_audio_embeds, (
            f"Mismatch: {num_audio_tokens} <audio> tokens but {num_audio_embeds} audio embeddings. "
            "This means some audio embeddings would be discarded by masked_scatter."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
