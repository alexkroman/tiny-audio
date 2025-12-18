"""Tests for DataCollator label masking behavior."""

import numpy as np
import pytest
from transformers import AutoTokenizer, WhisperFeatureExtractor

from src.train import DataCollator


@pytest.fixture
def tokenizer():
    """Load the SmolLM tokenizer."""
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


@pytest.fixture
def feature_extractor():
    """Load Whisper feature extractor."""
    return WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")


@pytest.fixture
def collator(tokenizer, feature_extractor):
    """Create DataCollator instance."""
    return DataCollator(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        sample_rate=16000,
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def collator_no_system(tokenizer, feature_extractor):
    """Create DataCollator without system prompt."""
    return DataCollator(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        sample_rate=16000,
        system_prompt=None,
    )


def create_sample(text: str, duration_sec: float = 1.0):
    """Create a sample with dummy audio."""
    sample_rate = 16000
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
