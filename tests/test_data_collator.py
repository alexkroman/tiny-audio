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
    existing_special = getattr(tok, "additional_special_tokens", None) or []
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
            i
            for i, (label, inp) in enumerate(zip(labels, input_ids))
            if label != -100 and inp != pad_id
        ]

        assert len(unmasked_positions) > 0, "No unmasked labels found"

        # Decode the unmasked tokens
        unmasked_tokens = [input_ids[i] for i in unmasked_positions]
        unmasked_text = tokenizer.decode(unmasked_tokens, skip_special_tokens=True)

        # The transcription text should be in the unmasked portion
        assert "hello" in unmasked_text.lower(), (
            f"Transcription not found in unmasked text: {unmasked_text}"
        )

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
            i
            for i, (label, inp) in enumerate(zip(labels, input_ids))
            if inp == im_end_id and label == im_end_id
        ]

        assert len(unmasked_im_end) > 0, (
            "No <|im_end|> token found in labels - model won't learn to stop"
        )

    def test_system_and_user_prompts_are_masked(self, collator, tokenizer):
        """Verify that system prompt and user instruction are masked (-100)."""
        text = "Transcription content here."
        samples = [create_sample(text)]

        batch = collator(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        # Decode full input to verify structure
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        # Check that one of the transcribe prompts is in the text
        from scripts.train import TRANSCRIBE_PROMPTS

        matched_prompt = None
        for prompt in TRANSCRIBE_PROMPTS:
            if prompt in full_text:
                matched_prompt = prompt
                break
        assert matched_prompt is not None, f"User instruction not in input. Got: {full_text}"

        # Find the prompt in input and verify it's masked
        prompt_tokens = tokenizer.encode(matched_prompt, add_special_tokens=False)
        for i in range(len(input_ids) - len(prompt_tokens)):
            if input_ids[i : i + len(prompt_tokens)] == prompt_tokens:
                # These positions should be masked
                assert all(labels[i + j] == -100 for j in range(len(prompt_tokens))), (
                    "User instruction should be masked"
                )
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
            1 for label, inp in zip(labels, input_ids) if label != -100 and inp != pad_id
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
                assert label == input_id, (
                    f"Label mismatch at position {i}: label={label}, input_id={input_id}"
                )


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

        # Expected: real_mel_len // 4 (Whisper stride-2 × projector stride-2)
        # Use attention mask to get actual audio length (not padded)
        real_mel_len = batch["audio_attention_mask"].sum().item()
        expected_audio_tokens = real_mel_len // 4

        assert num_audio_tokens == expected_audio_tokens, (
            f"Audio token count mismatch: got {num_audio_tokens}, "
            f"expected {expected_audio_tokens} (real_mel_len={real_mel_len})"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
