"""Tests for DataCollator label masking behavior."""

import numpy as np
import pytest
import torch
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
        err = f"Transcription not found in unmasked text: {unmasked_text}"
        assert "hello" in unmasked_text.lower(), err

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

        err = "No <|im_end|> token found in labels - model won't learn to stop"
        assert len(unmasked_im_end) > 0, err

    def test_system_and_user_prompts_are_masked(self, collator, tokenizer):
        """Verify that system prompt and user content are masked (-100)."""
        text = "Transcription content here."
        samples = [create_sample(text)]

        batch = collator(samples)

        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()

        # Decode full input to verify structure
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Verify audio tokens are present and user section exists
        assert "<audio>" in full_text, f"Audio tokens not in input. Got: {full_text}"
        assert "<|im_start|>user" in full_text, f"User section not in input. Got: {full_text}"

        # Verify that user section (audio tokens) is masked
        # Find audio token positions and verify they're masked
        audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
        audio_positions = [i for i, tok in enumerate(input_ids) if tok == audio_token_id]
        assert len(audio_positions) > 0, "No audio tokens found"
        assert all(labels[i] == -100 for i in audio_positions), "Audio tokens should be masked"

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
                err = f"Label mismatch at position {i}: label={label}, input_id={input_id}"
                assert label == input_id, err


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
            err = f"Audio token at position {pos} should be masked but has label {labels[pos]}"
            assert labels[pos].item() == -100, err


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


class TestAudioTokenCountsExposed:
    """Collator must expose audio_token_counts so the model does not recompute them."""

    def test_audio_token_counts_in_batch(self, collator):
        samples = [
            create_sample("hello", duration_sec=1.0),
            create_sample("world how are you", duration_sec=2.0),
        ]
        batch = collator(samples)
        err = "Collator must include audio_token_counts in the returned batch"
        assert "audio_token_counts" in batch, err
        assert batch["audio_token_counts"].dtype == torch.long
        assert batch["audio_token_counts"].shape == (2,)
        assert (batch["audio_token_counts"] > 0).all()


class TestExtractAudioArraysFilters:
    """Collator must drop rows that would poison training:
    pre-norm-empty text, post-norm-empty text (entire label was an annotation
    marker), and over-long audio (silently truncated while the label keeps its
    full transcript).

    The duration bound is read from DataCollator._MAX_AUDIO_SECONDS rather than
    hardcoded. It has been retuned before (30.0 -> 19.0, to cut mel-spec peak
    memory) and hardcoding it left these tests asserting the old window."""

    def test_drops_post_normalize_empty_text(self, collator):
        # Switchboard ships ~2% of rows where the whole label is `<noise>` —
        # passes the .strip() check but normalizes to empty, producing an
        # empty assistant turn that teaches the model to emit nothing.
        good = create_sample("hello world", duration_sec=1.0)
        bad = create_sample("<noise>", duration_sec=1.0)
        arrays, kept = collator._extract_audio_arrays([good, bad])
        assert len(arrays) == 1
        assert kept[0]["text"] == "hello world"

    def test_drops_audio_over_max_duration(self, collator):
        # Over-cap audio is silently truncated by the feature extractor while
        # the label keeps its full transcript, so the row teaches the model to
        # hallucinate the unheard tail — observed in EdAcc (max 46s) and
        # Earnings22 (max 26s).
        good = create_sample("normal length", duration_sec=1.0)
        too_long = create_sample("very long", duration_sec=DataCollator._MAX_AUDIO_SECONDS + 5.0)
        arrays, kept = collator._extract_audio_arrays([good, too_long])
        assert len(arrays) == 1
        assert kept[0]["text"] == "normal length"

    def test_keeps_audio_at_max_duration_boundary(self, collator):
        # The cap is strictly greater-than, so exactly _MAX_AUDIO_SECONDS passes.
        ok = create_sample("at the cap", duration_sec=DataCollator._MAX_AUDIO_SECONDS)
        arrays, _ = collator._extract_audio_arrays([ok])
        assert len(arrays) == 1

    def test_keeps_audio_at_min_duration_boundary(self, collator):
        # Mirror of the upper bound: the floor is strictly less-than, so a clip
        # of exactly _MIN_AUDIO_SECONDS is kept while anything under is dropped.
        ok = create_sample("at the floor", duration_sec=DataCollator._MIN_AUDIO_SECONDS)
        arrays, _ = collator._extract_audio_arrays([ok])
        assert len(arrays) == 1

    def test_drops_audio_under_min_duration(self, collator):
        # Sub-floor clips are boundary-cut segments and isolated backchannels
        # where the audio span and the reference transcript don't line up.
        good = create_sample("normal length", duration_sec=1.0)
        too_short = create_sample("yeah", duration_sec=DataCollator._MIN_AUDIO_SECONDS / 2)
        arrays, kept = collator._extract_audio_arrays([good, too_short])
        assert len(arrays) == 1
        assert kept[0]["text"] == "normal length"

    def test_drops_pre_normalize_empty_text(self, collator):
        # Existing behavior — empty-string labels were already dropped.
        good = create_sample("hi", duration_sec=1.0)
        empty = create_sample("", duration_sec=1.0)
        arrays, kept = collator._extract_audio_arrays([good, empty])
        assert len(arrays) == 1
        assert kept[0]["text"] == "hi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
