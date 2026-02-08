"""Integration tests for S2S training path through ASRModel.

Tests the full pipeline: ASRModel with use_audio_head=True, forward pass
with codec_labels + assistant_mask, state dict, and S2SDataCollator.
Uses small models (whisper-tiny, SmolLM2-135M) with synthetic data.
"""

import numpy as np
import pytest
import torch

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.audio_head import BOS_TOKEN, EOS_TOKEN, MIMI_VOCAB_SIZE, NUM_MIMI_CODEBOOKS


@pytest.fixture(scope="session")
def s2s_config():
    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
        use_audio_head=True,
        # Use small decoder for fast tests
        num_codebooks=NUM_MIMI_CODEBOOKS,
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=2,
    )


@pytest.fixture(scope="session")
def s2s_model(s2s_config):
    return ASRModel(s2s_config)


def _make_codec_inputs(batch_size, audio_len, device):
    """Build synthetic codec inputs (mimicking S2SDataCollator output)."""
    codec_input_ids = torch.randint(
        0, MIMI_VOCAB_SIZE, (batch_size, audio_len, NUM_MIMI_CODEBOOKS), device=device
    )
    codec_input_ids[:, 0, :] = BOS_TOKEN

    codec_labels = torch.randint(
        0, MIMI_VOCAB_SIZE, (batch_size, audio_len, NUM_MIMI_CODEBOOKS), device=device
    )
    codec_labels[codec_labels > 1800] = -100  # Mask ~12% as padding

    codec_attention_mask = torch.ones(batch_size, audio_len, dtype=torch.long, device=device)

    return codec_labels, codec_input_ids, codec_attention_mask


class TestS2SModelInit:
    def test_audio_head_exists(self, s2s_model):
        assert s2s_model.audio_head is not None

    def test_audio_head_trainable(self, s2s_model):
        trainable = sum(p.numel() for p in s2s_model.audio_head.parameters() if p.requires_grad)
        assert trainable > 0

    def test_audio_head_llm_dim_matches(self, s2s_model):
        llm_dim = s2s_model.language_model.config.hidden_size
        assert s2s_model.audio_head.llm_dim == llm_dim


class TestS2SForward:
    """Test forward pass with codec_labels through the full ASRModel."""

    def _build_inputs(self, model, batch_size=2, seq_len=20, audio_len=30):
        device = next(model.language_model.parameters()).device
        vocab_size = model.language_model.config.vocab_size

        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        assistant_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        assistant_mask[:, seq_len // 2 :] = True

        codec_labels, codec_input_ids, codec_attention_mask = _make_codec_inputs(
            batch_size, audio_len, device
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "codec_labels": codec_labels,
            "codec_input_ids": codec_input_ids,
            "codec_attention_mask": codec_attention_mask,
        }

    def test_returns_loss(self, s2s_model):
        inputs = self._build_inputs(s2s_model)
        outputs = s2s_model(**inputs)
        assert outputs.loss is not None
        assert outputs.loss.dim() == 0
        assert not torch.isnan(outputs.loss)

    def test_loss_is_differentiable(self, s2s_model):
        inputs = self._build_inputs(s2s_model)
        outputs = s2s_model(**inputs)
        outputs.loss.backward()

        has_grad = False
        for _name, p in s2s_model.audio_head.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        assert has_grad, "Audio head should have gradients after backward"

        s2s_model.zero_grad()

    def test_without_codec_labels_returns_lm_output(self, s2s_model):
        """Without codec_labels, forward pass should skip audio head."""
        device = next(s2s_model.language_model.parameters()).device
        vocab_size = s2s_model.language_model.config.vocab_size

        input_ids = torch.randint(1, vocab_size, (1, 10), device=device)
        labels = input_ids.clone()

        outputs = s2s_model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None

    def test_missing_assistant_mask_raises(self, s2s_model):
        device = next(s2s_model.language_model.parameters()).device
        vocab_size = s2s_model.language_model.config.vocab_size
        codec_labels, codec_input_ids, codec_attention_mask = _make_codec_inputs(1, 20, device)

        inputs = {
            "input_ids": torch.randint(1, vocab_size, (1, 10), device=device),
            "attention_mask": torch.ones(1, 10, dtype=torch.long, device=device),
            "codec_labels": codec_labels,
            "codec_input_ids": codec_input_ids,
            "codec_attention_mask": codec_attention_mask,
        }

        with pytest.raises(ValueError, match="assistant_mask is required"):
            s2s_model(**inputs)


class TestS2SStateDict:
    def test_contains_audio_head_keys(self, s2s_model):
        state = s2s_model.state_dict()
        audio_head_keys = [k for k in state if k.startswith("audio_head.")]
        assert len(audio_head_keys) > 0

    def test_contains_projector_keys(self, s2s_model):
        state = s2s_model.state_dict()
        projector_keys = [k for k in state if k.startswith("projector.")]
        assert len(projector_keys) > 0

    def test_no_encoder_or_lm_keys(self, s2s_model):
        state = s2s_model.state_dict()
        for key in state:
            assert not key.startswith("language_model."), f"LM key leaked: {key}"
            assert not key.startswith("audio_encoder."), f"Encoder key leaked: {key}"


class TestS2SDataCollator:
    """Test S2SDataCollator with synthetic features."""

    @pytest.fixture
    def collator(self, s2s_model):
        from scripts.train import S2SDataCollator

        return S2SDataCollator(
            tokenizer=s2s_model.tokenizer,
            feature_extractor=s2s_model.feature_extractor,
            sample_rate=16000,
            projector=s2s_model.projector,
            encoder_conv_layers=s2s_model.config.encoder_conv_layers,
            system_prompt="You are a helpful assistant.",
        )

    def _make_sample(self, duration_sec=1.0, sample_rate=16000):
        num_samples = int(duration_sec * sample_rate)
        audio_array = np.random.randn(num_samples).astype(np.float32) * 0.1
        num_frames = int(duration_sec * 12.5)  # Mimi: 12.5 tokens/sec
        codes = np.random.randint(0, MIMI_VOCAB_SIZE, (NUM_MIMI_CODEBOOKS, num_frames)).tolist()
        return {
            "audio": {"array": audio_array, "sampling_rate": sample_rate},
            "text": "hello world",
            "codes": codes,
        }

    def test_collator_produces_required_keys(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "input_features" in batch
        assert "codec_labels" in batch
        assert "codec_input_ids" in batch
        assert "codec_attention_mask" in batch
        assert "assistant_mask" in batch

    def test_codec_labels_shape(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)

        # codec_labels: [batch, audio_len, num_codebooks]
        assert batch["codec_labels"].dim() == 3
        assert batch["codec_labels"].shape[0] == 2
        assert batch["codec_labels"].shape[2] == NUM_MIMI_CODEBOOKS

    def test_codec_input_ids_shape(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)

        # codec_input_ids: [batch, audio_len, num_codebooks]
        assert batch["codec_input_ids"].dim() == 3
        assert batch["codec_input_ids"].shape[0] == 2
        assert batch["codec_input_ids"].shape[2] == NUM_MIMI_CODEBOOKS

    def test_codec_input_starts_with_bos(self, collator):
        features = [self._make_sample()]
        batch = collator(features)
        # First position should be BOS for all codebooks
        assert (batch["codec_input_ids"][0, 0, :] == BOS_TOKEN).all()

    def test_codec_labels_has_eos(self, collator):
        features = [self._make_sample(duration_sec=1.0)]
        batch = collator(features)
        # Should have EOS somewhere in labels
        assert (batch["codec_labels"] == EOS_TOKEN).any()

    def test_missing_codes_raises(self, collator):
        sample = self._make_sample()
        del sample["codes"]

        with pytest.raises(ValueError, match="No codec codes found"):
            collator([sample])


class TestS2STrainingStep:
    """Test that one full training step completes through the HF Trainer."""

    @pytest.fixture
    def collator(self, s2s_model):
        from scripts.train import S2SDataCollator

        return S2SDataCollator(
            tokenizer=s2s_model.tokenizer,
            feature_extractor=s2s_model.feature_extractor,
            sample_rate=16000,
            projector=s2s_model.projector,
            encoder_conv_layers=s2s_model.config.encoder_conv_layers,
            system_prompt="You are a helpful assistant.",
        )

    def _make_sample(self, duration_sec=1.0, sample_rate=16000):
        num_samples = int(duration_sec * sample_rate)
        audio_array = np.random.randn(num_samples).astype(np.float32) * 0.1
        num_frames = int(duration_sec * 12.5)  # Mimi: 12.5 tokens/sec
        codes = np.random.randint(0, MIMI_VOCAB_SIZE, (NUM_MIMI_CODEBOOKS, num_frames)).tolist()
        return {
            "audio": {"array": audio_array, "sampling_rate": sample_rate},
            "text": "hello world",
            "codes": codes,
        }

    def test_one_training_step(self, s2s_model, collator):
        """Verify one training step: collator -> forward -> backward -> optimizer.step()."""
        # Snapshot audio head weights before training
        weights_before = {k: v.clone() for k, v in s2s_model.audio_head.state_dict().items()}

        # Build batch via collator (same path as real training)
        # Use 30s audio to get 3000 mel frames required by Whisper encoder
        samples = [self._make_sample(duration_sec=30.0) for _ in range(2)]
        batch = collator(samples)

        # Move to model device
        device = next(s2s_model.audio_head.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # One training step
        s2s_model.train()
        optimizer = torch.optim.Adam(s2s_model.audio_head.parameters(), lr=1e-3)
        optimizer.zero_grad()

        outputs = s2s_model(**batch)
        loss = outputs.loss

        assert loss is not None, "Forward pass should produce a loss"
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"

        loss.backward()
        optimizer.step()

        # Audio head weights changed
        weights_after = s2s_model.audio_head.state_dict()
        changed = any(not torch.equal(weights_before[k], weights_after[k]) for k in weights_before)
        assert changed, "Audio head weights should update after one training step"
