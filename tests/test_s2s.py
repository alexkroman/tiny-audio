"""Integration tests for S2S training path through ASRModel.

Tests the full pipeline: ASRModel with use_audio_head=True, forward pass
with dia_labels + assistant_mask, state dict, and S2SDataCollator.
Uses small models (whisper-tiny, SmolLM2-135M) with synthetic data.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.audio_head import DAC_VOCAB_SIZE, NUM_DAC_CODEBOOKS, AudioHead


class MockDiaModel(nn.Module):
    """Mock Dia decoder for S2S integration tests."""

    def __init__(self, hidden_dim=1024, vocab_size=1028):
        super().__init__()
        self.config = SimpleNamespace(
            delay_pattern=[0, 8, 9, 10, 11, 12, 13, 14, 15],
            pad_token_id=1025,
            eos_token_id=1024,
            bos_token_id=1026,
            decoder_config=SimpleNamespace(num_channels=9, vocab_size=vocab_size),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.requires_grad_(False)

    def forward(self, encoder_outputs=None, labels=None, **kwargs):
        hidden = encoder_outputs.last_hidden_state
        pooled = hidden.mean(dim=(0, 1), keepdim=True)
        loss = None
        if labels is not None:
            expanded = pooled.expand(labels.shape[0], labels.shape[1], -1)
            logits = self.lm_head(expanded)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss)


@pytest.fixture(scope="session")
def s2s_config():
    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
        use_audio_head=True,
    )


@pytest.fixture(scope="session")
def s2s_model(s2s_config):
    model = ASRModel(s2s_config)
    model.audio_head.dia_model = MockDiaModel(AudioHead.DIA_DIM, DAC_VOCAB_SIZE)
    return model


def _make_dia_inputs(batch_size, seq_len, device):
    """Build synthetic Dia-ready inputs (mimicking S2SDataCollator output)."""
    dia_labels = torch.randint(
        0, DAC_VOCAB_SIZE, (batch_size * NUM_DAC_CODEBOOKS, seq_len), device=device
    )
    dia_labels[dia_labels > 900] = -100
    dia_decoder_input_ids = torch.randint(
        0, DAC_VOCAB_SIZE, (batch_size, seq_len, NUM_DAC_CODEBOOKS), device=device
    )
    dia_decoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return dia_labels, dia_decoder_input_ids, dia_decoder_attention_mask


class TestS2SModelInit:
    def test_audio_head_exists(self, s2s_model):
        assert s2s_model.audio_head is not None

    def test_audio_head_projector_trainable(self, s2s_model):
        for p in s2s_model.audio_head.projector.parameters():
            assert p.requires_grad

    def test_audio_head_llm_dim_matches(self, s2s_model):
        llm_dim = s2s_model.language_model.config.hidden_size
        assert s2s_model.audio_head.llm_dim == llm_dim


class TestS2SForward:
    """Test forward pass with dia_labels through the full ASRModel."""

    def _build_inputs(self, model, batch_size=2, seq_len=20, audio_len=50):
        device = next(model.language_model.parameters()).device
        vocab_size = model.language_model.config.vocab_size

        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        assistant_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        assistant_mask[:, seq_len // 2 :] = True

        dia_labels, dia_decoder_input_ids, dia_decoder_attention_mask = _make_dia_inputs(
            batch_size, audio_len, device
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "dia_labels": dia_labels,
            "dia_decoder_input_ids": dia_decoder_input_ids,
            "dia_decoder_attention_mask": dia_decoder_attention_mask,
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

        for name, p in s2s_model.audio_head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for audio_head.{name}"

        s2s_model.zero_grad()

    def test_without_dia_labels_returns_lm_output(self, s2s_model):
        """Without dia_labels, forward pass should skip audio head."""
        device = next(s2s_model.language_model.parameters()).device
        vocab_size = s2s_model.language_model.config.vocab_size

        input_ids = torch.randint(1, vocab_size, (1, 10), device=device)
        labels = input_ids.clone()

        outputs = s2s_model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None

    def test_missing_assistant_mask_raises(self, s2s_model):
        device = next(s2s_model.language_model.parameters()).device
        vocab_size = s2s_model.language_model.config.vocab_size
        dia_labels, dia_decoder_input_ids, dia_decoder_attention_mask = _make_dia_inputs(
            1, 20, device
        )

        inputs = {
            "input_ids": torch.randint(1, vocab_size, (1, 10), device=device),
            "attention_mask": torch.ones(1, 10, dtype=torch.long, device=device),
            "dia_labels": dia_labels,
            "dia_decoder_input_ids": dia_decoder_input_ids,
            "dia_decoder_attention_mask": dia_decoder_attention_mask,
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
        num_frames = int(duration_sec * 86)
        codes = np.random.randint(0, DAC_VOCAB_SIZE, (NUM_DAC_CODEBOOKS, num_frames)).tolist()
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
        assert "dia_labels" in batch
        assert "dia_decoder_input_ids" in batch
        assert "dia_decoder_attention_mask" in batch
        assert "assistant_mask" in batch

    def test_dia_labels_shape(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)

        # dia_labels: [batch * 9, seq_len]
        assert batch["dia_labels"].dim() == 2
        assert batch["dia_labels"].shape[0] == 2 * NUM_DAC_CODEBOOKS

    def test_dia_decoder_input_ids_shape(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)

        # dia_decoder_input_ids: [batch, seq_len, 9]
        assert batch["dia_decoder_input_ids"].dim() == 3
        assert batch["dia_decoder_input_ids"].shape[0] == 2
        assert batch["dia_decoder_input_ids"].shape[2] == NUM_DAC_CODEBOOKS

    def test_missing_codes_raises(self, collator):
        sample = self._make_sample()
        del sample["codes"]

        with pytest.raises(ValueError, match="No codec codes found"):
            collator([sample])
