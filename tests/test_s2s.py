"""Tests for standalone AudioHead S2S training path.

Tests the standalone pipeline: AudioHead with text tokens + NeuCodec codes,
and S2SDataCollator.
"""

import numpy as np
import pytest
import torch

from tiny_audio.audio_head import (
    BOS_TOKEN,
    EOS_TOKEN,
    NEUCODEC_VOCAB_SIZE,
    AudioHead,
    AudioHeadConfig,
    AudioHeadOutput,
)

TEXT_VOCAB_SIZE = 1000
DECODER_DIM = 64


@pytest.fixture(scope="session")
def audio_head():
    config = AudioHeadConfig(
        decoder_dim=DECODER_DIM,
        decoder_layers=2,
        decoder_heads=2,
        text_vocab_size=TEXT_VOCAB_SIZE,
        max_audio_tokens=50,
    )
    return AudioHead(config)


def _make_codec_inputs(batch_size, audio_len):
    """Build synthetic codec inputs."""
    codec_input_ids = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    codec_input_ids[:, 0] = BOS_TOKEN
    codec_labels = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    codec_attention_mask = torch.ones(batch_size, audio_len, dtype=torch.long)
    return codec_labels, codec_input_ids, codec_attention_mask


class TestStandaloneForward:
    """Test AudioHead forward pass with text tokens directly."""

    def test_returns_loss(self, audio_head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = audio_head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert isinstance(output, AudioHeadOutput)
        assert output.loss is not None
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)

    def test_loss_is_differentiable(self, audio_head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = audio_head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()
        assert audio_head.text_embedding.weight.grad is not None
        audio_head.zero_grad()

    def test_inference_returns_codes(self, audio_head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (1, 10))
        audio_head.eval()
        with torch.no_grad():
            output = audio_head(tokens)
        assert isinstance(output, AudioHeadOutput)
        assert output.codes is not None
        assert output.codes.dim() == 2
        assert output.codes.shape[0] == 1
        audio_head.train()


class TestS2SDataCollator:
    """Test S2SDataCollator with synthetic features."""

    @pytest.fixture
    def collator(self):
        from transformers import AutoTokenizer

        from scripts.train import S2SDataCollator

        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return S2SDataCollator(tokenizer=tokenizer)

    def _make_sample(self, duration_sec=1.0):
        num_frames = int(duration_sec * 50)  # NeuCodec: 50 tokens/sec
        codes = np.random.randint(0, NEUCODEC_VOCAB_SIZE, (num_frames,)).tolist()
        return {"text": "hello world", "codes": codes}

    def test_collator_produces_required_keys(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)

        assert "text_token_ids" in batch
        assert "attention_mask" in batch
        assert "codec_labels" in batch
        assert "codec_input_ids" in batch
        assert "codec_attention_mask" in batch

    def test_no_audio_keys(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)
        assert "input_features" not in batch
        assert "audio_attention_mask" not in batch

    def test_codec_labels_shape(self, collator):
        features = [self._make_sample(), self._make_sample()]
        batch = collator(features)
        assert batch["codec_labels"].dim() == 2
        assert batch["codec_labels"].shape[0] == 2

    def test_codec_input_starts_with_bos(self, collator):
        features = [self._make_sample()]
        batch = collator(features)
        assert (batch["codec_input_ids"][0, 0] == BOS_TOKEN).item()

    def test_codec_labels_has_eos(self, collator):
        features = [self._make_sample(duration_sec=1.0)]
        batch = collator(features)
        assert (batch["codec_labels"] == EOS_TOKEN).any()

    def test_missing_codes_raises(self, collator):
        sample = self._make_sample()
        del sample["codes"]
        with pytest.raises(ValueError, match="No codec codes found"):
            collator([sample])


class TestTrainerCompatibility:
    """Test that AudioHead works directly with HF Trainer (no wrapper needed)."""

    def test_forward_with_collator_keys(self, audio_head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        text_mask = torch.ones(2, 10, dtype=torch.long)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)

        output = audio_head(
            text_token_ids=tokens,
            attention_mask=text_mask,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert isinstance(output, AudioHeadOutput)
        assert output.loss is not None
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)

    def test_training_step(self, audio_head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        text_mask = torch.ones(2, 10, dtype=torch.long)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)

        weights_before = audio_head.text_embedding.weight.data.clone()

        optimizer = torch.optim.Adam(audio_head.parameters(), lr=1e-3)
        optimizer.zero_grad()

        output = audio_head(
            text_token_ids=tokens,
            attention_mask=text_mask,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()
        optimizer.step()

        weights_after = audio_head.text_embedding.weight.data
        assert not torch.equal(weights_before, weights_after)
