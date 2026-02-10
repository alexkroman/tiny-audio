"""Tests for the trainable AR decoder AudioHead for speech-to-speech."""

import pytest
import torch

from tiny_audio.audio_head import (
    BOS_TOKEN,
    NEUCODEC_VOCAB_SIZE,
    AudioHead,
    AudioHeadConfig,
    AudioHeadOutput,
)

TEXT_VOCAB_SIZE = 1000
DECODER_DIM = 64  # Small for fast tests


def _make_codec_inputs(batch_size=2, audio_len=20):
    """Build synthetic codec inputs (mimicking S2SDataCollator output)."""
    codec_input_ids = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    codec_input_ids[:, 0] = BOS_TOKEN  # BOS at start

    codec_labels = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    # Mask ~20% as padding
    mask = torch.rand(batch_size, audio_len) > 0.8
    codec_labels[mask] = -100

    codec_attention_mask = torch.ones(batch_size, audio_len, dtype=torch.long)

    return codec_labels, codec_input_ids, codec_attention_mask


def _make_config(**kwargs):
    """Create an AudioHeadConfig with test defaults."""
    defaults = {
        "decoder_dim": DECODER_DIM,
        "decoder_layers": 2,
        "decoder_heads": 2,
        "text_vocab_size": TEXT_VOCAB_SIZE,
        "max_audio_tokens": 50,
    }
    defaults.update(kwargs)
    return AudioHeadConfig(**defaults)


@pytest.fixture
def config():
    return _make_config


@pytest.fixture
def head(config):
    """Create a small AudioHead for testing."""
    return AudioHead(config())


class TestInit:
    def test_default_vocab(self):
        cfg = AudioHeadConfig()
        head = AudioHead(cfg)
        assert head.text_vocab_size == 32000

    def test_custom_vocab(self):
        cfg = _make_config(text_vocab_size=1000)
        head = AudioHead(cfg)
        assert head.text_vocab_size == 1000

    def test_vocab_size(self, head):
        assert head.vocab_size == NEUCODEC_VOCAB_SIZE

    def test_text_embedding_exists(self, head):
        assert hasattr(head, "text_embedding")

    def test_token_embedding_exists(self, head):
        assert hasattr(head, "token_embedding")

    def test_decoder_exists(self, head):
        assert hasattr(head, "decoder")

    def test_uses_token_embedding_as_head(self, head):
        assert hasattr(head, "token_embedding")
        assert not hasattr(head, "head")

    def test_no_neucodec_model_by_default(self, head):
        assert head.neucodec_model is None

    def test_is_pretrained_model(self, head):
        from transformers import PreTrainedModel

        assert isinstance(head, PreTrainedModel)

    def test_config_class(self):
        assert AudioHead.config_class is AudioHeadConfig


class TestTextEmbeddingShape:
    def test_embed_output_dim(self, head):
        x = torch.randint(0, TEXT_VOCAB_SIZE, (1, 10))
        out = head.text_embedding(x)
        assert out.shape == (1, 10, DECODER_DIM)

    def test_embed_batch(self, head):
        x = torch.randint(0, TEXT_VOCAB_SIZE, (4, 5))
        out = head.text_embedding(x)
        assert out.shape == (4, 5, DECODER_DIM)


class TestForwardTraining:
    def test_returns_audio_head_output(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert isinstance(output, AudioHeadOutput)
        assert output.loss is not None
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

    def test_loss_differentiable(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()
        assert head.text_embedding.weight.grad is not None

    def test_batch_size_one(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (1, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)


class TestForwardInference:
    def test_no_targets_returns_codes(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (1, 10))
        output = head(tokens)
        assert isinstance(output, AudioHeadOutput)
        assert output.codes is not None
        assert output.codes.dim() == 2
        assert output.codes.shape[0] == 1


class TestStateDict:
    def test_contains_all_trainable_params(self, head):
        state = head.state_dict()
        has_text_emb = any(k.startswith("text_embedding.") for k in state)
        has_codec_emb = any(k.startswith("token_embedding.") for k in state)
        has_decoder = any(k.startswith("decoder.") for k in state)
        assert has_text_emb
        assert has_codec_emb
        assert has_decoder

    def test_not_empty(self, head):
        state = head.state_dict()
        assert len(state) > 0

    def test_load_state_dict(self, head):
        original = head.state_dict()
        with torch.no_grad():
            head.text_embedding.weight.fill_(0.0)
        head.load_state_dict(original)
        restored = head.state_dict()
        for key in original:
            assert torch.allclose(original[key], restored[key])


class TestGradientFlow:
    def test_gradients_flow_to_all_components(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()

        assert head.text_embedding.weight.grad is not None, "No gradient for text_embedding"
        assert not torch.isnan(head.text_embedding.weight.grad).any()

        has_decoder_grad = False
        for name, param in head.decoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_decoder_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient for decoder.{name}"
        assert has_decoder_grad, "Decoder should have gradients"

        # token_embedding is used as both embedding and output head (weight tying)
        assert head.token_embedding.weight.grad is not None, (
            "token_embedding should have gradients (used as output head)"
        )

    def test_gradient_magnitude_reasonable(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()
        for name, param in head.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"


class TestParameterCount:
    def test_has_parameters(self, head):
        total = sum(p.numel() for p in head.parameters())
        assert total > 0

    def test_all_trainable(self, head):
        total = sum(p.numel() for p in head.parameters())
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        assert trainable == total

    def test_no_separate_head_layer(self):
        """Prediction uses token_embedding weight directly (no separate head)."""
        cfg = _make_config()
        head = AudioHead(cfg)
        assert not hasattr(head, "head"), (
            "head layer should not exist; uses token_embedding.weight via F.linear"
        )


class TestTrainMode:
    def test_train_mode(self, head):
        head.train()
        assert head.training is True
        assert head.text_embedding.training is True
        assert head.decoder.training is True

    def test_eval_mode(self, head):
        head.eval()
        assert head.training is False
        assert head.text_embedding.training is False
        assert head.decoder.training is False


class TestLossDecreases:
    def test_loss_decreases(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)

        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

        loss_initial = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        ).loss.item()

        for _ in range(10):
            optimizer.zero_grad()
            output = head(
                tokens,
                codec_labels=labels,
                codec_input_ids=inp_ids,
                codec_attention_mask=attn_mask,
            )
            output.loss.backward()
            optimizer.step()

        loss_final = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        ).loss.item()
        assert loss_final < loss_initial


class TestEdgeCases:
    def test_short_sequence(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (1, 3))
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 5)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(output.loss)

    def test_long_sequence(self, head):
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (1, 50))
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 100)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(output.loss)

    def test_high_token_ids(self, head):
        tokens = torch.randint(TEXT_VOCAB_SIZE - 10, TEXT_VOCAB_SIZE, (2, 10))
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

    def test_zero_token_ids(self, head):
        tokens = torch.zeros(2, 10, dtype=torch.long)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(output.loss)

    def test_decode_without_neucodec_raises(self, head):
        assert head.neucodec_model is None


class TestDevicePlacement:
    def test_to_device(self, head):
        head.to(device="cpu")
        for param in head.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_device(self, head):
        device = torch.device("cpu")
        head.to(device)
        tokens = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10), device=device)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        labels, inp_ids, attn_mask = labels.to(device), inp_ids.to(device), attn_mask.to(device)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert output.loss.device == device


class TestSaveLoad:
    def test_save_and_load_pretrained(self, head, tmp_path):
        head.save_pretrained(tmp_path)
        loaded = AudioHead.from_pretrained(tmp_path)
        assert isinstance(loaded, AudioHead)
        assert loaded.config.decoder_dim == head.config.decoder_dim
        assert loaded.config.text_vocab_size == head.config.text_vocab_size
        for key in head.state_dict():
            assert torch.allclose(head.state_dict()[key], loaded.state_dict()[key]), (
                f"Mismatch for {key}"
            )

    def test_config_serialization(self, head, tmp_path):
        head.config.save_pretrained(tmp_path)
        loaded_config = AudioHeadConfig.from_pretrained(tmp_path)
        assert loaded_config.decoder_dim == head.config.decoder_dim
        assert loaded_config.text_vocab_size == head.config.text_vocab_size
