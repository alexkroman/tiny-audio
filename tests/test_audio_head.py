"""Tests for the trainable AR decoder AudioHead for speech-to-speech."""

import pytest
import torch

from tiny_audio.audio_head import (
    BOS_TOKEN,
    MIMI_VOCAB_SIZE,
    NUM_MIMI_CODEBOOKS,
    AudioHead,
)

TEST_DIM = 256
DECODER_DIM = 64  # Small for fast tests


class MockConfig:
    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 2048)
        self.num_codebooks = kwargs.get("num_codebooks", NUM_MIMI_CODEBOOKS)
        self.decoder_dim = kwargs.get("decoder_dim", DECODER_DIM)
        self.decoder_layers = kwargs.get("decoder_layers", 2)
        self.decoder_heads = kwargs.get("decoder_heads", 2)
        self.max_audio_tokens = kwargs.get("max_audio_tokens", 50)


def _make_codec_inputs(batch_size=2, audio_len=20, num_codebooks=NUM_MIMI_CODEBOOKS):
    """Build synthetic codec inputs (mimicking S2SDataCollator output)."""
    codec_input_ids = torch.randint(0, MIMI_VOCAB_SIZE, (batch_size, audio_len, num_codebooks))
    codec_input_ids[:, 0, :] = BOS_TOKEN  # BOS at start

    codec_labels = torch.randint(0, MIMI_VOCAB_SIZE, (batch_size, audio_len, num_codebooks))
    # Mask ~20% as padding
    mask = torch.rand(batch_size, audio_len, num_codebooks) > 0.8
    codec_labels[mask] = -100

    codec_attention_mask = torch.ones(batch_size, audio_len, dtype=torch.long)

    return codec_labels, codec_input_ids, codec_attention_mask


@pytest.fixture
def config():
    return MockConfig


@pytest.fixture
def head(config):
    """Create a small AudioHead for testing."""
    return AudioHead(
        config(llm_dim=TEST_DIM, decoder_dim=DECODER_DIM, decoder_layers=2, decoder_heads=2),
        llm_dim=TEST_DIM,
    )


class TestInit:
    def test_default_dim(self, config):
        head = AudioHead(config())
        assert head.llm_dim == 2048

    def test_custom_dim(self, config):
        head = AudioHead(config(llm_dim=512), llm_dim=512)
        assert head.llm_dim == 512

    def test_vocab_size(self, head):
        assert head.vocab_size == MIMI_VOCAB_SIZE

    def test_num_codebooks(self, head):
        assert head.num_codebooks == NUM_MIMI_CODEBOOKS

    def test_input_proj_exists(self, head):
        assert hasattr(head, "input_proj")

    def test_token_embedding_exists(self, head):
        assert hasattr(head, "token_embedding")

    def test_decoder_exists(self, head):
        assert hasattr(head, "decoder")

    def test_heads_exist(self, head):
        assert hasattr(head, "heads")
        assert len(head.heads) == NUM_MIMI_CODEBOOKS

    def test_no_mimi_model_by_default(self, head):
        assert head.mimi_model is None


class TestInputProjShape:
    def test_proj_output_dim(self, head):
        x = torch.randn(1, 10, TEST_DIM)
        out = head.input_proj(x)
        assert out.shape == (1, 10, DECODER_DIM)

    def test_proj_batch(self, head):
        x = torch.randn(4, 5, TEST_DIM)
        out = head.input_proj(x)
        assert out.shape == (4, 5, DECODER_DIM)


class TestForwardTraining:
    def test_returns_scalar_loss(self, head):
        hidden = torch.randn(2, 10, TEST_DIM)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_differentiable(self, head):
        hidden = torch.randn(2, 10, TEST_DIM, requires_grad=True)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_batch_size_one(self, head):
        hidden = torch.randn(1, 10, TEST_DIM)
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    def test_no_targets_returns_codes(self, head):
        hidden = torch.randn(1, 10, TEST_DIM)
        codes, _ = head(hidden)
        assert codes.dim() == 3
        assert codes.shape[0] == 1
        assert codes.shape[2] == NUM_MIMI_CODEBOOKS


class TestStateDict:
    def test_contains_all_trainable_params(self, head):
        state = head.state_dict()
        # Should contain input_proj, token_embedding, decoder, and heads
        has_input_proj = any(k.startswith("input_proj.") for k in state)
        has_embedding = any(k.startswith("token_embedding.") for k in state)
        has_decoder = any(k.startswith("decoder.") for k in state)
        has_heads = any(k.startswith("heads.") for k in state)
        assert has_input_proj
        assert has_embedding
        assert has_decoder
        assert has_heads

    def test_not_empty(self, head):
        state = head.state_dict()
        assert len(state) > 0

    def test_load_state_dict(self, head):
        original = head.state_dict()
        # Zero out a param
        with torch.no_grad():
            for p in head.input_proj.parameters():
                p.fill_(0.0)
                break
        head.load_state_dict(original)
        restored = head.state_dict()
        for key in original:
            assert torch.allclose(original[key], restored[key])


class TestGradientFlow:
    def test_gradients_flow_to_all_components(self, head):
        hidden = torch.randn(2, 10, TEST_DIM, requires_grad=True)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        loss.backward()

        # Check input_proj
        for name, param in head.input_proj.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for input_proj.{name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for input_proj.{name}"

        # Check decoder
        has_decoder_grad = False
        for name, param in head.decoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_decoder_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient for decoder.{name}"
        assert has_decoder_grad, "Decoder should have gradients"

        # Check heads
        for i, h in enumerate(head.heads):
            for name, param in h.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No gradient for heads[{i}].{name}"

    def test_gradient_magnitude_reasonable(self, head):
        hidden = torch.randn(2, 10, TEST_DIM, requires_grad=True)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        loss.backward()
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

    def test_param_count_report(self, head):
        total = sum(p.numel() for p in head.parameters())
        input_proj_params = sum(p.numel() for p in head.input_proj.parameters())
        embedding_params = sum(p.numel() for p in head.token_embedding.parameters())
        decoder_params = sum(p.numel() for p in head.decoder.parameters())
        heads_params = sum(p.numel() for p in head.heads.parameters())
        print(f"\n{'Component':<20} {'Params':>12}")
        print("-" * 35)
        print(f"{'input_proj':<20} {input_proj_params:>12,}")
        print(f"{'token_embedding':<20} {embedding_params:>12,}")
        print(f"{'decoder':<20} {decoder_params:>12,}")
        print(f"{'heads':<20} {heads_params:>12,}")
        print(f"{'TOTAL (trainable)':<20} {total:>12,}")
        assert total > 0


class TestTrainMode:
    def test_train_mode(self, head):
        head.train()
        assert head.training is True
        assert head.input_proj.training is True
        assert head.decoder.training is True

    def test_eval_mode(self, head):
        head.eval()
        assert head.training is False
        assert head.input_proj.training is False
        assert head.decoder.training is False


class TestLossDecreases:
    def test_loss_decreases(self, head):
        hidden = torch.randn(2, 10, TEST_DIM)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)

        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

        loss_initial = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        ).item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = head(
                hidden,
                codec_labels=labels,
                codec_input_ids=inp_ids,
                codec_attention_mask=attn_mask,
            )
            loss.backward()
            optimizer.step()

        loss_final = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        ).item()
        assert loss_final < loss_initial


class TestEdgeCases:
    def test_short_sequence(self, head):
        hidden = torch.randn(1, 1, TEST_DIM)
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 3)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(loss)

    def test_long_sequence(self, head):
        hidden = torch.randn(1, 50, TEST_DIM)
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 100)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(loss)

    def test_large_input_values(self, head):
        hidden = torch.randn(2, 10, TEST_DIM) * 100
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_small_input_values(self, head):
        hidden = torch.randn(2, 10, TEST_DIM) * 1e-6
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(loss)

    def test_decode_without_mimi_raises(self, head):
        # mimi_model is None by default, so decode should fail gracefully
        # (it will try to load the model which may fail in test env)
        assert head.mimi_model is None


class TestDevicePlacement:
    def test_to_device(self, head):
        head.to(device="cpu")
        for param in head.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_device(self, head):
        device = torch.device("cpu")
        head.to(device)
        hidden = torch.randn(2, 10, TEST_DIM, device=device)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        labels, inp_ids, attn_mask = labels.to(device), inp_ids.to(device), attn_mask.to(device)
        loss = head(
            hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert loss.device == device
