"""Tests for the Dia-based AudioHead for speech-to-speech."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tiny_audio.audio_head import DAC_VOCAB_SIZE, NUM_DAC_CODEBOOKS, AudioHead


class MockDiaModel(nn.Module):
    """Mock Dia decoder for testing without downloading 1.6B params.

    Produces a differentiable loss from encoder_outputs so gradients
    flow back to the MLP projector, matching real Dia's behavior.
    """

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

    def forward(
        self,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        hidden = encoder_outputs.last_hidden_state
        pooled = hidden.mean(dim=(0, 1), keepdim=True)

        loss = None
        if labels is not None:
            # labels: [batch * num_channels, seq_len]
            expanded = pooled.expand(labels.shape[0], labels.shape[1], -1)
            logits = self.lm_head(expanded)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss)

    def generate(self, encoder_outputs=None, max_new_tokens=None, **kwargs):
        batch = encoder_outputs.last_hidden_state.shape[0]
        return torch.randint(0, self.vocab_size, (batch, max_new_tokens or 90, 9))


TEST_DIM = 256


class MockConfig:
    def __init__(self, **kwargs):
        self.llm_dim = kwargs.get("llm_dim", 2048)
        self.dia_model_id = kwargs.get("dia_model_id", "nari-labs/Dia-1.6B-0626")
        self.max_audio_tokens = kwargs.get("max_audio_tokens", 500)


def _make_dia_inputs(batch_size=2, seq_len=30):
    """Build synthetic Dia-ready inputs (mimicking S2SDataCollator output)."""
    dia_labels = torch.randint(0, DAC_VOCAB_SIZE, (batch_size * NUM_DAC_CODEBOOKS, seq_len))
    # Mask ~20% of positions as padding
    dia_labels[dia_labels > 900] = -100
    dia_decoder_input_ids = torch.randint(
        0, DAC_VOCAB_SIZE, (batch_size, seq_len, NUM_DAC_CODEBOOKS)
    )
    dia_decoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return dia_labels, dia_decoder_input_ids, dia_decoder_attention_mask


@pytest.fixture
def config():
    return MockConfig


@pytest.fixture
def head(config):
    """Create a small AudioHead with mock Dia for testing."""
    h = AudioHead(config(llm_dim=TEST_DIM), llm_dim=TEST_DIM)
    h.dia_model = MockDiaModel(AudioHead.DIA_DIM, DAC_VOCAB_SIZE)
    return h


class TestInit:
    def test_default_dim(self, config):
        head = AudioHead(config())
        assert head.llm_dim == 2048

    def test_custom_dim(self, config):
        head = AudioHead(config(llm_dim=512), llm_dim=512)
        assert head.llm_dim == 512

    def test_vocab_size(self, head):
        assert head.vocab_size == DAC_VOCAB_SIZE

    def test_num_codebooks(self, head):
        assert head.num_codebooks == NUM_DAC_CODEBOOKS

    def test_projector_exists(self, head):
        assert hasattr(head, "projector")

    def test_no_codebook_heads(self, head):
        assert not hasattr(head, "codebook_heads")

    def test_dia_loaded(self, head):
        assert head.dia_model is not None


class TestProjectorShape:
    def test_projector_output_dim(self, head):
        x = torch.randn(1, 10, TEST_DIM)
        out = head.projector(x)
        assert out.shape == (1, 10, AudioHead.DIA_DIM)

    def test_projector_batch(self, head):
        x = torch.randn(4, 5, TEST_DIM)
        out = head.projector(x)
        assert out.shape == (4, 5, AudioHead.DIA_DIM)


class TestForwardTraining:
    def test_returns_scalar_loss(self, head):
        hidden = torch.randn(2, 10, TEST_DIM)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_differentiable(self, head):
        hidden = torch.randn(2, 10, TEST_DIM, requires_grad=True)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape

    def test_batch_size_one(self, head):
        hidden = torch.randn(1, 10, TEST_DIM)
        labels, dec_ids, dec_mask = _make_dia_inputs(1, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestForwardInference:
    def test_no_targets_returns_codes(self, head):
        hidden = torch.randn(1, 10, TEST_DIM)
        codes, _ = head(hidden)
        assert codes.dim() == 3
        assert codes.shape[2] == NUM_DAC_CODEBOOKS


class TestStateDict:
    def test_only_projector(self, head):
        state = head.state_dict()
        for key in state:
            assert key.startswith("projector."), f"Unexpected key in state_dict: {key}"

    def test_no_dia_keys(self, head):
        state = head.state_dict()
        for key in state:
            assert "dia_model" not in key, f"Dia key leaked into state_dict: {key}"

    def test_not_empty(self, head):
        state = head.state_dict()
        assert len(state) > 0

    def test_load_state_dict(self, head):
        original = head.state_dict()
        with torch.no_grad():
            for p in head.projector.parameters():
                p.fill_(0.0)
                break
        head.load_state_dict(original, strict=False)
        restored = head.state_dict()
        for key in original:
            assert torch.allclose(original[key], restored[key])


class TestGradientFlow:
    def test_gradients_flow_to_projector(self, head):
        hidden = torch.randn(2, 10, TEST_DIM, requires_grad=True)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        loss.backward()
        for name, param in head.projector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for projector.{name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for projector.{name}"

    def test_dia_params_no_grad(self, head):
        hidden = torch.randn(2, 10, TEST_DIM)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        loss.backward()
        for name, param in head.dia_model.named_parameters():
            assert param.grad is None, f"Dia param {name} should not have grad"

    def test_gradient_magnitude_reasonable(self, head):
        hidden = torch.randn(2, 10, TEST_DIM, requires_grad=True)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        loss.backward()
        for name, param in head.projector.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"


class TestParameterCount:
    def test_has_parameters(self, head):
        total = sum(p.numel() for p in head.projector.parameters())
        assert total > 0

    def test_all_projector_trainable(self, head):
        total = sum(p.numel() for p in head.projector.parameters())
        trainable = sum(p.numel() for p in head.projector.parameters() if p.requires_grad)
        assert trainable == total

    def test_param_count_report(self, head):
        projector_params = sum(p.numel() for p in head.projector.parameters())
        print(f"\n{'Component':<20} {'Params':>12}")
        print("-" * 35)
        print(f"{'projector':<20} {projector_params:>12,}")
        print(f"{'TOTAL (trainable)':<20} {projector_params:>12,}")
        assert projector_params > 0


class TestTrainMode:
    def test_train_keeps_dia_eval(self, head):
        head.train()
        assert head.projector.training is True
        assert head.dia_model.training is False

    def test_eval_mode(self, head):
        head.eval()
        assert head.projector.training is False
        assert head.dia_model.training is False


class TestLossDecreases:
    def test_loss_decreases(self, head):
        hidden = torch.randn(2, 10, TEST_DIM)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)

        head.train()
        optimizer = torch.optim.Adam(head.projector.parameters(), lr=1e-3)

        loss_initial = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        ).item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = head(
                hidden,
                dia_labels=labels,
                dia_decoder_input_ids=dec_ids,
                dia_decoder_attention_mask=dec_mask,
            )
            loss.backward()
            optimizer.step()

        loss_final = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        ).item()
        assert loss_final < loss_initial


class TestEdgeCases:
    def test_short_sequence(self, head):
        hidden = torch.randn(1, 1, TEST_DIM)
        labels, dec_ids, dec_mask = _make_dia_inputs(1, 3)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert not torch.isnan(loss)

    def test_long_sequence(self, head):
        hidden = torch.randn(1, 50, TEST_DIM)
        labels, dec_ids, dec_mask = _make_dia_inputs(1, 200)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert not torch.isnan(loss)

    def test_large_input_values(self, head):
        hidden = torch.randn(2, 10, TEST_DIM) * 100
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_small_input_values(self, head):
        hidden = torch.randn(2, 10, TEST_DIM) * 1e-6
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert not torch.isnan(loss)

    def test_decode_without_dia_processor_raises(self, head):
        codes = torch.randint(0, DAC_VOCAB_SIZE, (1, 20, NUM_DAC_CODEBOOKS))
        head.dia_processor = None
        with pytest.raises(RuntimeError, match="Dia not loaded"):
            head.decode_to_audio(codes)


class TestDevicePlacement:
    def test_to_device(self, head):
        head.to(device="cpu")
        for param in head.projector.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_device(self, head):
        device = torch.device("cpu")
        head.to(device)
        hidden = torch.randn(2, 10, TEST_DIM, device=device)
        labels, dec_ids, dec_mask = _make_dia_inputs(2, 30)
        labels, dec_ids, dec_mask = labels.to(device), dec_ids.to(device), dec_mask.to(device)
        loss = head(
            hidden,
            dia_labels=labels,
            dia_decoder_input_ids=dec_ids,
            dia_decoder_attention_mask=dec_mask,
        )
        assert loss.device == device
