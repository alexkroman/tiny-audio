"""Tests for AudioHead: frozen LLM + frozen neutts-nano backbone + trainable projector.

Architecture under test:
  Text tokens → frozen LLM (SmolLM3) → hidden states (llm_dim)
  → Projector MLP (trainable, llm_dim → backbone_dim)
  → Concat with codec embeddings → neutts-nano backbone (frozen)
  → lm_head → speech token logits → NeuCodec codes
"""

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


@pytest.fixture(scope="session")
def config():
    """AudioHeadConfig using real models (downloaded once per session)."""
    return AudioHeadConfig(
        tts_model_id="neuphonic/neutts-nano",
        llm_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",  # Small LLM for fast tests
        projector_hidden=128,  # Small for fast tests
        max_audio_tokens=20,
    )


@pytest.fixture(scope="session")
def head(config):
    """Create AudioHead with real frozen models."""
    return AudioHead(config)


def _make_codec_inputs(batch_size=2, audio_len=20):
    """Build synthetic codec inputs (mimicking S2SDataCollator output).

    Uses BOS_TOKEN/EOS_TOKEN constants that the S2SDataCollator produces.
    """
    codec_input_ids = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    codec_input_ids[:, 0] = BOS_TOKEN  # BOS at start

    codec_labels = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    codec_labels[:, -1] = EOS_TOKEN  # EOS at end
    # Mask ~20% as padding
    mask = torch.rand(batch_size, audio_len) > 0.8
    codec_labels[mask] = -100

    codec_attention_mask = torch.ones(batch_size, audio_len, dtype=torch.long)

    return codec_labels, codec_input_ids, codec_attention_mask


def _make_text_tokens(head, batch_size=2, seq_len=10):
    """Create random text tokens within the LLM's vocab range."""
    # Use a safe range within the LLM's vocabulary
    return torch.randint(0, 1000, (batch_size, seq_len))


def _make_llm_hidden_states(head, batch_size=2, seq_len=10):
    """Create synthetic LLM hidden states matching the LLM's hidden dim."""
    llm_dim = head.llm.config.hidden_size
    return torch.randn(batch_size, seq_len, llm_dim, dtype=torch.bfloat16)


class TestInit:
    def test_backbone_is_frozen(self, head):
        """All backbone parameters should have requires_grad=False."""
        for name, param in head.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

    def test_llm_is_frozen(self, head):
        """All LLM parameters should have requires_grad=False."""
        for name, param in head.llm.named_parameters():
            assert not param.requires_grad, f"LLM param {name} should be frozen"

    def test_projector_is_trainable(self, head):
        """All projector parameters should have requires_grad=True."""
        for name, param in head.projector.named_parameters():
            assert param.requires_grad, f"Projector param {name} should be trainable"

    def test_backbone_in_eval_mode(self, head):
        """Backbone should be in eval mode (no dropout, etc.)."""
        assert not head.backbone.training

    def test_llm_in_eval_mode(self, head):
        """LLM should be in eval mode."""
        assert not head.llm.training

    def test_speech_token_offset_resolved(self, head):
        """speech_token_offset should be a valid token ID."""
        assert head.speech_token_offset > 0
        assert isinstance(head.speech_token_offset, int)

    def test_speech_start_id_resolved(self, head):
        assert head.speech_start_id > 0

    def test_speech_end_id_resolved(self, head):
        assert head.speech_end_id > 0

    def test_projector_shape(self, head):
        """Projector: Linear → RMSNorm → GELU → Linear → RMSNorm."""
        backbone_dim = head.backbone.config.hidden_size
        llm_dim = head.llm.config.hidden_size
        # First linear: llm_dim → projector_hidden
        assert head.projector[0].in_features == llm_dim
        assert head.projector[0].out_features == head.config.projector_hidden
        # Second linear: projector_hidden → backbone_dim (index 3)
        assert head.projector[3].in_features == head.config.projector_hidden
        assert head.projector[3].out_features == backbone_dim
        # Final RMSNorm on backbone_dim (index 4)
        assert head.projector[4].weight.shape[0] == backbone_dim

    def test_no_neucodec_model_by_default(self, head):
        assert head.neucodec_model is None

    def test_is_pretrained_model(self, head):
        from transformers import PreTrainedModel

        assert isinstance(head, PreTrainedModel)

    def test_config_class(self):
        assert AudioHead.config_class is AudioHeadConfig


class TestTokenMapping:
    def test_codec_to_speech_ids(self, head):
        """NeuCodec code 0 should map to speech_token_offset."""
        codes = torch.tensor([0, 1, 100, 65535])
        speech_ids = head._codec_to_speech_ids(codes)
        assert speech_ids[0].item() == head.speech_token_offset
        assert speech_ids[1].item() == head.speech_token_offset + 1
        assert speech_ids[3].item() == head.speech_token_offset + 65535

    def test_speech_ids_to_codec_roundtrip(self, head):
        """Mapping should be invertible."""
        codes = torch.tensor([0, 42, 65535])
        speech_ids = head._codec_to_speech_ids(codes)
        recovered = head._speech_ids_to_codec(speech_ids)
        assert torch.equal(codes, recovered)

    def test_map_collator_ids_bos(self, head):
        """BOS_TOKEN (65536) should map to audio_start."""
        ids = torch.tensor([[BOS_TOKEN]])
        mapped = head._map_collator_ids_to_speech(ids)
        assert mapped[0, 0].item() == head.speech_start_id

    def test_map_collator_ids_eos(self, head):
        """EOS_TOKEN (65537) should map to audio_end."""
        ids = torch.tensor([[EOS_TOKEN]])
        mapped = head._map_collator_ids_to_speech(ids)
        assert mapped[0, 0].item() == head.speech_end_id

    def test_map_collator_ids_codec(self, head):
        """Regular codec codes should map to speech token IDs."""
        ids = torch.tensor([[0, 100, 65535]])
        mapped = head._map_collator_ids_to_speech(ids)
        assert mapped[0, 0].item() == head.speech_token_offset
        assert mapped[0, 1].item() == head.speech_token_offset + 100
        assert mapped[0, 2].item() == head.speech_token_offset + 65535

    def test_map_collator_labels_ignore(self, head):
        """Labels with -100 should stay as -100."""
        labels = torch.tensor([[-100, 42, -100]])
        mapped = head._map_collator_labels_to_speech(labels)
        assert mapped[0, 0].item() == -100
        assert mapped[0, 2].item() == -100

    def test_map_collator_labels_eos(self, head):
        """EOS_TOKEN in labels should map to audio_end."""
        labels = torch.tensor([[42, EOS_TOKEN]])
        mapped = head._map_collator_labels_to_speech(labels)
        assert mapped[0, 1].item() == head.speech_end_id


class TestForwardTraining:
    def test_with_text_token_ids(self, head):
        """Forward with text_token_ids runs through frozen LLM."""
        tokens = _make_text_tokens(head, 2, 10)
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

    def test_with_llm_hidden_states(self, head):
        """Forward with pre-computed LLM hidden states (pipeline mode)."""
        hidden = _make_llm_hidden_states(head, 2, 10)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            llm_hidden_states=hidden,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert isinstance(output, AudioHeadOutput)
        assert output.loss is not None
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)

    def test_loss_differentiable(self, head):
        tokens = _make_text_tokens(head, 2, 10)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()
        # Projector should have gradients
        has_projector_grad = any(p.grad is not None for p in head.projector.parameters())
        assert has_projector_grad, "Projector should have gradients"
        head.zero_grad()

    def test_batch_size_one(self, head):
        tokens = _make_text_tokens(head, 1, 10)
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 20)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert output.loss.dim() == 0
        assert not torch.isnan(output.loss)

    def test_raises_without_input(self, head):
        """Must provide either text_token_ids or llm_hidden_states."""
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)
        with pytest.raises(ValueError, match="Either text_token_ids or llm_hidden_states"):
            head(
                codec_labels=labels,
                codec_input_ids=inp_ids,
                codec_attention_mask=attn_mask,
            )


class TestForwardInference:
    def test_no_targets_returns_codes(self, head):
        tokens = _make_text_tokens(head, 1, 10)
        head_eval = head
        head_eval.eval()
        with torch.no_grad():
            output = head_eval(tokens)
        assert isinstance(output, AudioHeadOutput)
        assert output.codes is not None
        assert output.codes.dim() == 2
        assert output.codes.shape[0] == 1
        head.train()

    def test_codes_in_valid_range(self, head):
        tokens = _make_text_tokens(head, 1, 5)
        head.eval()
        with torch.no_grad():
            output = head(tokens)
        if output.codes.numel() > 0:
            assert (output.codes >= 0).all()
            assert (output.codes < NEUCODEC_VOCAB_SIZE).all()
        head.train()

    def test_inference_with_hidden_states(self, head):
        """Inference works with pre-computed hidden states."""
        hidden = _make_llm_hidden_states(head, 1, 10)
        head.eval()
        with torch.no_grad():
            output = head(llm_hidden_states=hidden)
        assert output.codes is not None
        assert output.codes.dim() == 2
        head.train()


class TestGradientFlow:
    def test_gradients_only_flow_to_projector(self, head):
        """Backbone and LLM should be frozen; only projector gets gradients."""
        tokens = _make_text_tokens(head, 2, 10)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)

        head.zero_grad()
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()

        # Projector should have gradients
        projector_has_grad = False
        for name, param in head.projector.named_parameters():
            if param.grad is not None:
                projector_has_grad = True
                assert not torch.isnan(param.grad).any(), f"NaN grad in projector.{name}"
        assert projector_has_grad, "Projector should have gradients"

        # Backbone should NOT have gradients (frozen)
        for name, param in head.backbone.named_parameters():
            assert param.grad is None, f"Backbone param {name} should not have gradients"

        # LLM should NOT have gradients (frozen)
        for name, param in head.llm.named_parameters():
            assert param.grad is None, f"LLM param {name} should not have gradients"

        head.zero_grad()

    def test_gradient_magnitude_reasonable(self, head):
        tokens = _make_text_tokens(head, 2, 10)
        labels, inp_ids, attn_mask = _make_codec_inputs(2, 20)

        head.zero_grad()
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        output.loss.backward()

        for name, param in head.projector.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1e6, f"Exploding gradient for projector.{name}: {grad_norm}"

        head.zero_grad()


class TestStateDict:
    def test_contains_only_projector_keys(self, head):
        """state_dict should only contain projector weights."""
        state = head.state_dict()
        assert len(state) > 0
        for key in state:
            assert key.startswith("projector."), f"Unexpected key in state_dict: {key}"

    def test_projector_weight_count(self, head):
        """Should have weights + biases for 2 linear layers + 2 RMSNorm weights."""
        state = head.state_dict()
        # projector = Sequential(Linear, RMSNorm, GELU, Linear, RMSNorm)
        # Linear has weight + bias = 2 params each → 4, 2x RMSNorm weight → 2 = 6 total
        assert len(state) == 6, (
            f"Expected 6 projector params, got {len(state)}: {list(state.keys())}"
        )


class TestParameterCount:
    def test_trainable_is_projector_only(self, head):
        """Only projector parameters should be trainable."""
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        projector_params = sum(p.numel() for p in head.projector.parameters())
        assert trainable == projector_params

    def test_total_much_larger_than_trainable(self, head):
        """Total params should be >> trainable (LLM + backbone are frozen)."""
        total = sum(p.numel() for p in head.parameters())
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        assert total > trainable * 10, "Frozen models should dwarf projector in param count"


class TestTrainMode:
    def test_train_mode_keeps_frozen_models_eval(self, head):
        """Even in train mode, backbone and LLM should stay in eval mode."""
        head.train()
        assert head.training is True
        # Backbone and LLM should remain in eval mode to disable dropout etc.
        assert not head.backbone.training
        assert not head.llm.training

    def test_eval_mode(self, head):
        head.eval()
        assert head.training is False
        head.train()  # Reset


class TestEdgeCases:
    def test_short_sequence(self, head):
        tokens = _make_text_tokens(head, 1, 3)
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 5)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(output.loss)

    def test_long_sequence(self, head):
        tokens = _make_text_tokens(head, 1, 50)
        labels, inp_ids, attn_mask = _make_codec_inputs(1, 100)
        output = head(
            tokens,
            codec_labels=labels,
            codec_input_ids=inp_ids,
            codec_attention_mask=attn_mask,
        )
        assert not torch.isnan(output.loss)

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


class TestDevicePlacement:
    def test_to_device(self, head):
        head.to(device="cpu")
        for param in head.projector.parameters():
            assert param.device.type == "cpu"

    def test_forward_respects_device(self, head):
        device = torch.device("cpu")
        head.to(device)
        tokens = _make_text_tokens(head, 2, 10).to(device)
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
        assert loaded.config.tts_model_id == head.config.tts_model_id
        assert loaded.config.llm_model_id == head.config.llm_model_id
        assert loaded.config.projector_hidden == head.config.projector_hidden
        # Projector weights should match
        for key in head.state_dict():
            assert torch.allclose(head.state_dict()[key], loaded.state_dict()[key]), (
                f"Mismatch for {key}"
            )

    def test_config_serialization(self, head, tmp_path):
        head.config.save_pretrained(tmp_path)
        loaded_config = AudioHeadConfig.from_pretrained(tmp_path)
        assert loaded_config.tts_model_id == head.config.tts_model_id
        assert loaded_config.llm_model_id == head.config.llm_model_id
        assert loaded_config.projector_hidden == head.config.projector_hidden
        assert loaded_config.max_audio_tokens == head.config.max_audio_tokens
