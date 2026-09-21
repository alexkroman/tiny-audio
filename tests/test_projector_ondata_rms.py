"""The on-data projector output-RMS diagnostic.

`measure_output_rms()` probes with isotropic noise, which understates a trained
projector because real encoder features are directional and `linear_1` learns
to align with them (1.46x on the shipped top4 checkpoint). These tests pin the
on-data companion series that `_encode_audio` stashes for the trainer.
"""

import math

import pytest
import torch

from scripts.train import ASRTrainer


@pytest.fixture
def model(base_asr_model):
    """`base_asr_model` is session-scoped, so restore what these tests mutate.

    Both the training flag and the stash leak into every later test otherwise,
    and the device is whatever an earlier test last moved the model to.
    """
    was_training = base_asr_model.training
    previous = getattr(base_asr_model, "_last_audio_embed_rms", None)
    try:
        yield base_asr_model
    finally:
        base_asr_model.train(was_training)
        base_asr_model._last_audio_embed_rms = previous


def _encode(model, batch=2):
    """Mirror the token-count arithmetic the collator does at train.py:965."""
    mel_len = 3000
    device = next(model.parameters()).device
    audio_attention_mask = torch.ones(batch, mel_len, dtype=torch.long, device=device)
    encoder_lengths = model._compute_encoder_output_lengths(audio_attention_mask)
    counts = model.projector.get_output_length(encoder_lengths).to(torch.long)
    return model._encode_audio(
        audio_features=torch.zeros(batch, 80, mel_len, device=device),
        expected_token_counts=counts,
    )


class TestOnDataRMSStash:
    def test_training_forward_stashes_a_scalar_tensor(self, model):
        model._last_audio_embed_rms = None
        model.train()
        with torch.no_grad():
            _encode(model)

        rms = model._last_audio_embed_rms
        assert isinstance(rms, torch.Tensor)
        assert rms.ndim == 0

    def test_stashed_value_is_finite_and_positive(self, model):
        model.train()
        with torch.no_grad():
            _encode(model)

        value = model._last_audio_embed_rms.item()
        assert math.isfinite(value)
        assert value > 0

    def test_matches_rms_of_the_returned_embeddings(self, model):
        """The stash must measure the packed tensor, not the padded one."""
        model.train()
        with torch.no_grad():
            packed = _encode(model)

        expected = packed.float().pow(2).mean().sqrt().item()
        assert model._last_audio_embed_rms.item() == expected

    def test_eval_forward_does_not_update_the_stash(self, model):
        """Gated on self.training so generate() stays free of the reduction."""
        model.train()
        with torch.no_grad():
            _encode(model)
        during_training = model._last_audio_embed_rms.clone()

        model.eval()
        with torch.no_grad():
            _encode(model, batch=1)

        assert model._last_audio_embed_rms.item() == during_training.item()


class TestTrainerMetric:
    """`_projector_output_rms` only reads `self.model`, so a stub suffices."""

    def test_emits_both_series_once_the_stash_exists(self, model):
        model.train()
        with torch.no_grad():
            _encode(model)

        stub = type("S", (), {"model": model})()
        metrics = ASRTrainer._projector_output_rms(stub)

        assert "projector/output_rms_over_embed" in metrics
        assert metrics["projector/output_rms_over_embed_ondata"] > 0

    def test_probe_series_survives_a_missing_stash(self, model):
        """First log fires before any training step; the probe must still emit."""
        model._last_audio_embed_rms = None
        stub = type("S", (), {"model": model})()
        metrics = ASRTrainer._projector_output_rms(stub)

        assert "projector/output_rms_over_embed" in metrics
        assert "projector/output_rms_over_embed_ondata" not in metrics
