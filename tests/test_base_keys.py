"""Tests for trained-to-base tensor name mapping (scripts/debug/base_keys.py)."""

import torch

from scripts.debug.base_keys import (
    all_base_key_candidates,
    base_key_candidates,
    encoder_base_key_candidates,
    resolve_base_key,
    resolve_base_tensor,
)

PLAIN = "model.layers.0.mlp.gate_proj.weight"
NESTED = "model.language_model.layers.0.mlp.gate_proj.weight"
TRAINED = "language_model.model.layers.0.mlp.gate_proj.weight"


class TestBaseKeyCandidates:
    """Tests for candidate name generation."""

    def test_offers_both_plain_and_nested_forms(self):
        assert base_key_candidates(TRAINED) == (PLAIN, NESTED)

    def test_plain_form_comes_first(self):
        """Plain CausalLM bases are the common case, so they should match first."""
        assert base_key_candidates(TRAINED)[0] == PLAIN

    def test_non_decoder_tensor_yields_no_candidates(self):
        """Empty tuple distinguishes 'not an LM tensor' from 'base lacks it'."""
        assert base_key_candidates("projector.linear_1.weight") == ()
        assert base_key_candidates("audio_tower.encoder.layers.0.weight") == ()

    def test_lm_tensor_outside_model_subtree(self):
        assert base_key_candidates("language_model.lm_head.weight") == ("lm_head.weight",)


class TestResolveBaseKey:
    """Tests for resolution against an actual base weight map."""

    def test_resolves_plain_base(self):
        """Qwen3-0.6B / SmolLM3 layout."""
        assert resolve_base_key(TRAINED, {PLAIN: 1}) == PLAIN

    def test_resolves_nested_base(self):
        """Qwen3.5-2B ships as Qwen3_5ForConditionalGeneration — one level deeper."""
        assert resolve_base_key(TRAINED, {NESTED: 1}) == NESTED

    def test_returns_none_when_absent(self):
        assert resolve_base_key(TRAINED, {"model.layers.0.mlp.up_proj.weight": 1}) is None

    def test_handles_empty_and_none_base(self):
        assert resolve_base_key(TRAINED, {}) is None
        assert resolve_base_key(TRAINED, None) is None


class TestResolveBaseTensor:
    """Tests for tensor retrieval."""

    def test_returns_tensor_from_nested_base(self):
        t = torch.ones(2, 2)
        assert torch.equal(resolve_base_tensor(TRAINED, {NESTED: t}), t)

    def test_returns_none_without_match(self):
        assert resolve_base_tensor(TRAINED, {"unrelated": torch.ones(1)}) is None
        assert resolve_base_tensor("projector.linear_1.weight", {PLAIN: torch.ones(1)}) is None


class TestEncoderBaseKeyCandidates:
    """Tests for the encoder half of the mapping.

    tiny-audio saves the encoder flat under ``audio_tower.``; where the base
    keeps it depends on the architecture. Verified against the shipped
    safetensors headers: granite-speech-5.0 stores all 1,100 tensors under
    ``encoder.``, whisper-* under ``model.encoder.``.

    Before this existed, `compare_to_base` skipped every ``audio_tower.*`` key
    via its ``if not base_key_candidates(tk): continue`` guard and reported
    "Matched tensors: 0" rather than failing -- so the encoder-drift check
    granite_qwen_full prescribes had no instrument behind it.
    """

    TRAINED_ENC = "audio_tower.layers.0.conv.norm.running_mean"

    def test_granite_layout(self):
        base = {"encoder.layers.0.conv.norm.running_mean": 1}
        assert resolve_base_key(self.TRAINED_ENC, base) == "encoder.layers.0.conv.norm.running_mean"

    def test_whisper_layout(self):
        assert resolve_base_key("audio_tower.conv1.weight", {"model.encoder.conv1.weight": 1}) == (
            "model.encoder.conv1.weight"
        )

    def test_glm_composite_layout(self):
        assert resolve_base_key("audio_tower.x.weight", {"model.audio_tower.x.weight": 1}) == (
            "model.audio_tower.x.weight"
        )

    def test_bare_encoder_checkpoint(self):
        assert resolve_base_key("audio_tower.x.weight", {"x.weight": 1}) == "x.weight"

    def test_qualified_name_wins_over_bare(self):
        """Most-specific prefix first, so a base carrying both is unambiguous."""
        base = {"encoder.x.weight": 1, "x.weight": 2}
        assert resolve_base_key("audio_tower.x.weight", base) == "encoder.x.weight"

    def test_non_encoder_tensor_yields_no_candidates(self):
        assert encoder_base_key_candidates("projector.linear_1.weight") == ()
        assert encoder_base_key_candidates(TRAINED) == ()

    def test_decoder_and_encoder_candidates_stay_disjoint(self):
        """The union must never mix the two towers' namespaces."""
        assert all_base_key_candidates(TRAINED) == base_key_candidates(TRAINED)
        assert all_base_key_candidates(self.TRAINED_ENC) == encoder_base_key_candidates(
            self.TRAINED_ENC
        )
        assert all_base_key_candidates("projector.linear_1.weight") == ()

    def test_encoder_tensor_does_not_resolve_against_a_text_base(self):
        """analyze_weights passes decoder tensors with a text base; no crosstalk."""
        assert resolve_base_key(self.TRAINED_ENC, {PLAIN: 1, NESTED: 2}) is None

    def test_resolve_tensor_returns_the_encoder_value(self):
        want = torch.ones(3)
        assert torch.equal(resolve_base_tensor("audio_tower.x", {"encoder.x": want}), want)
