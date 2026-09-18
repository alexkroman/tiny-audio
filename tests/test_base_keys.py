"""Tests for decoder-to-base tensor name mapping (scripts/debug/base_keys.py)."""

import torch

from scripts.debug.base_keys import base_key_candidates, resolve_base_key, resolve_base_tensor

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
