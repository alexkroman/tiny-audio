"""Tests for per-module LoRA rank/alpha overrides (`lora_rank_pattern`).

These exist because `all-linear` is blind to matrix shape. On Qwen3.5 the
gated-DeltaNet gates `in_proj_a` / `in_proj_b` are (16, 2048), so a rank-64
adapter on them is capped at rank 16 by the output dimension and spends 4x
what full fine-tuning the same matrix would cost.

The trap this file guards is the SECOND half of that fix: PEFT resolves
`rank_pattern` and `alpha_pattern` with independent lookups, each falling back
to `r` / `lora_alpha`. A recipe that overrides rank alone does not get "the
same adapter, smaller" -- it gets alpha/r = 128/16 = 8.0 instead of 2.0, a 4x
HOTTER update on the exact modules it was trying to spend less on. That is
silent: the config looks conservative, the checkpoint loads, and the scale is
wrong. So the scale assertion here matters more than the shape one.
"""

from typing import Any

import pytest
import torch
from conftest import stub
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel


class GateBlock(nn.Module):
    """Mirrors Qwen3.5's linear_attn geometry: two (16, 2048) gates, two wide ones."""

    def __init__(self, dim: int = 2048, heads: int = 16) -> None:
        super().__init__()
        self.in_proj_a = nn.Linear(dim, heads, bias=False)
        self.in_proj_b = nn.Linear(dim, heads, bias=False)
        self.in_proj_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class FakeDecoderConfig(PretrainedConfig):
    model_type = "fake_decoder"


class FakeDecoder(PreTrainedModel):
    config_class = FakeDecoderConfig

    def __init__(self, depth: int = 2) -> None:
        super().__init__(FakeDecoderConfig())
        self.layers = nn.ModuleList(GateBlock() for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


TARGETS = ["in_proj_a", "in_proj_b", "in_proj_qkv", "out_proj"]


def _identity_peft(model: nn.Module, _config: object) -> nn.Module:
    return model


def _adapters(model: nn.Module) -> dict[str, tuple[int, float, float]]:
    """Map leaf module name -> (r, alpha, scaling) for layer 0's adapters."""
    out: dict[str, tuple[int, float, float]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer) and ".layers.0." in f".{name}.":
            leaf = name.split(".")[-1]
            out[leaf] = (mod.r["default"], mod.lora_alpha["default"], mod.scaling["default"])
    return out


class TestASRConfigFields:
    def test_defaults_are_empty_dicts_not_none(self) -> None:
        # PEFT's LoraConfig defaults these to {} and `get_pattern_key` iterates
        # the keys unconditionally, so None would be a TypeError at setup time.
        config = ASRConfig()
        assert config.lora_rank_pattern == {}
        assert config.lora_alpha_pattern == {}

    def test_survives_json_round_trip(self) -> None:
        # These land in config.json and have to come back on from_pretrained.
        config = ASRConfig(
            lora_rank_pattern={"in_proj_a": 16}, lora_alpha_pattern={"in_proj_a": 32}
        )
        revived = ASRConfig.from_dict(config.to_dict())
        assert revived.lora_rank_pattern == {"in_proj_a": 16}
        assert revived.lora_alpha_pattern == {"in_proj_a": 32}


class TestPatternApplication:
    def test_rank_override_applies_only_to_named_modules(self) -> None:
        model = get_peft_model(
            FakeDecoder(),
            LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=TARGETS,
                rank_pattern={"in_proj_a": 16, "in_proj_b": 16},
                alpha_pattern={"in_proj_a": 32, "in_proj_b": 32},
                bias="none",
            ),
        )
        got = _adapters(model)
        assert got["in_proj_a"][0] == 16
        assert got["in_proj_b"][0] == 16
        # Unnamed targets keep the global rank.
        assert got["in_proj_qkv"][0] == 64
        assert got["out_proj"][0] == 64

    def test_scale_is_preserved_across_the_override(self) -> None:
        """The whole reason `alpha_pattern` is set alongside `rank_pattern`."""
        model = get_peft_model(
            FakeDecoder(),
            LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=TARGETS,
                rank_pattern={"in_proj_a": 16, "in_proj_b": 16},
                alpha_pattern={"in_proj_a": 32, "in_proj_b": 32},
                bias="none",
            ),
        )
        for leaf, (_, _, scaling) in _adapters(model).items():
            assert scaling == 2.0, f"{leaf} scale drifted to {scaling}"

    def test_rank_without_alpha_silently_multiplies_the_scale(self) -> None:
        """Pin the failure mode, so nobody 'simplifies' alpha_pattern away.

        This is not desired behaviour being asserted -- it is the booby trap,
        recorded so a future edit that drops `alpha_pattern` fails here loudly
        instead of shipping a 4x-hot adapter.
        """
        model = get_peft_model(
            FakeDecoder(),
            LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=TARGETS,
                rank_pattern={"in_proj_a": 16},
                bias="none",
            ),
        )
        got = _adapters(model)
        assert got["in_proj_a"][2] == 8.0  # 128 / 16, not the intended 2.0
        assert got["out_proj"][2] == 2.0

    def test_override_shrinks_the_parameter_budget(self) -> None:
        def lora_params(**kwargs: Any) -> int:
            model = get_peft_model(
                FakeDecoder(),
                LoraConfig(r=64, lora_alpha=128, target_modules=TARGETS, bias="none", **kwargs),
            )
            return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)

        patterned = lora_params(
            rank_pattern={"in_proj_a": 16, "in_proj_b": 16},
            alpha_pattern={"in_proj_a": 32, "in_proj_b": 32},
        )
        flat = lora_params()
        # 2 gates x 2 layers x ((64-16)*2048 + (16*64 - 16*16)) = 396,288
        assert flat - patterned == 396_288


class TestSetupLoraPassthrough:
    def test_setup_lora_forwards_both_patterns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ASRModel._setup_lora must hand the config's patterns to PEFT."""
        captured: dict[str, Any] = {}

        class FakeLoraConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr("tiny_audio.asr_modeling.LoraConfig", FakeLoraConfig)
        monkeypatch.setattr("tiny_audio.asr_modeling.get_peft_model", _identity_peft)

        # `_setup_lora` only reads `config` and rebinds `self.language_model`,
        # so a namespace stands in for the half-built model without dragging in
        # nn.Module's attribute machinery.
        holder = stub(language_model=nn.Linear(2, 2))
        config = ASRConfig(
            use_lora=True,
            lora_rank=64,
            lora_alpha=128,
            lora_target_modules=["q_proj"],
            lora_rank_pattern={"in_proj_a": 16},
            lora_alpha_pattern={"in_proj_a": 32},
        )
        ASRModel._setup_lora(holder, config)

        assert captured["rank_pattern"] == {"in_proj_a": 16}
        assert captured["alpha_pattern"] == {"in_proj_a": 32}

    def test_setup_lora_tolerates_a_config_predating_the_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checkpoint configs written before these fields must still load."""
        captured: dict[str, Any] = {}

        class FakeLoraConfig:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr("tiny_audio.asr_modeling.LoraConfig", FakeLoraConfig)
        monkeypatch.setattr("tiny_audio.asr_modeling.get_peft_model", _identity_peft)

        config = ASRConfig(use_lora=True, lora_rank=64, lora_alpha=128)
        del config.lora_rank_pattern
        del config.lora_alpha_pattern

        holder = stub(language_model=nn.Linear(2, 2))
        ASRModel._setup_lora(holder, config)

        assert captured["rank_pattern"] == {}
        assert captured["alpha_pattern"] == {}
