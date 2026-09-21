"""Tests for the drift comparator (scripts/debug/compare_to_base.py).

These cover the encoder path specifically. `compare_to_base` was decoder-only
in practice: every `audio_tower.*` tensor fell through its candidate guard, so
it printed "Matched tensors: 0" instead of failing, and the encoder-drift check
that `granite_qwen_full` prescribes had no instrument behind it.

Making the encoder comparable pulls BatchNorm buffers into the diff, and those
break two assumptions the decoder path never exercised: `running_mean` ships as
exact zeros (a zero-norm base), and `num_batches_tracked` is an integer counter
rather than a weight.
"""

import math
from typing import ClassVar

import torch
from safetensors.torch import save_file

import scripts.debug.compare_to_base as ctb
from scripts.debug.compare_to_base import classify_component, compare_tensors


class TestCompareTensorsHandlesZeroNormBases:
    """BatchNorm's `running_mean` is all zeros in every pretrained checkpoint."""

    def test_unchanged_zero_tensor_is_zero_drift_not_nan(self):
        z = torch.zeros(4)
        stats = compare_tensors(z, z)
        assert stats["rel_change"] == 0.0

    def test_unchanged_zero_tensor_is_maximally_similar(self):
        """cosine_similarity against a zero vector returns 0 -- the wrong story."""
        z = torch.zeros(4)
        assert compare_tensors(z, z)["cosine"] == 1.0

    def test_moving_off_a_zero_base_is_unbounded_not_nan(self):
        """A pinned buffer that moved should be loud, and inf is loud."""
        stats = compare_tensors(torch.ones(4), torch.zeros(4))
        assert math.isinf(stats["rel_change"])

    def test_ordinary_tensors_are_unaffected(self):
        base = torch.full((4,), 2.0)
        stats = compare_tensors(base + 1.0, base)
        assert stats["rel_change"] == 0.5


class TestWeightedIgnoresNonFinite:
    """One inf must not take the whole report with it."""

    def test_infinite_entry_is_excluded(self):
        rows = [
            {"rel_change": 0.2, "numel": 10},
            {"rel_change": float("inf"), "numel": 10},
        ]
        assert ctb._weighted(rows, "rel_change") == 0.2

    def test_parameter_weighting_still_applies(self):
        rows = [
            {"rel_change": 0.0, "numel": 30},
            {"rel_change": 1.0, "numel": 10},
        ]
        assert ctb._weighted(rows, "rel_change") == 0.25

    def test_all_non_finite_yields_nan_rather_than_dividing_by_zero(self):
        rows = [{"rel_change": float("inf"), "numel": 10}]
        assert math.isnan(ctb._weighted(rows, "rel_change"))


class TestEncoderComponentClassification:
    """A conformer block has parts a transformer decoder does not."""

    def test_conformer_parts_are_labelled(self):
        cases = {
            "encoder.layers.0.conv.depthwise_conv.weight": "enc.conv.depthwise",
            "encoder.layers.0.conv.pointwise_lin1.weight": "enc.conv.pointwise",
            "encoder.layers.0.conv.norm.running_mean": "enc.conv.batchnorm",
            "encoder.layers.0.feed_forward1.linear1.weight": "enc.feed_forward",
            "encoder.layers.0.norm_conv.weight": "enc.norms",
            "encoder.layers.0.self_attn.rel_pos_emb.weight": "enc.self_attn.rel_pos_emb",
            "encoder.input_linear.weight": "enc.input_linear",
            "encoder.out.weight": "enc.out",
            "encoder.out_mid.bias": "enc.out",
        }
        for key, want in cases.items():
            assert classify_component(key) == want, key

    def test_shared_attention_labels_cover_both_towers(self):
        """q/k/v/o are the same mechanism in either stack, so same label."""
        assert classify_component("encoder.layers.0.self_attn.q_proj.weight") == "self_attn.q_proj"

    def test_nothing_encoder_shaped_falls_through_to_other(self):
        assert classify_component("encoder.layers.0.conv.depthwise_conv.weight") != "other"


def _write_checkpoint(directory, weights):
    directory.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(directory / "model.safetensors"))
    return directory


class TestEncoderDriftEndToEnd:
    """The whole point: an encoder checkpoint against its audio base."""

    BASE: ClassVar[dict] = {
        "encoder.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 0.5),
        "encoder.layers.0.conv.norm.weight": torch.ones(4),
        "encoder.layers.0.conv.norm.running_mean": torch.zeros(4),
        "encoder.layers.0.conv.norm.running_var": torch.ones(4),
        "encoder.layers.0.conv.norm.num_batches_tracked": torch.tensor(11),
    }

    def _run(self, tmp_path, monkeypatch, *, pin_batchnorm):
        monkeypatch.setattr(ctb, "load_base_weights", lambda _: dict(self.BASE))
        trained = {}
        for key, value in self.BASE.items():
            flat = "audio_tower." + key.removeprefix("encoder.")
            pinned = pin_batchnorm and "running_" in key
            trained[flat] = (
                value.clone() if pinned or not value.is_floating_point() else value + 0.1
            )
        _write_checkpoint(tmp_path / "ck", trained)
        return ctb.compare_to_base(str(tmp_path / "ck"), "fake/base", top_k=3)

    def test_encoder_tensors_actually_match(self, tmp_path, monkeypatch, capsys):
        assert self._run(tmp_path, monkeypatch, pin_batchnorm=True) is True
        out = capsys.readouterr().out
        assert "Matched tensors:        4" in out

    def test_integer_buffers_are_skipped(self, tmp_path, monkeypatch, capsys):
        """num_batches_tracked is a counter; a norm ratio on it is meaningless."""
        self._run(tmp_path, monkeypatch, pin_batchnorm=True)
        out = capsys.readouterr().out
        assert "num_batches_tracked" not in out

    def test_pinned_batchnorm_reports_no_unbounded_drift(self, tmp_path, monkeypatch, capsys):
        self._run(tmp_path, monkeypatch, pin_batchnorm=True)
        assert "zero-norm base" not in capsys.readouterr().out

    def test_unpinned_batchnorm_is_surfaced(self, tmp_path, monkeypatch, capsys):
        """running_mean leaving its zero base means the pin is not holding."""
        self._run(tmp_path, monkeypatch, pin_batchnorm=False)
        out = capsys.readouterr().out
        assert "zero-norm base" in out
        assert "running_mean" in out

    def test_aggregate_survives_an_unbounded_tensor(self, tmp_path, monkeypatch, capsys):
        self._run(tmp_path, monkeypatch, pin_batchnorm=False)
        out = capsys.readouterr().out
        assert "Weighted relative drift:     nan" not in out

    def test_local_path_is_accepted(self, tmp_path, monkeypatch):
        """The CLI always advertised local paths; it used to call hf_hub_download."""
        assert self._run(tmp_path, monkeypatch, pin_batchnorm=True) is True

    def test_missing_local_checkpoint_fails_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ctb, "load_base_weights", lambda _: dict(self.BASE))
        (tmp_path / "empty").mkdir()
        assert ctb.compare_to_base(str(tmp_path / "empty"), "fake/base") is False


class TestSegmentCosine:
    """Drift magnitude alone cannot separate diffusion from adaptation."""

    BASE: ClassVar[dict] = {"encoder.w": torch.zeros(64)}

    def _run(self, tmp_path, monkeypatch, second_step):
        monkeypatch.setattr(ctb, "load_base_weights", lambda _: dict(self.BASE))
        torch.manual_seed(0)
        first = torch.randn(64)
        _write_checkpoint(tmp_path / "a", {"audio_tower.w": first})
        _write_checkpoint(tmp_path / "b", {"audio_tower.w": first + second_step})
        return ctb.compare_to_base(
            str(tmp_path / "b"), "fake/base", previous_id=str(tmp_path / "a")
        )

    def test_aligned_segments_are_reported_as_consistent_travel(
        self, tmp_path, monkeypatch, capsys
    ):
        torch.manual_seed(0)
        same_direction = torch.randn(64)
        self._run(tmp_path, monkeypatch, same_direction)
        assert "strongly aligned" in capsys.readouterr().out

    def test_orthogonal_segments_are_called_diffusion(self, tmp_path, monkeypatch, capsys):
        """The failure mode that put the encoder LR 2.4x too low."""
        torch.manual_seed(0)
        first = torch.randn(64)
        orthogonal = torch.randn(64)
        orthogonal -= first * (orthogonal @ first) / (first @ first)
        _write_checkpoint(tmp_path / "a", {"audio_tower.w": first})
        _write_checkpoint(tmp_path / "b", {"audio_tower.w": first + orthogonal})
        monkeypatch.setattr(ctb, "load_base_weights", lambda _: dict(self.BASE))
        ctb.compare_to_base(str(tmp_path / "b"), "fake/base", previous_id=str(tmp_path / "a"))
        assert "diffusion" in capsys.readouterr().out

    def test_no_previous_checkpoint_means_no_segment_section(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ctb, "load_base_weights", lambda _: dict(self.BASE))
        _write_checkpoint(tmp_path / "b", {"audio_tower.w": torch.ones(64)})
        ctb.compare_to_base(str(tmp_path / "b"), "fake/base")
        assert "SEGMENT COSINE" not in capsys.readouterr().out
