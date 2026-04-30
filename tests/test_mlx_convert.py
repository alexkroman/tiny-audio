"""Tests for projector weight conversion + atomic cache markers."""

import pytest
import torch

mx = pytest.importorskip("mlx.core")


def test_convert_projector_weights_strips_prefix():
    from tiny_audio.mlx.convert import convert_projector_weights

    fake_state = {
        "projector.linear_1.weight": torch.randn(2048, 32),
        "projector.norm.weight": torch.ones(32),
        "projector.linear_2.weight": torch.randn(32, 64),
    }
    out = convert_projector_weights(fake_state)
    assert set(out.keys()) == {"linear_1.weight", "norm.weight", "linear_2.weight"}
    assert out["linear_1.weight"].shape == (2048, 32)
    assert out["linear_2.weight"].shape == (32, 64)
    # Should be fp16
    assert out["linear_1.weight"].dtype == mx.float16


def test_convert_projector_weights_skips_non_projector_keys():
    from tiny_audio.mlx.convert import convert_projector_weights

    fake_state = {
        "projector.linear_1.weight": torch.randn(8, 4),
        "language_model.layers.0.weight": torch.randn(8, 4),  # should be ignored
        "audio_tower.conv1.weight": torch.randn(8, 4),  # should be ignored
    }
    out = convert_projector_weights(fake_state)
    assert set(out.keys()) == {"linear_1.weight"}


def test_convert_projector_weights_raises_on_empty():
    from tiny_audio.mlx.convert import convert_projector_weights

    fake_state = {"language_model.weight": torch.randn(4, 4)}
    with pytest.raises(ValueError, match="No projector.* keys"):
        convert_projector_weights(fake_state)


def test_atomic_cache_marker_round_trip(tmp_path):
    from tiny_audio.mlx.convert import is_cache_valid, mark_cache_complete

    cache = tmp_path / "test_repo"
    cache.mkdir()

    # No marker
    assert not is_cache_valid(cache)

    # Write marker
    mark_cache_complete(cache)
    assert is_cache_valid(cache)


def test_cache_marker_version_invalidation(tmp_path):
    from tiny_audio.mlx.convert import is_cache_valid, mark_cache_complete

    cache = tmp_path / "test_repo"
    cache.mkdir()

    mark_cache_complete(cache, version=1)
    assert is_cache_valid(cache, expected_version=1)
    # A bumped expected version invalidates the cache
    assert not is_cache_valid(cache, expected_version=2)
    # An older version also invalidates (different version)
    assert not is_cache_valid(cache, expected_version=0)


def test_safe_repo_id_replaces_slashes():
    from tiny_audio.mlx.convert import safe_repo_id

    assert safe_repo_id("mazesmazes/tiny-audio-embedded") == "mazesmazes__tiny-audio-embedded"
    assert safe_repo_id("local-only") == "local-only"


def test_cache_marker_corrupted_json_invalidates(tmp_path):
    """If the marker file is corrupted, is_cache_valid returns False (not raises)."""
    from tiny_audio.mlx.convert import is_cache_valid

    cache = tmp_path / "test_repo"
    cache.mkdir()
    (cache / ".mlx_converted").write_text("not valid json {{{")

    assert not is_cache_valid(cache)


def test_default_cache_root_under_home(monkeypatch, tmp_path):
    from tiny_audio.mlx.convert import default_cache_root

    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_cache_root() == tmp_path / ".cache" / "tiny-audio" / "mlx"
