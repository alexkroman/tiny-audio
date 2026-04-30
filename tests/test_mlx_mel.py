"""Smoke + shape tests for compute_mel_unpadded."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("audio_seconds", [1.0, 3.5, 7.0])
def test_compute_mel_unpadded_shape_and_length(audio_seconds):
    from tiny_audio.mlx.encoder import compute_mel_unpadded

    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(16000 * audio_seconds)).astype(np.float32)

    mel, mel_len = compute_mel_unpadded(audio)

    # Shape: [1, 128, T_mel]
    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 128
    # Audio is hop_length=160 → ~100 mel frames per second
    expected_frames = int(audio_seconds * 100)
    # Allow small variance from edge effects (typically ±1 frame)
    assert abs(mel_len - expected_frames) <= 2, (
        f"audio_seconds={audio_seconds}, mel_len={mel_len}, expected~{expected_frames}"
    )
    # The unpadded length should not exceed mel.shape[2]
    assert mel_len <= mel.shape[2]


def test_compute_mel_matches_pt_feature_extractor():
    """The MLX wrapper must produce the same mel as the PT pipeline (it uses the same extractor)."""
    from transformers import AutoFeatureExtractor

    from tiny_audio.mlx.encoder import compute_mel_unpadded

    rng = np.random.default_rng(1)
    audio = rng.standard_normal(16000 * 3).astype(np.float32)

    fe = AutoFeatureExtractor.from_pretrained("zai-org/GLM-ASR-Nano-2512")
    fe.padding = False
    pt_out = fe(audio, sampling_rate=16000, return_attention_mask=True, return_tensors="np")
    pt_mel = pt_out["input_features"]
    pt_len = int(pt_out["attention_mask"].sum())

    mx_mel, mx_len = compute_mel_unpadded(audio)

    np.testing.assert_array_equal(np.array(mx_mel), pt_mel)
    assert mx_len == pt_len


def test_compute_mel_rejects_2d_audio():
    from tiny_audio.mlx.encoder import compute_mel_unpadded

    audio_2d = np.zeros((2, 16000), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1D"):
        compute_mel_unpadded(audio_2d)
