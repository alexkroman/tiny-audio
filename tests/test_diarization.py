"""Tests for SpectralCluster and diarization helpers (synthetic data only)."""

import numpy as np
import pytest


class TestSpectralClusterSimMat:
    """SpectralCluster.get_sim_mat builds cosine similarity."""

    def test_identical_rows_have_similarity_one(self):
        from tiny_audio.diarization import SpectralCluster

        sc = SpectralCluster()
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        sim = sc.get_sim_mat(embeddings)
        assert sim.shape == (2, 2)
        np.testing.assert_allclose(sim, np.ones((2, 2)), atol=1e-6)

    def test_orthogonal_rows_have_zero_similarity(self):
        from tiny_audio.diarization import SpectralCluster

        sc = SpectralCluster()
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = sc.get_sim_mat(embeddings)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[1, 1] == pytest.approx(1.0)
        assert sim[0, 1] == pytest.approx(0.0)


class TestSpectralClusterPruning:
    """p_pruning keeps only the top-k entries per row."""

    def test_pruning_zeroes_low_values(self):
        from tiny_audio.diarization import SpectralCluster

        # Use n=20 so that 6.0/n=0.3 < pval=0.5, ensuring pval dominates.
        # k_keep = max(1, int(0.5 * 20)) = 10 per row.
        n = 20
        pval = 0.5
        sc = SpectralCluster(pval=pval)

        rng = np.random.default_rng(7)
        # Build a matrix where the top-10 per row are clearly distinguishable
        affinity = np.zeros((n, n))
        for i in range(n):
            vals = rng.uniform(0.0, 0.4, size=n)
            top_idx = rng.choice(n, size=10, replace=False)
            vals[top_idx] = rng.uniform(0.7, 1.0, size=10)
            affinity[i] = vals

        pruned = sc.p_pruning(affinity.copy())
        expected_k = max(1, int(max(pval, 6.0 / n) * n))  # 10

        # Each row must have exactly expected_k non-zero entries
        for row in pruned:
            non_zero_count = int(np.sum(row > 0))
            assert non_zero_count == expected_k


class TestSpectralClusterEnd2End:
    """SpectralCluster.__call__ returns valid cluster labels for synthetic data."""

    def test_two_distinct_clusters(self):
        from tiny_audio.diarization import SpectralCluster

        # Two well-separated clusters of 5 points each in 4D
        rng = np.random.default_rng(42)
        cluster_a = rng.normal(loc=[1.0, 1.0, 0.0, 0.0], scale=0.05, size=(5, 4))
        cluster_b = rng.normal(loc=[0.0, 0.0, 1.0, 1.0], scale=0.05, size=(5, 4))
        embeddings = np.vstack([cluster_a, cluster_b])

        sc = SpectralCluster(min_num_spks=1, max_num_spks=4)
        labels = sc(embeddings, oracle_num=2)

        assert labels.shape == (10,)
        # Should split into two distinct labels
        unique_labels = set(labels.tolist())
        assert len(unique_labels) == 2
        # First 5 should share a label, last 5 should share a label
        assert len(set(labels[:5].tolist())) == 1
        assert len(set(labels[5:].tolist())) == 1
        # And the two groups should differ
        assert labels[0] != labels[5]

    def test_oracle_num_one_returns_single_cluster(self):
        from tiny_audio.diarization import SpectralCluster

        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(8, 4))
        sc = SpectralCluster()
        labels = sc(embeddings, oracle_num=1)
        # All same label
        assert len(set(labels.tolist())) == 1


class TestDeviceHelper:
    """_get_device returns a torch.device."""

    def test_returns_device(self):
        import torch

        from tiny_audio.diarization import _get_device

        device = _get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")


class TestVadFraming:
    """_get_speech_segments must process every complete VAD frame."""

    HOP_SIZE = 256

    def _run(self, mocker, num_samples: int):
        """Run the VAD loop over `num_samples` of always-speech audio."""
        from tiny_audio.diarization import SpeakerDiarizer

        vad = mocker.MagicMock()
        vad.process.return_value = (0.9, True)
        mocker.patch.object(SpeakerDiarizer, "_get_ten_vad_model", return_value=vad)

        audio = np.ones(num_samples, dtype=np.float32) * 0.5
        _segments, vad_frames = SpeakerDiarizer._get_speech_segments(audio, sample_rate=16000)
        return vad, vad_frames

    @pytest.mark.parametrize("num_samples", [256, 1024, 1280, 4096])
    def test_processes_every_complete_frame(self, mocker, num_samples):
        """Stopping at `len - hop` dropped the last complete frame.

        For 1024 samples that is 3 frames instead of 4, so the trailing 16ms of
        speech is truncated and a silence cut is forced at the end of the clip.
        """
        vad, vad_frames = self._run(mocker, num_samples)

        expected = num_samples // self.HOP_SIZE
        assert vad.process.call_count == expected
        assert len(vad_frames) == expected

    def test_does_not_process_a_partial_trailing_frame(self, mocker):
        """A trailing remainder shorter than one hop is still left out."""
        vad, _ = self._run(mocker, 1023)

        assert vad.process.call_count == 1023 // self.HOP_SIZE

    def test_audio_shorter_than_one_frame_yields_nothing(self, mocker):
        vad, vad_frames = self._run(mocker, 255)

        assert vad.process.call_count == 0
        assert vad_frames == []
