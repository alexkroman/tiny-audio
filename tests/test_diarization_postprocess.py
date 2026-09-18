"""Tests for the model-free parts of tiny_audio.diarization.

VAD hysteresis, vote-grid resampling, segment merging and the clustering
front-end are all plain numpy; none of these tests load TEN-VAD or ECAPA.
"""

import numpy as np
import pytest

from tiny_audio.diarization import SpeakerClusterer, SpeakerDiarizer, SpectralCluster

SR = 16000


def seg(start: float, end: float) -> dict:
    """A VAD segment dict in the shape `_get_speech_segments` emits."""
    return {
        "start": start,
        "end": end,
        "start_sample": int(start * SR),
        "end_sample": int(end * SR),
    }


class TestVadHysteresis:
    """Gap filling, minimum duration, and onset/offset padding."""

    def test_empty_input(self):
        assert SpeakerDiarizer._apply_vad_hysteresis([]) == []

    def test_short_gap_is_bridged(self):
        gap = SpeakerDiarizer.VAD_MAX_GAP / 2
        out = SpeakerDiarizer._apply_vad_hysteresis([seg(0.5, 1.0), seg(1.0 + gap, 2.0)])
        assert len(out) == 1
        assert out[0]["end"] == pytest.approx(2.0 + SpeakerDiarizer.VAD_PAD_OFFSET)

    def test_long_gap_is_kept(self):
        gap = SpeakerDiarizer.VAD_MAX_GAP * 2
        out = SpeakerDiarizer._apply_vad_hysteresis([seg(0.5, 1.0), seg(1.0 + gap, 3.0)])
        assert len(out) == 2

    def test_unsorted_input_is_sorted_before_merging(self):
        out = SpeakerDiarizer._apply_vad_hysteresis([seg(3.0, 4.0), seg(0.5, 1.0)])
        assert [round(s["start"], 2) for s in out] == [0.45, 2.95]

    def test_too_short_segment_is_dropped(self):
        too_short = SpeakerDiarizer.VAD_MIN_DURATION / 2
        out = SpeakerDiarizer._apply_vad_hysteresis([seg(0.5, 0.5 + too_short), seg(5.0, 6.0)])
        assert [s["start"] for s in out] == [pytest.approx(5.0 - SpeakerDiarizer.VAD_PAD_ONSET)]

    def test_padding_is_clamped_at_zero_and_samples_follow(self):
        out = SpeakerDiarizer._apply_vad_hysteresis([seg(0.0, 1.0)], sample_rate=SR)
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == pytest.approx(1.0 + SpeakerDiarizer.VAD_PAD_OFFSET)
        assert out[0]["start_sample"] == 0
        assert out[0]["end_sample"] == int(out[0]["end"] * SR)

    def test_input_dicts_are_not_mutated(self):
        original = seg(0.5, 1.0)
        SpeakerDiarizer._apply_vad_hysteresis([original])
        assert original == seg(0.5, 1.0)


class TestResampleVad:
    """16 ms VAD frames are mapped onto the 10 ms voting grid."""

    def test_empty_vad_means_silence_everywhere(self):
        out = SpeakerDiarizer._resample_vad([], 7)
        assert out.dtype == bool
        assert out.shape == (7,)
        assert not out.any()

    def test_boundary_lands_on_the_right_voting_frame(self):
        vad = [False] * 10 + [True] * 10  # speech starts at 10 * 16 ms = 0.16 s
        out = SpeakerDiarizer._resample_vad(vad, 32)
        assert not out[:16].any()  # 0.00 .. 0.15 s
        assert out[16:].all()  # 0.16 s onwards

    def test_voting_frames_past_the_vad_tail_reuse_the_last_decision(self):
        out = SpeakerDiarizer._resample_vad([True, True], 100)
        assert out.all()


class TestMergeShortSegments:
    """Flicker removal keeps the timeline monotonic."""

    def test_empty(self):
        assert SpeakerDiarizer._merge_short_segments([]) == []

    def test_short_same_speaker_blip_extends_previous(self):
        short = SpeakerDiarizer.MIN_SEGMENT_DURATION / 2
        segs = [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_0", "start": 1.05, "end": 1.05 + short},
        ]
        out = SpeakerDiarizer._merge_short_segments(segs)
        assert out == [{"speaker": "SPEAKER_0", "start": 0.0, "end": 1.05 + short}]

    def test_short_other_speaker_blip_is_dropped(self):
        short = SpeakerDiarizer.MIN_SEGMENT_DURATION / 2
        segs = [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_1", "start": 1.0, "end": 1.0 + short},
            {"speaker": "SPEAKER_0", "start": 1.0 + short, "end": 2.0},
        ]
        out = SpeakerDiarizer._merge_short_segments(segs)
        assert [s["speaker"] for s in out] == ["SPEAKER_0"]
        assert out[0]["end"] == 2.0

    def test_same_speaker_within_gap_merges(self):
        gap = SpeakerDiarizer.SAME_SPEAKER_GAP / 2
        segs = [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_0", "start": 1.0 + gap, "end": 2.0},
        ]
        assert SpeakerDiarizer._merge_short_segments(segs) == [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 2.0}
        ]

    def test_same_speaker_beyond_gap_stays_split(self):
        gap = SpeakerDiarizer.SAME_SPEAKER_GAP * 2
        segs = [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_0", "start": 1.0 + gap, "end": 3.0},
        ]
        assert len(SpeakerDiarizer._merge_short_segments(segs)) == 2

    def test_different_speakers_stay_split(self):
        segs = [
            {"speaker": "SPEAKER_0", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_1", "start": 1.0, "end": 2.0},
        ]
        assert SpeakerDiarizer._merge_short_segments(segs) == segs


class TestPostprocessSegments:
    """Frame-level voting turns overlapping windows into speaker turns."""

    @staticmethod
    def vad_all(duration: float, speech: bool = True) -> list[bool]:
        return [speech] * int(np.ceil(duration / (256 / 16000)))

    def test_empty_windows(self):
        assert SpeakerDiarizer._postprocess_segments([], np.array([]), 1.0, []) == []

    def test_two_clean_turns(self):
        windows = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
        out = SpeakerDiarizer._postprocess_segments(
            windows, np.array([0, 1]), 2.0, self.vad_all(2.0)
        )
        assert [s["speaker"] for s in out] == ["SPEAKER_0", "SPEAKER_1"]
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == pytest.approx(1.0)
        assert out[1]["start"] == pytest.approx(1.0)
        assert out[1]["end"] >= 2.0

    def test_labels_are_renumbered_contiguously(self):
        windows = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
        out = SpeakerDiarizer._postprocess_segments(
            windows, np.array([7, 3]), 2.0, self.vad_all(2.0)
        )
        assert {s["speaker"] for s in out} == {"SPEAKER_0", "SPEAKER_1"}

    def test_vad_silence_suppresses_speech(self):
        windows = [{"start": 0.0, "end": 1.0}]
        out = SpeakerDiarizer._postprocess_segments(
            windows, np.array([0]), 1.0, self.vad_all(1.0, speech=False)
        )
        assert out == []

    def test_uncovered_frames_are_silence(self):
        # One window over the first second of a two-second clip: the second
        # second gets no votes and must not be assigned to anyone.
        windows = [{"start": 0.0, "end": 1.0}]
        out = SpeakerDiarizer._postprocess_segments(windows, np.array([0]), 2.0, self.vad_all(2.0))
        assert len(out) == 1
        assert out[0]["end"] == pytest.approx(1.0)


class TestSpeakerClustererEdgeCases:
    """Tiny inputs bypass spectral clustering entirely."""

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="Expected 2D array"):
            SpeakerClusterer()(np.zeros(4))

    def test_no_embeddings(self):
        out = SpeakerClusterer()(np.zeros((0, 8)))
        assert out.shape == (0,)
        assert out.dtype == int

    def test_single_embedding(self):
        assert SpeakerClusterer()(np.ones((1, 8))).tolist() == [0]

    @pytest.mark.parametrize("n", [2, 5])
    def test_fewer_than_six_embeddings_are_one_speaker(self, n):
        assert SpeakerClusterer()(np.random.default_rng(0).normal(size=(n, 8))).tolist() == [0] * n

    def test_oracle_count_separates_two_clear_speakers(self):
        rng = np.random.default_rng(0)
        a = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)) + rng.normal(scale=0.01, size=(5, 4))
        b = np.tile([0.0, 1.0, 0.0, 0.0], (5, 1)) + rng.normal(scale=0.01, size=(5, 4))
        embs = np.vstack([a, b])
        embs[0, 3] = np.nan  # must be scrubbed, not propagated
        labels = SpeakerClusterer()(embs, num_speakers=2)
        assert set(labels.tolist()) == {0, 1}
        assert len(set(labels[:5].tolist())) == 1
        assert len(set(labels[5:].tolist())) == 1
        assert labels[0] != labels[5]

    def test_oracle_count_does_not_leak_into_later_calls(self):
        clusterer = SpeakerClusterer(min_num_spks=2, max_num_spks=10)
        rng = np.random.default_rng(0)
        embs = rng.normal(size=(8, 4))
        clusterer(embs, num_speakers=2)
        spectral = clusterer._get_spectral_cluster()
        assert spectral.min_num_spks == 2
        assert spectral.max_num_spks == 10


class TestMergeByCos:
    """Near-duplicate centroids collapse to one speaker."""

    def test_single_label_is_returned_unchanged(self):
        labels = np.zeros(4, dtype=int)
        out = SpeakerClusterer()._merge_by_cos(
            labels, np.random.default_rng(0).normal(size=(4, 3)), 0.9
        )
        assert out is labels

    def test_identical_centroids_merge(self):
        embs = np.tile([1.0, 0.0, 0.0], (6, 1))
        labels = np.array([0, 0, 0, 1, 1, 1])
        merged = SpeakerClusterer()._merge_by_cos(labels, embs, cos_thr=0.9)
        assert len(np.unique(merged)) == 1

    def test_orthogonal_centroids_stay_apart(self):
        embs = np.vstack([np.tile([1.0, 0.0, 0.0], (3, 1)), np.tile([0.0, 1.0, 0.0], (3, 1))])
        labels = np.array([0, 0, 0, 1, 1, 1])
        merged = SpeakerClusterer()._merge_by_cos(labels, embs, cos_thr=0.9)
        assert len(np.unique(merged)) == 2


class TestSpectralClusterInternals:
    """Laplacian, eigen-gap and oracle paths."""

    def test_eigen_gaps_are_consecutive_differences(self):
        gaps = SpectralCluster().get_eigen_gaps(np.array([0.0, 0.1, 0.5, 2.0]))
        assert gaps.tolist() == pytest.approx([0.1, 0.4, 1.5])

    def test_laplacian_zeroes_diagonal_and_has_zero_row_sums(self):
        sim = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
        lap = SpectralCluster().get_laplacian(sim)
        assert np.allclose(lap.sum(axis=1), 0.0)
        assert np.allclose(np.diag(lap), [0.7, 0.8, 0.5])

    def test_oracle_count_selects_that_many_eigenvectors(self):
        sim = np.eye(6)
        lap = SpectralCluster().get_laplacian(sim.copy())
        emb, k = SpectralCluster().get_spec_embs(lap, k_oracle=2)
        assert k == 2
        assert emb.shape == (6, 2)
