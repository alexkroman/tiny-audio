"""`ta analysis compare` must read one sweep, not the newest dir per dataset.

Mixing sweeps is how the corpus WER came to compare different data between
models: one model's pool was 53% LibriSpeech by reference word (its only
n=500 runs) against 18% for the other.
"""

from pathlib import Path

import pytest

from scripts.analysis import _latest_sweep, _recompute_matched_corpus, collect_model_metrics


def _write(root: Path, ts: str, model: str, ds: str, run_id: str | None, pairs, wer: float):
    d = root / f"{ts}_{model}_{ds}"
    d.mkdir(parents=True)
    header = f"Model: {model}\nDataset: {ds}\nTimestamp: {ts}\n"
    if run_id:
        header += f"Run ID: {run_id}\n"
    (d / "metrics.txt").write_text(header + "-" * 40 + f"\nwer: {wer}\nnum_samples: {len(pairs)}\n")
    (d / "results.txt").write_text(
        "".join(
            f"Sample {i} - WER: 0.00%\nGround Truth: {r}\nPrediction: {p}\n"
            f"Ground Truth Raw: {r}\nPrediction Raw: {p}\n" + "-" * 80 + "\n"
            for i, (r, p) in enumerate(pairs, 1)
        )
    )
    return d


MATCH = [("alpha beta gamma", "alpha beta gamma")] * 5
MISS = [("alpha beta gamma", "alpha beta delta")] * 5


class TestLatestSweep:
    def test_picks_newest_run_id_and_drops_older_sweep(self, tmp_path):
        _write(tmp_path, "20260101_000000", "modelA", "ami", "oldrun", MATCH, 0.0)
        _write(tmp_path, "20260102_000000", "modelA", "ami", "newrun", MISS, 33.3)
        dirs, rid = _latest_sweep(sorted(tmp_path.iterdir()))
        assert rid == "newrun"
        assert [d.name for d in dirs] == ["20260102_000000_modelA_ami"]

    def test_directories_without_a_run_id_are_dropped(self, tmp_path):
        _write(tmp_path, "20260101_000000", "modelA", "ami", None, MATCH, 0.0)
        dirs, rid = _latest_sweep(sorted(tmp_path.iterdir()))
        assert dirs == []
        assert rid == ""

    def test_stale_dataset_does_not_leak_into_the_new_sweep(self, tmp_path):
        """The regression: `latest=True` resolves per dataset, so a dataset the
        new sweep also covered could still be served from the old one."""
        _write(tmp_path, "20260101_000000", "modelA", "ami", "oldrun", MATCH, 0.0)
        _write(tmp_path, "20260102_000000", "modelA", "ami", "newrun", MISS, 33.3)
        _write(tmp_path, "20260102_001000", "modelA", "librispeech", "newrun", MISS, 33.3)
        m = collect_model_metrics("modelA", tmp_path, [])
        assert m["sweep"] == "newrun"
        assert sorted(m["datasets"]) == ["ami", "librispeech"]
        assert m["datasets"]["ami"]["wer_calculated"] == pytest.approx(33.3, abs=0.2)


class TestMatchedCorpus:
    def test_corpus_uses_only_datasets_every_model_has(self, tmp_path):
        _write(tmp_path, "20260102_000000", "modelA", "ami", "ra", MISS, 33.3)
        _write(tmp_path, "20260102_001000", "modelA", "tedlium", "ra", MISS, 33.3)
        _write(tmp_path, "20260102_010000", "modelB", "ami", "rb", MATCH, 0.0)
        mm = {
            "a": collect_model_metrics("modelA", tmp_path, []),
            "b": collect_model_metrics("modelB", tmp_path, []),
        }
        _recompute_matched_corpus(mm)
        assert mm["a"]["corpus_datasets"] == ["ami"], "tedlium is not in modelB's sweep"
        assert mm["a"]["corpus_wer"] == pytest.approx(33.3, abs=0.2)
        assert mm["b"]["corpus_wer"] == pytest.approx(0.0, abs=0.2)

    def test_corpus_truncates_to_the_shared_row_count(self, tmp_path):
        """An n=100 run against an n=500 run must not weight the larger one more."""
        _write(tmp_path, "20260102_000000", "modelA", "ami", "ra", MISS * 4, 33.3)
        _write(tmp_path, "20260102_010000", "modelB", "ami", "rb", MATCH, 0.0)
        mm = {
            "a": collect_model_metrics("modelA", tmp_path, []),
            "b": collect_model_metrics("modelB", tmp_path, []),
        }
        _recompute_matched_corpus(mm)
        assert len(mm["a"]["datasets"]["ami"]["refs"]) == 20
        assert mm["a"]["corpus_datasets"] == ["ami"]
        # Both corpora scored over the shared first 5 rows.
        assert mm["a"]["corpus_wer"] == pytest.approx(33.3, abs=0.2)

    def test_dataset_with_different_references_is_excluded(self, tmp_path):
        """Different eval pools must not be pooled -- the pre-shuffle trap."""
        _write(tmp_path, "20260102_000000", "modelA", "ami", "ra", MISS, 33.3)
        _write(
            tmp_path,
            "20260102_010000",
            "modelB",
            "ami",
            "rb",
            [("totally different text here", "totally different text here")] * 5,
            0.0,
        )
        mm = {
            "a": collect_model_metrics("modelA", tmp_path, []),
            "b": collect_model_metrics("modelB", tmp_path, []),
        }
        _recompute_matched_corpus(mm)
        assert mm["a"]["corpus_datasets"] == []
        assert "ami" in mm["a"]["corpus_excluded"]
        assert "corpus_wer" not in mm["a"]
