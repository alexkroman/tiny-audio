"""Tests for scripts.turn_aware.transcripts: local JSONL cache and Hub mirror (Hub stubbed)."""

from __future__ import annotations

import json

import datasets

from scripts.turn_aware import transcripts

MODEL = "Qwen/Qwen3-ASR-0.6B-hf"


def _write(path, rows, tail=""):
    path.write_text("".join(json.dumps({"key": k, "text": t}) + "\n" for k, t in rows) + tail)


def test_read_cache_skips_cut_off_line_and_reterminates(tmp_path):
    cache = tmp_path / "transcripts-train.jsonl"
    _write(cache, [("a", "hello")], tail='{"key": "b", "te')
    assert transcripts.read_cache(cache) == {"a": "hello"}
    assert cache.read_text().endswith("\n")
    assert transcripts.read_cache(tmp_path / "missing.jsonl") == {}


def test_merge_appends_only_missing_keys(tmp_path):
    cache = tmp_path / "transcripts-train.jsonl"
    _write(cache, [("a", "hello")])
    assert transcripts.merge_into_cache(cache, {"a": "CHANGED", "b": "world"}) == 1
    assert transcripts.read_cache(cache) == {"a": "hello", "b": "world"}  # local copy wins


def test_pull_keeps_only_the_configured_models_rows(tmp_path, monkeypatch):
    hub = datasets.Dataset.from_dict(
        {"key": ["a", "b"], "text": ["hi", "yo"], "target_model": [MODEL, "other/model"]}
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, split: hub)
    cache = tmp_path / "transcripts-train.jsonl"
    assert transcripts.pull("org/repo", "train", cache, MODEL) == 1
    assert transcripts.read_cache(cache) == {"a": "hi"}


def test_pull_from_a_repo_that_does_not_exist_yet_is_a_no_op(tmp_path, monkeypatch):
    def missing(repo, split):
        raise FileNotFoundError(repo)

    monkeypatch.setattr(datasets, "load_dataset", missing)
    assert transcripts.pull("org/repo", "train", tmp_path / "t.jsonl", MODEL) == 0


def test_push_uploads_every_cached_row_with_its_model(tmp_path, monkeypatch):
    pushed = {}

    def fake_push(self, repo_id, split, private):
        pushed.update(repo=repo_id, split=split, private=private, rows=self.to_dict())

    monkeypatch.setattr(datasets.Dataset, "push_to_hub", fake_push)
    cache = tmp_path / "transcripts-validation.jsonl"
    _write(cache, [("a", "hello"), ("b", "world")])
    assert transcripts.push("org/repo", "validation", cache, MODEL) == 2
    assert pushed["private"] is True
    assert pushed["split"] == "validation"
    assert pushed["rows"] == {
        "key": ["a", "b"],
        "text": ["hello", "world"],
        "target_model": [MODEL] * 2,
    }
