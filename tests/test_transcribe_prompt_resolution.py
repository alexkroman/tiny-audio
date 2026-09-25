"""Tests for resolving the inference prompt a checkpoint is saved with."""

from scripts.train import (
    TRANSCRIBE_PROMPT_PUNCT,
    _resolve_transcribe_prompt,
)


class TestResolveTranscribePrompt:
    def test_unset_with_punct_source_resolves_to_punct(self):
        datasets = [
            {"path": "tedlium", "text_punct": False, "train_splits": ["train"]},
            {"path": "cv", "text_punct": True, "train_splits": ["train"]},
        ]
        assert _resolve_transcribe_prompt(None, datasets) == TRANSCRIBE_PROMPT_PUNCT

    def test_explicit_prompt_is_kept(self):
        datasets = [{"path": "cv", "text_punct": True}]
        assert _resolve_transcribe_prompt("custom", datasets) == "custom"

    def test_no_punct_source_stays_unset(self):
        datasets = [{"path": "ami", "text_punct": False}, {"path": "undeclared"}]
        assert _resolve_transcribe_prompt(None, datasets) is None

    def test_eval_only_punct_source_does_not_count(self):
        datasets = [
            {"path": "ami", "text_punct": False},
            {"path": "gs-dev", "text_punct": True, "train_splits": []},
        ]
        assert _resolve_transcribe_prompt(None, datasets) is None

    def test_train_splits_default_counts_as_training(self):
        assert _resolve_transcribe_prompt(None, [{"text_punct": True}]) == TRANSCRIBE_PROMPT_PUNCT
