"""Tests for scripts.eval.speaker_metrics and speaker-labelled scoring in `ta eval`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.eval.speaker_metrics import (
    cp_errors,
    has_speakers,
    parse_turns,
    plain_text,
    serialize_turns,
    speaker_metrics,
    word_errors,
)


def test_parse_turns_and_plain_text():
    text = "lead<SPK_1>Hi.<SPK_2> Hello. <SPK_3><SPK_1>Bye."
    assert parse_turns(text) == [(0, "lead"), (1, "Hi."), (2, "Hello."), (1, "Bye.")]
    assert plain_text(text) == "lead Hi. Hello. Bye."


def test_word_errors():
    assert word_errors(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    assert word_errors([], ["a", "b"]) == 2
    assert word_errors(["a", "b"], []) == 2


def test_cpwer_is_permutation_free():
    ref = "<SPK_1>hello there<SPK_2>good morning"
    assert cp_errors(ref, "<SPK_2>hello there<SPK_1>good morning") == (0, 4)


def test_cpwer_charges_misattributed_words_twice():
    ref = "<SPK_1>a b<SPK_2>c d"
    # 'c' given to speaker 1: a deletion from speaker 2 and an insertion on speaker 1.
    assert cp_errors(ref, "<SPK_1>a b c<SPK_2>d") == (2, 4)
    # No speaker tokens at all: everything lands on one voice, the other is all deletions.
    assert cp_errors(ref, "a b c d") == (4, 4)


def test_speaker_metrics_separates_recognition_from_attribution():
    refs = ["<SPK_1>a b<SPK_2>c d", "<SPK_1>e f"]
    hyps = ["<SPK_1>a b c d", "<SPK_1>e f"]
    m = speaker_metrics(refs, hyps)
    assert m["wer"] == 0.0
    assert m["cpwer"] == pytest.approx(4 / 6)  # speaker 2's words: 2 ins + 2 del
    assert m["attribution_gap"] == pytest.approx(4 / 6)
    assert m["speaker_count_acc"] == 0.5
    assert (m["cpwer_1spk"], m["cpwer_2spk"]) == (0.0, 1.0)


def test_serialize_turns_renumbers_by_first_appearance_and_merges():
    turns = [("B", "Hi."), ("A", "Hello."), ("A", "How are you?"), ("C", ""), ("B", "Fine.")]
    assert serialize_turns(turns) == "<SPK_1>Hi.<SPK_2>Hello. How are you?<SPK_1>Fine."
    assert has_speakers("<SPK_3>x")
    assert not has_speakers("plain text")


def test_assemblyai_utterances_become_speaker_text():
    from scripts.eval.evaluators.asr import speaker_text

    utt = SimpleNamespace
    transcript = SimpleNamespace(
        text="so yes", utterances=[utt(speaker="B", text="So"), utt(speaker="A", text="yes")]
    )
    assert speaker_text(transcript) == "<SPK_1>So<SPK_2>yes"
    assert speaker_text(SimpleNamespace(text="hi", utterances=None)) == "hi"


class _FixedEvaluator:
    """An Evaluator whose transcript per sample comes from a list."""

    @staticmethod
    def make(predictions):
        from scripts.eval.evaluators.base import Evaluator

        class Fixed(Evaluator):
            def transcribe(self, audio):
                return predictions[audio], 0.0, None

        evaluator = Fixed()
        # Whisper's normalizer downloads a spelling table; lowercasing is all
        # these references need.
        evaluator.normalizer = SimpleNamespace(normalize=str.lower)
        return evaluator


def test_eval_scores_words_without_tokens_and_adds_cpwer():
    evaluator = _FixedEvaluator.make(["<SPK_1>hello there good morning", "<SPK_1>okay"])
    dataset = [
        {"audio": 0, "text": "<SPK_1>HELLO THERE<SPK_2>GOOD MORNING"},
        {"audio": 1, "text": "<SPK_1>OKAY"},
    ]
    results = evaluator.evaluate(dataset, speakers=True)
    assert [r.wer for r in results] == [0.0, 0.0]  # words are right; tokens never count
    m = evaluator.compute_metrics()
    assert m["wer"] == 0.0
    assert m["cpwer"] == pytest.approx(100 * 4 / 5)  # 'good morning': 2 ins + 2 del
    assert m["attribution_gap"] == pytest.approx(m["cpwer"])
    assert m["speaker_count_acc"] == 0.5


def test_plain_datasets_get_no_speaker_metrics():
    evaluator = _FixedEvaluator.make(["hello"])
    evaluator.evaluate([{"audio": 0, "text": "hello"}])
    assert "cpwer" not in evaluator.compute_metrics()


def test_ami_speakers_is_registered_but_not_in_all():
    from scripts.eval.cli import ALL_DATASETS
    from scripts.eval.datasets import DATASET_REGISTRY

    assert DATASET_REGISTRY["ami-speakers"].speakers
    assert "ami-speakers" not in ALL_DATASETS
    assert not DATASET_REGISTRY["ami"].speakers
