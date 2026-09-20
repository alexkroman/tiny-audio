"""`ta analysis compare` must report the WER that `ta eval` printed.

Both sides score the Whisper-normalized pair that `save_results` writes to
results.txt, and both get their number from `jiwer`. Nothing in between may
re-normalize, re-tokenize, or re-derive the formula.
"""

from pathlib import Path

import jiwer
import pytest

from scripts.analysis import collect_model_metrics
from scripts.eval.cli import save_results
from scripts.eval.evaluators.base import EvalResult, Evaluator


def _run(pairs: list[tuple[str, str]], tmp_path: Path, dataset: str = "earnings22") -> None:
    """Score `pairs` the way the harness does, then persist the run."""
    evaluator = Evaluator()
    normalize = evaluator.normalizer.normalize
    results = []
    for ref, hyp in pairs:
        norm_ref, norm_hyp = normalize(ref), normalize(hyp)
        results.append(
            EvalResult(
                prediction=hyp,
                reference=ref,
                wer=jiwer.wer(norm_ref, norm_hyp) * 100 if norm_ref else 0.0,
                time=0.1,
                norm_prediction=norm_hyp,
                norm_reference=norm_ref,
            )
        )
    evaluator.results = results
    metrics = evaluator.compute_metrics()
    save_results("testmodel", dataset, evaluator.results, metrics, str(tmp_path))


def _both_wers(tmp_path: Path, dataset: str = "earnings22") -> tuple[float, float]:
    collected = collect_model_metrics("testmodel", tmp_path, [])
    ds = collected["datasets"][dataset]
    return ds["wer"], ds["wer_calculated"]


@pytest.mark.parametrize(
    ("name", "pairs"),
    [
        (
            # `%` and currency: the exact case that made compare disagree.
            # A second normalizer split "25%" into "25 percent", inflating the
            # reference word count (1581 -> 1585 on the real earnings22 run).
            "percent_and_currency",
            [
                ("We grew 25% this quarter.", "We grew 20% this quarter."),
                ("Capex was £125 million.", "Capex was 125 million."),
                ("Margins hit 46% of €71 million.", "Margins hit 46% of 71 million."),
            ],
        ),
        (
            # An empty hypothesis is a 100%-deletion row; the parser used to
            # drop it, which lowered the compare-side WER.
            "empty_prediction",
            [
                ("the quick brown fox", ""),
                ("jumps over the lazy dog", "jumps over the lazy dog"),
            ],
        ),
        (
            "plain",
            [("hello world", "hello word"), ("good morning", "good morning")],
        ),
    ],
)
def test_compare_reproduces_the_harness_wer(name, pairs, tmp_path):
    _run(pairs, tmp_path)
    harness, compared = _both_wers(tmp_path)
    msg = f"{name}: ta eval reported {harness:.4f}%, compare reported {compared:.4f}%"
    assert compared == pytest.approx(harness, abs=5e-4), msg


def test_compare_wer_is_jiwers_own_number(tmp_path):
    """Not a re-derivation of (S+D+I)/(H+S+D) that could drift from jiwer."""
    pairs = [
        ("the board approved 25% of the buyback", "the board approved 20% of a buyback"),
        ("revenue rose sharply", ""),
        ("thank you", "thank you very much"),
    ]
    _run(pairs, tmp_path)

    normalizer = Evaluator().normalizer
    refs = [normalizer.normalize(r) for r, _ in pairs]
    hyps = [normalizer.normalize(h) for _, h in pairs]
    expected = jiwer.wer([r for r in refs if r], [h for r, h in zip(refs, hyps) if r]) * 100

    harness, compared = _both_wers(tmp_path)
    # Both are read back from metrics.txt / results.txt, which carry 4 decimals.
    assert harness == pytest.approx(expected, abs=5e-4)
    assert compared == pytest.approx(expected, abs=5e-4)
