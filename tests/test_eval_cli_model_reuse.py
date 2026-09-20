"""`ta eval -d a -d b` must load the model once, not once per dataset."""

from typer.testing import CliRunner

from scripts.eval import cli as eval_cli
from scripts.eval.evaluators.base import Evaluator

runner = CliRunner()


class RecordingEvaluator(Evaluator):
    """Counts constructions and records the fields each `evaluate` ran with."""

    builds = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).builds += 1
        self.seen: list[tuple[str, str]] = []

    def evaluate(self, dataset, max_samples=None, **kwargs):
        results = super().evaluate(dataset, max_samples, **kwargs)
        self.seen.append((self.audio_field, self.text_field))
        return results

    def transcribe(self, audio):
        return str(audio), 0.01, None


def test_model_is_built_once_for_a_multi_dataset_sweep(monkeypatch, tmp_path):
    RecordingEvaluator.builds = 0
    built: list[RecordingEvaluator] = []

    def fake_build(**kwargs):
        evaluator = RecordingEvaluator()
        built.append(evaluator)
        return "fake-model", evaluator

    # loquacious reads `wav`/`text`; earnings22 reads `audio`/`sentence`.
    # The rows carry both shapes so either field set resolves.
    def fake_load(name, split, config=None):
        return [{"wav": "a", "audio": "a", "text": "hi", "sentence": "hi"}]

    monkeypatch.setattr(eval_cli, "_build_evaluator", fake_build)
    monkeypatch.setattr(eval_cli, "load_eval_dataset", fake_load)

    result = runner.invoke(
        eval_cli.app,
        ["-m", "fake", "-d", "loquacious", "-d", "earnings22", "-o", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert RecordingEvaluator.builds == 1, "evaluator was rebuilt per dataset"
    # Each dataset still scored with its own column names.
    assert built[0].seen == [("wav", "text"), ("audio", "sentence")]
    # And both datasets produced their own output directory.
    assert len(list(tmp_path.iterdir())) == 2
