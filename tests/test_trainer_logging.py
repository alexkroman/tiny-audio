"""ASRTrainer.log attaches the projector's output-scale diagnostic.

`projector/output_rms_over_embed` is the one custom training metric this repo
keeps, and it only earns that place if it is readable against the loss. Its
previous implementation logged from inside a `_clip_grad_norm` override, which
emitted a wandb step of its own: every diagnostic row had NaN for loss and vice
versa, so the two could not be plotted or correlated. These tests pin the
metric to the training-loss log entry and keep it off eval entries.
"""

import pytest


@pytest.fixture
def trainer(base_asr_model, tmp_path):
    from transformers import TrainingArguments

    from scripts.train import ASRTrainer

    args = TrainingArguments(output_dir=str(tmp_path), report_to=[])
    return ASRTrainer(model=base_asr_model, args=args)


class TestProjectorRmsLogging:
    def test_rides_the_training_loss_entry(self, trainer):
        trainer.log({"loss": 0.5})
        entry = trainer.state.log_history[-1]
        # Fresh projector calibrated by `projector_output_rms: auto`, so the
        # ratio it is measuring starts at ~1.0 by construction.
        assert entry["projector/output_rms_over_embed"] == pytest.approx(1.0, rel=0.2)

    def test_absent_from_eval_entries(self, trainer):
        trainer.log({"eval_loss": 0.4})
        assert "projector/output_rms_over_embed" not in trainer.state.log_history[-1]

    def test_restores_projector_training_mode(self, trainer):
        trainer.model.projector.train()
        trainer.log({"loss": 0.5})
        assert trainer.model.projector.training

    def test_survives_a_projector_without_the_probe(self, trainer, monkeypatch):
        """Diagnostics must never take a run down."""
        monkeypatch.delattr(type(trainer.model.projector), "measure_output_rms")
        trainer.log({"loss": 0.5})
        assert trainer.state.log_history[-1]["loss"] == 0.5
