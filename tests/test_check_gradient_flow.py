"""Tests for scripts/debug/check_gradient_flow.py on the tiny offline model pair."""

import pytest
import torch

from scripts.debug.check_gradient_flow import (
    grad_norm,
    load_experiment_config,
    report,
    synthetic_batch,
    training_knobs,
)
from tests.conftest import make_asr_config
from tiny_audio.asr_modeling import ASRModel


@pytest.fixture(scope="module")
def trainable_model() -> ASRModel:
    # Fresh (not the session fixture): report() runs backward and leaves .grad set.
    return ASRModel(make_asr_config(freeze_language_model=False))


def test_grad_norm_matches_flattened_l2_and_handles_no_grads() -> None:
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))  # upcast before summing
    unused = torch.nn.Parameter(torch.zeros(5))
    assert grad_norm([a, b]) == 0.0  # no .grad anywhere: empty list, not an error
    a.grad = torch.tensor([1.0, 2.0, 2.0])
    b.grad = torch.full((2, 2), 2.0, dtype=torch.bfloat16)
    assert grad_norm([a, b, unused]) == pytest.approx(5.0)  # sqrt(9 + 16)


def test_experiment_config_supplies_trainer_knobs() -> None:
    """Composed, so knobs the experiment inherits from config.yaml are present."""
    cfg = load_experiment_config("stage_1")
    assert cfg.training.attn_implementation == "eager"
    knobs = training_knobs(cfg)
    assert knobs["learning_rate"] == pytest.approx(1e-3)
    assert knobs["decoder_learning_rate"] == pytest.approx(2e-5)
    assert knobs["projector_weight_decay"] == 0.0


def test_batch_comes_from_the_training_collator(trainable_model: ASRModel) -> None:
    batch = synthetic_batch(trainable_model)
    ids = batch["input_ids"]
    assert ids.shape[0] == 2
    # The collator emits model.audio_token, one per projected audio frame.
    n_audio = (ids == trainable_model.audio_token_id).sum(dim=1)
    assert torch.equal(n_audio, batch["audio_token_counts"])
    # Prompt (audio included) is masked out of the loss; the response is not.
    assert bool((batch["labels"][ids == trainable_model.audio_token_id] == -100).all())
    assert bool((batch["labels"] != -100).any())


def test_report_passes_on_a_full_decoder_finetune(
    trainable_model: ASRModel, capsys: pytest.CaptureFixture[str]
) -> None:
    knobs = training_knobs(load_experiment_config("stage_1"))
    report(trainable_model, torch.float32, "cpu", "stage_1", knobs)
    out = capsys.readouterr().out
    assert "[FAIL]" not in out
    assert "[OK] gradient flow matches stage_1's freeze settings" in out
