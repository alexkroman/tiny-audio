"""Resuming from a checkpoint restores the LoRA adapter, not just the projector.

HF Trainer's `_load_from_checkpoint` loads model.safetensors (the projector)
and, when that file exists, never reaches its PEFT branch. A resumed run
therefore restarted LoRA from its initialization under the trained optimizer
state (granite-qwen-verbatim-aug-2 at step 52,000, 2026-10-19: loss 0.28 ->
1.7). `ASRTrainer` loads adapter_model.safetensors as well.
"""

from pathlib import Path

import pytest
import torch
from conftest import make_asr_config
from safetensors.torch import save_file
from transformers import Trainer, TrainingArguments

from scripts.train import ASRTrainer
from tiny_audio.asr_modeling import ASRModel


@pytest.fixture(scope="module")
def lora_model() -> ASRModel:
    """A LoRA model of its own: these tests overwrite its weights."""
    return ASRModel(make_asr_config(use_lora=True, lora_rank=8, lora_alpha=16))


def _lora(model: ASRModel) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def _projector(model: ASRModel) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.projector.named_parameters()}


Weights = dict[str, torch.Tensor]


def _train_save_and_reset(model: ASRModel, checkpoint: Path) -> tuple[Weights, Weights]:
    """Give the model "trained" weights, save them, then scramble them."""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.normal_(std=0.02)
    trained = (_lora(model), _projector(model))
    model.save_pretrained(checkpoint)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    return trained


def _trainer(cls: type[Trainer], model: ASRModel, tmp_path: Path) -> Trainer:
    # use_cpu: a Trainer otherwise moves the shared model to MPS/CUDA in place.
    args = TrainingArguments(output_dir=str(tmp_path / "out"), report_to=[], use_cpu=True)
    return cls(model=model, args=args)


def _matches(now: dict[str, torch.Tensor], saved: dict[str, torch.Tensor]) -> bool:
    return now.keys() == saved.keys() and all(torch.equal(now[k], saved[k]) for k in saved)


def test_resume_restores_projector_and_lora(lora_model: ASRModel, tmp_path: Path) -> None:
    lora, projector = _train_save_and_reset(lora_model, tmp_path / "checkpoint-52000")
    assert lora
    trainer = _trainer(ASRTrainer, lora_model, tmp_path)
    trainer._load_from_checkpoint(str(tmp_path / "checkpoint-52000"))
    assert _matches(_projector(lora_model), projector)
    assert _matches(_lora(lora_model), lora)


def test_stock_trainer_skips_the_adapter(lora_model: ASRModel, tmp_path: Path) -> None:
    """The bug the override exists for; if this starts failing, HF fixed it."""
    lora, projector = _train_save_and_reset(lora_model, tmp_path / "checkpoint")
    _trainer(Trainer, lora_model, tmp_path)._load_from_checkpoint(str(tmp_path / "checkpoint"))
    assert _matches(_projector(lora_model), projector)
    assert not _matches(_lora(lora_model), lora)


def test_load_lora_checkpoint_edges(lora_model: ASRModel, tmp_path: Path) -> None:
    assert lora_model.load_lora_checkpoint(tmp_path) is False  # no adapter file
    save_file({"not.a.lora.weight": torch.zeros(1)}, tmp_path / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="match no LoRA parameter"):
        lora_model.load_lora_checkpoint(tmp_path)
