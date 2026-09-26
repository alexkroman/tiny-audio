#!/usr/bin/env python3
"""Turn-aware LoRA training for Qwen3-ASR (config: configs/turn_aware.yaml).

    poetry run ta turn-aware build-pool --split train --split validation
    poetry run python scripts/turn_aware/train.py
    poetry run python scripts/turn_aware/train.py training.max_steps=50   # smoke

Separate from scripts/train.py on purpose: that script builds ASRModel
(encoder -> fresh projector -> LLM). Qwen3-ASR is already an aligned
audio-LLM, and re-wiring its tower through a fresh projector would throw away
exactly the alignment this recipe is meant to keep frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments, set_seed

from scripts.turn_aware.data import (
    TurnAudioStore,
    TurnAwareCollator,
    TurnAwareDataset,
    assemble_audio,
    marker_metrics,
    stratified_subset,
)
from scripts.turn_aware.model import (
    add_end_of_turn_token,
    apply_lora,
    decode_batch,
    load_model,
    load_processor,
)

logger = logging.getLogger(__name__)


def read_manifest(pool_dir: str, split: str) -> list[dict]:
    path = Path(to_absolute_path(pool_dir)) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with `ta turn-aware build-pool --split {split}`"
        )
    return pd.read_parquet(path).to_dict("records")


def predict_rows(model, processor, rows, store, batch_size: int, progress: bool = False):
    """Greedy-decode manifest rows; return (transcript, fired, marker_margin) per row."""
    starts = range(0, len(rows), batch_size)
    if progress:
        from rich.progress import track

        starts = track(starts, description="decoding")
    preds: list[tuple[str, bool, float]] = []
    for i in starts:
        chunk = rows[i : i + batch_size]
        audios = [assemble_audio(store.get(r["turn_key"]), r) for r in chunk]
        preds += decode_batch(
            model, processor, audios, [r["ctx"] for r in chunk], return_margin=True
        )
    return preds


def evaluate_markers(model, processor, rows, store, batch_size: int, normalize=None) -> dict:
    """Greedy-decode `rows` and score them with `marker_metrics`."""
    preds = predict_rows(model, processor, rows, store, batch_size)
    return marker_metrics(rows, [(t, f) for t, f, _ in preds], normalize=normalize)


class TurnAwareTrainer(Trainer):
    """Trainer whose evaluate() also decodes a balanced subset and scores markers.

    eval/loss alone cannot tell a model that fires correctly from one that
    has learned the marker's average frequency; the decode can.
    """

    def __init__(self, *args, marker_eval: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.marker_eval = marker_eval

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.marker_eval and self.is_world_process_zero():
            was_training = self.model.training
            self.model.eval()
            scores = evaluate_markers(self.model, **self.marker_eval)
            self.model.train(was_training)
            scores = {f"eval_markers/{k}": v for k, v in scores.items() if k != "n"}
            self.log(scores)
            metrics.update(scores)
        return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="turn_aware")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.training.seed)
    dtype = getattr(torch, cfg.model.dtype)

    processor = load_processor(cfg.model.model_id)
    model = load_model(cfg.model.model_id, dtype=dtype)
    token_id = add_end_of_turn_token(model, processor, seed=cfg.training.seed)
    model = apply_lora(
        model, token_id, cfg.model.lora_rank, cfg.model.lora_alpha, cfg.model.lora_dropout
    )
    model.print_trainable_parameters()

    train_rows = read_manifest(cfg.data.pool_dir, cfg.data.train_split)
    eval_rows = stratified_subset(
        read_manifest(cfg.data.pool_dir, cfg.data.eval_split), cfg.data.eval_samples
    )
    train_store = TurnAudioStore(cfg.data.train_split, cfg.data.dataset_id)
    eval_store = TurnAudioStore(cfg.data.eval_split, cfg.data.dataset_id)
    logger.info("pool: %d train rows, %d eval rows", len(train_rows), len(eval_rows))

    from scripts.eval.audio import TextNormalizer

    marker_rows = stratified_subset(eval_rows, cfg.data.marker_eval_samples, seed=1)
    trainer = TurnAwareTrainer(
        model=model,
        args=TrainingArguments(
            **OmegaConf.to_container(cfg.training, resolve=True),
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        train_dataset=TurnAwareDataset(train_rows, train_store),
        eval_dataset=TurnAwareDataset(eval_rows, eval_store),
        data_collator=TurnAwareCollator(processor),
        marker_eval={
            "processor": processor,
            "rows": marker_rows,
            "store": eval_store,
            "batch_size": cfg.training.per_device_eval_batch_size,
            "normalize": TextNormalizer().normalize,
        },
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.evaluate()

    # Merge the adapter and the marker row into plain weights, so the result
    # loads with Qwen3ASRForConditionalGeneration.from_pretrained and needs
    # neither PEFT nor this package.
    final_dir = Path(cfg.training.output_dir) / "final"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("merged model written to %s", final_dir.resolve())
    if cfg.hub_model_id:
        merged.push_to_hub(cfg.hub_model_id)
        processor.push_to_hub(cfg.hub_model_id)


if __name__ == "__main__":
    main()
