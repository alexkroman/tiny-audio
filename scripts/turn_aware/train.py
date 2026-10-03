#!/usr/bin/env python3
"""Turn-aware LoRA training for Qwen3-ASR (config: configs/turn_aware/).

    poetry run ta turn-aware build-pool +experiment=v2
    poetry run python -m scripts.turn_aware.train +experiment=v2
    poetry run python -m scripts.turn_aware.train +experiment=v2 training.max_steps=50  # smoke

Pass the same overrides to both: training refuses a pool whose recorded
pool_config.json does not match the resolved `pool:` section.

Separate from scripts/train.py on purpose: that script builds ASRModel
(encoder -> fresh projector -> LLM). Qwen3-ASR is already an aligned
audio-LLM, and re-wiring its tower through a fresh projector would throw away
exactly the alignment this recipe is meant to keep frozen.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, set_seed

from scripts.turn_aware.config import (
    pool_signature,
    read_signature,
    signature_diff,
    training_arguments,
)
from scripts.turn_aware.data import (
    TurnAudioStore,
    TurnAwareCollator,
    TurnAwareDataset,
    marker_metrics,
    predict_rows,
    stratified_subset,
)
from scripts.turn_aware.model import (
    add_end_of_turn_token,
    apply_lora,
    cut_grad_into_frozen_audio_tower,
)
from tiny_audio.turns import load_model, register_end_of_turn, set_end_of_turn_threshold

logger = logging.getLogger(__name__)


def read_manifest(pool_dir: str, split: str) -> list[dict]:
    path = Path(to_absolute_path(pool_dir)) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with `ta turn-aware build-pool --split {split}` "
            "and the same overrides as this run"
        )
    return pd.read_parquet(path).to_dict("records")


def check_pool(cfg: DictConfig, split: str) -> None:
    """Refuse a manifest built with different pool settings than this run's.

    The pool is built by a separate command; without this, a stale
    data/turn_aware_v2 from an earlier recipe would train silently under the
    new run's name.
    """
    pool_dir = Path(to_absolute_path(cfg.data.pool_dir))
    built = read_signature(pool_dir, split)
    if built is None:
        logger.warning("%s/%s.parquet has no recorded pool settings", pool_dir, split)
        return
    diff = signature_diff(built, pool_signature(cfg, built.get("max_samples")))
    if diff:
        raise ValueError(
            f"{pool_dir}/{split}.parquet was built with different pool settings:\n  "
            + "\n  ".join(diff)
            + "\nRebuild with `ta turn-aware build-pool` and this run's overrides."
        )
    if built.get("max_samples"):
        logger.warning("%s pool is a %d-observation sample", split, built["max_samples"])


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
            ev = self.marker_eval
            preds = predict_rows(
                self.model, ev["processor"], ev["rows"], ev["store"], ev["batch_size"]
            )
            scores = marker_metrics(
                ev["rows"], [p.fired for p in preds], [p.text for p in preds], ev["normalize"]
            )
            self.model.train(was_training)
            scores = {f"eval_markers/{k}": v for k, v in scores.items() if k != "n"}
            self.log(scores)
            metrics.update(scores)
        return metrics


@hydra.main(version_base=None, config_path="../../configs/turn_aware", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.training.seed)
    if HydraConfig.get().runtime.choices.get("experiment") is None:
        logger.warning(
            "No +experiment=<v1|v2a|v2> preset: training the base recipe under the scratch "
            "identity (pool %s, hub_model_id %s).",
            cfg.data.pool_dir,
            cfg.hub_model_id,
        )
    for split in (cfg.data.train_split, cfg.data.eval_split):
        check_pool(cfg, split)
    dtype = getattr(torch, cfg.model.dtype)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.model.model_id)
    token_is_new = register_end_of_turn(processor)
    model = load_model(cfg.model.model_id, dtype=dtype)
    token_id = add_end_of_turn_token(
        model, processor, init_row=token_is_new, seed=cfg.training.seed
    )
    model = apply_lora(
        model, token_id, cfg.model.lora_rank, cfg.model.lora_alpha, cfg.model.lora_dropout
    )
    model.print_trainable_parameters()
    cut_grad_into_frozen_audio_tower(model)

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
        args=training_arguments(cfg, remove_unused_columns=False, label_names=["labels"]),
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
    report_to = cfg.training.report_to
    if trainer.is_world_process_zero() and "wandb" in (
        [report_to] if isinstance(report_to, str) else list(report_to)
    ):
        import wandb

        # Trainer logs only TrainingArguments; the pool recipe, LoRA shape and
        # threshold live elsewhere in cfg. An existing run is reused by
        # Trainer's WandbCallback.
        wandb.init(
            # Trainer's WandbCallback default; without it this run landed in
            # W&B's `uncategorized` while v1 sat in `huggingface`.
            project=os.environ.get("WANDB_PROJECT", "huggingface"),
            name=cfg.training.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.evaluate()

    # Merge the adapter and the marker row into plain weights, so the result
    # loads with Qwen3ASRForConditionalGeneration.from_pretrained and needs
    # neither PEFT nor this package.
    final_dir = Path(cfg.training.output_dir) / "final"
    merged = trainer.model.merge_and_unload()
    set_end_of_turn_threshold(merged.generation_config, token_id, cfg.model.end_of_turn_threshold)
    merged.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("merged model written to %s", final_dir.resolve())
    if cfg.hub_model_id:
        merged.push_to_hub(cfg.hub_model_id)
        processor.push_to_hub(cfg.hub_model_id)


if __name__ == "__main__":
    main()
