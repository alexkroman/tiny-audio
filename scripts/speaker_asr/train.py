#!/usr/bin/env python3
"""Speaker-attributed LoRA training for Qwen3-ASR (config: configs/speaker_asr/).

    poetry run ta speaker-asr build-pool +experiment=v1
    poetry run python -m scripts.speaker_asr.train +experiment=v1
    poetry run python -m scripts.speaker_asr.train +experiment=v1 training.max_steps=50 \\
        hub_model_id=null                                                     # smoke

Pass the same overrides to both: training refuses a pool whose recorded
pool_config.json does not match the resolved `pool:` and `data:` source.
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
from transformers import Trainer, TrainingArguments, set_seed

from scripts.speaker_asr.config import pool_signature
from scripts.speaker_asr.data import SpeakerASRDataset, UtteranceStore, speaker_metrics, subset
from scripts.speaker_asr.model import (
    apply_lora,
    init_speaker_rows,
    predict_rows,
    register_speaker_tokens,
    speaker_token_ids,
)
from scripts.turn_aware.config import read_signature, signature_diff

logger = logging.getLogger(__name__)


def read_manifest(pool_dir: str, split: str) -> list[dict]:
    path = Path(to_absolute_path(pool_dir)) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with `ta speaker-asr build-pool --split {split}` "
            "and the same overrides as this run"
        )
    return pd.read_parquet(path).to_dict("records")


def check_pool(cfg: DictConfig, split: str) -> None:
    """Refuse a manifest built with different settings than this run's."""
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
            + "\nRebuild with `ta speaker-asr build-pool` and this run's overrides."
        )
    if built.get("max_samples"):
        logger.warning("%s pool is a %d-utterance sample", split, built["max_samples"])


class SpeakerASRTrainer(Trainer):
    """Trainer whose evaluate() also decodes a subset and scores cpWER.

    eval/loss is dominated by transcript tokens the base model already gets
    right; whether the speaker tokens land on the right voice only shows up
    in a free-running decode.
    """

    def __init__(self, *args, decode_eval: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_eval = decode_eval

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if self.decode_eval and self.is_world_process_zero():
            was_training = self.model.training
            self.model.eval()
            ev = self.decode_eval
            preds = predict_rows(
                self.model,
                ev["processor"],
                ev["dataset"],
                ev["n_speakers"],
                ev["batch_size"],
                ev["max_new_tokens"],
            )
            refs = [r["target"] for r in ev["dataset"].rows]
            scores = speaker_metrics(refs, preds, ev["normalize"])
            self.model.train(was_training)
            scores = {f"eval_speakers/{k}": v for k, v in scores.items() if k != "n"}
            self.log(scores)
            metrics.update(scores)
        return metrics


@hydra.main(version_base=None, config_path="../../configs/speaker_asr", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.training.seed)
    if HydraConfig.get().runtime.choices.get("experiment") is None:
        logger.warning(
            "No +experiment=<v1|tower> preset: training under the scratch identity "
            "(pool %s, hub_model_id %s).",
            cfg.data.pool_dir,
            cfg.hub_model_id,
        )
    for split in (cfg.data.train_split, cfg.data.eval_split):
        check_pool(cfg, split)

    from transformers import AutoProcessor

    from scripts.turn_aware.data import TurnAwareCollator
    from tiny_audio.turns import load_model

    n_speakers = cfg.pool.max_speakers
    processor = AutoProcessor.from_pretrained(cfg.model.model_id)
    tokens_are_new = register_speaker_tokens(processor, n_speakers)
    token_ids = speaker_token_ids(processor, n_speakers)
    model = load_model(cfg.model.model_id, dtype=getattr(torch, cfg.model.dtype))
    if tokens_are_new:  # continuing from a speaker-ASR checkpoint keeps its trained rows
        init_speaker_rows(model, token_ids, seed=cfg.training.seed)
    model = apply_lora(
        model,
        token_ids,
        cfg.model.lora_target_modules,
        cfg.model.lora_rank,
        cfg.model.lora_alpha,
        cfg.model.lora_dropout,
    )
    model.print_trainable_parameters()

    columns = OmegaConf.to_container(cfg.data.columns)
    stores = {
        split: UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
        for split in (cfg.data.train_split, cfg.data.eval_split)
    }
    train_rows = read_manifest(cfg.data.pool_dir, cfg.data.train_split)
    eval_rows = subset(read_manifest(cfg.data.pool_dir, cfg.data.eval_split), cfg.data.eval_samples)
    logger.info("pool: %d train windows, %d eval windows", len(train_rows), len(eval_rows))

    from scripts.eval.audio import TextNormalizer

    eval_store = stores[cfg.data.eval_split]
    trainer = SpeakerASRTrainer(
        model=model,
        args=TrainingArguments(
            **OmegaConf.to_container(cfg.training, resolve=True),
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        train_dataset=SpeakerASRDataset(train_rows, stores[cfg.data.train_split]),
        eval_dataset=SpeakerASRDataset(eval_rows, eval_store),
        # Same chat layout and transcript-only labels as turn-aware; the
        # target simply carries speaker tokens instead of an end-of-turn one.
        data_collator=TurnAwareCollator(processor),
        decode_eval={
            "processor": processor,
            "dataset": SpeakerASRDataset(
                subset(eval_rows, cfg.data.decode_eval_samples, seed=1), eval_store
            ),
            "n_speakers": n_speakers,
            "batch_size": cfg.training.per_device_eval_batch_size,
            "max_new_tokens": cfg.model.max_new_tokens,
            "normalize": TextNormalizer().normalize,
        },
    )
    report_to = cfg.training.report_to
    if trainer.is_world_process_zero() and "wandb" in (
        [report_to] if isinstance(report_to, str) else list(report_to)
    ):
        import wandb

        # Trainer logs only TrainingArguments; the pool recipe and LoRA shape
        # live elsewhere in cfg. Trainer's WandbCallback reuses this run.
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "huggingface"),
            name=cfg.training.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.evaluate()

    # Merge the adapter and speaker rows into plain weights: the result loads
    # with Qwen3ASRForConditionalGeneration.from_pretrained, no PEFT needed.
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
