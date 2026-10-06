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

import json
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

from scripts.speaker_asr.chunks import build_short_rows, busy_weights
from scripts.speaker_asr.config import (
    context_config,
    embedding_config,
    junction_config,
    n_speaker_tokens,
    pool_signature,
    short_chunk_config,
)
from scripts.speaker_asr.context import build_context_rows
from scripts.speaker_asr.data import (
    SpeakerASRCollator,
    SpeakerASRDataset,
    UtteranceStore,
    load_texts,
    subset,
)
from scripts.speaker_asr.embedding import (
    META_KEYS,
    LastHidden,
    MeetingBatchSampler,
    SpeakerHead,
    distill_loss,
    pool_turns,
    supcon_loss,
    turn_speakers,
)
from scripts.speaker_asr.junction import build_junction_rows, junction_count
from scripts.speaker_asr.metrics import CONTEXT_END, speaker_metrics
from scripts.speaker_asr.model import (
    apply_lora,
    init_speaker_rows,
    predict_rows,
    register_speaker_tokens,
    speaker_token_ids,
)
from scripts.turn_aware.config import read_signature, signature_diff, training_arguments
from scripts.turn_aware.model import cut_grad_into_frozen_audio_tower

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


def context_rows(cfg: DictConfig, split: str, rows: list[dict], stores: dict) -> list[dict]:
    """Pool rows -> context-prefixed rows (scripts/speaker_asr/context.py)."""
    meta = stores[split].meta
    starts = dict(zip(meta["id"], meta["start"].astype(float), strict=True))
    out = build_context_rows(
        rows, starts, load_texts(cfg, split, meta), context_config(cfg), cfg.pool.seed
    )
    with_memory = sum(r["n_memory"] > 0 for r in out)
    logger.info("%s: %d context rows (%d with voice samples)", split, len(out), with_memory)
    return out


def junction_rows(cfg: DictConfig, split: str, n_rows: int, stores: dict, seed: int) -> list[dict]:
    """`n_rows` junction windows over one split's utterances (scripts/speaker_asr/junction.py)."""
    meta = stores[split].meta
    utts = meta[["row", "id", "group", "speaker", "start", "end"]].to_dict("records")
    out = build_junction_rows(
        utts,
        load_texts(cfg, split, meta),
        junction_config(cfg),
        n_rows,
        max_speakers=cfg.pool.max_speakers,
        max_audio_s=cfg.pool.max_audio_s,
        seed=seed,
    )
    same = sum(r["n_same"] for r in out)
    new = sum(r["n_new"] for r in out)
    logger.info(
        "%s: %d junction windows (%d same-speaker / %d new-speaker junctions)",
        split, len(out), same, new,
    )  # fmt: skip
    return out


def annotate_turns(cfg: DictConfig, split: str, rows: list[dict], stores: dict) -> list[dict]:
    """Rows + `turn_speakers` / `turn_ids` (who says each target turn), for the embedding head.

    Context rows carry prefix clips in `parts`; only their live window
    (`live_ids`) makes the target's turns.
    """
    texts = load_texts(cfg, split, stores[split].meta)
    out = []
    for row in rows:
        parts = json.loads(row["parts"])
        if row.get("live_ids"):
            live = set(json.loads(row["live_ids"]))
            parts = [p for p in parts if p["id"] in live]
        speakers, ids = turn_speakers(parts, texts)
        out.append({**row, "turn_speakers": speakers, "turn_ids": ids})
    return out


def attach_ecapa(cfg: DictConfig, split: str, rows: list[dict], stores: dict) -> list[dict]:
    """Rows + `turn_ecapa`: per turn, the unit mean ECAPA-TDNN embedding of its clean headset clips.

    Utterance embeddings are cached in <pool_dir>/ecapa-<split>.npz (id -> 192-d).
    """
    import numpy as np

    store = stores[split]
    cache = Path(to_absolute_path(cfg.data.pool_dir)) / f"ecapa-{split}.npz"
    known: dict = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["ids"].tolist(), z["embs"], strict=True))
    needed = sorted({u for r in rows for ids in r["turn_ids"] for u in ids} - set(known))
    if needed:
        from tiny_audio.diarization import SpeakerDiarizer

        ecapa = SpeakerDiarizer._get_ecapa_model()
        index = dict(zip(store.meta["id"], store.meta["row"], strict=True))
        logger.info("%s: ECAPA embeddings for %d utterances", split, len(needed))
        meta = store.meta.set_index("id")
        duration = (meta["end"] - meta["start"]).to_dict()
        # length-sorted batches (little padding), audio loaded one batch at a time
        by_length = sorted(needed, key=lambda u: duration.get(u, 0.0))
        with torch.no_grad():
            for start in range(0, len(by_length), 32):
                batch = by_length[start : start + 32]
                clips = {u: store.get(index[u]) for u in batch}
                longest = max(len(clips[u]) for u in batch)
                wavs = torch.zeros(len(batch), longest)
                for i, u in enumerate(batch):
                    wavs[i, : len(clips[u])] = torch.from_numpy(clips[u]).float()
                lens = torch.tensor([len(clips[u]) / longest for u in batch])
                out = ecapa.encode_batch(wavs, wav_lens=lens).reshape(len(batch), -1).cpu().numpy()
                for u, v in zip(batch, out, strict=True):
                    known[u] = v / (np.linalg.norm(v) + 1e-9)
        ids = sorted(known)
        np.savez(cache, ids=np.array(ids), embs=np.stack([known[i] for i in ids]))
    out = []
    for r in rows:
        turns = []
        for ids in r["turn_ids"]:
            v = np.mean([known[u] for u in ids], axis=0)
            turns.append((v / (np.linalg.norm(v) + 1e-9)).tolist())
        out.append({**r, "turn_ecapa": turns})
    return out


def short_rows(cfg: DictConfig, split: str, stores: dict, seed: int) -> list[dict]:
    """Short real-timeline chunk rows for a split, cached in <pool_dir>/short-<split>-h<s>.parquet.

    The human-text threshold is in the file name: rows built with other targets are not reused.
    """
    scfg = short_chunk_config(cfg)
    tag = f"h{scfg.human_text_below_s:g}" + ("-e" if scfg.keep_empty else "")
    cache = Path(to_absolute_path(cfg.data.pool_dir)) / f"short-{split}-{tag}.parquet"
    if cache.exists():
        rows = pd.read_parquet(cache).to_dict("records")
    else:
        store = stores[split]
        texts = load_texts(cfg, split, store.meta)
        rows = build_short_rows(cfg, split, store, texts, short_chunk_config(cfg), seed)
        pd.DataFrame(rows).to_parquet(cache)
    logger.info(
        "%s: %d short real chunks (%d with >= 2 speakers, %d with no words)",
        split, len(rows), sum(int(r["n_speakers"]) >= 2 for r in rows),
        sum(not r["target"] for r in rows),
    )  # fmt: skip
    return rows


def weighted_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, speaker_ids: torch.Tensor, weight: float
) -> torch.Tensor:
    """Next-token cross-entropy with `weight` on speaker-token targets (mean over weighted targets).

    Only supervised positions are gathered before the fp32 upcast: the full
    [batch, seq, vocab] logits in fp32 run to ~10 GiB on long rehearsal rows.
    """
    import torch.nn.functional as F  # noqa: N812

    labels = labels[:, 1:].to(logits.device)
    keep = labels != -100
    picked = logits[:, :-1][keep].float()  # [n_targets, vocab]
    targets = labels[keep]
    ce = F.cross_entropy(picked, targets, reduction="none")
    w = torch.ones_like(ce)
    w[torch.isin(targets, speaker_ids.to(targets.device))] = weight
    return (ce * w).sum() / w.sum().clamp(min=1.0)


class SpeakerASRTrainer(Trainer):
    """Trainer whose evaluate() also decodes subsets and scores cpWER.

    eval/loss is dominated by transcript tokens the base model already gets
    right; whether the speaker tokens land on the right voice only shows up
    in a free-running decode. Each entry of `decode_evals` is logged under its
    own `prefix` (eval_speakers/*, eval_junction/*).
    """

    def __init__(
        self,
        *args,
        decode_evals: list[dict] | None = None,
        embed: dict | None = None,
        sample_weights: list[float] | None = None,
        speaker_loss: tuple[torch.Tensor, float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decode_evals = decode_evals or []
        # Short real chunks: busy-chunk sampling weights and (speaker token ids, loss weight).
        self.sample_weights = sample_weights
        self.speaker_loss = speaker_loss
        # Speaker-embedding head: {"head", "cfg" (EmbeddingConfig), "eval_dataset"}.
        self.embed = embed
        self.hidden = LastHidden(self.model) if embed else None
        self._embed_logs: dict[str, list[float]] = {}

    def _get_train_sampler(self, *args, **kwargs):
        if not self.embed and self.sample_weights:
            generator = torch.Generator().manual_seed(self.args.seed)
            return torch.utils.data.WeightedRandomSampler(
                self.sample_weights, len(self.sample_weights), replacement=True, generator=generator
            )
        if not self.embed:
            return super()._get_train_sampler(*args, **kwargs)
        groups = [row.get("group") for row in self.train_dataset.rows]
        return MeetingBatchSampler(
            groups,
            self.args.per_device_train_batch_size,
            self.embed["cfg"].meetings_per_batch,
            seed=self.args.seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        meta = {k: inputs.pop(k) for k in META_KEYS if k in inputs}
        if self.speaker_loss and model.training and "labels" in inputs:
            # Our weighted loss replaces the model's: call it without labels so it does not
            # also build its own full-vocab fp32 loss on the same logits.
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            ids, weight = self.speaker_loss
            loss = weighted_lm_loss(outputs.logits, labels, ids, weight)
            inputs["labels"] = labels
        else:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        if self.embed and "turn_index" in meta:
            ecfg = self.embed["cfg"]
            pooled = pool_turns(self.hidden.value, meta["turn_index"], len(meta["turn_speaker"]))
            self.hidden.value = None  # don't keep this step's graph alive into the next forward
            z, teacher_space = self.embed["head"](pooled)
            contrast = supcon_loss(z, meta["turn_speaker"], meta["turn_group"], ecfg.temperature)
            loss = loss + ecfg.weight * contrast
            self._embed_logs.setdefault("embed/supcon", []).append(float(contrast.detach()))
            if ecfg.distill_weight and teacher_space is not None and "turn_ecapa" in meta:
                distill = distill_loss(teacher_space, meta["turn_ecapa"])
                loss = loss + ecfg.distill_weight * distill
                self._embed_logs.setdefault("embed/distill", []).append(float(distill.detach()))
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._embed_logs:
            logs.update({k: sum(v) / len(v) for k, v in self._embed_logs.items()})
            self._embed_logs = {}
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.embed:
            save_head(self.embed["head"], output_dir or self.args.output_dir)

    @torch.no_grad()
    def embed_auc(self) -> float | None:
        """Same- vs different-speaker AUC over turn pairs from different windows of one meeting."""
        from sklearn.metrics import roc_auc_score

        ds = self.embed.get("eval_dataset")
        if not ds:
            return None
        was_training = self.model.training
        self.model.eval()
        vecs, keys = [], []
        bs = self.args.per_device_eval_batch_size
        for start in range(0, len(ds), bs):
            batch = self.data_collator([ds[j] for j in range(start, min(start + bs, len(ds)))])
            if "turn_index" not in batch:
                continue
            meta = {k: batch.pop(k) for k in META_KEYS if k in batch}
            self.model(**self._prepare_inputs(batch))
            z, _ = self.embed["head"](
                pool_turns(self.hidden.value, meta["turn_index"], len(meta["turn_speaker"]))
            )
            self.hidden.value = None
            vecs.append(z.float().cpu())
            keys += [(g, s, start + b) for g, s, b in meta["turn_keys"]]
        self.model.train(was_training)
        if not vecs:
            return None
        v = torch.cat(vecs)
        sim = (v @ v.T).numpy()
        y, score = [], []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i][0] == keys[j][0] and keys[i][2] != keys[j][2]:
                    y.append(keys[i][1] != keys[j][1])
                    score.append(1.0 - sim[i, j])
        return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if not self.is_world_process_zero():
            return metrics
        for ev in self.decode_evals:
            was_training = self.model.training
            self.model.eval()
            preds = predict_rows(
                self.model,
                ev["processor"],
                ev["dataset"],
                ev["n_speakers"],
                ev["batch_size"],
                ev["max_new_tokens"],
            )
            refs = [r["target"] for r in ev["dataset"].rows]
            scores = speaker_metrics(refs, preds, ev["normalize"], ev["fixed_labels"])
            self.model.train(was_training)
            scores = {f"{ev['prefix']}/{k}": v for k, v in scores.items() if k != "n"}
            self.log(scores)
            metrics.update(scores)
        if self.embed:
            auc = self.embed_auc()
            if auc is not None:
                self.log({"eval_embed/auc": auc})
                metrics["eval_embed/auc"] = auc
        return metrics


def save_head(head: SpeakerHead, directory) -> None:
    """speaker_head.pt next to a checkpoint/model: weights + the shape needed to rebuild it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": head.proj[0].in_features,
            "dim": head.proj[-1].out_features,
            "distill": head.to_ecapa is not None,
        },
        directory / "speaker_head.pt",
    )


@hydra.main(version_base=None, config_path="../../configs/speaker_asr", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(cfg.training.seed)
    if HydraConfig.get().runtime.choices.get("experiment") is None:
        logger.warning(
            "No +experiment=<v1|tower|context|junction|embed|short> preset: training under the scratch identity "
            "(pool %s, hub_model_id %s).",
            cfg.data.pool_dir,
            cfg.hub_model_id,
        )
    for split in (cfg.data.train_split, cfg.data.eval_split):
        check_pool(cfg, split)

    from transformers import AutoProcessor

    from tiny_audio.turns import load_model

    context = bool(cfg.context.enabled)
    junction = bool(cfg.junction.enabled)
    if context and junction:
        raise ValueError("context and junction windows are separate recipes; enable one")
    n_speakers = n_speaker_tokens(cfg)
    processor = AutoProcessor.from_pretrained(cfg.model.model_id)
    new_ids = register_speaker_tokens(processor, n_speakers, context)
    token_ids = speaker_token_ids(processor, n_speakers, context)
    model = load_model(cfg.model.model_id, dtype=getattr(torch, cfg.model.dtype))
    if new_ids:  # continuing from a speaker-ASR checkpoint keeps its trained rows
        init_speaker_rows(model, new_ids, seed=cfg.training.seed)
    model = apply_lora(
        model,
        token_ids,
        cfg.model.lora_target_modules,
        cfg.model.lora_rank,
        cfg.model.lora_alpha,
        cfg.model.lora_dropout,
    )
    model.print_trainable_parameters()
    cut_grad_into_frozen_audio_tower(model)

    columns = OmegaConf.to_container(cfg.data.columns)
    stores = {
        split: UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
        for split in (cfg.data.train_split, cfg.data.eval_split)
    }
    train_rows = read_manifest(cfg.data.pool_dir, cfg.data.train_split)
    eval_rows = read_manifest(cfg.data.pool_dir, cfg.data.eval_split)
    if context:
        train_rows = context_rows(cfg, cfg.data.train_split, train_rows, stores)
        eval_rows = context_rows(cfg, cfg.data.eval_split, eval_rows, stores)
    embedding = bool(cfg.embedding.enabled)
    embed_eval = []
    if embedding:
        ecfg = embedding_config(cfg)
        train_rows = annotate_turns(cfg, cfg.data.train_split, train_rows, stores)
        # AUC needs several windows per meeting: take whole meetings, in order
        by_meeting = annotate_turns(cfg, cfg.data.eval_split, eval_rows, stores)
        by_meeting.sort(key=lambda r: (str(r.get("group")), r["key"]))
        embed_eval = by_meeting[: ecfg.eval_rows]
        if ecfg.distill_weight:
            train_rows = attach_ecapa(cfg, cfg.data.train_split, train_rows, stores)
    eval_rows = subset(eval_rows, cfg.data.eval_samples)
    short = bool(cfg.short_chunks.enabled)
    short_eval, sample_weights, speaker_loss, augment_prob = [], None, None, 0.0
    if short:
        import random as _random

        scfg = short_chunk_config(cfg)
        prefix = CONTEXT_END if context else ""
        short_train = [
            {**r, "prefix": prefix}
            for r in short_rows(cfg, cfg.data.train_split, stores, cfg.pool.seed)
        ]
        n_rehearsal = min(
            len(train_rows), round(len(short_train) * scfg.rehearsal / (1 - scfg.rehearsal))
        )
        rehearsal = _random.Random(cfg.pool.seed).sample(train_rows, n_rehearsal)
        train_rows = short_train + rehearsal
        sample_weights = busy_weights(short_train, scfg.busy_oversample) + [1.0] * len(rehearsal)
        speaker_loss = (torch.tensor(token_ids), scfg.speaker_loss_weight)
        augment_prob = scfg.augment_prob
        held = short_rows(cfg, cfg.data.eval_split, stores, cfg.pool.seed + 1)
        short_eval = subset([{**r, "prefix": prefix} for r in held], scfg.eval_rows, seed=2)
        logger.info("short: %d short chunks + %d rehearsal rows", len(short_train), n_rehearsal)
    junction_eval = []
    if junction:
        jcfg = junction_config(cfg)
        n_junction = junction_count(len(train_rows), jcfg.ratio)
        train_rows = train_rows + junction_rows(
            cfg, cfg.data.train_split, n_junction, stores, cfg.pool.seed
        )
        junction_eval = junction_rows(
            cfg, cfg.data.eval_split, jcfg.eval_rows, stores, cfg.pool.seed + 1
        )
    logger.info("pool: %d train windows, %d eval windows", len(train_rows), len(eval_rows))

    from scripts.eval.audio import TextNormalizer

    eval_store = stores[cfg.data.eval_split]
    decode_spec = {
        "processor": processor,
        "n_speakers": n_speakers,
        "batch_size": cfg.training.per_device_eval_batch_size,
        "max_new_tokens": cfg.model.max_new_tokens,
        "normalize": TextNormalizer().normalize,
        "fixed_labels": context,
    }
    decode_evals = [
        {
            **decode_spec,
            "prefix": "eval_speakers",
            "dataset": SpeakerASRDataset(
                subset(eval_rows, cfg.data.decode_eval_samples, seed=1), eval_store
            ),
        }
    ]
    if short_eval:
        decode_evals.append(
            {
                **decode_spec,
                "prefix": "eval_short",
                "dataset": SpeakerASRDataset(short_eval, eval_store),
            }
        )
    if junction_eval:
        decode_evals.append(
            {
                **decode_spec,
                "prefix": "eval_junction",
                "dataset": SpeakerASRDataset(junction_eval, eval_store),
            }
        )
    embed = None
    if embedding:
        hidden_size = model.get_base_model().config.get_text_config().hidden_size
        head = SpeakerHead(hidden_size, ecfg.dim, distill=bool(ecfg.distill_weight))
        resume = cfg.resume_from_checkpoint
        if resume and (Path(resume) / "speaker_head.pt").exists():
            head.load_state_dict(
                torch.load(Path(resume) / "speaker_head.pt", weights_only=True)["state_dict"]
            )
        # A submodule of the PEFT model: moved to the device and optimised with
        # the adapter; merge_and_unload() drops it, so it is saved on its own.
        model.speaker_head = head
        embed = {
            "head": head,
            "cfg": ecfg,
            "eval_dataset": SpeakerASRDataset(embed_eval, eval_store) if embed_eval else None,
        }
    trainer = SpeakerASRTrainer(
        model=model,
        args=training_arguments(cfg, remove_unused_columns=False, label_names=["labels"]),
        train_dataset=SpeakerASRDataset(
            train_rows,
            stores[cfg.data.train_split],
            augment_prob=augment_prob,
            seed=cfg.training.seed,
        ),
        eval_dataset=SpeakerASRDataset(eval_rows, eval_store),
        # Same chat layout and transcript-only labels as turn-aware; the
        # target carries speaker tokens, and a context prefix is unsupervised.
        data_collator=SpeakerASRCollator(processor),
        decode_evals=decode_evals,
        embed=embed,
        sample_weights=sample_weights,
        speaker_loss=speaker_loss,
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
    if embed:
        save_head(embed["head"], final_dir)
    logger.info("merged model written to %s", final_dir.resolve())
    if cfg.hub_model_id:
        merged.push_to_hub(cfg.hub_model_id)
        processor.push_to_hub(cfg.hub_model_id)
        if embed:
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=str(final_dir / "speaker_head.pt"),
                path_in_repo="speaker_head.pt",
                repo_id=cfg.hub_model_id,
            )


if __name__ == "__main__":
    main()
