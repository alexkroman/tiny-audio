"""`ta speaker-asr`: build the AMI window pool and score speaker-attributed decodes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# Heavy imports (torch, transformers, datasets, tiny_audio) stay inside the
# commands so `ta --help` does not pay for them.

app = typer.Typer(
    no_args_is_help=True, help="Speaker-attributed ASR on Qwen3-ASR (diarization by decoding)."
)
console = Console()

OVERRIDES_HELP = "Hydra overrides for configs/speaker_asr (e.g. +experiment=v1)"


def _sample_groups(utts, max_samples: int, seed: int):
    """Whole meetings, in seeded random order, until >= max_samples utterances.

    Sampling utterances would leave windows full of holes; sampling meetings
    keeps every kept window's turn-taking intact.
    """
    groups = sorted(utts["group"].unique())
    random.Random(seed).shuffle(groups)
    sizes = utts["group"].value_counts()
    keep, total = [], 0
    for g in groups:
        if total >= max_samples:
            break
        keep.append(g)
        total += int(sizes[g])
    return utts[utts["group"].isin(keep)]


def _self_transcripts(parts: list[dict], store, model_id: str, cache: Path, batch_size: int):
    """Base-model transcript of every utterance, cached as JSONL and resumable.

    Each utterance is transcribed ALONE (its own headset clip), never inside
    the mix: that is the clean per-speaker text the window target is built
    from. Longest-first so batches pad evenly; a crash costs one batch.
    """
    from rich.progress import track

    from scripts.turn_aware.transcripts import read_cache
    from tiny_audio.turns import load_model, load_processor, transcribe

    done = read_cache(cache)
    todo = {p["id"]: p for p in parts if p["id"] not in done}
    if not todo:
        return done
    records = sorted(todo.values(), key=lambda p: -p["dur_s"])
    processor = load_processor(model_id)
    model = load_model(model_id).eval()
    with cache.open("a") as fh:
        for i in track(
            range(0, len(records), batch_size), description=f"transcribing {cache.stem}"
        ):
            chunk = records[i : i + batch_size]
            decoded = transcribe(model, processor, [store.get(p["row"]) for p in chunk])
            for part, d in zip(chunk, decoded, strict=True):
                done[part["id"]] = d.text
                fh.write(json.dumps({"key": part["id"], "text": d.text}) + "\n")
            fh.flush()
    return done


@app.command("build-pool")
def build_pool_cmd(
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
    splits: Annotated[
        list[str] | None,
        typer.Option(
            "--split",
            help="Dataset split(s) to build; repeat for several (default: train, validation)",
        ),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option(
            "--max-samples", "-n", help="Utterances per split, as whole meetings (smoke runs)"
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Decode batch size for self-transcription")
    ] = 32,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Rebuild even if the manifest is up to date")
    ] = False,
):
    """Build {split}.parquet window manifests in data.pool_dir from configs/speaker_asr.

    Pass the SAME overrides you train with; a split already built with
    identical settings is skipped (pool_config.json records them). With
    pool.hub_repo set (the presets set it), a split built elsewhere with the
    same settings is pulled instead of rebuilt, and every split built here is
    pushed -- so build locally, then `ta runpod train-speaker-asr` reuses it.
    """
    import pandas as pd
    from omegaconf import OmegaConf

    from scripts.speaker_asr import hub
    from scripts.speaker_asr.config import load_config, pool_signature, window_config
    from scripts.speaker_asr.data import UtteranceStore, build_pool, plan_windows
    from scripts.turn_aware.config import pool_is_current, write_signature

    cfg = load_config(overrides)
    if cfg.pool.target_text not in ("self", "dataset"):
        raise typer.BadParameter("pool.target_text must be 'self' or 'dataset'")
    output_dir = Path(cfg.data.pool_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.pool.transcript_cache or output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = pool_signature(cfg, max_samples)
    columns = OmegaConf.to_container(cfg.data.columns)

    repo = cfg.pool.hub_repo
    for split in splits or ["train", "validation"]:
        if not force and pool_is_current(output_dir, split, signature):
            console.print(f"[green]{split}[/]: {output_dir}/{split}.parquet is up to date")
            continue
        cache = cache_dir / f"transcripts-{split}.jsonl"
        if repo and not force:
            pulled = hub.pull(repo, split, output_dir, cache, signature)
            if pulled == "pulled":
                console.print(f"[green]{split}[/]: pulled the matching pool from {repo}")
                continue
            if pulled == "transcripts":
                console.print(f"{split}: merged {repo}'s transcripts; settings differ, rebuilding")
        store = UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
        utts = store.meta
        if max_samples and max_samples < len(utts):
            utts = _sample_groups(utts, max_samples, cfg.pool.seed)
        windows = plan_windows(utts.to_dict("records"), window_config(cfg), cfg.pool.seed)
        parts = [p for w in windows for p in w["parts"]]
        if cfg.pool.target_text == "self":
            texts = _self_transcripts(parts, store, cfg.pool.target_model, cache, batch_size)
        else:
            texts = dict(zip(utts["id"], utts["text"].fillna(""), strict=True))
        pool = pd.DataFrame(build_pool(windows, texts))
        path = output_dir / f"{split}.parquet"
        pool.to_parquet(path, index=False)
        write_signature(output_dir, split, signature)
        console.print(
            f"[green]{split}[/]: {len(utts)} utterances -> {len(pool)} windows "
            f"({pool['duration_s'].sum() / 3600:.1f} h) -> {path}"
        )
        console.print(pool["n_speakers"].value_counts().sort_index().to_string())
        if repo:
            try:  # a mirror, never a reason to fail the build (e.g. read-only token)
                hub.push(repo, split, output_dir, cache, signature)
                console.print(f"pushed {split} pool to {repo}")
            except Exception as exc:
                console.print(f"[yellow]pool push to {repo} failed ({exc}); continuing")


@app.command("export-windows")
def export_windows_cmd(
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
    split: Annotated[
        str, typer.Option("--split", help="Pool split to export (build it first)")
    ] = "test",
    max_samples: Annotated[
        int,
        typer.Option("--max-samples", "-n", help="Windows, balanced by speaker count (0 = all)"),
    ] = 0,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", "-r", help="Push to this Hub dataset (private), as `split`"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Also write {split}.parquet here"),
    ] = None,
):
    """Export a pool split as mixed audio + references, for `ta eval -d ami-speakers`.

    Columns: `audio` (16 kHz mix), `text` (AMI's HUMAN transcripts as
    `<SPK_n>` turns -- the reference `ta eval` scores, fair to any system),
    `text_self` (the training-style self-distilled target), `key`,
    `n_speakers`, `duration_s`. AMI references are uppercase and unpunctuated;
    WER and cpWER are computed after normalisation, so that costs nothing.
    """
    import pandas as pd
    from datasets import Audio, Dataset, Features, Value
    from omegaconf import OmegaConf

    from scripts.speaker_asr.config import load_config
    from scripts.speaker_asr.data import (
        SAMPLE_RATE,
        UtteranceStore,
        format_target,
        mix_window,
        subset,
    )

    if not repo_id and not output_dir:
        raise typer.BadParameter("pass --repo-id and/or --output-dir")
    cfg = load_config(overrides)
    path = Path(cfg.data.pool_dir) / f"{split}.parquet"
    if not path.exists():
        raise typer.BadParameter(
            f"{path} not found; run `ta speaker-asr build-pool --split {split}`"
        )
    rows = subset(pd.read_parquet(path).to_dict("records"), max_samples)
    columns = OmegaConf.to_container(cfg.data.columns)
    store = UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
    store.verify(rows)
    human = dict(zip(store.meta["id"], store.meta["text"].fillna(""), strict=True))

    def examples():
        for row in rows:
            parts = json.loads(row["parts"])
            text, n_speakers = format_target(parts, human)
            if not text:
                continue
            audio = mix_window(parts, store, row["tail_s"])
            yield {
                "audio": {"array": audio, "sampling_rate": SAMPLE_RATE},
                "text": text,
                "text_self": row["target"],
                "key": row["key"],
                "n_speakers": n_speakers,
                "duration_s": len(audio) / SAMPLE_RATE,
            }

    features = Features(
        {
            "audio": Audio(sampling_rate=SAMPLE_RATE),
            "text": Value("string"),
            "text_self": Value("string"),
            "key": Value("string"),
            "n_speakers": Value("int32"),
            "duration_s": Value("float32"),
        }
    )
    ds = Dataset.from_generator(examples, features=features)
    console.print(f"{split}: {len(ds)} windows, {sum(ds['duration_s']) / 3600:.1f} h")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(output_dir / f"{split}.parquet"))
        console.print(f"wrote {output_dir / f'{split}.parquet'}")
    if repo_id:
        ds.push_to_hub(repo_id, split=split, private=True)
        console.print(f"pushed {split} to {repo_id}")


def _metrics_table(scores: dict, title: str) -> Table:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in scores.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    return table


@app.command("evaluate")
def evaluate_cmd(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Speaker-ASR checkpoint (Hub ID or local path)")
    ],
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
    split: Annotated[
        str, typer.Option("--split", help="Pool split to score (build it first)")
    ] = "test",
    max_samples: Annotated[
        int,
        typer.Option("--max-samples", "-n", help="Windows, balanced by speaker count (0 = all)"),
    ] = 0,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Decode batch size")] = 16,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Write predictions.jsonl and metrics.json here"),
    ] = None,
):
    """Decode a pool split and report WER, cpWER and speaker-count accuracy.

    A base Qwen3-ASR scores too: it emits no speaker tokens, so all its words
    land on one voice -- the "no diarization" floor for cpWER.
    """
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    from scripts.eval.audio import TextNormalizer
    from scripts.speaker_asr.config import load_config
    from scripts.speaker_asr.data import SpeakerASRDataset, UtteranceStore, subset
    from scripts.speaker_asr.metrics import speaker_metrics
    from scripts.speaker_asr.model import predict_rows, register_speaker_tokens
    from tiny_audio.turns import load_model

    cfg = load_config(overrides)
    path = Path(cfg.data.pool_dir) / f"{split}.parquet"
    if not path.exists():
        raise typer.BadParameter(
            f"{path} not found; run `ta speaker-asr build-pool --split {split}`"
        )
    rows = subset(pd.read_parquet(path).to_dict("records"), max_samples)
    columns = OmegaConf.to_container(cfg.data.columns)
    store = UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
    dataset = SpeakerASRDataset(rows, store)

    n_speakers = cfg.pool.max_speakers
    processor = AutoProcessor.from_pretrained(model)
    register_speaker_tokens(processor, n_speakers)  # no-op on a trained checkpoint
    net = load_model(model).eval()
    preds = predict_rows(
        net, processor, dataset, n_speakers, batch_size, cfg.model.max_new_tokens, progress=True
    )
    scores = speaker_metrics([r["target"] for r in rows], preds, TextNormalizer().normalize)
    console.print(_metrics_table(scores, f"{model} on {split} ({len(rows)} windows)"))
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "predictions.jsonl").open("w") as fh:
            for row, pred in zip(rows, preds, strict=True):
                record = {
                    "key": row["key"],
                    "n_speakers": int(row["n_speakers"]),
                    "duration_s": float(row["duration_s"]),
                    "target": row["target"],
                    "prediction": pred,
                }
                fh.write(json.dumps(record) + "\n")
        (output_dir / "metrics.json").write_text(json.dumps(scores, indent=2))
        console.print(f"wrote {output_dir}/predictions.jsonl and metrics.json")
