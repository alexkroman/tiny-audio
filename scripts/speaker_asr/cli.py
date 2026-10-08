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
    land on one voice -- the "no diarization" floor for cpWER. With
    `+experiment=context` the windows carry context prefixes and `sawer`
    (labels compared as given) is reported too.
    """
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    from scripts.eval.audio import TextNormalizer
    from scripts.eval.speaker_metrics import speaker_metrics
    from scripts.speaker_asr.config import context_config, load_config, n_speaker_tokens
    from scripts.speaker_asr.context import build_context_rows
    from scripts.speaker_asr.data import SpeakerASRDataset, UtteranceStore, load_texts, subset
    from scripts.speaker_asr.model import predict_rows, register_speaker_tokens
    from tiny_audio.turns import load_model

    cfg = load_config(overrides)
    path = Path(cfg.data.pool_dir) / f"{split}.parquet"
    if not path.exists():
        raise typer.BadParameter(
            f"{path} not found; run `ta speaker-asr build-pool --split {split}`"
        )
    rows = pd.read_parquet(path).to_dict("records")
    columns = OmegaConf.to_container(cfg.data.columns)
    store = UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
    context = bool(cfg.context.enabled)
    if context:  # prefixes need the whole meeting's history, so before subsetting
        starts = dict(zip(store.meta["id"], store.meta["start"].astype(float), strict=True))
        texts = load_texts(cfg, split, store.meta)
        rows = build_context_rows(rows, starts, texts, context_config(cfg), cfg.pool.seed)
    rows = subset(rows, max_samples)
    dataset = SpeakerASRDataset(rows, store)

    n_speakers = n_speaker_tokens(cfg)
    processor = AutoProcessor.from_pretrained(model)
    register_speaker_tokens(processor, n_speakers, context)  # no-op on a trained checkpoint
    net = load_model(model).eval()
    preds = predict_rows(
        net, processor, dataset, n_speakers, batch_size, cfg.model.max_new_tokens, progress=True
    )
    scores = speaker_metrics(
        [r["target"] for r in rows], preds, TextNormalizer().normalize, fixed_labels=context
    )
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


# ------------------------------------------------------------- long recordings

MAX_MINUTES_HELP = "Keep only each meeting's first N minutes (0 = whole meeting)"


def _meetings(cfg, split: str, max_samples: int, max_minutes: float):
    """(store, [(meeting id, parts at real time)]) for the first `max_samples` meetings."""
    from omegaconf import OmegaConf

    from scripts.speaker_asr.data import UtteranceStore, meeting_parts

    columns = OmegaConf.to_container(cfg.data.columns)
    store = UtteranceStore(cfg.data.dataset_id, cfg.data.data_files, split, columns)
    groups = sorted(store.meta["group"].unique())
    if max_samples:
        groups = groups[:max_samples]
    max_s = max_minutes * 60 if max_minutes else None
    return store, [(g, meeting_parts(store.meta, g, max_s)) for g in groups]


def _reference(parts: list[dict], texts: dict[str, str]) -> str:
    """A whole meeting's `<SPK_n>` reference, turns in start-time order."""
    from scripts.eval.speaker_metrics import serialize_turns

    ordered = sorted(parts, key=lambda p: (p["offset_s"], p["dur_s"]))
    return serialize_turns((p["speaker"], texts.get(p["id"], "")) for p in ordered)


def _load_checkpoint(model: str):
    """Processor + model for any Qwen3-ASR checkpoint (base, v1, context)."""
    from transformers import AutoProcessor

    from tiny_audio.turns import load_model

    return AutoProcessor.from_pretrained(model), load_model(model).eval()


@app.command("export-meetings")
def export_meetings_cmd(
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
    split: Annotated[str, typer.Option("--split", help="AMI split to export")] = "test",
    max_samples: Annotated[
        int, typer.Option("--max-samples", "-n", help="Meetings to export (0 = all)")
    ] = 0,
    max_minutes: Annotated[float, typer.Option("--max-minutes", help=MAX_MINUTES_HELP)] = 0,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", "-r", help="Push to this Hub dataset (private), as `split`"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Also write {split}.parquet here"),
    ] = None,
):
    """Export whole AMI meetings as long recordings, for `ta eval -d ami-speakers-long`.

    Each row is one meeting: every headset segment mixed at its real time
    (`audio`, ~20-50 min) and the meeting's `<SPK_n>` reference over AMI's
    human transcripts (`text`). cpWER on it scores linking speakers across
    the whole recording -- for this model's long-form decoding and for
    AssemblyAI alike. No pool is needed.
    """
    from datasets import Audio, Dataset, Features, Value

    from scripts.speaker_asr.config import load_config
    from scripts.speaker_asr.data import SAMPLE_RATE, mix_window

    if not repo_id and not output_dir:
        raise typer.BadParameter("pass --repo-id and/or --output-dir")
    cfg = load_config(overrides)
    store, meetings = _meetings(cfg, split, max_samples, max_minutes)
    human = dict(zip(store.meta["id"], store.meta["text"].fillna(""), strict=True))

    def examples():
        for group, parts in meetings:
            audio = mix_window(parts, store)
            yield {
                "audio": {"array": audio, "sampling_rate": SAMPLE_RATE},
                "text": _reference(parts, human),
                "key": group,
                "n_speakers": len({p["speaker"] for p in parts if human.get(p["id"], "").strip()}),
                "duration_s": len(audio) / SAMPLE_RATE,
            }

    features = Features(
        {
            "audio": Audio(sampling_rate=SAMPLE_RATE),
            "text": Value("string"),
            "key": Value("string"),
            "n_speakers": Value("int32"),
            "duration_s": Value("float32"),
        }
    )
    ds = Dataset.from_generator(examples, features=features)
    console.print(f"{split}: {len(ds)} meetings, {sum(ds['duration_s']) / 3600:.1f} h")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(output_dir / f"{split}.parquet"))
        console.print(f"wrote {output_dir / f'{split}.parquet'}")
    if repo_id:
        ds.push_to_hub(repo_id, split=split, private=True)
        console.print(f"pushed {split} to {repo_id}")


def _turns_json(result) -> list[dict]:
    return [
        {"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in result.turns
    ]


@app.command("evaluate-long")
def evaluate_long_cmd(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Qwen3-ASR checkpoint (Hub ID or local path)")
    ],
    overrides: Annotated[list[str] | None, typer.Argument(help=OVERRIDES_HELP)] = None,
    split: Annotated[str, typer.Option("--split", help="AMI split to score")] = "test",
    max_samples: Annotated[
        int, typer.Option("--max-samples", "-n", help="Meetings to score (0 = all)")
    ] = 0,
    max_minutes: Annotated[float, typer.Option("--max-minutes", help=MAX_MINUTES_HELP)] = 0,
    reference: Annotated[
        str,
        typer.Option(
            "--reference", help="'human' (AMI transcripts) or 'self' (the pool's transcripts)"
        ),
    ] = "human",
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Write predictions.jsonl and metrics.json here"),
    ] = None,
):
    """Recording-level cpWER: decode whole AMI meetings with long-form decoding.

    Each meeting is rebuilt as one recording (headset segments at their real
    times) and decoded chunk by chunk (scripts/speaker_asr/longform.py), so
    cpWER over the meeting scores whether a speaker keeps one label across
    the whole recording. A checkpoint without context training decodes chunks
    independently -- the no-linking baseline to compare against.
    """
    from scripts.eval.audio import TextNormalizer
    from scripts.eval.speaker_metrics import cp_errors, speaker_metrics
    from scripts.speaker_asr.config import context_config, load_config
    from scripts.speaker_asr.data import load_texts, mix_window
    from scripts.speaker_asr.longform import transcribe_long

    if reference not in ("human", "self"):
        raise typer.BadParameter("--reference must be 'human' or 'self'")
    cfg = load_config(overrides)
    store, meetings = _meetings(cfg, split, max_samples, max_minutes)
    if reference == "human":
        texts = dict(zip(store.meta["id"], store.meta["text"].fillna(""), strict=True))
    else:
        texts = load_texts(cfg, split, store.meta)
    processor, net = _load_checkpoint(model)
    normalize = TextNormalizer().normalize
    ccfg = context_config(cfg)

    refs, preds, records = [], [], []
    for group, parts in meetings:
        audio = mix_window(parts, store)
        result = transcribe_long(
            net, processor, audio, ccfg, max_new_tokens=cfg.model.max_new_tokens
        )
        ref = _reference(parts, texts)
        errors, words = cp_errors(ref, result.text, normalize)
        console.print(
            f"{group}: {len(audio) / 16000 / 60:.1f} min, {len(result.chunks)} chunks, "
            f"cpWER {errors / max(words, 1):.3f}, speakers "
            f"{len({t.speaker for t in result.turns})} (ref {len({p['speaker'] for p in parts})})"
        )
        refs.append(ref)
        preds.append(result.text)
        records.append(
            {"key": group, "target": ref, "prediction": result.text, "turns": _turns_json(result)}
        )
    scores = speaker_metrics(refs, preds, normalize)
    console.print(_metrics_table(scores, f"{model} on {split} ({len(refs)} meetings)"))
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "predictions.jsonl").open("w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        (output_dir / "metrics.json").write_text(json.dumps(scores, indent=2))
        console.print(f"wrote {output_dir}/predictions.jsonl and metrics.json")


@app.command("transcribe-long")
def transcribe_long_cmd(
    audio: Annotated[Path, typer.Argument(help="Audio file of any length (resampled to 16 kHz)")],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Qwen3-ASR checkpoint (Hub ID or local path)")
    ],
    overrides: Annotated[
        list[str] | None, typer.Argument(help="Hydra overrides for the context: section")
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Write <name>.json (timed turns) and <name>.txt"),
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="clustered: short chunks + ECAPA + spectral clustering (best); "
            "prefix: context-prefix chunk linking",
        ),
    ] = "clustered",
):
    """Speaker-attributed transcript of one audio file, minutes to hours long.

    Prints `[start-end] SPEAKER n: text` per turn; speakers are numbered by
    first appearance over the whole recording.
    """
    import librosa

    from scripts.speaker_asr.config import context_config, load_config

    if method not in ("clustered", "prefix"):
        raise typer.BadParameter("--method must be 'clustered' or 'prefix'")
    cfg = load_config(overrides)
    wav, _ = librosa.load(str(audio), sr=16000, mono=True)
    processor, net = _load_checkpoint(model)
    if method == "clustered":
        from scripts.speaker_asr.clustered import ClusterConfig, transcribe_clustered

        result = transcribe_clustered(
            net, processor, wav, ClusterConfig(max_new_tokens=cfg.model.max_new_tokens)
        )
    else:
        from scripts.speaker_asr.longform import transcribe_long

        result = transcribe_long(
            net, processor, wav, context_config(cfg), max_new_tokens=cfg.model.max_new_tokens
        )
    order = {}
    for t in result.turns:
        order.setdefault(t.speaker, len(order) + 1)

    def clock(seconds):
        return "  ?  " if seconds is None else f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"

    for t in result.turns:
        console.print(f"[{clock(t.start)}-{clock(t.end)}] SPEAKER {order[t.speaker]}: {t.text}")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        turns = [{**turn, "speaker": order[turn["speaker"]]} for turn in _turns_json(result)]
        (output_dir / f"{audio.stem}.json").write_text(
            json.dumps({"audio": str(audio), "chunks": result.chunks, "turns": turns}, indent=2)
        )
        (output_dir / f"{audio.stem}.txt").write_text(result.text + "\n")
        console.print(f"wrote {output_dir / audio.stem}.json and .txt")
