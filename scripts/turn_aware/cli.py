"""`ta turn-aware`: build the pool, score markers, replay against commercial endpointers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from scripts.turn_aware.data import SAMPLE_RATE

app = typer.Typer(no_args_is_help=True, help="Turn-aware ASR on Qwen3-ASR (end-of-turn token).")
console = Console()

# `evaluate`'s default: the recipe-neutral pool from `build-pool +experiment=eval`.
DEFAULT_POOL_DIR = "data/turn_aware_eval"


def _print_metrics(title: str, metrics: dict) -> None:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


def _self_transcripts(obs, store, model_id: str, cache: Path, batch_size: int) -> dict[str, str]:
    """Base-model transcripts of every trimmed prefix, cached as JSONL and resumable.

    Longest-first so each batch pads to similar lengths, and a crash costs at
    most one batch.
    """
    from rich.progress import track

    from scripts.turn_aware.data import assemble_audio
    from scripts.turn_aware.model import decode_batch, load_model, load_processor

    done: dict[str, str] = {}
    if cache.exists():
        raw = cache.read_text()
        for line in raw.splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a write cut off by a kill: that clip is simply redone
            done[rec["key"]] = rec["text"]
        if raw and not raw.endswith("\n"):
            with cache.open("a") as fh:  # so the next record starts on its own line
                fh.write("\n")
    todo = obs[~obs["key"].isin(done)].sort_values("speech_end_s", ascending=False)
    if todo.empty:
        return done

    processor = load_processor(model_id)
    model = load_model(model_id).eval()
    records = todo.to_dict("records")
    with cache.open("a") as fh:
        for i in track(
            range(0, len(records), batch_size), description=f"transcribing {cache.stem}"
        ):
            chunk = records[i : i + batch_size]
            audios = [
                assemble_audio(store.get(r["turn_key"]), {**r, "lead_s": 0.0, "tail_s": 0.0})
                for r in chunk
            ]
            for rec, (text, _) in zip(chunk, decode_batch(model, processor, audios), strict=True):
                done[rec["key"]] = text
                fh.write(json.dumps({"key": rec["key"], "text": text}) + "\n")
            fh.flush()
    return done


def _prepare_observations(obs, store, target_text, model, cache, batch_size):
    """Add speech_end_s, agent_turn and target to observation rows (a DataFrame).

    The cut instant is pulled back to the last speech frame, so every example's
    silence is exactly the silence build_pool adds. Rows with no speech left
    are dropped.
    """
    from scripts.turn_aware.data import trim_tail_silence

    obs = obs[obs["turn_key"].isin(store.meta)].sort_values("turn_key").copy()
    obs["speech_end_s"] = [
        trim_tail_silence(store.get(k)[: round(at_s * SAMPLE_RATE)]) / SAMPLE_RATE
        for k, at_s in zip(obs["turn_key"], obs["at_s"], strict=True)
    ]
    obs["agent_turn"] = [store.meta[k]["agent_turn"] for k in obs["turn_key"]]
    obs = obs[obs["speech_end_s"] > 0]
    if target_text == "self":
        texts = _self_transcripts(obs, store, model, cache, batch_size)
        obs["target"] = [texts[k] for k in obs["key"]]
    else:
        obs["target"] = obs["text"].fillna("")
    return obs


def _load_extra(source: str, split: str):
    """Prepared extra observations for `split`: a local dir of {split}.parquet or a Hub repo."""
    import pandas as pd

    local = Path(source) / f"{split}.parquet"
    if local.exists():
        return pd.read_parquet(local)
    from datasets import load_dataset

    return load_dataset(source, split=split).to_pandas()


@app.command("build-pool")
def build_pool_cmd(
    overrides: Annotated[
        list[str] | None,
        typer.Argument(
            help="Hydra overrides for configs/turn_aware (e.g. +experiment=v2 pool.ctx_prob=0.8)"
        ),
    ] = None,
    splits: Annotated[
        list[str] | None,
        typer.Option(
            "--split",
            help="Dataset split(s) to build; repeat for several (default: train, validation)",
        ),
    ] = None,
    max_samples: Annotated[
        int | None,
        typer.Option("--max-samples", "-n", help="Random observations per split (smoke runs)"),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Decode batch size for self-transcription")
    ] = 32,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Rebuild even if the manifest is up to date")
    ] = False,
):
    """Build {split}.parquet manifests in data.pool_dir from configs/turn_aware.

    Every data setting comes from the Hydra config -- pool dir, mined extras,
    transcript cache, silence/oversampling knobs -- so pass the SAME overrides
    you train with. A split whose manifest was built with identical settings
    is skipped; pool_config.json records them.
    """
    import pandas as pd

    from scripts.turn_aware.config import (
        load_config,
        pool_config,
        pool_is_current,
        pool_signature,
        write_signature,
    )
    from scripts.turn_aware.data import TurnAudioStore, build_pool, load_split

    cfg = load_config(overrides)
    if cfg.pool.target_text not in ("self", "dataset"):
        raise typer.BadParameter("pool.target_text must be 'self' or 'dataset'")
    output_dir = Path(cfg.data.pool_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.pool.transcript_cache or output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = pool_signature(cfg, max_samples)
    config = pool_config(cfg)

    for split in splits or ["train", "validation"]:
        if not force and pool_is_current(output_dir, split, signature):
            console.print(f"[green]{split}[/]: {output_dir}/{split}.parquet is up to date")
            continue
        obs = load_split("observations", split, cfg.data.dataset_id).to_pandas()
        frac = 1.0
        if max_samples and max_samples < len(obs):
            frac = max_samples / len(obs)
            obs = obs.sample(n=max_samples, random_state=cfg.pool.seed)
        store = TurnAudioStore(split, cfg.data.dataset_id)
        cache = cache_dir / f"transcripts-{split}.jsonl"
        obs = _prepare_observations(
            obs, store, cfg.pool.target_text, cfg.pool.target_model, cache, batch_size
        )
        if cfg.pool.extra:
            more = _load_extra(cfg.pool.extra, split)
            if frac < 1.0:  # keep the extra rows in proportion to a smoke sample
                more = more.sample(frac=frac, random_state=cfg.pool.seed)
            obs = pd.concat([obs, more[more["turn_key"].isin(store.meta)]], ignore_index=True)
            console.print(f"merged {len(more)} extra observations from {cfg.pool.extra}")

        pool = pd.DataFrame(build_pool(obs.to_dict("records"), config, seed=cfg.pool.seed))
        path = output_dir / f"{split}.parquet"
        pool.to_parquet(path, index=False)
        write_signature(output_dir, split, signature)
        console.print(
            f"[green]{split}[/]: {len(obs)} observations -> {len(pool)} examples -> {path}"
        )
        console.print(pool["schema"].value_counts().to_string())


@app.command("mine-pauses")
def mine_pauses_cmd(
    splits: Annotated[
        list[str] | None,
        typer.Option(
            "--split",
            help="Split(s) to mine; repeat for several (default: train, validation, test)",
        ),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Where {split}.parquet observations go")
    ] = Path("data/turn_aware_pauses"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="Base model that writes self-distilled targets")
    ] = "Qwen/Qwen3-ASR-0.6B-hf",
    max_samples: Annotated[
        int | None,
        typer.Option("--max-samples", "-n", help="Random mined pauses per split (smoke runs)"),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Decode batch size for self-transcription")
    ] = 32,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", "-r", help="Also push the prepared splits to this Hub dataset"),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for --max-samples")] = 0,
):
    """Mine unlabelled mid-sentence pauses as label-0 observations, ready for --extra."""
    import pandas as pd

    from scripts.turn_aware.data import TurnAudioStore, load_split, mine_pauses

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits or ["train", "validation", "test"]:
        obs = load_split("observations", split).to_pandas()
        labelled = {
            (k, round(float(a), 2)) for k, a in zip(obs["turn_key"], obs["at_s"], strict=True)
        }
        turns = load_split("turns", split).remove_columns(["audio"]).to_pandas()
        mined = pd.DataFrame(mine_pauses(turns.to_dict("records"), labelled))
        if max_samples and max_samples < len(mined):
            mined = mined.sample(n=max_samples, random_state=seed)
        store = TurnAudioStore(split)
        cache = output_dir / f"transcripts-{split}.jsonl"
        mined = _prepare_observations(mined, store, "self", model, cache, batch_size)
        path = output_dir / f"{split}.parquet"
        mined.to_parquet(path, index=False)
        console.print(f"[green]{split}[/]: {len(mined)} mid-sentence pauses -> {path}")
        if repo_id:
            from datasets import Dataset

            Dataset.from_pandas(mined, preserve_index=False).push_to_hub(
                repo_id, split=split, private=True
            )
            console.print(f"pushed {split} to {repo_id}")


# Margin thresholds swept by `evaluate`. 0 is plain greedy; positive values
# demand more confidence before firing (fewer false fires, some lost recall).
SWEEP_TAUS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


def _print_columns(title: str, columns: dict[str, dict], metrics) -> None:
    """One row per metric, one column per named metrics dict ("-" when absent)."""
    table = Table(title=title)
    table.add_column("metric")
    for name in columns:
        table.add_column(name, justify="right")
    for key in metrics:
        cells = [columns[name].get(key) for name in columns]
        table.add_row(
            key,
            *("-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v) for v in cells),
        )
    console.print(table)


@app.command("evaluate")
def evaluate_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="Trained model dir or Hub ID")],
    pool_dir: Annotated[
        Path, typer.Option("--pool-dir", help="Directory holding {split}.parquet manifests")
    ] = Path(DEFAULT_POOL_DIR),
    split: Annotated[str, typer.Option("--split", help="Manifest split to score")] = "validation",
    max_samples: Annotated[
        int, typer.Option("--max-samples", "-n", help="Schema-balanced sample size (0 = all)")
    ] = 2000,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Decode batch size")] = 32,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", "-o", help="Write metrics.json + per-row predictions.parquet here"
        ),
    ] = None,
):
    """Score end-of-turn firing (P/R/F1, per kind) and transcript drift on a manifest.

    Also splits the scores by whether the agent's question was given as
    context, and sweeps a confidence threshold on the marker: fire only when
    logit(<END_OF_TURN>) - logit(<|im_end|>) > tau at the end of the transcript.
    """
    import pandas as pd

    from scripts.eval.audio import TextNormalizer
    from scripts.turn_aware.data import (
        SWEEP_METRICS,
        TurnAudioStore,
        marker_metrics,
        stratified_subset,
        threshold_sweep,
    )
    from scripts.turn_aware.model import load_model, load_processor
    from scripts.turn_aware.train import predict_rows

    rows = stratified_subset(
        pd.read_parquet(pool_dir / f"{split}.parquet").to_dict("records"), max_samples
    )
    processor = load_processor(model)
    net = load_model(model).eval()
    preds = predict_rows(net, processor, rows, TurnAudioStore(split), batch_size, progress=True)
    greedy = [(t, f) for t, f, _ in preds]
    margins = [m for _, _, m in preds]

    metrics = marker_metrics(rows, greedy, normalize=TextNormalizer().normalize)
    _print_metrics(f"{model} on {split}", metrics)

    by_context = {"all": metrics}
    for name, has_ctx in (("with context", True), ("no context", False)):
        idx = [i for i, r in enumerate(rows) if bool(r["ctx"]) == has_ctx]
        if idx:
            by_context[name] = marker_metrics(
                [rows[i] for i in idx], [greedy[i] for i in idx], text_wer=False
            )
    _print_columns("by agent-question context", by_context, ("n", *SWEEP_METRICS))

    sweep = threshold_sweep(rows, margins, SWEEP_TAUS)
    table = Table(title="marker threshold sweep: fire iff margin > tau (tau=0 is greedy)")
    for col in ("tau", *(k.split("/")[-1] for k in SWEEP_METRICS)):
        table.add_column(col, justify="right")
    for s in sweep:
        table.add_row(f"{s['tau']:g}", *(f"{s[k]:.3f}" if k in s else "-" for k in SWEEP_METRICS))
    console.print(table)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {**metrics, "by_context": by_context, "threshold_sweep": sweep}
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
        frame = pd.DataFrame(rows)
        frame["pred_text"] = [t for t, _ in greedy]
        frame["pred_fired"] = [f for _, f in greedy]
        frame["marker_margin"] = margins
        frame.to_parquet(output_dir / "predictions.parquet", index=False)
        console.print(f"wrote {output_dir / 'metrics.json'} and predictions.parquet")


@app.command("set-threshold")
def set_threshold_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="Model dir or Hub ID to read")],
    tau: Annotated[
        float, typer.Option("--tau", help="Fire only when the marker margin exceeds this (0 = off)")
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Write the updated generation_config.json here"),
    ] = None,
    repo_id: Annotated[
        str | None,
        typer.Option(
            "--repo-id", "-r", help="Push ONLY generation_config.json to this Hub model repo"
        ),
    ] = None,
):
    """Ship a marker threshold with a checkpoint via generation_config.json.

    Weights are untouched: the threshold is a `sequence_bias` of -tau on
    <END_OF_TURN> that stock `generate()` applies.
    """
    from transformers import GenerationConfig

    from scripts.turn_aware.data import END_OF_TURN
    from scripts.turn_aware.model import load_processor, set_end_of_turn_threshold

    if output_dir is None and repo_id is None:
        raise typer.BadParameter("pass --output-dir and/or --repo-id", param_hint="--output-dir")
    token_id = load_processor(model).tokenizer.convert_tokens_to_ids(END_OF_TURN)
    config = GenerationConfig.from_pretrained(model)
    set_end_of_turn_threshold(config, token_id, tau)
    console.print(f"sequence_bias -> {config.sequence_bias}")
    if output_dir:
        config.save_pretrained(output_dir)
        console.print(f"wrote {output_dir / 'generation_config.json'}")
    if repo_id:
        config.push_to_hub(repo_id, commit_message=f"Set end-of-turn threshold tau={tau:g}")
        console.print(f"pushed generation_config.json to {repo_id}")


def _fire_times(
    net, processor, audio, ctx, hop_s, gate_s, tail_s, batch_size, taus
) -> dict[float, float | None]:
    """Stream `audio` + `tail_s` of silence; first window end that fires, per tau.

    Every `hop_s` the model sees the WHOLE prefix so far, as in training. The
    `gate_s` energy gate only skips decodes that could not fire anyway (a fire
    needs >=0.3 s of observed silence), which keeps replay affordable. One pass
    serves every tau: decoding stops once the largest tau has fired, and every
    smaller tau fired at or before that window.
    """
    from scripts.turn_aware.data import first_fire_times, trailing_is_silent
    from scripts.turn_aware.model import decode_batch

    padded = np.concatenate([audio, np.zeros(int(tail_s * SAMPLE_RATE), dtype=np.float32)])
    times = np.arange(hop_s, len(padded) / SAMPLE_RATE + 1e-6, hop_s)
    candidates = [t for t in times if trailing_is_silent(padded[: int(t * SAMPLE_RATE)], gate_s)]
    seen_t: list[float] = []
    seen_m: list[float] = []
    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i : i + batch_size]
        prefixes = [padded[: int(t * SAMPLE_RATE)] for t in chunk]
        preds = decode_batch(
            net, processor, prefixes, [ctx] * len(chunk), max_new_tokens=160, return_margin=True
        )
        seen_t += [float(t) for t in chunk]
        seen_m += [m for _, _, m in preds]
        if any(m > max(taus) for _, _, m in preds):
            break
    return first_fire_times(seen_t, seen_m, taus)


@app.command("replay")
def replay_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="Trained model dir or Hub ID")],
    max_samples: Annotated[
        int | None, typer.Option("--max-samples", "-n", help="Number of test turns (default: all)")
    ] = None,
    hop_s: Annotated[float, typer.Option("--hop-s", help="Seconds between decodes")] = 0.16,
    gate_s: Annotated[
        float,
        typer.Option("--gate-s", help="Decode only when this much tail is silent (0 = always)"),
    ] = 0.1,
    tail_s: Annotated[
        float, typer.Option("--tail-s", help="Silence appended after the turn")
    ] = 2.0,
    context: Annotated[
        bool, typer.Option("--context/--no-context", help="Give the agent's question as context")
    ] = False,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Windows decoded per batch")] = 16,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the --max-samples draw")] = 0,
    taus: Annotated[
        list[float] | None,
        typer.Option(
            "--tau",
            help="Fire when marker margin > tau (default 0 = greedy); repeat to score several",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Write per-turn results + summary here"),
    ] = None,
):
    """Streaming replay on the test turns the commercial endpointers were run on."""
    import pandas as pd
    from rich.progress import track

    from scripts.turn_aware.data import TurnAudioStore, endpoint_summary, load_split
    from scripts.turn_aware.model import load_model, load_processor

    endpoints = load_split("endpoints", "test").to_pandas()
    # A seeded random sample, never the first N: turn keys sort by style, so the
    # first 30 are all `afterthought_*` multi-sentence turns.
    keys = sorted(endpoints["turn_key"].unique())
    if max_samples and max_samples < len(keys):
        keys = sorted(random.Random(seed).sample(keys, max_samples))
    store = TurnAudioStore("test")
    processor = load_processor(model)
    net = load_model(model).eval()

    taus = sorted(set(taus or [0.0]))
    results = []
    for key in track(keys, description="replaying"):
        ctx = store.meta[key]["agent_turn"] if context else ""
        fires = _fire_times(
            net, processor, store.get(key), ctx, hop_s, gate_s, tail_s, batch_size, taus
        )
        results.append(
            {
                "turn_key": key,
                "audio_duration_s": store.meta[key]["audio_duration_s"],
                **{f"fire_s_tau{tau:g}": fires[tau] for tau in taus},
            }
        )
    ours = pd.DataFrame(results)

    summaries = {
        f"turn-aware tau={tau:g}": endpoint_summary(
            ours[f"fire_s_tau{tau:g}"], ours["audio_duration_s"]
        )
        for tau in taus
    }
    arms = endpoints[endpoints["turn_key"].isin(keys)]
    for arm, grp in arms.groupby("arm"):
        summaries[arm] = endpoint_summary(grp["first_endpoint_s"], grp["audio_duration_s"])

    table = Table(title=f"streaming replay, {len(keys)} test turns")
    for col in ("system", "n", "cut_early", "missed", "latency_p50_s", "latency_p90_s"):
        table.add_column(col, justify="right")
    for name, s in summaries.items():
        table.add_row(
            name,
            str(s["n"]),
            f"{s['cut_early']:.1%}",
            f"{s['missed']:.1%}",
            f"{s['latency_p50_s']:.2f}",
            f"{s['latency_p90_s']:.2f}",
        )
    console.print(table)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ours.to_csv(output_dir / "replay.csv", index=False)
        (output_dir / "replay_summary.json").write_text(json.dumps(summaries, indent=2))
