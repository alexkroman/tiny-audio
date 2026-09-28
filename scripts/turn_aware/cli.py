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

# tiny_audio.turns is imported inside each command, never here: importing it
# loads the whole tiny_audio package (~3.5 s), which every `ta --help` and
# `ta turn-aware ...` invocation would otherwise pay.

app = typer.Typer(no_args_is_help=True, help="Turn-aware ASR on Qwen3-ASR (end-of-turn token).")
console = Console()


def _self_transcripts(
    obs, store, model_id: str, cache: Path, batch_size: int, key_col: str = "key", with_ctx=False
) -> dict[str, str]:
    """Base-model transcripts of every trimmed prefix, cached as JSONL and resumable.

    Longest-first so each batch pads to similar lengths, and a crash costs at
    most one batch. With `with_ctx`, each prefix is transcribed with its row's
    `ctx` as the system prompt, cached under `key_col` (see transcript_key).
    """
    from rich.progress import track

    from scripts.turn_aware.data import assemble_audio
    from scripts.turn_aware.transcripts import read_cache
    from tiny_audio.turns import load_model, load_processor, transcribe

    done = read_cache(cache)
    todo = obs[~obs[key_col].isin(done)].drop_duplicates(key_col)
    todo = todo.sort_values("speech_end_s", ascending=False)
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
            contexts = [r["ctx"] for r in chunk] if with_ctx else None
            decoded_batch = transcribe(model, processor, audios, contexts)
            for rec, decoded in zip(chunk, decoded_batch, strict=True):
                done[rec[key_col]] = decoded.text
                fh.write(json.dumps({"key": rec[key_col], "text": decoded.text}) + "\n")
            fh.flush()
    return done


def _trim_observations(obs, store):
    """Add speech_end_s and agent_turn to observation rows (a DataFrame).

    The cut instant is pulled back to the last speech frame, so every example's
    silence is exactly the silence build_pool adds. Rows with no speech left
    are dropped.
    """
    from scripts.turn_aware.data import trim_tail_silence
    from tiny_audio.turns import SAMPLE_RATE

    obs = obs[obs["turn_key"].isin(store.meta)].sort_values("turn_key").copy()
    obs["speech_end_s"] = [
        trim_tail_silence(store.get(k)[: round(at_s * SAMPLE_RATE)]) / SAMPLE_RATE
        for k, at_s in zip(obs["turn_key"], obs["at_s"], strict=True)
    ]
    obs["agent_turn"] = [store.meta[k]["agent_turn"] for k in obs["turn_key"]]
    return obs[obs["speech_end_s"] > 0]


def _attach_targets(obs, store, target_text, model, cache, batch_size, with_ctx=False):
    """Set `target`: the base model's transcript (optionally with each row's ctx) or `text`."""
    from scripts.turn_aware.data import transcript_key

    obs = obs.copy()
    if target_text != "self":
        obs["target"] = obs["text"].fillna("")
        return obs
    obs["tkey"] = [transcript_key(r) for r in obs.to_dict("records")] if with_ctx else obs["key"]
    texts = _self_transcripts(obs, store, model, cache, batch_size, "tkey", with_ctx)
    obs["target"] = [texts[k] for k in obs["tkey"]]
    return obs.drop(columns=["tkey"])


def _prepare_observations(obs, store, target_text, model, cache, batch_size):
    """_trim_observations then _attach_targets without context (mine-pauses)."""
    obs = _trim_observations(obs, store)
    return _attach_targets(obs, store, target_text, model, cache, batch_size)


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

    from scripts.turn_aware import transcripts
    from scripts.turn_aware.config import (
        load_config,
        pool_config,
        pool_is_current,
        pool_signature,
        write_signature,
    )
    from scripts.turn_aware.data import TurnAudioStore, assign_contexts, build_pool, load_split

    cfg = load_config(overrides)
    if cfg.pool.target_text not in ("self", "dataset"):
        raise typer.BadParameter("pool.target_text must be 'self' or 'dataset'")
    if cfg.pool.target_with_context and cfg.pool.target_text != "self":
        raise typer.BadParameter("pool.target_with_context needs pool.target_text=self")
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
        repo, model = cfg.pool.transcript_repo, cfg.pool.target_model
        if repo and cfg.pool.target_text == "self":
            pulled = transcripts.pull(repo, split, cache, model)
            console.print(f"pulled {pulled} new transcripts for {split} from {repo}")
        before = len(transcripts.read_cache(cache))
        obs = _trim_observations(obs, store)
        with_ctx = bool(cfg.pool.target_with_context)
        if not with_ctx:  # extras below keep the targets they were prepared with
            obs = _attach_targets(obs, store, cfg.pool.target_text, model, cache, batch_size)
        if cfg.pool.extra:
            more = _load_extra(cfg.pool.extra, split)
            if cfg.pool.extra_labels:
                if "completeness" not in more:
                    raise typer.BadParameter(
                        f"pool.extra_labels is set but {cfg.pool.extra} has no `completeness` "
                        "column; run `ta turn-aware label-pauses` first"
                    )
                more = more[more["completeness"].isin(list(cfg.pool.extra_labels))]
            if frac < 1.0:  # keep the extra rows in proportion to a smoke sample
                more = more.sample(frac=frac, random_state=cfg.pool.seed)
            obs = pd.concat([obs, more[more["turn_key"].isin(store.meta)]], ignore_index=True)
            console.print(f"merged {len(more)} extra observations from {cfg.pool.extra}")
        if with_ctx or config.ctx_distractor_prob > 0:
            obs = pd.DataFrame(assign_contexts(obs.to_dict("records"), config, cfg.pool.seed))
        if with_ctx:  # every row, extras included, transcribed with the context it shows
            obs = _attach_targets(obs, store, "self", model, cache, batch_size, with_ctx=True)
        added = len(transcripts.read_cache(cache)) - before
        if repo and added:
            try:  # a backup, never a reason to fail the build (e.g. read-only token)
                transcripts.push(repo, split, cache, model)
                console.print(f"pushed {split} transcripts (+{added}) to {repo}")
            except Exception as exc:
                console.print(f"[yellow]transcript push to {repo} failed ({exc}); continuing")

        pool = pd.DataFrame(build_pool(obs.to_dict("records"), config, seed=cfg.pool.seed))
        path = output_dir / f"{split}.parquet"
        pool.to_parquet(path, index=False)
        write_signature(output_dir, split, signature)
        console.print(
            f"[green]{split}[/]: {len(obs)} observations -> {len(pool)} examples -> {path}"
        )
        console.print(pool["schema"].value_counts().to_string())


@app.command("push-transcripts")
def push_transcripts_cmd(
    overrides: Annotated[
        list[str] | None,
        typer.Argument(
            help="Hydra overrides (pool.transcript_cache / transcript_repo / target_model)"
        ),
    ] = None,
    splits: Annotated[
        list[str] | None,
        typer.Option(
            "--split", help="Split(s) to push (default: every transcripts-*.jsonl present)"
        ),
    ] = None,
):
    """Back up the base-model transcript cache to pool.transcript_repo on the Hub.

    Pulls first, so rows another machine pushed are kept rather than
    overwritten. build-pool pulls this repo before transcribing, so a fresh
    pod skips the ~1 h transcription pass entirely.
    """
    from scripts.turn_aware import transcripts
    from scripts.turn_aware.config import load_config

    cfg = load_config(overrides)
    repo, model = cfg.pool.transcript_repo, cfg.pool.target_model
    if not repo:
        raise typer.BadParameter("pool.transcript_repo is not set")
    cache_dir = Path(cfg.pool.transcript_cache)
    found = sorted(
        p.stem.removeprefix("transcripts-") for p in cache_dir.glob("transcripts-*.jsonl")
    )
    for split in splits or found:
        cache = cache_dir / f"transcripts-{split}.jsonl"
        if not cache.exists():
            console.print(f"[yellow]{cache} missing; skipped")
            continue
        pulled = transcripts.pull(repo, split, cache, model)
        n = transcripts.push(repo, split, cache, model)
        console.print(
            f"[green]{split}[/]: pushed {n} transcripts to {repo} ({pulled} merged from it first)"
        )


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
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Base model that writes self-distilled targets (default: tiny_audio.turns.MODEL_ID)",
        ),
    ] = None,
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
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="'boundary': mid-sentence chunk ends (script text); "
            "'intra': silences >= 0.3 s inside chunks (base-model transcript as text)",
        ),
    ] = "boundary",
):
    """Mine unlabelled pauses as label-0 observations, for `pool.extra`.

    Must use the same base model as the pool's `pool.target_model`, so mined
    targets are self-distilled from the same transcriber. Transcripts are
    pulled from / pushed to pool.transcript_repo (split `<kind>_<split>`), so
    an interrupted or repeated run never re-transcribes.
    """
    import pandas as pd

    from scripts.turn_aware import transcripts
    from scripts.turn_aware.config import load_config
    from scripts.turn_aware.data import (
        INTRA_KIND,
        TurnAudioStore,
        find_intra_pauses,
        load_split,
        mine_pauses,
    )
    from tiny_audio.turns import MODEL_ID

    if kind not in ("boundary", "intra"):
        raise typer.BadParameter("must be 'boundary' or 'intra'", param_hint="--kind")
    model = model or MODEL_ID
    mirror = load_config().pool.transcript_repo

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits or ["train", "validation", "test"]:
        obs = load_split("observations", split).to_pandas()
        labelled = {
            (k, round(float(a), 2)) for k, a in zip(obs["turn_key"], obs["at_s"], strict=True)
        }
        turns = load_split("turns", split).remove_columns(["audio"]).to_pandas()
        store = TurnAudioStore(split)
        if kind == "boundary":
            mined = pd.DataFrame(mine_pauses(turns.to_dict("records"), labelled))
        else:
            cuts = obs.groupby("turn_key")["at_s"].apply(list).to_dict()
            rows = []
            for turn in turns.itertuples():
                avoid = list(turn.chunk_ends_s) + cuts.get(turn.turn_key, [])
                for t in find_intra_pauses(store.get(turn.turn_key), avoid):
                    rows.append(
                        {
                            "key": f"{turn.turn_key}@{t:g}#intra",
                            "turn_key": turn.turn_key,
                            # Inside the silence; _prepare_observations trims back to speech.
                            "at_s": round(t + 0.1, 2),
                            "label": 0,
                            "kind": INTRA_KIND,
                            "style": turn.style,
                            "domain": turn.domain,
                        }
                    )
            mined = pd.DataFrame(rows)
        if max_samples and max_samples < len(mined):
            mined = mined.sample(n=max_samples, random_state=seed)
        cache = output_dir / f"transcripts-{split}.jsonl"
        mirror_split = f"{kind}_{split}"
        if mirror:
            n = transcripts.pull(mirror, mirror_split, cache, model)
            console.print(f"pulled {n} transcripts from {mirror} [{mirror_split}]")
        mined = _prepare_observations(mined, store, "self", model, cache, batch_size)
        if kind == "intra":  # no script text inside a chunk: the words are the transcript
            mined["text"] = mined["target"]
        if mirror:
            try:
                transcripts.push(mirror, mirror_split, cache, model)
                console.print(f"pushed transcripts to {mirror} [{mirror_split}]")
            except Exception as exc:
                console.print(f"[yellow]transcript push to {mirror} failed ({exc}); continuing")
        path = output_dir / f"{split}.parquet"
        mined.to_parquet(path, index=False)
        console.print(f"[green]{split}[/]: {len(mined)} {kind} pauses -> {path}")
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


@app.command("label-pauses")
def label_pauses_cmd(
    pauses_dir: Annotated[
        Path, typer.Option("--dir", help="mine-pauses output: {split}.parquet files")
    ] = Path("data/turn_aware_pauses"),
    splits: Annotated[
        list[str] | None,
        typer.Option("--split", help="Split(s) to label (default: train, validation, test)"),
    ] = None,
    pilot: Annotated[
        int,
        typer.Option(
            "--pilot", help="Label N random rows with direct calls, print them, write nothing"
        ),
    ] = 0,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", "-r", help="Push the labelled splits to this Hub dataset"),
    ] = None,
    poll_s: Annotated[int, typer.Option("--poll-s", help="Seconds between batch polls")] = 60,
):
    """Add a Claude `completeness` label (incomplete/complete/ambiguous) to mined pauses.

    Runs one Message Batch (50% price, usually < 1 h). The batch id is saved in
    <dir>/label_batch.json, so rerunning after an interruption resumes polling
    instead of paying twice. Rows the batch failed on are retried directly.
    Needs the `anthropic` package and credentials (ANTHROPIC_API_KEY).
    """
    import time

    import anthropic
    import pandas as pd
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from scripts.turn_aware.label import parse_label, request_params

    client = anthropic.Anthropic()
    splits = splits or ["train", "validation", "test"]
    frames = {s: pd.read_parquet(pauses_dir / f"{s}.parquet") for s in splits}

    if pilot:
        sample = pd.concat(frames.values()).sample(n=pilot, random_state=0)
        rows = []
        for r in sample.itertuples():
            msg = client.messages.create(**request_params(r.agent_turn, r.text))
            rows.append(
                {"label": parse_label(msg), "caller": r.text[-90:], "agent": r.agent_turn[:50]}
            )
        table = Table(title=f"pilot: {pilot} mined pauses")
        for col in ("label", "caller so far (tail)", "agent asked"):
            table.add_column(col)
        for r in rows:
            table.add_row(str(r["label"]), r["caller"], r["agent"])
        console.print(table)
        console.print(pd.Series([r["label"] for r in rows]).value_counts().to_string())
        return

    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$, so rows are addressed by position.
    state_path = pauses_dir / "label_batch.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("splits") != splits:
        requests = [
            Request(
                custom_id=f"{s}-{i}",
                params=MessageCreateParamsNonStreaming(**request_params(r.agent_turn, r.text)),
            )
            for s, df in frames.items()
            for i, r in enumerate(df.itertuples())
        ]
        batch = client.messages.batches.create(requests=requests)
        state = {"batch_id": batch.id, "splits": splits}
        state_path.write_text(json.dumps(state))
        console.print(f"submitted batch {batch.id} with {len(requests)} requests")

    while (
        batch := client.messages.batches.retrieve(state["batch_id"])
    ).processing_status != "ended":
        c = batch.request_counts
        console.print(
            f"{batch.processing_status}: {c.succeeded} ok, {c.errored} err, {c.processing} left"
        )
        time.sleep(poll_s)

    labels: dict[str, str | None] = {}
    for result in client.messages.batches.results(state["batch_id"]):
        ok = result.result.type == "succeeded"
        labels[result.custom_id] = parse_label(result.result.message) if ok else None
    for s, df in frames.items():
        todo = [i for i in range(len(df)) if labels.get(f"{s}-{i}") is None]
        for i in todo:  # batch failures / refusals / unparsable: retry directly
            r = df.iloc[i]
            labels[f"{s}-{i}"] = parse_label(
                client.messages.create(**request_params(r.agent_turn, r.text))
            )
        df["completeness"] = [labels.get(f"{s}-{i}") or "unlabelled" for i in range(len(df))]
        df.to_parquet(pauses_dir / f"{s}.parquet", index=False)
        console.print(
            f"[green]{s}[/] ({len(todo)} retried): {df['completeness'].value_counts().to_dict()}"
        )
        if repo_id:
            from datasets import Dataset

            Dataset.from_pandas(df, preserve_index=False).push_to_hub(
                repo_id, split=s, private=True
            )
            console.print(f"pushed {s} to {repo_id}")


@app.command("evaluate")
def evaluate_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="Trained model dir or Hub ID")],
    overrides: Annotated[
        list[str] | None,
        typer.Argument(
            help="Hydra overrides picking the manifest (default: +experiment=eval, the neutral pool)"
        ),
    ] = None,
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
    context: Annotated[
        str,
        typer.Option(
            "--context",
            help="Agent question as context: 'pool' (manifest's own mix), 'always', or 'never'",
        ),
    ] = "pool",
):
    """Score end-of-turn firing (P/R/F1, per kind) and transcript drift on a manifest.

    The manifest is data.pool_dir/{split}.parquet from the resolved config --
    build it first with `build-pool` and the same overrides. Also splits the
    scores by whether the agent's question was given as context, and sweeps a
    confidence threshold on the marker: fire only when
    logit(<END_OF_TURN>) - logit(<|im_end|>) > tau at the end of the transcript.
    """
    import pandas as pd

    from scripts.eval.audio import TextNormalizer
    from scripts.turn_aware.config import load_config
    from scripts.turn_aware.data import (
        SWEEP_METRICS,
        TurnAudioStore,
        marker_metrics,
        predict_rows,
        stratified_subset,
        threshold_sweep,
        with_context,
    )
    from tiny_audio.turns import load_model, load_processor

    if context not in ("pool", "always", "never"):
        raise typer.BadParameter("must be pool, always or never", param_hint="--context")
    cfg = load_config(overrides or ["+experiment=eval"])
    manifest = Path(cfg.data.pool_dir) / f"{split}.parquet"
    rows = stratified_subset(pd.read_parquet(manifest).to_dict("records"), max_samples)
    processor = load_processor(model)
    net = load_model(model).eval()
    store = TurnAudioStore(split, cfg.data.dataset_id)
    rows = with_context(rows, context, store.meta)
    preds = predict_rows(net, processor, rows, store, batch_size, progress=True)
    fired = [p.fired for p in preds]
    margins = [p.margin for p in preds]

    metrics = marker_metrics(rows, fired, [p.text for p in preds], TextNormalizer().normalize)
    _print_columns(f"{model} on {manifest}", {"value": metrics}, metrics)

    by_context = {"all": metrics}
    for name, has_ctx in (("with context", True), ("no context", False)):
        idx = [i for i, r in enumerate(rows) if bool(r["ctx"]) == has_ctx]
        if idx:
            by_context[name] = marker_metrics([rows[i] for i in idx], [fired[i] for i in idx])
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
        frame["pred_text"] = [p.text for p in preds]
        frame["pred_fired"] = fired
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

    from tiny_audio.turns import END_OF_TURN, load_processor, set_end_of_turn_threshold

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
    from tiny_audio.turns import SAMPLE_RATE, load_model, load_processor, stream_fire_times

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
        padded = np.concatenate(
            [store.get(key), np.zeros(int(tail_s * SAMPLE_RATE), dtype=np.float32)]
        )
        fires = stream_fire_times(
            net, processor, padded, taus, hop_s, gate_s, batch_size, context=ctx
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
