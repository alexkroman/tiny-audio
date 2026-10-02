"""Publish a built pool to a public Hub dataset, so it is built once.

`build-pool`'s expensive step is self-transcribing every AMI utterance (~1 h
of GPU); the manifests are small (row indices and offsets, no audio). With
`pool.hub_repo` set, build-pool pushes each split it builds and, before
building, pulls a split whose recorded settings match this run's -- so a pool
built on one machine (a laptop, a cheap GPU) is reused by `ta runpod
train-speaker-asr` without re-transcribing.

Repo layout, per split: `{split}.parquet`, `transcripts-{split}.jsonl` and
`signatures/{split}.json` (the settings that built it), plus a README crediting
AMI (CC BY 4.0, which requires attribution). The repo is public: it holds no
audio, only AMI row indices, offsets and base-model transcripts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _download(repo: str, filename: str) -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        return Path(hf_hub_download(repo, filename, repo_type="dataset"))
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def _merge_transcripts(src: Path, cache: Path) -> int:
    """Append rows of `src` whose key `cache` lacks; returns how many."""
    from scripts.turn_aware.transcripts import read_cache

    have = read_cache(cache)
    added = 0
    with cache.open("a") as fh:
        for line in src.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec["key"] not in have:
                fh.write(json.dumps(rec) + "\n")
                have[rec["key"]] = rec["text"]
                added += 1
    return added


def pull(repo: str, split: str, pool_dir: Path, cache: Path, signature: dict) -> str:
    """Fetch `split` from `repo` into pool_dir; returns what happened.

    'pulled': the remote manifest was built with exactly `signature` and is now
    local (with its signature recorded). 'transcripts': settings differ, but
    the transcripts came from the same target model and were merged into the
    cache, so a rebuild skips transcription. 'none': nothing usable.
    """
    from scripts.turn_aware.config import write_signature

    sig_path = _download(repo, f"signatures/{split}.json")
    if sig_path is None:
        return "none"
    remote = json.loads(sig_path.read_text())
    same_model = remote["pool"].get("target_model") == signature["pool"].get("target_model")
    texts = _download(repo, f"transcripts-{split}.jsonl") if same_model else None
    if texts is not None:
        _merge_transcripts(texts, cache)
    if remote != signature:
        return "transcripts" if texts is not None else "none"
    manifest = _download(repo, f"{split}.parquet")
    if manifest is None:
        return "transcripts" if texts is not None else "none"
    shutil.copyfile(manifest, Path(pool_dir) / f"{split}.parquet")
    write_signature(Path(pool_dir), split, signature)
    return "pulled"


CARD = """---
license: cc-by-4.0
language: en
tags: [speaker-diarization, speaker-attributed-asr]
configs:
- config_name: default
  data_files:
{splits}
---

# Speaker-ASR window pool (AMI)

Window manifests for tiny-audio's speaker-attributed ASR recipe
(`ta speaker-asr build-pool`, configs/speaker_asr). Each row of
`{{split}}.parquet` lists the AMI utterances of one window (row index into
`{dataset_id}` `{data_files}`, utterance id, speaker, offset, duration) and the
`<SPK_n>` target; `transcripts-{{split}}.jsonl` holds the base model's
transcript of each utterance; `signatures/{{split}}.json` the settings that
built the split. No audio is stored here.

Derived from the AMI Meeting Corpus (Carletta et al., 2005), CC BY 4.0:
https://groups.inf.ed.ac.uk/ami/corpus/
"""


def push(repo: str, split: str, pool_dir: Path, cache: Path, signature: dict) -> None:
    """Upload one split's manifest, transcripts and signature to a PUBLIC dataset repo."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    # Declared, not inferred: the Hub would otherwise read transcripts-train.jsonl
    # as part of the `train` split alongside train.parquet.
    present = {f.removesuffix(".parquet") for f in api.list_repo_files(repo, repo_type="dataset")
               if f.endswith(".parquet") and "/" not in f}  # fmt: skip
    splits = "\n".join(f"  - split: {s}\n    path: {s}.parquet" for s in sorted(present | {split}))
    card = CARD.format(splits=splits, **signature["source"]).encode()
    ops = [
        CommitOperationAdd("README.md", card),
        CommitOperationAdd(f"{split}.parquet", str(Path(pool_dir) / f"{split}.parquet")),
        CommitOperationAdd(
            f"signatures/{split}.json", json.dumps(signature, indent=2, sort_keys=True).encode()
        ),
    ]
    if cache.exists():
        ops.append(CommitOperationAdd(f"transcripts-{split}.jsonl", str(cache)))
    api.create_commit(repo, ops, repo_type="dataset", commit_message=f"speaker-asr pool: {split}")
