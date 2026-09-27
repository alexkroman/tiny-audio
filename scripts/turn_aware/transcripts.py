"""The base-model transcript cache, locally (JSONL) and on the Hub.

`build-pool`'s self-distillation pass -- the base model transcribing every
trimmed prefix, ~100k clips, about an hour of GPU -- is the one expensive
artifact in the pipeline; manifests rebuild from it in minutes. The cache is
`transcripts-{split}.jsonl` keyed by observation `key`. Mirroring it to a
private Hub dataset (`pool.transcript_repo`) means a fresh pod never
re-transcribes: build-pool pulls before transcribing and pushes what it adds.

Rows carry the `target_model` that wrote them, and pulls keep only rows from
the configured model, since a transcript is only a valid target for the
model that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_cache(path: Path) -> dict[str, str]:
    """key -> transcript from a JSONL cache, skipping a line cut off by a kill.

    Re-terminates a file whose last write was cut off, so the next appended
    record starts on its own line.
    """
    if not path.exists():
        return {}
    done: dict[str, str] = {}
    raw = path.read_text()
    for line in raw.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # that clip is simply redone
        done[rec["key"]] = rec["text"]
    if raw and not raw.endswith("\n"):
        with path.open("a") as fh:
            fh.write("\n")
    return done


def merge_into_cache(path: Path, records: dict[str, str]) -> int:
    """Append the records whose key the cache lacks; return how many were added."""
    have = read_cache(path)
    new = {k: v for k, v in records.items() if k not in have}
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for key, text in new.items():
                fh.write(json.dumps({"key": key, "text": text}) + "\n")
    return len(new)


def pull(repo_id: str, split: str, path: Path, target_model: str) -> int:
    """Merge `repo_id`'s transcripts for `split` into the local cache; 0 if none exist yet."""
    from datasets import load_dataset

    try:
        ds = load_dataset(repo_id, split=split)
    except Exception:  # repo or split not created yet (first run) -- nothing to pull
        return 0
    ds = ds.filter(lambda m: m == target_model, input_columns="target_model")
    return merge_into_cache(path, dict(zip(ds["key"], ds["text"], strict=True)))


def push(repo_id: str, split: str, path: Path, target_model: str) -> int:
    """Upload the whole local cache for `split` (replacing the Hub copy); return rows pushed."""
    from datasets import Dataset

    done = read_cache(path)
    Dataset.from_dict(
        {"key": list(done), "text": list(done.values()), "target_model": [target_model] * len(done)}
    ).push_to_hub(repo_id, split=split, private=True)
    return len(done)
