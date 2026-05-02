# scripts/publish_mlx_bundle.py
"""Publish a tiny-audio MLX bundle to the Hugging Face Hub.

Reads an existing tiny-audio embedded checkpoint (PT-format projector + config),
runs the encoder weight-load + 4-bit quantize step in Python, mirrors the
Qwen3-MLX-4bit decoder weights, writes a manifest with SHA256 + format version,
and pushes everything to a target Hub repo.

Usage:
    python scripts/publish_mlx_bundle.py \\
        --source-repo mazesmazes/tiny-audio-embedded \\
        --target-repo mazesmazes/tiny-audio-mlx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from huggingface_hub import HfApi, snapshot_download
from mlx.utils import tree_flatten

from tiny_audio.mlx import MLXASRModel

# Bump this when bundle layout changes incompatibly.
MLX_FORMAT_VERSION = 1

# Renames applied when mirroring files from the upstream decoder repo.
# Keys are upstream filenames; values are the names used inside the bundle.
DECODER_FILE_RENAMES = {
    "model.safetensors": "decoder.safetensors",
    "config.json": "decoder_config.json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_module_safetensors(module, out_path: Path) -> None:
    """Flatten a Module's parameters and save as a single safetensors file."""
    flat = dict(tree_flatten(module.parameters()))
    mx.save_safetensors(str(out_path), flat)


def build_bundle(work: Path, source_repo: str, decoder_repo: str) -> None:
    """Build the MLX bundle into *work*, writing all files except manifest.json last."""
    print(f"Loading source MLX model from {source_repo}...")
    model = MLXASRModel.from_pretrained(source_repo)

    print("Saving encoder.safetensors (4-bit quantized)...")
    _save_module_safetensors(model.encoder, work / "encoder.safetensors")

    print("Saving projector.safetensors (fp16)...")
    _save_module_safetensors(model.projector, work / "projector.safetensors")

    print(f"Mirroring decoder weights from {decoder_repo}...")
    decoder_src = Path(snapshot_download(decoder_repo))
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json"):
        src = decoder_src / name
        if src.exists():
            shutil.copy(src, work / DECODER_FILE_RENAMES.get(name, name))

    missing = []
    if not (work / "decoder.safetensors").exists():
        missing.append("decoder.safetensors (from upstream model.safetensors)")
    if not (work / "decoder_config.json").exists():
        missing.append("decoder_config.json (from upstream config.json)")
    if missing:
        upstream_files = sorted(p.name for p in decoder_src.iterdir() if p.is_file())
        raise FileNotFoundError(
            f"Decoder mirror is missing required files: {missing}. "
            f"Upstream {decoder_repo} contains: {upstream_files}. "
            f"Sharded weights (model-XXXXX-of-NNNNN.safetensors) are not yet supported by this script."
        )

    encoder_cfg = {
        "encoder_dim": int(model.encoder.cfg.hidden_size),
        "n_mels": int(model.encoder.cfg.n_mels),
        "num_layers": int(model.encoder.cfg.num_layers),
        "num_heads": int(model.encoder.cfg.num_heads),
        "head_dim": int(model.encoder.cfg.head_dim),
        "intermediate_size": int(model.encoder.cfg.intermediate_size),
        "rope_theta": float(model.encoder.cfg.rope_theta),
    }
    bundle_config = {
        "mlx_format_version": MLX_FORMAT_VERSION,
        "encoder": encoder_cfg,
        "projector": {
            "encoder_dim": encoder_cfg["encoder_dim"],
            "llm_dim": int(model.projector.linear_2.weight.shape[0]),
            "hidden_dim": int(model.projector.linear_1.weight.shape[0]),
            "pool_stride": int(model.projector.k),
        },
        "audio_token": "<audio>",
        "audio_token_id": int(model.audio_token_id),
        "eos_token_ids": sorted(int(t) for t in model.eos_token_ids),
        "encoder_conv_layers": [[1, 3, 1], [1, 3, 2]],
        "hop_length": 160,
    }
    (work / "config.json").write_text(json.dumps(bundle_config, indent=2))

    files_for_manifest = [
        p for p in work.iterdir() if p.is_file() and p.name not in ("manifest.json",)
    ]
    manifest = {
        "format_version": MLX_FORMAT_VERSION,
        "files": {
            p.name: {"sha256": _sha256(p), "size": p.stat().st_size} for p in files_for_manifest
        },
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default="mazesmazes/tiny-audio-embedded")
    parser.add_argument("--target-repo", default="mazesmazes/tiny-audio-mlx")
    parser.add_argument("--decoder-repo", default="Qwen/Qwen3-0.6B-MLX-4bit")
    parser.add_argument("--dry-run", action="store_true", help="Build locally; skip push.")
    args = parser.parse_args()

    if args.dry_run:
        work = Path(tempfile.mkdtemp(prefix="tiny-audio-mlx-"))
        print(f"Building bundle in {work}")
        build_bundle(work, args.source_repo, args.decoder_repo)
        print(f"Dry run — bundle prepared at {work}. Skipping push.")
        return

    with tempfile.TemporaryDirectory(prefix="tiny-audio-mlx-") as work_str:
        work = Path(work_str)
        print(f"Building bundle in {work}")
        build_bundle(work, args.source_repo, args.decoder_repo)
        api = HfApi()
        api.create_repo(args.target_repo, exist_ok=True)
        print(f"Uploading to {args.target_repo}...")
        api.upload_folder(folder_path=str(work), repo_id=args.target_repo)
        print(f"Done. Bundle published to https://huggingface.co/{args.target_repo}")


if __name__ == "__main__":
    main()
