"""Extract a fine-tuned tiny-audio decoder and convert to MLX 4-bit (local output).

Run once per fine-tuned projector checkpoint (i.e. trained with
`freeze_language_model: false`). Produces a self-contained mlx-lm directory
ready to be passed to `ta mlx build-bundle --decoder <out_dir>`.

The actual HF -> MLX conversion + 4-bit quantization is done by
`mlx_lm.convert`. This module's only job is the tiny-audio-specific step
of materializing the fine-tuned `language_model` (with the resized
`<audio>` vocab) as a clean Qwen3 directory that mlx_lm.convert can read.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path


def _patch_config_for_mlx_lm(hf_dir: Path) -> None:
    """Hoist `rope_parameters.rope_theta` to a top-level `rope_theta`.

    transformers v5 nests rope params under `rope_parameters`, but mlx_lm's
    Qwen3 `ModelArgs` requires `rope_theta` as a flat required field.
    """
    cfg_path = hf_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if "rope_theta" not in cfg:
        rope_params = cfg.get("rope_parameters") or {}
        cfg["rope_theta"] = float(rope_params.get("rope_theta", 10000.0))
        cfg_path.write_text(json.dumps(cfg, indent=2))


def convert_decoder(
    checkpoint: str,
    out_dir: Path,
    *,
    q_bits: int = 4,
    q_group_size: int = 64,
    q_mode: str = "affine",
    quantize: bool = True,
) -> None:
    from mlx_lm import convert as mlx_lm_convert

    from tiny_audio.asr_modeling import ASRModel

    print(f"Loading tiny-audio checkpoint from {checkpoint}...")
    model = ASRModel.from_pretrained(checkpoint)

    if getattr(model.config, "freeze_language_model", True) and not hasattr(
        model.language_model, "peft_config"
    ):
        print(
            f"Note: {checkpoint} was trained with freeze_language_model=true — "
            f"the LM weights match the upstream base model. Converting anyway to "
            f"keep the bundle's decoder quantization consistent across "
            f"frozen-LM and fine-tuned-LM checkpoints."
        )

    # Bake any LoRA adapters into the base LM so the exported HF directory is a
    # plain Qwen3 (or whatever) checkpoint that mlx_lm.convert can read. PeftModel's
    # own save_pretrained only writes adapter_model.safetensors + adapter_config.json,
    # which is not a full HF model.
    if hasattr(model.language_model, "peft_config"):
        print("Merging LoRA adapters into base LM...")
        model.language_model = model.language_model.merge_and_unload()

    out_dir = Path(out_dir)
    if out_dir.exists():
        # mlx_lm.convert refuses to overwrite an existing path; clear it so reruns are idempotent.
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tiny-audio-decoder-hf-") as hf_str:
        hf_dir = Path(hf_str)

        print(f"Saving language_model + tokenizer to {hf_dir}...")
        model.language_model.save_pretrained(str(hf_dir))
        model.tokenizer.save_pretrained(str(hf_dir))
        _patch_config_for_mlx_lm(hf_dir)

        if quantize:
            print(
                f"Converting + quantizing ({q_mode}, {q_bits}-bit, group={q_group_size}) -> {out_dir}..."
            )
            mlx_lm_convert(
                hf_path=str(hf_dir),
                mlx_path=str(out_dir),
                quantize=True,
                q_bits=q_bits,
                q_group_size=q_group_size,
                q_mode=q_mode,
            )
        else:
            print(f"Converting (fp16, no quantization) -> {out_dir}...")
            mlx_lm_convert(
                hf_path=str(hf_dir),
                mlx_path=str(out_dir),
                quantize=False,
            )

    print(f"Done. MLX-LM decoder written to {out_dir}")
