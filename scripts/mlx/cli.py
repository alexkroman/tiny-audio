"""CLI for MLX bundle build utilities."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from scripts.utils import get_project_root

app = typer.Typer(help="MLX bundle build utilities", no_args_is_help=True)

_DEFAULT_DECODER_CACHE = get_project_root() / ".cache" / "decoder-mlx"
_DEFAULT_BUNDLE_DIR = get_project_root() / "swift/Sources/TinyAudio/Resources/Model"
_STOCK_DECODER_REPO = "Qwen/Qwen3-0.6B-MLX-4bit"
_DEFAULT_CHECKPOINT = "mazesmazes/tiny-audio-embedded"


def _projector_needs_converted_decoder(repo_or_path: str) -> bool:
    """A converted decoder is needed when the projector checkpoint shipped its
    own fine-tuned LM weights (`freeze_language_model=false`) or LoRA adapters
    (`use_lora=true`). Pure projector-only training can reuse the stock MLX
    decoder."""
    from huggingface_hub import hf_hub_download

    p = Path(repo_or_path)
    if p.is_dir():
        cfg_path = p / "config.json"
    else:
        cfg_path = Path(hf_hub_download(repo_or_path, "config.json"))
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("freeze_language_model", True):
        return True
    return bool(cfg.get("use_lora", False))


def _resolve_default_decoder(projector: str) -> str:
    """Pick the decoder source for a build-bundle invocation that didn't pass
    --decoder. Caches per-projector under .cache/decoder-mlx/<sanitized-repo>
    and auto-runs convert-decoder for fine-tuned/LoRA checkpoints so the
    bundle's decoder matches the projector it was trained against."""
    if not _projector_needs_converted_decoder(projector):
        return _STOCK_DECODER_REPO

    sanitized = projector.replace("/", "--")
    cache_path = _DEFAULT_DECODER_CACHE / sanitized
    if not cache_path.is_dir():
        from scripts.mlx.convert_decoder import convert_decoder

        typer.echo(f"No cached decoder for {projector}; running convert-decoder...")
        convert_decoder(checkpoint=projector, out_dir=cache_path, q_bits=8)
    return str(cache_path)


@app.command("convert-decoder")
def convert_decoder_cmd(
    checkpoint: Annotated[
        str,
        typer.Option(
            "--checkpoint",
            "-c",
            help="HF repo id or local path of a tiny-audio checkpoint with full LM weights.",
        ),
    ] = _DEFAULT_CHECKPOINT,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            "-o",
            help="Where to write the MLX-LM 4-bit decoder. `build-bundle` reads this path by default.",
        ),
    ] = _DEFAULT_DECODER_CACHE,
    q_bits: Annotated[
        int,
        typer.Option(
            "--q-bits",
            help=(
                "Bits per weight. Default 8: full-decoder fine-tuning tightens weight "
                "distributions enough that 4-bit affine quant degrades EOS prediction "
                "(over-generation, repetition). 8-bit recovers PT-equivalent WER."
            ),
        ),
    ] = 8,
    q_group_size: Annotated[
        int, typer.Option("--q-group-size", help="Quantization group size.")
    ] = 64,
    q_mode: Annotated[
        str,
        typer.Option(
            "--q-mode",
            help=(
                "Quantization mode: affine (default), mxfp4, mxfp8, nvfp4. "
                "mxfp4/nvfp4 hit an mlx-swift bug today (biases-not-null); use affine."
            ),
        ),
    ] = "affine",
) -> None:
    """Extract the fine-tuned LM from a tiny-audio checkpoint and convert to MLX 4-bit (local)."""
    from scripts.mlx.convert_decoder import convert_decoder

    convert_decoder(
        checkpoint=checkpoint,
        out_dir=out_dir,
        q_bits=q_bits,
        q_group_size=q_group_size,
        q_mode=q_mode,
    )


@app.command("build-bundle")
def build_bundle_cmd(
    projector: Annotated[
        str,
        typer.Option("--projector", "-p", help="HF repo id of the trained projector checkpoint."),
    ],
    decoder: Annotated[
        Optional[str],
        typer.Option(
            "--decoder",
            "-d",
            help=(
                "Local mlx-lm directory or HF repo id. Defaults to a per-projector "
                f"cache under {_DEFAULT_DECODER_CACHE}/<repo>, auto-populated by "
                f"convert-decoder when the projector was trained with a fine-tuned "
                f"LM or LoRA adapters. Falls back to the stock {_STOCK_DECODER_REPO} "
                f"for projector-only checkpoints."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Where to write the bundle."),
    ] = _DEFAULT_BUNDLE_DIR,
    encoder: Annotated[
        str,
        typer.Option("--encoder", help="HF repo id of the upstream PT encoder."),
    ] = "zai-org/GLM-ASR-Nano-2512",
    q_bits: Annotated[int, typer.Option("--q-bits", help="Bits per encoder weight.")] = 8,
    q_group_size: Annotated[
        int, typer.Option("--q-group-size", help="Encoder quantization group size.")
    ] = 64,
) -> None:
    """Assemble the Swift SDK's MLX bundle from projector + decoder + upstream encoder."""
    from scripts.mlx.build_bundle import build_bundle

    if decoder is None:
        decoder = _resolve_default_decoder(projector)
        typer.echo(f"Using decoder: {decoder}")

    build_bundle(
        projector_repo=projector,
        encoder_repo=encoder,
        decoder=decoder,
        output_dir=output_dir,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )
