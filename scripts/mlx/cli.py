"""CLI for MLX bundle build utilities."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from scripts.utils import get_project_root

app = typer.Typer(help="MLX bundle build utilities", no_args_is_help=True)

_DEFAULT_DECODER_CACHE = get_project_root() / ".cache" / "decoder-mlx"
_DEFAULT_BUNDLE_DIR = get_project_root() / "swift/Sources/TinyAudio/Resources/Model"
_STOCK_DECODER_REPO = "Qwen/Qwen3-0.6B-MLX-4bit"
_DEFAULT_CHECKPOINT = "mazesmazes/tiny-audio-embedded"


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
                "Local mlx-lm directory or HF repo id. "
                f"Defaults to the convert-decoder cache ({_DEFAULT_DECODER_CACHE}) if it exists, "
                f"else the stock {_STOCK_DECODER_REPO}."
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
    q_bits: Annotated[int, typer.Option("--q-bits", help="Bits per encoder weight.")] = 4,
    q_group_size: Annotated[
        int, typer.Option("--q-group-size", help="Encoder quantization group size.")
    ] = 64,
) -> None:
    """Assemble the Swift SDK's MLX bundle from projector + decoder + upstream encoder."""
    from scripts.mlx.build_bundle import build_bundle

    if decoder is None:
        decoder = (
            str(_DEFAULT_DECODER_CACHE) if _DEFAULT_DECODER_CACHE.is_dir() else _STOCK_DECODER_REPO
        )
        typer.echo(f"Using decoder: {decoder}")

    build_bundle(
        projector_repo=projector,
        encoder_repo=encoder,
        decoder=decoder,
        output_dir=output_dir,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )
