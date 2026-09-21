#!/usr/bin/env python3
"""Compare a fine-tuned model's weights against its base to measure drift."""

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import torch
import typer
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file

from scripts.debug.analyze_weights import load_base_weights
from scripts.debug.base_keys import all_base_key_candidates, resolve_base_key

app = typer.Typer(help="Compare fine-tuned weights against base-model weights")
console = Console()


COMPONENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "embed_tokens": re.compile(r"\.embed_tokens\."),
    "self_attn.q_proj": re.compile(r"\.self_attn\.q_proj\."),
    "self_attn.k_proj": re.compile(r"\.self_attn\.k_proj\."),
    "self_attn.v_proj": re.compile(r"\.self_attn\.v_proj\."),
    "self_attn.o_proj": re.compile(r"\.self_attn\.o_proj\."),
    "self_attn.q_norm": re.compile(r"\.self_attn\.q_norm\."),
    "self_attn.k_norm": re.compile(r"\.self_attn\.k_norm\."),
    # Qwen3.5's linear-attention block. Without these its 18 hybrid layers --
    # 379M params, the largest single group -- collapse into "other".
    "linear_attn.in_proj_qkv": re.compile(r"\.linear_attn\.in_proj_qkv\."),
    "linear_attn.in_proj_z": re.compile(r"\.linear_attn\.in_proj_z\."),
    "linear_attn.in_proj_ab": re.compile(r"\.linear_attn\.in_proj_[ab]\."),
    "linear_attn.out_proj": re.compile(r"\.linear_attn\.out_proj\."),
    "linear_attn.conv1d": re.compile(r"\.linear_attn\.conv1d\."),
    "linear_attn.norm": re.compile(r"\.linear_attn\.norm\."),
    "linear_attn.ssm": re.compile(r"\.linear_attn\.(?:A_log|dt_bias)"),
    "mlp.gate_proj": re.compile(r"\.mlp\.gate_proj\."),
    "mlp.up_proj": re.compile(r"\.mlp\.up_proj\."),
    "mlp.down_proj": re.compile(r"\.mlp\.down_proj\."),
    "input_layernorm": re.compile(r"\.input_layernorm\."),
    "post_attention_layernorm": re.compile(r"\.post_attention_layernorm\."),
    # Optional `language_model.` segment: multimodal bases (Qwen3.5-2B) nest
    # the final norm one level deeper than plain CausalLM bases.
    "model.norm": re.compile(r"^model\.(?:language_model\.)?norm\."),
    "lm_head": re.compile(r"^lm_head\."),
    # ---- Granite Speech conformer encoder ---------------------------------
    # The q/k/v/o patterns above already match the encoder's self-attention,
    # which is what we want: same label, both towers. These cover the parts a
    # conformer block has and a transformer decoder does not. Without them the
    # whole encoder collapses into "other" and the component table says
    # nothing.
    "enc.self_attn.rel_pos_emb": re.compile(r"\.self_attn\.rel_pos_emb\."),
    "enc.conv.depthwise": re.compile(r"\.conv\.depthwise_conv\."),
    "enc.conv.pointwise": re.compile(r"\.conv\.pointwise_lin[12]\."),
    # BatchNorm. `weight`/`bias` are trainable affine params; `running_mean`/
    # `running_var` are BUFFERS and are the reason this label is worth having
    # on its own. _apply_train_mode pins them to the pretrained values
    # whenever the encoder is unfrozen, so a non-zero drift here means that
    # pin is not holding -- a failure invisible to every LR-based check,
    # because buffers carry no gradient and sit in no optimizer group.
    "enc.conv.batchnorm": re.compile(r"\.conv\.norm\."),
    "enc.feed_forward": re.compile(r"\.feed_forward[12]\.linear[12]\."),
    "enc.norms": re.compile(r"\.norm_(?:conv|feed_forward[12]|out|self_att)\."),
    "enc.input_linear": re.compile(r"(?:^|\.)input_linear\."),
    "enc.out": re.compile(r"(?:^|\.)out(?:_mid)?\.(?:weight|bias)$"),
}

LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def classify_component(base_key: str) -> str:
    """Return a component label (e.g. 'self_attn.q_proj') for a base-model key."""
    for label, pattern in COMPONENT_PATTERNS.items():
        if pattern.search(base_key):
            return label
    return "other"


def layer_index(base_key: str) -> int | None:
    match = LAYER_INDEX_RE.search(base_key)
    return int(match.group(1)) if match else None


def compare_tensors(trained: torch.Tensor, base: torch.Tensor) -> dict:
    """Compute drift metrics between two same-shape tensors.

    Both are cast to fp32 for stable norms/cosine.
    """
    t = trained.float()
    b = base.float()
    delta = t - b

    base_norm = b.norm().item()
    delta_norm = delta.norm().item()
    # A zero-norm base tensor is not an error and is not undefined. BatchNorm
    # ships `running_mean` as exact zeros, and once the encoder is comparable
    # at all those buffers are in the diff -- so this branch is now routinely
    # taken. Relative change against zero is 0 when nothing moved and
    # unbounded when something did; returning nan for both made the
    # parameter-weighted aggregate nan and took the whole report with it.
    if base_norm > 0:
        rel_change = delta_norm / base_norm
    else:
        rel_change = 0.0 if delta_norm == 0 else float("inf")

    t_flat = t.flatten()
    b_flat = b.flatten()
    # cosine_similarity against a zero vector is 0 by convention, which reads
    # as "maximally different" when the truthful answer is "identical".
    if base_norm == 0 and delta_norm == 0:
        cos = 1.0
    else:
        cos = torch.nn.functional.cosine_similarity(t_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    return {
        "shape": list(t.shape),
        "numel": t.numel(),
        "base_norm": base_norm,
        "trained_norm": t.norm().item(),
        "delta_norm": delta_norm,
        "rel_change": rel_change,
        "cosine": cos,
        "delta_abs_mean": delta.abs().mean().item(),
        "delta_abs_max": delta.abs().max().item(),
    }


def _weighted(rows, field: str) -> float:
    """Parameter-weighted mean of `field`, ignoring non-finite entries.

    A tensor whose base norm is zero and which then moved has unbounded
    relative change. That is a real and interesting result per-tensor, but a
    single inf makes every aggregate containing it inf, so it is excluded here
    and surfaced separately by the caller.
    """
    finite = [r for r in rows if math.isfinite(r[field])]
    params = sum(r["numel"] for r in finite)
    if not params:
        return float("nan")
    return sum(r[field] * r["numel"] for r in finite) / params


def _drift_severity(rel_change: float) -> str:
    if rel_change < 0.05:
        return "[green]small[/green]"
    if rel_change < 0.15:
        return "[yellow]moderate[/yellow]"
    if rel_change < 0.30:
        return "[orange1]large[/orange1]"
    return "[red]extreme[/red]"


def load_trained_weights(model_id: str) -> dict[str, torch.Tensor] | None:
    """Load a tiny-audio checkpoint from a local path or the Hub.

    The local branch is what makes mid-run drift checks possible at all: the
    checkpoints this is meant to be pointed at (`checkpoint-2000/`, etc.) live
    on the training box and are never pushed. The CLI advertised "or local
    path" while unconditionally calling `hf_hub_download`, so a path argument
    failed as a malformed repo id.
    """
    local = Path(model_id)
    if local.exists():
        candidate = local / "model.safetensors" if local.is_dir() else local
        if not candidate.exists():
            console.print(f"[red]No model.safetensors under {local}[/red]")
            return None
        return load_file(str(candidate))
    try:
        return load_file(hf_hub_download(repo_id=model_id, filename="model.safetensors"))
    except Exception as e:
        console.print(f"[red]Error downloading weights: {e}[/red]")
        return None


def compare_to_base(
    trained_id: str,
    base_id: str,
    show_per_layer: bool = False,
    top_k: int = 15,
    previous_id: str | None = None,
) -> bool:
    console.rule(f"[bold]Drift Comparison: {trained_id} vs {base_id}[/bold]")

    trained_weights = load_trained_weights(trained_id)
    if trained_weights is None:
        return False

    previous_weights: dict[str, torch.Tensor] | None = None
    if previous_id:
        previous_weights = load_trained_weights(previous_id)
        if previous_weights is None:
            return False
    # Shared with analyze-weights for its single-file -> sharded-index fallback.
    # Requesting a bare `model.safetensors` fails outright on bases that ship
    # shards, which is how Qwen3.5-2B is published.
    base_weights = load_base_weights(base_id)
    if not base_weights:
        return False

    matched: dict[str, dict] = {}
    unmatched_trained: list[str] = []
    unmatched_base: set[str] = set(base_weights.keys())

    segments: list[dict] = []

    for tk in trained_weights:
        # Empty candidates = not a decoder or encoder tensor (projector, etc.)
        # and not expected in the base. Candidates that all miss = a tensor the
        # base genuinely lacks, which is worth reporting.
        if not all_base_key_candidates(tk):
            continue
        bk = resolve_base_key(tk, base_weights)
        if bk is None:
            unmatched_trained.append(tk)
            continue
        t_tensor = trained_weights[tk]
        b_tensor = base_weights[bk]
        # Integer buffers (BatchNorm's num_batches_tracked, rotary caches)
        # are counters, not weights: a norm ratio and a cosine on a 0-d int64
        # are meaningless and pollute the parameter-weighted aggregates.
        if not t_tensor.is_floating_point():
            unmatched_base.discard(bk)
            continue
        if tuple(t_tensor.shape) != tuple(b_tensor.shape):
            console.print(
                f"[yellow]Shape mismatch on {bk}: trained={list(t_tensor.shape)} base={list(b_tensor.shape)} — skipping[/yellow]"
            )
            unmatched_base.discard(bk)
            continue
        matched[bk] = compare_tensors(t_tensor, b_tensor)
        unmatched_base.discard(bk)

        # Segment cosine: cos(W_now - W_prev, W_prev - W_base). Drift
        # magnitude alone cannot tell a diffusing tower from an adapting one
        # -- that conflation is what put granite_qwen_full's encoder LR 2.4x
        # too low. Successive displacement vectors that are statistically
        # orthogonal mean gradient noise dominates gradient signal (the LR is
        # below the useful band); a persistently positive cosine with drift
        # growing superlinearly means the tower is being rewritten.
        if previous_weights is not None and tk in previous_weights:
            p_tensor = previous_weights[tk]
            if tuple(p_tensor.shape) == tuple(t_tensor.shape):
                seg_now = (t_tensor.float() - p_tensor.float()).flatten()
                seg_prev = (p_tensor.float() - b_tensor.float()).flatten()
                n_now, n_prev = seg_now.norm().item(), seg_prev.norm().item()
                if n_now > 0 and n_prev > 0:
                    segments.append(
                        {
                            "key": bk,
                            "numel": t_tensor.numel(),
                            "cosine": torch.nn.functional.cosine_similarity(
                                seg_now.unsqueeze(0), seg_prev.unsqueeze(0)
                            ).item(),
                            "seg_now": n_now,
                            "seg_prev": n_prev,
                        }
                    )

    console.print(f"  Matched tensors:        {len(matched)}")
    console.print(f"  Trained-only tensors:   {len(unmatched_trained)}")
    console.print(f"  Base-only tensors:      {len(unmatched_base)}")
    if unmatched_base:
        for k in sorted(unmatched_base):
            console.print(f"    [dim]base-only: {k}[/dim]")

    if not matched:
        console.print("[red]No matched tensors — abort.[/red]")
        return False

    console.rule("[bold]OVERALL DRIFT (parameter-weighted)[/bold]")

    rows_all = list(matched.values())
    total_params = sum(s["numel"] for s in rows_all)
    weighted_rel = _weighted(rows_all, "rel_change")
    weighted_cos = _weighted(rows_all, "cosine")
    finite_rel = [s["rel_change"] for s in rows_all if math.isfinite(s["rel_change"])]
    max_rel = max(finite_rel) if finite_rel else float("nan")
    min_cos = min(s["cosine"] for s in rows_all)
    unbounded = [k for k, s in matched.items() if not math.isfinite(s["rel_change"])]

    console.print(f"\n  Parameters compared:    {total_params:,}")
    console.print(
        f"  Weighted relative drift: {weighted_rel:>7.2%}  {_drift_severity(weighted_rel)}"
    )
    console.print(f"  Weighted cosine sim:     {weighted_cos:>7.4f}")
    console.print(f"  Max per-tensor drift:    {max_rel:>7.2%}")
    console.print(f"  Min per-tensor cosine:   {min_cos:>7.4f}")
    if unbounded:
        # Almost always a BatchNorm running_mean that left its all-zero base:
        # i.e. the statistics pin in _apply_train_mode is not holding.
        console.print(
            f"  [orange1]Moved off a zero-norm base: {len(unbounded)} tensor(s) "
            f"(excluded from the weighted means)[/orange1]"
        )
        for k in sorted(unbounded)[:5]:
            console.print(f"    [dim]{k}[/dim]")

    if segments:
        console.rule(f"[bold]SEGMENT COSINE vs {previous_id}[/bold]")
        seg_params = sum(s["numel"] for s in segments)
        seg_cos = sum(s["cosine"] * s["numel"] for s in segments) / seg_params
        path = sum(s["seg_now"] + s["seg_prev"] for s in segments)
        net = sum(
            (
                s["seg_now"] ** 2
                + s["seg_prev"] ** 2
                + 2 * s["cosine"] * s["seg_now"] * s["seg_prev"]
            )
            ** 0.5
            for s in segments
        )
        console.print(f"\n  Tensors with both segments: {len(segments):,}")
        console.print(f"  Weighted segment cosine:    {seg_cos:>7.4f}")
        console.print(f"  Net / path travelled:       {net / path:>7.4f}")
        # sqrt(2)/2 = 0.707 is the net/path of a pure two-step random walk.
        if seg_cos < 0.05:
            console.print(
                "  [yellow]Successive segments are ~orthogonal: this is diffusion, "
                "not learning.[/yellow]\n"
                "     The LR for this tower is likely below the useful band."
            )
        elif seg_cos > 0.5:
            console.print(
                "  [orange1]Successive segments strongly aligned: the tower is "
                "travelling in a consistent direction.[/orange1]\n"
                "     Healthy if drift is decelerating; check the magnitude if not."
            )
        else:
            console.print("  [green]Mixed alignment -- typical of healthy adaptation.[/green]")

    console.rule("[bold]DRIFT BY COMPONENT[/bold]")

    by_comp: dict[str, list[dict]] = defaultdict(list)
    for bk, stats in matched.items():
        by_comp[classify_component(bk)].append({**stats, "key": bk})

    table = Table(show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Tensors", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Mean rel Δ", justify="right")
    table.add_column("Max rel Δ", justify="right")
    table.add_column("Mean cos", justify="right")
    table.add_column("Min cos", justify="right")
    table.add_column("Severity")

    comp_summary: list[tuple[str, float]] = []
    for comp in sorted(by_comp):
        rows = by_comp[comp]
        params = sum(r["numel"] for r in rows)
        mean_rel = _weighted(rows, "rel_change")
        finite_c = [r["rel_change"] for r in rows if math.isfinite(r["rel_change"])]
        max_rel_c = max(finite_c) if finite_c else float("inf")
        mean_cos = _weighted(rows, "cosine")
        min_cos_c = min(r["cosine"] for r in rows)
        comp_summary.append((comp, mean_rel))
        table.add_row(
            comp,
            str(len(rows)),
            f"{params:,}",
            f"{mean_rel:.2%}",
            f"{max_rel_c:.2%}",
            f"{mean_cos:.4f}",
            f"{min_cos_c:.4f}",
            _drift_severity(mean_rel),
        )

    console.print(table)

    console.rule(f"[bold]TOP {top_k} MOST-DRIFTED TENSORS[/bold]")

    ranked = sorted(matched.items(), key=lambda kv: -kv[1]["rel_change"])[:top_k]
    table = Table(show_header=True, header_style="bold")
    table.add_column("Tensor", style="cyan", overflow="fold")
    table.add_column("Shape", justify="right")
    table.add_column("Rel Δ", justify="right")
    table.add_column("Cosine", justify="right")
    table.add_column("ΔL2", justify="right")
    table.add_column("|Δ| max", justify="right")
    for key, s in ranked:
        table.add_row(
            key,
            str(s["shape"]),
            f"{s['rel_change']:.2%}",
            f"{s['cosine']:.4f}",
            f"{s['delta_norm']:.4f}",
            f"{s['delta_abs_max']:.4f}",
        )
    console.print(table)

    if show_per_layer:
        console.rule("[bold]PER-LAYER DRIFT (attention vs MLP, weighted by params)[/bold]")
        by_layer: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for bk, stats in matched.items():
            li = layer_index(bk)
            if li is None:
                continue
            comp = classify_component(bk)
            # `linear_attn` belongs in `attn`: on a hybrid decoder like Qwen3.5
            # it IS the attention mechanism for 18 of 24 layers. Without it
            # those layers fall to `other`, which this table never prints --
            # 379.1M params, the single largest group, silently rendered as
            # "—". Same failure class the component table above already fixed.
            #
            # Order matters and matches `analyze_weights._decoder_module`:
            # attention is tested before norm, so `self_attn.q_norm` and
            # `linear_attn.norm` group with attention while the standalone
            # `input_layernorm` / `post_attention_layernorm` still land in
            # `norm`. Keeping the two scripts consistent is the point.
            if comp.startswith(("self_attn", "linear_attn")):
                bucket = "attn"
            elif comp.startswith("mlp"):
                bucket = "mlp"
            elif "layernorm" in comp or "norm" in comp:
                bucket = "norm"
            else:
                bucket = "other"
            by_layer[li][bucket].append(stats)

        table = Table(show_header=True, header_style="bold")
        table.add_column("Layer", justify="right")
        table.add_column("Attn rel Δ", justify="right")
        table.add_column("MLP rel Δ", justify="right")
        table.add_column("Norm rel Δ", justify="right")
        for li in sorted(by_layer):
            cells = [str(li)]
            for bucket in ("attn", "mlp", "norm"):
                rows = by_layer[li][bucket]
                if not rows:
                    cells.append("—")
                    continue
                params = sum(r["numel"] for r in rows)
                rel = _weighted(rows, "rel_change")
                cells.append(f"{rel:.2%}")
            table.add_row(*cells)
        console.print(table)

    console.rule("[bold]VERDICT[/bold]")

    if weighted_rel < 0.02:
        console.print("  [yellow]🟡 LM barely moved (<2% weighted drift).[/yellow]")
        console.print("     Either training was very short, decoder LR is too low,")
        console.print("     or the projector's gradient through the frozen forward path is small.")
    elif weighted_rel < 0.10:
        console.print("  [green]🟢 Moderate drift (2-10%).[/green]")
        console.print("     Typical of healthy adapter-style fine-tuning.")
    elif weighted_rel < 0.20:
        console.print("  [orange1]🟠 Significant drift (10-20%).[/orange1]")
        console.print("     Watch for catastrophic forgetting on general-LM behavior.")
    else:
        console.print("  [red]🔴 Heavy drift (>20%).[/red]")
        console.print("     LM has substantially specialized; verify it still")
        console.print("     handles non-ASR inputs if that matters for your use case.")

    return True


@app.command()
def main(
    model: Annotated[
        str,
        typer.Argument(help="Fine-tuned HuggingFace model ID (or local path)"),
    ] = "mazesmazes/tiny-audio-embedded-2",
    base_model: Annotated[
        str,
        typer.Option("--base-model", help="Base model ID to compare against"),
    ] = "Qwen/Qwen3-0.6B",
    per_layer: Annotated[
        bool,
        typer.Option("--per-layer", help="Show per-layer attn/mlp/norm drift table"),
    ] = False,
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Show top-K most-drifted tensors"),
    ] = 15,
    previous: Annotated[
        str | None,
        typer.Option(
            "--previous",
            help=(
                "Earlier checkpoint (path or repo id). Adds the segment cosine, "
                "which separates a diffusing tower from an adapting one."
            ),
        ),
    ] = None,
):
    """Compare a fine-tuned model against its base, reporting drift metrics.

    Works on the encoder as well as the decoder. For the encoder-drift check
    granite_qwen_full prescribes, point `--base-model` at the audio base and
    read the `enc.*` rows plus the segment cosine:

        ta debug compare-to-base out/checkpoint-4000 \\
          --base-model ibm-granite/granite-speech-5.0-470m-turboctc \\
          --previous out/checkpoint-2000
    """
    success = compare_to_base(
        model, base_model, show_per_layer=per_layer, top_k=top_k, previous_id=previous
    )
    raise typer.Exit(0 if success else 1)


if __name__ == "__main__":
    app()
