#!/usr/bin/env python3
"""Compare a fine-tuned model's weights against its base to measure drift."""

import re
from collections import defaultdict
from typing import Annotated

import torch
import typer
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file

from scripts.debug.analyze_weights import load_base_weights
from scripts.debug.base_keys import base_key_candidates, resolve_base_key

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
    rel_change = delta_norm / base_norm if base_norm > 0 else float("nan")

    t_flat = t.flatten()
    b_flat = b.flatten()
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


def _drift_severity(rel_change: float) -> str:
    if rel_change < 0.05:
        return "[green]small[/green]"
    if rel_change < 0.15:
        return "[yellow]moderate[/yellow]"
    if rel_change < 0.30:
        return "[orange1]large[/orange1]"
    return "[red]extreme[/red]"


def compare_to_base(
    trained_id: str,
    base_id: str,
    show_per_layer: bool = False,
    top_k: int = 15,
) -> bool:
    console.rule(f"[bold]Drift Comparison: {trained_id} vs {base_id}[/bold]")

    try:
        trained_path = hf_hub_download(repo_id=trained_id, filename="model.safetensors")
    except Exception as e:
        console.print(f"[red]Error downloading weights: {e}[/red]")
        return False

    trained_weights = load_file(trained_path)
    # Shared with analyze-weights for its single-file -> sharded-index fallback.
    # Requesting a bare `model.safetensors` fails outright on bases that ship
    # shards, which is how Qwen3.5-2B is published.
    base_weights = load_base_weights(base_id)
    if not base_weights:
        return False

    matched: dict[str, dict] = {}
    unmatched_trained: list[str] = []
    unmatched_base: set[str] = set(base_weights.keys())

    for tk in trained_weights:
        # Empty candidates = not a decoder tensor (projector, etc.) and not
        # expected in the base. Candidates that all miss = a decoder tensor the
        # base genuinely lacks, which is worth reporting.
        if not base_key_candidates(tk):
            continue
        bk = resolve_base_key(tk, base_weights)
        if bk is None:
            unmatched_trained.append(tk)
            continue
        t_tensor = trained_weights[tk]
        b_tensor = base_weights[bk]
        if tuple(t_tensor.shape) != tuple(b_tensor.shape):
            console.print(
                f"[yellow]Shape mismatch on {bk}: trained={list(t_tensor.shape)} base={list(b_tensor.shape)} — skipping[/yellow]"
            )
            unmatched_base.discard(bk)
            continue
        matched[bk] = compare_tensors(t_tensor, b_tensor)
        unmatched_base.discard(bk)

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

    total_params = sum(s["numel"] for s in matched.values())
    weighted_rel = sum(s["rel_change"] * s["numel"] for s in matched.values()) / total_params
    weighted_cos = sum(s["cosine"] * s["numel"] for s in matched.values()) / total_params
    max_rel = max(s["rel_change"] for s in matched.values())
    min_cos = min(s["cosine"] for s in matched.values())

    console.print(f"\n  Parameters compared:    {total_params:,}")
    console.print(
        f"  Weighted relative drift: {weighted_rel:>7.2%}  {_drift_severity(weighted_rel)}"
    )
    console.print(f"  Weighted cosine sim:     {weighted_cos:>7.4f}")
    console.print(f"  Max per-tensor drift:    {max_rel:>7.2%}")
    console.print(f"  Min per-tensor cosine:   {min_cos:>7.4f}")

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
        mean_rel = sum(r["rel_change"] * r["numel"] for r in rows) / params
        max_rel_c = max(r["rel_change"] for r in rows)
        mean_cos = sum(r["cosine"] * r["numel"] for r in rows) / params
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
                rel = sum(r["rel_change"] * r["numel"] for r in rows) / params
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
):
    """Compare a fine-tuned model against its base, reporting drift metrics."""
    success = compare_to_base(model, base_model, show_per_layer=per_layer, top_k=top_k)
    raise typer.Exit(0 if success else 1)


if __name__ == "__main__":
    app()
