#!/usr/bin/env python3
"""Static analysis of model weights for training health diagnostics."""

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table
from safetensors.torch import load_file

from scripts.debug.base_keys import resolve_base_tensor

app = typer.Typer(help="Analyze model weights for training health")
console = Console()


def _threshold_status(value: float, warn: float, high: float) -> str:
    if value < warn:
        return "✅ OK"
    if value < high:
        return "⚠️  WARNING"
    return "❌ HIGH"


COMPONENT_FILTERS = {
    "projector": "projector",
    "decoder": "language_model",
    "encoder": "audio_tower",
    "all": "",
}


def _tensor_kind(name: str, shape: tuple) -> str:
    """Classify tensor for type-aware analysis.

    Returns one of: 'norm', 'embed', 'lm_head', 'bias', 'linear', 'other'.
    Used to skip false-positive >1 warnings on RMSNorm gain weights and to
    drive embed_tokens-specific row-norm analysis.
    """
    nl = name.lower()
    if "norm" in nl and len(shape) == 1:
        return "norm"
    if "embed_tokens" in nl:
        return "embed"
    if "lm_head" in nl:
        return "lm_head"
    if "bias" in nl and len(shape) == 1:
        return "bias"
    if len(shape) == 2:
        return "linear"
    return "other"


def _extract_layer_idx(name: str) -> int | None:
    """Pull the decoder layer index out of names like 'language_model.model.layers.5.self_attn.q_proj.weight'."""
    import re

    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def load_base_weights(text_model_id: str) -> dict[str, torch.Tensor] | None:
    """Download and load base LM weights for delta-from-base analysis.

    Tries single-file model.safetensors first, then falls back to sharded
    safetensors via the index manifest. Returns None on any failure so
    callers can degrade gracefully to absolute-value analysis.
    """
    try:
        try:
            path = hf_hub_download(repo_id=text_model_id, filename="model.safetensors")
            return load_file(path)
        except Exception:
            index_path = hf_hub_download(
                repo_id=text_model_id, filename="model.safetensors.index.json"
            )
            with Path(index_path).open() as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            weights: dict[str, torch.Tensor] = {}
            for shard in shard_files:
                shard_path = hf_hub_download(repo_id=text_model_id, filename=shard)
                weights.update(load_file(shard_path))
            return weights
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not load base weights from {text_model_id}: {e}[/yellow]")
        console.print("[yellow]    Falling back to absolute-value analysis (no delta).[/yellow]")
        return None


def _decoder_module(name: str) -> str:
    """Classify a decoder tensor's module: 'attn', 'mlp', 'norm', 'embed', 'lm_head', 'other'."""
    nl = name.lower()
    if "embed_tokens" in nl:
        return "embed"
    if "lm_head" in nl:
        return "lm_head"
    # `linear_attn` is Qwen3.5's hybrid linear-attention block, which holds 18
    # of its 24 layers. Without it those layers fall through to the "norm"/
    # "other" buckets and the drift table reports 0.0000 attention drift for
    # three quarters of the decoder. Checked before the "norm" branch so
    # `linear_attn.norm` groups with attention, matching how `self_attn.q_norm`
    # already does.
    if (
        "self_attn" in nl
        or "linear_attn" in nl
        or any(x in nl for x in [".q_proj.", ".k_proj.", ".v_proj.", ".o_proj."])
    ):
        return "attn"
    if ".mlp." in nl or any(x in nl for x in [".gate_proj.", ".up_proj.", ".down_proj."]):
        return "mlp"
    if "norm" in nl:
        return "norm"
    return "other"


def estimate_effective_rank(tensor: torch.Tensor, threshold: float = 0.99) -> tuple[int, int]:
    """Estimate effective rank of a weight matrix using SVD.

    Returns (effective_rank, full_rank) where effective_rank is the number
    of singular values needed to explain `threshold` of the variance.
    """
    if len(tensor.shape) != 2:
        return (0, 0)

    t = tensor.float()
    try:
        if min(t.shape) > 1000:
            _, singular_values, _ = torch.svd_lowrank(t, q=min(500, *t.shape))
        else:
            _, singular_values, _ = torch.linalg.svd(t, full_matrices=False)
    except Exception:
        return (0, min(t.shape))

    total_var = (singular_values**2).sum()
    cumvar = torch.cumsum(singular_values**2, dim=0) / total_var

    effective_rank = (cumvar < threshold).sum().item() + 1
    full_rank = min(t.shape)

    return (effective_rank, full_rank)


def analyze_tensor(
    name: str, tensor: torch.Tensor, verbose: bool = False, compute_rank: bool = True
) -> dict:
    """Analyze a single tensor and return health metrics.

    Args:
        name: Tensor name (used for type classification).
        tensor: The weight tensor.
        verbose: Include percentile and distribution analysis.
        compute_rank: Run SVD-based effective-rank estimate. Disable for
            large decoder runs where 300+ SVDs dominate runtime.
    """
    t = tensor.float()
    numel = t.numel()

    # Basic statistics
    stats = {
        "name": name,
        "shape": list(t.shape),
        "numel": numel,
        "dtype": str(tensor.dtype),
        "tensor_kind": _tensor_kind(name, tuple(t.shape)),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "var": t.var().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "abs_mean": t.abs().mean().item(),
    }

    # Health checks
    stats["nan_count"] = torch.isnan(t).sum().item()
    stats["inf_count"] = torch.isinf(t).sum().item()
    stats["exact_zero_count"] = (t == 0).sum().item()
    stats["near_zero_count"] = (t.abs() < 1e-7).sum().item()
    stats["small_count"] = (t.abs() < 1e-4).sum().item()
    stats["large_count"] = (t.abs() > 1).sum().item()
    stats["very_large_count"] = (t.abs() > 5).sum().item()

    # Percentages
    stats["exact_zero_pct"] = 100 * stats["exact_zero_count"] / numel
    stats["near_zero_pct"] = 100 * stats["near_zero_count"] / numel
    stats["large_pct"] = 100 * stats["large_count"] / numel

    # Xavier/Kaiming comparison for weight matrices
    if "weight" in name and len(t.shape) == 2:
        fan_in, fan_out = t.shape[1], t.shape[0]
        stats["xavier_std"] = np.sqrt(2.0 / (fan_in + fan_out))
        stats["kaiming_std"] = np.sqrt(2.0 / fan_in)
        stats["xavier_ratio"] = stats["std"] / stats["xavier_std"]
        stats["kaiming_ratio"] = stats["std"] / stats["kaiming_std"]

    # Neuron health for weight matrices
    if len(t.shape) == 2:
        row_norms = t.norm(dim=1)
        col_norms = t.norm(dim=0)
        stats["row_norm_mean"] = row_norms.mean().item()
        stats["row_norm_std"] = row_norms.std().item()
        stats["row_norm_min"] = row_norms.min().item()
        stats["row_norm_max"] = row_norms.max().item()
        stats["col_norm_mean"] = col_norms.mean().item()
        stats["col_norm_std"] = col_norms.std().item()
        stats["col_norm_min"] = col_norms.min().item()
        stats["col_norm_max"] = col_norms.max().item()
        stats["dead_rows"] = (row_norms < 1e-5).sum().item()
        stats["dead_cols"] = (col_norms < 1e-5).sum().item()

        # Effective rank (training capacity indicator). Skipped for large
        # decoder runs because SVD on 300+ Qwen3 matrices is dominantly slow.
        if compute_rank:
            eff_rank, full_rank = estimate_effective_rank(t)
            stats["effective_rank"] = eff_rank
            stats["full_rank"] = full_rank
            stats["rank_utilization"] = eff_rank / full_rank if full_rank > 0 else 0

    # Value distribution (binned)
    if verbose:
        abs_vals = t.abs().flatten()
        bins = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
        distribution = {}
        for i in range(len(bins) - 1):
            count = ((abs_vals >= bins[i]) & (abs_vals < bins[i + 1])).sum().item()
            key = f">={bins[i]}" if bins[i + 1] == float("inf") else f"{bins[i]}-{bins[i + 1]}"
            distribution[key] = {"count": count, "pct": 100 * count / numel}
        stats["distribution"] = distribution

        # Percentiles (sample if tensor is large)
        flat = t.flatten()
        if flat.numel() > 1_000_000:
            indices = torch.randperm(flat.numel())[:1_000_000]
            sample = flat[indices]
        else:
            sample = flat
        percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        stats["percentiles"] = {p: torch.quantile(sample, p / 100).item() for p in percentiles}

    return stats


def print_tensor_analysis(stats: dict, verbose: bool = False):
    """Print analysis for a single tensor."""
    console.print(f"\n{'─' * 70}")
    console.print(f"📊 [bold]{stats['name']}[/bold]")
    console.print(f"{'─' * 70}")
    console.print(f"  Shape: {stats['shape']}")
    console.print(f"  Total params: {stats['numel']:,}")
    console.print(f"  Dtype: {stats['dtype']}")

    # Basic statistics
    console.print("\n  📈 Basic Statistics:")
    console.print(f"    Mean:     {stats['mean']:>14.8f}")
    console.print(f"    Std:      {stats['std']:>14.8f}")
    console.print(f"    Variance: {stats['var']:>14.8f}")
    console.print(f"    Min:      {stats['min']:>14.8f}")
    console.print(f"    Max:      {stats['max']:>14.8f}")
    console.print(f"    Abs Mean: {stats['abs_mean']:>14.8f}")

    # Percentiles
    if verbose and "percentiles" in stats:
        console.print("\n  📊 Percentile Distribution:")
        for p, val in stats["percentiles"].items():
            console.print(f"    {p:>5.1f}%: {val:>14.8f}")

    # Value distribution
    if verbose and "distribution" in stats:
        console.print("\n  📊 Value Distribution (binned by absolute value):")
        for key, data in stats["distribution"].items():
            bar = "█" * int(data["pct"] / 2)
            console.print(f"    |x| {key:15s}: {data['count']:>12,} ({data['pct']:>6.2f}%) {bar}")

    console.print("\n  🔍 Health Checks:")
    nan_status = "❌ CRITICAL" if stats["nan_count"] > 0 else "✅ OK"
    inf_status = "❌ CRITICAL" if stats["inf_count"] > 0 else "✅ OK"
    zero_status = _threshold_status(stats["exact_zero_pct"], 1, 10)
    near_zero_status = _threshold_status(stats["near_zero_pct"], 1, 10)
    very_large_pct = 100 * stats["very_large_count"] / stats["numel"]
    very_large_status = _threshold_status(very_large_pct, 1, 5)
    # For RMSNorm gain weights, values > 1 are expected (per-channel amplification)
    # and the linear-weight threshold (5%/20%) doesn't apply.
    if stats.get("tensor_kind") == "norm":
        large_status = "✅ OK (norm gain)"
    else:
        large_status = _threshold_status(stats["large_pct"], 5, 20)

    console.print(f"    NaN values:        {stats['nan_count']:>12,} {nan_status}")
    console.print(f"    Inf values:        {stats['inf_count']:>12,} {inf_status}")
    console.print(
        f"    Exact zeros:       {stats['exact_zero_count']:>12,} ({stats['exact_zero_pct']:>6.2f}%) {zero_status}"
    )
    console.print(
        f"    Near-zero (<1e-7): {stats['near_zero_count']:>12,} ({stats['near_zero_pct']:>6.2f}%) {near_zero_status}"
    )
    console.print(
        f"    Small (<1e-4):     {stats['small_count']:>12,} ({100 * stats['small_count'] / stats['numel']:>6.2f}%)"
    )
    console.print(
        f"    Large (>1):        {stats['large_count']:>12,} ({stats['large_pct']:>6.2f}%) {large_status}"
    )
    console.print(
        f"    Very large (>5):   {stats['very_large_count']:>12,} ({very_large_pct:>6.2f}%) {very_large_status}"
    )

    # Xavier/Kaiming comparison
    if "xavier_std" in stats:
        console.print("\n  📐 Initialization Comparison:")
        console.print(f"    Actual std:       {stats['std']:>14.8f}")
        console.print(
            f"    Xavier expected:  {stats['xavier_std']:>14.8f} (ratio: {stats['xavier_ratio']:>6.2f}x)"
        )
        console.print(
            f"    Kaiming expected: {stats['kaiming_std']:>14.8f} (ratio: {stats['kaiming_ratio']:>6.2f}x)"
        )

        if stats["xavier_ratio"] > 1.5:
            console.print(
                f"    📝 Note: Std is {stats['xavier_ratio']:.1f}x Xavier - weights diverged during training (normal)"
            )
        elif stats["xavier_ratio"] < 0.5:
            console.print("    ⚠️  Warning: Std is very low - possible vanishing gradients")
        else:
            console.print("    ✅ Std is within expected range")

    # Neuron health
    if "row_norm_mean" in stats:
        console.print("\n  📊 Neuron Health Analysis:")
        console.print(f"    Output neurons (rows={stats['shape'][0]}):")
        console.print(
            f"      L2 norm: mean={stats['row_norm_mean']:.4f}, std={stats['row_norm_std']:.4f}"
        )
        console.print(
            f"      L2 norm: min={stats['row_norm_min']:.6f}, max={stats['row_norm_max']:.4f}"
        )
        console.print(f"    Input neurons (cols={stats['shape'][1]}):")
        console.print(
            f"      L2 norm: mean={stats['col_norm_mean']:.4f}, std={stats['col_norm_std']:.4f}"
        )
        console.print(
            f"      L2 norm: min={stats['col_norm_min']:.6f}, max={stats['col_norm_max']:.4f}"
        )

        if stats["dead_rows"] > 0:
            console.print(f"    ❌ Dead output neurons: {stats['dead_rows']}")
        if stats["dead_cols"] > 0:
            console.print(f"    ❌ Dead input neurons: {stats['dead_cols']}")
        if stats["dead_rows"] == 0 and stats["dead_cols"] == 0:
            console.print("    ✅ All neurons are active and healthy")

        # Effective rank (training capacity)
        if "effective_rank" in stats and stats["full_rank"] > 0:
            eff = stats["effective_rank"]
            full = stats["full_rank"]
            util = stats["rank_utilization"]
            console.print("\n  📈 Training Capacity (Effective Rank):")
            console.print(f"    Effective rank: {eff} / {full} ({util:.1%} utilization)")
            if util < 0.3:
                console.print(
                    "    📝 Low rank utilization - model may be underfitting or has significant capacity remaining"
                )
            elif util < 0.7:
                console.print(
                    "    ✅ Moderate rank utilization - healthy training, capacity remains"
                )
            else:
                console.print(
                    "    ⚠️  High rank utilization - approaching full capacity, may need more parameters"
                )


def gain_offset(name: str, text_model_id: str | None) -> float:
    """Offset converting a stored RMSNorm weight into its EFFECTIVE GAIN.

    Two conventions coexist inside a single Qwen3.5 decoder, verified in
    transformers source:

      Qwen3_5RMSNorm       weight init zeros, forward `output * (1.0 + weight)`
                           -> gain = 1 + w, so offset 1.0
      Qwen3_5RMSNormGated  weight init ones,  forward `weight * hidden_states`
                           -> gain = w,     so offset 0.0

    Llama / Qwen3 / most families are ones-init throughout (`self.weight *
    hidden_states`), so offset 0.0. Qwen3.5 is the one that mixes them:
    everything except `linear_attn.norm` is zero-centered.

    This matters for any relative-drift metric. Dividing ||Δ|| by ||w|| on a
    zero-centered tensor divides by the DEVIATION norm instead of the gain
    norm, which over-reports drift substantially and, fed to a hardcoded
    threshold ladder, fires early.

    Returns 0.0 for anything that is not a norm gain, so callers that run over
    mixed groups (linear weights and norms together) can apply it blindly.
    """
    if "norm" not in name.lower():
        return 0.0
    if not text_model_id or "qwen3.5" not in text_model_id.lower():
        return 0.0
    # The gated delta-net gain is the one ones-init tensor in Qwen3.5.
    return 0.0 if "linear_attn.norm" in name else 1.0


def print_decoder_summary(
    weights: dict[str, torch.Tensor],
    all_stats: list[dict],
    base_weights: dict[str, torch.Tensor] | None = None,
    top_k: int = 10,
    text_model_id: str | None = None,
) -> dict:
    """Decoder-specific aggregate view.

    When `base_weights` is provided, all sections show drift *from base* —
    far more useful than absolute values because pretrained LM weights
    (especially RMSNorm gains) already have non-uniform structure that
    swamps any fine-tuning signal in absolute terms.

    Returns a small dict of summary metrics used by the verdict at the end.
    """
    summary: dict = {}
    if base_weights is None:
        console.print(
            "\n[dim]Note: no base-model weights loaded — showing absolute values. "
            "Pass --compare-base to see fine-tuning delta from base.[/dim]"
        )

    layer_groups: dict[int, list[dict]] = {}
    for s in all_stats:
        idx = _extract_layer_idx(s["name"])
        if idx is not None:
            layer_groups.setdefault(idx, []).append(s)

    if layer_groups:
        console.rule(
            "[bold]"
            + ("DECODER PER-LAYER DRIFT" + (" (Δ FROM BASE)" if base_weights else ""))
            + "[/bold]"
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("Layer", justify="right")
        if base_weights:
            table.add_column("Δ attn", justify="right")
            table.add_column("Δ mlp", justify="right")
            table.add_column("Δ norm", justify="right")
            table.add_column("max Δ ch", justify="right")
        else:
            table.add_column("attn std", justify="right")
            table.add_column("mlp std", justify="right")
            table.add_column("norm mean", justify="right")
            table.add_column("norm range", justify="right")

        for idx in sorted(layer_groups):
            stats_for_layer = layer_groups[idx]
            attn = [s for s in stats_for_layer if _decoder_module(s["name"]) == "attn"]
            mlp = [s for s in stats_for_layer if _decoder_module(s["name"]) == "mlp"]
            norm = [s for s in stats_for_layer if _decoder_module(s["name"]) == "norm"]

            if base_weights:
                # Relative L2 drift: ||trained - base|| / ||base||. Skip
                # tensors with shape mismatches (e.g., resized vocab) — those
                # are reported separately in the embed_tokens section.
                def rel_drift(group: list[dict]) -> float:
                    total_d, total_b = 0.0, 0.0
                    for s in group:
                        base = resolve_base_tensor(s["name"], base_weights)
                        if base is None:
                            continue
                        cur = weights[s["name"]].float()
                        b = base.float()
                        if cur.shape != b.shape:
                            continue
                        # Denominator is the effective gain for norm tensors --
                        # a no-op (identity 0.0) for linear weights. See
                        # gain_offset().
                        gain_b = b + gain_offset(s["name"], text_model_id)
                        total_d += float((cur - b).pow(2).sum().item())
                        total_b += float(gain_b.pow(2).sum().item())
                    return (total_d / total_b) ** 0.5 if total_b > 0 else 0.0

                def max_channel_delta(group: list[dict]) -> float:
                    m = 0.0
                    for s in group:
                        base = resolve_base_tensor(s["name"], base_weights)
                        if base is None:
                            continue
                        cur = weights[s["name"]].float()
                        b = base.float()
                        if cur.shape != b.shape:
                            continue
                        m = max(m, float((cur - b).abs().max().item()))
                    return m

                attn_d = rel_drift(attn)
                mlp_d = rel_drift(mlp)
                norm_d = rel_drift(norm)
                norm_max_ch = max_channel_delta(norm)

                table.add_row(
                    str(idx),
                    f"{attn_d:.4f}",
                    f"{mlp_d:.4f}",
                    f"{norm_d:.4f}",
                    f"{norm_max_ch:.4f}",
                )
            else:
                attn_std = sum(s["std"] for s in attn) / len(attn) if attn else 0.0
                mlp_std = sum(s["std"] for s in mlp) / len(mlp) if mlp else 0.0
                norm_mean = sum(s["mean"] for s in norm) / len(norm) if norm else 0.0
                norm_min = min((s["min"] for s in norm), default=0.0)
                norm_max = max((s["max"] for s in norm), default=0.0)
                table.add_row(
                    str(idx),
                    f"{attn_std:.4f}",
                    f"{mlp_std:.4f}",
                    f"{norm_mean:.4f}",
                    f"[{norm_min:.3f}, {norm_max:.3f}]",
                )
        console.print(table)
        if base_weights:
            console.print(
                "  [dim]Δ attn/mlp/norm = relative L2 drift ||trained-base|| / ||base||. "
                "max Δ ch = largest single-channel absolute change in a norm weight.[/dim]"
            )

    embed_stats = next((s for s in all_stats if s.get("tensor_kind") == "embed"), None)
    if embed_stats is not None and embed_stats["name"] in weights:
        console.rule(
            "[bold]"
            + ("EMBED_TOKENS" + (" DRIFT FROM BASE" if base_weights else " ROW-NORM DISTRIBUTION"))
            + "[/bold]"
        )
        embed = weights[embed_stats["name"]].float()
        base_embed_tensor = resolve_base_tensor(embed_stats["name"], base_weights)
        base_embed = None if base_embed_tensor is None else base_embed_tensor.float()

        if base_embed is not None:
            # Vocab sizes commonly differ between base and fine-tuned: the
            # tiny-audio model adds <audio>, drops base's vocab padding, and
            # ends up at 151,670 vs Qwen3's 151,936. Token IDs 0..n-1 are
            # aligned in both (same BPE), so compare the overlapping prefix.
            n = min(embed.shape[0], base_embed.shape[0])
            if embed.shape[0] != base_embed.shape[0]:
                console.print(
                    f"  [yellow]Vocab size mismatch — trained {embed.shape[0]:,} "
                    f"vs base {base_embed.shape[0]:,}. "
                    f"Comparing first {n:,} aligned rows.[/yellow]"
                )
            embed = embed[:n]
            base_embed = base_embed[:n]
            delta = embed - base_embed
            delta_norms = delta.norm(dim=1)
            base_norms = base_embed.norm(dim=1)
            console.print(
                f"  Shape: {list(embed.shape)} ({embed.shape[0]:,} tokens × {embed.shape[1]} dim)"
            )
            console.print(
                f"  Per-row Δ L2 norm: mean={delta_norms.mean().item():.4f}, "
                f"std={delta_norms.std().item():.4f}, "
                f"min={delta_norms.min().item():.4f}, "
                f"max={delta_norms.max().item():.4f}"
            )
            relative = delta_norms / base_norms.clamp(min=1e-6)
            console.print(
                f"  Relative drift  (Δ/base): mean={relative.mean().item():.4f}, "
                f"max={relative.max().item():.4f}"
            )

            top_indices = torch.topk(delta_norms, k=top_k).indices.tolist()
            console.print(f"\n  Top {top_k} most-drifted tokens (||Δ row||):")
            for i in top_indices:
                console.print(
                    f"    token_id {i:>6d}: ||Δ||={delta_norms[i].item():.4f} "
                    f"(base ||row||={base_norms[i].item():.4f}, "
                    f"relative={relative[i].item():.2%})"
                )

            mean_drift = delta_norms.mean().item()
            max_drift = delta_norms.max().item()
            summary["embed_mean_drift"] = mean_drift
            summary["embed_max_drift"] = max_drift
            if mean_drift == 0.0 and max_drift == 0.0:
                # freeze_text_embed_tokens=true keeps every row at its base value.
                console.print(
                    "\n  ✅ Embed_tokens frozen — zero drift, rare-token vocab fully preserved."
                )
            # Heuristic: if max drift dominates mean by >100x, a small number
            # of tokens have shifted dramatically (rare-token drift signature).
            elif max_drift > 100 * mean_drift:
                console.print(
                    f"\n  ⚠️  Max drift ({max_drift:.3f}) dominates mean ({mean_drift:.3f}) "
                    "by >100× — a few tokens have moved dramatically (possible rare-token drift)."
                )
            else:
                console.print(
                    f"\n  ✅ Drift distribution balanced "
                    f"(max/mean = {max_drift / mean_drift:.1f}×) — "
                    "no runaway tokens detected."
                )
        else:
            # Fallback to absolute row-norm distribution.
            row_norms = embed.norm(dim=1)
            console.print(
                f"  Shape: {list(embed.shape)} ({embed.shape[0]:,} tokens × {embed.shape[1]} dim)"
            )
            console.print(
                f"  Row norms: mean={row_norms.mean().item():.4f}, "
                f"std={row_norms.std().item():.4f}, "
                f"min={row_norms.min().item():.4f}, "
                f"max={row_norms.max().item():.4f}"
            )
            ratio = (row_norms.max() / row_norms.min().clamp(min=1e-6)).item()
            console.print(f"  Max/min ratio: {ratio:.2f}x")

    norm_stats = [s for s in all_stats if s.get("tensor_kind") == "norm"]
    if norm_stats:
        console.rule(
            "[bold]"
            + ("RMSNORM GAIN" + (" Δ FROM BASE" if base_weights else " COHERENCE"))
            + "[/bold]"
        )

        if base_weights:
            deltas: list[float] = []
            max_layer = (None, 0.0)
            for s in norm_stats:
                base = resolve_base_tensor(s["name"], base_weights)
                if base is None:
                    continue
                cur = weights[s["name"]].float()
                b = base.float()
                if cur.shape != b.shape:
                    continue
                # Measure against the EFFECTIVE GAIN, not the stored weight.
                # On a zero-centered RMSNorm the stored tensor is the deviation
                # from identity, so ||b|| is not the gain magnitude; see
                # gain_offset(). The numerator is unaffected either way,
                # since (id+cur) - (id+b) == cur - b.
                offset = gain_offset(s["name"], text_model_id)
                gain_b = b + offset
                d = float((cur - b).pow(2).sum().sqrt().item())
                rel = d / float(gain_b.pow(2).sum().sqrt().clamp(min=1e-6).item())
                deltas.append(rel)
                if rel > max_layer[1]:
                    max_layer = (s["name"], rel)

            if deltas:
                avg_rel = sum(deltas) / len(deltas)
                summary["norm_mean_rel_drift"] = avg_rel
                summary["norm_max_rel_drift"] = max_layer[1]
                console.print(f"  Number of norm tensors: {len(deltas)}")
                console.print(
                    f"  Relative drift ||Δ||/||base||: avg={avg_rel:.4f}, max={max_layer[1]:.4f}"
                )
                console.print(f"  Largest drift: {max_layer[0]}")

                # Calibration: at 7000-step fine-tune, expect avg <0.05 (5% L2
                # drift) for healthy WD routing. If the WD-on-RMSNorm bug were
                # still pulling norms toward zero, we'd see avg drift much
                # larger than the linear-weight drift (since norms have higher
                # effective LR under Adam).
                if avg_rel < 0.05:
                    console.print(
                        f"\n  ✅ Norm gains close to base (avg {avg_rel:.1%} drift) — "
                        "WD-on-RMSNorm routing is healthy."
                    )
                elif avg_rel < 0.2:
                    console.print(
                        f"\n  ✅ Norm gains moderately drifted "
                        f"(avg {avg_rel:.1%}) — within normal fine-tuning range."
                    )
                else:
                    console.print(
                        f"\n  ⚠️  Norm gains significantly drifted "
                        f"(avg {avg_rel:.1%}) — check WD routing if larger than "
                        "linear-weight drift."
                    )
        else:
            # Fallback: absolute stats, uncalibrated against base.
            #
            # Split by convention rather than averaged. One mean over both is
            # not a gain: on Qwen3.5 the zero-centered text norms sit near
            # 0.28 (true gain 1.28) while the gated ones sit near 0.91, so a
            # combined average is a number with no interpretation.
            by_conv: dict[float, list[dict]] = {0.0: [], 1.0: []}
            for s in norm_stats:
                by_conv[gain_offset(s["name"], text_model_id)].append(s)

            console.print(f"  Number of norm tensors: {len(norm_stats)}")
            for offset, label in ((1.0, "zero-centered (1 + w)"), (0.0, "ones-init (w)")):
                group = by_conv[offset]
                if not group:
                    continue
                gains = [s["mean"] + offset for s in group]
                stds = [s["std"] for s in group]
                gain_avg = sum(gains) / len(gains)
                std_avg = sum(stds) / len(stds)
                console.print(
                    f"  {label:<24} n={len(group):<3} "
                    f"effective gain avg={gain_avg:.4f}, "
                    f"range=[{min(gains):.4f}, {max(gains):.4f}], "
                    f"std avg={std_avg:.4f}"
                )
            console.print(
                "\n  [dim]Note: absolute norm gains reflect base-LM pretraining "
                "as much as fine-tuning. Pass --compare-base for delta analysis.[/dim]"
            )

    return summary


def analyze_weights(
    model_id: str,
    filter_prefix: str | None = None,
    verbose: bool = False,
    component: str = "projector",
    per_tensor: bool = True,
    skip_rank: bool = False,
    compare_base: bool = True,
):
    """Analyze model weights for training health."""
    console.rule(f"[bold]Weight Analysis: {model_id}[/bold]")

    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        return False

    with Path(config_path).open() as f:
        config = json.load(f)

    console.print("\n[bold]Model Configuration[/bold]")
    important_keys = [
        "model_type",
        "projector_type",
        "encoder_dim",
        "llm_dim",
        "projector_pool_stride",
        "audio_model_id",
        "text_model_id",
        "use_lora",
    ]
    for key in important_keys:
        if key in config:
            console.print(f"  {key}: {config[key]}")

    weights = load_file(weights_path)

    if filter_prefix:
        weights = {k: v for k, v in weights.items() if filter_prefix in k}
        if not weights:
            console.print(f"[red]No weights found matching '{filter_prefix}'[/red]")
            return False
        console.print(f"\nFiltering to weights containing '{filter_prefix}'")

    console.rule("[bold]WEIGHT TENSORS[/bold]")

    # Determine trainability from the saved config rather than hardcoding
    # "projector" — language_model.* is trainable when freeze_language_model
    # is False (the default for embedded.yaml's full-decoder fine-tune).
    freeze_lm = config.get("freeze_language_model", True)
    use_lora = bool(config.get("use_lora", False))

    def _is_trainable(name: str) -> bool:
        if "projector" in name:
            return True
        if "language_model" in name and not freeze_lm:
            return True
        return bool(use_lora and "lora" in name.lower())

    total_params = 0
    trainable_params = 0

    # In decoder mode, a 311-row table is noise — collapse to a summary count.
    # Per-tensor inspection still available via --per-tensor.
    collapse_table = component == "decoder" and not per_tensor

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Type")

    for name in sorted(weights.keys()):
        tensor = weights[name]
        params = tensor.numel()
        total_params += params
        is_trainable = _is_trainable(name)
        if is_trainable:
            trainable_params += params
            marker = "🎯 trainable"
        else:
            marker = "❄️ frozen"
        if not collapse_table:
            table.add_row(name, str(list(tensor.shape)), f"{params:,}", marker)

    if collapse_table:
        console.print(
            f"  [{len(weights)} decoder tensors — table suppressed in decoder mode "
            "(pass --per-tensor to expand)]"
        )
    else:
        console.print(table)
    console.print(f"\n  Total parameters:     {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,}")
    console.print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    # Decoder mode: 311 Qwen3 tensors at full per-tensor verbosity is unreadable
    # and the effective-rank SVDs dominate runtime. Default to aggregate view;
    # opt back in to the firehose with --per-tensor.
    is_decoder_mode = component == "decoder"
    show_per_tensor = (not is_decoder_mode) or per_tensor
    compute_rank = (not skip_rank) and show_per_tensor

    if show_per_tensor:
        console.rule("[bold]DETAILED WEIGHT ANALYSIS[/bold]")

    all_stats = []
    for name in sorted(weights.keys()):
        stats = analyze_tensor(name, weights[name], verbose=verbose, compute_rank=compute_rank)
        all_stats.append(stats)
        if show_per_tensor:
            print_tensor_analysis(stats, verbose=verbose)

    if is_decoder_mode:
        base_weights = None
        if compare_base:
            text_model_id = config.get("text_model_id")
            if text_model_id:
                console.print(
                    f"\n[dim]Loading base weights from {text_model_id} for delta analysis...[/dim]"
                )
                base_weights = load_base_weights(text_model_id)
        print_decoder_summary(
            weights,
            all_stats,
            base_weights=base_weights,
            text_model_id=config.get("text_model_id"),
        )

    console.rule("[bold]OVERALL TRAINING HEALTH SUMMARY[/bold]")

    total_nans = sum(s["nan_count"] for s in all_stats)
    total_infs = sum(s["inf_count"] for s in all_stats)
    total_zeros = sum(s["exact_zero_count"] for s in all_stats)
    total_params_analyzed = sum(s["numel"] for s in all_stats)
    total_dead_neurons = sum(s.get("dead_rows", 0) + s.get("dead_cols", 0) for s in all_stats)

    # Split dead units into inherited (already dead in the base checkpoint) and
    # acquired (killed by this run). Only the second kind is a training issue.
    #
    # Qwen3.5-2B ships 14 dead rows in layers.0.linear_attn.in_proj_qkv: 7
    # channels whose Q and K halves both hold denormals (~1.2e-37, so the fp32
    # norm underflows to 0). They cannot recover -- each half's gradient is
    # proportional to the other, so it underflows too, and over 18.5k steps
    # they moved by 3e-40. Counting those against the run made the verdict read
    # "Issues detected" on every single checkpoint.
    inherited_dead, acquired_dead, acquired_where = 0, 0, []
    if base_weights:
        for s in all_stats:
            if not (s.get("dead_rows", 0) or s.get("dead_cols", 0)):
                continue
            cur = weights.get(s["name"])
            base = resolve_base_tensor(s["name"], base_weights)
            if cur is None or cur.dim() != 2:
                continue
            t = cur.float()
            t_dead = {("r", i) for i in (t.norm(dim=1) < 1e-5).nonzero().flatten().tolist()} | {
                ("c", i) for i in (t.norm(dim=0) < 1e-5).nonzero().flatten().tolist()
            }
            if base is None or base.shape != cur.shape:
                acquired_dead += len(t_dead)
                acquired_where.append((s["name"], len(t_dead), "base tensor unavailable"))
                continue
            b = base.float()
            b_dead = {("r", i) for i in (b.norm(dim=1) < 1e-5).nonzero().flatten().tolist()} | {
                ("c", i) for i in (b.norm(dim=0) < 1e-5).nonzero().flatten().tolist()
            }
            inherited_dead += len(t_dead & b_dead)
            new = t_dead - b_dead
            if new:
                acquired_dead += len(new)
                acquired_where.append((s["name"], len(new), "killed during training"))

    console.print(f"\n  📦 Analyzed {len(all_stats)} tensors, {total_params_analyzed:,} parameters")

    console.print("\n  🔬 Numerical Stability:")
    console.print(f"     NaN values: {total_nans} {'❌' if total_nans > 0 else '✅'}")
    console.print(f"     Inf values: {total_infs} {'❌' if total_infs > 0 else '✅'}")
    console.print(
        f"     Exact zeros: {total_zeros} ({100 * total_zeros / total_params_analyzed:.4f}%)"
    )
    if base_weights and total_dead_neurons:
        mark = "❌" if acquired_dead > 0 else "✅"
        console.print(
            f"     Dead neurons: {total_dead_neurons} "
            f"({inherited_dead} inherited from base, {acquired_dead} acquired) {mark}"
        )
        for name, n, why in acquired_where:
            console.print(f"       ⚠️  {n} in {name} ({why})")
    else:
        console.print(
            f"     Dead neurons: {total_dead_neurons} {'❌' if total_dead_neurons > 0 else '✅'}"
            + (
                "  [dim](pass --compare-base to separate inherited)[/dim]"
                if total_dead_neurons
                else ""
            )
        )

    # Per-tensor listing is useful for projector (4 tensors) but unreadable
    # for decoder (311 tensors); decoder mode prints per-layer aggregates via
    # print_decoder_summary instead.
    if not is_decoder_mode:
        console.print("\n  📊 Layer-wise Weight Statistics:")
        for s in all_stats:
            short_name = s["name"].replace("projector.", "").replace("audio_tower.", "")
            console.print(
                f"     {short_name:25s}: mean={s['mean']:>10.6f}, "
                f"std={s['std']:>8.6f}, range=[{s['min']:>8.4f}, {s['max']:>7.4f}]"
            )

    console.print("\n  📈 Training Capacity Analysis:")

    rank_stats = [
        (s["name"], s["effective_rank"], s["full_rank"], s["rank_utilization"])
        for s in all_stats
        if "effective_rank" in s and s["full_rank"] > 0
    ]

    if rank_stats:
        avg_rank_util = sum(r[3] for r in rank_stats) / len(rank_stats)
        min_rank_util = min(r[3] for r in rank_stats)
        max_rank_util = max(r[3] for r in rank_stats)

        console.print("     Rank utilization across layers:")
        for name, eff, full, util in rank_stats:
            short_name = name.replace("projector.", "")
            bar = "█" * int(util * 20)
            console.print(f"       {short_name:25s}: {eff:>4}/{full:<4} ({util:>5.1%}) {bar}")

        console.print(
            f"\n     Summary: avg={avg_rank_util:.1%}, min={min_rank_util:.1%}, max={max_rank_util:.1%}"
        )

        if avg_rank_util < 0.4:
            capacity_status = "🟢 HIGH CAPACITY REMAINING"
            capacity_msg = "Model is using < 40% of its representational capacity. Significant room for continued training."
        elif avg_rank_util < 0.7:
            capacity_status = "🟡 MODERATE CAPACITY REMAINING"
            capacity_msg = (
                "Model is using 40-70% of capacity. Healthy training with room for improvement."
            )
        elif avg_rank_util < 0.9:
            capacity_status = "🟠 LIMITED CAPACITY REMAINING"
            capacity_msg = (
                "Model is using 70-90% of capacity. May benefit from increased projector size."
            )
        else:
            capacity_status = "🔴 NEAR FULL CAPACITY"
            capacity_msg = "Model is using > 90% of capacity. Consider larger projector or LoRA for further training."

        console.print(f"\n     {capacity_status}")
        console.print(f"     {capacity_msg}")

    xavier_ratios = [s["xavier_ratio"] for s in all_stats if "xavier_ratio" in s]
    if xavier_ratios:
        avg_xavier = sum(xavier_ratios) / len(xavier_ratios)
        console.print("\n     Weight divergence from initialization:")
        console.print(f"       Average Xavier ratio: {avg_xavier:.1f}x")
        if avg_xavier < 2:
            console.print(
                "       📝 Weights close to initialization - early in training or learning slowly"
            )
        elif avg_xavier < 5:
            console.print("       ✅ Healthy divergence - actively training")
        elif avg_xavier < 10:
            console.print("       ✅ Significant divergence - well-trained")
        else:
            console.print("       ⚠️  Very high divergence - check for instability")

    issues = []
    if total_nans > 0:
        issues.append("NaN values detected")
    if total_infs > 0:
        issues.append("Inf values detected")
    if 100 * total_zeros / total_params_analyzed > 10:
        issues.append("High percentage of zero weights")
    # Only units this run killed count as an issue; inherited ones are a
    # property of the base checkpoint. Without a base to compare against we
    # cannot tell them apart, so fall back to flagging the total.
    if base_weights:
        if acquired_dead > 0:
            issues.append(f"{acquired_dead} dead neurons acquired during training")
    elif total_dead_neurons > 0:
        issues.append(f"{total_dead_neurons} dead neurons detected (no base for comparison)")

    console.print(f"\n  {'─' * 60}")
    if not issues:
        console.print("  🎉 [bold green]VERDICT: Model is HEALTHY[/bold green]")
        console.print("     ✅ No numerical instabilities (NaN/Inf)")
        console.print("     ✅ No dead neurons")
        console.print("     ✅ Weight magnitudes are reasonable")
        console.print("     ✅ Weight standard deviations show training occurred")
        return True
    console.print("  ❌ [bold red]VERDICT: Issues detected[/bold red]")
    for issue in issues:
        console.print(f"     ❌ {issue}")
    return False


@app.command()
def main(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID"),
    ] = "mazesmazes/tiny-audio",
    component: Annotated[
        str,
        typer.Option(
            "--component",
            "-c",
            help="Which model component to analyze: projector / decoder / encoder / all",
        ),
    ] = "projector",
    filter_: Annotated[
        str | None,
        typer.Option(
            "--filter",
            "-f",
            help="Override --component with an arbitrary substring filter",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed percentile and distribution analysis"),
    ] = False,
    per_tensor: Annotated[
        bool,
        typer.Option(
            "--per-tensor",
            help="Force per-tensor printout (default off for decoder, on for everything else)",
        ),
    ] = False,
    skip_rank: Annotated[
        bool,
        typer.Option("--skip-rank", help="Skip effective-rank SVDs (slow on large decoder runs)"),
    ] = False,
    compare_base: Annotated[
        bool,
        typer.Option(
            "--compare-base/--no-compare-base",
            help="(decoder) Download base LM weights to report drift from base "
            "instead of absolute values. Default on; pass --no-compare-base to "
            "skip the download.",
        ),
    ] = True,
):
    """Analyze model weights for training health diagnostics.

    Checks for:
    - NaN/Inf values
    - Dead neurons
    - Weight magnitude issues
    - Initialization divergence
    - (decoder) per-layer attn/mlp/norm drift
    - (decoder) embed_tokens row-norm distribution (rare-token drift)
    - (decoder) RMSNorm gain coherence (WD-on-norm routing health)
    """
    if filter_ is not None:
        # Explicit filter — try to auto-detect component for display mode.
        effective_filter = filter_
        for comp, fil in COMPONENT_FILTERS.items():
            if fil and fil in filter_:
                component = comp
                break
    else:
        effective_filter = COMPONENT_FILTERS.get(component)
        if effective_filter is None:
            console.print(f"[red]Unknown --component: {component}[/red]")
            sys.exit(2)

    success = analyze_weights(
        model_id,
        filter_prefix=effective_filter or None,
        verbose=verbose,
        component=component,
        per_tensor=per_tensor,
        skip_rank=skip_rank,
        compare_base=compare_base,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()
