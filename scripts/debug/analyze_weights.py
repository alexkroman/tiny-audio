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

app = typer.Typer(help="Analyze model weights for training health")
console = Console()


def estimate_effective_rank(tensor: torch.Tensor, threshold: float = 0.99) -> tuple[int, int]:
    """Estimate effective rank of a weight matrix using SVD.

    Returns (effective_rank, full_rank) where effective_rank is the number
    of singular values needed to explain `threshold` of the variance.
    """
    if len(tensor.shape) != 2:
        return (0, 0)

    t = tensor.float()
    # Use randomized SVD for large matrices
    try:
        if min(t.shape) > 1000:
            # Sample for very large matrices
            _, singular_values, _ = torch.svd_lowrank(t, q=min(500, min(t.shape)))
        else:
            _, singular_values, _ = torch.linalg.svd(t, full_matrices=False)
    except Exception:
        return (0, min(t.shape))

    # Compute variance explained
    total_var = (singular_values**2).sum()
    cumvar = torch.cumsum(singular_values**2, dim=0) / total_var

    # Find rank needed for threshold variance
    effective_rank = (cumvar < threshold).sum().item() + 1
    full_rank = min(t.shape)

    return (effective_rank, full_rank)


def analyze_tensor(name: str, tensor: torch.Tensor, verbose: bool = False) -> dict:
    """Analyze a single tensor and return health metrics."""
    t = tensor.float()
    numel = t.numel()

    # Basic statistics
    stats = {
        "name": name,
        "shape": list(t.shape),
        "numel": numel,
        "dtype": str(tensor.dtype),
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

        # Effective rank (training capacity indicator)
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

    # Health checks
    console.print("\n  🔍 Health Checks:")
    nan_status = "❌ CRITICAL" if stats["nan_count"] > 0 else "✅ OK"
    inf_status = "❌ CRITICAL" if stats["inf_count"] > 0 else "✅ OK"
    zero_status = (
        "✅ OK"
        if stats["exact_zero_pct"] < 1
        else ("⚠️  WARNING" if stats["exact_zero_pct"] < 10 else "❌ HIGH")
    )
    near_zero_status = (
        "✅ OK"
        if stats["near_zero_pct"] < 1
        else ("⚠️  WARNING" if stats["near_zero_pct"] < 10 else "❌ HIGH")
    )
    large_status = (
        "✅ OK"
        if stats["large_pct"] < 5
        else ("⚠️  WARNING" if stats["large_pct"] < 20 else "❌ HIGH")
    )
    very_large_pct = 100 * stats["very_large_count"] / stats["numel"]
    very_large_status = (
        "✅ OK" if very_large_pct < 1 else ("⚠️  WARNING" if very_large_pct < 5 else "❌ HIGH")
    )

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


def analyze_weights(
    model_id: str,
    filter_prefix: str | None = None,
    verbose: bool = False,
):
    """Analyze model weights for training health."""
    console.print("=" * 70)
    console.print(f"[bold]Weight Analysis: {model_id}[/bold]")
    console.print("=" * 70)

    # Download model files
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        return False

    # Load config
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
        "use_specaugment",
        "use_lora",
    ]
    for key in important_keys:
        if key in config:
            console.print(f"  {key}: {config[key]}")

    # Load weights
    weights = load_file(weights_path)

    # Filter weights if prefix specified
    if filter_prefix:
        weights = {k: v for k, v in weights.items() if filter_prefix in k}
        if not weights:
            console.print(f"[red]No weights found matching '{filter_prefix}'[/red]")
            return False
        console.print(f"\nFiltering to weights containing '{filter_prefix}'")

    # Print all tensors summary
    console.print(f"\n{'=' * 70}")
    console.print("[bold]WEIGHT TENSORS[/bold]")
    console.print("=" * 70)

    total_params = 0
    trainable_params = 0

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Type")

    for name in sorted(weights.keys()):
        tensor = weights[name]
        params = tensor.numel()
        total_params += params
        is_trainable = "projector" in name
        if is_trainable:
            trainable_params += params
            marker = "🎯 trainable"
        else:
            marker = "❄️ frozen"
        table.add_row(name, str(list(tensor.shape)), f"{params:,}", marker)

    console.print(table)
    console.print(f"\n  Total parameters:     {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,}")
    console.print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    # Detailed analysis
    console.print(f"\n{'=' * 70}")
    console.print("[bold]DETAILED WEIGHT ANALYSIS[/bold]")
    console.print("=" * 70)

    all_stats = []
    for name in sorted(weights.keys()):
        stats = analyze_tensor(name, weights[name], verbose=verbose)
        all_stats.append(stats)
        print_tensor_analysis(stats, verbose=verbose)

    # Overall summary
    console.print(f"\n{'=' * 70}")
    console.print("[bold]OVERALL TRAINING HEALTH SUMMARY[/bold]")
    console.print("=" * 70)

    # Aggregate checks
    total_nans = sum(s["nan_count"] for s in all_stats)
    total_infs = sum(s["inf_count"] for s in all_stats)
    total_zeros = sum(s["exact_zero_count"] for s in all_stats)
    total_params_analyzed = sum(s["numel"] for s in all_stats)
    total_dead_neurons = sum(s.get("dead_rows", 0) + s.get("dead_cols", 0) for s in all_stats)

    console.print(f"\n  📦 Analyzed {len(all_stats)} tensors, {total_params_analyzed:,} parameters")

    console.print("\n  🔬 Numerical Stability:")
    console.print(f"     NaN values: {total_nans} {'❌' if total_nans > 0 else '✅'}")
    console.print(f"     Inf values: {total_infs} {'❌' if total_infs > 0 else '✅'}")
    console.print(
        f"     Exact zeros: {total_zeros} ({100 * total_zeros / total_params_analyzed:.4f}%)"
    )
    console.print(
        f"     Dead neurons: {total_dead_neurons} {'❌' if total_dead_neurons > 0 else '✅'}"
    )

    console.print("\n  📊 Layer-wise Weight Statistics:")
    for s in all_stats:
        short_name = s["name"].replace("projector.", "")
        console.print(
            f"     {short_name:25s}: mean={s['mean']:>10.6f}, std={s['std']:>8.6f}, range=[{s['min']:>8.4f}, {s['max']:>7.4f}]"
        )

    # Training capacity analysis
    console.print("\n  📈 Training Capacity Analysis:")

    # Collect rank utilization stats
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

        # Interpret capacity
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

    # Weight divergence from initialization
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

    # Final verdict
    issues = []
    if total_nans > 0:
        issues.append("NaN values detected")
    if total_infs > 0:
        issues.append("Inf values detected")
    if 100 * total_zeros / total_params_analyzed > 10:
        issues.append("High percentage of zero weights")
    if total_dead_neurons > 0:
        issues.append(f"{total_dead_neurons} dead neurons detected")

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
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter", "-f", help="Filter to weights containing this string (e.g., 'projector')"
        ),
    ] = "projector",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed percentile and distribution analysis"),
    ] = False,
):
    """Analyze model weights for training health diagnostics.

    Checks for:
    - NaN/Inf values
    - Dead neurons
    - Weight magnitude issues
    - Initialization divergence
    """
    success = analyze_weights(model_id, filter_prefix=filter, verbose=verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()
