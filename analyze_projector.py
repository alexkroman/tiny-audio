#!/usr/bin/env python3
"""
Analyze the audio projector to determine if it's appropriately sized.
Checks for signs of over-parameterization or under-utilization.
"""

import argparse
import json

import numpy as np
import torch
from transformers.utils import cached_file


def load_projector_weights(model_path: str):
    """Load projector weights from Hub or local path."""
    print(f"Loading projector weights from: {model_path}")

    # Try loading from last-checkpoint first
    try:
        projector_file = cached_file(
            model_path,
            "projector.safetensors",
            subfolder="last-checkpoint",
            _raise_exceptions_for_missing_entries=False,
        )
    except Exception:
        # Fallback to main directory
        projector_file = cached_file(
            model_path,
            "projector.safetensors",
            _raise_exceptions_for_missing_entries=False,
        )

    if not projector_file:
        raise FileNotFoundError(f"Could not find projector.safetensors in {model_path}")

    from safetensors.torch import load_file
    weights = load_file(projector_file)

    print(f"✓ Loaded {len(weights)} weight tensors")
    return weights


def analyze_weight_matrix(weight: torch.Tensor, name: str):
    """Analyze a single weight matrix using SVD and statistics."""

    # Convert to float32 for SVD (not supported for bfloat16 on CPU)
    weight = weight.float()

    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Normalize singular values
    S_normalized = S / S.sum()

    # Calculate metrics

    # 1. Effective rank (entropy-based)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    # 2. Stable rank (Frobenius norm / spectral norm)
    stable_rank = (S ** 2).sum().item() / (S[0] ** 2).item()

    # 3. Intrinsic dimensionality (90% of variance)
    cumsum = torch.cumsum(S_normalized, dim=0)
    intrinsic_dim_90 = (cumsum < 0.90).sum().item() + 1
    intrinsic_dim_95 = (cumsum < 0.95).sum().item() + 1
    intrinsic_dim_99 = (cumsum < 0.99).sum().item() + 1

    # 4. Spectral decay (how fast singular values drop)
    # Measure the ratio of top-k to total
    matrix_rank = min(weight.shape)
    top_10_percent = max(1, matrix_rank // 10)
    top_25_percent = max(1, matrix_rank // 4)
    top_50_percent = max(1, matrix_rank // 2)

    spectral_concentration_10 = S[:top_10_percent].sum() / S.sum()
    spectral_concentration_25 = S[:top_25_percent].sum() / S.sum()
    spectral_concentration_50 = S[:top_50_percent].sum() / S.sum()

    # 5. Weight statistics
    weight_norm = torch.norm(weight).item()
    weight_mean = weight.abs().mean().item()
    weight_std = weight.std().item()

    # 6. Dead neurons (very small weights)
    # For biases or per-neuron analysis
    if len(weight.shape) == 1:
        dead_threshold = weight.abs().max().item() * 0.01
        dead_neurons = (weight.abs() < dead_threshold).sum().item()
        dead_percentage = (dead_neurons / weight.shape[0]) * 100
    else:
        # For weight matrices, check rows
        row_norms = torch.norm(weight, dim=1)
        dead_threshold = row_norms.max().item() * 0.01
        dead_neurons = (row_norms < dead_threshold).sum().item()
        dead_percentage = (dead_neurons / weight.shape[0]) * 100

    # 7. Condition number
    condition_number = S[0].item() / (S[-1].item() + 1e-10)

    return {
        "name": name,
        "shape": list(weight.shape),
        "total_params": weight.numel(),
        "matrix_rank": matrix_rank,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "rank_utilization": effective_rank / matrix_rank,
        "intrinsic_dim_90": intrinsic_dim_90,
        "intrinsic_dim_95": intrinsic_dim_95,
        "intrinsic_dim_99": intrinsic_dim_99,
        "spectral_concentration_10": spectral_concentration_10.item(),
        "spectral_concentration_25": spectral_concentration_25.item(),
        "spectral_concentration_50": spectral_concentration_50.item(),
        "weight_norm": weight_norm,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "dead_neurons": dead_neurons,
        "dead_percentage": dead_percentage,
        "condition_number": condition_number,
    }


def generate_projector_recommendations(results: dict):
    """Generate recommendations based on projector analysis."""

    print("\n" + "=" * 80)
    print("PROJECTOR SIZING ANALYSIS")
    print("=" * 80)

    # Calculate architecture info
    gate_proj = results["gate_proj.weight"]
    up_proj = results["up_proj.weight"]
    down_proj = results["down_proj.weight"]

    hidden_dim = gate_proj["shape"][0]
    input_dim = gate_proj["shape"][1]
    output_dim = down_proj["shape"][0]

    total_params = sum(r["total_params"] for r in results.values())

    print(f"\nProjector Architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Total parameters: {total_params:,}")

    # Analyze each component
    print(f"\n{'Layer':<20} {'Rank Util':<12} {'Dead %':<10} {'Spectral 50%':<15} {'Status'}")
    print("-" * 80)

    recommendations = []

    for name, r in results.items():
        if name.endswith(".weight"):
            rank_util = r["rank_utilization"]
            dead_pct = r["dead_percentage"]
            spec_50 = r["spectral_concentration_50"]

            # Determine status
            if rank_util < 0.5 or dead_pct > 10:
                status = "⚠️  OVER-PARAMETERIZED"
            elif rank_util > 0.9 and spec_50 > 0.9:
                status = "⚠️  UNDER-PARAMETERIZED"
            else:
                status = "✓ GOOD"

            print(f"  {name:<18} {rank_util:>6.1%}      {dead_pct:>5.1f}%     {spec_50:>6.1%}         {status}")

    # Overall assessment
    avg_rank_util = np.mean([r["rank_utilization"] for r in results.values() if "weight" in r["name"]])
    avg_dead_pct = np.mean([r["dead_percentage"] for r in results.values() if "weight" in r["name"]])
    avg_spec_50 = np.mean([r["spectral_concentration_50"] for r in results.values() if "weight" in r["name"]])

    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    print(f"\nAverage Metrics:")
    print(f"  Rank utilization: {avg_rank_util:.1%}")
    print(f"  Dead neurons: {avg_dead_pct:.1f}%")
    print(f"  Spectral concentration (top 50%): {avg_spec_50:.1%}")

    # Generate recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # 1. Hidden dimension sizing
    if avg_rank_util < 0.40:
        recommendations.append({
            "issue": "HIDDEN DIM TOO LARGE",
            "severity": "⚠️  WARNING",
            "finding": f"Only {avg_rank_util:.1%} of hidden dimension capacity is used",
            "current": f"hidden_dim = {hidden_dim}",
            "suggestion": f"REDUCE hidden_dim to {hidden_dim // 2} or {hidden_dim // 4}",
            "rationale": "Projector is over-parameterized, wasting compute and memory"
        })
    elif avg_rank_util < 0.60:
        recommendations.append({
            "issue": "HIDDEN DIM MODERATELY LARGE",
            "severity": "💡 OPTIMIZATION",
            "finding": f"Only {avg_rank_util:.1%} of hidden dimension capacity is used",
            "current": f"hidden_dim = {hidden_dim}",
            "suggestion": f"Consider reducing hidden_dim to {hidden_dim // 2}",
            "rationale": "Could reduce parameters without hurting performance"
        })
    elif avg_rank_util > 0.85:
        recommendations.append({
            "issue": "HIDDEN DIM MAY BE TOO SMALL",
            "severity": "⚠️  WARNING",
            "finding": f"Using {avg_rank_util:.1%} of hidden dimension capacity",
            "current": f"hidden_dim = {hidden_dim}",
            "suggestion": f"Consider INCREASING hidden_dim to {hidden_dim * 2}",
            "rationale": "Projector may be capacity-limited, hurting adaptation quality"
        })
    else:
        recommendations.append({
            "issue": "HIDDEN DIM SIZING",
            "severity": "✓ GOOD",
            "finding": f"Using {avg_rank_util:.1%} of hidden dimension capacity",
            "current": f"hidden_dim = {hidden_dim}",
            "suggestion": "No change needed",
            "rationale": "Hidden dimension is appropriately sized"
        })

    # 2. Dead neurons
    if avg_dead_pct > 15:
        recommendations.append({
            "issue": "MANY DEAD NEURONS",
            "severity": "⚠️  WARNING",
            "finding": f"{avg_dead_pct:.1f}% of neurons have minimal activation",
            "current": "May indicate poor initialization or training",
            "suggestion": "Reduce hidden_dim or improve initialization",
            "rationale": "Dead neurons waste parameters without contributing to learning"
        })
    elif avg_dead_pct > 5:
        recommendations.append({
            "issue": "SOME DEAD NEURONS",
            "severity": "💡 OPTIMIZATION",
            "finding": f"{avg_dead_pct:.1f}% of neurons have minimal activation",
            "current": "Acceptable but not optimal",
            "suggestion": "Could reduce hidden_dim slightly",
            "rationale": "Minor parameter efficiency gain possible"
        })

    # 3. Spectral concentration
    if avg_spec_50 > 0.90:
        recommendations.append({
            "issue": "HIGH SPECTRAL CONCENTRATION",
            "severity": "💡 INFO",
            "finding": f"Top 50% of dimensions capture {avg_spec_50:.1%} of variance",
            "current": "Projector learning is concentrated in few directions",
            "suggestion": "This is normal - projector compresses information",
            "rationale": "Expected behavior for a dimensionality reduction layer"
        })

    # 4. Intrinsic dimensionality comparison
    intrinsic_90 = np.mean([r["intrinsic_dim_90"] for r in results.values() if "weight" in r["name"]])
    intrinsic_95 = np.mean([r["intrinsic_dim_95"] for r in results.values() if "weight" in r["name"]])

    hidden_dim_float = float(hidden_dim)
    if intrinsic_95 < hidden_dim_float * 0.3:
        recommendations.append({
            "issue": "LOW INTRINSIC DIMENSIONALITY",
            "severity": "💡 OPTIMIZATION",
            "finding": f"95% of variance captured by only {intrinsic_95:.0f} dims (out of {hidden_dim})",
            "current": f"Effective dimensionality is {intrinsic_95 / hidden_dim_float:.1%} of hidden_dim",
            "suggestion": f"Could reduce hidden_dim to {int(intrinsic_95 * 2)} or {int(intrinsic_95 * 3)}",
            "rationale": "Most of the hidden dimension is redundant"
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']} - {rec['severity']}")
        print(f"   Finding: {rec['finding']}")
        print(f"   Current: {rec['current']}")
        print(f"   Suggestion: {rec['suggestion']}")
        print(f"   Rationale: {rec['rationale']}")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze projector sizing and efficiency")
    parser.add_argument("--model-path", type=str, default="mazesmazes/tiny-audio",
                       help="HuggingFace Hub model path or local directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Optional JSON output file")

    args = parser.parse_args()

    print("=" * 80)
    print("AUDIO PROJECTOR ANALYSIS")
    print("=" * 80)

    # Load weights
    weights = load_projector_weights(args.model_path)

    # Analyze each weight matrix (skip 1D tensors like LayerNorm)
    results = {}
    for name, weight in weights.items():
        if len(weight.shape) < 2:
            print(f"\nSkipping {name} (1D tensor - LayerNorm)")
            continue
        print(f"\nAnalyzing {name}...")
        results[name] = analyze_weight_matrix(weight, name)

    # Generate recommendations
    recommendations = generate_projector_recommendations(results)

    # Save results if requested
    if args.output:
        output_data = {
            "model_path": args.model_path,
            "layer_results": results,
            "recommendations": recommendations,
        }

        # Convert numpy types to Python types for JSON serialization
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output_data = convert_to_python(output_data)

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
