#!/usr/bin/env python3
"""
Analyze decoder LoRA adapters from a trained model to determine:
1. How well the LoRA is being trained (rank utilization)
2. Whether rank/alpha should be adjusted
3. Training quality metrics and recommendations
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers.utils import cached_file


def load_lora_weights(model_path: str):
    """Load decoder LoRA weights from Hub or local path."""
    print(f"Loading decoder LoRA weights from: {model_path}")

    # Try loading from last-checkpoint first
    try:
        decoder_file = cached_file(
            model_path,
            "decoder.safetensors",
            subfolder="last-checkpoint",
            _raise_exceptions_for_missing_entries=False,
        )
    except Exception:
        # Fallback to main directory
        decoder_file = cached_file(
            model_path,
            "decoder.safetensors",
            _raise_exceptions_for_missing_entries=False,
        )

    if not decoder_file:
        raise FileNotFoundError(f"Could not find decoder.safetensors in {model_path}")

    from safetensors.torch import load_file
    weights = load_file(decoder_file)

    print(f"✓ Loaded {len(weights)} weight tensors")
    return weights


def analyze_lora_layer(lora_A: torch.Tensor, lora_B: torch.Tensor, layer_name: str):
    """
    Analyze a single LoRA layer using SVD and other metrics.

    Args:
        lora_A: Down-projection matrix [r, d]
        lora_B: Up-projection matrix [d, r]
        layer_name: Name of the layer

    Returns:
        dict with analysis results
    """
    # LoRA matrices: output = B @ A @ input
    # Effective weight update: ΔW = B @ A (shape [d_out, d_in])
    effective_weight = lora_B @ lora_A

    # Compute SVD of effective weight
    U, S, Vh = torch.linalg.svd(effective_weight, full_matrices=False)

    # Normalize singular values
    S_normalized = S / S.sum()

    # Calculate metrics
    rank = lora_A.shape[0]  # LoRA rank

    # 1. Effective rank (number of "significant" singular values)
    # Based on Shannon entropy
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    # 2. Stable rank (Frobenius norm / spectral norm)
    stable_rank = (S ** 2).sum().item() / (S[0] ** 2).item()

    # 3. Rank utilization (how much of allocated rank is used)
    rank_utilization = effective_rank / rank

    # 4. Spectral decay (how fast singular values decay)
    # Measure ratio of top-k to total
    top_k = min(rank // 2, len(S))
    spectral_concentration = S[:top_k].sum() / S.sum()

    # 5. Weight magnitude statistics
    weight_norm = torch.norm(effective_weight).item()
    weight_mean = effective_weight.abs().mean().item()
    weight_std = effective_weight.std().item()

    # 6. Gradient flow proxy (ratio of smallest to largest SV)
    condition_number = S[0].item() / (S[-1].item() + 1e-10)

    return {
        "layer_name": layer_name,
        "rank": rank,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "rank_utilization": rank_utilization,
        "spectral_concentration": spectral_concentration.item(),
        "weight_norm": weight_norm,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "condition_number": condition_number,
        "singular_values": S.cpu().numpy().tolist(),
    }


def generate_recommendations(results: list[dict], lora_config: dict):
    """Generate recommendations based on analysis results."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    avg_utilization = np.mean([r["rank_utilization"] for r in results])
    avg_effective_rank = np.mean([r["effective_rank"] for r in results])
    avg_concentration = np.mean([r["spectral_concentration"] for r in results])
    current_rank = results[0]["rank"]

    print(f"\nCurrent LoRA Configuration:")
    print(f"  Rank (r): {lora_config.get('r', current_rank)}")
    print(f"  Alpha: {lora_config.get('lora_alpha', 'unknown')}")
    print(f"  Target modules: {lora_config.get('target_modules', 'unknown')}")

    print(f"\nTraining Quality Metrics:")
    print(f"  Average rank utilization: {avg_utilization:.1%}")
    print(f"  Average effective rank: {avg_effective_rank:.2f} / {current_rank}")
    print(f"  Spectral concentration (top 50%): {avg_concentration:.1%}")

    recommendations = []

    # Recommendation 1: Rank sizing
    if avg_utilization > 0.85:
        recommendations.append({
            "issue": "HIGH RANK UTILIZATION",
            "severity": "⚠️  WARNING",
            "finding": f"Average rank utilization is {avg_utilization:.1%}",
            "explanation": "LoRA adapters are using most of their allocated capacity",
            "action": f"Consider INCREASING rank from r={current_rank} to r={current_rank * 2}",
            "rationale": "High utilization suggests the model needs more capacity to represent the learned adaptation"
        })
    elif avg_utilization < 0.50:
        recommendations.append({
            "issue": "LOW RANK UTILIZATION",
            "severity": "💡 OPTIMIZATION",
            "finding": f"Average rank utilization is only {avg_utilization:.1%}",
            "explanation": "LoRA adapters are underutilizing their allocated capacity",
            "action": f"Consider DECREASING rank from r={current_rank} to r={max(4, current_rank // 2)}",
            "rationale": "Lower rank would be more parameter-efficient without losing performance"
        })
    else:
        recommendations.append({
            "issue": "RANK SIZING",
            "severity": "✓ GOOD",
            "finding": f"Rank utilization is {avg_utilization:.1%}",
            "explanation": "LoRA rank appears well-sized for the task",
            "action": "No change needed",
            "rationale": "Current rank provides good balance of capacity and efficiency"
        })

    # Recommendation 2: Alpha tuning
    current_alpha = lora_config.get('lora_alpha', current_rank)
    scaling_factor = current_alpha / current_rank

    if scaling_factor < 0.5:
        recommendations.append({
            "issue": "LOW ALPHA SCALING",
            "severity": "💡 OPTIMIZATION",
            "finding": f"Alpha/rank ratio is {scaling_factor:.2f} (alpha={current_alpha}, r={current_rank})",
            "explanation": "Low scaling factor may cause slow adaptation",
            "action": f"Consider INCREASING alpha from {current_alpha} to {current_rank}",
            "rationale": "Higher alpha (alpha/r = 1.0) is standard and may improve training speed"
        })
    elif scaling_factor > 2.0:
        recommendations.append({
            "issue": "HIGH ALPHA SCALING",
            "severity": "💡 OPTIMIZATION",
            "finding": f"Alpha/rank ratio is {scaling_factor:.2f} (alpha={current_alpha}, r={current_rank})",
            "explanation": "High scaling may cause training instability",
            "action": f"Consider DECREASING alpha from {current_alpha} to {current_rank * 2}",
            "rationale": "Lower alpha (alpha/r = 1-2) provides more stable training"
        })
    else:
        recommendations.append({
            "issue": "ALPHA SCALING",
            "severity": "✓ GOOD",
            "finding": f"Alpha/rank ratio is {scaling_factor:.2f}",
            "explanation": "Alpha scaling factor is in recommended range",
            "action": "No change needed",
            "rationale": "Current alpha provides appropriate learning rate scaling"
        })

    # Recommendation 3: Training quality
    if avg_concentration > 0.85:
        recommendations.append({
            "issue": "HIGH SPECTRAL CONCENTRATION",
            "severity": "⚠️  WARNING",
            "finding": f"Top 50% of singular values account for {avg_concentration:.1%} of total",
            "explanation": "Weight updates are concentrated in few directions",
            "action": "Training may be converging to a low-rank solution",
            "rationale": "Consider increasing rank or adding more diverse training data"
        })

    # Recommendation 4: Gradient flow
    avg_condition = np.mean([r["condition_number"] for r in results])
    if avg_condition > 100:
        recommendations.append({
            "issue": "POOR CONDITIONING",
            "severity": "⚠️  WARNING",
            "finding": f"Average condition number is {avg_condition:.1f}",
            "explanation": "Large gap between largest and smallest singular values",
            "action": "May indicate gradient flow issues",
            "rationale": "Consider lowering learning rate or adding gradient clipping"
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']} - {rec['severity']}")
        print(f"   Finding: {rec['finding']}")
        print(f"   Explanation: {rec['explanation']}")
        print(f"   Action: {rec['action']}")
        print(f"   Rationale: {rec['rationale']}")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze decoder LoRA training quality")
    parser.add_argument("--model-path", type=str, default="mazesmazes/tiny-audio",
                       help="HuggingFace Hub model path or local directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Optional JSON output file")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed per-layer analysis")

    args = parser.parse_args()

    print("=" * 80)
    print("DECODER LORA ANALYSIS")
    print("=" * 80)

    # Load weights
    weights = load_lora_weights(args.model_path)

    # Load LoRA config
    try:
        from transformers.utils import cached_file
        config_file = cached_file(
            args.model_path,
            "decoder_lora_config.json",
            subfolder="last-checkpoint",
            _raise_exceptions_for_missing_entries=False,
        )
        if not config_file:
            config_file = cached_file(args.model_path, "decoder_lora_config.json")

        with open(config_file) as f:
            lora_config = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LoRA config: {e}")
        lora_config = {}

    # Group weights by layer
    lora_layers = {}
    for name, weight in weights.items():
        if "lora_A" in name or "lora_B" in name:
            # Extract base layer name (handle .default.weight suffix)
            base_name = name.replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
            base_name = base_name.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            if base_name not in lora_layers:
                lora_layers[base_name] = {}

            if "lora_A" in name:
                lora_layers[base_name]["A"] = weight
            else:
                lora_layers[base_name]["B"] = weight

    print(f"\nFound {len(lora_layers)} LoRA adapter pairs")

    # Analyze each layer
    results = []
    for layer_name, matrices in lora_layers.items():
        if "A" not in matrices or "B" not in matrices:
            continue

        result = analyze_lora_layer(matrices["A"], matrices["B"], layer_name)
        results.append(result)

        if args.verbose:
            print(f"\n{layer_name}:")
            print(f"  Rank: {result['rank']}")
            print(f"  Effective rank: {result['effective_rank']:.2f}")
            print(f"  Rank utilization: {result['rank_utilization']:.1%}")
            print(f"  Spectral concentration: {result['spectral_concentration']:.1%}")

    # Generate recommendations
    recommendations = generate_recommendations(results, lora_config)

    # Save results if requested
    if args.output:
        output_data = {
            "model_path": args.model_path,
            "lora_config": lora_config,
            "layer_results": results,
            "recommendations": recommendations,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
