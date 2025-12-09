#!/usr/bin/env python3
"""
MOSA Checkpoint Forensic Analysis Tool

Usage:
    python scripts/analyze_moe.py path/to/checkpoint/model.safetensors

Analyzes MOSA-style projector (Mixture of Simple Adapters):
1. Integrity: Checks for NaNs or Infinity values
2. Router Health: Checks router weights and biases
3. Conv Module: Analyzes convolutional downsampling layers
4. Expert Diversity: Checks if adapters are specialized or clones
5. Spectral Health: Uses SVD to check for rank collapse
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as f
from safetensors.torch import load_file


def draw_ascii_bar(
    values: list[float],
    labels: Optional[list[str]] = None,
    max_width: int = 40,
    denominator: Optional[float] = None,
):
    """Draws a visual comparison bar chart with optional percentage."""
    if len(values) == 0:
        return

    values = np.array(values)
    min_val, max_val = values.min(), values.max()
    range_val = max_val - min_val if max_val != min_val else 1.0

    print("-" * 75)
    for i, val in enumerate(values):
        normalized = (val - min_val) / range_val if range_val > 0 else 0.5
        width = int(normalized * max_width)
        bar = "█" * width

        label = f"{labels[i]:<6}" if labels else f"{i:<6}"
        if denominator is not None:
            pct = (val / denominator) * 100
            print(f"{label} | {bar:<{max_width}} {val:.1f} ({pct:.1f}%)")
        else:
            print(f"{label} | {bar:<{max_width}} {val:.6f}")
    print("-" * 75)


def compute_effective_rank(matrix: torch.Tensor) -> float:
    """
    Computes Shannon Entropy of Singular Values.
    High Rank = Matrix is using all its dimensions (Healthy).
    Low Rank = Matrix has collapsed to a simple linear transformation (Bad).
    """
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)

    matrix = matrix.float().cpu()

    try:
        s = torch.linalg.svdvals(matrix)
        p = s / s.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        return torch.exp(entropy).item()
    except Exception:
        return 0.0


def check_integrity(name: str, tensor: torch.Tensor) -> bool:
    """Returns True if tensor is healthy, False if NaN/Inf."""
    if torch.isnan(tensor).any():
        print(f"   ❌ CRITICAL FAILURE: NaNs detected in {name}")
        return False
    if torch.isinf(tensor).any():
        print(f"   ❌ CRITICAL FAILURE: Infinity detected in {name}")
        return False
    return True


def resolve_model_path(path_or_repo: str) -> str:
    """Handles local paths or HuggingFace repo downloads."""
    path = Path(path_or_repo)
    if path.exists():
        if path.is_dir():
            candidate = path / "model.safetensors"
            if candidate.exists():
                return str(candidate)
        return path_or_repo

    if "/" in path_or_repo and not path_or_repo.endswith(".safetensors"):
        try:
            from huggingface_hub import hf_hub_download

            print(f"📥 Downloading from Hub: {path_or_repo}")
            return hf_hub_download(repo_id=path_or_repo, filename="model.safetensors")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return path_or_repo
    return path_or_repo


def analyze_checkpoint(file_path: str):
    print("\n🔬 MOSA PROJECTOR FORENSIC ANALYSIS")
    print(f"   Target: {file_path}")
    print("=" * 65)

    file_path = resolve_model_path(file_path)
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    try:
        tensors = load_file(file_path)
    except Exception as e:
        print(f"❌ Failed to load safetensors: {e}")
        return

    # Filter for projector weights
    proj_weights = {k: v for k, v in tensors.items() if "projector" in k}

    if not proj_weights:
        print("⚠️  No 'projector' prefix found. Searching for generic keys...")
        proj_weights = {
            k: v for k, v in tensors.items() if "experts" in k or "router" in k or "conv" in k
        }

    if not proj_weights:
        print("❌ No MOSA parameters found in this checkpoint.")
        return

    print(f"📊 Loaded {len(proj_weights)} projector parameters.\n")

    # Print all parameter names and shapes
    print("   Parameters found:")
    for k, v in sorted(proj_weights.items()):
        print(f"      {k}: {list(v.shape)}")
    print()

    # Organize weights by component
    conv_weights = {}
    router_weights = {}
    experts_map = defaultdict(dict)

    healthy = True
    for k, v in proj_weights.items():
        if not check_integrity(k, v):
            healthy = False

        # Conv layers
        if "conv" in k.lower():
            conv_weights[k] = v

        # Router layers
        if "router" in k.lower():
            router_weights[k] = v

        # Expert/Adapter layers
        match = re.search(r"experts\.(\d+)\.(fc1|fc2)\.(weight|bias)", k)
        if match:
            idx = int(match.group(1))
            layer = match.group(2)
            param = match.group(3)
            experts_map[idx][f"{layer}.{param}"] = v

    if not healthy:
        print("\n❌ ABORTING ANALYSIS: Model weights are corrupted (NaN/Inf).")
        return

    sorted_indices = sorted(experts_map.keys())
    num_experts = len(sorted_indices)

    print(
        f"   Found: {len(conv_weights)} conv params, {len(router_weights)} router params, {num_experts} experts"
    )

    # 1. CONV MODULE DIAGNOSTICS
    print("\n" + "=" * 65)
    print("[1] CONV MODULE DIAGNOSTICS")
    print("=" * 65)
    if conv_weights:
        for name, w in sorted(conv_weights.items()):
            if "weight" in name:
                std = w.float().std().item()
                mean = w.float().mean().item()
                norm = torch.linalg.norm(w.float()).item()
                print(f"   {name}:")
                print(
                    f"      Shape: {list(w.shape)}, Std: {std:.5f}, Mean: {mean:.5f}, Norm: {norm:.2f}"
                )
    else:
        print("   ❌ No conv weights found!")

    # 2. ROUTER DIAGNOSTICS
    print("\n" + "=" * 65)
    print("[2] ROUTER DIAGNOSTICS")
    print("=" * 65)
    if router_weights:
        for name, w in sorted(router_weights.items()):
            if "weight" in name:
                std = w.float().std().item()
                mean = w.float().mean().item()
                norm = torch.linalg.norm(w.float()).item()
                print(f"   {name}:")
                print(
                    f"      Shape: {list(w.shape)}, Std: {std:.5f}, Mean: {mean:.5f}, Norm: {norm:.2f}"
                )

                # For the final router layer (outputs num_experts), check per-expert norms
                if w.shape[0] == num_experts:
                    expert_norms = torch.linalg.norm(w.float(), dim=1).cpu().numpy()
                    print("\n   Per-Expert Router Output Norms:")
                    draw_ascii_bar(
                        expert_norms,
                        labels=[f"E{i}" for i in range(len(expert_norms))],
                        max_width=40,
                    )

                    norm_std = np.std(expert_norms)
                    norm_ratio = np.max(expert_norms) / (np.min(expert_norms) + 1e-9)
                    print(f"   Norm Std Dev: {norm_std:.5f}, Max/Min Ratio: {norm_ratio:.2f}x")

                    if norm_ratio > 3.0:
                        print("   ⚠️  Uneven router norms: Some experts may be favored.")
                    else:
                        print("   ✅ Router output weights look balanced.")
    else:
        print("   ❌ No router weights found!")

    # 3. EXPERT DIVERSITY
    print("\n" + "=" * 65)
    print("[3] EXPERT DIVERSITY (Adapter Specialization)")
    print("=" * 65)

    if num_experts > 1:
        # Get fc1 weights for each expert
        fc1_weights = []
        for idx in sorted_indices:
            if "fc1.weight" in experts_map[idx]:
                fc1_weights.append(experts_map[idx]["fc1.weight"])

        if len(fc1_weights) > 1:
            # Stack and flatten: [Num_Experts, Total_Params]
            flat_experts = torch.stack([w.view(-1) for w in fc1_weights]).float()
            norm_experts = f.normalize(flat_experts, p=2, dim=1)

            # Cosine Similarity Matrix
            sim_matrix = torch.mm(norm_experts, norm_experts.t())

            # Mask diagonal
            mask = ~torch.eye(num_experts, dtype=torch.bool, device=sim_matrix.device)
            off_diag = sim_matrix[mask]

            avg_sim = off_diag.mean().item()
            max_sim = off_diag.max().item()
            min_sim = off_diag.min().item()

            print("   FC1 Weight Similarity (cosine):")
            print(f"      Avg: {avg_sim:.4f}, Max: {max_sim:.4f}, Min: {min_sim:.4f}")

            if avg_sim > 0.98:
                print("   ❌ RED ZONE: Experts are identical clones - no specialization!")
            elif avg_sim > 0.80:
                print("   ⚠️  YELLOW ZONE: Experts are highly correlated.")
            elif avg_sim > 0.50:
                print("   ✅ GREEN ZONE: Experts are moderately specialized.")
            else:
                print("   ✅ GREEN ZONE: Experts are highly specialized (diverse).")

            # Show pairwise similarities
            print("\n   Pairwise Similarity Matrix:")
            print("        ", end="")
            for i in sorted_indices:
                print(f"  E{i}   ", end="")
            print()
            for i, idx_i in enumerate(sorted_indices):
                print(f"   E{idx_i}  ", end="")
                for j, _idx_j in enumerate(sorted_indices):
                    sim = sim_matrix[i, j].item()
                    print(f" {sim:.3f} ", end="")
                print()
    else:
        print("   ⚠️  Only 1 expert found - cannot compute diversity.")

    # 4. EXPERT MAGNITUDE ANALYSIS
    print("\n" + "=" * 65)
    print("[4] EXPERT MAGNITUDE ANALYSIS")
    print("=" * 65)

    fc1_mags = []
    fc2_mags = []
    for idx in sorted_indices:
        if "fc1.weight" in experts_map[idx]:
            fc1_mags.append(experts_map[idx]["fc1.weight"].abs().mean().item())
        if "fc2.weight" in experts_map[idx]:
            fc2_mags.append(experts_map[idx]["fc2.weight"].abs().mean().item())

    if fc1_mags:
        print("   FC1 Weight Magnitudes:")
        draw_ascii_bar(fc1_mags, labels=[f"E{i}" for i in sorted_indices], max_width=40)

    if fc2_mags:
        print("   FC2 Weight Magnitudes:")
        draw_ascii_bar(fc2_mags, labels=[f"E{i}" for i in sorted_indices], max_width=40)

    # Check for magnitude imbalance
    if fc1_mags:
        ratio = max(fc1_mags) / (min(fc1_mags) + 1e-9)
        if ratio > 2.0:
            print(f"   ⚠️  FC1 magnitude imbalance: {ratio:.2f}x between experts")
        else:
            print(f"   ✅ FC1 magnitudes balanced (ratio: {ratio:.2f}x)")

    # 5. SPECTRAL HEALTH
    print("\n" + "=" * 65)
    print("[5] SPECTRAL HEALTH (Effective Rank)")
    print("=" * 65)

    if num_experts > 0 and "fc1.weight" in experts_map[sorted_indices[0]]:
        ranks = []
        for idx in sorted_indices:
            if "fc1.weight" in experts_map[idx]:
                rank = compute_effective_rank(experts_map[idx]["fc1.weight"])
                ranks.append(rank)

        if ranks:
            full_rank = min(experts_map[sorted_indices[0]]["fc1.weight"].shape)
            avg_rank = np.mean(ranks)

            print("   FC1 Effective Rank (higher = more expressive):")
            print(
                f"      Average: {avg_rank:.1f} / {full_rank} ({(avg_rank / full_rank) * 100:.1f}%)"
            )
            draw_ascii_bar(ranks, labels=[f"E{i}" for i in sorted_indices], denominator=full_rank)

            if avg_rank / full_rank < 0.3:
                print("   ⚠️  Low effective rank - possible feature collapse.")
            else:
                print("   ✅ Healthy effective rank.")

    # 6. WEIGHT MAGNITUDE ANALYSIS (Weight Decay Optimization)
    print("\n" + "=" * 65)
    print("[6] WEIGHT MAGNITUDE ANALYSIS (Weight Decay Optimization)")
    print("=" * 65)

    # Collect all weight tensors by layer type
    all_weights = {"conv": [], "router": [], "expert_fc1": [], "expert_fc2": [], "ln": []}

    for name, w in proj_weights.items():
        if "weight" not in name:
            continue
        w_flat = w.float().view(-1)
        stats = {
            "name": name,
            "l2_norm": torch.linalg.norm(w_flat).item(),
            "mean_abs": w_flat.abs().mean().item(),
            "std": w_flat.std().item(),
            "max_abs": w_flat.abs().max().item(),
            "sparsity": (w_flat.abs() < 1e-6).float().mean().item() * 100,
        }

        if "conv" in name.lower():
            all_weights["conv"].append(stats)
        elif "router" in name.lower():
            all_weights["router"].append(stats)
        elif "fc1" in name.lower():
            all_weights["expert_fc1"].append(stats)
        elif "fc2" in name.lower():
            all_weights["expert_fc2"].append(stats)
        elif "ln" in name.lower() or "norm" in name.lower():
            all_weights["ln"].append(stats)

    print("\n   Weight Statistics by Layer Type:")
    print("   " + "-" * 70)
    print(
        f"   {'Layer Type':<15} {'L2 Norm':<12} {'Mean |W|':<12} {'Std':<12} {'Max |W|':<12} {'Sparsity':<10}"
    )
    print("   " + "-" * 70)

    layer_summaries = {}
    for layer_type, weights_list in all_weights.items():
        if not weights_list:
            continue
        avg_l2 = np.mean([w["l2_norm"] for w in weights_list])
        avg_mean = np.mean([w["mean_abs"] for w in weights_list])
        avg_std = np.mean([w["std"] for w in weights_list])
        avg_max = np.mean([w["max_abs"] for w in weights_list])
        avg_sparsity = np.mean([w["sparsity"] for w in weights_list])
        layer_summaries[layer_type] = {
            "l2": avg_l2,
            "mean": avg_mean,
            "std": avg_std,
            "max": avg_max,
            "sparsity": avg_sparsity,
        }
        print(
            f"   {layer_type:<15} {avg_l2:<12.4f} {avg_mean:<12.6f} {avg_std:<12.6f} {avg_max:<12.4f} {avg_sparsity:<10.2f}%"
        )

    print("\n   Weight Decay Recommendations:")

    # Check for weight magnitude issues
    all_l2_norms = [s["l2"] for s in layer_summaries.values() if "l2" in s]
    if all_l2_norms:
        avg_l2 = np.mean(all_l2_norms)
        if avg_l2 > 100:
            print("   ⚠️  HIGH weight magnitudes detected (L2 > 100)")
            print("      → Consider INCREASING weight_decay (try 0.05-0.1)")
            print("      → This may indicate insufficient regularization")
        elif avg_l2 < 1:
            print("   ⚠️  LOW weight magnitudes detected (L2 < 1)")
            print("      → Consider DECREASING weight_decay (try 0.001-0.005)")
            print("      → Weights may be over-regularized")
        else:
            print(f"   ✅ Weight magnitudes look healthy (avg L2: {avg_l2:.2f})")
            print("      → Current weight_decay setting appears appropriate")

    # Check sparsity
    all_sparsity = [s["sparsity"] for s in layer_summaries.values() if "sparsity" in s]
    if all_sparsity:
        avg_sparsity = np.mean(all_sparsity)
        if avg_sparsity > 10:
            print(f"   ⚠️  High sparsity detected ({avg_sparsity:.1f}% near-zero weights)")
            print("      → May indicate over-regularization or dead neurons")

    # 7. DROPOUT ANALYSIS
    print("\n" + "=" * 65)
    print("[7] DROPOUT ANALYSIS (Activation Patterns)")
    print("=" * 65)

    # Analyze weight distributions for signs of dropout effects
    print("\n   Expert Weight Distribution Analysis:")

    expert_stats = []
    for idx in sorted_indices:
        if "fc1.weight" in experts_map[idx] and "fc2.weight" in experts_map[idx]:
            fc1_w = experts_map[idx]["fc1.weight"].float()
            fc2_w = experts_map[idx]["fc2.weight"].float()

            # Compute statistics
            fc1_var = fc1_w.var().item()
            fc2_var = fc2_w.var().item()
            fc1_kurtosis = (
                ((fc1_w - fc1_w.mean()) ** 4).mean() / (fc1_w.var() ** 2 + 1e-10)
            ).item()
            fc2_kurtosis = (
                ((fc2_w - fc2_w.mean()) ** 4).mean() / (fc2_w.var() ** 2 + 1e-10)
            ).item()

            # Check for "bursty" patterns (high kurtosis = heavy tails)
            expert_stats.append(
                {
                    "idx": idx,
                    "fc1_var": fc1_var,
                    "fc2_var": fc2_var,
                    "fc1_kurtosis": fc1_kurtosis,
                    "fc2_kurtosis": fc2_kurtosis,
                }
            )

    if expert_stats:
        print(
            f"   {'Expert':<10} {'FC1 Var':<12} {'FC2 Var':<12} {'FC1 Kurt':<12} {'FC2 Kurt':<12}"
        )
        print("   " + "-" * 58)
        for s in expert_stats:
            print(
                f"   E{s['idx']:<9} {s['fc1_var']:<12.6f} {s['fc2_var']:<12.6f} {s['fc1_kurtosis']:<12.2f} {s['fc2_kurtosis']:<12.2f}"
            )

        avg_kurtosis = np.mean([s["fc1_kurtosis"] + s["fc2_kurtosis"] for s in expert_stats]) / 2
        var_ratio = np.std([s["fc1_var"] for s in expert_stats]) / (
            np.mean([s["fc1_var"] for s in expert_stats]) + 1e-10
        )

        print("\n   Dropout Recommendations:")
        if avg_kurtosis > 5:
            print(f"   ⚠️  High kurtosis detected (avg: {avg_kurtosis:.2f})")
            print("      → Weight distributions have heavy tails")
            print("      → Consider INCREASING dropout (try 0.1-0.2)")
            print("      → This can help prevent co-adaptation")
        elif avg_kurtosis < 2.5:
            print(f"   ℹ️  Low kurtosis (avg: {avg_kurtosis:.2f}) - near-Gaussian")
            print("      → Weights are well-distributed")
            print("      → Current dropout appears appropriate")
        else:
            print(f"   ✅ Kurtosis in normal range (avg: {avg_kurtosis:.2f})")

        if var_ratio > 0.5:
            print(f"   ⚠️  High variance ratio across experts ({var_ratio:.2f})")
            print("      → Experts have uneven weight scales")
            print("      → Dropout may help equalize expert contributions")

    # Check for dead/saturated neurons by looking at bias distributions
    print("\n   Bias Analysis (Dead Neuron Detection):")
    bias_stats = []
    for name, w in proj_weights.items():
        if "bias" in name and "fc" in name:
            w_flat = w.float()
            negative_pct = (w_flat < -1).float().mean().item() * 100
            large_pct = (w_flat.abs() > 2).float().mean().item() * 100
            bias_stats.append({"name": name, "neg_pct": negative_pct, "large_pct": large_pct})

    if bias_stats:
        avg_neg = np.mean([b["neg_pct"] for b in bias_stats])
        avg_large = np.mean([b["large_pct"] for b in bias_stats])

        if avg_neg > 20:
            print(f"   ⚠️  {avg_neg:.1f}% of FC biases are strongly negative (<-1)")
            print("      → May indicate dead ReLU neurons")
            print("      → Consider reducing dropout or using LeakyReLU")
        elif avg_large > 10:
            print(f"   ⚠️  {avg_large:.1f}% of FC biases have large magnitude (>2)")
            print("      → May indicate saturation issues")
        else:
            print("   ✅ Bias distributions look healthy")

    # 8. ROUTER ENTROPY ANALYSIS
    print("\n" + "=" * 65)
    print("[8] ROUTER ENTROPY ANALYSIS")
    print("=" * 65)

    # Find the final router layer (outputs to num_experts)
    final_router_weight = None
    final_router_bias = None
    for name, w in router_weights.items():
        if "weight" in name and w.shape[0] == num_experts:
            final_router_weight = w.float()
        if "bias" in name and w.shape[0] == num_experts:
            final_router_bias = w.float()

    if final_router_weight is not None:
        # Analyze the bias to understand routing preferences
        if final_router_bias is not None:
            # Softmax of biases gives "default" routing without input
            default_probs = torch.softmax(final_router_bias, dim=0)
            entropy = -torch.sum(default_probs * torch.log(default_probs + 1e-10)).item()
            max_entropy = np.log(num_experts)
            normalized_entropy = entropy / max_entropy

            print("   Router Bias Analysis (default routing preferences):")
            print(f"      Bias values: {final_router_bias.cpu().numpy().round(3)}")
            print(f"      Default probs: {default_probs.cpu().numpy().round(3)}")
            print(
                f"      Entropy: {entropy:.3f} / {max_entropy:.3f} ({normalized_entropy * 100:.1f}% of max)"
            )

            draw_ascii_bar(
                default_probs.cpu().numpy(),
                labels=[f"E{i}" for i in range(num_experts)],
                max_width=40,
            )

            print("\n   Router Recommendations:")
            if normalized_entropy > 0.95:
                print("   ✅ Near-uniform routing - all experts contribute equally")
                print("      → No router regularization needed")
            elif normalized_entropy > 0.7:
                print("   ✅ Balanced routing with some specialization")
            elif normalized_entropy > 0.4:
                print("   ⚠️  Moderate routing imbalance")
                print("      → Consider adding load balancing loss")
            else:
                print("   ❌ Severe routing collapse - few experts dominate")
                print("      → Strongly recommend adding auxiliary load balancing loss")
                print("      → Or increase router temperature during training")
        else:
            print("   ⚠️  No router bias found - cannot analyze default routing")

    # 9. CONDITION NUMBER ANALYSIS
    print("\n" + "=" * 65)
    print("[9] CONDITION NUMBER ANALYSIS (Numerical Stability)")
    print("=" * 65)

    condition_numbers = {"conv": [], "router": [], "expert": []}

    for name, w in proj_weights.items():
        if "weight" not in name:
            continue
        if w.dim() < 2:
            continue

        # Reshape to 2D for SVD
        w_2d = w.float().view(w.shape[0], -1)
        if w_2d.shape[0] > 1 and w_2d.shape[1] > 1:
            try:
                s = torch.linalg.svdvals(w_2d)
                cond = (s[0] / (s[-1] + 1e-10)).item()

                if "conv" in name.lower():
                    condition_numbers["conv"].append(cond)
                elif "router" in name.lower():
                    condition_numbers["router"].append(cond)
                elif "fc" in name.lower():
                    condition_numbers["expert"].append(cond)
            except Exception:
                pass

    print("\n   Condition Numbers by Layer Type (lower is better):")
    print("   " + "-" * 50)
    for layer_type, conds in condition_numbers.items():
        if conds:
            avg_cond = np.mean(conds)
            max_cond = np.max(conds)
            status = "✅" if max_cond < 100 else "⚠️" if max_cond < 1000 else "❌"
            print(f"   {layer_type:<10} Avg: {avg_cond:<10.1f} Max: {max_cond:<10.1f} {status}")

    all_conds = sum(condition_numbers.values(), [])
    if all_conds:
        max_cond = max(all_conds)
        print("\n   Learning Rate Recommendations:")
        if max_cond > 1000:
            print(f"   ❌ Very high condition number ({max_cond:.0f})")
            print("      → DECREASE learning rate (try 0.5x current)")
            print("      → High condition numbers cause unstable gradients")
        elif max_cond > 100:
            print(f"   ⚠️  Elevated condition number ({max_cond:.0f})")
            print("      → Consider slightly lower learning rate")
        else:
            print(f"   ✅ Condition numbers healthy (max: {max_cond:.0f})")
            print("      → Learning rate appears appropriate")

    # 10. INITIALIZATION DEVIATION ANALYSIS
    print("\n" + "=" * 65)
    print("[10] INITIALIZATION DEVIATION ANALYSIS")
    print("=" * 65)

    # Compare current std to typical init std (0.02 for this model)
    init_std = 0.02
    print(f"\n   Comparing to expected init std: {init_std}")
    print("   " + "-" * 60)

    deviation_ratios = []
    for name, w in proj_weights.items():
        if "weight" not in name:
            continue
        if "ln" in name.lower() or "norm" in name.lower():
            continue  # Skip norm layers

        current_std = w.float().std().item()
        ratio = current_std / init_std
        deviation_ratios.append({"name": name, "std": current_std, "ratio": ratio})

    if deviation_ratios:
        # Sort by ratio
        deviation_ratios.sort(key=lambda x: x["ratio"], reverse=True)

        print(f"   {'Layer':<45} {'Std':<10} {'Ratio':<10}")
        print("   " + "-" * 65)
        for d in deviation_ratios[:10]:  # Show top 10
            status = "⚠️" if d["ratio"] > 5 or d["ratio"] < 0.2 else ""
            print(f"   {d['name']:<45} {d['std']:<10.4f} {d['ratio']:<10.2f}x {status}")

        avg_ratio = np.mean([d["ratio"] for d in deviation_ratios])
        max_ratio = max(d["ratio"] for d in deviation_ratios)
        min_ratio = min(d["ratio"] for d in deviation_ratios)

        print(f"\n   Average deviation: {avg_ratio:.2f}x init")
        print(f"   Range: {min_ratio:.2f}x - {max_ratio:.2f}x")

        print("\n   Training Stability Recommendations:")
        if max_ratio > 10:
            print(f"   ⚠️  Some weights grew very large ({max_ratio:.1f}x init)")
            print("      → May indicate exploding gradients or high LR")
            print("      → Consider gradient clipping or lower LR")
        if min_ratio < 0.1:
            print(f"   ⚠️  Some weights shrank significantly ({min_ratio:.2f}x init)")
            print("      → May indicate vanishing gradients or over-regularization")
        if 0.5 <= avg_ratio <= 3.0:
            print("   ✅ Weight magnitudes in reasonable range")

    # 11. SUMMARY
    print("\n" + "=" * 65)
    print("[SUMMARY]")
    print("=" * 65)
    total_params = sum(v.numel() for v in proj_weights.values())
    print(f"   Total projector parameters: {total_params:,}")
    print(f"   Number of experts: {num_experts}")
    print(f"   Architecture: MOSA (Conv + Router + {num_experts}x SimpleAdapter)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe.py <path_to_model.safetensors>")
        sys.exit(1)
    analyze_checkpoint(sys.argv[1])
