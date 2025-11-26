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

    # 6. SUMMARY
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
