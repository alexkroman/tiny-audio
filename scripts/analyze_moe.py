#!/usr/bin/env python3
"""
MoE Checkpoint Forensic Analysis Tool

Usage:
    python scripts/analyze_moe.py path/to/checkpoint/model.safetensors

Diagnoses:
1. Integrity: Checks for NaNs or Infinity values (common in BF16 training).
2. Router Health: Checks if the router is biased or exploding.
3. Load Balance: Checks the uniformity of the expert_load buffer (CRITICAL NEW CHECK).
4. Diversity: Checks if experts are identical clones (bad) or specialized (good).
5. Spectral Health: Uses SVD to check for rank collapse (feature richness).
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as f
from safetensors.torch import load_file

# --- Helper Functions ---


def draw_ascii_bar(
    values: List[float],
    labels: Optional[List[str]] = None,
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
        # Normalize to 0-1 for width
        normalized = (val - min_val) / range_val if range_val > 0 else 0.5

        width = int(normalized * max_width)
        bar = "█" * width

        label = f"{labels[i]:<4}" if labels else f"{i:<4}"
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

    # Move to float32 and CPU for SVD stability
    matrix = matrix.float().cpu()

    try:
        # We only need singular values (s)
        s = torch.linalg.svdvals(matrix)
        # Normalize to treat as probabilities
        p = s / s.sum()
        # Compute entropy
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        # Effective rank = e^entropy
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


# --- Main Analysis Logic ---


def analyze_checkpoint(file_path: str):
    print("\n🔬 MOE FORENSIC ANALYSIS")
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
        print("⚠️  No 'projector' prefix found. Searching for generic expert keys...")
        proj_weights = {k: v for k, v in tensors.items() if "experts" in k or "router" in k}

    if not proj_weights:
        print("❌ No MoE parameters found in this checkpoint.")
        return

    print(f"📊 Loaded {len(proj_weights)} MoE-related parameters.")

    # 1. ORGANIZE WEIGHTS
    shared_w12 = None
    shared_w3 = None
    router_w = None
    expert_load_buffer = None  # NEW: For load balance tracking
    experts_map = defaultdict(dict)

    # Scan for NaNs while loading
    healthy = True
    for k, v in proj_weights.items():
        if not check_integrity(k, v):
            healthy = False

        k_clean = k.lower()

        if "shared_expert.w12" in k_clean:
            shared_w12 = v
        elif "shared_expert.w3" in k_clean:
            shared_w3 = v
        elif "router_weights" in k_clean:
            router_w = v
        # NEW: Load expert_load buffer
        elif "expert_load" in k_clean:
            expert_load_buffer = v

        # Regex to find expert index (e.g., experts.0.w12)
        match = re.search(r"experts\.(\d+)\.", k)
        if match:
            idx = int(match.group(1))
            if "w12" in k:
                experts_map[idx]["w12"] = v
            elif "w3" in k:
                experts_map[idx]["w3"] = v

    if not healthy:
        print("\n❌ ABORTING ANALYSIS: Model weights are corrupted (NaN/Inf).")
        return

    sorted_indices = sorted(experts_map.keys())
    if not sorted_indices:
        print("❌ No routed experts found. Check variable naming convention.")
        return

    num_experts = len(sorted_indices)
    routed_w12s = [experts_map[i]["w12"] for i in sorted_indices if "w12" in experts_map[i]]
    routed_w3s = [experts_map[i]["w3"] for i in sorted_indices if "w3" in experts_map[i]]

    print(f"   Found {num_experts} Routed Experts + 1 Shared Expert.")

    # 2. ROUTER DIAGNOSTICS (Original [1])
    print("\n[1] ROUTER DIAGNOSTICS")
    if router_w is not None:
        r_std = router_w.std().item()
        r_norm = torch.linalg.norm(router_w).item()

        print(f"   Router Weight Std:  {r_std:.5f}")
        print(f"   Router Weight Mean: {router_w.mean().item():.5f}")
        print(f"   Router Weight Norm: {r_norm:.5f}")

        if r_std > 1.0:
            print("   ⚠️  High Variance: Router might be over-confident (Gradient Explosion).")
        elif r_std < 1e-4:
            print("   ⚠️  Low Variance: Router gradients might be vanishing.")
        else:
            print("   ✅ Router weights look healthy.")
    else:
        print("   ❌ Router weights missing!")

    # 3. LOAD BALANCE DIAGNOSTICS (NEW CRITICAL SECTION)
    print("\n[2] LOAD BALANCE DIAGNOSTICS (Router Effectiveness)")
    if expert_load_buffer is not None:
        load_values = expert_load_buffer.float().cpu().numpy()
        
        load_std = np.std(load_values)
        load_max = np.max(load_values)
        load_min = np.min(load_values)
        
        print(f"   Buffer Std Dev: {load_std:.6f} (Lower is better)")
        
        if load_min > 1e-9:
            ratio = load_max / load_min
            print(f"   Max/Min Ratio: {ratio:.2f}x")
        else:
            print("   ❌ CRITICAL FAILURE: At least one expert load is near zero.")
            ratio = float('inf')
            
        print("\n   Current Load per Expert (Visual Check):")
        draw_ascii_bar(load_values, labels=[str(i) for i in range(num_experts)], max_width=40)
        
        if ratio > 5.0 or load_min < 1e-9:
            print("   ❌ RED ZONE: Load Imbalance is significant. Router is failing to use all experts.")
        elif ratio > 2.0 or load_std > 0.05:
            print("   ⚠️  YELLOW ZONE: Imbalance is present. Consider increasing router_bias_scale.")
        else:
            print("   ✅ Load balance appears stable and uniform.")
    else:
        print("   ❌ Expert Load Buffer missing!")


    # 4. EXPERT DIVERSITY (Original [2])
    print("\n[3] EXPERT DIVERSITY (Input Weight W12)")
    if len(routed_w12s) > 1:
        # Stack and flatten: [Num_Experts, Total_Params]
        flat_experts = torch.stack([w.view(-1) for w in routed_w12s]).float()
        norm_experts = f.normalize(flat_experts, p=2, dim=1)

        # Cosine Similarity Matrix
        sim_matrix = torch.mm(norm_experts, norm_experts.t())

        # Mask diagonal
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]

        avg_sim = off_diag.mean().item()

        print(f"   Avg Similarity: {avg_sim:.4f} (Lower is better)")
        print(f"   Max Similarity: {off_diag.max().item():.4f}")

        if avg_sim > 0.98:
            print("   ❌ RED ZONE: Experts are identical clones.")
        elif avg_sim > 0.80:
            print("   ⚠️  YELLOW ZONE: Experts are highly correlated.")
        else:
            print("   ✅ GREEN ZONE: Experts are specializing.")

    # 5. INITIALIZATION / FADE-IN (Original [3])
    print("\n[4] INITIALIZATION / FADE-IN STATUS")
    if shared_w3 is not None and len(routed_w3s) > 0:
        shared_mag = shared_w3.abs().mean().item()
        expert_mags = [w.abs().mean().item() for w in routed_w3s]
        avg_routed_mag = np.mean(expert_mags)

        print(f"   Shared W3 Mean Mag: {shared_mag:.6f}")
        print(f"   Routed W3 Mean Mag: {avg_routed_mag:.6f}")

        print("\n   W3 Magnitude per Expert:")
        draw_ascii_bar(expert_mags, labels=[str(i) for i in sorted_indices])

        # Heuristic check
        ratio = avg_routed_mag / (shared_mag + 1e-9)
        if ratio > 0.8 and ratio < 1.2:
            print("   ✅ Standard Initialization detected (Balanced).")
        else:
            print(f"   ℹ️  Ratio: {ratio:.2f}x (Check if this matches your init strategy).")

    # 6. SPECTRAL HEALTH (Original [4])
    print("\n[5] SPECTRAL HEALTH (Effective Rank)")
    if shared_w12 is not None:
        rank = compute_effective_rank(shared_w12)
        full_rank = min(shared_w12.shape)
        pct = (rank / full_rank) * 100
        print(f"   Shared Expert Rank: {rank:.1f} / {full_rank} ({pct:.1f}%)")

    if len(routed_w12s) > 0:
        ranks = [compute_effective_rank(w) for w in routed_w12s]
        avg_rank = np.mean(ranks)
        full_rank = min(routed_w12s[0].shape)

        print(
            f"   Avg Routed Expert Rank: {avg_rank:.1f} / {full_rank} ({(avg_rank / full_rank) * 100:.1f}%)"
        )
        print("\n   Rank per Expert:")
        draw_ascii_bar(ranks, labels=[str(i) for i in sorted_indices], denominator=full_rank)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe.py <path_to_model.safetensors>")
        sys.exit(1)
    analyze_checkpoint(sys.argv[1])