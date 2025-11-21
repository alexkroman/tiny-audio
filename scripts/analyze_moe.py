#!/usr/bin/env python3
"""
MoE Checkpoint Forensic Analysis Tool

Analyzes a trained MoE projector checkpoint to diagnose training issues:
- Weight movement (did it learn?)
- Router health (is gating working?)
- Expert diversity (are experts specialized?)
- Shared vs Routed balance
- Spectral analysis (information capacity)

Usage:
    python scripts/analyze_moe.py path/to/model.safetensors
    python scripts/analyze_moe.py outputs/checkpoint-1000/model.safetensors
"""

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np
import sys
import os


def compute_effective_rank(matrix):
    """
    Computes the 'Effective Rank' using Shannon Entropy of Singular Values.
    Returns: (effective_rank, stable_rank)
    """
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)

    # SVD is expensive, convert to float32
    try:
        # Standard SVD
        U, S, V = torch.svd(matrix.float())

        # 1. Stable Rank (Sum / Max) - Robust to outliers
        stable_rank = (S.sum() ** 2) / (S ** 2).sum()

        # 2. Effective Rank (Entropy) - Information capacity
        # Normalize singular values to probability distribution
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        effective_rank = torch.exp(entropy)

        return effective_rank.item(), stable_rank.item()
    except:
        return 0.0, 0.0


def analyze_checkpoint(file_path):
    print(f"\n🔬 STARTING FORENSIC ANALYSIS: {file_path}")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    # 1. Load Tensors
    try:
        tensors = load_file(file_path)
        print(f"✅ Loaded {len(tensors)} tensors successfully.")
    except Exception as e:
        print(f"❌ Failed to load safetensors: {e}")
        return

    # 2. Filter for Projector
    # Handle standard keys or PEFT 'base_model.model.' prefixes
    projector_weights = {
        k.replace("base_model.model.", "").replace("model.", ""): v
        for k, v in tensors.items()
        if "projector" in k
    }

    if len(projector_weights) == 0:
        print("❌ CRITICAL: No projector weights found! Is this a full checkpoint?")
        return

    print(f"📊 Found {len(projector_weights)} projector parameters.")

    # --- CHECK 1: WEIGHT MOVEMENT (Did it learn?) ---
    print("\n[1] TRAINING HEARTBEAT (Weight Movement)")
    # We know init std was 0.02. Did it broaden?

    shared_w12 = next(
        (v for k, v in projector_weights.items() if "shared_expert.w12" in k), None
    )
    if shared_w12 is not None:
        std = shared_w12.std().item()
        mean = shared_w12.mean().item()
        kurtosis = torch.mean((shared_w12 - mean) ** 4) / (std**4)

        print(f"   Shared Expert Std: {std:.5f} (Target > 0.02)")
        print(f"   Shared Expert Kurtosis: {kurtosis:.2f} (Higher = Specialized)")

        if std <= 0.02001:
            print(
                "   ⚠️  WARNING: Weights are remarkably close to init. Learning Rate might be too low."
            )
        else:
            print("   ✅  PASS: Weights have drifted from initialization.")
    else:
        print("   ⚠️  Shared expert not found in keys.")

    # --- CHECK 2: ROUTER HEALTH ---
    print("\n[2] ROUTER HEALTH (Gating Network)")
    router_weight = next(
        (v for k, v in projector_weights.items() if "router.weight" in k), None
    )

    if router_weight is not None:
        # Router weights should NOT be uniform
        r_std = router_weight.std().item()
        print(f"   Router Weight Std: {r_std:.5f}")

        if r_std < 0.001:
            print(
                "   ❌ CRITICAL: Router weights are dead/uniform. Experts are not being selected."
            )
        elif r_std > 0.5:
            print(
                "   ⚠️  WARNING: Router weights are very large. Risk of 'Hard Routing' (gradient saturation)."
            )
        else:
            print("   ✅  PASS: Router has healthy variance.")
    else:
        print("   ❌ Router weights not found!")

    # --- CHECK 3: EXPERT COLLAPSE (Cosine Similarity) ---
    print("\n[3] EXPERT DIVERSITY CHECK (Are experts different?)")

    # Collect all routed expert w12 weights
    routed_w12s = []
    expert_indices = []
    for k, v in projector_weights.items():
        if "routed_experts" in k and "w12.weight" in k:
            # Extract index from string "routed_experts.5.w12..."
            parts = k.split(".")
            try:
                idx = int(parts[parts.index("routed_experts") + 1])
                expert_indices.append(idx)
                routed_w12s.append(v.float())  # float32 for sim calc
            except:
                continue

    if len(routed_w12s) > 1:
        # Sort by index
        zipped = sorted(zip(expert_indices, routed_w12s))
        routed_w12s = [z[1] for z in zipped]

        # Stack: [Num_Experts, Features]
        stack = torch.stack(routed_w12s).view(len(routed_w12s), -1)

        # Normalize
        stack = F.normalize(stack, p=2, dim=1)

        # Cosine Sim Matrix
        sim_matrix = torch.mm(stack, stack.t())

        # Remove diagonal (1.0)
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
        off_diag = sim_matrix[mask]

        avg_sim = off_diag.mean().item()
        max_sim = off_diag.max().item()
        min_sim = off_diag.min().item()

        print(f"   Analyzed {len(routed_w12s)} Experts.")
        print(f"   Average Similarity: {avg_sim:.4f} (Lower is better)")
        print(f"   Max Similarity:     {max_sim:.4f}")

        if avg_sim > 0.9:
            print("   ❌ CRITICAL: EXPERT COLLAPSE. All experts are identical.")
        elif avg_sim > 0.5:
            print(
                "   ⚠️  WARNING: Experts are highly correlated. Aux Loss might be too low."
            )
        else:
            print("   ✅  PASS: Experts are specializing (divergent weights).")

    else:
        print("   ⚠️  Not enough experts found to compare.")

    # --- CHECK 4: SHARED vs ROUTED BALANCE ---
    print("\n[4] SHARED vs ROUTED BALANCE")
    # Check magnitude of weights to see if one dominates
    if len(routed_w12s) > 0 and shared_w12 is not None:
        shared_mag = shared_w12.abs().mean().item()
        routed_mag = torch.stack(routed_w12s).abs().mean().item()

        ratio = routed_mag / shared_mag
        print(f"   Shared Magnitude: {shared_mag:.5f}")
        print(f"   Routed Magnitude: {routed_mag:.5f}")
        print(f"   Ratio (Routed/Shared): {ratio:.2f}")

        if ratio < 0.5:
            print(
                "   ⚠️  WARNING: Shared expert is dominating. Routed experts might be starving."
            )
            print("       -> Consider increasing 'routed_scaling_factor'.")
        elif ratio > 2.0:
            print(
                "   ⚠️  WARNING: Routed experts are dominating. Shared expert is passive."
            )
        else:
            print("   ✅  PASS: Balanced contribution.")

    # --- CHECK 5: SPECTRAL ANALYSIS (Effective Rank) ---
    print("\n[5] SPECTRUM ANALYSIS (Information Capacity)")
    # This checks if the matrices are actually using their full dimensions
    # or if they have collapsed into a low-rank subspace (lazy learning).

    if shared_w12 is not None:
        e_rank, s_rank = compute_effective_rank(shared_w12)
        full_rank = min(shared_w12.shape)
        utilization = (e_rank / full_rank) * 100

        print(
            f"   Shared Expert Effective Rank: {e_rank:.1f} / {full_rank} ({utilization:.1f}%)"
        )

        if utilization < 5.0:
            print(
                "   ❌ CRITICAL: Dimensional Collapse. The model is ignoring 95% of feature space."
            )
        elif utilization < 20.0:
            print(
                "   ⚠️  WARNING: Low Rank. The model might be underfitting complex features."
            )
        else:
            print("   ✅  PASS: High Effective Rank. Dense information encoding.")

    if len(routed_w12s) > 0:
        # Check random routed expert
        e_rank_r, _ = compute_effective_rank(routed_w12s[0])
        full_rank_r = min(routed_w12s[0].shape)
        utilization_r = (e_rank_r / full_rank_r) * 100
        print(
            f"   Routed Expert 0 Effective Rank: {e_rank_r:.1f} / {full_rank_r} ({utilization_r:.1f}%)"
        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe.py <path_to_model.safetensors>")
        print("\nExample:")
        print("  python scripts/analyze_moe.py outputs/checkpoint-1000/model.safetensors")
        sys.exit(1)
    else:
        analyze_checkpoint(sys.argv[1])
