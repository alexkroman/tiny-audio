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
    python scripts/analyze_moe.py mazesmazes/tiny-audio  # Download from HF Hub
"""

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np
import sys
import os
from pathlib import Path


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


def resolve_model_path(path_or_repo):
    """
    Resolves a path to a model file, downloading from HF Hub if needed.

    Args:
        path_or_repo: Either a local file path or HF repo ID (e.g., 'mazesmazes/tiny-audio')

    Returns:
        Path to the model.safetensors file
    """
    # Check if it's a local file that exists
    if os.path.exists(path_or_repo):
        return path_or_repo

    # Check if it looks like a HF repo ID (contains / and doesn't end with .safetensors)
    if "/" in path_or_repo and not path_or_repo.endswith(".safetensors"):
        try:
            from huggingface_hub import hf_hub_download
            print(f"📥 Downloading model from Hugging Face Hub: {path_or_repo}")

            # Try to download model.safetensors
            model_file = hf_hub_download(
                repo_id=path_or_repo,
                filename="model.safetensors",
                repo_type="model"
            )
            print(f"✅ Downloaded to: {model_file}")
            return model_file
        except Exception as e:
            print(f"❌ Failed to download from Hub: {e}")
            print(f"   Trying to load as local path...")
            return path_or_repo

    # Return as-is (will fail later if doesn't exist)
    return path_or_repo


def analyze_checkpoint(file_path):
    print(f"\n🔬 STARTING FORENSIC ANALYSIS: {file_path}")
    print("=" * 60)

    # Resolve the path (download from Hub if needed)
    file_path = resolve_model_path(file_path)

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

    # --- CHECK 1: TRAINING HEARTBEAT (Weight Movement) ---
    print("\n[1] TRAINING HEARTBEAT (Weight Movement)")

    shared_w12 = next(
        (v for k, v in projector_weights.items() if "shared_expert.w12" in k), None
    )
    if shared_w12 is not None:
        std = shared_w12.std().item()
        mean = shared_w12.mean().item()
        kurtosis = torch.mean((shared_w12 - mean) ** 4) / (std**4)

        print(f"   Shared Expert Std: {std:.5f}")
        print(f"   Shared Expert Kurtosis: {kurtosis:.2f}")

        if std < 0.005:
            print("   ❌ RED ZONE: Signal vanishing. Learning rate too low or frozen.")
        elif 0.012 <= std <= 0.018:
            print("   ✅ GREEN ZONE: Healthy weight variance.")
        else:
            print("   ⚠️  Within acceptable range.")

        if kurtosis < 2.0:
            print("   ❌ RED ZONE: Weights too uniform. No specialization.")
        elif kurtosis > 20.0:
            print("   ❌ RED ZONE: Extreme sparsity. Most weights dead.")
        elif 4.0 <= kurtosis <= 8.0:
            print("   ✅ GREEN ZONE: Healthy sparsification.")
        elif 2.0 <= kurtosis < 4.0:
            print("   ⚠️  Early training. Expect kurtosis to rise.")
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

    experts_w12 = next(
        (v for k, v in projector_weights.items() if "experts_w12" in k), None
    )

    routed_w12s = []
    if experts_w12 is not None:
        num_experts = experts_w12.shape[0]
        for i in range(num_experts):
            routed_w12s.append(experts_w12[i].float())

    if len(routed_w12s) > 1:
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
        print(f"   Average Similarity: {avg_sim:.4f}")
        print(f"   Max Similarity:     {max_sim:.4f}")

        if avg_sim > 0.50:
            print("   ❌ RED ZONE: Expert collapse. Router failing.")
        elif avg_sim <= 0.20:
            print("   ✅ GREEN ZONE: Experts are diverse and specialized.")
        else:
            print("   ⚠️  Experts partially specialized. Monitor closely.")

        # Calculate Kurtosis for each expert
        expert_kurtosis = []
        for w in routed_w12s:
            std = w.std().item()
            mean = w.mean().item()
            k = torch.mean((w - mean) ** 4) / (std**4)
            expert_kurtosis.append(k.item())

        if len(expert_kurtosis) > 0:
            avg_k = sum(expert_kurtosis) / len(expert_kurtosis)
            min_k = min(expert_kurtosis)
            max_k = max(expert_kurtosis)

            print(f"   Expert Kurtosis Range: {min_k:.2f} - {max_k:.2f} (Avg: {avg_k:.2f})")

            if max_k > 50.0:
                print("   ⚠️  WARNING: Some experts are extremely sparse (dying?).")
            elif max_k - min_k < 1.0:
                print("   ⚠️  WARNING: All experts look structurally identical.")
            else:
                print("   ✅  PASS: Experts show structural diversity.")

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
        print(f"   Balance Ratio (Routed/Shared): {ratio:.2f}")

        if ratio < 0.2 or ratio > 5.0:
            print("   ❌ RED ZONE: Severe imbalance. One pathway dominates.")
        elif 0.8 <= ratio <= 1.5:
            print("   ✅ GREEN ZONE: Balanced partnership.")
        else:
            print("   ⚠️  Moderate imbalance. Monitor for divergence.")

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

        if utilization < 10.0:
            print("   ❌ RED ZONE: Dimensional collapse. Model ignoring most features.")
        elif 40.0 <= utilization <= 60.0:
            print("   ✅ GREEN ZONE: Healthy compression. Efficient feature usage.")
        elif utilization > 80.0:
            print("   ⚠️  Early training. Expect rank to drop as model optimizes.")
        else:
            print("   ⚠️  Within acceptable range.")

    if len(routed_w12s) > 0:
        for i, expert_w12 in enumerate(routed_w12s):
            e_rank_r, _ = compute_effective_rank(expert_w12)
            full_rank_r = min(expert_w12.shape)
            utilization_r = (e_rank_r / full_rank_r) * 100
            print(
                f"   Routed Expert {i} Effective Rank: {e_rank_r:.1f} / {full_rank_r} ({utilization_r:.1f}%)"
            )

            if utilization_r < 10.0:
                print(f"      ❌ RED ZONE: Expert {i} collapsed.")
            elif 40.0 <= utilization_r <= 60.0:
                print(f"      ✅ GREEN ZONE: Expert {i} healthy compression.")
            elif utilization_r > 80.0:
                print(f"      ⚠️  Early training phase for expert {i}.")
            else:
                print(f"      ⚠️  Acceptable range for expert {i}.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe.py <path_or_repo_id>")
        print("\nExamples:")
        print("  # Local checkpoint")
        print("  python scripts/analyze_moe.py outputs/checkpoint-1000/model.safetensors")
        print()
        print("  # Download from Hugging Face Hub")
        print("  python scripts/analyze_moe.py mazesmazes/tiny-audio")
        sys.exit(1)
    else:
        analyze_checkpoint(sys.argv[1])
