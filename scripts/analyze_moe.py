#!/usr/bin/env python3
"""
MoE Checkpoint Forensic Analysis Tool (Enhanced)

Diagnoses:
- Expert Aggregation (ModuleList vs Stacked)
- Router Bias (The #1 cause of collapse)
- Feature Collapse (SVD/Rank)
- Dead Experts (Gradient starvation)
"""

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np
import sys
import os
import re
from collections import defaultdict

def draw_ascii_bar(values, labels=None, max_width=40):
    """Draws a simple ASCII bar chart for visualizations."""
    if len(values) == 0:
        return
    
    values = np.array(values)
    min_val, max_val = values.min(), values.max()
    range_val = max_val - min_val if max_val != min_val else 1.0
    
    print("-" * 60)
    for i, val in enumerate(values):
        # Normalize to 0-1 for width
        normalized = (val - min_val) / range_val
        width = int(normalized * max_width)
        
        bar = "█" * width
        label = f"{labels[i]:<10}" if labels else f"Exp {i:<4}"
        print(f"{label} | {bar:<{max_width}} {val:.4f}")
    print("-" * 60)

def compute_effective_rank(matrix):
    """Computes Shannon Entropy of Singular Values (Information Capacity)."""
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)
    
    # Move to float32 and CPU for SVD
    matrix = matrix.float().cpu()
    
    try:
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
        # Normalize singular values
        p = S / S.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        effective_rank = torch.exp(entropy).item()
        return effective_rank
    except:
        return 0.0

def resolve_model_path(path_or_repo):
    if os.path.exists(path_or_repo):
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

def analyze_checkpoint(file_path):
    print(f"\n🔬 MOE FORENSIC ANALYSIS: {file_path}")
    print("=" * 60)

    file_path = resolve_model_path(file_path)
    if not os.path.exists(file_path):
        print(f"❌ File not found.")
        return

    try:
        tensors = load_file(file_path)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return

    # Filter for projector weights
    proj_weights = {k: v for k, v in tensors.items() if "projector" in k}
    if not proj_weights:
        print("❌ No projector keys found. Is this a full model checkpoint?")
        # Try looking for just 'model.' keys if standard naming
        proj_weights = {k: v for k, v in tensors.items() if "experts" in k or "router" in k}
    
    print(f"📊 Loaded {len(proj_weights)} projector parameters.")

    # =========================================================================
    # 1. ORGANIZE WEIGHTS (Handle ModuleList logic)
    # =========================================================================
    shared_w12 = None
    shared_w3 = None
    router_w = None
    router_b = None
    
    # Dictionary to hold experts: {index: {'w12': tensor, 'w3': tensor}}
    experts_map = defaultdict(dict)

    for k, v in proj_weights.items():
        if "shared_expert.w12" in k:
            shared_w12 = v
        elif "shared_expert.w3" in k:
            shared_w3 = v
        elif "router.weight" in k:
            router_w = v
        elif "router.bias" in k:
            router_b = v
        elif "experts" in k:
            # Regex to find expert index in "experts.0.w12" or "experts.15.w3"
            match = re.search(r"experts\.(\d+)\.", k)
            if match:
                idx = int(match.group(1))
                if "w12" in k:
                    experts_map[idx]['w12'] = v
                elif "w3" in k:
                    experts_map[idx]['w3'] = v

    # Convert map to lists
    num_experts = len(experts_map)
    sorted_indices = sorted(experts_map.keys())
    
    routed_w12s = [experts_map[i]['w12'] for i in sorted_indices if 'w12' in experts_map[i]]
    routed_w3s = [experts_map[i]['w3'] for i in sorted_indices if 'w3' in experts_map[i]]

    print(f"   Found {num_experts} Routed Experts.")

    # =========================================================================
    # 2. ROUTER HEALTH (The "Bias" Check)
    # =========================================================================
    print("\n[1] ROUTER DIAGNOSTICS")
    if router_w is not None:
        r_std = router_w.std().item()
        print(f"   Router Weight Std: {r_std:.5f}")
        
        if router_b is not None:
            print("   Router Bias Distribution (High variance = Collapse):")
            bias_vals = router_b.float().cpu().numpy()
            draw_ascii_bar(bias_vals, labels=[f"Exp {i}" for i in range(len(bias_vals))])
            
            bias_spread = bias_vals.max() - bias_vals.min()
            if bias_spread > 2.0:
                 print(f"   ❌ CRITICAL: Bias spread is {bias_spread:.2f}. Router has likely collapsed to one expert.")
            else:
                 print(f"   ✅ PASS: Bias spread is {bias_spread:.2f}. Load balancing likely working.")
        else:
            print("   ⚠️  No Router Bias found (Did you set bias=False?). Harder to balance early training.")
    else:
        print("   ❌ Router weights missing!")

    # =========================================================================
    # 3. EXPERT DIVERSITY (Cosine Sim)
    # =========================================================================
    print("\n[2] EXPERT DIVERSITY (Cosine Similarity)")
    if len(routed_w12s) > 1:
        # Stack all w12s: [Num_Experts, Hidden_Dim]
        # We flatten inputs to (N, -1)
        flat_experts = torch.stack([w.view(-1) for w in routed_w12s]).float()
        
        # Normalize
        norm_experts = F.normalize(flat_experts, p=2, dim=1)
        
        # Similarity Matrix
        sim_matrix = torch.mm(norm_experts, norm_experts.t())
        
        # Mask diagonal
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]
        
        avg_sim = off_diag.mean().item()
        max_sim = off_diag.max().item()
        
        print(f"   Avg Similarity: {avg_sim:.4f}")
        print(f"   Max Similarity: {max_sim:.4f}")
        
        if avg_sim > 0.90:
             print("   ❌ RED ZONE: Experts are identical. Initialization failed or noise too low.")
        elif avg_sim > 0.50:
             print("   ⚠️  YELLOW ZONE: Experts are still very similar (Sparse Upcycling Start).")
        elif avg_sim < 0.20:
             print("   ✅ GREEN ZONE: High specialization.")
    else:
        print("   ⚠️  Not enough experts to compare.")

    # =========================================================================
    # 4. SHARED VS ROUTED (Fade-In Check)
    # =========================================================================
    print("\n[3] FADE-IN STATUS (Output Weights)")
    if shared_w3 is not None and len(routed_w3s) > 0:
        shared_mag = shared_w3.abs().mean().item()
        
        # Check average magnitude of expert outputs
        expert_mags = [w.abs().mean().item() for w in routed_w3s]
        avg_routed_mag = np.mean(expert_mags)
        
        print(f"   Shared Output Mag: {shared_mag:.6f}")
        print(f"   Routed Output Mag: {avg_routed_mag:.6f}")
        
        ratio = avg_routed_mag / (shared_mag + 1e-9)
        print(f"   Ratio (Routed / Shared): {ratio:.4f}")
        
        if ratio < 0.01:
            print("   ✅ FADE-IN ACTIVE: Routed path is quiet. Training stability is high.")
        elif ratio > 0.5:
            print("   ⚠️  FADE-IN COMPLETE: Experts are contributing significantly.")
        
        print("   Expert Output Magnitudes:")
        draw_ascii_bar(expert_mags)
    else:
        print("   ⚠️  Cannot compare shared vs routed outputs.")

    # =========================================================================
    # 5. SPECTRAL HEALTH
    # =========================================================================
    print("\n[4] SPECTRAL HEALTH (Effective Rank)")
    if shared_w12 is not None:
        rank = compute_effective_rank(shared_w12)
        full_rank = min(shared_w12.shape)
        print(f"   Shared Expert Rank: {rank:.1f} / {full_rank} ({rank/full_rank*100:.1f}%)")
    
    if len(routed_w12s) > 0:
        ranks = [compute_effective_rank(w) for w in routed_w12s[:4]] # Check first 4
        print(f"   First 4 Experts Ranks: {[f'{r:.1f}' for r in ranks]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe.py <path_to_model.safetensors>")
        sys.exit(1)
    analyze_checkpoint(sys.argv[1])