#!/usr/bin/env python3
"""
MoE Checkpoint Forensic Analysis Tool

Usage:
    python scripts/analyze_moe.py path/to/checkpoint/model.safetensors

Diagnoses:
- Expert Aggregation (ModuleList vs Stacked)
- Router Bias (The #1 cause of collapse)
- Feature Collapse (SVD/Rank)
- Dead Experts (Gradient starvation)
- Initialization Safety (Magnitude checks)
"""

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np
import sys
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

def draw_ascii_bar(values: List[float], labels: Optional[List[str]] = None, max_width: int = 40):
    """Draws a simple ASCII bar chart for visualizations."""
    if len(values) == 0:
        return
    
    values = np.array(values)
    min_val, max_val = values.min(), values.max()
    # Avoid div by zero
    range_val = max_val - min_val if max_val != min_val else 1.0
    
    print("-" * 65)
    for i, val in enumerate(values):
        # Normalize to 0-1 for width
        if range_val > 0:
            normalized = (val - min_val) / range_val
        else:
            normalized = 0.5
            
        width = int(normalized * max_width)
        
        bar = "█" * width
        label = f"{labels[i]:<10}" if labels else f"Exp {i:<4}"
        print(f"{label} | {bar:<{max_width}} {val:.6f}")
    print("-" * 65)

def compute_effective_rank(matrix: torch.Tensor) -> float:
    """Computes Shannon Entropy of Singular Values (Information Capacity)."""
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)
    
    # Move to float32 and CPU for SVD
    matrix = matrix.float().cpu()
    
    try:
        # We only need singular values (S)
        S = torch.linalg.svdvals(matrix)
        # Normalize singular values to treat them as probabilities
        p = S / S.sum()
        # Compute entropy
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        # Effective rank = e^entropy
        effective_rank = torch.exp(entropy).item()
        return effective_rank
    except Exception as e:
        return 0.0

def resolve_model_path(path_or_repo: str) -> str:
    """Handles local paths or simple huggingface repo downloads."""
    if os.path.exists(path_or_repo):
        # Check if directory, look for safe tensors
        if os.path.isdir(path_or_repo):
            candidate = os.path.join(path_or_repo, "model.safetensors")
            if os.path.exists(candidate):
                return candidate
        return path_or_repo
        
    # If it looks like a repo ID
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
    print(f"\n🔬 MOE FORENSIC ANALYSIS")
    print(f"   Target: {file_path}")
    print("=" * 65)

    file_path = resolve_model_path(file_path)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        tensors = load_file(file_path)
    except Exception as e:
        print(f"❌ Failed to load safetensors: {e}")
        return

    # Filter for projector weights
    # We look for 'projector' keys first, fallback to generic if not found
    proj_weights = {k: v for k, v in tensors.items() if "projector" in k}
    
    # Debug: Print first few keys if projector not found
    if not proj_weights:
        print("⚠️  No 'projector' prefix found. Searching for generic expert keys...")
        proj_weights = {k: v for k, v in tensors.items() if "experts" in k or "router" in k}
    
    if not proj_weights:
        print("❌ No MoE parameters found in this checkpoint.")
        return

    print(f"📊 Loaded {len(proj_weights)} MoE-related parameters.")

    # =========================================================================
    # 1. ORGANIZE WEIGHTS (Parsing the ModuleList)
    # =========================================================================
    shared_w12 = None
    shared_w3 = None
    router_w = None
    # We might not have router bias if using simple Linear without bias
    router_b = None 
    
    # Dictionary to hold experts: {index: {'w12': tensor, 'w3': tensor}}
    experts_map = defaultdict(dict)

    for k, v in proj_weights.items():
        # Clean the key name for easier matching
        k_clean = k.lower()
        
        if "shared_expert.w12" in k_clean:
            shared_w12 = v
        elif "shared_expert.w3" in k_clean:
            shared_w3 = v
        elif "router_weights" in k_clean or ("router" in k_clean and "weight" in k_clean):
            router_w = v
        
        # Regex to find expert index in "experts.0.w12" or "experts.15.w3"
        # Matches: projector.experts.0.w12.weight
        match = re.search(r"experts\.(\d+)\.", k)
        if match:
            idx = int(match.group(1))
            if "w12" in k:
                experts_map[idx]['w12'] = v
            elif "w3" in k:
                experts_map[idx]['w3'] = v

    # Convert map to lists, ensuring sorted order by index
    sorted_indices = sorted(experts_map.keys())
    if not sorted_indices:
        print("❌ No routed experts found (parsing failed). check key names.")
        # Debug print
        print("First 5 keys:", list(proj_weights.keys())[:5])
        return

    num_experts = len(sorted_indices)
    routed_w12s = [experts_map[i]['w12'] for i in sorted_indices if 'w12' in experts_map[i]]
    routed_w3s = [experts_map[i]['w3'] for i in sorted_indices if 'w3' in experts_map[i]]

    print(f"   Found {num_experts} Routed Experts + 1 Shared Expert.")

    # =========================================================================
    # 2. ROUTER HEALTH (The "Bias/Scale" Check)
    # =========================================================================
    print("\n[1] ROUTER DIAGNOSTICS")
    if router_w is not None:
        r_std = router_w.std().item()
        r_mean = router_w.mean().item()
        r_norm = torch.linalg.norm(router_w).item()
        
        print(f"   Router Weight Std:  {r_std:.5f}")
        print(f"   Router Weight Mean: {r_mean:.5f}")
        print(f"   Router Weight Norm: {r_norm:.5f}")
        
        # In a cosine router, the weights should be somewhat normalized or small.
        # If they are exploding, gradients are likely huge.
        if r_std > 1.0:
            print("   ⚠️  Router weights have high variance. Check Router Scale.")
        elif r_std < 1e-4:
            print("   ⚠️  Router weights are vanishingly small. Learning might be stalled.")
        else:
            print("   ✅ Router weights look healthy.")
            
    else:
        print("   ❌ Router weights missing!")

    # =========================================================================
    # 3. EXPERT DIVERSITY (Cosine Sim of Input Weights)
    # =========================================================================
    print("\n[2] EXPERT DIVERSITY (Input Weight W12)")
    # Are the experts becoming identical?
    if len(routed_w12s) > 1:
        # Stack all w12s: [Num_Experts, In_Dim * Hidden_Dim]
        # Flatten each expert's matrix to a single vector
        flat_experts = torch.stack([w.view(-1) for w in routed_w12s]).float()
        
        # Normalize vectors
        norm_experts = F.normalize(flat_experts, p=2, dim=1)
        
        # Similarity Matrix = XX^T
        sim_matrix = torch.mm(norm_experts, norm_experts.t())
        
        # Mask diagonal (self-similarity is always 1.0)
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]
        
        avg_sim = off_diag.mean().item()
        max_sim = off_diag.max().item()
        min_sim = off_diag.min().item()
        
        print(f"   Avg Similarity: {avg_sim:.4f} (Lower is better)")
        print(f"   Max Similarity: {max_sim:.4f}")
        
        if avg_sim > 0.98:
             print("   ❌ RED ZONE: Experts are identical. Initialization failed or noise too low.")
        elif avg_sim > 0.80:
             print("   ⚠️  YELLOW ZONE: Experts are highly correlated. Might be collapsing.")
        else:
             print("   ✅ GREEN ZONE: Experts are diverging and specializing.")
    else:
        print("   ⚠️  Not enough experts to compare.")

    # =========================================================================
    # 4. FADE-IN STATUS (Output Weights Magnitude)
    # =========================================================================
    print("\n[3] INITIALIZATION / FADE-IN STATUS")
    # This checks the fix: Routed experts should be comparable to Shared expert
    # OR small if using Zero-Init (which we moved away from, but good to check).
    
    if shared_w3 is not None and len(routed_w3s) > 0:
        shared_mag = shared_w3.abs().mean().item()
        
        # Check average magnitude of expert outputs
        expert_mags = [w.abs().mean().item() for w in routed_w3s]
        avg_routed_mag = np.mean(expert_mags)
        
        print(f"   Shared W3 Mean Mag: {shared_mag:.6f}")
        print(f"   Routed W3 Mean Mag: {avg_routed_mag:.6f}")
        
        # Ratio check
        ratio = avg_routed_mag / (shared_mag + 1e-9)
        
        # Visualization
        print("\n   W3 Magnitude per Expert:")
        draw_ascii_bar(expert_mags, labels=[str(i) for i in sorted_indices])
        
        if shared_mag < 1e-5:
            print("   ⚠️  Shared Expert is near-zero. This is unusual (but allowed).")
        
        if avg_routed_mag < 1e-5:
            print("   ℹ️  Routed Experts are initialized near-zero (Fade-in Ready).")
        elif ratio > 0.8 and ratio < 1.2:
            print("   ✅ Standard Initialization detected (Experts balanced with Shared).")
        else:
            print(f"   ℹ️  Experts are {ratio:.2f}x scale of Shared.")

    else:
        print("   ⚠️  Cannot compare shared vs routed outputs.")

    # =========================================================================
    # 5. SPECTRAL HEALTH (Rank Analysis)
    # =========================================================================
    print("\n[4] SPECTRAL HEALTH (Effective Rank)")
    # A random matrix has high rank. A collapsed matrix has low rank.
    
    if shared_w12 is not None:
        rank = compute_effective_rank(shared_w12)
        full_rank = min(shared_w12.shape)
        pct = (rank / full_rank) * 100
        print(f"   Shared Expert Rank: {rank:.1f} / {full_rank} ({pct:.1f}%)")
        if pct < 10.0:
            print("   ❌ Shared Expert is undergoing Rank Collapse.")
    
    if len(routed_w12s) > 0:
        ranks = [compute_effective_rank(w) for w in routed_w12s]
        avg_rank = np.mean(ranks)
        full_rank = min(routed_w12s[0].shape)
        pct = (avg_rank / full_rank) * 100
        
        print(f"   Avg Routed Expert Rank: {avg_rank:.1f} / {full_rank} ({pct:.1f}%)")
        
        print("\n   Rank per Expert:")
        draw_ascii_bar(ranks, labels=[str(i) for i in sorted_indices])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe.py <path_to_model.safetensors or dir>")
        sys.exit(1)
    analyze_checkpoint(sys.argv[1])