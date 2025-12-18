#!/usr/bin/env python3
"""
SharedMoE Checkpoint Forensic Analysis Tool

Usage:
    python scripts/analyze_moe_shared.py path/to/checkpoint/model.safetensors

Analyzes SharedMoEAudioProjector architecture:
1. Shared vs Routed Expert Balance: Compares contribution levels
2. Router Diagnostics: Routing patterns and load balancing
3. SwiGLU Expert Analysis: Gate/Up/Down projection health
4. Expert Diversity: Checks if routed experts are specialized
5. Grow-in Analysis: Checks if routed experts are "growing in" from zero-init
6. Spectral Health: Uses SVD to check for rank collapse (effective rank)
7. Weight Magnitude Analysis: L2 norms, mean absolute values
8. Condition Number Analysis: Numerical stability checks
9. Initialization Deviation: Compares to expected init values
10. Dead Neuron Detection: Finds near-zero rows/columns
11. Gate-Up Alignment: SwiGLU redundancy check
12. Weight Distribution Health: Kurtosis and skewness
13. Router Decision Sharpness: Expert distinguishability
14. Singular Value Distribution: Energy concentration analysis
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional


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

        label = f"{labels[i]:<8}" if labels else f"{i:<8}"
        if denominator is not None:
            pct = (val / denominator) * 100
            print(f"{label} | {bar:<{max_width}} {val:.4f} ({pct:.1f}%)")
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
    print("\n🔬 SHARED MOE PROJECTOR FORENSIC ANALYSIS")
    print(f"   Target: {file_path}")
    print("=" * 70)

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
        print("⚠️  No 'projector' prefix found. Searching for moe/expert keys...")
        proj_weights = {
            k: v for k, v in tensors.items() if "moe" in k or "expert" in k or "router" in k
        }

    if not proj_weights:
        print("❌ No SharedMoE parameters found in this checkpoint.")
        return

    print(f"📊 Loaded {len(proj_weights)} projector parameters.\n")

    # Print all parameter names and shapes
    print("   Parameters found:")
    for k, v in sorted(proj_weights.items()):
        print(f"      {k}: {list(v.shape)}")
    print()

    # Organize weights by component
    shared_expert = {}  # shared_expert.{gate,up,down}_proj.weight
    routed_experts = defaultdict(dict)  # experts.{idx}.{gate,up,down}_proj.weight
    router_weights = {}

    healthy = True
    for k, v in proj_weights.items():
        if not check_integrity(k, v):
            healthy = False

        # Router
        if "router" in k.lower():
            router_weights[k] = v

        # Shared expert
        match = re.search(r"shared_expert\.(gate_proj|up_proj|down_proj)\.weight", k)
        if match:
            proj_type = match.group(1)
            shared_expert[proj_type] = v

        # Routed experts
        match = re.search(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", k)
        if match:
            idx = int(match.group(1))
            proj_type = match.group(2)
            routed_experts[idx][proj_type] = v

    if not healthy:
        print("\n❌ ABORTING ANALYSIS: Model weights are corrupted (NaN/Inf).")
        return

    sorted_indices = sorted(routed_experts.keys())
    num_experts = len(sorted_indices)

    print(
        f"   Found: 1 shared expert, {num_experts} routed experts, "
        f"{len(router_weights)} router params"
    )

    # 1. SHARED VS ROUTED EXPERT BALANCE
    print("\n" + "=" * 70)
    print("[1] SHARED VS ROUTED EXPERT CONTRIBUTION ANALYSIS")
    print("=" * 70)

    if shared_expert and routed_experts:
        # Compare down_proj norms (output contribution)
        shared_down_norm = (
            torch.linalg.norm(shared_expert["down_proj"].float()).item()
            if "down_proj" in shared_expert
            else 0
        )

        routed_down_norms = []
        for idx in sorted_indices:
            if "down_proj" in routed_experts[idx]:
                norm = torch.linalg.norm(routed_experts[idx]["down_proj"].float()).item()
                routed_down_norms.append(norm)

        avg_routed_norm = np.mean(routed_down_norms) if routed_down_norms else 0
        total_norm = shared_down_norm + sum(routed_down_norms)

        print("\n   Output Contribution (down_proj L2 norms):")
        print(
            f"      Shared expert:     {shared_down_norm:.4f} ({shared_down_norm / total_norm * 100:.1f}%)"
        )
        print(f"      Routed avg:        {avg_routed_norm:.4f}")
        print(
            f"      Routed total:      {sum(routed_down_norms):.4f} ({sum(routed_down_norms) / total_norm * 100:.1f}%)"
        )

        # Visual comparison
        all_down_norms = [shared_down_norm] + routed_down_norms
        all_labels = ["Shared"] + [f"E{i}" for i in sorted_indices]
        print("\n   Down Projection Norms:")
        draw_ascii_bar(all_down_norms, labels=all_labels, max_width=40)

        # Check grow-in status
        ratio = avg_routed_norm / (shared_down_norm + 1e-9)
        print(f"\n   Routed/Shared Ratio: {ratio:.3f}")
        if ratio < 0.1:
            print("   ⚠️  Routed experts barely grown in - may need more training")
        elif ratio < 0.5:
            print("   ✅ Routed experts growing in gradually (healthy)")
        elif ratio < 2.0:
            print("   ✅ Routed experts well established")
        else:
            print("   ⚠️  Routed experts dominating - shared expert underutilized")

    # 2. ROUTER DIAGNOSTICS
    print("\n" + "=" * 70)
    print("[2] ROUTER DIAGNOSTICS")
    print("=" * 70)

    if router_weights:
        for name, w in sorted(router_weights.items()):
            std = w.float().std().item()
            mean = w.float().mean().item()
            norm = torch.linalg.norm(w.float()).item()
            max_val = w.float().abs().max().item()
            print(f"\n   {name}:")
            print(f"      Shape: {list(w.shape)}")
            print(f"      Std: {std:.5f}, Mean: {mean:.5f}, Norm: {norm:.4f}, Max: {max_val:.4f}")

            # Check if router has learned from zero-init
            if std < 0.001:
                print("      ⚠️  Router weights near zero - may not have learned routing")
            elif std < 0.01:
                print("      ✅ Router learning subtle routing patterns")
            else:
                print("      ✅ Router has learned distinct routing patterns")

            # Per-expert router weight analysis
            if w.shape[0] == num_experts:
                expert_norms = torch.linalg.norm(w.float(), dim=1).cpu().numpy()
                print("\n   Per-Expert Router Weight Norms:")
                draw_ascii_bar(
                    expert_norms,
                    labels=[f"E{i}" for i in range(len(expert_norms))],
                    max_width=40,
                )

                norm_std = np.std(expert_norms)
                norm_ratio = np.max(expert_norms) / (np.min(expert_norms) + 1e-9)
                print(f"   Norm Std Dev: {norm_std:.5f}, Max/Min Ratio: {norm_ratio:.2f}x")

                if norm_ratio > 3.0:
                    print("   ⚠️  Uneven router norms: Some experts may be favored")
                else:
                    print("   ✅ Router output weights look balanced")
    else:
        print("   ❌ No router weights found!")

    # 3. SWIGLU ANALYSIS
    print("\n" + "=" * 70)
    print("[3] SWIGLU EXPERT ANALYSIS")
    print("=" * 70)

    def analyze_swiglu_expert(name: str, expert_dict: dict):
        """Analyze a SwiGLU expert's gate/up/down projections."""
        print(f"\n   {name}:")
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if proj_name not in expert_dict:
                continue
            w = expert_dict[proj_name].float()
            std = w.std().item()
            norm = torch.linalg.norm(w).item()
            print(
                f"      {proj_name:<12} Shape: {str(list(w.shape)):<20} Std: {std:.5f}, Norm: {norm:.4f}"
            )

    if shared_expert:
        analyze_swiglu_expert("Shared Expert", shared_expert)

    # Analyze first and last routed expert as samples
    if num_experts > 0:
        analyze_swiglu_expert("Routed Expert 0", routed_experts[sorted_indices[0]])
        if num_experts > 1:
            analyze_swiglu_expert(
                f"Routed Expert {sorted_indices[-1]}", routed_experts[sorted_indices[-1]]
            )

    # 4. EXPERT DIVERSITY
    print("\n" + "=" * 70)
    print("[4] ROUTED EXPERT DIVERSITY (Specialization)")
    print("=" * 70)

    if num_experts > 1:
        # Compare gate_proj weights across routed experts
        gate_weights = []
        for idx in sorted_indices:
            if "gate_proj" in routed_experts[idx]:
                gate_weights.append(routed_experts[idx]["gate_proj"])

        if len(gate_weights) > 1:
            flat_experts = torch.stack([w.view(-1) for w in gate_weights]).float()
            norm_experts = functional.normalize(flat_experts, p=2, dim=1)

            sim_matrix = torch.mm(norm_experts, norm_experts.t())

            mask = ~torch.eye(num_experts, dtype=torch.bool, device=sim_matrix.device)
            off_diag = sim_matrix[mask]

            avg_sim = off_diag.mean().item()
            max_sim = off_diag.max().item()
            min_sim = off_diag.min().item()

            print("   Gate Projection Similarity (cosine):")
            print(f"      Avg: {avg_sim:.4f}, Max: {max_sim:.4f}, Min: {min_sim:.4f}")

            if avg_sim > 0.98:
                print("   ❌ RED ZONE: Experts are identical clones - no specialization!")
            elif avg_sim > 0.80:
                print("   ⚠️  YELLOW ZONE: Experts are highly correlated")
            elif avg_sim > 0.50:
                print("   ✅ GREEN ZONE: Experts are moderately specialized")
            else:
                print("   ✅ GREEN ZONE: Experts are highly specialized (diverse)")

            # Show pairwise similarities
            if num_experts <= 8:
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
        print("   ⚠️  Only 1 routed expert - cannot compute diversity")

    # 5. GROW-IN ANALYSIS (Zero-init down_proj)
    print("\n" + "=" * 70)
    print("[5] GROW-IN ANALYSIS (Routed Expert Activation)")
    print("=" * 70)

    print("\n   Routed experts start with zero-initialized down_proj weights.")
    print("   As training progresses, they 'grow in' from zero contribution.")

    down_proj_stats = []
    for idx in sorted_indices:
        if "down_proj" in routed_experts[idx]:
            w = routed_experts[idx]["down_proj"].float()
            norm = torch.linalg.norm(w).item()
            std = w.std().item()
            near_zero_pct = (w.abs() < 1e-6).float().mean().item() * 100
            down_proj_stats.append(
                {
                    "idx": idx,
                    "norm": norm,
                    "std": std,
                    "near_zero_pct": near_zero_pct,
                }
            )

    if down_proj_stats:
        print(f"\n   {'Expert':<10} {'L2 Norm':<12} {'Std':<12} {'Near-Zero %':<12}")
        print("   " + "-" * 46)
        for s in down_proj_stats:
            status = "🌱" if s["norm"] < 0.1 else "🌿" if s["norm"] < 1.0 else "🌳"
            print(
                f"   E{s['idx']:<9} {s['norm']:<12.4f} {s['std']:<12.6f} {s['near_zero_pct']:<12.2f}% {status}"
            )

        avg_norm = np.mean([s["norm"] for s in down_proj_stats])
        print(f"\n   Average down_proj norm: {avg_norm:.4f}")
        print("   Legend: 🌱 barely grown, 🌿 growing, 🌳 fully grown")

        # Compare to shared expert
        if "down_proj" in shared_expert:
            shared_norm = torch.linalg.norm(shared_expert["down_proj"].float()).item()
            print(f"\n   Shared expert down_proj norm: {shared_norm:.4f}")
            print(f"   Routed/Shared ratio: {avg_norm / shared_norm:.3f}")

    # 6. SPECTRAL HEALTH
    print("\n" + "=" * 70)
    print("[6] SPECTRAL HEALTH (Effective Rank)")
    print("=" * 70)

    # Check shared expert
    if "gate_proj" in shared_expert:
        rank = compute_effective_rank(shared_expert["gate_proj"])
        full_rank = min(shared_expert["gate_proj"].shape)
        print("\n   Shared Expert gate_proj:")
        print(f"      Effective Rank: {rank:.1f} / {full_rank} ({rank / full_rank * 100:.1f}%)")

    # Check routed experts
    if num_experts > 0 and "gate_proj" in routed_experts[sorted_indices[0]]:
        ranks = []
        for idx in sorted_indices:
            if "gate_proj" in routed_experts[idx]:
                rank = compute_effective_rank(routed_experts[idx]["gate_proj"])
                ranks.append(rank)

        if ranks:
            full_rank = min(routed_experts[sorted_indices[0]]["gate_proj"].shape)
            avg_rank = np.mean(ranks)

            print("\n   Routed Expert gate_proj Effective Rank:")
            print(
                f"      Average: {avg_rank:.1f} / {full_rank} ({avg_rank / full_rank * 100:.1f}%)"
            )
            draw_ascii_bar(ranks, labels=[f"E{i}" for i in sorted_indices], denominator=full_rank)

            if avg_rank / full_rank < 0.3:
                print("   ⚠️  Low effective rank - possible feature collapse")
            else:
                print("   ✅ Healthy effective rank")

    # 7. WEIGHT MAGNITUDE ANALYSIS
    print("\n" + "=" * 70)
    print("[7] WEIGHT MAGNITUDE ANALYSIS")
    print("=" * 70)

    all_weights = {"shared": [], "routed": [], "router": []}

    for name, w in proj_weights.items():
        if "weight" not in name:
            continue
        w_flat = w.float().view(-1)
        stats = {
            "name": name,
            "l2_norm": torch.linalg.norm(w_flat).item(),
            "mean_abs": w_flat.abs().mean().item(),
            "std": w_flat.std().item(),
        }

        if "shared_expert" in name:
            all_weights["shared"].append(stats)
        elif "experts" in name:
            all_weights["routed"].append(stats)
        elif "router" in name:
            all_weights["router"].append(stats)

    print("\n   Weight Statistics by Component:")
    print("   " + "-" * 60)
    print(f"   {'Component':<15} {'Avg L2 Norm':<15} {'Avg |W|':<15} {'Avg Std':<15}")
    print("   " + "-" * 60)

    for component, weights_list in all_weights.items():
        if not weights_list:
            continue
        avg_l2 = np.mean([w["l2_norm"] for w in weights_list])
        avg_mean = np.mean([w["mean_abs"] for w in weights_list])
        avg_std = np.mean([w["std"] for w in weights_list])
        print(f"   {component:<15} {avg_l2:<15.4f} {avg_mean:<15.6f} {avg_std:<15.6f}")

    # 8. CONDITION NUMBER ANALYSIS
    print("\n" + "=" * 70)
    print("[8] CONDITION NUMBER ANALYSIS (Numerical Stability)")
    print("=" * 70)

    condition_numbers = {"shared": [], "routed": [], "router": []}

    for name, w in proj_weights.items():
        if "weight" not in name:
            continue
        if w.dim() < 2:
            continue

        w_2d = w.float().view(w.shape[0], -1)
        if w_2d.shape[0] > 1 and w_2d.shape[1] > 1:
            try:
                s = torch.linalg.svdvals(w_2d)
                cond = (s[0] / (s[-1] + 1e-10)).item()

                if "shared_expert" in name:
                    condition_numbers["shared"].append(cond)
                elif "experts" in name:
                    condition_numbers["routed"].append(cond)
                elif "router" in name:
                    condition_numbers["router"].append(cond)
            except Exception:
                pass

    print("\n   Condition Numbers by Component (lower is better):")
    print("   " + "-" * 50)
    for component, conds in condition_numbers.items():
        if conds:
            avg_cond = np.mean(conds)
            max_cond = np.max(conds)
            status = "✅" if max_cond < 100 else "⚠️" if max_cond < 1000 else "❌"
            print(f"   {component:<10} Avg: {avg_cond:<10.1f} Max: {max_cond:<10.1f} {status}")

    # 9. INITIALIZATION DEVIATION
    print("\n" + "=" * 70)
    print("[9] INITIALIZATION DEVIATION ANALYSIS")
    print("=" * 70)

    # Expected init std based on input dim (approximation)
    # SharedMoE uses std = 1.0 / sqrt(in_dim)
    # We'll estimate based on weight shape
    print("\n   Comparing current weights to expected initialization...")

    deviation_stats = []
    for name, w in proj_weights.items():
        if "weight" not in name:
            continue

        current_std = w.float().std().item()

        # Estimate expected init std
        in_dim = w.shape[1] if w.dim() >= 2 else w.shape[0]
        if "down_proj" in name and "experts" in name:
            # Tiny orthogonal init (gain=0.01) for routed down_proj
            expected_std = 0.01 / (in_dim**0.5)
            ratio = current_std / (expected_std + 1e-10)
            if ratio > 3.0:
                deviation_stats.append(
                    {
                        "name": name,
                        "current_std": current_std,
                        "expected_std": expected_std,
                        "note": f"grown {ratio:.1f}x from init",
                    }
                )
        else:
            expected_std = 1.0 / (in_dim**0.5)
            ratio = current_std / (expected_std + 1e-10)
            if ratio < 0.3 or ratio > 3.0:
                deviation_stats.append(
                    {
                        "name": name,
                        "current_std": current_std,
                        "expected_std": expected_std,
                        "ratio": ratio,
                    }
                )

    if deviation_stats:
        print("\n   Notable deviations from expected initialization:")
        for d in deviation_stats[:10]:
            if "ratio" in d:
                print(f"      {d['name']}: {d['ratio']:.2f}x expected")
            else:
                print(f"      {d['name']}: {d['note']} (std={d['current_std']:.4f})")
    else:
        print("   ✅ All weights within expected initialization range")

    # 10. DEAD NEURON DETECTION
    print("\n" + "=" * 70)
    print("[10] DEAD NEURON DETECTION")
    print("=" * 70)

    dead_neuron_threshold = 1e-4
    dead_neuron_report = []

    for name, w in proj_weights.items():
        if "weight" not in name or w.dim() < 2:
            continue

        w_float = w.float()
        row_norms = w_float.norm(dim=1)
        col_norms = w_float.norm(dim=0)
        dead_rows = (row_norms < dead_neuron_threshold).sum().item()
        dead_cols = (col_norms < dead_neuron_threshold).sum().item()

        if dead_rows > 0 or dead_cols > 0:
            dead_neuron_report.append(
                {
                    "name": name,
                    "dead_rows": dead_rows,
                    "total_rows": w.shape[0],
                    "dead_cols": dead_cols,
                    "total_cols": w.shape[1],
                }
            )

    if dead_neuron_report:
        print("\n   ⚠️  Dead neurons detected (norm < 1e-4):")
        for d in dead_neuron_report:
            print(f"      {d['name']}:")
            print(
                f"         Rows: {d['dead_rows']}/{d['total_rows']} dead ({d['dead_rows'] / d['total_rows'] * 100:.1f}%)"
            )
            print(
                f"         Cols: {d['dead_cols']}/{d['total_cols']} dead ({d['dead_cols'] / d['total_cols'] * 100:.1f}%)"
            )
    else:
        print("\n   ✅ No dead neurons detected")

    # 11. GATE-UP ALIGNMENT (SwiGLU-specific)
    print("\n" + "=" * 70)
    print("[11] GATE-UP ALIGNMENT (SwiGLU Redundancy Check)")
    print("=" * 70)

    print("\n   In SwiGLU, gate and up projections are multiplied together.")
    print("   Low similarity = complementary patterns (good)")
    print("   High similarity = redundant capacity (bad)")

    gate_up_sims = []

    # Shared expert
    if "gate_proj" in shared_expert and "up_proj" in shared_expert:
        gate = shared_expert["gate_proj"].float().flatten()
        up = shared_expert["up_proj"].float().flatten()
        sim = functional.cosine_similarity(gate, up, dim=0).item()
        gate_up_sims.append(("Shared", sim))

    # Routed experts
    for idx in sorted_indices:
        if "gate_proj" in routed_experts[idx] and "up_proj" in routed_experts[idx]:
            gate = routed_experts[idx]["gate_proj"].float().flatten()
            up = routed_experts[idx]["up_proj"].float().flatten()
            sim = functional.cosine_similarity(gate, up, dim=0).item()
            gate_up_sims.append((f"E{idx}", sim))

    if gate_up_sims:
        print(f"\n   {'Expert':<10} {'Gate-Up Cosine Sim':<20} {'Status'}")
        print("   " + "-" * 45)
        for name, sim in gate_up_sims:
            status = (
                "✅ complementary"
                if abs(sim) < 0.3
                else "⚠️ correlated"
                if abs(sim) < 0.7
                else "❌ redundant"
            )
            print(f"   {name:<10} {sim:<20.4f} {status}")

        avg_sim = np.mean([abs(s[1]) for s in gate_up_sims])
        print(f"\n   Average |similarity|: {avg_sim:.4f}")
        if avg_sim < 0.3:
            print("   ✅ Gate and Up projections are learning complementary patterns")
        elif avg_sim < 0.7:
            print("   ⚠️  Some redundancy between Gate and Up projections")
        else:
            print("   ❌ High redundancy - Gate and Up learning similar patterns")

    # 12. WEIGHT DISTRIBUTION HEALTH
    print("\n" + "=" * 70)
    print("[12] WEIGHT DISTRIBUTION HEALTH (Kurtosis & Skewness)")
    print("=" * 70)

    print("\n   Healthy weights: kurtosis ≈ 0 (Gaussian), skewness ≈ 0 (symmetric)")
    print("   High kurtosis = heavy tails/outliers, High skewness = asymmetric")

    dist_stats = []
    for name, w in proj_weights.items():
        if "weight" not in name:
            continue
        w_flat = w.float().flatten()
        mean = w_flat.mean()
        std = w_flat.std()
        if std < 1e-10:
            continue

        centered = w_flat - mean
        kurtosis = (centered**4).mean() / (std**4) - 3  # Excess kurtosis
        skewness = (centered**3).mean() / (std**3)

        dist_stats.append(
            {
                "name": name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name,
                "kurtosis": kurtosis.item(),
                "skewness": skewness.item(),
            }
        )

    if dist_stats:
        # Group by component type
        print(f"\n   {'Component':<35} {'Kurtosis':<12} {'Skewness':<12} {'Status'}")
        print("   " + "-" * 70)
        for s in dist_stats:
            k_status = "✅" if abs(s["kurtosis"]) < 1 else "⚠️" if abs(s["kurtosis"]) < 3 else "❌"
            s_status = "✅" if abs(s["skewness"]) < 0.5 else "⚠️" if abs(s["skewness"]) < 1 else "❌"
            status = k_status if abs(s["kurtosis"]) > abs(s["skewness"]) else s_status
            print(f"   {s['name']:<35} {s['kurtosis']:<12.4f} {s['skewness']:<12.4f} {status}")

    # 13. ROUTER DECISION SHARPNESS
    print("\n" + "=" * 70)
    print("[13] ROUTER DECISION SHARPNESS")
    print("=" * 70)

    print("\n   How distinguishable are experts from the router's perspective?")
    print("   Higher distances = sharper routing decisions")

    if router_weights:
        for name, w in router_weights.items():
            if w.dim() < 2:
                continue

            w_float = w.float()
            n_experts = w_float.shape[0]

            if n_experts < 2:
                continue

            # Pairwise L2 distances between expert routing vectors
            distances = []
            for i in range(n_experts):
                for j in range(i + 1, n_experts):
                    d = (w_float[i] - w_float[j]).norm().item()
                    distances.append((f"E{i}-E{j}", d))

            if distances:
                dists_only = [d[1] for d in distances]
                print(f"\n   Router: {name}")
                print(f"      Mean distance: {np.mean(dists_only):.4f}")
                print(f"      Min distance:  {np.min(dists_only):.4f}")
                print(f"      Max distance:  {np.max(dists_only):.4f}")

                min_dist = np.min(dists_only)
                if min_dist < 0.01:
                    print("   ❌ Some expert pairs are nearly indistinguishable")
                elif min_dist < 0.1:
                    print("   ⚠️  Some expert pairs have low separation")
                else:
                    print("   ✅ All expert pairs are well-separated")

                # Show pairwise distances
                if len(distances) <= 10:
                    print("\n   Pairwise distances:")
                    for pair, d in sorted(distances, key=lambda x: x[1]):
                        print(f"      {pair}: {d:.4f}")

    # 14. SINGULAR VALUE DISTRIBUTION
    print("\n" + "=" * 70)
    print("[14] SINGULAR VALUE DISTRIBUTION")
    print("=" * 70)

    print("\n   Analyzing concentration of singular values (energy distribution)")

    sv_stats = []
    for expert_name, expert_dict in [("Shared", shared_expert)] + [
        (f"E{i}", routed_experts[i]) for i in sorted_indices
    ]:
        if "gate_proj" not in expert_dict:
            continue

        w = expert_dict["gate_proj"].float()
        try:
            s = torch.linalg.svdvals(w)
            s_sq = s**2
            total_energy = s_sq.sum().item()

            sv_stats.append(
                {
                    "name": expert_name,
                    "sv_max": s[0].item(),
                    "sv_min": s[-1].item(),
                    "sv_median": s[len(s) // 2].item(),
                    "top10_energy": (s_sq[:10].sum() / total_energy * 100).item(),
                    "top50_energy": (s_sq[:50].sum() / total_energy * 100).item(),
                }
            )
        except Exception:
            pass

    if sv_stats:
        print(
            f"\n   {'Expert':<10} {'σ_max':<10} {'σ_min':<10} {'σ_med':<10} {'Top-10 %':<10} {'Top-50 %':<10}"
        )
        print("   " + "-" * 60)
        for s in sv_stats:
            print(
                f"   {s['name']:<10} {s['sv_max']:<10.2f} {s['sv_min']:<10.4f} {s['sv_median']:<10.4f} {s['top10_energy']:<10.1f} {s['top50_energy']:<10.1f}"
            )

        avg_top10 = np.mean([s["top10_energy"] for s in sv_stats])
        print(f"\n   Average top-10 singular value energy: {avg_top10:.1f}%")
        if avg_top10 > 50:
            print("   ⚠️  Energy concentrated in top singular values (potential low-rank)")
        elif avg_top10 > 30:
            print("   ✅ Moderate energy distribution")
        else:
            print("   ✅ Energy well-distributed across singular values")

    # 15. SUMMARY
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)

    total_params = sum(v.numel() for v in proj_weights.values())
    shared_params = sum(v.numel() for k, v in proj_weights.items() if "shared_expert" in k)
    routed_params = sum(v.numel() for k, v in proj_weights.items() if "experts." in k)
    router_params = sum(v.numel() for k, v in proj_weights.items() if "router" in k)

    print(f"\n   Total projector parameters: {total_params:,}")
    print(f"      Shared expert: {shared_params:,} ({shared_params / total_params * 100:.1f}%)")
    print(f"      Routed experts: {routed_params:,} ({routed_params / total_params * 100:.1f}%)")
    print(f"      Router: {router_params:,} ({router_params / total_params * 100:.1f}%)")
    print(f"   Number of routed experts: {num_experts}")
    print(f"   Architecture: SharedMoE (1 shared + {num_experts} routed SwiGLU experts)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_moe_shared.py <path_to_model.safetensors>")
        sys.exit(1)
    analyze_checkpoint(sys.argv[1])
