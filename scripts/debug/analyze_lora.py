#!/usr/bin/env python3
"""Analyze LoRA adapter weights to determine training effectiveness per module."""

import argparse
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def analyze_lora_adapter(repo_id: str = "mazesmazes/tiny-audio"):
    """Download and analyze LoRA adapter weights."""

    print(f"Downloading adapter from {repo_id}...")
    adapter_path = hf_hub_download(repo_id=repo_id, filename="adapter_model.safetensors")

    print(f"Loading weights from {adapter_path}...")
    state_dict = load_file(adapter_path)

    # Group by module type (q_proj, k_proj, etc.)
    module_stats = defaultdict(lambda: {"A_norms": [], "B_norms": [], "combined_norms": [],
                                         "ranks": [], "effective_ranks": [], "energy_concentrations": [], "params": 0})

    # Parse tensor names and compute stats
    lora_pairs = {}  # Map base name to (A, B) tensors

    for name, tensor in state_dict.items():
        # Extract module type from name like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        parts = name.split(".")

        # Find module type (q_proj, k_proj, etc.)
        module_type = None
        for part in parts:
            if part in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                module_type = part
                break

        if module_type is None:
            continue

        # Determine if this is lora_A or lora_B
        if "lora_A" in name:
            base_name = name.replace(".lora_A.weight", "")
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]["A"] = tensor
            lora_pairs[base_name]["module_type"] = module_type
        elif "lora_B" in name:
            base_name = name.replace(".lora_B.weight", "")
            if base_name not in lora_pairs:
                lora_pairs[base_name] = {}
            lora_pairs[base_name]["B"] = tensor
            lora_pairs[base_name]["module_type"] = module_type

    # Analyze each LoRA pair
    print("\n" + "=" * 80)
    print("PER-LAYER ANALYSIS")
    print("=" * 80)

    total_params = 0
    all_effective_ratios = []

    for base_name, pair in sorted(lora_pairs.items()):
        if "A" not in pair or "B" not in pair:
            continue

        A = pair["A"].float()  # Shape: (rank, in_features)
        B = pair["B"].float()  # Shape: (out_features, rank)
        module_type = pair["module_type"]

        rank = A.shape[0]

        # Compute norms
        A_norm = torch.norm(A).item()
        B_norm = torch.norm(B).item()

        # Compute combined weight matrix W = B @ A
        W = B @ A
        W_norm = torch.norm(W).item()

        # Compute effective rank via SVD
        _, S, _ = torch.linalg.svd(W, full_matrices=False)

        # Only look at top `rank` singular values (the rest are numerical noise)
        S_topk = S[:rank]

        # Effective rank via entropy-based measure (more informative than threshold)
        # Normalized singular values as probability distribution
        S_norm = S_topk / S_topk.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
        # Effective rank: exp(entropy), normalized to [0, 1]
        effective_rank = torch.exp(torch.tensor(entropy)).item()
        rank_utilization = effective_rank / rank if rank > 0 else 0

        # Also compute how much energy is in top half vs bottom half of rank
        top_half_energy = (S_topk[:rank//2] ** 2).sum().item()
        total_energy = (S_topk ** 2).sum().item()
        energy_concentration = top_half_energy / total_energy if total_energy > 0 else 0

        all_effective_ratios.append(rank_utilization)

        # Store stats
        params = A.numel() + B.numel()
        total_params += params
        module_stats[module_type]["A_norms"].append(A_norm)
        module_stats[module_type]["B_norms"].append(B_norm)
        module_stats[module_type]["combined_norms"].append(W_norm)
        module_stats[module_type]["ranks"].append(rank)
        module_stats[module_type]["effective_ranks"].append(effective_rank)
        module_stats[module_type]["energy_concentrations"].append(energy_concentration)
        module_stats[module_type]["params"] += params

    # Print summary by module type
    print("\n" + "=" * 80)
    print("SUMMARY BY MODULE TYPE")
    print("=" * 80)
    print(f"\n{'Module':<12} {'Count':>6} {'Params':>10} {'Avg Norm':>12} {'Eff Rank':>10} {'Rank Util':>11} {'Top50% E':>10}")
    print("-" * 90)

    module_importance = []

    for module_type in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        stats = module_stats[module_type]
        if not stats["combined_norms"]:
            continue

        count = len(stats["combined_norms"])
        params = stats["params"]
        avg_norm = sum(stats["combined_norms"]) / count
        avg_eff_rank = sum(stats["effective_ranks"]) / count
        avg_rank = sum(stats["ranks"]) / count
        rank_util = avg_eff_rank / avg_rank if avg_rank > 0 else 0
        avg_energy_conc = sum(stats["energy_concentrations"]) / count

        # Norm per parameter (normalized importance)
        norm_per_param = avg_norm / (params / count) if params > 0 else 0
        module_importance.append((module_type, avg_norm, norm_per_param, rank_util, avg_energy_conc))

        print(f"{module_type:<12} {count:>6} {params:>10,} {avg_norm:>12.4f} {avg_eff_rank:>10.1f} {rank_util:>10.1%} {avg_energy_conc:>10.1%}")

    # Overall stats
    print("-" * 80)
    print(f"{'TOTAL':<12} {'':<6} {total_params:>10,}")

    avg_rank_util = sum(all_effective_ratios) / len(all_effective_ratios) if all_effective_ratios else 0

    # Recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    # Sort by normalized importance
    module_importance.sort(key=lambda x: x[2], reverse=True)

    print("\nModule importance (by norm per parameter):")
    for module_type, avg_norm, norm_per_param, rank_util, energy_conc in module_importance:
        bar = "█" * int(norm_per_param * 1e6)  # Scale for visibility
        print(f"  {module_type:<12} {bar}")

    print(f"\nOverall effective rank utilization: {avg_rank_util:.1%}")
    avg_energy_conc = sum(x[4] for x in module_importance) / len(module_importance) if module_importance else 0
    print(f"Average energy in top 50% of ranks: {avg_energy_conc:.1%}")

    if avg_energy_conc > 0.95:
        print("  → Energy highly concentrated in top ranks. Could try LOWER rank.")
    elif avg_energy_conc > 0.85:
        print("  → Energy moderately concentrated. Current rank is reasonable.")
    elif avg_energy_conc > 0.70:
        print("  → Energy spread across ranks. Current rank is well-utilized.")
    else:
        print("  → Energy evenly distributed. May benefit from HIGHER rank.")

    # Identify least important modules
    print("\nModule efficiency (norm per param, lower = less efficient):")
    module_importance.sort(key=lambda x: x[2])
    for module_type, avg_norm, norm_per_param, rank_util, energy_conc in module_importance[:3]:
        print(f"  {module_type}: {norm_per_param:.2e} (energy conc: {energy_conc:.1%})")

    print("\nLeast utilized modules could potentially be removed to reduce parameters.")

    # Get rank from first pair
    if lora_pairs:
        first_pair = next(iter(lora_pairs.values()))
        if "A" in first_pair:
            actual_rank = first_pair["A"].shape[0]
            print(f"\nCurrent LoRA rank: {actual_rank}")
            print(f"Total trainable params: {total_params:,} ({total_params/1e6:.2f}M)")


def main():
    parser = argparse.ArgumentParser(description="Analyze LoRA adapter weights")
    parser.add_argument("--repo", default="mazesmazes/tiny-audio", help="HuggingFace repo ID")
    args = parser.parse_args()

    analyze_lora_adapter(args.repo)


if __name__ == "__main__":
    main()
