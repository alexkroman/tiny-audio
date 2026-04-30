#!/usr/bin/env python3
"""Analyze LoRA adapter weights to determine training effectiveness per module."""

from collections import defaultdict
from typing import Annotated

import torch
import typer
from huggingface_hub import hf_hub_download
from rich.console import Console
from safetensors.torch import load_file

app = typer.Typer(help="Analyze LoRA adapter weights")
console = Console()

LORA_MODULE_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _section(title: str) -> None:
    console.print("\n" + "=" * 80)
    console.print(f"[bold]{title}[/bold]")
    console.print("=" * 80)


def analyze_lora_adapter(repo_id: str = "mazesmazes/tiny-audio"):
    """Download and analyze LoRA adapter weights."""

    console.print(f"Downloading adapter from {repo_id}...")
    adapter_path = hf_hub_download(repo_id=repo_id, filename="adapter_model.safetensors")  # nosec B615

    console.print(f"Loading weights from {adapter_path}...")
    state_dict = load_file(adapter_path)

    module_stats = defaultdict(
        lambda: {
            "combined_norms": [],
            "ranks": [],
            "effective_ranks": [],
            "energy_concentrations": [],
            "params": 0,
        }
    )

    lora_pairs: dict[str, dict] = {}
    for name, tensor in state_dict.items():
        module_type = next((p for p in name.split(".") if p in LORA_MODULE_TYPES), None)
        if module_type is None:
            continue
        for ab in ("A", "B"):
            marker = f".lora_{ab}.weight"
            if marker in name:
                base_name = name.replace(marker, "")
                pair = lora_pairs.setdefault(base_name, {})
                pair[ab] = tensor
                pair["module_type"] = module_type
                break

    _section("PER-LAYER ANALYSIS")

    total_params = 0
    all_effective_ratios = []

    for _base_name, pair in sorted(lora_pairs.items()):
        if "A" not in pair or "B" not in pair:
            continue

        A = pair["A"].float()
        B = pair["B"].float()
        module_type = pair["module_type"]

        rank = A.shape[0]
        W = B @ A
        W_norm = torch.norm(W).item()

        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        # Top-`rank` singular values are signal; the rest is numerical noise.
        S_topk = S[:rank]

        # Effective rank as exp(entropy of normalized singular values).
        S_norm = S_topk / S_topk.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
        effective_rank = torch.exp(torch.tensor(entropy)).item()
        rank_utilization = effective_rank / rank if rank > 0 else 0

        top_half_energy = (S_topk[: rank // 2] ** 2).sum().item()
        total_energy = (S_topk**2).sum().item()
        energy_concentration = top_half_energy / total_energy if total_energy > 0 else 0

        all_effective_ratios.append(rank_utilization)

        params = A.numel() + B.numel()
        total_params += params
        stats = module_stats[module_type]
        stats["combined_norms"].append(W_norm)
        stats["ranks"].append(rank)
        stats["effective_ranks"].append(effective_rank)
        stats["energy_concentrations"].append(energy_concentration)
        stats["params"] += params

    _section("SUMMARY BY MODULE TYPE")
    console.print(
        f"\n{'Module':<12} {'Count':>6} {'Params':>10} {'Avg Norm':>12} {'Eff Rank':>10} {'Rank Util':>11} {'Top50% E':>10}"
    )
    console.print("-" * 90)

    module_importance = []

    for module_type in LORA_MODULE_TYPES:
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

        norm_per_param = avg_norm / (params / count) if params > 0 else 0
        module_importance.append(
            (module_type, avg_norm, norm_per_param, rank_util, avg_energy_conc)
        )

        console.print(
            f"{module_type:<12} {count:>6} {params:>10,} {avg_norm:>12.4f} {avg_eff_rank:>10.1f} {rank_util:>10.1%} {avg_energy_conc:>10.1%}"
        )

    console.print("-" * 80)
    console.print(f"{'TOTAL':<12} {'':<6} {total_params:>10,}")

    avg_rank_util = (
        sum(all_effective_ratios) / len(all_effective_ratios) if all_effective_ratios else 0
    )

    _section("ANALYSIS & RECOMMENDATIONS")

    module_importance.sort(key=lambda x: x[2], reverse=True)

    console.print("\nModule importance (by norm per parameter):")
    for module_type, _avg_norm, norm_per_param, _rank_util, _energy_conc in module_importance:
        bar = "█" * int(norm_per_param * 1e6)
        console.print(f"  {module_type:<12} {bar}")

    console.print(f"\nOverall effective rank utilization: {avg_rank_util:.1%}")
    avg_energy_conc = (
        sum(x[4] for x in module_importance) / len(module_importance) if module_importance else 0
    )
    console.print(f"Average energy in top 50% of ranks: {avg_energy_conc:.1%}")

    if avg_energy_conc > 0.95:
        console.print("  → Energy highly concentrated in top ranks. Could try LOWER rank.")
    elif avg_energy_conc > 0.85:
        console.print("  → Energy moderately concentrated. Current rank is reasonable.")
    elif avg_energy_conc > 0.70:
        console.print("  → Energy spread across ranks. Current rank is well-utilized.")
    else:
        console.print("  → Energy evenly distributed. May benefit from HIGHER rank.")

    # Identify least important modules
    console.print("\nModule efficiency (norm per param, lower = less efficient):")
    module_importance.sort(key=lambda x: x[2])
    for module_type, _avg_norm, norm_per_param, _rank_util, energy_conc in module_importance[:3]:
        console.print(f"  {module_type}: {norm_per_param:.2e} (energy conc: {energy_conc:.1%})")

    console.print("\nLeast utilized modules could potentially be removed to reduce parameters.")

    # Get rank from first pair
    if lora_pairs:
        first_pair = next(iter(lora_pairs.values()))
        if "A" in first_pair:
            actual_rank = first_pair["A"].shape[0]
            console.print(f"\nCurrent LoRA rank: {actual_rank}")
            console.print(f"Total trainable params: {total_params:,} ({total_params / 1e6:.2f}M)")


@app.command()
def main(
    repo_id: Annotated[
        str,
        typer.Option("--repo-id", "-r", help="HuggingFace model ID"),
    ] = "mazesmazes/tiny-audio",
):
    """Analyze LoRA adapter weights."""
    analyze_lora_adapter(repo_id)


if __name__ == "__main__":
    app()
