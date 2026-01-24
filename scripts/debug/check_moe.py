#!/usr/bin/env python3
"""Check MoE model for router collapse and training health using real audio."""

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as functional
import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from safetensors.torch import load_file

app = typer.Typer(help="Check MoE model for router health")
console = Console()

# Target metrics for MoE with shared expert + sparse routed experts
TARGETS = {
    "entropy_min": 0.50,  # Minimum healthy entropy for routed experts
    "entropy_max": 0.90,  # Maximum (above = uniform, not specialized)
    "entropy_ideal": 0.70,  # Ideal entropy
    "expert_min": 0.15,  # Minimum routed expert selection rate
    "expert_max": 0.60,  # Maximum (one expert shouldn't dominate)
    "load_balance_max": 0.3,  # Maximum load imbalance (std of expert usage)
}


def download_sample_audio(num_samples: int = 20):
    """Download sample audio files for testing."""
    from datasets import load_dataset

    console.print(
        f"Downloading {num_samples} sample(s) from hf-internal-testing/librispeech_asr_dummy..."
    )
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )

    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]
        samples.append((audio, sr, text))
        if num_samples <= 5:
            text_preview = f"'{text[:60]}...'" if len(text) > 60 else f"'{text}'"
            console.print(f"  Sample {i + 1}: {len(audio) / sr:.1f}s - {text_preview}")

    console.print(f"Loaded {len(samples)} samples")
    return samples


def analyze_routing(probs, top_k_indices, num_experts, top_k, label=""):
    """Analyze MoE routing probabilities and top-k selection."""
    if label:
        console.print(f"\n{label}")
        console.print("-" * 40)

    # Sigmoid probabilities (before top-k)
    mean_probs = probs.mean(dim=0)
    std_probs = probs.std(dim=0)

    console.print("Sigmoid Router Probabilities (before top-k selection):")
    console.print(f"  {'Expert':<8} {'Mean':>7} {'Std':>7}  Distribution")
    console.print(f"  {'-' * 8} {'-' * 7} {'-' * 7}  {'-' * 20}")

    for i, (mean_p, std_p) in enumerate(zip(mean_probs, std_probs)):
        bar = "█" * int(mean_p * 40)
        console.print(f"  Expert {i}: {mean_p * 100:5.1f}% {std_p * 100:5.1f}%  {bar}")

    # Top-k selection frequency
    console.print(f"\nTop-{top_k} Selection Frequency:")
    selection_counts = torch.zeros(num_experts)
    for k in range(top_k):
        for expert_idx in range(num_experts):
            selection_counts[expert_idx] += (top_k_indices[:, k] == expert_idx).sum().item()

    total_selections = top_k_indices.numel()
    selection_freq = selection_counts / total_selections * top_k  # Normalize to percentage

    console.print(f"  {'Expert':<8} {'Selected':>10} {'Frequency':>10}  Distribution")
    console.print(f"  {'-' * 8} {'-' * 10} {'-' * 10}  {'-' * 20}")

    for i, (count, freq) in enumerate(zip(selection_counts, selection_freq)):
        bar = "█" * int(freq * 40)
        console.print(f"  Expert {i}: {int(count):>10} {freq * 100:>8.1f}%  {bar}")

    # Load balance analysis
    load_balance_std = selection_freq.std().item()
    ideal_freq = 1.0 / num_experts
    console.print("\nLoad Balance:")
    console.print(f"  Ideal frequency: {ideal_freq * 100:.1f}% per expert")
    console.print(f"  Actual std: {load_balance_std * 100:.1f}%")
    if load_balance_std < TARGETS["load_balance_max"]:
        console.print("  -> Good load balance")
    else:
        console.print("  -> Imbalanced (some experts over/under-used)")

    # Entropy of selection (how uniform is top-k choice)
    selection_probs = selection_freq / selection_freq.sum()
    entropy = -(selection_probs * (selection_probs + 1e-10).log()).sum()
    max_entropy = torch.log(torch.tensor(float(num_experts)))
    entropy_ratio = (entropy / max_entropy).item()

    console.print(
        f"\nRouting entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy_ratio:.1%} of max)"
    )
    console.print(f"  Target range: {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%}")

    if entropy_ratio < TARGETS["entropy_min"]:
        console.print("  -> LOW: Some experts rarely selected (potential collapse)")
    elif entropy_ratio > TARGETS["entropy_max"]:
        console.print("  -> HIGH: Very uniform selection (not specialized)")
    else:
        console.print("  -> IN TARGET RANGE")

    return selection_freq, entropy_ratio, load_balance_std


def check_moe(
    model_id: str = "mazesmazes/tiny-audio",
    force_download: bool = True,
    num_samples: int = 20,
):
    """Analyze MoE model weights for training health using real audio."""
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel

    console.print("=" * 80)
    console.print(f"[bold]MoE Health Check: {model_id}[/bold]")
    console.print("=" * 80)

    # Download latest model
    if force_download:
        cache_path = (
            Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"
        )
        if cache_path.exists():
            shutil.rmtree(cache_path)
            console.print("Cleared cache, downloading fresh...")

    path = snapshot_download(model_id)
    console.print(f"Model path: {path}")

    weights = load_file(f"{path}/model.safetensors")

    # Check if this is an MoE model
    router_key = "projector.moe.router.weight"
    if router_key not in weights:
        console.print("[red]ERROR: Router weight not found. Available keys:[/red]")
        for k in sorted(weights.keys()):
            if "projector" in k:
                console.print(f"  {k}")
        raise KeyError(f"Missing {router_key} - is this an MoE model?")

    router_weight = weights[router_key].float()
    num_experts = router_weight.shape[0]
    input_dim = router_weight.shape[1]

    console.print("\nMoE Architecture:")
    console.print(f"  Router input dim: {input_dim}")
    console.print(f"  Number of routed experts: {num_experts}")

    # Check for shared expert
    shared_expert_key = "projector.moe.shared_expert.fc1.weight"
    has_shared_expert = shared_expert_key in weights
    console.print(f"  Has shared expert: {has_shared_expert}")

    # Get top_k from config
    config_path = Path(path) / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    top_k = config.get("num_experts_per_tok", 2)
    console.print(f"  Top-k routing: {top_k}")

    # Download sample audio
    console.print()
    samples = download_sample_audio(num_samples)

    # Get encoder model from config
    encoder_id = config.get("audio_model_id", "zai-org/GLM-ASR-Nano-2512")

    console.print(f"\nLoading encoder: {encoder_id}")
    encoder = AutoModel.from_pretrained(encoder_id, trust_remote_code=True)
    encoder.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id, trust_remote_code=True)

    # Get projector config
    pool_stride = config.get("projector_pool_stride", 4)

    # Process all samples and collect routing data
    all_probs = []
    all_top_k_indices = []
    per_sample_stats = []
    model_dtype = next(encoder.parameters()).dtype

    for i, (audio, sr, _) in enumerate(samples):
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Get encoder outputs
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            input_features = inputs.input_features.to(model_dtype)

            if hasattr(encoder, "encoder"):
                encoder_outputs = encoder.encoder(input_features)
            elif hasattr(encoder, "audio_tower"):
                encoder_outputs = encoder.audio_tower(input_features)
            else:
                encoder_outputs = encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state

        # Apply frame stacking (matches MoE projector)
        x = hidden_states.squeeze(0).float()
        seq_len = x.shape[0]
        out_len = (seq_len - pool_stride) // pool_stride + 1
        x = x[: out_len * pool_stride, :]
        x = x.reshape(out_len, -1)  # Frame stacking

        # Run through router (sigmoid routing)
        router_logits = functional.linear(x, router_weight)
        router_probs = torch.sigmoid(router_logits)

        # Top-k selection
        _, top_k_indices = torch.topk(router_probs, top_k, dim=-1)

        all_probs.append(router_probs)
        all_top_k_indices.append(top_k_indices)

        # Per-sample stats
        per_sample_stats.append(
            {
                "sample_idx": i,
                "num_tokens": router_probs.shape[0],
                "mean_probs": router_probs.mean(dim=0).tolist(),
                "top_k_selection": top_k_indices.tolist(),
            }
        )

    # Concatenate all routing data
    probs = torch.cat(all_probs, dim=0)
    top_k_indices = torch.cat(all_top_k_indices, dim=0)

    console.print(
        f"\nProcessed {len(samples)} samples, {probs.shape[0]} total tokens (after frame stacking)"
    )

    # Analyze routing
    console.print(f"\n1. ROUTER BEHAVIOR ({len(samples)} sample(s), {probs.shape[0]} tokens)")
    selection_freq, entropy_ratio, load_balance_std = analyze_routing(
        probs, top_k_indices, num_experts, top_k
    )

    # Show per-sample variation
    if len(samples) > 1:
        console.print("\nPer-sample top-k selection patterns (first 10 samples):")
        for stat in per_sample_stats[:10]:
            # Count which experts are selected most often in this sample
            sample_selections = torch.tensor(stat["top_k_selection"])
            sample_counts = torch.zeros(num_experts)
            for k in range(top_k):
                for expert_idx in range(num_experts):
                    sample_counts[expert_idx] += (
                        (sample_selections[:, k] == expert_idx).sum().item()
                    )
            sample_freq = sample_counts / sample_counts.sum()
            freq_str = " ".join([f"E{i}:{f * 100:.0f}%" for i, f in enumerate(sample_freq)])
            console.print(f"  Sample {stat['sample_idx']}: {freq_str}")

    # Expert weight differentiation
    console.print("\n2. EXPERT DIFFERENTIATION")
    console.print("-" * 40)

    expert_weights = []
    for i in range(num_experts):
        key = f"projector.moe.experts.{i}.fc1.weight"
        if key in weights:
            expert_weights.append(weights[key].float().flatten())

    if expert_weights:
        flat = torch.stack(expert_weights)
        flat_norm = flat / flat.norm(dim=1, keepdim=True)
        cosine_sim = flat_norm @ flat_norm.T

        console.print("Pairwise cosine similarity between routed experts:")
        header = "        " + "".join([f"  E{i:>5}" for i in range(num_experts)])
        console.print(header)
        for i in range(num_experts):
            row = f"  E{i}:   "
            for j in range(num_experts):
                if i == j:
                    row += "      -"
                else:
                    row += f"  {cosine_sim[i, j]:.4f}"
            console.print(row)

        avg_sim = (cosine_sim.sum() - num_experts) / (num_experts * (num_experts - 1))
        console.print(f"\nAverage pairwise similarity: {avg_sim:.4f}")
        if avg_sim > 0.9:
            console.print("[red]WARNING: Experts are nearly identical (collapsed)[/red]")
        elif avg_sim > 0.7:
            console.print("[yellow]WARNING: Experts converging (losing diversity)[/yellow]")
        else:
            console.print("[green]Good expert diversity[/green]")

    # Check shared expert vs routed experts
    if has_shared_expert:
        console.print("\n3. SHARED EXPERT ANALYSIS")
        console.print("-" * 40)

        shared_weight = weights[shared_expert_key].float().flatten()
        shared_norm = shared_weight / shared_weight.norm()

        console.print("Cosine similarity between shared expert and routed experts:")
        for i, ew in enumerate(expert_weights):
            ew_norm = ew / ew.norm()
            sim = (shared_norm @ ew_norm).item()
            console.print(f"  Shared <-> Expert {i}: {sim:.4f}")

    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 80)

    issues = []

    # Check for expert collapse
    min_selection = selection_freq.min().item()
    if min_selection < TARGETS["expert_min"]:
        issues.append(
            f"Expert underutilization: one expert selected only {min_selection * 100:.1f}% of time"
        )

    max_selection = selection_freq.max().item()
    if max_selection > TARGETS["expert_max"]:
        issues.append(f"Expert dominance: one expert selected {max_selection * 100:.1f}% of time")

    if entropy_ratio < TARGETS["entropy_min"]:
        issues.append(f"Low routing entropy ({entropy_ratio:.1%}) - experts not balanced")

    if load_balance_std > TARGETS["load_balance_max"]:
        issues.append(f"Poor load balance (std={load_balance_std:.1%})")

    if expert_weights and avg_sim > 0.9:
        issues.append(f"Expert collapse detected (avg similarity={avg_sim:.4f})")

    if issues:
        console.print("[red]ISSUES DETECTED:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
    else:
        console.print("[green]No issues detected.[/green]")

    # MoE health summary
    console.print("\nMoE Health:")
    console.print(f"  Routed experts: {num_experts}")
    console.print(f"  Top-k: {top_k}")
    console.print(f"  Has shared expert: {has_shared_expert}")
    console.print(
        f"  Routing entropy: {entropy_ratio:.1%} (target: {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%})"
    )
    console.print(
        f"  Load balance std: {load_balance_std:.1%} (target: <{TARGETS['load_balance_max']:.0%})"
    )

    return len(issues) == 0


@app.command()
def main(
    model_id: Annotated[
        str,
        typer.Argument(help="HuggingFace model ID"),
    ] = "mazesmazes/tiny-audio",
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Use cached model if available"),
    ] = False,
    num_samples: Annotated[
        int,
        typer.Option("-n", "--num-samples", help="Number of audio samples to process"),
    ] = 20,
):
    """Check MoE model for router health and expert balance."""
    success = check_moe(model_id, force_download=not no_cache, num_samples=num_samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()
