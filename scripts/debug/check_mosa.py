#!/usr/bin/env python3
"""Check MOSA model for router collapse and training health using real audio."""

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

app = typer.Typer(help="Check MOSA model for router collapse")
console = Console()

# Target metrics for end of training (4-expert dense MoE per MOSA-Base)
# Note: MOSA paper shows one expert often becomes "shared" with higher weight
TARGETS = {
    "entropy_min": 0.60,  # Minimum healthy entropy
    "entropy_max": 0.85,  # Maximum (above = still learning)
    "entropy_ideal": 0.70,  # Ideal entropy
    "expert_min": 0.10,  # Minimum expert weight (10%) - some can be specialists
    "expert_max": 0.40,  # Maximum expert weight (40%) - allows for shared expert
    "expert_ideal": 0.25,  # Ideal = uniform (25% for 4 experts)
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


def analyze_routing(probs, num_experts, label=""):
    """Analyze routing probabilities and print stats."""
    if label:
        console.print(f"\n{label}")
        console.print("-" * 40)

    mean_probs = probs.mean(dim=0)
    min_per_expert = probs.min(dim=0).values
    max_per_expert = probs.max(dim=0).values
    std_per_expert = probs.std(dim=0)

    console.print(
        f"Expert Weight Distribution (target: {TARGETS['expert_min'] * 100:.0f}-{TARGETS['expert_max'] * 100:.0f}% each):"
    )
    console.print(f"  {'Expert':<8} {'Mean':>7} {'Min':>7} {'Max':>7} {'Std':>7}  Distribution")
    console.print(f"  {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 20}")

    max_prob = 0
    min_prob = 1.0
    dominant_expert = 0
    for i, (mean_p, min_p, max_p, std_p) in enumerate(
        zip(mean_probs, min_per_expert, max_per_expert, std_per_expert)
    ):
        bar = "█" * int(mean_p * 40)
        console.print(
            f"  Expert {i}: {mean_p * 100:5.1f}% {min_p * 100:5.1f}% {max_p * 100:5.1f}% {std_p * 100:5.1f}%  {bar}"
        )
        if mean_p > max_prob:
            max_prob = mean_p.item()
            dominant_expert = i
        if mean_p < min_prob:
            min_prob = mean_p.item()

    # Per-token analysis
    console.print("\nPer-token routing variance:")
    total_variance = probs.var(dim=0).mean().item()
    console.print(f"  Avg variance across experts: {total_variance:.4f}")
    if total_variance < 0.001:
        console.print("  -> Very low variance: router gives similar weights to all tokens")
    elif total_variance < 0.01:
        console.print("  -> Low variance: router starting to differentiate")
    else:
        console.print("  -> Good variance: router differentiates between tokens")

    # Entropy
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(num_experts)))
    entropy_ratio = (entropy.mean() / max_entropy).item()
    entropy_std = (entropy / max_entropy).std().item()

    # Interpret entropy relative to targets
    if entropy_ratio < 0.5:
        entropy_status = "COLLAPSED"
    elif entropy_ratio < TARGETS["entropy_min"]:
        entropy_status = "over-specialized (below target)"
    elif entropy_ratio <= TARGETS["entropy_max"]:
        entropy_status = "IN TARGET RANGE"
    elif entropy_ratio < 0.95:
        entropy_status = "still learning (above target)"
    else:
        entropy_status = "very uniform (early training)"

    console.print(
        f"\nRouting entropy: {entropy.mean():.4f} / {max_entropy:.4f} ({entropy_ratio:.1%} of max)"
    )
    console.print(f"  Std across tokens: {entropy_std:.1%}")
    console.print(f"  Target range: {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%}")
    console.print(f"  -> {entropy_status}")

    return max_prob, min_prob, dominant_expert, entropy_ratio


def check_mosa(
    model_id: str = "mazesmazes/tiny-audio",
    force_download: bool = True,
    num_samples: int = 20,
):
    """Analyze MOSA model weights for training health using real audio."""
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel

    console.print("=" * 80)
    console.print(f"[bold]MOSA Health Check: {model_id}[/bold]")
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

    # Build router forward function
    # MOSA-Base uses 2-layer router: Linear -> ReLU -> Linear (indices 0 and 2)
    router_layers = []
    for i in [0, 2]:
        key_w = f"projector.router.{i}.weight"
        key_b = f"projector.router.{i}.bias"
        if key_w not in weights:
            console.print(f"[red]ERROR: Router layer {i} not found. Available keys:[/red]")
            for k in sorted(weights.keys()):
                if "router" in k or "projector" in k:
                    console.print(f"  {k}")
            raise KeyError(f"Missing {key_w} - is this a MOSA model?")
        w = weights[key_w].float()
        b = weights[key_b].float()
        router_layers.append((w, b))

    def forward_router(x):
        for i, (w, b) in enumerate(router_layers):
            x = functional.linear(x, w, b)
            if i < len(router_layers) - 1:
                x = functional.relu(x)  # ReLU per MOSA paper
        return x

    num_experts = router_layers[-1][0].shape[0]
    encoder_dim = router_layers[0][0].shape[1]

    # Download sample audio
    console.print()
    samples = download_sample_audio(num_samples)

    # Get encoder model from config
    config_path = Path(path) / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    encoder_id = config.get("audio_model_id", "zai-org/GLM-ASR-Nano-2512")

    console.print(f"\nLoading encoder: {encoder_id}")
    encoder = AutoModel.from_pretrained(encoder_id, trust_remote_code=True)
    encoder.eval()

    # Use feature extractor (works for both Whisper and GLM-ASR)
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id, trust_remote_code=True)

    # Process all samples and collect routing probabilities
    all_probs = []
    per_sample_stats = []
    model_dtype = next(encoder.parameters()).dtype

    for i, (audio, sr, _text) in enumerate(samples):
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

        # Run through router
        x_real = hidden_states.squeeze(0).float()
        logits = forward_router(x_real)
        probs = functional.softmax(logits, dim=-1)
        all_probs.append(probs)

        # Per-sample stats
        mean_probs = probs.mean(dim=0)
        per_sample_stats.append(
            {
                "sample_idx": i,
                "num_tokens": probs.shape[0],
                "expert_means": mean_probs.tolist(),
                "expert_stds": probs.std(dim=0).tolist(),
            }
        )

    # Concatenate all routing probs across samples
    probs = torch.cat(all_probs, dim=0)
    console.print(f"\nProcessed {len(samples)} samples, {probs.shape[0]} total tokens")
    console.print(f"Router expects: {encoder_dim}-dim input, {num_experts} experts")

    # Identify shared expert (highest mean usage) per MOSA architecture
    mean_probs_all = probs.mean(dim=0)
    shared_expert = mean_probs_all.argmax().item()
    specialist_experts = [i for i in range(num_experts) if i != shared_expert]

    console.print("\nMOSA Architecture Analysis:")
    console.print(
        f"  Shared expert: Expert {shared_expert} ({mean_probs_all[shared_expert] * 100:.1f}% avg)"
    )
    console.print(f"  Specialist experts: {specialist_experts}")

    # Show per-sample variation if multiple samples
    if len(samples) > 1:
        console.print("\nPer-sample expert distribution (mean %):")
        header = f"  {'Sample':<8}"
        for e in range(num_experts):
            label = f"E{e}*" if e == shared_expert else f"E{e}"
            header += f" {label:>7}"
        console.print(header)
        for stat in per_sample_stats[:10]:  # Show first 10
            line = f"  {stat['sample_idx']:<8}"
            for m in stat["expert_means"]:
                line += f"   {m * 100:5.1f}%"
            console.print(line)
        if len(per_sample_stats) > 10:
            console.print(f"  ... and {len(per_sample_stats) - 10} more samples")

        # Cross-sample variance
        expert_means_per_sample = torch.tensor([s["expert_means"] for s in per_sample_stats])
        cross_sample_std = expert_means_per_sample.std(dim=0)
        console.print("\nCross-sample consistency (std of expert means across samples):")
        for e in range(num_experts):
            consistency = "consistent" if cross_sample_std[e] < 0.05 else "varies"
            console.print(f"  Expert {e}: {cross_sample_std[e] * 100:.1f}% std ({consistency})")

    # Within-sample routing dynamics
    console.print("\nWithin-sample routing dynamics:")
    avg_within_sample_std = torch.tensor([s["expert_stds"] for s in per_sample_stats]).mean(dim=0)
    for e in range(num_experts):
        if e == shared_expert:
            console.print(
                f"  Expert {e} (shared): {avg_within_sample_std[e] * 100:.1f}% avg std within samples"
            )
        else:
            console.print(
                f"  Expert {e} (specialist): {avg_within_sample_std[e] * 100:.1f}% avg std within samples"
            )

    # Specialist activation analysis
    console.print("\nSpecialist expert activation:")
    for e in specialist_experts:
        # When does this expert get > 25% routing weight?
        activation_rate = (probs[:, e] > 0.25).float().mean().item()
        peak_activation = probs[:, e].max().item()
        console.print(
            f"  Expert {e}: activates >25% on {activation_rate * 100:.1f}% of tokens, peak={peak_activation * 100:.1f}%"
        )

    console.print(f"\n1. ROUTER BEHAVIOR ({len(samples)} sample(s), {probs.shape[0]} tokens)")
    console.print("-" * 40)

    # Recompute logits stats from all samples
    all_logits = []
    for audio, sr, _ in samples:
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
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
        x_real = hidden_states.squeeze(0).float()
        all_logits.append(forward_router(x_real))
    logits = torch.cat(all_logits, dim=0)

    console.print(f"Logit mean: {logits.mean():.4f}, std: {logits.std():.4f}")
    console.print(f"Logit range: [{logits.min():.4f}, {logits.max():.4f}]")

    logit_exploded = logits.std() > 100
    if logit_exploded:
        console.print("[red]WARNING: Router logits have exploded![/red]")

    max_prob, min_prob, dominant_expert, entropy_ratio = analyze_routing(probs, num_experts)

    # Expert differentiation
    console.print("\n2. EXPERT DIFFERENTIATION")
    console.print("-" * 40)

    if "projector.experts.0.gate_proj.weight" in weights:
        expert_weights = [
            weights[f"projector.experts.{i}.gate_proj.weight"].float().flatten()
            for i in range(num_experts)
        ]
    elif "projector.experts.0.fc1.weight" in weights:
        expert_weights = [
            weights[f"projector.experts.{i}.fc1.weight"].float().flatten()
            for i in range(num_experts)
        ]
    else:
        console.print("Unknown expert architecture")
        expert_weights = None

    if expert_weights:
        flat = torch.stack(expert_weights)
        flat_norm = flat / flat.norm(dim=1, keepdim=True)
        cosine_sim = flat_norm @ flat_norm.T
        avg_sim = (cosine_sim.sum() - num_experts) / (num_experts * (num_experts - 1))
        console.print(f"Average pairwise cosine similarity: {avg_sim:.6f}")
        if avg_sim > 0.5:
            console.print("[yellow]WARNING: Experts are converging (losing diversity)[/yellow]")

    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 80)

    # Check for critical issues (MOSA-aware)
    issues = []
    if logit_exploded:
        issues.append("Router logits exploded (std > 100)")
    if entropy_ratio < 0.5:
        issues.append(f"Low routing entropy ({entropy_ratio:.1%}) - router may be collapsed")

    if issues:
        console.print("[red]ISSUES DETECTED:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
    else:
        console.print("[green]No issues detected.[/green]")

    # MOSA health summary
    console.print("\nMOSA Health:")
    console.print(f"  Shared expert (E{shared_expert}): {max_prob * 100:.1f}% avg routing")
    console.print(f"  Specialist experts: {[f'E{e}' for e in specialist_experts]}")
    console.print(
        f"  Routing entropy: {entropy_ratio:.1%} of max (target: {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%})"
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
    """Check MOSA model for router collapse."""
    success = check_mosa(model_id, force_download=not no_cache, num_samples=num_samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    app()
