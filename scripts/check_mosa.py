#!/usr/bin/env python3
"""Check MOSA model for router collapse and training health using real audio."""

import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

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


def download_sample_audio():
    """Download a sample audio file from LibriSpeech for testing."""
    from datasets import load_dataset

    print("Downloading sample audio from LibriSpeech...")
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    sample = next(iter(ds))
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    text = sample["text"]
    print(f"Sample text: '{text[:80]}...'") if len(text) > 80 else print(f"Sample text: '{text}'")
    print(f"Duration: {len(audio) / sr:.1f}s, Sample rate: {sr}Hz")
    return audio, sr


def analyze_routing(probs, num_experts, label=""):
    """Analyze routing probabilities and print stats."""
    if label:
        print(f"\n{label}")
        print("-" * 40)

    mean_probs = probs.mean(dim=0)
    min_per_expert = probs.min(dim=0).values
    max_per_expert = probs.max(dim=0).values
    std_per_expert = probs.std(dim=0)

    print(
        f"Expert Weight Distribution (target: {TARGETS['expert_min'] * 100:.0f}-{TARGETS['expert_max'] * 100:.0f}% each):"
    )
    print(f"  {'Expert':<8} {'Mean':>7} {'Min':>7} {'Max':>7} {'Std':>7}  Distribution")
    print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*20}")

    max_prob = 0
    min_prob = 1.0
    dominant_expert = 0
    for i, (mean_p, min_p, max_p, std_p) in enumerate(
        zip(mean_probs, min_per_expert, max_per_expert, std_per_expert)
    ):
        bar = "█" * int(mean_p * 40)
        range_bar = f"[{min_p * 100:4.1f}-{max_p * 100:4.1f}%]"
        print(
            f"  Expert {i}: {mean_p * 100:5.1f}% {min_p * 100:5.1f}% {max_p * 100:5.1f}% {std_p * 100:5.1f}%  {bar}"
        )
        if mean_p > max_prob:
            max_prob = mean_p.item()
            dominant_expert = i
        if mean_p < min_prob:
            min_prob = mean_p.item()

    # Per-token analysis
    print(f"\nPer-token routing variance:")
    total_variance = probs.var(dim=0).mean().item()
    print(f"  Avg variance across experts: {total_variance:.4f}")
    if total_variance < 0.001:
        print("  -> Very low variance: router gives similar weights to all tokens")
    elif total_variance < 0.01:
        print("  -> Low variance: router starting to differentiate")
    else:
        print("  -> Good variance: router differentiates between tokens")

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

    print(
        f"\nRouting entropy: {entropy.mean():.4f} / {max_entropy:.4f} ({entropy_ratio:.1%} of max)"
    )
    print(f"  Std across tokens: {entropy_std:.1%}")
    print(f"  Target range: {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%}")
    print(f"  -> {entropy_status}")

    return max_prob, min_prob, dominant_expert, entropy_ratio


def check_mosa(
    model_id: str = "mazesmazes/tiny-audio",
    force_download: bool = True,
):
    """Analyze MOSA model weights for training health using real audio."""
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel

    print("=" * 80)
    print(f"MOSA Health Check: {model_id}")
    print("=" * 80)

    # Download latest model
    if force_download:
        cache_path = (
            Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"
        )
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print("Cleared cache, downloading fresh...")

    path = snapshot_download(model_id)
    print(f"Model path: {path}")

    weights = load_file(f"{path}/model.safetensors")

    # Build router forward function
    # MOSA-Base uses 2-layer router: Linear -> ReLU -> Linear (indices 0 and 2)
    router_layers = []
    for i in [0, 2]:
        key_w = f"projector.router.{i}.weight"
        key_b = f"projector.router.{i}.bias"
        if key_w not in weights:
            print(f"ERROR: Router layer {i} not found. Available keys:")
            for k in sorted(weights.keys()):
                if "router" in k or "projector" in k:
                    print(f"  {k}")
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
    print()
    audio, sr = download_sample_audio()

    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Get encoder model from config
    config_path = Path(path) / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    encoder_id = config.get("audio_model_id", "openai/whisper-large-v3-turbo")

    print(f"\nLoading encoder: {encoder_id}")
    encoder = AutoModel.from_pretrained(encoder_id, trust_remote_code=True)
    encoder.eval()

    # Use feature extractor (works for both Whisper and GLM-ASR)
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id, trust_remote_code=True)

    # Get encoder outputs
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        # Handle different encoder architectures
        # Convert inputs to model dtype (handles bfloat16 models)
        model_dtype = next(encoder.parameters()).dtype
        input_features = inputs.input_features.to(model_dtype)

        if hasattr(encoder, "encoder"):
            # Whisper-style: model has separate encoder
            encoder_outputs = encoder.encoder(input_features)
        elif hasattr(encoder, "audio_tower"):
            # GLM-ASR style: audio_tower is the encoder
            encoder_outputs = encoder.audio_tower(input_features)
        else:
            encoder_outputs = encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (1, seq, dim)

    print(f"Encoder output shape: {hidden_states.shape}")
    print(f"Router expects: {encoder_dim}-dim input, {num_experts} experts")

    # Run through router
    x_real = hidden_states.squeeze(0).float()  # (seq, dim)
    logits = forward_router(x_real)
    probs = functional.softmax(logits, dim=-1)

    print("\n1. ROUTER BEHAVIOR (real speech)")
    print("-" * 40)
    print(f"Logit mean: {logits.mean():.4f}, std: {logits.std():.4f}")
    print(f"Logit range: [{logits.min():.4f}, {logits.max():.4f}]")

    logit_exploded = logits.std() > 100
    if logit_exploded:
        print("WARNING: Router logits have exploded!")

    max_prob, min_prob, dominant_expert, entropy_ratio = analyze_routing(probs, num_experts)

    # Expert differentiation
    print("\n2. EXPERT DIFFERENTIATION")
    print("-" * 40)

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
        print("Unknown expert architecture")
        expert_weights = None

    if expert_weights:
        flat = torch.stack(expert_weights)
        flat_norm = flat / flat.norm(dim=1, keepdim=True)
        cosine_sim = flat_norm @ flat_norm.T
        avg_sim = (cosine_sim.sum() - num_experts) / (num_experts * (num_experts - 1))
        print(f"Average pairwise cosine similarity: {avg_sim:.6f}")
        if avg_sim > 0.5:
            print("WARNING: Experts are converging (losing diversity)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Check for critical issues
    issues = []
    if logit_exploded:
        issues.append("Router logits exploded (std > 100)")
    if max_prob > 0.5:
        issues.append(f"Expert {dominant_expert} dominates with {max_prob * 100:.1f}%")
    if entropy_ratio < 0.5:
        issues.append(f"Low routing entropy ({entropy_ratio:.1%}) - COLLAPSED")

    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print()

    # Progress toward targets
    print("PROGRESS TO TARGET:")
    print("-" * 40)

    # Entropy progress
    if entropy_ratio > TARGETS["entropy_max"]:
        entropy_status = f"above target ({entropy_ratio:.1%} -> {TARGETS['entropy_max']:.0%})"
    elif entropy_ratio >= TARGETS["entropy_min"]:
        entropy_status = "IN TARGET RANGE"
    else:
        entropy_status = f"below target ({entropy_ratio:.1%} -> {TARGETS['entropy_min']:.0%})"

    print(f"  Entropy:     {entropy_status}")

    # Expert balance progress
    expert_in_range = TARGETS["expert_min"] <= min_prob and max_prob <= TARGETS["expert_max"]
    if expert_in_range:
        print(f"  Expert dist: IN TARGET RANGE ({min_prob * 100:.1f}%-{max_prob * 100:.1f}%)")
    else:
        if min_prob < TARGETS["expert_min"]:
            print(
                f"  Expert dist: min too low ({min_prob * 100:.1f}% < {TARGETS['expert_min'] * 100:.0f}%)"
            )
        if max_prob > TARGETS["expert_max"]:
            print(
                f"  Expert dist: max too high ({max_prob * 100:.1f}% > {TARGETS['expert_max'] * 100:.0f}%)"
            )

    print()
    print(f"Target entropy:      {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%}")
    print(
        f"Target expert range: {TARGETS['expert_min'] * 100:.0f}%-{TARGETS['expert_max'] * 100:.0f}%"
    )

    if not issues:
        print("\nAll health checks passed!")

    return len(issues) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check MOSA model for router collapse")
    parser.add_argument(
        "model_id", nargs="?", default="mazesmazes/tiny-audio", help="HuggingFace model ID"
    )
    parser.add_argument("--no-cache", action="store_true", help="Use cached model if available")
    args = parser.parse_args()

    success = check_mosa(args.model_id, force_download=not args.no_cache)
    sys.exit(0 if success else 1)
