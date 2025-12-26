#!/usr/bin/env python3
"""Check MOSA model for router collapse and training health."""

import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Target metrics for end of training (8-expert dense MoE)
# Note: MOSA paper shows one expert often becomes "shared" with higher weight
TARGETS = {
    "entropy_min": 0.60,  # Minimum healthy entropy
    "entropy_max": 0.80,  # Maximum (above = still learning)
    "entropy_ideal": 0.70,  # Ideal entropy
    "expert_min": 0.05,  # Minimum expert weight (5%) - some can be specialists
    "expert_max": 0.30,  # Maximum expert weight (30%) - allows for shared expert
    "expert_ideal": 0.125,  # Ideal = uniform (12.5% for 8 experts)
}


def analyze_routing(probs, num_experts, label=""):
    """Analyze routing probabilities and print stats."""
    if label:
        print(f"\n{label}")
        print("-" * 40)

    mean_probs = probs.mean(dim=0)
    print(
        f"Expert Weight Distribution (target: {TARGETS['expert_min'] * 100:.0f}-{TARGETS['expert_max'] * 100:.0f}% each):"
    )
    max_prob = 0
    min_prob = 1.0
    dominant_expert = 0
    for i, p in enumerate(mean_probs):
        bar = "█" * int(p * 100)
        status = ""
        if p > 0.5:
            status = " <- COLLAPSED"
        elif p > TARGETS["expert_max"]:
            status = " <- HIGH"
        elif p < TARGETS["expert_min"]:
            status = " <- LOW"
        print(f"  Expert {i}: {p:.3f} ({p * 100:5.1f}%) {bar}{status}")
        if p > max_prob:
            max_prob = p
            dominant_expert = i
        if p < min_prob:
            min_prob = p

    # Entropy
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(num_experts)))
    entropy_ratio = (entropy.mean() / max_entropy).item()

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
    print(f"  Target range: {TARGETS['entropy_min']:.0%}-{TARGETS['entropy_max']:.0%}")
    print(f"  -> {entropy_status}")

    return max_prob, min_prob, dominant_expert, entropy_ratio


def check_mosa(
    model_id: str = "mazesmazes/tiny-audio-mosa",
    force_download: bool = True,
    audio_path: str = None,
):
    """Analyze MOSA model weights for training health."""

    print("=" * 80)
    print(f"MOSA Health Check: {model_id}")
    print("=" * 80)

    # Download latest
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
    router_layers = []
    for i in range(0, 9, 2):
        w = weights[f"projector.router.{i}.weight"].float()
        b = weights[f"projector.router.{i}.bias"].float()
        router_layers.append((w, b))

    def forward_router(x):
        for i, (w, b) in enumerate(router_layers):
            x = functional.linear(x, w, b)
            if i < len(router_layers) - 1:
                x = functional.relu(x)  # ReLU per MOSA paper
        return x

    num_experts = router_layers[-1][0].shape[0]

    # 1. Random noise baseline
    print("\n1. ROUTER BEHAVIOR (random noise)")
    print("-" * 40)

    torch.manual_seed(42)
    x_random = torch.randn(5000, 1280)
    logits_random = forward_router(x_random)
    probs_random = functional.softmax(logits_random, dim=-1)

    print(f"Logit mean: {logits_random.mean():.4f}, std: {logits_random.std():.4f}")
    print(f"Logit range: [{logits_random.min():.4f}, {logits_random.max():.4f}]")

    logit_exploded = logits_random.std() > 100
    if logit_exploded:
        print("WARNING: Router logits have exploded!")

    max_prob, min_prob, dominant_expert, entropy_ratio = analyze_routing(probs_random, num_experts)

    # 2. Real audio (if provided)
    if audio_path:
        print("\n2. ROUTER BEHAVIOR (real audio)")
        print("-" * 40)

        try:
            import librosa
            from transformers import WhisperModel

            print(f"Loading audio: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"Audio duration: {len(audio) / sr:.1f}s")

            # Load Whisper encoder
            print("Loading Whisper encoder...")
            whisper = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo")
            whisper.eval()

            # Get encoder outputs
            from transformers import WhisperFeatureExtractor

            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-large-v3-turbo"
            )
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

            with torch.no_grad():
                encoder_outputs = whisper.encoder(inputs.input_features)
                hidden_states = encoder_outputs.last_hidden_state  # (1, seq, 1280)

            print(f"Encoder output shape: {hidden_states.shape}")

            # Run through router
            x_real = hidden_states.squeeze(0).float()  # (seq, 1280)
            logits_real = forward_router(x_real)
            probs_real = functional.softmax(logits_real, dim=-1)

            print(f"Logit mean: {logits_real.mean():.4f}, std: {logits_real.std():.4f}")
            print(f"Logit range: [{logits_real.min():.4f}, {logits_real.max():.4f}]")

            max_prob_real, min_prob_real, dominant_expert_real, entropy_ratio_real = (
                analyze_routing(probs_real, num_experts)
            )

            # Compare
            print("\n3. COMPARISON: Random vs Real Audio")
            print("-" * 40)
            print(f"Entropy (random): {entropy_ratio:.1%}")
            print(f"Entropy (real):   {entropy_ratio_real:.1%}")
            if entropy_ratio_real < entropy_ratio - 0.05:
                print("  -> Router is more selective with real audio!")
            elif entropy_ratio_real > entropy_ratio + 0.05:
                print("  -> Router is more uniform with real audio (unexpected)")
            else:
                print("  -> Similar behavior on random vs real")

        except Exception as e:
            print(f"Error processing audio: {e}")

    # Expert differentiation
    section_num = 3 if audio_path else 2
    print(f"\n{section_num}. EXPERT DIFFERENTIATION")
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
        "model_id", nargs="?", default="mazesmazes/tiny-audio-mosa", help="HuggingFace model ID"
    )
    parser.add_argument("--no-cache", action="store_true", help="Use cached model if available")
    parser.add_argument("--audio", type=str, help="Path to audio file for real encoder test")
    args = parser.parse_args()

    success = check_mosa(args.model_id, force_download=not args.no_cache, audio_path=args.audio)
    sys.exit(0 if success else 1)
