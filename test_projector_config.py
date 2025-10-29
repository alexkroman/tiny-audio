from types import SimpleNamespace

# Simulate different projector configurations
configs = [
    {
        "name": "Actual (Tiny Audio)",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 5,
        "hidden_dim": 8192,
    },
    {
        "name": "Smaller Projector",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 5,
        "hidden_dim": 4096,
    },
    {
        "name": "More Downsampling",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 8,
        "hidden_dim": 8192,
    },
    {
        "name": "Less Downsampling",
        "encoder_dim": 1280,
        "llm_dim": 2048,
        "downsample_rate": 2,
        "hidden_dim": 8192,
    },
]

def count_projector_params(config):
    """Calculate parameter count for a projector configuration."""
    stacked_dim = config["encoder_dim"] * config["downsample_rate"]
    hidden_dim = config["hidden_dim"]
    llm_dim = config["llm_dim"]

    # LayerNorm params (pre and post)
    ln_params = stacked_dim + llm_dim

    # Linear layer params (no bias)
    gate_params = stacked_dim * hidden_dim
    up_params = stacked_dim * hidden_dim
    down_params = hidden_dim * llm_dim

    total = ln_params + gate_params + up_params + down_params
    return total

def analyze_efficiency(config, audio_duration_sec=3.0):
    """Analyze computational efficiency."""
    # Assume 50 Hz encoder output (320x compression of 16kHz)
    encoder_frames = int(audio_duration_sec * 50)
    projector_frames = encoder_frames // config["downsample_rate"]

    return {
        "encoder_frames": encoder_frames,
        "projector_frames": projector_frames,
        "compression": encoder_frames / projector_frames,
        "ms_per_frame": (audio_duration_sec * 1000) / projector_frames,
    }

print("="*80)
print("PROJECTOR CONFIGURATION COMPARISON")
print("="*80)
print(f"{'Config':<25} {'Params':<15} {'Frames':<10} {'ms/frame':<12} {'Compression':<12}")
print("="*80)

for cfg in configs:
    params = count_projector_params(cfg)
    efficiency = analyze_efficiency(cfg)

    print(f"{cfg['name']:<25} {params:>14,} {efficiency['projector_frames']:>9} {efficiency['ms_per_frame']:>11.1f} {efficiency['compression']:>11.1f}x")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("Tradeoffs:")
print("  • Larger hidden_dim → More capacity, more parameters")
print("  • Higher downsample_rate → Fewer frames, faster inference, less detail")
print("  • Lower downsample_rate → More frames, slower inference, more detail")
print("\nTiny Audio's choice (5x, 8192 hidden):")
print("  ✓ Balances capacity with efficiency")
print("  ✓ ~122M params (trainable)")
print("  ✓ ~100ms per frame (good temporal resolution)")
