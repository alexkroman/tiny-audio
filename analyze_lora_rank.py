#!/usr/bin/env python3
"""
LoRA Rank Analysis Script

This script analyzes LoRA adapter weights to determine if the rank is sufficient
for both encoder and decoder. It computes several metrics:

1. Effective Rank (using SVD)
2. Stable Rank
3. Spectral Entropy
4. Rank Utilization Percentage

Low effective rank compared to configured rank suggests the rank might be too high.
High effective rank close to configured rank suggests good utilization or potential
rank insufficiency.
"""

import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, List, Tuple
import argparse


def load_lora_weights(checkpoint_dir: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load LoRA weights from safetensors files."""
    encoder_path = checkpoint_dir / "encoder.safetensors"
    decoder_path = checkpoint_dir / "decoder.safetensors"

    encoder_weights = load_file(str(encoder_path))
    decoder_weights = load_file(str(decoder_path))

    return encoder_weights, decoder_weights


def load_lora_config(checkpoint_dir: Path) -> Tuple[dict, dict]:
    """Load LoRA configurations."""
    encoder_config_path = checkpoint_dir / "encoder_lora_config.json"
    decoder_config_path = checkpoint_dir / "decoder_lora_config.json"

    with open(encoder_config_path) as f:
        encoder_config = json.load(f)

    with open(decoder_config_path) as f:
        decoder_config = json.load(f)

    return encoder_config, decoder_config


def compute_effective_rank(matrix: torch.Tensor) -> float:
    """
    Compute effective rank using singular values.
    Effective rank measures how many dimensions are meaningfully used.
    Formula: exp(entropy of normalized singular values)
    """
    # Convert to float32 for numerical stability
    matrix = matrix.float()

    # Compute SVD
    U, S, V = torch.svd(matrix)

    # Normalize singular values to form a probability distribution
    S_normalized = S / S.sum()

    # Compute entropy
    # Filter out very small values to avoid log(0)
    S_normalized = S_normalized[S_normalized > 1e-10]
    entropy = -(S_normalized * torch.log(S_normalized)).sum()

    # Effective rank is exp(entropy)
    effective_rank = torch.exp(entropy).item()

    return effective_rank


def compute_stable_rank(matrix: torch.Tensor) -> float:
    """
    Compute stable rank: ||A||_F^2 / ||A||_2^2
    This is the ratio of Frobenius norm squared to spectral norm squared.
    """
    matrix = matrix.float()

    frobenius_norm_sq = torch.norm(matrix, p='fro') ** 2
    spectral_norm_sq = torch.norm(matrix, p=2) ** 2

    stable_rank = (frobenius_norm_sq / spectral_norm_sq).item()

    return stable_rank


def compute_spectral_entropy(matrix: torch.Tensor) -> float:
    """
    Compute spectral entropy: entropy of normalized singular values.
    Higher entropy means more uniform distribution of information across dimensions.
    """
    matrix = matrix.float()

    # Compute SVD
    _, S, _ = torch.svd(matrix)

    # Normalize singular values
    S_normalized = S / S.sum()

    # Compute entropy
    S_normalized = S_normalized[S_normalized > 1e-10]
    entropy = -(S_normalized * torch.log(S_normalized)).sum().item()

    return entropy


def analyze_lora_layer(lora_A: torch.Tensor, lora_B: torch.Tensor,
                       layer_name: str, configured_rank: int) -> Dict:
    """
    Analyze a single LoRA layer (lora_A and lora_B).
    LoRA decomposition: ΔW = B @ A, where A has shape (r, d_in) and B has shape (d_out, r)
    """
    print(f"\n{'='*80}")
    print(f"Layer: {layer_name}")
    print(f"{'='*80}")
    print(f"LoRA A shape: {lora_A.shape}")
    print(f"LoRA B shape: {lora_B.shape}")
    print(f"Configured rank: {configured_rank}")

    # Compute metrics for both A and B matrices
    metrics_A = {
        'effective_rank': compute_effective_rank(lora_A),
        'stable_rank': compute_stable_rank(lora_A),
        'spectral_entropy': compute_spectral_entropy(lora_A),
    }

    metrics_B = {
        'effective_rank': compute_effective_rank(lora_B),
        'stable_rank': compute_stable_rank(lora_B),
        'spectral_entropy': compute_spectral_entropy(lora_B),
    }

    # Compute combined delta W = B @ A
    delta_W = torch.matmul(lora_B.float(), lora_A.float())
    metrics_combined = {
        'effective_rank': compute_effective_rank(delta_W),
        'stable_rank': compute_stable_rank(delta_W),
        'spectral_entropy': compute_spectral_entropy(delta_W),
    }

    # Print analysis
    print(f"\nLoRA A (rank={lora_A.shape[0]}):")
    print(f"  Effective Rank: {metrics_A['effective_rank']:.2f} ({metrics_A['effective_rank']/lora_A.shape[0]*100:.1f}% utilization)")
    print(f"  Stable Rank: {metrics_A['stable_rank']:.2f}")
    print(f"  Spectral Entropy: {metrics_A['spectral_entropy']:.4f}")

    print(f"\nLoRA B (rank={lora_B.shape[1]}):")
    print(f"  Effective Rank: {metrics_B['effective_rank']:.2f} ({metrics_B['effective_rank']/lora_B.shape[1]*100:.1f}% utilization)")
    print(f"  Stable Rank: {metrics_B['stable_rank']:.2f}")
    print(f"  Spectral Entropy: {metrics_B['spectral_entropy']:.4f}")

    print(f"\nCombined ΔW = B @ A (shape: {delta_W.shape}):")
    min_dim = min(delta_W.shape[0], delta_W.shape[1])
    print(f"  Effective Rank: {metrics_combined['effective_rank']:.2f} ({metrics_combined['effective_rank']/min_dim*100:.1f}% of min dimension)")
    print(f"  Stable Rank: {metrics_combined['stable_rank']:.2f}")
    print(f"  Spectral Entropy: {metrics_combined['spectral_entropy']:.4f}")

    # Compute rank utilization percentage
    rank_utilization = metrics_A['effective_rank'] / configured_rank * 100

    # Provide recommendations
    print(f"\n{'─'*80}")
    print("ANALYSIS:")
    print(f"{'─'*80}")

    if rank_utilization > 90:
        print("✅ HIGH RANK UTILIZATION (>90%)")
        print("   → Current rank is being fully utilized")
        print("   → Consider INCREASING rank if loss plateaus")
        print("   → Model may benefit from more parameters")
    elif rank_utilization > 70:
        print("✓ GOOD RANK UTILIZATION (70-90%)")
        print("   → Current rank seems appropriate")
        print("   → Most dimensions are being effectively used")
    elif rank_utilization > 50:
        print("⚠ MODERATE RANK UTILIZATION (50-70%)")
        print("   → Some capacity is unused")
        print("   → Could potentially reduce rank slightly")
        print("   → Monitor training dynamics before changing")
    else:
        print("⚠ LOW RANK UTILIZATION (<50%)")
        print("   → Significant unused capacity")
        print("   → Consider DECREASING rank")
        print("   → Current rank may be unnecessarily high")

    return {
        'layer_name': layer_name,
        'configured_rank': configured_rank,
        'lora_A': metrics_A,
        'lora_B': metrics_B,
        'combined': metrics_combined,
        'rank_utilization_pct': rank_utilization,
    }


def extract_lora_pairs(weights: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract LoRA A and B weight pairs from the weight dictionary.
    Returns a dict mapping layer names to (lora_A, lora_B) tuples.
    """
    lora_pairs = {}

    # Group weights by layer name
    for key in weights.keys():
        if 'lora_A' in key:
            # Extract base name (everything before .lora_A)
            base_name = key.replace('.lora_A.weight', '').replace('.lora_A', '')

            # Find corresponding lora_B
            lora_B_key = key.replace('lora_A', 'lora_B')

            if lora_B_key in weights:
                lora_pairs[base_name] = (weights[key], weights[lora_B_key])

    return lora_pairs


def analyze_model_component(component_name: str, weights: Dict[str, torch.Tensor],
                           config: dict) -> List[Dict]:
    """Analyze all LoRA layers in a model component (encoder or decoder)."""
    print(f"\n{'#'*80}")
    print(f"# ANALYZING {component_name.upper()}")
    print(f"{'#'*80}")
    print(f"\nConfiguration:")
    print(f"  Rank (r): {config['r']}")
    print(f"  LoRA Alpha: {config.get('lora_alpha', 'N/A')}")
    print(f"  Target Modules: {config['target_modules']}")
    print(f"  Dropout: {config.get('lora_dropout', 0.0)}")

    lora_pairs = extract_lora_pairs(weights)

    print(f"\nFound {len(lora_pairs)} LoRA layer pairs")

    results = []
    for layer_name, (lora_A, lora_B) in lora_pairs.items():
        result = analyze_lora_layer(lora_A, lora_B, layer_name, config['r'])
        results.append(result)

    # Compute aggregate statistics
    if results:
        avg_utilization = np.mean([r['rank_utilization_pct'] for r in results])
        min_utilization = np.min([r['rank_utilization_pct'] for r in results])
        max_utilization = np.max([r['rank_utilization_pct'] for r in results])

        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {component_name.upper()}")
        print(f"{'='*80}")
        print(f"Average Rank Utilization: {avg_utilization:.1f}%")
        print(f"Min Rank Utilization: {min_utilization:.1f}%")
        print(f"Max Rank Utilization: {max_utilization:.1f}%")
        print(f"\n{'─'*80}")
        print("OVERALL RECOMMENDATION:")
        print(f"{'─'*80}")

        if avg_utilization > 85:
            print(f"🚀 {component_name.upper()}: Consider INCREASING rank (current: r={config['r']})")
            print(f"   → Suggest trying r={config['r']*2} or r={config['r']*4}")
            print(f"   → High utilization indicates the model could benefit from more capacity")
        elif avg_utilization > 60:
            print(f"✓ {component_name.upper()}: Current rank seems appropriate (r={config['r']})")
            print(f"   → Rank is being efficiently utilized")
        else:
            print(f"💡 {component_name.upper()}: Consider DECREASING rank (current: r={config['r']})")
            print(f"   → Suggest trying r={config['r']//2} or r={max(4, config['r']//4)}")
            print(f"   → Lower utilization indicates some wasted capacity")

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze LoRA rank sufficiency')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='./downloaded_model/last-checkpoint',
                       help='Path to checkpoint directory')
    parser.add_argument('--output', type=str, default='lora_analysis.json',
                       help='Output JSON file for detailed results')

    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    print("="*80)
    print("LoRA RANK ANALYSIS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_dir}")

    # Load configurations
    encoder_config, decoder_config = load_lora_config(checkpoint_dir)

    # Load weights
    encoder_weights, decoder_weights = load_lora_weights(checkpoint_dir)

    # Analyze encoder
    encoder_results = analyze_model_component('ENCODER', encoder_weights, encoder_config)

    # Analyze decoder
    decoder_results = analyze_model_component('DECODER', decoder_weights, decoder_config)

    # Load training state if available
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        print(f"\n{'#'*80}")
        print("# TRAINING PROGRESS")
        print(f"{'#'*80}")
        print(f"Global Step: {trainer_state['global_step']}")
        print(f"Best Step: {trainer_state['best_global_step']}")
        print(f"Best Metric: {trainer_state['best_metric']:.4f}")

        # Get recent loss history
        log_history = trainer_state['log_history']
        recent_losses = [entry for entry in log_history if 'loss' in entry][-10:]

        if recent_losses:
            print(f"\nRecent Training Loss Trend (last 10 steps):")
            for entry in recent_losses:
                print(f"  Step {entry['step']}: {entry['loss']:.4f}")

            # Check if loss is plateauing
            if len(recent_losses) >= 5:
                recent_5 = [e['loss'] for e in recent_losses[-5:]]
                loss_std = np.std(recent_5)
                loss_mean = np.mean(recent_5)

                print(f"\nRecent Loss Statistics:")
                print(f"  Mean: {loss_mean:.4f}")
                print(f"  Std Dev: {loss_std:.4f}")
                print(f"  Coefficient of Variation: {loss_std/loss_mean*100:.2f}%")

                if loss_std / loss_mean < 0.05:
                    print("\n⚠ Loss appears to be PLATEAUING")
                    print("  → If rank utilization is high, consider increasing rank")
                    print("  → If rank utilization is low, model may need different architecture changes")

    # Save detailed results
    output_data = {
        'checkpoint_dir': str(checkpoint_dir),
        'encoder': {
            'config': encoder_config,
            'results': encoder_results,
        },
        'decoder': {
            'config': decoder_config,
            'results': decoder_results,
        }
    }

    # Convert tensors to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj

    output_data = convert_to_serializable(output_data)

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
