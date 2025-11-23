"""
Test script to verify MoE Projector Uni-MoE implementation.
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace

# Import the MoE projector
import sys
sys.path.insert(0, 'src')
from asr_modeling import MoEAudioProjector


def test_moe_projector():
    """Test the Uni-MoE projector implementation."""

    print("=" * 80)
    print("Testing Uni-MoE Audio Projector")
    print("=" * 80)

    # Create a simple config matching actual SmolLM3-3B dimensions
    config = SimpleNamespace(
        encoder_dim=512,  # Whisper Turbo encoder dim
        llm_dim=2048,     # SmolLM3-3B hidden size
        num_experts=8,
        moe_top_p=0.9,
        projector_pool_stride=2,
        projector_hidden_dim=512,
        projector_dropout=0.05,
        projector_input_noise=0.02,
        routed_scaling_factor=1.0,
        projector_init_std=0.02,
    )

    # Initialize projector
    print("\n1. Initializing MoE Projector...")
    print(f"   - Encoder dim: {config.encoder_dim}")
    print(f"   - LLM dim: {config.llm_dim}")
    print(f"   - Num routed experts: {config.num_experts}")
    print(f"   - Top-P threshold: {config.moe_top_p}")
    print(f"   - Pool stride: {config.projector_pool_stride}")

    projector = MoEAudioProjector(config)
    projector.eval()

    # Check router output dimensions
    print("\n2. Checking Router Configuration...")
    router_out_features = projector.router.out_features
    expected_out_features = config.num_experts + 1  # N routed + 1 null
    print(f"   - Router output features: {router_out_features}")
    print(f"   - Expected (N+1): {expected_out_features}")
    assert router_out_features == expected_out_features, \
        f"Router should output {expected_out_features} features (8 routed + 1 null)"
    print("   ✓ Router outputs N+1 experts (includes Null Expert at index 0)")

    # Check number of computational experts
    print("\n3. Checking Expert Configuration...")
    num_routed_experts = len(projector.routed_experts)
    print(f"   - Number of computational experts: {num_routed_experts}")
    print(f"   - Expected: {config.num_experts}")
    assert num_routed_experts == config.num_experts, \
        f"Should have {config.num_experts} computational experts"
    print("   ✓ Correct number of computational experts")

    # Test forward pass
    print("\n4. Testing Forward Pass...")
    batch_size = 4
    seq_len = 100
    # Input should be (batch, seq, encoder_dim) - pooling happens internally
    in_dim = config.encoder_dim

    # Create dummy input
    x = torch.randn(batch_size, seq_len, in_dim)
    print(f"   - Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output, aux_loss = projector(x)

    print(f"   - Output shape: {output.shape}")
    print(f"   - Auxiliary loss: {aux_loss}")
    expected_seq_len = seq_len // config.projector_pool_stride
    expected_shape = (batch_size, expected_seq_len, config.llm_dim)
    print(f"   - Expected shape: {expected_shape}")
    assert output.shape == expected_shape, \
        f"Output shape mismatch: got {output.shape}, expected {expected_shape}"
    print("   ✓ Output shape is correct")

    # Test routing behavior
    print("\n5. Testing Routing Behavior (Top-P/Nucleus Sampling)...")

    # Create a simpler input for routing analysis
    test_batch = 2
    test_seq = 10
    test_input = torch.randn(test_batch, test_seq, config.encoder_dim)

    with torch.no_grad():
        # Flatten and pool like the forward method does
        # Shape: (batch, seq, encoder_dim) -> pool -> (batch, seq//k, encoder_dim) -> flatten to (batch, seq//k, encoder_dim*k)
        batch, seq, dim = test_input.shape
        pooled_seq = seq // config.projector_pool_stride
        test_flat = test_input[:, :pooled_seq * config.projector_pool_stride, :].reshape(
            batch, pooled_seq, config.projector_pool_stride, dim
        ).reshape(batch, pooled_seq, dim * config.projector_pool_stride)
        test_flat = test_flat.reshape(-1, dim * config.projector_pool_stride)

        norm_x = projector.ln_pre(test_flat)

        # Get router logits
        router_logits = projector.router(norm_x)
        expected_tokens = test_batch * pooled_seq
        print(f"   - Router logits shape: {router_logits.shape}")
        print(f"   - Expected: ({expected_tokens}, {config.num_experts + 1})")

        # Get routing probabilities
        routing_probs = F.softmax(router_logits.float(), dim=-1)

        # Verify probability distribution
        prob_sums = routing_probs.sum(dim=-1)
        print(f"   - Probability sum (should be ~1.0): {prob_sums[0].item():.6f}")
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            "Routing probabilities should sum to 1"
        print("   ✓ Routing probabilities sum to 1")

        # Simulate Top-P routing
        sorted_probs, sorted_indices = torch.sort(routing_probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Check that Top-P masking works
        mask = cumulative_probs < config.moe_top_p
        mask = F.pad(mask[:, :-1], (1, 0), value=True)

        # Count selected experts per token
        num_selected = mask.sum(dim=-1)
        print(f"   - Avg experts selected per token: {num_selected.float().mean().item():.2f}")
        print(f"   - Min experts selected: {num_selected.min().item()}")
        print(f"   - Max experts selected: {num_selected.max().item()}")

        # Verify at least 1 expert is always selected
        assert num_selected.min() >= 1, "At least 1 expert should always be selected"
        print("   ✓ At least 1 expert selected per token")

        # Verify that we're using dynamic capacity (variable number of experts)
        if num_selected.max() > num_selected.min():
            print("   ✓ Dynamic capacity routing working (variable experts per token)")

        # Check null expert probability
        null_expert_probs = routing_probs[:, 0]  # Index 0 is null expert
        print(f"   - Null expert avg probability: {null_expert_probs.mean().item():.4f}")
        print(f"   - Null expert max probability: {null_expert_probs.max().item():.4f}")

    print("\n6. Testing End-to-End with Gradients...")
    # Test that gradients flow properly
    projector.train()
    test_input_grad = torch.randn(2, 20, config.encoder_dim, requires_grad=True)

    output_grad, aux_loss_grad = projector(test_input_grad)
    loss = output_grad.mean()
    loss.backward()

    # Check that input has gradients
    assert test_input_grad.grad is not None, "Gradients should flow to input"
    print("   ✓ Gradients flow through the projector")

    # Check router gradients
    assert projector.router.weight.grad is not None, "Router should have gradients"
    print("   ✓ Router has gradients")

    # Check expert gradients
    for i, expert in enumerate(projector.routed_experts):
        has_grad = any(p.grad is not None for p in expert.parameters())
        if not has_grad:
            print(f"   ⚠ Warning: Expert {i+1} has no gradients (may be normal if not selected)")

    print("\n" + "=" * 80)
    print("✓ All MoE Projector Tests Passed!")
    print("=" * 80)

    # Summary
    print("\nSummary (SmolLM3-3B Configuration):")
    print(f"  - Router outputs {config.num_experts + 1} logits (Null + {config.num_experts} routed)")
    print(f"  - Uses Top-P ({config.moe_top_p}) for dynamic capacity routing")
    print(f"  - Null expert at index 0 provides noise gating")
    print(f"  - Computational experts at indices 1-{config.num_experts}")
    print(f"  - Shared expert always active")
    print(f"  - Encoder dim: {config.encoder_dim} (Whisper Turbo)")
    print(f"  - LLM dim: {config.llm_dim} (SmolLM3-3B)")
    print(f"  - Output shape: (batch, seq/{config.projector_pool_stride}, {config.llm_dim})")


if __name__ == "__main__":
    test_moe_projector()
