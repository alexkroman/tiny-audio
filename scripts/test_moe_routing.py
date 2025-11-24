#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import sys
import os
import re

# ==============================================================================
# 1. MODEL DEFINITION
#    (Paste your class here if not importing from another file)
# ==============================================================================

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, dropout_rate=0.05):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        w12_out = self.w12(x)
        x_gate, x_val = w12_out.chunk(2, dim=-1)
        x = F.silu(x_gate) * x_val
        x = self.w3(x)
        x = self.dropout(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MoEAudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 2)
        self.num_experts = getattr(config, "num_experts", 8)
        self.top_k = getattr(config, "moe_top_k", 2)
        self.router_scale = 12.0
        self.jitter_noise = 0.0

        in_dim = config.encoder_dim * self.k
        self.out_dim = config.llm_dim
        expert_hidden = getattr(config, "projector_hidden_dim", 512)

        self.ln_pre = RMSNorm(in_dim, eps=1e-5)
        self.ln_post = RMSNorm(self.out_dim, eps=1e-5)
        self.router_weights = nn.Parameter(torch.randn(self.num_experts, in_dim) * 0.02)
        self.shared_expert = SwiGLU(in_dim, expert_hidden, self.out_dim)
        self.experts = nn.ModuleList([
            SwiGLU(in_dim, expert_hidden, self.out_dim) for _ in range(self.num_experts)
        ])

    def _safe_normalize(self, x, dim=-1, eps=1e-4):
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
        norm = torch.clamp(norm, min=eps)
        return x / norm.type_as(x)

    def forward(self, x):
        # Placeholder forward - we will monkey patch this during testing
        pass

# ==============================================================================
# 2. UTILITIES
# ==============================================================================

class MockConfig:
    def __init__(self):
        # Defaults - will be overwritten by auto-inference
        self.projector_pool_stride = 2
        self.num_experts = 8
        self.moe_top_k = 2
        self.encoder_dim = 1024 
        self.llm_dim = 4096       
        self.projector_hidden_dim = 512

def resolve_model_path(path_or_repo):
    """
    Handles local paths AND Hugging Face Hub IDs (e.g., 'user/repo').
    """
    # 1. If it exists locally, use it
    if os.path.exists(path_or_repo):
        if os.path.isdir(path_or_repo):
            # Try finding a safetensors file inside the dir
            candidates = ["model.safetensors", "adapter_model.safetensors"]
            for c in candidates:
                p = os.path.join(path_or_repo, c)
                if os.path.exists(p):
                    return p
        return path_or_repo

    # 2. If it looks like a HF Repo ID (has a slash and isn't a file)
    if "/" in path_or_repo and not path_or_repo.endswith(".safetensors"):
        try:
            print(f"🌐 Detected Hugging Face Repo: '{path_or_repo}'")
            print("   ⏳ Downloading model.safetensors...")
            from huggingface_hub import hf_hub_download
            
            # Try fetching model.safetensors, fall back to adapter_model if needed
            try:
                file_path = hf_hub_download(repo_id=path_or_repo, filename="model.safetensors")
            except:
                print("   ⚠️  'model.safetensors' not found, trying 'adapter_model.safetensors'...")
                file_path = hf_hub_download(repo_id=path_or_repo, filename="adapter_model.safetensors")
            
            print(f"   ✅ Downloaded to: {file_path}")
            return file_path
        except ImportError:
            print("❌ Error: You provided a HF Repo ID but 'huggingface_hub' is not installed.")
            print("   Run: pip install huggingface_hub")
            sys.exit(1)
        except Exception as e:
            print(f"❌ HF Download failed: {e}")
            sys.exit(1)
            
    return path_or_repo

def auto_infer_config(state_dict, config):
    """
    Updates MockConfig by inspecting the loaded weight shapes.
    Prevents dimension mismatch errors.
    """
    print("   🕵️  Auto-detecting model dimensions from weights...")
    
    # 1. Detect Num Experts & Input Dim from Router
    if 'router_weights' in state_dict:
        rw = state_dict['router_weights'] # [num_experts, in_dim]
        config.num_experts = rw.shape[0]
        actual_in_dim = rw.shape[1]
        
        # Infer encoder_dim based on stride (k)
        # in_dim = encoder_dim * k
        # We assume k=2 default, but you might need to guess k if it changes
        config.encoder_dim = actual_in_dim // config.projector_pool_stride
        print(f"      - Num Experts: {config.num_experts}")
        print(f"      - In Dim (Stride*{config.projector_pool_stride}): {actual_in_dim}")

    # 2. Detect LLM Dim from Shared Expert Output
    if 'shared_expert.w3.weight' in state_dict:
        w3 = state_dict['shared_expert.w3.weight'] # [out_dim, hidden]
        config.llm_dim = w3.shape[0]
        config.projector_hidden_dim = w3.shape[1]
        print(f"      - LLM Dim: {config.llm_dim}")
        print(f"      - Hidden Dim: {config.projector_hidden_dim}")
        
    return config

# ==============================================================================
# 3. TEST LOGIC
# ==============================================================================

def test_routing(path_or_repo, batch_size=4, seq_len=150):
    print(f"\n🚀 STARTING LIVE ROUTING TEST")
    
    # 1. Resolve Path (Local or Hub)
    checkpoint_path = resolve_model_path(path_or_repo)
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Could not locate file at {checkpoint_path}")
        return

    # 2. Load Weights First (to infer config)
    try:
        state_dict = load_file(checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load safetensors: {e}")
        return

    # 3. Auto-Configure & Init Model
    config = MockConfig()
    config = auto_infer_config(state_dict, config)
    
    model = MoEAudioProjector(config)
    
    # Load weights into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   ⚠️  Missing keys (ignorable if non-MoE): {len(missing)}")
    
    model.eval()
    model.float()

    # 4. Instrument Forward Pass (Monkey Patch)
    # We replace the empty forward() with the logic to capture indices
    captured_data = {}

    def instrumented_forward(x):
        batch_size, seq_len, dim = x.size()
        
        # Stride Pooling
        remainder = seq_len % model.k
        if remainder:
            x = F.pad(x, (0, 0, 0, model.k - remainder))
        x = x.view(batch_size, -1, dim * model.k)
        x_flat = x.view(-1, dim * model.k)
        
        # Router Logic
        norm_x = model.ln_pre(x_flat)
        input_normed = model._safe_normalize(norm_x, dim=-1)
        router_normed = model._safe_normalize(model.router_weights, dim=-1)
        router_logits = F.linear(input_normed, router_normed) * model.router_scale
        routing_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_probs, model.top_k, dim=-1)
        
        captured_data['indices'] = top_k_indices
        captured_data['probs'] = routing_probs
        return None, None

    model.forward = instrumented_forward

    # 5. Run Inference with Dummy Data
    print(f"   🎲 Generating random input noise: [{batch_size}, {seq_len}, {config.encoder_dim}]")
    dummy_input = torch.randn(batch_size, seq_len, config.encoder_dim)

    with torch.no_grad():
        model(dummy_input)

    # 6. Analyze Results
    if 'indices' not in captured_data:
        print("   ❌ Failed to capture routing indices.")
        return

    indices = captured_data['indices'].flatten()
    probs = captured_data['probs']
    
    total_tokens = indices.numel()
    counts = torch.bincount(indices, minlength=config.num_experts)
    percentages = (counts.float() / total_tokens) * 100
    avg_conf = probs.max(dim=-1).values.mean().item()

    print(f"\n📊 --- LIVE ROUTING DISTRIBUTION (Random Input) ---")
    print(f"   Router Confidence: {avg_conf:.4f} (1.0 = Certain, ~0.12 = Random Guessing)")
    print("-" * 65)
    print(f"{'Expert':<6} | {'Count':<8} | {'Usage %':<10} | {'Status'}")
    print("-" * 65)

    ideal = 100 / config.num_experts
    
    for i in range(config.num_experts):
        pct = percentages[i].item()
        count = counts[i].item()
        bar = "█" * int(pct / 2)
        
        status = "✅ Balanced"
        if pct < (ideal * 0.2): status = "⚠️  Starved"
        if pct > (ideal * 2.0): status = "🔥 Overloaded"
        
        print(f"{i:<6} | {count:<8} | {pct:5.1f}%      | {bar} {status}")

    print("-" * 65)

def test_routing_advanced(model, batch_size=4, seq_len=150, encoder_dim=1024):
    """
    Run advanced routing diagnostics: determinism and gradient flow.

    Args:
        model: MoEAudioProjector instance
        batch_size: Number of samples
        seq_len: Sequence length
        encoder_dim: Encoder dimension
    """
    print("\n" + "=" * 80)
    print("ADVANCED MoE ROUTING DIAGNOSTICS")
    print("=" * 80)

    # Force model to float32 and enable gradients
    model = model.float()
    model.train()

    # 1. ROUTING CONSISTENCY CHECK (Determinism)
    print("\n[TEST A] ROUTING CONSISTENCY (Jitter Check)")
    print("-" * 80)

    dummy_input = torch.randn(batch_size, seq_len, encoder_dim)

    # Temporarily disable jitter noise to test consistency
    old_jitter = model.jitter_noise
    model.jitter_noise = 0.0

    # Capture routing from two runs
    captured_data1 = {}
    captured_data2 = {}

    def make_instrumented_forward(capture_dict):
        def forward(x):
            batch_size, seq_len, dim = x.size()
            remainder = seq_len % model.k
            if remainder:
                x = F.pad(x, (0, 0, 0, model.k - remainder))
            x = x.view(batch_size, -1, dim * model.k)
            x_flat = x.view(-1, dim * model.k)

            norm_x = model.ln_pre(x_flat)
            input_normed = model._safe_normalize(norm_x, dim=-1)
            router_normed = model._safe_normalize(model.router_weights, dim=-1)
            router_logits = F.linear(input_normed, router_normed) * model.router_scale

            if model.training and model.jitter_noise > 0:
                router_logits = router_logits + (torch.randn_like(router_logits) * model.jitter_noise)

            routing_probs = F.softmax(router_logits.float(), dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_probs, model.top_k, dim=-1)

            capture_dict['indices'] = top_k_indices
            capture_dict['probs'] = routing_probs

            # Full forward for gradient test
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            top_k_weights = top_k_weights.to(x.dtype)

            shared_out = model.shared_expert(norm_x)
            routed_out = torch.zeros_like(shared_out)

            for k in range(model.top_k):
                indices_k = top_k_indices[:, k]
                weights_k = top_k_weights[:, k].unsqueeze(-1)
                for expert_idx, expert in enumerate(model.experts):
                    mask = (indices_k == expert_idx)
                    if mask.any():
                        routed_out[mask] += expert(norm_x[mask]) * weights_k[mask]

            final_out = model.ln_post(shared_out + routed_out)
            return final_out.view(batch_size, -1, model.out_dim), 0.0
        return forward

    # Run 1
    model.forward = make_instrumented_forward(captured_data1)
    with torch.no_grad():
        out1, _ = model(dummy_input)

    # Run 2 (Same Input)
    model.forward = make_instrumented_forward(captured_data2)
    with torch.no_grad():
        out2, _ = model(dummy_input)

    # Compare
    diff = (out1 - out2).abs().sum().item()
    if diff < 1e-5:
        print("   ✅ Router is Deterministic (Stable).")
    else:
        print(f"   ⚠️  Router is Flipping! Diff: {diff:.5f}")
        print("      (This causes 'forgetting'. Ensure jitter is off during inference).")

    model.jitter_noise = old_jitter  # Restore jitter

    # 2. DEAD GRADIENT CHECK (The "Zombie Expert" Detector)
    print("\n[TEST B] GRADIENT FLOW (Backward Pass Simulation)")
    print("-" * 80)

    # Reset gradients
    model.zero_grad()

    # Forward Pass with gradients enabled
    output, aux_loss = model(dummy_input)

    # Create a fake loss to force backprop
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)

    # Backward Pass
    print("   📉 Running Backward Pass...")
    loss.backward()

    print("\n   Gradient Magnitude per Expert (w12 weights):")
    print("-" * 80)

    has_dead_experts = False
    grad_mags = []

    for i, expert in enumerate(model.experts):
        # Check w12 gradients
        if expert.w12.weight.grad is None:
            print(f"Expert {i} | ❌ NO GRADIENT (Disconnected?)")
            has_dead_experts = True
            continue

        grad_mag = expert.w12.weight.grad.abs().mean().item()
        grad_mags.append(grad_mag)

        # Check if gradient is dangerously low
        status = "✅ Flowing"
        if grad_mag < 1e-9:
            status = "⚠️  Vanishing / Dead"
            has_dead_experts = True
        elif grad_mag > 1.0:
            status = "🔥 Exploding"

        # Draw Bar (Log scale)
        bar_len = int(min(grad_mag * 10000, 40))
        bar = "█" * bar_len
        print(f"Expert {i} | {grad_mag:.8f} | {bar} {status}")

    print("-" * 80)
    if not has_dead_experts:
        print("   ✅ ALL SYSTEMS GO: Every expert is learning.")
    else:
        print("   ❌ WARNING: Some experts are not receiving updates.")

    # Check shared expert
    if model.shared_expert.w12.weight.grad is not None:
        shared_grad = model.shared_expert.w12.weight.grad.abs().mean().item()
        print(f"\n   Shared Expert gradient: {shared_grad:.8f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_moe_routing.py <path_or_repo_id> [--advanced]")
        print("Example: python scripts/test_moe_routing.py my-user/audio-moe-projector")
        print("\nOptions:")
        print("  --advanced    Run advanced diagnostics (determinism & gradient flow)")
    else:
        path_or_repo = sys.argv[1]
        run_advanced = '--advanced' in sys.argv

        # Basic routing test
        test_routing(path_or_repo)

        # Advanced tests if requested
        if run_advanced:
            checkpoint_path = resolve_model_path(path_or_repo)
            if os.path.exists(checkpoint_path):
                state_dict = load_file(checkpoint_path)
                config = MockConfig()
                config = auto_infer_config(state_dict, config)
                model = MoEAudioProjector(config)
                model.load_state_dict(state_dict, strict=False)
                test_routing_advanced(model, encoder_dim=config.encoder_dim)