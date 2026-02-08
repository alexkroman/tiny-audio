#!/usr/bin/env python3
"""Analyze norm placement in the ASR encoder→LLM projector (MLPAudioProjector).

Current architecture: Linear → RMSNorm → GELU → Linear (with frame stacking).
Compares variants: no norm, pre-norm, mid-norm (current), post-norm, pre+post-norm,
and RMSNorm vs LayerNorm.

Uses a mock LLM decoder to measure loss convergence, activation stats, and gradients.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# --- Config ---
ENCODER_DIM = 384  # whisper-tiny encoder dim
LLM_DIM = 576  # SmolLM2-135M hidden dim
POOL_STRIDE = 4
IN_DIM = ENCODER_DIM * POOL_STRIDE  # 1536 after frame stacking
VOCAB_SIZE = 49152  # SmolLM2 vocab
STEPS = 50
LR = 1e-3


class MockLMHead(nn.Module):
    """Mock LLM decoder: just a linear head for computing cross-entropy loss."""

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.requires_grad_(False)

    def forward(self, hidden_states, labels=None):
        logits = self.head(hidden_states)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class ProjectorVariant(nn.Module):
    """Projector with configurable norm placement, matching MLPAudioProjector's frame stacking."""

    def __init__(self, layers):
        super().__init__()
        self.k = POOL_STRIDE
        self.net = nn.Sequential(*layers)

    def get_output_length(self, input_length):
        return (input_length - self.k) // self.k + 1

    def forward(self, x):
        batch, seq, dim = x.shape
        out_len = (seq - self.k) // self.k + 1
        x = x[:, : out_len * self.k, :]
        x = x.reshape(batch, out_len, dim * self.k)
        return self.net(x)


def build_variants():
    return {
        "no norm": ProjectorVariant(
            [
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
            ]
        ),
        "RMSNorm mid (current)": ProjectorVariant(
            [
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                LlamaRMSNorm(LLM_DIM, eps=1e-6),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
            ]
        ),
        "LayerNorm mid": ProjectorVariant(
            [
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                nn.LayerNorm(LLM_DIM),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
            ]
        ),
        "RMSNorm pre": ProjectorVariant(
            [
                LlamaRMSNorm(IN_DIM, eps=1e-6),
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
            ]
        ),
        "RMSNorm post": ProjectorVariant(
            [
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
                LlamaRMSNorm(LLM_DIM, eps=1e-6),
            ]
        ),
        "RMSNorm pre+post": ProjectorVariant(
            [
                LlamaRMSNorm(IN_DIM, eps=1e-6),
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
                LlamaRMSNorm(LLM_DIM, eps=1e-6),
            ]
        ),
        "RMSNorm mid+post": ProjectorVariant(
            [
                nn.Linear(IN_DIM, LLM_DIM, bias=False),
                LlamaRMSNorm(LLM_DIM, eps=1e-6),
                nn.GELU(),
                nn.Linear(LLM_DIM, LLM_DIM, bias=False),
                LlamaRMSNorm(LLM_DIM, eps=1e-6),
            ]
        ),
    }


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def make_data(batch=4, enc_seq=100):
    """Simulate whisper-tiny encoder output."""
    encoder_out = torch.randn(batch, enc_seq, ENCODER_DIM)
    # After frame stacking + projector, output length:
    out_len = (enc_seq - POOL_STRIDE) // POOL_STRIDE + 1  # 24
    labels = torch.randint(0, VOCAB_SIZE, (batch, out_len))
    return encoder_out, labels


def run_variant(projector, lm_head, encoder_out, labels):
    optimizer = torch.optim.Adam(projector.parameters(), lr=LR)

    # Activation stats before training (per-layer)
    layer_stats = []
    x = encoder_out.clone()
    batch, seq, dim = x.shape
    out_len = (seq - POOL_STRIDE) // POOL_STRIDE + 1
    x = x[:, : out_len * POOL_STRIDE, :]
    x = x.reshape(batch, out_len, dim * POOL_STRIDE)

    with torch.no_grad():
        for i, layer in enumerate(projector.net):
            x = layer(x)
            layer_stats.append(
                {
                    "name": f"{i}: {layer.__class__.__name__}",
                    "mean": x.mean().item(),
                    "std": x.std().item(),
                    "max": x.abs().max().item(),
                }
            )

    # Training loop
    losses = []
    grad_norms = []
    grad_maxes = []

    for _step in range(STEPS):
        optimizer.zero_grad()
        projected = projector(encoder_out)
        output = lm_head(projected, labels=labels)
        loss = output.loss
        loss.backward()

        total_norm = 0.0
        max_grad = 0.0
        for p in projector.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
                max_grad = max(max_grad, p.grad.abs().max().item())
        grad_norms.append(total_norm**0.5)
        grad_maxes.append(max_grad)
        losses.append(loss.item())
        optimizer.step()

    return {
        "layer_stats": layer_stats,
        "losses": losses,
        "avg_grad_norm": sum(grad_norms) / len(grad_norms),
        "avg_grad_max": sum(grad_maxes) / len(grad_maxes),
    }


def main():
    torch.manual_seed(42)
    encoder_out, labels = make_data()

    print("=" * 110)
    print("ASR PROJECTOR (encoder → LLM) NORM ANALYSIS")
    print(
        f"Architecture: Whisper-tiny ({ENCODER_DIM}d) → frame stack (×{POOL_STRIDE}={IN_DIM}d) → MLP → SmolLM ({LLM_DIM}d)"
    )
    print("=" * 110)

    header = (
        f"{'Variant':<25} {'Init Loss':>10} {'Final Loss':>11} {'Reduction':>10}"
        f" | {'Act Std':>10} {'Act Max':>10}"
        f" | {'Grad Norm':>10} {'Grad Max':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    all_results = {}
    for name, projector in build_variants().items():
        torch.manual_seed(42)
        init_weights(projector)
        lm_head = MockLMHead(LLM_DIM, VOCAB_SIZE)
        result = run_variant(projector, lm_head, encoder_out, labels)
        all_results[name] = result

        r = result
        init_loss = r["losses"][0]
        final_loss = r["losses"][-1]
        reduction = (init_loss - final_loss) / init_loss * 100
        # Use last layer stats for summary
        last = r["layer_stats"][-1]
        print(
            f"{name:<25} {init_loss:>10.4f} {final_loss:>11.4f} {reduction:>9.1f}%"
            f" | {last['std']:>10.4f} {last['max']:>10.4f}"
            f" | {r['avg_grad_norm']:>10.4f} {r['avg_grad_max']:>10.4f}"
        )

    # Per-layer activation stats
    print(f"\n{'=' * 110}")
    print("PER-LAYER ACTIVATION STATS (before training)")
    print("=" * 110)

    for name, result in all_results.items():
        print(f"\n  {name}:")
        print(f"    {'Layer':<30} {'Mean':>10} {'Std':>10} {'Max':>10}")
        print(f"    {'-' * 65}")
        for ls in result["layer_stats"]:
            print(f"    {ls['name']:<30} {ls['mean']:>10.4f} {ls['std']:>10.4f} {ls['max']:>10.4f}")

    # Loss curves
    print(f"\n{'=' * 110}")
    print("LOSS CURVES (every 5 steps)")
    print("=" * 110)

    step_header = f"{'Step':<8}"
    for name in all_results:
        step_header += f"{name:<27}"
    print(step_header)
    print("-" * len(step_header))

    for step in list(range(0, STEPS, 5)) + [STEPS - 1]:
        row = f"{step:<8}"
        for name in all_results:
            row += f"{all_results[name]['losses'][step]:<27.4f}"
        print(row)


if __name__ == "__main__":
    main()
