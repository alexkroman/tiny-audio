#!/usr/bin/env python3
"""Analyze whether adding LayerNorm to the AudioHead MLP projector helps.

Compares 5 variants: no norm, pre-norm, mid-norm, post-norm, pre+post-norm.
Measures activation stats, gradient stats, and loss convergence over 50 steps.
Uses MockDiaModel to avoid downloading 1.6B params.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from tiny_audio.audio_head import DAC_VOCAB_SIZE, NUM_DAC_CODEBOOKS


class MockDia(nn.Module):
    def __init__(self, hidden_dim=1024, vocab_size=1028):
        super().__init__()
        self.config = SimpleNamespace(
            delay_pattern=[0, 8, 9, 10, 11, 12, 13, 14, 15],
            pad_token_id=1025,
            eos_token_id=1024,
            bos_token_id=1026,
            decoder_config=SimpleNamespace(num_channels=9, vocab_size=vocab_size),
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.requires_grad_(False)

    def forward(self, encoder_outputs=None, labels=None, **kw):
        h = encoder_outputs.last_hidden_state
        p = h.mean(dim=(0, 1), keepdim=True)
        loss = None
        if labels is not None:
            e = p.expand(labels.shape[0], labels.shape[1], -1)
            logits = self.lm_head(e)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss)


LLM_DIM = 2048
DIA_DIM = 1024
STEPS = 50
LR = 1e-3


def build_variants():
    return {
        "baseline (no norm)": nn.Sequential(
            nn.Linear(LLM_DIM, LLM_DIM),
            nn.GELU(),
            nn.Linear(LLM_DIM, DIA_DIM),
        ),
        "pre-norm": nn.Sequential(
            nn.LayerNorm(LLM_DIM),
            nn.Linear(LLM_DIM, LLM_DIM),
            nn.GELU(),
            nn.Linear(LLM_DIM, DIA_DIM),
        ),
        "mid-norm (after GELU)": nn.Sequential(
            nn.Linear(LLM_DIM, LLM_DIM),
            nn.GELU(),
            nn.LayerNorm(LLM_DIM),
            nn.Linear(LLM_DIM, DIA_DIM),
        ),
        "post-norm": nn.Sequential(
            nn.Linear(LLM_DIM, LLM_DIM),
            nn.GELU(),
            nn.Linear(LLM_DIM, DIA_DIM),
            nn.LayerNorm(DIA_DIM),
        ),
        "pre+post-norm": nn.Sequential(
            nn.LayerNorm(LLM_DIM),
            nn.Linear(LLM_DIM, LLM_DIM),
            nn.GELU(),
            nn.Linear(LLM_DIM, DIA_DIM),
            nn.LayerNorm(DIA_DIM),
        ),
    }


def init_weights(projector):
    for m in projector.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def make_data(batch=2, seq_len=15, audio_seq=30):
    hidden = torch.randn(batch, seq_len, LLM_DIM)
    labels = torch.randint(0, DAC_VOCAB_SIZE, (batch * NUM_DAC_CODEBOOKS, audio_seq))
    labels[labels > 900] = -100
    dec_ids = torch.randint(0, DAC_VOCAB_SIZE, (batch, audio_seq, NUM_DAC_CODEBOOKS))
    dec_mask = torch.ones(batch, audio_seq, dtype=torch.long)
    return hidden, labels, dec_ids, dec_mask


def run_variant(projector, hidden, labels):
    dia = MockDia(DIA_DIM, DAC_VOCAB_SIZE)
    optimizer = torch.optim.Adam(projector.parameters(), lr=LR)

    # Activation stats before training
    with torch.no_grad():
        acts = projector(hidden)
    act_stats = {
        "mean": acts.mean().item(),
        "std": acts.std().item(),
        "max": acts.abs().max().item(),
    }

    # Per-layer activation stats
    layer_stats = []
    x = hidden
    with torch.no_grad():
        for i, layer in enumerate(projector):
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
        projected = projector(hidden)
        encoder_outputs = BaseModelOutput(last_hidden_state=projected)
        output = dia(encoder_outputs=encoder_outputs, labels=labels)
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
        "act_stats": act_stats,
        "layer_stats": layer_stats,
        "losses": losses,
        "avg_grad_norm": sum(grad_norms) / len(grad_norms),
        "avg_grad_max": sum(grad_maxes) / len(grad_maxes),
    }


def main():
    torch.manual_seed(42)
    hidden, labels, _, _ = make_data()

    print("=" * 100)
    print("AUDIO HEAD MLP NORM ANALYSIS")
    print("=" * 100)

    # --- Summary table ---
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
        result = run_variant(projector, hidden, labels)
        all_results[name] = result

        r = result
        init_loss = r["losses"][0]
        final_loss = r["losses"][-1]
        reduction = (init_loss - final_loss) / init_loss * 100

        print(
            f"{name:<25} {init_loss:>10.4f} {final_loss:>11.4f} {reduction:>9.1f}%"
            f" | {r['act_stats']['std']:>10.4f} {r['act_stats']['max']:>10.4f}"
            f" | {r['avg_grad_norm']:>10.4f} {r['avg_grad_max']:>10.4f}"
        )

    # --- Per-layer activation stats ---
    print(f"\n{'=' * 100}")
    print("PER-LAYER ACTIVATION STATS (before training)")
    print("=" * 100)

    for name, result in all_results.items():
        print(f"\n  {name}:")
        print(f"    {'Layer':<30} {'Mean':>10} {'Std':>10} {'Max':>10}")
        print(f"    {'-' * 65}")
        for ls in result["layer_stats"]:
            print(f"    {ls['name']:<30} {ls['mean']:>10.4f} {ls['std']:>10.4f} {ls['max']:>10.4f}")

    # --- Loss curves ---
    print(f"\n{'=' * 100}")
    print("LOSS CURVES (every 5 steps)")
    print("=" * 100)

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
