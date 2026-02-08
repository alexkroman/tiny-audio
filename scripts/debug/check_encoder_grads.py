#!/usr/bin/env python3
"""Check if gradients flow through Dia encoder layers back to the MLP projector."""

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


class MiniEncoder(nn.Module):
    """Minimal encoder that mimics Dia's encoder structure with self-attention layers."""

    def __init__(self, hidden_dim=1024, num_layers=4, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.requires_grad_(False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MiniDecoder(nn.Module):
    """Minimal decoder with cross-attention to encoder output."""

    def __init__(self, hidden_dim=1024, vocab_size=1028):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.requires_grad_(False)

    def forward(self, encoder_output, labels):
        # Use encoder output as both key/value, mean-pool as query
        query = encoder_output.mean(dim=1, keepdim=True).expand_as(encoder_output)
        attn_out, _ = self.cross_attn(query, encoder_output, encoder_output)
        expanded = attn_out.mean(dim=(0, 1), keepdim=True).expand(
            labels.shape[0], labels.shape[1], -1
        )
        logits = self.lm_head(expanded)
        return nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )


def check_gradient_flow():
    torch.manual_seed(42)

    llm_dim = 2048
    dia_dim = 1024
    batch = 2
    seq_len = 15
    audio_seq = 30
    num_codebooks = 9

    # Trainable projector (same as AudioHead)
    projector = nn.Sequential(
        nn.Linear(llm_dim, llm_dim),
        nn.GELU(),
        nn.Linear(llm_dim, dia_dim),
    )

    # Frozen encoder + decoder
    encoder = MiniEncoder(dia_dim, num_layers=4)
    decoder = MiniDecoder(dia_dim)

    # Synthetic data
    embeddings = torch.randn(batch, seq_len, llm_dim)
    labels = torch.randint(0, 1028, (batch * num_codebooks, audio_seq))

    # === Test 1: Direct path (no encoder) ===
    projector.zero_grad()
    projected = projector(embeddings)
    encoder_out = BaseModelOutput(last_hidden_state=projected)
    loss_direct = decoder(encoder_out.last_hidden_state, labels)
    loss_direct.backward()

    grad_norm_direct = (
        sum(p.grad.norm().item() ** 2 for p in projector.parameters() if p.grad is not None) ** 0.5
    )
    has_grads_direct = all(
        p.grad is not None and p.grad.abs().max() > 0 for p in projector.parameters()
    )

    print("Direct (no encoder):")
    print(f"  Loss: {loss_direct.item():.4f}")
    print(f"  Grad norm: {grad_norm_direct:.6f}")
    print(f"  All params have grads: {has_grads_direct}")
    for name, p in projector.named_parameters():
        if p.grad is not None:
            print(f"  {name}: grad norm={p.grad.norm():.6f}, max={p.grad.abs().max():.6f}")

    # === Test 2: Through frozen encoder ===
    projector.zero_grad()
    projected = projector(embeddings)
    encoder_output = encoder(projected)
    loss_encoder = decoder(encoder_output, labels)
    loss_encoder.backward()

    grad_norm_encoder = (
        sum(p.grad.norm().item() ** 2 for p in projector.parameters() if p.grad is not None) ** 0.5
    )
    has_grads_encoder = all(
        p.grad is not None and p.grad.abs().max() > 0 for p in projector.parameters()
    )

    print("\nThrough frozen encoder (4 layers):")
    print(f"  Loss: {loss_encoder.item():.4f}")
    print(f"  Grad norm: {grad_norm_encoder:.6f}")
    print(f"  All params have grads: {has_grads_encoder}")
    for name, p in projector.named_parameters():
        if p.grad is not None:
            print(f"  {name}: grad norm={p.grad.norm():.6f}, max={p.grad.abs().max():.6f}")

    # === Test 3: Through frozen encoder with gradient checkpointing ===
    projector.zero_grad()
    projected = projector(embeddings)
    hidden = projected
    for layer in encoder.layers:
        hidden = torch.utils.checkpoint.checkpoint(
            layer,
            hidden,
            use_reentrant=False,
        )
    hidden = encoder.norm(hidden)
    loss_ckpt = decoder(hidden, labels)
    loss_ckpt.backward()

    grad_norm_ckpt = (
        sum(p.grad.norm().item() ** 2 for p in projector.parameters() if p.grad is not None) ** 0.5
    )
    has_grads_ckpt = all(
        p.grad is not None and p.grad.abs().max() > 0 for p in projector.parameters()
    )

    print("\nThrough frozen encoder with gradient checkpointing:")
    print(f"  Loss: {loss_ckpt.item():.4f}")
    print(f"  Grad norm: {grad_norm_ckpt:.6f}")
    print(f"  All params have grads: {has_grads_ckpt}")
    for name, p in projector.named_parameters():
        if p.grad is not None:
            print(f"  {name}: grad norm={p.grad.norm():.6f}, max={p.grad.abs().max():.6f}")

    # === Summary ===
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Direct grad norm:       {grad_norm_direct:.6f}")
    print(f"Encoder grad norm:      {grad_norm_encoder:.6f}")
    print(f"Checkpoint grad norm:   {grad_norm_ckpt:.6f}")
    print(f"Encoder/Direct ratio:   {grad_norm_encoder / grad_norm_direct:.4f}x")
    print(f"Checkpoint/Direct ratio: {grad_norm_ckpt / grad_norm_direct:.4f}x")


if __name__ == "__main__":
    check_gradient_flow()
