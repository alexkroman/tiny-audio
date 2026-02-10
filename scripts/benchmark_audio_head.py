"""Sweep AudioHead transformer configurations to find best architecture.

Tests different combinations of layers, dims, and heads on synthetic
(text_tokens, neucodec_codes) pairs with fresh batches each step.

Usage:
    poetry run python scripts/benchmark_audio_head.py
"""

import time

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812

# NeuCodec FSQ constants (4^8 = 65536 codes)
NEUCODEC_VOCAB_SIZE = 65536
BOS_TOKEN = 65536
TOTAL_VOCAB = NEUCODEC_VOCAB_SIZE + 3  # 65539

# Training constants
TEXT_VOCAB_SIZE = 32000
BATCH_SIZE = 16
TEXT_LEN = 30
AUDIO_LEN = 60
NUM_STEPS = 300
LR = 1e-3


def make_batch(batch_size=BATCH_SIZE, text_len=TEXT_LEN, audio_len=AUDIO_LEN):
    """Create a fresh random (text_tokens, codec) batch each call."""
    text_tokens = torch.randint(0, TEXT_VOCAB_SIZE, (batch_size, text_len))
    codec_input_ids = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    codec_input_ids[:, 0] = BOS_TOKEN
    codec_labels = torch.randint(0, NEUCODEC_VOCAB_SIZE, (batch_size, audio_len))
    return text_tokens, codec_input_ids, codec_labels


class AudioHeadVariant(nn.Module):
    """Configurable AudioHead for benchmarking."""

    def __init__(self, decoder_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.text_embedding = nn.Embedding(TEXT_VOCAB_SIZE, decoder_dim)
        self.token_embedding = nn.Embedding(TOTAL_VOCAB, decoder_dim)

        from transformers import LlamaConfig, LlamaModel

        config = LlamaConfig(
            hidden_size=decoder_dim,
            intermediate_size=decoder_dim * 4,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            vocab_size=TOTAL_VOCAB,
            max_position_embeddings=4096,
        )
        self.decoder = LlamaModel(config)
        self.decoder.embed_tokens = None
        self.head = nn.Linear(decoder_dim, TOTAL_VOCAB)

    def forward(self, text_tokens, codec_input_ids, codec_labels):
        prefix = self.text_embedding(text_tokens)
        token_emb = self.token_embedding(codec_input_ids)
        hidden = torch.cat([prefix, token_emb], dim=1)

        text_len = prefix.shape[1]
        total_len = hidden.shape[1]
        batch_size = hidden.shape[0]

        causal_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=hidden.device), diagonal=1
        )
        causal_mask[:text_len, :text_len] = 0.0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        position_ids = (
            torch.arange(total_len, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
        )
        outputs = self.decoder(
            inputs_embeds=hidden, attention_mask=causal_mask, position_ids=position_ids
        )
        audio_hidden = outputs.last_hidden_state[:, text_len:]

        logits = self.head(audio_hidden).reshape(-1, TOTAL_VOCAB)
        return F.cross_entropy(logits, codec_labels.reshape(-1), ignore_index=-100)


def benchmark(decoder_dim, num_layers, num_heads, num_steps=NUM_STEPS):
    """Train a configuration and return metrics."""
    model = AudioHeadVariant(decoder_dim=decoder_dim, num_layers=num_layers, num_heads=num_heads)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    losses = []

    # Warmup
    for _ in range(5):
        text_tokens, codec_input_ids, codec_labels = make_batch()
        loss = model(text_tokens, codec_input_ids, codec_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    start = time.time()
    for _ in range(num_steps):
        text_tokens, codec_input_ids, codec_labels = make_batch()
        optimizer.zero_grad()
        loss = model(text_tokens, codec_input_ids, codec_labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    elapsed = time.time() - start

    return {
        "params": total_params,
        "final_loss": sum(losses[-20:]) / 20,  # avg of last 20 steps
        "min_loss": min(losses),
        "loss_25": losses[num_steps // 4],
        "loss_50": losses[num_steps // 2],
        "loss_75": losses[3 * num_steps // 4],
        "steps_per_sec": num_steps / elapsed,
        "loss_per_param": (sum(losses[-20:]) / 20) / (total_params / 1e6),  # efficiency
    }


def main():
    configs = [
        # (dim, layers, heads, label)
        (128, 2, 2, "128d/2L/2H"),
        (128, 4, 2, "128d/4L/2H"),
        (256, 2, 4, "256d/2L/4H"),
        (256, 4, 4, "256d/4L/4H"),
        (256, 6, 4, "256d/6L/4H"),
        (512, 2, 8, "512d/2L/8H"),
        (512, 4, 8, "512d/4L/8H"),
        (512, 6, 8, "512d/6L/8H (current)"),
        (512, 8, 8, "512d/8L/8H"),
        (768, 4, 12, "768d/4L/12H"),
        (768, 6, 12, "768d/6L/12H"),
    ]

    print("=" * 100)
    print("AudioHead Configuration Sweep")
    print(f"Batch={BATCH_SIZE}, TextLen={TEXT_LEN}, AudioLen={AUDIO_LEN}, Steps={NUM_STEPS}")
    print("Fresh random batches each step (no overfitting)")
    print("=" * 100)

    results = []
    for dim, layers, heads, label in configs:
        print(f"\n  {label}...", end="", flush=True)
        r = benchmark(dim, layers, heads)
        r["label"] = label
        results.append(r)
        print(
            f" {r['params'] / 1e6:.1f}M params, loss={r['final_loss']:.3f}, {r['steps_per_sec']:.1f} steps/s"
        )

    # Sort by final loss
    results.sort(key=lambda x: x["final_loss"])

    print("\n" + "=" * 100)
    print(
        f"{'Config':<25} {'Params':>8} {'Final':>8} {'Min':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'Steps/s':>8} {'Loss/M':>8}"
    )
    print("-" * 105)
    for r in results:
        print(
            f"{r['label']:<25} "
            f"{r['params'] / 1e6:>7.1f}M "
            f"{r['final_loss']:>8.3f} "
            f"{r['min_loss']:>8.3f} "
            f"{r['loss_25']:>8.3f} "
            f"{r['loss_50']:>8.3f} "
            f"{r['loss_75']:>8.3f} "
            f"{r['steps_per_sec']:>8.1f} "
            f"{r['loss_per_param']:>8.2f}"
        )

    print("\n(Lower Final = better generalization. Lower Loss/M = more param-efficient.)")
    print("(Steps/s = training throughput. Higher = faster training.)")

    # Best efficiency pick
    best_eff = min(results, key=lambda x: x["loss_per_param"])
    best_loss = results[0]  # already sorted
    print(f"\nBest loss: {best_loss['label']} (loss={best_loss['final_loss']:.3f})")
    print(f"Best efficiency: {best_eff['label']} (loss/M={best_eff['loss_per_param']:.2f})")


if __name__ == "__main__":
    main()
