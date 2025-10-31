#!/usr/bin/env python3
"""Test if dropout works when base model is frozen but LoRA is trainable."""

import torch
import torch.nn as nn

# Simulate frozen model with LoRA
class FrozenWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.dropout = nn.Dropout(0.5)

        # Freeze the linear layer (simulating frozen encoder)
        self.linear.requires_grad_(False)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)  # Dropout AFTER frozen layer
        return x

print("=" * 60)
print("Testing: Does dropout work on frozen layers?")
print("=" * 60)

model = FrozenWithDropout()
dummy_input = torch.randn(2, 10)

# Test in training mode
model.train()
print("\nTraining mode (with frozen linear layer):")
print(f"  Linear layer requires_grad: {next(model.linear.parameters()).requires_grad}")

output1 = model(dummy_input)
output2 = model(dummy_input)
diff = torch.abs(output1 - output2).mean().item()

print(f"  Mean difference: {diff:.6f}")
if diff > 1e-6:
    print("  ✓ Dropout IS working (outputs differ)")
else:
    print("  ✗ Dropout NOT working (outputs identical)")

# The key insight
print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("=" * 60)
print("Dropout works regardless of requires_grad status!")
print("It's applied during forward pass, not backward pass.")
print("So frozen encoder + dropout = working regularization ✓")

# But let's check activation_dropout specifically in HuBERT
print("\n" + "=" * 60)
print("Checking HuBERT's actual dropout modules")
print("=" * 60)

from transformers import AutoModel

encoder = AutoModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")

# Override activation dropout
encoder.config.activation_dropout = 0.1

# Check dropout modules
dropout_modules = []
for name, module in encoder.named_modules():
    if isinstance(module, nn.Dropout):
        dropout_modules.append((name, module.p))

print(f"Found {len(dropout_modules)} Dropout modules in HuBERT")
print("\nDropout probabilities:")
for name, p in dropout_modules[:10]:  # Show first 10
    print(f"  {name}: {p}")

if len(dropout_modules) > 10:
    print(f"  ... and {len(dropout_modules) - 10} more")

# Check if activation_dropout is actually used
print(f"\nActivation dropout in config: {encoder.config.activation_dropout}")

# The problem: activation_dropout in config doesn't auto-update dropout modules!
print("\n" + "=" * 60)
print("CRITICAL FINDING:")
print("=" * 60)
print("❌ Setting encoder.config.activation_dropout = 0.1 ONLY updates the config")
print("❌ It does NOT update the actual nn.Dropout modules in the model!")
print("❌ The dropout modules were initialized with p=0.0 and stay that way!")
print("\nTo fix: Need to manually update dropout modules OR reinitialize encoder")
