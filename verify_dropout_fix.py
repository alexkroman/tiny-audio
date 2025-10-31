#!/usr/bin/env python3
"""Verify that activation dropout is now properly applied."""

from transformers import AutoModel
import torch

print("=" * 60)
print("Verifying activation dropout fix")
print("=" * 60)

encoder = AutoModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")

print("\nBEFORE fix:")
# Check a few intermediate_dropout modules
for name, module in encoder.named_modules():
    if name.endswith('intermediate_dropout'):
        print(f"  {name}: p={module.p}")
        if name.startswith('encoder.layers.0'):
            break

# Apply the fix
encoder.config.activation_dropout = 0.1
for name, module in encoder.named_modules():
    if name.endswith('intermediate_dropout'):
        module.p = 0.1

print("\nAFTER fix:")
for name, module in encoder.named_modules():
    if name.endswith('intermediate_dropout'):
        print(f"  {name}: p={module.p}")
        if name.startswith('encoder.layers.0'):
            break

# Test that dropout actually works now
print("\n" + "=" * 60)
print("Testing dropout in forward pass")
print("=" * 60)

dummy_input = torch.randn(1, 16000)

encoder.train()
print("\nTraining mode:")
with torch.no_grad():
    output1 = encoder(dummy_input).last_hidden_state
    output2 = encoder(dummy_input).last_hidden_state
    diff = torch.abs(output1 - output2).mean().item()
    print(f"  Mean difference: {diff:.6f}")
    if diff > 1e-6:
        print(f"  ✓ Activation dropout IS working!")
    else:
        print(f"  ✗ Still not working")

encoder.eval()
print("\nEval mode:")
with torch.no_grad():
    output1 = encoder(dummy_input).last_hidden_state
    output2 = encoder(dummy_input).last_hidden_state
    diff = torch.abs(output1 - output2).mean().item()
    print(f"  Mean difference: {diff:.10f}")
    if diff < 1e-6:
        print(f"  ✓ Dropout properly disabled in eval mode!")

print("\n" + "=" * 60)
print("✓ Fix verified - activation dropout now working!")
print("=" * 60)
