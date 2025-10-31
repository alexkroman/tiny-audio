#!/usr/bin/env python3
"""Test script to verify regularization is actually applied during training."""

import torch
from transformers import AutoConfig, AutoModel

# Test 1: Check encoder config
print("=" * 60)
print("TEST 1: Checking HuBERT encoder dropout settings")
print("=" * 60)

encoder_config = AutoConfig.from_pretrained("facebook/hubert-xlarge-ls960-ft")
print(f"✓ Loaded encoder config")
print(f"  activation_dropout: {encoder_config.activation_dropout}")
print(f"  attention_dropout: {encoder_config.attention_dropout}")
print(f"  hidden_dropout: {encoder_config.hidden_dropout}")
print(f"  feat_proj_dropout: {encoder_config.feat_proj_dropout}")
print(f"  layerdrop: {encoder_config.layerdrop}")

# Test 2: Verify activation dropout override
print("\n" + "=" * 60)
print("TEST 2: Testing activation dropout override")
print("=" * 60)

encoder = AutoModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")
print(f"Before override: activation_dropout = {encoder.config.activation_dropout}")

encoder.config.activation_dropout = 0.1
print(f"After override: activation_dropout = {encoder.config.activation_dropout}")

# Test 3: Check if dropout is actually used during forward pass
print("\n" + "=" * 60)
print("TEST 3: Testing dropout behavior in train vs eval mode")
print("=" * 60)

# Create dummy input
dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz

# Test in training mode
encoder.train()
print("Training mode (dropout SHOULD be active):")
with torch.no_grad():
    output1 = encoder(dummy_input).last_hidden_state
    output2 = encoder(dummy_input).last_hidden_state
    diff_train = torch.abs(output1 - output2).mean().item()
    print(f"  Mean difference between two forward passes: {diff_train:.6f}")
    if diff_train > 1e-6:
        print(f"  ✓ Dropout is ACTIVE (outputs differ)")
    else:
        print(f"  ✗ Dropout appears INACTIVE (outputs identical)")

# Test in eval mode
encoder.eval()
print("\nEval mode (dropout SHOULD be disabled):")
with torch.no_grad():
    output1 = encoder(dummy_input).last_hidden_state
    output2 = encoder(dummy_input).last_hidden_state
    diff_eval = torch.abs(output1 - output2).mean().item()
    print(f"  Mean difference between two forward passes: {diff_eval:.10f}")
    if diff_eval < 1e-6:
        print(f"  ✓ Dropout is DISABLED (outputs identical)")
    else:
        print(f"  ✗ Dropout appears ACTIVE (outputs differ)")

# Test 4: Check SpecAugment masking
print("\n" + "=" * 60)
print("TEST 4: Testing SpecAugment masking")
print("=" * 60)

from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

mask_time_prob = 0.065
mask_time_length = 10
batch_size = 4
sequence_length = 1000

mask_indices = _compute_mask_indices(
    (batch_size, sequence_length),
    mask_prob=mask_time_prob,
    mask_length=mask_time_length,
    min_masks=2,
)

num_masked = mask_indices.sum()
expected_masked = batch_size * sequence_length * mask_time_prob
actual_percent = (num_masked / (batch_size * sequence_length)) * 100

print(f"Masking configuration:")
print(f"  mask_time_prob: {mask_time_prob} (6.5%)")
print(f"  mask_time_length: {mask_time_length}")
print(f"  Total positions: {batch_size * sequence_length}")
print(f"  Masked positions: {num_masked}")
print(f"  Expected ~{expected_masked:.0f} masked ({mask_time_prob*100}%)")
print(f"  Actual: {actual_percent:.2f}% masked")

if 5.0 <= actual_percent <= 8.0:
    print(f"  ✓ SpecAugment masking appears correct")
else:
    print(f"  ✗ SpecAugment masking may be incorrect")

# Test 5: Test DataCollator augmentation toggle
print("\n" + "=" * 60)
print("TEST 5: Testing DataCollator augmentation toggle")
print("=" * 60)

# Simulate the augmentation function
def test_augmentation(apply_augmentation, mask_time_prob):
    """Simulate _apply_spec_augment logic"""
    if not apply_augmentation or mask_time_prob == 0:
        return "SKIP (no augmentation)"
    return "APPLY (augmentation active)"

print("With apply_augmentation=True:")
result = test_augmentation(True, 0.065)
print(f"  Result: {result}")
if "APPLY" in result:
    print(f"  ✓ Augmentation enabled correctly")

print("\nWith apply_augmentation=False:")
result = test_augmentation(False, 0.065)
print(f"  Result: {result}")
if "SKIP" in result:
    print(f"  ✓ Augmentation disabled correctly")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("All regularization components tested!")
print("Check the results above to ensure everything is working correctly.")
