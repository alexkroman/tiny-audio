#!/usr/bin/env python3
"""Test SODA audio dataset loading."""

from datasets import load_dataset, Audio

# Load the SODA audio dataset
print("Loading SODA audio dataset...")
ds = load_dataset(
    "fixie-ai/soda-audio",
    name="default",
    split="train",
    streaming=True,
)

# Take first sample
print("Taking first sample...")
sample = next(iter(ds))

# Check what columns are available
print(f"Available columns: {list(sample.keys())}")

# Check the structure of audio_second_last_turn
if "audio_second_last_turn" in sample:
    audio = sample["audio_second_last_turn"]
    print(f"Type of audio_second_last_turn: {type(audio)}")
    if hasattr(audio, '__dict__'):
        print(f"Audio attributes: {audio.__dict__.keys()}")

# Check alt_last_turn
if "alt_last_turn" in sample:
    text = sample["alt_last_turn"]
    print(f"alt_last_turn text: {text[:100]}...")

# Try casting the audio column
print("\nTrying to cast audio column...")
ds = ds.cast_column("audio_second_last_turn", Audio(sampling_rate=16000))

# Take another sample after casting
sample = next(iter(ds))
audio = sample["audio_second_last_turn"]
print(f"After casting - Type of audio: {type(audio)}")
if hasattr(audio, '__dict__'):
    print(f"After casting - Audio attributes: {audio.__dict__.keys()}")

# Try renaming column
print("\nRenaming audio column...")
ds = ds.rename_column("audio_second_last_turn", "audio")
sample = next(iter(ds))
print(f"After renaming - Available columns: {list(sample.keys())}")