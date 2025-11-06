#!/usr/bin/env python3
"""Test AudioSet Strong dataset loading."""

from datasets import load_dataset, Audio

# Load the AudioSet Strong dataset
print("Loading AudioSet Strong dataset...")
ds = load_dataset(
    "CLAPv2/audioset_strong",
    split="train",
    streaming=True,
)

# Take first sample
print("Taking first sample...")
sample = next(iter(ds))

# Check what columns are available
print(f"Available columns: {list(sample.keys())}")

# Check the audio column
if "audio" in sample:
    audio = sample["audio"]
    print(f"Type of audio: {type(audio)}")
    if hasattr(audio, '__dict__'):
        print(f"Audio attributes: {audio.__dict__.keys()}")

# Check the text column
if "text" in sample:
    text = sample["text"]
    print(f"Text description: {text}")

# Try casting the audio column
print("\nTrying to cast audio column...")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Take another sample after casting
sample = next(iter(ds))
audio = sample["audio"]
print(f"After casting - Type of audio: {type(audio)}")
if hasattr(audio, '__dict__'):
    print(f"After casting - Audio attributes: {audio.__dict__.keys()}")

print(f"\nFinal sample text: {sample['text']}")