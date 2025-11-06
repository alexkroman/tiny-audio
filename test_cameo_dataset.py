#!/usr/bin/env python3
"""Test CAMEO dataset loading."""

from datasets import load_dataset, Audio

# Load the CAMEO dataset - crema_d split (lowercase with underscore)
print("Loading CAMEO crema_d dataset...")
ds = load_dataset(
    "amu-cai/CAMEO",
    split="crema_d[:10]",  # crema_d is the correct split name
    streaming=False,  # Not streaming to see structure
)

# Check first sample
print("Taking first sample...")
sample = ds[0]

# Check what columns are available
print(f"Available columns: {list(sample.keys())}")

# Check the audio column
if "audio" in sample:
    audio = sample["audio"]
    print(f"Type of audio: {type(audio)}")
    if isinstance(audio, dict):
        print(f"Audio keys: {audio.keys()}")
        if "array" in audio:
            print(f"Audio array shape: {audio['array'].shape if hasattr(audio['array'], 'shape') else len(audio['array'])}")
        if "sampling_rate" in audio:
            print(f"Sampling rate: {audio['sampling_rate']}")

# Check the emotion column
if "emotion" in sample:
    emotion = sample["emotion"]
    print(f"Emotion label: {emotion}")

# Get all unique emotions in this subset
emotions = set(ds["emotion"])
print(f"\nUnique emotions in subset: {emotions}")

# Try casting the audio column
print("\nTrying to cast audio column...")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Check after casting
sample = ds[0]
audio = sample["audio"]
print(f"After casting - Type of audio: {type(audio)}")
if isinstance(audio, dict):
    print(f"After casting - Audio keys: {audio.keys()}")