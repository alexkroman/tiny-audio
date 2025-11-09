#!/usr/bin/env python3
"""Simple debug script to check label alignment without loading full model."""

import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B", trust_remote_code=True)

# Add audio token if needed
existing_special = tokenizer.additional_special_tokens or []
if "<audio>" not in existing_special:
    special_tokens = {"additional_special_tokens": existing_special + ["<audio>"]}
    tokenizer.add_special_tokens(special_tokens)

# Create the exact same message structure as training
text = "my name is alex"
instruction = "Transcribe: <audio>"
system_prompt = "You are a helpful voice assistant."

messages = []
messages.append({"role": "system", "content": system_prompt})
messages.append({"role": "user", "content": instruction})
messages.append({"role": "assistant", "content": text})

# Tokenize (same as training)
tokens = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    truncation=True,
    max_length=256,
    enable_thinking=False,
)

# Create labels (same logic as training - only train on assistant content)
labels = [-100] * len(tokens)

# Get special token IDs
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")
audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")

# Find assistant message start
assistant_start_idx = -1
for i in range(len(tokens) - 1):
    if tokens[i] == im_start_id and tokens[i + 1] == assistant_id:
        assistant_start_idx = i
        break

# Find where actual content starts (after </think> if present, else after assistant header)
content_start = -1
think_end_id = tokenizer.convert_tokens_to_ids("</think>")

# Look for </think> ONLY within the assistant message
if assistant_start_idx >= 0:
    for i in range(assistant_start_idx, len(tokens)):
        if tokens[i] == think_end_id:
            # Skip the </think> token and any newlines after it
            content_start = i + 1
            # Skip newlines
            while (
                content_start < len(tokens)
                and tokenizer.decode([tokens[content_start]]).strip() == ""
            ):
                content_start += 1
            break

# If no thinking tags, start right after "assistant\n"
if content_start == -1 and assistant_start_idx >= 0:
    content_start = assistant_start_idx + 3  # Skip <|im_start|>, assistant, \n

# Find the closing <|im_end|> for the assistant message
content_end = -1
if content_start > 0:
    for i in range(content_start, len(tokens)):
        if tokens[i] == im_end_id:
            content_end = i
            break

# Unmask only the actual transcription text (NOT thinking tags)
# Include <|im_end|> so model learns when to stop generating
if content_start > 0 and content_end > 0 and content_start < content_end:
    for i in range(content_start, content_end + 1):  # +1 to include <|im_end|>
        labels[i] = tokens[i]

print("=" * 100)
print("TRAINING LABEL ANALYSIS")
print("=" * 100)

print(f"\nFull tokenized sequence ({len(tokens)} tokens):")
print(tokenizer.decode(tokens))

print(f"\n\nLabeled content (what model is trained to predict):")
labeled_tokens = [t for t, l in zip(tokens, labels) if l != -100]
print(tokenizer.decode(labeled_tokens))

print(f"\n\nTOKEN-BY-TOKEN BREAKDOWN:")
print("-" * 100)
print(f"{'Pos':<5} {'Token ID':<10} {'Token Text':<40} {'Has Label?':<12} {'Notes':<20}")
print("-" * 100)

audio_token_pos = None
for i, (tid, label) in enumerate(zip(tokens, labels)):
    token_text = repr(tokenizer.decode([tid]))
    has_label = "YES ✓" if label != -100 else "NO"
    notes = ""

    if tid == audio_token_id:
        notes = "<AUDIO TOKEN>"
        audio_token_pos = i
    elif tid == im_start_id:
        notes = "<START>"
    elif tid == im_end_id:
        notes = "<END>"
    elif tid == assistant_id:
        notes = "assistant"
    elif i == content_start:
        notes = "← LABELS START"

    print(f"{i:<5} {tid:<10} {token_text:<40} {has_label:<12} {notes:<20}")

print("\n" + "=" * 100)
print("CRITICAL ANALYSIS:")
print("=" * 100)

if audio_token_pos is not None:
    print(f"\n✓ <audio> token found at position {audio_token_pos}")
    print(f"  Content start (first labeled token): {content_start}")
    print(f"  Content end (last labeled token): {content_end}")

    # Check if audio token has a label
    if labels[audio_token_pos] != -100:
        print(f"\n⚠️  PROBLEM: <audio> token itself IS being trained on (label = {labels[audio_token_pos]})")
        print("   The model learns to predict the <audio> token, not use its embeddings!")
    else:
        print(f"\n✓ <audio> token is NOT labeled (good)")

    # Check if tokens before assistant are labeled
    tokens_before_assistant = sum(1 for l in labels[:content_start] if l != -100)
    if tokens_before_assistant > 0:
        print(f"\n⚠️  PROBLEM: {tokens_before_assistant} tokens before assistant response are labeled")
        print("   Model can learn patterns from the prompt without using audio!")
    else:
        print(f"\n✓ No tokens before assistant are labeled (good)")

    # The key question: Can the model predict the transcription without audio?
    print("\n" + "=" * 100)
    print("CAN THE MODEL IGNORE AUDIO?")
    print("=" * 100)
    print("\nSequence the model sees during training:")
    print(f"  1. System prompt: '{system_prompt}'")
    print(f"  2. User: 'Transcribe: <audio>' (token at pos {audio_token_pos})")
    print(f"  3. Assistant: '{text}' ← THESE TOKENS ARE LABELED")
    print("\nDuring forward pass:")
    print(f"  - The <audio> TOKEN (pos {audio_token_pos}) is replaced with audio EMBEDDINGS")
    print(f"  - But the word 'Transcribe:' is still in the sequence as TEXT")
    print(f"  - Model MIGHT learn: 'After Transcribe: → generate plausible text'")
    print(f"  - Model MIGHT NOT use the audio embeddings at all!")

    print("\nPotential fix:")
    print("  1. Mask out more of the prompt from input_ids (replace with padding)")
    print("  2. Add audio embedding dropout to force model to use audio")
    print("  3. Add a verification task (e.g., audio matching) to ensure audio is attended to")
else:
    print("\n❌ ERROR: <audio> token not found in sequence!")

print("\n" + "=" * 100)
