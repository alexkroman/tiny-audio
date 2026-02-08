#!/usr/bin/env python3
"""Check S2S label construction for bugs: masking ratio, delay pattern, shapes."""

import torch
from transformers.models.dia.processing_dia import DiaProcessor

DELAY_PATTERN = [0, 8, 9, 10, 11, 12, 13, 14, 15]
PAD_TOKEN = 1025
BOS_TOKEN = 1026
EOS_TOKEN = 1024
NUM_CODEBOOKS = 9
MAX_DELAY = max(DELAY_PATTERN)


def check_labels():
    # Simulate what S2SDataCollator does with realistic data
    batch_size = 2
    code_lengths = [50, 70]  # realistic audio code lengths
    max_audio_len = max(code_lengths)

    # Simulate codes [9, seq_len] -> transpose to [seq_len, 9]
    code_list = []
    for length in code_lengths:
        codes = torch.randint(0, 1024, (length, NUM_CODEBOOKS))  # valid DAC codes 0-1023
        code_list.append(codes)

    # Build [batch, seq_len, 9] with BOS + codes + EOS + PAD for delay
    total_len = max_audio_len + 2 + MAX_DELAY  # +2 for BOS and EOS
    prefill = torch.full((batch_size, total_len, NUM_CODEBOOKS), PAD_TOKEN, dtype=torch.long)
    for i, codes_t in enumerate(code_list):
        seq_len = codes_t.shape[0]
        prefill[i, 0] = BOS_TOKEN
        prefill[i, 1 : 1 + seq_len] = codes_t
        prefill[i, 1 + seq_len] = EOS_TOKEN

    print(f"Prefill shape: {prefill.shape}")
    print(f"Total len: {total_len} = max_audio({max_audio_len}) + 2(BOS/EOS) + {MAX_DELAY}(delay)")
    print()

    # Check prefill content for sample 0
    print("Sample 0 prefill (first 5 timesteps, first 3 codebooks):")
    print(f"  t=0 (should be BOS={BOS_TOKEN}): {prefill[0, 0, :3].tolist()}")
    print(f"  t=1 (should be codes): {prefill[0, 1, :3].tolist()}")
    print(f"  t={code_lengths[0]} (last code): {prefill[0, code_lengths[0], :3].tolist()}")
    print(
        f"  t={code_lengths[0] + 1} (should be EOS={EOS_TOKEN}): {prefill[0, code_lengths[0] + 1, :3].tolist()}"
    )
    print(
        f"  t={code_lengths[0] + 2} (should be PAD={PAD_TOKEN}): {prefill[0, code_lengths[0] + 2, :3].tolist()}"
    )
    print()

    # Apply delay pattern
    precomputed_idx = DiaProcessor.build_indices(
        bsz=batch_size,
        seq_len=total_len,
        num_channels=NUM_CODEBOOKS,
        delay_pattern=DELAY_PATTERN,
        revert=False,
    )
    delayed = DiaProcessor.apply_audio_delay(
        audio=prefill,
        pad_token_id=PAD_TOKEN,
        bos_token_id=BOS_TOKEN,
        precomputed_idx=precomputed_idx,
    )

    print(f"Delayed shape: {delayed.shape}")
    print(f"Delayed sample 0, t=0, all codebooks: {delayed[0, 0].tolist()}")
    print(f"Delayed sample 0, t=1, all codebooks: {delayed[0, 1].tolist()}")
    print()

    # Labels: shift left, mask special tokens
    labels = delayed[:, 1:].clone()
    print(f"Labels shape before masking: {labels.shape}")

    total_tokens = labels.numel()
    pad_count = (labels == PAD_TOKEN).sum().item()
    bos_count = (labels == BOS_TOKEN).sum().item()
    eos_count = (labels == EOS_TOKEN).sum().item()
    labels[labels == PAD_TOKEN] = -100
    labels[labels == BOS_TOKEN] = -100

    masked_count = (labels == -100).sum().item()
    valid_count = total_tokens - masked_count

    print(f"Total tokens: {total_tokens}")
    print(f"PAD tokens (masked): {pad_count} ({pad_count / total_tokens * 100:.1f}%)")
    print(f"BOS tokens (masked): {bos_count} ({bos_count / total_tokens * 100:.1f}%)")
    print(f"EOS tokens (kept as target): {eos_count} ({eos_count / total_tokens * 100:.1f}%)")
    print(f"Valid training tokens: {valid_count} ({valid_count / total_tokens * 100:.1f}%)")
    print(f"Masked tokens (-100): {masked_count} ({masked_count / total_tokens * 100:.1f}%)")
    print()

    # Flatten to dia_labels format
    dia_labels = labels.transpose(1, 2).reshape(batch_size * NUM_CODEBOOKS, -1).contiguous().long()
    print(f"dia_labels shape: {dia_labels.shape}")
    print(f"  Expected: [{batch_size * NUM_CODEBOOKS}, {total_len - 1}]")
    print()

    # Check per-codebook masking (delay pattern should cause different masking per codebook)
    print("Per-codebook valid token counts (sample 0):")
    for cb in range(NUM_CODEBOOKS):
        row = dia_labels[cb]
        valid = (row != -100).sum().item()
        total = row.shape[0]
        print(
            f"  Codebook {cb} (delay={DELAY_PATTERN[cb]}): {valid}/{total} valid ({valid / total * 100:.1f}%)"
        )

    # Check decoder_input_ids
    decoder_input_ids = delayed[:, :-1]
    print(f"\ndecoder_input_ids shape: {decoder_input_ids.shape}")
    print(f"  Expected: [{batch_size}, {total_len - 1}, {NUM_CODEBOOKS}]")

    # Check decoder attention mask
    mask_len = total_len - 1
    decoder_attention_mask = torch.zeros(batch_size, mask_len, dtype=torch.long)
    for i in range(batch_size):
        valid = code_lengths[i] + 2 + MAX_DELAY
        decoder_attention_mask[i, : min(valid, mask_len)] = 1

    print(f"decoder_attention_mask shape: {decoder_attention_mask.shape}")
    for i in range(batch_size):
        valid_positions = decoder_attention_mask[i].sum().item()
        print(f"  Sample {i}: {valid_positions}/{mask_len} valid positions")

    # === KEY CHECK: Does output_hidden_states=True cause issues? ===
    print(f"\n{'=' * 60}")
    print("CRITICAL CHECK: output_hidden_states=True in forward")
    print("=" * 60)
    print("Line 540: kwargs['output_hidden_states'] = True")
    print("This makes the LLM return ALL layer hidden states.")
    print("Line 567: all_hidden_states = outputs.hidden_states[-1]")
    print("Takes only the last layer. But ALL are returned and stored.")
    print()

    # === Check assistant_mask alignment ===
    print(f"{'=' * 60}")
    print("ASSISTANT MASK ALIGNMENT CHECK")
    print("=" * 60)
    print("The assistant_mask selects which LLM hidden states go to the audio head.")
    print("These hidden states are then projected and used as encoder_outputs for Dia.")
    print("The dia_labels have a DIFFERENT sequence length (audio code length).")
    print()
    # Typical assistant response might be 10-30 tokens for a short transcription
    simulated_assistant_tokens = [12, 18]
    for i, n_tokens in enumerate(simulated_assistant_tokens):
        label_seq_len = dia_labels.shape[1]
        valid_labels = (dia_labels[i * NUM_CODEBOOKS] != -100).sum().item()
        print(f"  Sample {i}: {n_tokens} assistant tokens -> encoder seq_len for Dia")
        print(f"    dia_labels seq_len: {label_seq_len}")
        print(f"    valid labels (codebook 0): {valid_labels}")
        print(f"    Ratio (labels/encoder): {label_seq_len / n_tokens:.1f}x")
        print(
            f"    This means Dia decoder must generate {label_seq_len / n_tokens:.1f}x more tokens than encoder inputs"
        )


if __name__ == "__main__":
    check_labels()
