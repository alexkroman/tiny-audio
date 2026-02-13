#!/usr/bin/env python3
"""Encode audio columns in a HuggingFace dataset to xcodec2 codes and push to Hub.

Usage:
    poetry run python scripts/encode_xcodec2.py \
        --dataset GSQA/spoken-alpaca-gpt4 \
        --audio-columns input_audio output_audio \
        --output-repo user/spoken-alpaca-gpt4-xcodec2 \
        --batch-size 32
"""

import argparse

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from tqdm.auto import tqdm  # noqa: F401 — used by datasets .map() progress


def load_xcodec2(device: str = "cuda"):
    """Load xcodec2 model."""
    from xcodec2.modeling_xcodec2 import XCodec2Model

    return XCodec2Model.from_pretrained("HKUSTAudio/xcodec2").to(device).eval()


def encode_column(dataset, model, column: str, batch_size: int, device: str):
    """Encode a single audio column to xcodec2 codes using .map()."""
    target_sr = 16000
    codes_column = f"{column}_codes"

    def encode_batch(batch):
        waveforms = []
        for audio in batch[column]:
            if audio is None:
                waveforms.append(None)
                continue

            wav = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            # Resample to 16kHz if needed
            if sr != target_sr:
                wav_tensor = torch.from_numpy(wav).unsqueeze(0)
                wav_tensor = torchaudio.functional.resample(wav_tensor, sr, target_sr)
                wav = wav_tensor.squeeze(0).numpy()

            waveforms.append(wav)

        # Encode non-None waveforms
        all_codes = []
        for wav in waveforms:
            if wav is None:
                all_codes.append([])
                continue

            wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
            with torch.no_grad():
                codes = model.encode_code(wav_tensor)
            # codes shape: [batch, codebook, seq_len] → flatten to 1D list
            codes_1d = codes[0, 0].cpu().tolist()
            all_codes.append(codes_1d)

        return {codes_column: all_codes}

    return dataset.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        desc=f"Encoding {column}",
    )


def main():
    parser = argparse.ArgumentParser(description="Encode audio columns to xcodec2 codes")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset ID")
    parser.add_argument("--audio-columns", nargs="+", required=True, help="Audio columns to encode")
    parser.add_argument("--output-repo", required=True, help="Hub repo to push encoded dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default=None, help="Dataset split (default: all)")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    if args.split:
        ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    else:
        ds = load_dataset(args.dataset, trust_remote_code=True)

    print(f"Loading xcodec2 model on {args.device}")
    model = load_xcodec2(args.device)

    # Handle DatasetDict (multiple splits) vs single Dataset
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        for split_name in ds:
            print(f"\nProcessing split: {split_name}")
            for col in args.audio_columns:
                if col in ds[split_name].column_names:
                    ds[split_name] = encode_column(
                        ds[split_name], model, col, args.batch_size, args.device
                    )
                else:
                    print(f"  Warning: column '{col}' not found in split '{split_name}', skipping")

            # Drop original audio columns
            drop_cols = [c for c in args.audio_columns if c in ds[split_name].column_names]
            if drop_cols:
                ds[split_name] = ds[split_name].remove_columns(drop_cols)
    else:
        for col in args.audio_columns:
            if col in ds.column_names:
                ds = encode_column(ds, model, col, args.batch_size, args.device)
            else:
                print(f"  Warning: column '{col}' not found, skipping")

        # Drop original audio columns
        drop_cols = [c for c in args.audio_columns if c in ds.column_names]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)

    print(f"\nPushing to Hub: {args.output_repo}")
    ds.push_to_hub(args.output_repo)
    print("Done!")


if __name__ == "__main__":
    main()
