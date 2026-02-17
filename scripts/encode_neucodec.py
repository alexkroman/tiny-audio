#!/usr/bin/env python3
"""Encode audio datasets to neucodec codes and produce TRL-compatible ChatML datasets.

Supports two modes:

1. Raw encoding: Encode audio columns to neucodec code columns (original behavior).
2. ChatML TTS: Encode audio, convert to speech tokens, and produce a single `messages`
   column ready for TRL's SFTTrainer with no extra processing needed.

Usage:
    # Raw encoding (Stage 1/2)
    poetry run python scripts/encode_neucodec.py \
        --dataset GSQA/spoken-alpaca-gpt4 \
        --audio-columns input_audio output_audio \
        --output-repo user/spoken-alpaca-gpt4-neucodec \
        --batch-size 32

    # ChatML TTS with voice cloning from LibriTTS
    # Uses a different utterance from the same speaker as voice reference
    poetry run python scripts/encode_neucodec.py \
        --dataset mythicinfinity/libritts \
        --chatml-tts \
        --speaker-column speaker_id \
        --text-column text_normalized \
        --audio-column audio \
        --output-repo user/libritts-tts-chatml \
        --split train.clean.360 \
        --config clean \
        --batch-size 32

    # ChatML TTS without voice reference (simple text → speech)
    poetry run python scripts/encode_neucodec.py \
        --dataset mythicinfinity/libritts \
        --chatml-tts \
        --text-column text_normalized \
        --output-repo user/libritts-tts-chatml \
        --split train.clean.360 \
        --config clean \
        --batch-size 32

    # Raw encoding + reasoning for Stage 3
    poetry run python scripts/encode_neucodec.py \
        --dataset GSQA/spoken-alpaca-gpt4 \
        --audio-columns input_audio output_audio \
        --output-repo user/spoken-alpaca-gpt4-neucodec \
        --generate-reasoning \
        --reasoning-model Qwen/Qwen3-0.6B \
        --batch-size 32
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from tqdm.auto import tqdm

from tiny_audio.lm import codes_to_speech_text

CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def save_checkpoint(output_dir: str, dataset: Dataset, step: int):
    """Save a processed shard to disk."""
    path = Path(output_dir) / f"{CHECKPOINT_PREFIX}-{step}"
    dataset.save_to_disk(str(path))
    print(f"  Saved checkpoint-{step} ({len(dataset):,} samples)")


def get_last_checkpoint_step(folder: str) -> int:
    """Find the last completed checkpoint step, or 0 if none."""
    folder_path = Path(folder)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        return 0
    content = [p.name for p in folder_path.iterdir()]
    checkpoints = [p for p in content if _RE_CHECKPOINT.match(p)]
    if not checkpoints:
        return 0
    last = max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.match(x).group(1)))
    return int(_RE_CHECKPOINT.match(last).group(1))


def load_all_checkpoints(folder: str) -> Dataset:
    """Load and concatenate all checkpoint shards."""
    paths = sorted(
        Path(folder).glob(f"{CHECKPOINT_PREFIX}-*"), key=lambda p: int(p.name.split("-")[1])
    )
    datasets = [load_from_disk(str(p)) for p in paths]
    print(f"Loaded {len(datasets)} checkpoints ({sum(len(d) for d in datasets):,} total samples)")
    return concatenate_datasets(datasets)


def load_codec(device: str = "cuda"):
    """Load NeuCodec model for audio encoding."""
    from neucodec import NeuCodec

    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


TARGET_SR = 16000


def resample_audio(audio: dict | None) -> np.ndarray | None:
    """Extract and resample a single audio dict to 16kHz numpy array."""
    if audio is None:
        return None
    wav = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if sr != TARGET_SR:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, TARGET_SR)
        wav = wav_tensor.squeeze(0).numpy()
    return wav


def batch_encode(waveforms: list[np.ndarray | None], model, device: str) -> list[list[int]]:
    """Encode a list of waveforms in a single batched GPU call.

    Pads waveforms to equal length, encodes as one batch, then trims
    each sample's codes based on its original length.
    """
    # Separate valid waveforms from Nones
    valid = [(i, w) for i, w in enumerate(waveforms) if w is not None and len(w) > 0]
    results: list[list[int]] = [[] for _ in waveforms]
    if not valid:
        return results

    indices, wavs = zip(*valid)
    lengths = [len(w) for w in wavs]
    max_len = max(lengths)

    # Pad to max length and stack into [batch, 1, waveform_length] for NeuCodec
    padded = torch.zeros(len(wavs), 1, max_len)
    for j, w in enumerate(wavs):
        padded[j, 0, : len(w)] = torch.from_numpy(w)

    with torch.no_grad():
        codes = model.encode_code(padded.to(device))

    # codes shape: [batch, 1, seq_len] or [batch, seq_len]
    total_code_len = codes.shape[-1]

    for j, orig_idx in enumerate(indices):
        # Trim codes proportional to original waveform length
        expected_len = max(1, round(lengths[j] / max_len * total_code_len))
        if codes.dim() == 3:
            sample_codes = codes[j, 0, :expected_len].cpu().tolist()
        else:
            sample_codes = codes[j, :expected_len].cpu().tolist()
        results[orig_idx] = sample_codes

    return results


def encode_column(dataset, model, column: str, batch_size: int, device: str):
    """Encode a single audio column to neucodec codes using .map()."""
    codes_column = f"{column}_codes"

    def encode_batch_fn(batch):
        waveforms = [resample_audio(audio) for audio in batch[column]]
        return {codes_column: batch_encode(waveforms, model, device)}

    return dataset.map(
        encode_batch_fn,
        batched=True,
        batch_size=batch_size,
        desc=f"Encoding {column}",
    )


def encode_chatml_tts(
    dataset,
    model,
    audio_column: str,
    text_column: str,
    batch_size: int,
    device: str,
    both_directions: bool = False,
):
    """Encode audio and produce a single `messages` column for TRL ChatML training.

    Simple mode (no voice reference):
      user=text, assistant=speech

    With both_directions=True, each row produces TWO conversations:
      1. Speak:       user=text, assistant=speech
      2. Transcribe:  user=speech, assistant=text
    """

    def encode_and_format_batch(batch):
        texts = batch[text_column]
        audios = batch[audio_column]

        # Batch resample + encode
        waveforms = [resample_audio(audio) for audio in audios]
        all_codes = batch_encode(waveforms, model, device)

        all_messages = []
        for text, codes_1d in zip(texts, all_codes):
            if text is None or not codes_1d:
                continue

            text = text.strip()
            if not text:
                continue

            speech_text = codes_to_speech_text(codes_1d)

            # Speak direction: text → speech
            all_messages.append(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": speech_text},
                ]
            )

            if both_directions:
                # Transcribe direction: speech → text
                all_messages.append(
                    [
                        {"role": "user", "content": speech_text},
                        {"role": "assistant", "content": text},
                    ]
                )

        return {"messages": all_messages}

    return dataset.map(
        encode_and_format_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        desc="Encoding audio → ChatML messages",
    )


def encode_chatml_voice_clone(
    dataset,
    model,
    audio_column: str,
    text_column: str,
    speaker_column: str,
    batch_size: int,
    device: str,
    max_ref_codes: int = 500,
):
    """Encode audio and produce ChatML messages with voice reference from same speaker.

    For each sample, picks a different utterance from the same speaker as the voice
    reference. The user message contains the reference audio + target text, and the
    assistant message contains the target speech.

    Format:
      user:      <|audio_start|>...<|audio_end|> {text}
      assistant: <|audio_start|>...<|audio_end|>

    Args:
        max_ref_codes: Truncate reference audio to this many codes (~10s at 50 codes/sec).
    """
    # Step 1: Encode all audio to codes
    print("Step 1/2: Encoding all audio to neucodec codes...")
    dataset = encode_column(dataset, model, audio_column, batch_size, device)
    codes_col = f"{audio_column}_codes"

    # Step 2: Build speaker → indices mapping
    print("Step 2/2: Pairing samples by speaker for voice cloning...")
    speakers = dataset[speaker_column]
    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speakers):
        speaker_to_indices[spk].append(i)

    # Filter to speakers with at least 2 samples
    valid_speakers = {spk for spk, indices in speaker_to_indices.items() if len(indices) >= 2}
    print(
        f"  {len(valid_speakers)} speakers with 2+ samples "
        f"(skipping {len(speaker_to_indices) - len(valid_speakers)} single-sample speakers)"
    )

    # Step 3: Build paired messages
    all_messages = []
    texts = dataset[text_column]
    all_codes = dataset[codes_col]
    skipped = 0

    for i in tqdm(range(len(dataset)), desc="Pairing voice clone samples"):
        spk = speakers[i]
        if spk not in valid_speakers:
            skipped += 1
            continue

        text = texts[i]
        target_codes = all_codes[i]

        if not text or not text.strip() or not target_codes:
            skipped += 1
            continue

        text = text.strip()

        # Pick a random different sample from the same speaker as voice reference
        candidates = speaker_to_indices[spk]
        ref_idx = i
        while ref_idx == i:
            ref_idx = random.choice(candidates)

        ref_codes = all_codes[ref_idx]
        if not ref_codes:
            skipped += 1
            continue

        # Truncate reference to max_ref_codes
        if len(ref_codes) > max_ref_codes:
            ref_codes = ref_codes[:max_ref_codes]

        ref_speech = codes_to_speech_text(ref_codes)
        target_speech = codes_to_speech_text(target_codes)

        all_messages.append(
            [
                {"role": "user", "content": f"{ref_speech} {text}"},
                {"role": "assistant", "content": target_speech},
            ]
        )

    print(f"  Created {len(all_messages):,} voice clone samples ({skipped:,} skipped)")
    return Dataset.from_dict({"messages": all_messages})


REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given an instruction and its response, "
    "write a brief internal reasoning (1-3 sentences) that explains the thought "
    "process for producing the response. Be concise and focus on the key logic. "
    "Output only the reasoning, nothing else."
)


def generate_reasoning(dataset, model_id: str, batch_size: int, device: str):
    """Generate reasoning text for each (instruction, input, output) triple."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading reasoning model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    def reasoning_batch(batch):
        instructions = batch.get("instruction", [""] * len(batch["output"]))
        inputs = batch.get("input", [""] * len(batch["output"]))
        outputs = batch["output"]

        prompts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            user_msg = f"Instruction: {inst}"
            if inp:
                user_msg += f"\nInput: {inp}"
            user_msg += f"\nResponse: {out}"

            messages = [
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        # Batch tokenize with left-padding for generation
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
            )

        # Decode only the new tokens
        reasoning_texts = []
        for out_ids in output_ids:
            new_ids = out_ids[encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            reasoning_texts.append(text)

        return {"reasoning": reasoning_texts}

    return dataset.map(
        reasoning_batch,
        batched=True,
        batch_size=batch_size,
        desc="Generating reasoning",
    )


def process_split(ds, args, codec_model):
    """Process a single dataset split: encode audio columns and optionally generate reasoning."""
    for col in args.audio_columns:
        if col in ds.column_names:
            ds = encode_column(ds, codec_model, col, args.batch_size, args.device)
        else:
            print(f"  Warning: column '{col}' not found, skipping")

    # Generate reasoning if requested
    if args.generate_reasoning:
        ds = generate_reasoning(
            ds,
            model_id=args.reasoning_model,
            batch_size=args.reasoning_batch_size,
            device=args.device,
        )

    # Drop original audio columns
    drop_cols = [c for c in args.audio_columns if c in ds.column_names]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    return ds


def process_chatml_split(ds, args, codec_model):
    """Process a single split in ChatML TTS mode."""
    if args.speaker_column:
        return encode_chatml_voice_clone(
            ds,
            codec_model,
            audio_column=args.audio_column,
            text_column=args.text_column,
            speaker_column=args.speaker_column,
            batch_size=args.batch_size,
            device=args.device,
            max_ref_codes=args.max_ref_codes,
        )
    return encode_chatml_tts(
        ds,
        codec_model,
        audio_column=args.audio_column,
        text_column=args.text_column,
        batch_size=args.batch_size,
        device=args.device,
        both_directions=args.chatml_both_directions,
    )


def process_sharded(ds, args, codec_model, checkpoint_dir: str):
    """Process dataset in shards with local checkpointing for resumability."""
    save_every = args.save_every
    total = len(ds)
    num_shards = (total + save_every - 1) // save_every
    process_fn = process_chatml_split if args.chatml_tts else process_split

    # Resume from last checkpoint
    last_step = get_last_checkpoint_step(checkpoint_dir)
    start_shard = (last_step // save_every) + 1 if last_step > 0 else 0
    if start_shard > 0:
        print(
            f"Resuming from shard {start_shard}/{num_shards} (last checkpoint at step {last_step})"
        )

    for shard_idx in range(start_shard, num_shards):
        start = shard_idx * save_every
        end = min(start + save_every, total)
        print(f"\nShard {shard_idx + 1}/{num_shards}: samples {start:,}-{end:,}")

        shard = ds.select(range(start, end))
        processed = process_fn(shard, args, codec_model)
        save_checkpoint(checkpoint_dir, processed, end)

    # Concatenate all checkpoints and push
    print("\nAll shards processed. Assembling final dataset...")
    final = load_all_checkpoints(checkpoint_dir)
    print(f"Pushing {len(final):,} samples to Hub: {args.output_repo}")
    final.push_to_hub(args.output_repo)


def main():
    parser = argparse.ArgumentParser(description="Encode audio columns to neucodec codes")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset ID")
    parser.add_argument("--output-repo", required=True, help="Hub repo to push encoded dataset")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Encoding batch size (default: 128, tuned for A40 48GB)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default=None, help="Dataset split (default: all)")
    parser.add_argument(
        "--config", default=None, help="Dataset config/subset name (e.g. 'clean', 'all')"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to process (for testing)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50000,
        help="Save a checkpoint every N samples for resumability (default: 50000)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/workspace/neucodec_checkpoints",
        help="Directory for local checkpoints (default: /workspace/neucodec_checkpoints)",
    )

    # ChatML TTS mode
    chatml_group = parser.add_argument_group("ChatML TTS mode")
    chatml_group.add_argument(
        "--chatml-tts",
        action="store_true",
        help="Produce a single `messages` column for direct TRL SFTTrainer use",
    )
    chatml_group.add_argument(
        "--audio-column",
        default="audio",
        help="Audio column name (default: audio)",
    )
    chatml_group.add_argument(
        "--text-column",
        default="text_normalized",
        help="Text column name (default: text_normalized)",
    )
    chatml_group.add_argument(
        "--speaker-column",
        default=None,
        help="Speaker ID column for voice cloning (e.g. speaker_id). "
        "When set, pairs samples by speaker: voice ref + text → speech",
    )
    chatml_group.add_argument(
        "--max-ref-codes",
        type=int,
        default=500,
        help="Max neucodec codes for voice reference (~50 codes/sec, default: 500 = ~10s)",
    )
    chatml_group.add_argument(
        "--chatml-both-directions",
        action="store_true",
        help="Include both speak (text→speech) and transcribe (speech→text) directions "
        "(only used without --speaker-column)",
    )

    # Raw encoding mode
    raw_group = parser.add_argument_group("Raw encoding mode")
    raw_group.add_argument("--audio-columns", nargs="+", help="Audio columns to encode")

    # Reasoning generation for Stage 3
    reasoning_group = parser.add_argument_group("Reasoning generation (Stage 3)")
    reasoning_group.add_argument(
        "--generate-reasoning",
        action="store_true",
        help="Generate reasoning text for Stage 3 chain-of-modality training",
    )
    reasoning_group.add_argument(
        "--reasoning-model",
        default="Qwen/Qwen3-0.6B",
        help="Model to generate reasoning (default: Qwen/Qwen3-0.6B)",
    )
    reasoning_group.add_argument(
        "--reasoning-batch-size",
        type=int,
        default=8,
        help="Batch size for reasoning generation (default: 8)",
    )
    args = parser.parse_args()

    # Validate args
    if not args.chatml_tts and not args.audio_columns:
        parser.error("Either --chatml-tts or --audio-columns is required")

    print(f"Loading dataset: {args.dataset}")
    load_kwargs = {"trust_remote_code": True}
    if args.config:
        load_kwargs["name"] = args.config
    if args.split:
        load_kwargs["split"] = args.split
    ds = load_dataset(args.dataset, **load_kwargs)

    # Truncate dataset for testing
    if args.max_samples:
        if isinstance(ds, DatasetDict):
            for split_name in ds:
                n = min(len(ds[split_name]), args.max_samples)
                ds[split_name] = ds[split_name].select(range(n))
                print(f"  Truncated {split_name} to {n} samples")
        else:
            n = min(len(ds), args.max_samples)
            ds = ds.select(range(n))
            print(f"  Truncated to {n} samples")

    print(f"Loading neucodec model on {args.device}")
    codec_model = load_codec(args.device)

    if isinstance(ds, DatasetDict):
        for split_name in ds:
            print(f"\nProcessing split: {split_name}")
            split_ckpt_dir = str(Path(args.checkpoint_dir) / split_name)
            process_sharded(ds[split_name], args, codec_model, split_ckpt_dir)
    else:
        process_sharded(ds, args, codec_model, args.checkpoint_dir)

    print("Done!")


if __name__ == "__main__":
    main()
