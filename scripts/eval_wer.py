#!/usr/bin/env python3
"""
WER Evaluation Script for Loquacious Dataset

Evaluates a trained ASR model on 200 samples from the loquacious dataset
and outputs results to console.

Usage:
    uv run scripts/eval_wer.py --model path/to/model
    uv run scripts/eval_wer.py --model mazesmazes/tiny-audio
    uv run scripts/eval_wer.py --model path/to/model --num-samples 100
"""

import argparse
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm

from modeling import ASRModel


def eval_wer(model_path: str, num_samples: int = 200) -> dict:
    """
    Evaluate WER on loquacious dataset.

    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        num_samples: Number of samples to evaluate (default: 200)

    Returns:
        dict with WER metrics and sample predictions
    """
    print(f"🎙️  Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model
    model = ASRModel.from_pretrained(model_path, low_cpu_mem_usage=False)
    model = model.to(device)
    model.eval()

    print(f"📊 Loading loquacious dataset (first {num_samples} samples)...")
    # Load loquacious dataset
    dataset = load_dataset(
        "speechcolab/loquacious",
        name="clean",
        split="test",
        streaming=True,
    )

    # Take only the requested number of samples
    samples = list(dataset.take(num_samples))

    print(f"🔬 Evaluating on {len(samples)} samples...")
    wer_metric = evaluate.load("wer")

    predictions = []
    references = []

    with torch.no_grad():
        for sample in tqdm(samples, desc="Transcribing"):
            # Extract audio array
            audio_array = sample["audio"]["array"]

            # Process audio with feature extractor
            inputs = model.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
            )

            # Generate transcription
            generated_ids = model.generate(
                input_features=inputs.input_features.to(device),
            )

            # Decode prediction
            prediction = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            reference = sample["text"]

            predictions.append(prediction)
            references.append(reference)

    # Compute WER
    wer = wer_metric.compute(predictions=predictions, references=references)

    # Print results
    print("\n" + "="*80)
    print(f"📈 RESULTS: WER Evaluation on Loquacious Dataset")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Samples evaluated: {len(samples)}")
    print(f"Word Error Rate (WER): {wer:.2%}")
    print("="*80)

    # Show first 5 examples
    print("\n📝 Sample Predictions (first 5):\n")
    for i in range(min(5, len(predictions))):
        print(f"Example {i+1}:")
        print(f"  Reference:  {references[i]}")
        print(f"  Prediction: {predictions[i]}")
        print()

    return {
        "wer": wer,
        "num_samples": len(samples),
        "predictions": predictions,
        "references": references,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WER on loquacious dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples to evaluate (default: 200)",
    )

    args = parser.parse_args()

    eval_wer(args.model, args.num_samples)


if __name__ == "__main__":
    main()
