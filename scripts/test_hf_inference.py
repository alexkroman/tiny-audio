import torch
from datasets import load_dataset
import evaluate

from src.asr_modeling import ASRModel


def main():
    model_id = "mazesmazes/tiny-audio"

    # System prompt override (set to None to use model's default)
    # Try a more direct prompt that matches training
    system_prompt = "/no_think /system_override You are a transcription model. Output only the transcribed text."  # Use model's default (which should match training)

    # Load WER metric
    wer_metric = evaluate.load("wer")

    print(f"Loading model from {model_id}...")

    model = ASRModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float32,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    print("\n" + "=" * 80)
    print("MODEL TOKENIZER INSPECTION")
    print("=" * 80)
    print(f"Tokenizer type: {type(model.tokenizer)}")
    print(f"Vocab size: {len(model.tokenizer)}")
    print(f"BOS token: {model.tokenizer.bos_token!r} (ID: {model.tokenizer.bos_token_id})")
    print(f"EOS token: {model.tokenizer.eos_token!r} (ID: {model.tokenizer.eos_token_id})")
    print(f"PAD token: {model.tokenizer.pad_token!r} (ID: {model.tokenizer.pad_token_id})")
    print(f"Padding side: {model.tokenizer.padding_side}")
    print("\n✅ Chat template enabled")
    if system_prompt:
        print(f"   System prompt (override): {system_prompt}")
    elif model.config.system_prompt:
        print(f"   System prompt (from model): {model.config.system_prompt[:50]}...")
    else:
        print("   No system prompt")

    print("\n" + "=" * 80)
    print("LOADING TEST AUDIO FROM LIBRISPEECH")
    print("=" * 80)

    ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    samples = list(ds.take(10))

    # Track all references and predictions for overall WER
    all_references = []
    all_predictions = []

    for i, sample in enumerate(samples):
        print(f"\n{'=' * 80}")
        print(f"TEST SAMPLE {i + 1}/{len(samples)}")
        print(f"{'=' * 80}")

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference_text = sample["text"]

        print(f"Audio shape: {audio.shape}")
        print(f"Sample rate: {sr}")
        print(f"Reference text: {reference_text}")

        inputs = model.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

        input_values = inputs["input_values"].to(device)
        print(f"\nInput values shape: {input_values.shape}")

        print("\nRunning inference with chat template...")
        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": 120,
                "num_beams": 3,
                "do_sample": False,
                "length_penalty": 0.5,
                "no_repeat_ngram_size": 3,
                "eos_token_id": model.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                "pad_token_id": model.tokenizer.pad_token_id,
                "system_prompt": system_prompt if system_prompt else model.config.system_prompt,
            }

            output_ids = model.generate(input_values, **generate_kwargs)

        transcription = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        transcription = transcription.strip()

        # Normalize text for WER calculation (ignore case and punctuation)
        import string
        def normalize_text(text):
            # Convert to uppercase
            text = text.upper()
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        # Normalize both reference and transcription
        normalized_ref = normalize_text(reference_text)
        normalized_pred = normalize_text(transcription)

        # Calculate WER for this sample
        sample_wer = wer_metric.compute(predictions=[normalized_pred], references=[normalized_ref])

        # Add normalized versions to overall tracking for consistent WER
        all_references.append(normalized_ref)
        all_predictions.append(normalized_pred)

        print(f"\n{'=' * 40}")
        print("RESULTS")
        print(f"{'=' * 40}")
        print(f"Reference:     {reference_text}")
        print(f"Transcription: {transcription}")
        print(f"WER:           {sample_wer:.2%}")
        print(f"\nRaw output IDs (first 50): {output_ids[0][:50].tolist()}")
        print(f"Output IDs shape: {output_ids.shape}")

        transcription_with_special = model.tokenizer.batch_decode(
            output_ids, skip_special_tokens=False
        )[0]
        print(f"\nWith special tokens: {transcription_with_special!r}")

        if not transcription or len(transcription.strip()) == 0:
            print("⚠️  WARNING: Empty transcription!")
        elif transcription.count(" ") < 2:
            print("⚠️  WARNING: Very short transcription!")

        print()

    # Calculate overall WER
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    overall_wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    print(f"Overall WER across {len(all_references)} samples: {overall_wer:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
