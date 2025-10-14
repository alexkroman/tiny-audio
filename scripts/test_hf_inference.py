import torch
import numpy as np
from datasets import load_dataset
from src.asr_modeling import ASRModel

def main():
    model_id = "mazesmazes/tiny-audio"
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

    print("\n" + "="*80)
    print("MODEL TOKENIZER INSPECTION")
    print("="*80)
    print(f"Tokenizer type: {type(model.tokenizer)}")
    print(f"Vocab size: {len(model.tokenizer)}")
    print(f"BOS token: {model.tokenizer.bos_token!r} (ID: {model.tokenizer.bos_token_id})")
    print(f"EOS token: {model.tokenizer.eos_token!r} (ID: {model.tokenizer.eos_token_id})")
    print(f"PAD token: {model.tokenizer.pad_token!r} (ID: {model.tokenizer.pad_token_id})")
    print(f"Padding side: {model.tokenizer.padding_side}")
    print(f"\n✅ Chat template enabled")
    print(f"   System prompt: {model.config.system_prompt[:50]}..." if model.config.system_prompt else "   No system prompt")

    print("\n" + "="*80)
    print("LOADING TEST AUDIO FROM LIBRISPEECH")
    print("="*80)

    ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    samples = list(ds.take(2))

    for i, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"TEST SAMPLE {i+1}/3")
        print(f"{'='*80}")

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference_text = sample["text"]

        print(f"Audio shape: {audio.shape}")
        print(f"Sample rate: {sr}")
        print(f"Reference text: {reference_text}")

        inputs = model.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].to(device)
        print(f"\nInput values shape: {input_values.shape}")

        print(f"\nRunning inference with chat template...")
        with torch.inference_mode():
            generate_kwargs = {
                "max_new_tokens": 200,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.1,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "eos_token_id": model.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                "pad_token_id": model.tokenizer.pad_token_id,
                "early_stopping": True,
                "system_prompt": model.config.system_prompt,
            }

            output_ids = model.generate(input_values, **generate_kwargs)

        transcription = model.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        print(f"\n{'='*40}")
        print("RESULTS")
        print(f"{'='*40}")
        print(f"Reference:     {reference_text}")
        print(f"Transcription: {transcription}")
        print(f"\nRaw output IDs (first 50): {output_ids[0][:50].tolist()}")
        print(f"Output IDs shape: {output_ids.shape}")

        transcription_with_special = model.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False
        )[0]
        print(f"\nWith special tokens: {transcription_with_special!r}")

        if not transcription or len(transcription.strip()) == 0:
            print("⚠️  WARNING: Empty transcription!")
        elif transcription.count(" ") < 2:
            print("⚠️  WARNING: Very short transcription!")

        print()

if __name__ == "__main__":
    main()
