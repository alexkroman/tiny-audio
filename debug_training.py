#!/usr/bin/env python3
"""Debug script to verify training data collation and label alignment."""

import torch
from transformers import AutoTokenizer
from src.train import DataCollator
from src.asr_modeling import ASRModel
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs/hydra", config_name="config")
def debug_collation(cfg: DictConfig):
    """Debug the data collation process to verify labels are correctly aligned."""

    # Load model to get tokenizer and feature extractor
    from src.asr_config import ASRConfig
    from transformers import AutoConfig as HFAutoConfig

    encoder_config = HFAutoConfig.from_pretrained(cfg.model.encoder_model_name)
    decoder_config = HFAutoConfig.from_pretrained(
        cfg.model.decoder_model_name, trust_remote_code=True
    )

    asr_config = ASRConfig(
        text_model_id=cfg.model.decoder_model_name,
        audio_model_id=cfg.model.encoder_model_name,
        attn_implementation=cfg.training.attn_implementation,
        model_dtype=cfg.training.model_dtype,
        audio_downsample_rate=cfg.model.audio_downsample_rate,
        system_prompt=cfg.model.system_prompt,
        encoder_dim=encoder_config.hidden_size,
        llm_dim=decoder_config.hidden_size,
        projector_hidden_dim=cfg.model.get("projector_hidden_dim", 2048),
    )

    model = ASRModel(asr_config)
    tokenizer = model.tokenizer
    feature_extractor = model.feature_extractor

    # Create a fake audio sample
    import numpy as np
    fake_audio_array = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

    # Create sample data
    fake_sample = {
        "audio": type('obj', (object,), {
            'get_all_samples': lambda: type('obj', (object,), {
                'data': torch.tensor(fake_audio_array).unsqueeze(0)
            })()
        })(),
        "text": "my name is alex",
        "task": "transcribe"
    }

    # Create collator
    collator = DataCollator(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        sample_rate=16000,
        system_prompt=cfg.model.system_prompt,
        apply_augmentation=False,
    )

    # Process the sample
    batch = collator([fake_sample])

    print("=" * 80)
    print("DEBUGGING TRAINING DATA COLLATION")
    print("=" * 80)

    # Decode input_ids
    print("\n1. INPUT IDS (full sequence):")
    print("-" * 80)
    input_ids = batch["input_ids"][0]
    decoded_full = tokenizer.decode(input_ids)
    print(f"Full decoded: {decoded_full}")
    print(f"Token IDs: {input_ids.tolist()}")

    # Find audio token
    audio_token_id = tokenizer.convert_tokens_to_ids("<audio>")
    audio_pos = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
    if len(audio_pos) > 0:
        print(f"\n<audio> token found at position: {audio_pos[0].item()}")
    else:
        print("\nWARNING: <audio> token NOT found!")

    # Decode labels
    print("\n2. LABELS (what the model is trained to predict):")
    print("-" * 80)
    labels = batch["labels"][0]

    # Show which tokens have labels (not -100)
    unmasked_positions = (labels != -100).nonzero(as_tuple=True)[0]
    print(f"Unmasked positions: {unmasked_positions.tolist()}")

    # Decode only the unmasked tokens
    unmasked_tokens = labels[unmasked_positions]
    decoded_labels = tokenizer.decode(unmasked_tokens)
    print(f"Decoded labels (what model trains on): {decoded_labels}")

    # Show token-by-token breakdown
    print("\n3. TOKEN-BY-TOKEN BREAKDOWN:")
    print("-" * 80)
    print(f"{'Pos':<5} {'Token ID':<10} {'Token':<30} {'Label':<10} {'Trained?':<10}")
    print("-" * 80)
    for i, (tid, lid) in enumerate(zip(input_ids.tolist(), labels.tolist())):
        token_str = tokenizer.decode([tid])
        label_str = str(lid) if lid == -100 else tokenizer.decode([lid])
        trained = "YES" if lid != -100 else "NO"
        print(f"{i:<5} {tid:<10} {repr(token_str):<30} {label_str:<10} {trained:<10}")

    # Check if prompt tokens are being trained on
    print("\n4. DIAGNOSIS:")
    print("-" * 80)

    # Find assistant marker
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")

    assistant_start = None
    for i in range(len(input_ids) - 1):
        if input_ids[i] == im_start_id and input_ids[i + 1] == assistant_id:
            assistant_start = i
            break

    if assistant_start is not None:
        print(f"Assistant message starts at position: {assistant_start}")

        # Check if any tokens before assistant are being trained on
        labels_before_assistant = labels[:assistant_start]
        if (labels_before_assistant != -100).any():
            print("⚠️  WARNING: Some tokens BEFORE assistant response are being trained!")
            print("   This means the model learns patterns from the prompt, not audio.")
        else:
            print("✓ Good: Only assistant response tokens are being trained.")

    # Check audio embedding info
    if "input_features" in batch:
        print(f"\nAudio features shape: {batch['input_features'].shape}")
    elif "input_values" in batch:
        print(f"\nAudio values shape: {batch['input_values'].shape}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    debug_collation()
