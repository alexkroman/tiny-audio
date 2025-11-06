#!/usr/bin/env python3
"""
Convert old checkpoint format to new separated format.

Old format:
- model.safetensors: Contains both projector and LoRA weights

New format:
- projector.safetensors: Contains only projector weights
- adapter_model.safetensors: Contains only LoRA weights (if present)
- adapter_config.json: PEFT adapter configuration (if LoRA weights present)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_checkpoint(checkpoint_dir: str, output_dir: str = None, force: bool = False):
    """
    Convert checkpoint from old format to new format.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        output_dir: Output directory (defaults to in-place conversion)
        force: Force overwrite if output files exist
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_path}")

    # Use same directory for output if not specified
    if output_dir is None:
        output_path = checkpoint_path
        print(f"Converting checkpoint in-place: {checkpoint_path}")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Converting checkpoint: {checkpoint_path} -> {output_path}")

        # Copy all non-model files to output directory
        if checkpoint_path != output_path:
            for file in checkpoint_path.glob("*"):
                if file.name not in [
                    "model.safetensors",
                    "pytorch_model.bin",
                    "projector.safetensors",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                ]:
                    dest = output_path / file.name
                    if file.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(file, dest)
                    else:
                        shutil.copy2(file, dest)

    # Check if already in new format
    new_projector_path = output_path / "projector.safetensors"
    if new_projector_path.exists() and not force:
        print("Checkpoint already in new format. Use --force to overwrite.")
        return

    # Load the old model file
    old_model_path = checkpoint_path / "model.safetensors"
    if not old_model_path.exists():
        old_model_path = checkpoint_path / "pytorch_model.bin"
        if not old_model_path.exists():
            print(f"No model.safetensors or pytorch_model.bin found in {checkpoint_path}")
            return

    print(f"Loading weights from {old_model_path.name}...")

    if old_model_path.suffix == ".safetensors":
        weights = load_file(old_model_path)
    else:
        weights = torch.load(old_model_path, map_location="cpu")
        if "state_dict" in weights:
            weights = weights["state_dict"]

    # Separate weights
    projector_weights = {}
    lora_weights = {}
    other_weights = {}

    for key, value in weights.items():
        if key.startswith("projector."):
            projector_weights[key] = value
        elif "lora_" in key and key.startswith("decoder."):
            lora_weights[key] = value
        else:
            other_weights[key] = value

    # Save projector weights
    if projector_weights:
        print(f"Saving {len(projector_weights)} projector weights to projector.safetensors")
        save_file(projector_weights, output_path / "projector.safetensors")
    else:
        print("No projector weights found")

    # Save LoRA weights in PEFT format
    if lora_weights:
        print(f"Saving {len(lora_weights)} LoRA weights to adapter_model.safetensors")

        # Convert to PEFT format (remove "decoder." prefix)
        peft_weights = {}
        for key, value in lora_weights.items():
            new_key = key.replace("decoder.", "")
            peft_weights[new_key] = value

        save_file(peft_weights, output_path / "adapter_model.safetensors")

        # Create adapter_config.json if it doesn't exist
        adapter_config_path = output_path / "adapter_config.json"
        if not adapter_config_path.exists():
            # Try to infer LoRA config from weight shapes
            r = None
            lora_alpha = 32  # Default value
            target_modules = set()

            for key in peft_weights:
                if "lora_A" in key:
                    # Extract r from lora_A weight shape
                    if r is None:
                        r = peft_weights[key].shape[0]

                    # Extract target module name
                    # Example: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                    parts = key.split(".")
                    for i, part in enumerate(parts):
                        if part in [
                            "q_proj",
                            "v_proj",
                            "k_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ]:
                            target_modules.add(part)
                            break

            # Check for existing peft_config.json for additional settings
            peft_config_path = checkpoint_path / "peft_config.json"
            if peft_config_path.exists():
                with open(peft_config_path) as f:
                    peft_config = json.load(f)
                    r = peft_config.get("r", r or 8)
                    lora_alpha = peft_config.get("lora_alpha", lora_alpha)
                    if peft_config.get("target_modules"):
                        target_modules = set(peft_config["target_modules"])

            adapter_config = {
                "r": r or 8,
                "lora_alpha": lora_alpha,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": (
                    sorted(list(target_modules)) if target_modules else ["q_proj", "v_proj"]
                ),
                "peft_type": "LORA",
                "base_model_name_or_path": "HuggingFaceTB/SmolLM3-3B",  # You may need to update this
            }

            print(
                f"Creating adapter_config.json with r={adapter_config['r']}, target_modules={adapter_config['target_modules']}"
            )
            with open(adapter_config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)
    else:
        print("No LoRA weights found")

    # Report any other weights
    if other_weights:
        print(f"\nWarning: Found {len(other_weights)} weights that are neither projector nor LoRA:")
        for key in list(other_weights.keys())[:10]:  # Show first 10
            print(f"  - {key}")
        if len(other_weights) > 10:
            print(f"  ... and {len(other_weights) - 10} more")

    # Remove old model file if converting in-place
    if output_path == checkpoint_path and (projector_weights or lora_weights):
        old_model_output = output_path / old_model_path.name
        if old_model_output.exists():
            backup_path = output_path / f"{old_model_path.name}.backup"
            print(f"\nBacking up old model file to {backup_path.name}")
            shutil.move(old_model_output, backup_path)

    print("\nConversion complete!")
    print(f"New checkpoint structure in {output_path}:")
    if projector_weights:
        print(f"  - projector.safetensors ({len(projector_weights)} weights)")
    if lora_weights:
        print(f"  - adapter_model.safetensors ({len(lora_weights)} weights)")
        print("  - adapter_config.json")


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to new separated format")
    parser.add_argument("checkpoint_dir", type=str, help="Path to checkpoint directory to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to in-place conversion)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite if output files already exist"
    )

    args = parser.parse_args()

    try:
        convert_checkpoint(args.checkpoint_dir, args.output_dir, args.force)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
