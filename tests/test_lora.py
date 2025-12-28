"""Tests for LoRA adapter functionality."""

import tempfile
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel


@pytest.fixture
def config():
    """Create a minimal config for testing."""
    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
    )


@pytest.fixture
def model(config):
    """Create an ASRModel instance for testing."""
    return ASRModel(config)


@pytest.fixture
def lora_config():
    """Create a LoRA config for testing."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )


class TestLoRAApplication:
    """Tests for applying LoRA to models."""

    def test_apply_lora_to_language_model(self, model, lora_config):
        """Test that LoRA can be applied to the language model."""
        original_params = sum(p.numel() for p in model.language_model.parameters())

        model.language_model = get_peft_model(model.language_model, lora_config)

        assert isinstance(model.language_model, PeftModel)
        # PEFT model should have more parameters (base + adapters)
        new_params = sum(p.numel() for p in model.language_model.parameters())
        assert new_params > original_params

    def test_lora_trainable_params(self, model, lora_config):
        """Test that only LoRA params are trainable after applying."""
        model.language_model = get_peft_model(model.language_model, lora_config)

        trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.language_model.parameters())

        # Trainable should be small fraction of total
        assert trainable > 0
        assert trainable < total * 0.1  # Less than 10% trainable

    def test_projector_frozen_with_lora(self, model, lora_config):
        """Test that projector can be frozen while LoRA is trainable."""
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.projector.requires_grad_(False)

        # Projector should be frozen
        for param in model.projector.parameters():
            assert not param.requires_grad

        # LoRA should still be trainable
        trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        assert trainable > 0


class TestLoRASaveLoad:
    """Tests for saving and loading LoRA adapters."""

    def test_save_adapter(self, model, lora_config):
        """Test that adapter can be saved."""
        model.language_model = get_peft_model(model.language_model, lora_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "adapter"
            model.language_model.save_pretrained(save_path)

            assert (save_path / "adapter_config.json").exists()
            assert (save_path / "adapter_model.safetensors").exists()

    def test_load_adapter(self, model, lora_config):
        """Test that adapter can be loaded."""
        # Apply and save
        model.language_model = get_peft_model(model.language_model, lora_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "adapter"
            model.language_model.save_pretrained(save_path)

            # Create fresh model and load adapter
            fresh_config = ASRConfig(
                audio_model_id="openai/whisper-tiny",
                text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                projector_type="mlp",
                model_dtype="float32",
                attn_implementation="eager",
            )
            fresh_model = ASRModel(fresh_config)

            # Load adapter
            fresh_model.language_model = PeftModel.from_pretrained(
                fresh_model.language_model, save_path, is_trainable=False
            )

            assert isinstance(fresh_model.language_model, PeftModel)

    def test_adapter_weights_preserved(self, model, lora_config):
        """Test that adapter weights are preserved after save/load."""
        model.language_model = get_peft_model(model.language_model, lora_config)

        # Modify a LoRA weight
        for name, param in model.language_model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                with torch.no_grad():
                    param.fill_(0.42)
                break

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "adapter"
            model.language_model.save_pretrained(save_path)

            # Load into fresh model
            fresh_config = ASRConfig(
                audio_model_id="openai/whisper-tiny",
                text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                projector_type="mlp",
                model_dtype="float32",
                attn_implementation="eager",
            )
            fresh_model = ASRModel(fresh_config)
            fresh_model.language_model = PeftModel.from_pretrained(
                fresh_model.language_model, save_path, is_trainable=False
            )

            # Check weight was preserved
            for name, param in fresh_model.language_model.named_parameters():
                if "lora_A" in name:
                    assert torch.allclose(param, torch.full_like(param, 0.42))
                    break


class TestLoRAConfig:
    """Tests for LoRA configuration options."""

    def test_different_ranks(self, model):
        """Test LoRA with different ranks."""
        for r in [4, 8, 16]:
            config = LoraConfig(
                r=r,
                lora_alpha=r * 2,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            # Create fresh model for each test
            fresh_config = ASRConfig(
                audio_model_id="openai/whisper-tiny",
                text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
                projector_type="mlp",
                model_dtype="float32",
                attn_implementation="eager",
            )
            fresh_model = ASRModel(fresh_config)
            fresh_model.language_model = get_peft_model(fresh_model.language_model, config)

            trainable = sum(
                p.numel() for p in fresh_model.language_model.parameters() if p.requires_grad
            )
            assert trainable > 0

    def test_target_modules_all_linear(self, model, lora_config):
        """Test that all-linear targets all linear layers."""
        model.language_model = get_peft_model(model.language_model, lora_config)

        # Check that LoRA was applied to multiple layers
        lora_layers = [n for n, _ in model.language_model.named_modules() if "lora" in n.lower()]
        assert len(lora_layers) > 0
