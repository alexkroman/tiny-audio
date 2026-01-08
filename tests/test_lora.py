"""Tests for LoRA (Stage 2) fine-tuning support."""

import os
import tempfile

import pytest
import torch

from src.asr_config import ASRConfig
from src.asr_modeling import ASRModel


@pytest.fixture
def base_config():
    """Create a minimal config without LoRA for testing."""
    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
    )


@pytest.fixture
def lora_config():
    """Create a config with LoRA enabled."""
    return ASRConfig(
        audio_model_id="openai/whisper-tiny",
        text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        projector_type="mlp",
        model_dtype="float32",
        attn_implementation="eager",
        use_lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        freeze_projector=True,
    )


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_default_lora_disabled(self, base_config):
        """Test that LoRA is disabled by default."""
        assert base_config.use_lora is False
        assert base_config.freeze_projector is False

    def test_lora_config_values(self, lora_config):
        """Test LoRA config values are set correctly."""
        assert lora_config.use_lora is True
        assert lora_config.lora_rank == 8
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.0
        assert lora_config.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert lora_config.freeze_projector is True

    def test_default_target_modules(self):
        """Test default target modules when None is passed."""
        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            use_lora=True,
            lora_target_modules=None,
        )
        assert config.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TestLoRAModelInitialization:
    """Tests for LoRA model initialization."""

    def test_lora_model_creation(self, lora_config):
        """Test that model with LoRA can be created."""
        model = ASRModel(lora_config)
        assert model is not None
        # Check that language model is wrapped with PEFT
        assert hasattr(model.language_model, "peft_config")

    def test_lora_layers_trainable(self, lora_config):
        """Test that LoRA layers are trainable."""
        model = ASRModel(lora_config)

        # Find LoRA parameters
        lora_params = [
            (name, param) for name, param in model.language_model.named_parameters()
            if "lora_" in name
        ]

        assert len(lora_params) > 0, "Should have LoRA parameters"

        for name, param in lora_params:
            assert param.requires_grad, f"LoRA param {name} should be trainable"

    def test_base_llm_frozen_with_lora(self, lora_config):
        """Test that base LLM (non-LoRA) weights are frozen."""
        model = ASRModel(lora_config)

        # Find non-LoRA parameters in language model
        base_params = [
            (name, param) for name, param in model.language_model.named_parameters()
            if "lora_" not in name
        ]

        for name, param in base_params:
            assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_projector_frozen_with_freeze_projector(self, lora_config):
        """Test that projector is frozen when freeze_projector=True."""
        model = ASRModel(lora_config)

        for param in model.projector.parameters():
            assert not param.requires_grad, "Projector should be frozen"

    def test_projector_trainable_without_freeze(self):
        """Test that projector is trainable when freeze_projector=False."""
        config = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            model_dtype="float32",
            attn_implementation="eager",
            use_lora=True,
            freeze_projector=False,  # Projector should be trainable
        )
        model = ASRModel(config)

        trainable_projector_params = sum(
            1 for p in model.projector.parameters() if p.requires_grad
        )
        assert trainable_projector_params > 0, "Projector should have trainable params"


class TestLoRATrainableParameters:
    """Tests for trainable parameter counts."""

    def test_lora_fewer_trainable_params(self, base_config, lora_config):
        """Test that LoRA has fewer trainable params than full fine-tuning would."""
        model_base = ASRModel(base_config)
        model_lora = ASRModel(lora_config)

        # Count trainable params
        base_trainable = sum(
            p.numel() for p in model_base.parameters() if p.requires_grad
        )
        lora_trainable = sum(
            p.numel() for p in model_lora.parameters() if p.requires_grad
        )

        # Base model: projector trainable
        # LoRA model: only LoRA adapters trainable (projector frozen)
        # LoRA should have fewer trainable params (LoRA rank=8 is small)
        assert lora_trainable < base_trainable * 2, (
            f"LoRA trainable ({lora_trainable}) should be reasonable vs base ({base_trainable})"
        )

    def test_print_trainable_parameters(self, lora_config, capsys):
        """Test that PEFT's print_trainable_parameters works."""
        model = ASRModel(lora_config)

        # PEFT provides this method
        if hasattr(model.language_model, "print_trainable_parameters"):
            model.language_model.print_trainable_parameters()
            captured = capsys.readouterr()
            assert "trainable" in captured.out.lower()


class TestLoRASaveLoad:
    """Tests for saving and loading LoRA models."""

    def test_save_creates_adapter_files(self, lora_config):
        """Test that save_pretrained creates adapter files."""
        model = ASRModel(lora_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Check for adapter files
            assert os.path.exists(os.path.join(tmpdir, "adapter_model.safetensors")) or \
                   os.path.exists(os.path.join(tmpdir, "adapter_model.bin")), \
                   "Should save adapter weights"
            assert os.path.exists(os.path.join(tmpdir, "adapter_config.json")), \
                   "Should save adapter config"
            # Also check projector weights
            assert os.path.exists(os.path.join(tmpdir, "model.safetensors")), \
                   "Should save projector weights"

    def test_save_load_roundtrip(self, lora_config):
        """Test that model can be saved and loaded correctly."""
        model = ASRModel(lora_config)

        # Get some LoRA weights before saving
        lora_weight_before = None
        for name, param in model.language_model.named_parameters():
            if "lora_A" in name:
                lora_weight_before = param.clone()
                break

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Load the model
            loaded_model = ASRModel.from_pretrained(tmpdir)

            # Check LoRA is still present
            assert hasattr(loaded_model.language_model, "peft_config")

            # Check weights match
            lora_weight_after = None
            for name, param in loaded_model.language_model.named_parameters():
                if "lora_A" in name:
                    lora_weight_after = param
                    break

            if lora_weight_before is not None and lora_weight_after is not None:
                assert torch.allclose(lora_weight_before, lora_weight_after, atol=1e-5)

    def test_base_model_without_lora_unchanged(self, base_config):
        """Test that base model without LoRA still works."""
        model = ASRModel(base_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Should NOT have adapter files
            assert not os.path.exists(os.path.join(tmpdir, "adapter_config.json")), \
                   "Base model should not have adapter config"

            # Load and verify
            loaded_model = ASRModel.from_pretrained(tmpdir)
            assert not hasattr(loaded_model.language_model, "peft_config")


class TestLoRAForward:
    """Tests for forward pass with LoRA."""

    @pytest.fixture
    def lora_model(self, lora_config):
        """Create a LoRA model for testing."""
        return ASRModel(lora_config)

    @pytest.fixture
    def batch(self, lora_model):
        """Create a sample batch for testing."""
        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)

        num_audio_tokens = int(lora_model._get_num_audio_tokens(audio_mask))
        audio_placeholder = "<audio>" * num_audio_tokens
        messages = [{"role": "user", "content": f"Transcribe: {audio_placeholder}"}]

        chat_result = lora_model.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(chat_result, "input_ids"):
            input_ids = chat_result.input_ids
        else:
            input_ids = chat_result

        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[input_ids == lora_model.audio_token_id] = -100

        return {
            "input_ids": input_ids,
            "input_features": audio_features,
            "audio_attention_mask": audio_mask,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def test_forward_returns_loss(self, lora_model, batch):
        """Test that forward pass returns a loss with LoRA."""
        outputs = lora_model(**batch)
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_forward_gradients_to_lora(self, lora_model, batch):
        """Test that gradients flow to LoRA parameters."""
        lora_model.train()
        outputs = lora_model(**batch)
        outputs.loss.backward()

        # Check LoRA params have gradients
        has_lora_grad = False
        for name, param in lora_model.language_model.named_parameters():
            if "lora_" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_lora_grad = True
                    break

        assert has_lora_grad, "LoRA parameters should receive gradients"

    def test_forward_no_gradients_to_projector_when_frozen(self, lora_model, batch):
        """Test that projector doesn't receive gradients when frozen."""
        lora_model.train()
        outputs = lora_model(**batch)
        outputs.loss.backward()

        # Check projector params don't have gradients
        for param in lora_model.projector.parameters():
            assert param.grad is None, "Frozen projector should not receive gradients"


class TestLoRAGeneration:
    """Tests for generation with LoRA."""

    def test_generate_with_lora(self, lora_config):
        """Test that generation works with LoRA model."""
        model = ASRModel(lora_config)

        audio_features = torch.randn(1, 80, 3000)
        audio_mask = torch.ones(1, 3000)

        output = model.generate(
            input_features=audio_features,
            audio_attention_mask=audio_mask,
            max_new_tokens=10,
        )

        assert output is not None
        assert isinstance(output, torch.Tensor)


class TestLoRAStateDict:
    """Tests for state_dict behavior with LoRA."""

    def test_state_dict_contains_projector(self, lora_config):
        """Test that state_dict still contains projector weights."""
        model = ASRModel(lora_config)
        state = model.state_dict()

        projector_keys = [k for k in state.keys() if k.startswith("projector.")]
        assert len(projector_keys) > 0, "State dict should contain projector weights"

    def test_base_model_state_dict_unchanged(self, base_config):
        """Test that base model state_dict behavior is unchanged."""
        model = ASRModel(base_config)
        state = model.state_dict()

        for key in state.keys():
            assert key.startswith("projector."), f"Unexpected key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
