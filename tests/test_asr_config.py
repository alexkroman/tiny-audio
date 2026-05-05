"""Tests for ASRConfig — round-trip serialization, AutoConfig registration, validation."""

import json

import transformers


class TestASRConfigDefaults:
    """ASRConfig with no overrides should produce a valid config."""

    def test_default_projector_type_is_mlp(self, base_asr_config):
        assert base_asr_config.projector_type == "mlp"

    def test_default_generation_params_are_greedy(self, base_asr_config):
        assert base_asr_config.do_sample is False
        assert base_asr_config.num_beams == 1
        assert base_asr_config.max_new_tokens == 128
        assert base_asr_config.repetition_penalty == 1.0

    def test_default_lora_disabled(self, base_asr_config):
        assert base_asr_config.use_lora is False

    def test_lora_target_modules_default(self, base_asr_config):
        # Default list set in __init__ when None passed
        assert "q_proj" in base_asr_config.lora_target_modules
        assert "v_proj" in base_asr_config.lora_target_modules

    def test_audio_config_attached(self, base_asr_config):
        assert base_asr_config.audio_config is not None
        assert hasattr(base_asr_config.audio_config, "model_type")

    def test_text_config_attached(self, base_asr_config):
        assert base_asr_config.text_config is not None

    def test_encoder_alias_points_to_audio_config(self, base_asr_config):
        # ASRConfig.__init__ sets self.encoder = self.audio_config
        assert base_asr_config.encoder is base_asr_config.audio_config

    def test_auto_map_registered(self, base_asr_config):
        assert base_asr_config.auto_map["AutoConfig"] == "asr_config.ASRConfig"
        assert base_asr_config.auto_map["AutoModel"] == "asr_modeling.ASRModel"

    def test_pipeline_metadata(self, base_asr_config):
        assert base_asr_config.pipeline_tag == "automatic-speech-recognition"
        assert base_asr_config.architectures == ["ASRModel"]


class TestASRConfigSerialization:
    """to_dict / to_json / save / load round-trips."""

    def test_to_dict_round_trip(self, base_asr_config):
        from tiny_audio.asr_config import ASRConfig

        d = base_asr_config.to_dict()
        # Reconstruct from dict
        cfg2 = ASRConfig(**d)
        assert cfg2.audio_model_id == base_asr_config.audio_model_id
        assert cfg2.text_model_id == base_asr_config.text_model_id
        assert cfg2.projector_type == base_asr_config.projector_type

    def test_to_json_string_is_valid_json(self, base_asr_config):
        s = base_asr_config.to_json_string()
        parsed = json.loads(s)
        assert parsed["audio_model_id"] == base_asr_config.audio_model_id

    def test_save_and_from_pretrained_round_trip(self, base_asr_config, tmp_path):
        from tiny_audio.asr_config import ASRConfig

        save_dir = tmp_path / "cfg"
        save_dir.mkdir()
        base_asr_config.save_pretrained(save_dir)

        # config.json should exist
        assert (save_dir / "config.json").exists()

        loaded = ASRConfig.from_pretrained(save_dir)
        assert loaded.audio_model_id == base_asr_config.audio_model_id
        assert loaded.text_model_id == base_asr_config.text_model_id
        assert loaded.projector_type == base_asr_config.projector_type
        assert loaded.use_lora == base_asr_config.use_lora

    def test_text_config_dict_round_trip(self):
        """text_config passed as dict should be reconstructed via AutoConfig."""
        from tiny_audio.asr_config import ASRConfig

        cfg = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
        )
        text_config_dict = cfg.text_config.to_dict()

        cfg2 = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            text_config=text_config_dict,
        )
        # Should have rebuilt a config object, not kept the dict
        assert not isinstance(cfg2.text_config, dict)
        assert cfg2.text_config.model_type == cfg.text_config.model_type


class TestASRConfigOverrides:
    """Explicit overrides win over defaults."""

    def test_explicit_max_new_tokens_overrides_default(self):
        from tiny_audio.asr_config import ASRConfig

        cfg = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            max_new_tokens=256,
        )
        assert cfg.max_new_tokens == 256

    def test_lora_enabled_with_custom_rank(self):
        from tiny_audio.asr_config import ASRConfig

        cfg = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            use_lora=True,
            lora_rank=16,
            lora_alpha=32,
        )
        assert cfg.use_lora is True
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32

    def test_custom_projector_type(self):
        from tiny_audio.asr_config import ASRConfig

        cfg = ASRConfig(
            audio_model_id="openai/whisper-tiny",
            text_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            attn_implementation="eager",
            model_dtype="float32",
            projector_type="mosa",
        )
        assert cfg.projector_type == "mosa"


class TestComputeEncoderOutputLength:
    """compute_encoder_output_length applies conv layer formulas."""

    def test_default_whisper_layers(self):
        from tiny_audio.asr_config import compute_encoder_output_length

        # Whisper default: [(1,3,1), (1,3,2)]
        # First: (3000 + 2 - 2 - 1) // 1 + 1 = 3000
        # Second: (3000 + 2 - 2 - 1) // 2 + 1 = 1500
        assert compute_encoder_output_length(3000) == 1500

    def test_custom_layers(self):
        from tiny_audio.asr_config import compute_encoder_output_length

        # Single layer: (100 + 0 - 0 - 1) // 1 + 1 = 100
        result = compute_encoder_output_length(100, conv_layers=[(0, 1, 1)])
        assert result == 100

    def test_works_with_torch_tensor(self):
        import torch

        from tiny_audio.asr_config import compute_encoder_output_length

        result = compute_encoder_output_length(torch.tensor([3000, 1500]))
        assert torch.equal(result, torch.tensor([1500, 750]))


class TestAutoConfigRegistration:
    """ASRConfig is registered with transformers.AutoConfig at import time."""

    def test_auto_config_resolves_asr_model(self):
        from tiny_audio.asr_config import ASRConfig

        # Register happens at module import. Confirm the registry has it.
        config_class = transformers.AutoConfig.for_model("asr_model").__class__
        assert config_class is ASRConfig
