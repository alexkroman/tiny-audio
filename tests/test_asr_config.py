"""Tests for ASRConfig — round-trip serialization, AutoConfig registration, validation."""

import json
from pathlib import Path

import transformers
from conftest import make_asr_config

from tiny_audio.asr_config import ASRConfig


class TestASRConfigDefaults:
    """ASRConfig with no overrides should produce a valid config."""

    def test_default_projector_type_is_mlp(self, base_asr_config: ASRConfig) -> None:
        assert base_asr_config.projector_type == "mlp"

    def test_default_lora_disabled(self, base_asr_config: ASRConfig) -> None:
        assert base_asr_config.use_lora is False

    def test_audio_config_attached(self, base_asr_config: ASRConfig) -> None:
        assert base_asr_config.audio_config is not None
        assert hasattr(base_asr_config.audio_config, "model_type")

    def test_text_config_attached(self, base_asr_config: ASRConfig) -> None:
        assert base_asr_config.text_config is not None

    def test_auto_map_registered(self, base_asr_config: ASRConfig) -> None:
        assert base_asr_config.auto_map["AutoConfig"] == "asr_config.ASRConfig"
        assert base_asr_config.auto_map["AutoModel"] == "asr_modeling.ASRModel"

    def test_pipeline_metadata(self, base_asr_config: ASRConfig) -> None:
        assert base_asr_config.pipeline_tag == "automatic-speech-recognition"
        assert base_asr_config.architectures == ["ASRModel"]


class TestASRConfigSerialization:
    """to_dict / to_json / save / load round-trips."""

    def test_to_dict_round_trip(self, base_asr_config: ASRConfig) -> None:
        d = base_asr_config.to_dict()
        # Reconstruct from dict
        cfg2 = ASRConfig(**d)
        assert cfg2.audio_model_id == base_asr_config.audio_model_id
        assert cfg2.text_model_id == base_asr_config.text_model_id
        assert cfg2.projector_type == base_asr_config.projector_type

    def test_to_json_string_is_valid_json(self, base_asr_config: ASRConfig) -> None:
        s = base_asr_config.to_json_string()
        parsed = json.loads(s)
        assert parsed["audio_model_id"] == base_asr_config.audio_model_id

    def test_save_and_from_pretrained_round_trip(
        self, base_asr_config: ASRConfig, tmp_path: Path
    ) -> None:
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


class TestASRConfigOverrides:
    """Explicit overrides win over defaults."""

    def test_explicit_max_new_tokens_overrides_default(self) -> None:
        cfg = make_asr_config(max_new_tokens=256)
        assert cfg.max_new_tokens == 256

    def test_lora_enabled_with_custom_rank(self) -> None:
        cfg = make_asr_config(use_lora=True, lora_rank=16, lora_alpha=32)
        assert cfg.use_lora is True
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32

    def test_custom_projector_type(self) -> None:
        cfg = make_asr_config(projector_type="mosa")
        assert cfg.projector_type == "mosa"


class TestAutoConfigRegistration:
    """ASRConfig is registered with transformers.AutoConfig at import time."""

    def test_auto_config_resolves_asr_model(self) -> None:
        # Register happens at module import. Confirm the registry has it.
        config = transformers.AutoConfig.for_model("asr_model")
        config_class = config.__class__
        assert config_class is ASRConfig
