"""Tests for the Hub-free surface of tiny_audio.asr_config.

Everything here builds `ASRConfig` from inline `audio_config` / `text_config`
dicts, so no test touches the network. The auto-detection helpers
(`native_audio_token`, `is_time_major_encoder`) and the conv-length formula are
pure functions and are exercised directly.
"""

import pytest
import torch

from tiny_audio.asr_config import (
    DEFAULT_ENCODER_CONV_LAYERS,
    GRANITE_ENCODER_CONV_LAYERS,
    ASRConfig,
    compute_encoder_output_length,
    is_time_major_encoder,
    native_audio_token,
)

# Minimal sub-configs: enough for `AutoConfig.for_model(model_type)` to
# reconstruct a real config class without downloading anything.
AUDIO_CFG = {"model_type": "whisper", "d_model": 64, "num_mel_bins": 80}
TEXT_CFG = {
    "model_type": "llama",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "vocab_size": 128,
}


def make_config(**overrides) -> ASRConfig:
    """ASRConfig that never calls the Hub."""
    kwargs = {
        "audio_model_id": "openai/whisper-tiny",
        "text_model_id": "Qwen/Qwen3-0.6B",
        "audio_config": dict(AUDIO_CFG),
        "text_config": dict(TEXT_CFG),
    }
    kwargs.update(overrides)
    return ASRConfig(**kwargs)


class TestNativeAudioToken:
    """Decoders with a pretrained audio placeholder reuse it."""

    @pytest.mark.parametrize(
        ("text_model_id", "expected"),
        [
            ("google/gemma-4-e2b-it", "<|audio|>"),
            ("Google/GEMMA-4-E4B", "<|audio|>"),
            ("Qwen/Qwen3-0.6B", None),
            ("HuggingFaceTB/SmolLM2-135M-Instruct", None),
            ("", None),
            (None, None),
        ],
    )
    def test_lookup(self, text_model_id, expected):
        assert native_audio_token(text_model_id) == expected


class TestIsTimeMajorEncoder:
    """Conformer-family encoders take (batch, time, feature) input_features."""

    @pytest.mark.parametrize(
        ("audio_model_id", "expected"),
        [
            ("ibm-granite/granite-speech-3.3-2b", True),
            ("nvidia/parakeet-ctc-0.6b", True),
            ("nvidia/Nemotron-Speech", True),
            ("openai/whisper-tiny", False),
            ("zai-org/GLM-ASR-Nano-2512", False),
            ("", False),
            (None, False),
        ],
    )
    def test_lookup(self, audio_model_id, expected):
        assert is_time_major_encoder(audio_model_id) is expected


class TestComputeEncoderOutputLength:
    """The conv-length formula reproduces the real encoders' downsampling."""

    @pytest.mark.parametrize(
        ("mel_length", "expected"),
        # Verified against the real Granite encoder at 1/2/5/10/20s; see the
        # comment on GRANITE_ENCODER_CONV_LAYERS.
        [(50, 12), (100, 25), (250, 62), (500, 125), (1000, 250)],
    )
    def test_granite_layers_match_documented_values(self, mel_length, expected):
        assert compute_encoder_output_length(mel_length, GRANITE_ENCODER_CONV_LAYERS) == expected

    @pytest.mark.parametrize(("mel_length", "expected"), [(3000, 1500), (2999, 1500), (1, 1)])
    def test_whisper_layers_halve(self, mel_length, expected):
        assert compute_encoder_output_length(mel_length) == expected
        assert compute_encoder_output_length(mel_length, DEFAULT_ENCODER_CONV_LAYERS) == expected

    def test_no_layers_is_identity(self):
        assert compute_encoder_output_length(123, []) == 123

    def test_granite_layers_accept_tensor_batches(self):
        lengths = torch.tensor([50, 100, 250, 500, 1000])
        out = compute_encoder_output_length(lengths, GRANITE_ENCODER_CONV_LAYERS)
        assert out.tolist() == [12, 25, 62, 125, 250]


class TestASRConfigAutoDetection:
    """Encoder-layout flags follow `audio_model_id` unless set explicitly."""

    def test_whisper_is_channel_major_without_encoder_mask(self):
        cfg = make_config(audio_model_id="openai/whisper-tiny")
        assert cfg.audio_features_time_major is False
        assert cfg.encoder_attention_mask is False

    def test_granite_is_time_major_with_encoder_mask(self):
        cfg = make_config(audio_model_id="ibm-granite/granite-speech-3.3-2b")
        assert cfg.audio_features_time_major is True
        assert cfg.encoder_attention_mask is True

    def test_explicit_flags_override_detection(self):
        cfg = make_config(
            audio_model_id="ibm-granite/granite-speech-3.3-2b",
            audio_features_time_major=False,
            encoder_attention_mask=False,
        )
        assert cfg.audio_features_time_major is False
        assert cfg.encoder_attention_mask is False

    def test_default_conv_layers(self):
        assert make_config().encoder_conv_layers == DEFAULT_ENCODER_CONV_LAYERS

    def test_custom_conv_layers_kept(self):
        cfg = make_config(encoder_conv_layers=GRANITE_ENCODER_CONV_LAYERS)
        assert cfg.encoder_conv_layers == GRANITE_ENCODER_CONV_LAYERS


class TestASRConfigAudioToken:
    """The placeholder token defaults to the decoder's native one when it has one."""

    def test_qwen_gets_generic_token(self):
        assert make_config(text_model_id="Qwen/Qwen3-0.6B").audio_token == "<audio>"

    def test_gemma_reuses_native_token(self):
        assert make_config(text_model_id="google/gemma-4-e2b-it").audio_token == "<|audio|>"

    def test_explicit_token_wins(self):
        cfg = make_config(text_model_id="google/gemma-4-e2b-it", audio_token="<snd>")
        assert cfg.audio_token == "<snd>"


class TestASRConfigDefaults:
    """Derived defaults that the training recipes rely on."""

    def test_projector_dtype_follows_model_dtype(self):
        assert make_config(model_dtype="bfloat16").projector_dtype == "bfloat16"

    def test_projector_dtype_can_be_pinned_separately(self):
        cfg = make_config(model_dtype="bfloat16", projector_dtype="float32")
        assert cfg.projector_dtype == "float32"
        assert cfg.model_dtype == "bfloat16"

    def test_lora_targets_default_to_all_linear_layers(self):
        assert make_config().lora_target_modules == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def test_custom_lora_targets_kept(self):
        assert make_config(lora_target_modules=["q_proj"]).lora_target_modules == ["q_proj"]

    def test_generation_defaults(self):
        cfg = make_config()
        assert cfg.max_new_tokens == 128
        assert cfg.use_cache is True

    def test_explicit_use_cache_false_is_kept(self):
        # `None` means "use the default"; an explicit False must not be
        # mistaken for it.
        assert make_config(use_cache=False).use_cache is False

    def test_sub_configs_are_reconstructed_from_dicts(self):
        cfg = make_config()
        assert cfg.audio_config.model_type == "whisper"
        assert cfg.text_config.model_type == "llama"
        assert cfg.encoder is cfg.audio_config

    def test_dict_round_trip_preserves_derived_fields(self):
        cfg = make_config(
            audio_model_id="ibm-granite/granite-speech-3.3-2b",
            text_model_id="google/gemma-4-e2b-it",
            encoder_conv_layers=GRANITE_ENCODER_CONV_LAYERS,
            projector_dtype="float32",
        )
        restored = ASRConfig(**cfg.to_dict())
        assert restored.audio_token == "<|audio|>"
        assert restored.audio_features_time_major is True
        assert restored.encoder_attention_mask is True
        assert [tuple(layer) for layer in restored.encoder_conv_layers] == [
            tuple(layer) for layer in GRANITE_ENCODER_CONV_LAYERS
        ]
        assert restored.projector_dtype == "float32"
