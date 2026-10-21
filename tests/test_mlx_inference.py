"""Tests for scripts/mlx_inference.py (MLX is macOS-only: conftest skips it elsewhere)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx_lm.models import qwen3
from safetensors.torch import save_file
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from scripts import mlx_inference as mi
from tiny_audio.projectors import MLPAudioProjector


class TestBannedNextTokens:
    """HF's no_repeat_ngram rule: ban whatever would complete a seen n-gram."""

    def test_bans_the_continuation_of_a_repeated_prefix(self) -> None:
        assert mi.banned_next_tokens([1, 2, 3, 1, 2], 3) == [3]

    def test_short_or_disabled_bans_nothing(self) -> None:
        assert mi.banned_next_tokens([1, 2], 3) == []
        assert mi.banned_next_tokens([1, 2, 1, 2], 0) == []

    def test_processor_sets_banned_logits_to_minus_inf(self) -> None:
        logits = mx.zeros((1, 5))
        out = mi.no_repeat_ngram_processor(2)(mx.array([4, 1, 4]), logits)
        assert np.isneginf(np.array(out)[0, 1])
        assert np.isfinite(np.array(out)[0, [0, 2, 3, 4]]).all()


def test_projector_matches_torch() -> None:
    """Same weights, same input: the MLX projector reproduces MLPAudioProjector."""
    torch.manual_seed(0)
    cfg = SimpleNamespace(
        encoder_dim=8, llm_dim=6, projector_pool_stride=2, projector_hidden_dim=16
    )
    ref = MLPAudioProjector(cfg).eval()
    x = torch.randn(1, 7, 8)
    mlx_proj = mi.Projector(8, 16, 6, 2)
    mlx_proj.load_weights([(k, mx.array(v.numpy())) for k, v in ref.state_dict().items()])
    with torch.no_grad():
        expected = ref(x).numpy()
    np.testing.assert_allclose(np.array(mlx_proj(mx.array(x.numpy()))), expected, atol=1e-5)


def test_weight_names_are_mapped() -> None:
    ckpt = {
        "language_model.model.norm.weight": 1,
        "language_model.lm_head.weight": 2,
        "projector.linear_1.weight": 3,
    }
    assert mi.decoder_weights(ckpt, tie_word_embeddings=True) == {"model.norm.weight": 1}
    assert "lm_head.weight" in mi.decoder_weights(ckpt, tie_word_embeddings=False)
    assert mi.projector_weights(ckpt) == {"linear_1.weight": 3}


def test_clean_text_matches_the_pipeline() -> None:
    assert mi.clean_text("  <think>\n</think>\nHello there. ") == "Hello there."


class TestConversionCache:
    """A Hub snapshot converts once per revision; a plain directory never caches."""

    def _checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True)
        save_file(
            {
                "language_model.model.norm.weight": torch.ones(4),
                "projector.linear_1.weight": torch.ones(2, 2),
            },
            str(directory / "model.safetensors"),
        )

    def test_snapshot_is_converted_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TINY_AUDIO_MLX_CACHE", str(tmp_path / "cache"))
        snapshot = tmp_path / "models--org--name" / "snapshots" / "abc123"
        self._checkpoint(snapshot)
        first = mi.converted_weights(snapshot, tie_word_embeddings=True)
        cached = tmp_path / "cache" / "org--name" / "abc123.safetensors"
        assert cached.exists()
        assert first["decoder.model.norm.weight"].dtype == mx.bfloat16
        assert first["projector.linear_1.weight"].dtype == mx.float32
        (snapshot / "model.safetensors").unlink()  # a second load must not need it
        assert set(mi.converted_weights(snapshot, tie_word_embeddings=True)) == set(first)

    def test_local_directory_is_not_cached(self, tmp_path: Path) -> None:
        local = tmp_path / "ckpt"
        self._checkpoint(local)
        assert mi._cache_file(local) is None
        assert "decoder.model.norm.weight" in mi.converted_weights(local, True)


def test_generation_settings_read_generation_config(tmp_path: Path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 64, "no_repeat_ngram_size": 12, "eos_token_id": [7, 9]})
    )
    gen = mi._generation_settings(tmp_path, {})
    assert (gen.max_new_tokens, gen.no_repeat_ngram_size, gen.eos_token_ids) == (64, 12, {7, 9})


class TestFromPretrainedRejects:
    """Unsupported checkpoints fail with a reason before anything is downloaded."""

    def _config(self, tmp_path: Path, **values: Any) -> str:
        (tmp_path / "config.json").write_text(json.dumps(values))
        return str(tmp_path)

    def test_non_granite_encoder(self, tmp_path: Path) -> None:
        path = self._config(tmp_path, audio_model_id="Qwen/Qwen3-ASR-1.7B-hf")
        with pytest.raises(ValueError, match=r"Granite Speech 5\.0"):
            mi.MLXTranscriber.from_pretrained(path)

    def test_lora_checkpoint(self, tmp_path: Path) -> None:
        path = self._config(
            tmp_path,
            audio_model_id="ibm-granite/granite-speech-5.0-470m-turboctc-nc",
            use_lora=True,
        )
        with pytest.raises(ValueError, match="LoRA"):
            mi.MLXTranscriber.from_pretrained(path)


@pytest.fixture
def tiny_transcriber() -> mi.MLXTranscriber:
    """Random tiny parts around the offline SmolLM2 tokenizer, with an <audio> token."""
    tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    tok.add_special_tokens({"additional_special_tokens": ["<audio>"]})
    decoder = qwen3.Model(
        qwen3.ModelArgs.from_dict(
            {
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": len(tok),
                "max_position_embeddings": 512,
                "rope_theta": 10000.0,
                "tie_word_embeddings": True,
            }
        )
    )

    def encoder(features: Any) -> Any:
        frames = int(features.shape[1]) // 4  # Granite's 4x subsampling
        return mx.ones((1, frames, 8))

    eos = cast(int, tok.eos_token_id)
    generation = mi.GenerationSettings(
        max_new_tokens=5, no_repeat_ngram_size=3, eos_token_ids={eos}
    )
    return mi.MLXTranscriber(
        encoder, mi.Projector(8, 16, 16, 1), decoder, tok, {"audio_token": "<audio>"}, generation
    )


class TestTranscribe:
    """The splice puts one projector frame on each placeholder, then decodes greedily."""

    def test_audio_replaces_exactly_the_placeholders(
        self, tiny_transcriber: mi.MLXTranscriber
    ) -> None:
        audio = mx.ones((3, 16)) * 7
        ids, embeds = tiny_transcriber._prompt_embeddings(audio)
        id_list = ids.tolist()
        assert id_list.count(tiny_transcriber.audio_token_id) == 3
        start = id_list.index(tiny_transcriber.audio_token_id)
        np.testing.assert_allclose(np.array(embeds[start : start + 3]), np.full((3, 16), 7.0))
        assert embeds.shape[0] == len(id_list)

    def test_decoding_stops_at_the_token_budget(self, tiny_transcriber: mi.MLXTranscriber) -> None:
        audio = tiny_transcriber._audio_embeddings(np.ones(16000, dtype=np.float32) * 0.1)
        tokens = list(tiny_transcriber._decode(*tiny_transcriber._prompt_embeddings(audio)))
        assert 0 < len(tokens) <= tiny_transcriber.generation.max_new_tokens
        assert isinstance(tiny_transcriber.transcribe(np.ones(16000, dtype=np.float32) * 0.1), str)

    def test_digital_silence_is_empty(self, tiny_transcriber: mi.MLXTranscriber) -> None:
        assert tiny_transcriber.transcribe(np.zeros(16000, dtype=np.float32)) == ""
