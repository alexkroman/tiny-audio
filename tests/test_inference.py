"""Tests for scripts/inference.py: CUDA-graph batching and NaN-safe MPS attention, on CPU."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts import inference
from scripts.inference import GraphedDecoder, _mps_safe_sdpa, missing_fast_kernels
from tiny_audio.asr_pipeline import ASRPipeline, PreparedChunk


def _chunk(frames: int) -> PreparedChunk:
    return {
        "input_features": torch.zeros(1, frames, 4),
        "attention_mask": torch.ones(1, frames, dtype=torch.long),
    }


class FakeCache:
    """Stands in for StaticCache: records construction and resets."""

    def __init__(self, config: object, max_cache_len: int) -> None:
        self.max_cache_len = max_cache_len
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def decoder(monkeypatch: pytest.MonkeyPatch) -> tuple[GraphedDecoder, list[dict[str, Any]]]:
    monkeypatch.setattr(inference, "StaticCache", FakeCache)
    calls: list[dict[str, Any]] = []

    def generate_prepared(prepared: list[PreparedChunk], **kwargs: Any) -> list[str]:
        calls.append({"frames": [int(p["attention_mask"].shape[-1]) for p in prepared], **kwargs})
        return [f"t{i}" for i in range(len(prepared))]

    model = SimpleNamespace(
        generation_config=SimpleNamespace(max_new_tokens=256),
        language_model=SimpleNamespace(config=object()),
    )
    pipe = SimpleNamespace(model=model, generate_prepared=generate_prepared)
    return GraphedDecoder(cast(ASRPipeline, pipe), max_batch_size=24), calls


class TestGraphedDecoder:
    def test_buckets_are_powers_of_two_up_to_the_max(
        self, decoder: tuple[GraphedDecoder, list[dict[str, Any]]]
    ) -> None:
        dec, _ = decoder
        assert dec.buckets == [1, 2, 4, 8, 16, 24]
        assert [dec.bucket(n) for n in (1, 3, 9, 17, 24)] == [1, 4, 16, 24, 24]
        assert dec.cache_len == 256 + 384

    def test_pads_to_the_bucket_and_drops_the_padding(
        self, decoder: tuple[GraphedDecoder, list[dict[str, Any]]]
    ) -> None:
        dec, calls = decoder
        texts = dec([_chunk(5), _chunk(7), _chunk(9)])
        assert texts == ["t0", "t1", "t2"]  # the padded fourth row is dropped
        assert calls[0]["frames"] == [5, 7, 9, 9]  # padded with copies of the last chunk

    def test_one_cache_per_bucket_reused_and_reset(
        self, decoder: tuple[GraphedDecoder, list[dict[str, Any]]]
    ) -> None:
        dec, calls = decoder
        dec([_chunk(5)])
        dec([_chunk(6)])
        dec([_chunk(5), _chunk(6)])
        one, again, two = (c["past_key_values"] for c in calls)
        assert one is again  # same object: the compiled graph's guards hold
        assert one is not two
        assert cast(FakeCache, one).resets == 1
        assert cast(FakeCache, two).resets == 0


class TestMpsSafeSdpa:
    """Fully-masked query rows (left padding) come out finite; real rows are unchanged."""

    @staticmethod
    def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(0)
        return tuple(torch.randn(1, 2, 4, 8, generator=gen) for _ in range(3))  # type: ignore[return-value]

    @pytest.mark.parametrize("boolean", [True, False])
    def test_padded_rows_finite_real_rows_exact(self, boolean: bool) -> None:
        q, k, v = self._qkv()
        # Row 0 is a left-pad position that may attend to nothing.
        allowed = torch.tensor(
            [
                [False] * 4,
                [False, True, False, False],
                [False, True, True, False],
                [False, True, True, True],
            ]
        )
        mask = allowed if boolean else torch.zeros(4, 4).masked_fill(~allowed, float("-inf"))
        mask = mask[None, None]
        module = torch.nn.Module()
        module.training = False
        out, _ = _mps_safe_sdpa(module, q, k, v, mask, is_causal=False)
        assert torch.isfinite(out).all()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, 1:], k, v, attn_mask=mask[:, :, 1:]
        ).transpose(1, 2)
        torch.testing.assert_close(out[:, 1:], expected)


def test_missing_fast_kernels_reports_pip_names(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_spec(_name: str) -> None:
        return None

    monkeypatch.setattr(inference.importlib.util, "find_spec", no_spec)
    assert missing_fast_kernels() == ["causal-conv1d", "flash-linear-attention"]
