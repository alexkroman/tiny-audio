"""Tests for scripts/inference.py: CUDA-graph batching, decode-step kernels and attention."""

import importlib.util
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts import inference
from scripts.inference import (
    GraphedDecoder,
    _gqa_decode_sdpa,
    _mps_safe_sdpa,
    contiguous_lm_head_input,
    missing_fast_kernels,
    use_fast_kernels_under_compile,
)
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_pipeline import ASRPipeline, PreparedChunk


def _chunk(frames: int) -> PreparedChunk:
    return {
        "input_features": torch.zeros(1, frames, 4),
        "attention_mask": torch.ones(1, frames, dtype=torch.long),
    }


class FakeCache:
    """Stands in for StaticCache: records its length and resets."""

    def __init__(self, config: object, max_cache_len: int) -> None:
        self.max_cache_len = max_cache_len
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


EOS = 99
LONG = 30  # a chunk with this many frames "talks" past any budget (short-tier length)


class FakeModel:
    """generate() returns, per row: the row's frame count, then EOS (or no EOS for LONG)."""

    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(max_new_tokens=256, eos_token_id=[EOS, 98])
        self.language_model = SimpleNamespace(config=object())
        self.device = torch.device("cpu")
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        frames = kwargs["audio_attention_mask"].sum(dim=-1).tolist()
        budget = kwargs["max_new_tokens"]
        self.calls.append({"frames": frames, **kwargs})
        rows = [[f] * budget if f == LONG else [f, EOS] + [0] * (budget - 2) for f in frames]
        return torch.tensor(rows)


@pytest.fixture
def decoder(monkeypatch: pytest.MonkeyPatch) -> tuple[GraphedDecoder, FakeModel]:
    monkeypatch.setattr(inference, "StaticCache", FakeCache)

    def prompt_253(pipe: object, prepared: object) -> int:
        return 253

    monkeypatch.setattr(inference, "prompt_tokens", prompt_253)
    model = FakeModel()

    def postprocess(outputs: dict[str, torch.Tensor]) -> dict[str, str]:
        return {"text": f"f{int(outputs['tokens'][0])}"}

    def prepare_chunk(chunk: Any, sample_rate: int) -> PreparedChunk:
        return _chunk(40)  # the longest chunk the pipeline cuts is 40 frames here

    pipe = SimpleNamespace(model=model, postprocess=postprocess, prepare_chunk=prepare_chunk)
    return GraphedDecoder(cast(ASRPipeline, pipe), max_batch_size=24), model


class TestGraphedDecoder:
    def test_buckets_and_tiers(self, decoder: tuple[GraphedDecoder, FakeModel]) -> None:
        dec, _ = decoder
        assert dec.buckets == [1, 2, 4, 8, 16, 24]
        assert [dec.bucket(n) for n in (1, 3, 9, 17, 24)] == [1, 4, 16, 24, 24]
        assert dec.short_enabled  # 253 prompt + 128 budget fits 416 slots
        assert dec.short_max_frames == 40
        assert (dec.full_cache_len, dec.full_budget) == (256 + 384, 256)

    def test_short_tier_pads_to_the_bucket_and_drops_the_padding(
        self, decoder: tuple[GraphedDecoder, FakeModel]
    ) -> None:
        dec, model = decoder
        assert dec([_chunk(5), _chunk(7), _chunk(9)]) == ["f5", "f7", "f9"]
        [call] = model.calls
        assert call["frames"] == [5, 7, 9, 9]  # padded with copies of the last chunk
        assert call["max_new_tokens"] == inference.SHORT_BUDGET
        assert call["past_key_values"].max_cache_len == inference.SHORT_CACHE_LEN

    def test_row_out_of_budget_is_redone_on_the_full_tier(
        self, decoder: tuple[GraphedDecoder, FakeModel]
    ) -> None:
        dec, model = decoder
        assert dec([_chunk(5), _chunk(LONG - 10)]) == ["f5", f"f{LONG - 10}"]
        model.calls.clear()
        texts = dec([_chunk(5), _chunk(LONG), _chunk(7)])
        short, full = model.calls
        assert full["frames"] == [LONG]  # only the cut-off row, alone
        assert full["max_new_tokens"] == 256
        assert full["past_key_values"].max_cache_len == 640
        assert texts == ["f5", f"f{LONG}", "f7"]
        assert dec.redone == 1
        assert short["max_new_tokens"] == inference.SHORT_BUDGET

    def test_chunk_longer_than_calibrated_goes_straight_to_full(
        self, decoder: tuple[GraphedDecoder, FakeModel]
    ) -> None:
        dec, model = decoder
        dec([_chunk(41)])
        [call] = model.calls
        assert call["max_new_tokens"] == 256

    def test_one_cache_per_bucket_and_tier_reused_and_reset(
        self, decoder: tuple[GraphedDecoder, FakeModel]
    ) -> None:
        dec, model = decoder
        dec([_chunk(5)])
        dec([_chunk(6)])
        dec([_chunk(5), _chunk(6)])
        one, again, two = (c["past_key_values"] for c in model.calls)
        assert one is again  # same object: the compiled graph's guards hold
        assert one is not two
        assert cast(FakeCache, one).resets == 1

    def test_short_tier_disabled_when_the_prompt_cannot_fit(
        self, monkeypatch: pytest.MonkeyPatch, decoder: tuple[GraphedDecoder, FakeModel]
    ) -> None:
        dec, model = decoder

        def prompt_300(pipe: object, prepared: object) -> int:
            return 300

        monkeypatch.setattr(inference, "prompt_tokens", prompt_300)
        dec2 = GraphedDecoder(dec.pipe, max_batch_size=4)
        assert not dec2.short_enabled
        dec2([_chunk(5)])
        assert model.calls[-1]["max_new_tokens"] == 256


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


class TestGqaDecodeSdpa:
    """One-token grouped-query attention without repeat_kv matches the stock sdpa path."""

    @staticmethod
    def _module() -> torch.nn.Module:
        # 8 query heads over 2 KV heads, as in Qwen3.5-2B.
        module = SimpleNamespace(training=False, num_key_value_groups=4, is_causal=True)
        return cast(torch.nn.Module, module)

    @pytest.mark.parametrize("mask_kind", ["none", "bool", "float"])
    @pytest.mark.parametrize("q_len", [1, 3])
    def test_matches_stock_sdpa(self, mask_kind: str, q_len: int) -> None:
        gen = torch.Generator().manual_seed(0)
        q = torch.randn(3, 8, q_len, 16, generator=gen)
        k, v = (torch.randn(3, 2, 6, 16, generator=gen) for _ in range(2))
        allowed = torch.ones(3, 1, q_len, 6, dtype=torch.bool)
        allowed[0, ..., :2] = False  # a left-padded row's pad keys
        mask = {
            "none": None,
            "bool": allowed,
            "float": torch.zeros(allowed.shape).masked_fill(~allowed, float("-inf")),
        }[mask_kind]
        module = self._module()
        out, _ = _gqa_decode_sdpa(module, q, k, v, mask, scaling=0.25)
        expected, _ = inference.sdpa_attention_forward(module, q, k, v, mask, scaling=0.25)
        assert out.shape == (3, q_len, 8, 16)
        torch.testing.assert_close(out, expected)


def _fake_modeling(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A modeling module holding one transformers kernel-or-reference wrapper.

    `operator.neg` stands in for the kernel; dynamo's `is_exporting` answer
    inside torch.compile is simulated by patching it True.
    """
    from transformers.integrations.hub_kernels import (  # noqa: PLC0415
        use_kernel_func_from_hub_with_fallback,
    )

    @use_kernel_func_from_hub_with_fallback("neg", "operator")
    def neg(a: int) -> str:
        return "reference"

    module = ModuleType("fake_modeling")
    module.neg = neg  # pyright: ignore[reportAttributeAccessIssue]
    module.helper = lambda: "untouched"  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
    return module


def test_fast_kernels_rebound_past_the_export_check(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _fake_modeling(monkeypatch)
    assert module.neg(3) == "reference"  # the bug: the wrapper falls back under compile
    assert use_fast_kernels_under_compile(module) == ["neg"]
    assert module.neg(3) == -3
    assert module.neg(3, unknown_kwarg=1) == -3  # still filtered like the wrapper does
    assert module.helper() == "untouched"


def test_lm_head_gets_a_contiguous_input() -> None:
    lm_head = torch.nn.Linear(4, 5, bias=False)
    seen: list[bool] = []
    lm_head.register_forward_hook(lambda _m, args, _out: seen.append(args[0].is_contiguous()))
    language_model = SimpleNamespace(get_output_embeddings=lambda: lm_head)
    contiguous_lm_head_input(cast(ASRModel, SimpleNamespace(language_model=language_model)))
    hidden = torch.randn(3, 7, 4)
    last = hidden[:, -1:, :]  # what generate's prefill keeps
    assert not last.is_contiguous()
    torch.testing.assert_close(lm_head(last), last @ lm_head.weight.T)
    assert seen == [True]


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="needs CUDA and flash-linear-attention",
)
def test_in_place_delta_step_matches_fla() -> None:
    from fla.ops.gated_delta_rule import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        fused_recurrent_gated_delta_rule,
    )

    gen = torch.Generator(device="cuda").manual_seed(0)
    batch, heads, dim = 4, 16, 128

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device="cuda", dtype=dtype)

    # q/k/v as strided views of one fused projection, like Qwen3.5's split.
    fused = randn(batch, 1, 3 * heads * dim)
    q, k, v = (t.view(batch, 1, heads, dim) for t in fused.split(heads * dim, dim=-1))
    g = -torch.rand(batch, 1, heads, generator=gen, device="cuda")
    beta = torch.rand(batch, 1, heads, generator=gen, device="cuda").to(torch.bfloat16)
    state = randn(batch, heads, dim, dim, dtype=torch.float32)
    kwargs: dict[str, Any] = {"g": g, "beta": beta, "output_final_state": True}
    expected_out, expected_state = fused_recurrent_gated_delta_rule(
        q, k, v, initial_state=state.clone(), use_qk_l2norm_in_kernel=True, **kwargs
    )
    out, new_state = inference._in_place_recurrent_step(  # pyright: ignore[reportPrivateUsage]
        q, k, v, initial_state=state, use_qk_l2norm_in_kernel=True, fallback=None, **kwargs
    )
    assert new_state is state  # updated in place: the cache's copy becomes a no-op
    assert torch.equal(out, expected_out)
    assert torch.equal(state, expected_state)
