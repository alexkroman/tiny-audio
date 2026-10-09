"""Tests for scripts/eval package.

Note: Audio utility tests (audio_to_wav_bytes, prepare_wav_bytes, TextNormalizer)
are in test_eval_audio.py to avoid duplication.
"""

import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import httpx
import numpy as np
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PretrainedConfig, PreTrainedModel

from scripts.eval.cli import _build_tiny_audio_evaluator
from scripts.eval.evaluators.asr import (
    DTYPE_CONFIG_FIELDS,
    AssemblyAIStreamingEvaluator,
    EndpointEvaluator,
    _resolve_local_runtime,
    _use_sdpa_where_safe,
)
from scripts.inference import merge_lora_adapters
from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel


class _DtypeOverrides(TypedDict):
    """One value per entry of DTYPE_CONFIG_FIELDS, as ASRConfig kwargs."""

    model_dtype: str
    projector_dtype: str
    encoder_dtype: str


class _StreamClosedError(RuntimeError):
    """A stream-close error carrying the close code the SDK attaches."""

    streaming_code: int


class TestResolveLocalRuntime:
    """Tests for _resolve_local_runtime, the local-model device/dtype policy.

    The regression these guard is silent: the wrong answer here costs 2x memory
    bandwidth on every decoded token and reports nothing, because fp32 is a
    perfectly valid dtype to run in.
    """

    @staticmethod
    def _resolver() -> Callable[[], tuple[int | str, str]]:
        return _resolve_local_runtime

    def test_cuda_prefers_bfloat16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert self._resolver()() == (0, "bfloat16")

    def test_mps_uses_bfloat16_not_float16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """bf16, not fp16.

        Measured equal in speed on torch 2.8 / Metal, so fp16 buys nothing
        while giving up the exponent range -- and both submodels of this stack
        are pretrained in bf16.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert self._resolver()() == ("mps", "bfloat16")

    def test_cpu_stays_float32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert self._resolver()() == (-1, "float32")

    def test_returns_dtype_as_string_not_torch_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Must be the string ASRConfig.model_dtype wants, not a torch.dtype.

        getattr(torch, config.model_dtype) in ASRModel.__init__ would raise on a
        torch.dtype instance, so this is the contract that makes the override
        land at all.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        _, dtype = self._resolver()()
        assert isinstance(dtype, str)
        assert getattr(torch, dtype) is torch.bfloat16


class TestModelDtypeIsTheWorkingOverride:
    """Pins WHY the fix routes dtype through model_kwargs instead of `dtype=`.

    ASRModel.__init__ reads `getattr(torch, config.model_dtype)` and then casts
    both submodules explicitly after load, so a `dtype=` kwarg on
    pipeline()/from_pretrained() is silently overwritten by the saved config.
    That is what left `ta eval` running granite-qwen's fp32 training weights at
    inference. If a future refactor makes `dtype=` authoritative, the first
    assertion here flips and this comment stops being true.
    """

    def test_dtype_kwarg_does_not_change_model_dtype(self, tmp_path: Path) -> None:
        ASRConfig(model_dtype="float32").save_pretrained(tmp_path)
        cfg = ASRConfig.from_pretrained(tmp_path, dtype=torch.bfloat16)
        assert cfg.model_dtype == "float32"

    def test_model_dtype_kwarg_does_change_it(self, tmp_path: Path) -> None:
        ASRConfig(model_dtype="float32").save_pretrained(tmp_path)
        cfg = ASRConfig.from_pretrained(tmp_path, model_dtype="bfloat16")
        assert cfg.model_dtype == "bfloat16"


class TestInferenceDtypeFieldsAllLand:
    """`model_dtype` alone does not reach the encoder or the projector.

    `projector_dtype` / `encoder_dtype` exist to hold fp32 MASTER WEIGHTS for
    AdamW, they are persisted into every checkpoint's config.json, and
    ASRModel prefers them over `model_dtype` when casting. So the bandwidth
    argument `TestModelDtypeIsTheWorkingOverride` pins for the decoder was
    only half-applied: a `-top4` eval held the 473M-param Granite encoder in
    fp32, and every eval held the projector there.
    """

    def test_all_three_fields_are_overridden_together(self, tmp_path: Path) -> None:
        ASRConfig(
            model_dtype="bfloat16", projector_dtype="float32", encoder_dtype="float32"
        ).save_pretrained(tmp_path)

        overrides: _DtypeOverrides = {
            "model_dtype": "bfloat16",
            "projector_dtype": "bfloat16",
            "encoder_dtype": "bfloat16",
        }
        assert set(overrides) == set(DTYPE_CONFIG_FIELDS)
        cfg = ASRConfig.from_pretrained(tmp_path, **overrides)

        assert [getattr(cfg, f) for f in DTYPE_CONFIG_FIELDS] == ["bfloat16"] * 3

    def test_model_dtype_alone_leaves_the_encoder_in_float32(self, tmp_path: Path) -> None:
        """The regression itself, so the constant cannot be quietly narrowed back."""
        ASRConfig(
            model_dtype="bfloat16", projector_dtype="float32", encoder_dtype="float32"
        ).save_pretrained(tmp_path)

        cfg = ASRConfig.from_pretrained(tmp_path, model_dtype="bfloat16")

        assert cfg.encoder_dtype == "float32"
        assert cfg.projector_dtype == "float32"


class TestMergeLoraAdapters:
    """Eval decodes through merged weights, not a live PeftModel.

    Unmerged adapters cost two extra matmuls per adapted linear per token, in
    fp32 against a bf16 base. Measured on an M-series Mac (granite-qwen-frozen,
    sdpa, batch 1): 20.6 -> 36.7 tok/s once merged.
    """

    @staticmethod
    def _peft_holder() -> types.SimpleNamespace:
        class Tiny(PreTrainedModel):
            config_class = PretrainedConfig

            def __init__(self) -> None:
                super().__init__(PretrainedConfig())
                self.lin = nn.Linear(4, 4)

        peft_model = get_peft_model(Tiny(), LoraConfig(target_modules=["lin"], r=2))
        return types.SimpleNamespace(language_model=peft_model)

    def test_merges_and_unwraps_a_peft_decoder(self) -> None:
        holder = self._peft_holder()
        assert merge_lora_adapters(cast(ASRModel, holder)) is True
        assert not isinstance(holder.language_model, PeftModel)

    def test_is_a_noop_without_lora(self) -> None:
        holder = types.SimpleNamespace(language_model=nn.Linear(4, 4))
        original = holder.language_model

        assert merge_lora_adapters(cast(ASRModel, holder)) is False
        assert holder.language_model is original


class TestUseSdpaWhereSafe:
    """The MPS attention policy has to be re-applied AFTER load.

    On the default eval path `trust_remote_code` imports the CHECKPOINT's
    asr_modeling.py, so the policy that ran is whatever was published with the
    weights -- and every checkpoint on the Hub carries the blanket
    "MPS -> eager" version that predates scoping it to sliding-window models.
    Qwen3.5-2B has `sliding_window: None` and cannot hit the Metal bug, so it
    was paying ~30% of decode throughput for a guard that did not apply to it.
    """

    @staticmethod
    def _holder(
        loaded_impl: str, requested: str = "sdpa"
    ) -> tuple[types.SimpleNamespace, list[str]]:
        calls: list[str] = []
        language_model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation=loaded_impl),
            set_attn_implementation=calls.append,
        )
        holder = types.SimpleNamespace(
            config=types.SimpleNamespace(
                attn_implementation=requested, text_model_id="stub/decoder"
            ),
            language_model=language_model,
        )
        return holder, calls

    @pytest.fixture
    def on_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    @staticmethod
    def _set_sliding_window(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
        def has_sliding_window(_model_id: str) -> bool:
            return value

        monkeypatch.setattr(
            "tiny_audio.asr_attention._has_sliding_window_attention", has_sliding_window
        )

    def test_corrects_a_stale_eager_decoder_to_sdpa(
        self, monkeypatch: pytest.MonkeyPatch, on_mps: None
    ) -> None:
        self._set_sliding_window(monkeypatch, False)
        holder, calls = self._holder("eager")

        _use_sdpa_where_safe(cast(ASRModel, holder))

        assert calls == ["sdpa"]

    def test_leaves_a_sliding_window_model_on_eager(
        self, monkeypatch: pytest.MonkeyPatch, on_mps: None
    ) -> None:
        """Metal's sdpa returns wrong results for cached decode against that mask."""
        self._set_sliding_window(monkeypatch, True)
        holder, calls = self._holder("eager")

        _use_sdpa_where_safe(cast(ASRModel, holder))

        assert calls == []

    def test_is_a_noop_when_load_already_resolved_correctly(
        self, monkeypatch: pytest.MonkeyPatch, on_mps: None
    ) -> None:
        """--local-code already gets this right; re-applying must not churn the model."""
        self._set_sliding_window(monkeypatch, False)
        holder, calls = self._holder("sdpa")

        _use_sdpa_where_safe(cast(ASRModel, holder))

        assert calls == []


def _no_pcm(_a: object) -> bytes:
    return b""


class TestStreamingRetry:
    """AssemblyAIStreamingEvaluator retries only on transient stream close codes."""

    @pytest.fixture
    def evaluator(
        self, monkeypatch: pytest.MonkeyPatch, no_retry_backoff: None
    ) -> AssemblyAIStreamingEvaluator:
        evaluator = AssemblyAIStreamingEvaluator(api_key="test")
        monkeypatch.setattr(evaluator, "_prepare_pcm", _no_pcm)
        return evaluator

    @staticmethod
    def _stream_error(code: int | None) -> RuntimeError:
        exc = _StreamClosedError(f"closed with {code}")
        if code is not None:
            exc.streaming_code = code
        return exc

    def test_transient_close_code_is_retried(
        self, evaluator: AssemblyAIStreamingEvaluator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def run_session(_pcm: bytes) -> tuple[str, float]:
            calls.append(1)
            if len(calls) < 3:
                raise self._stream_error(1013)
            return "hello", 0.5

        monkeypatch.setattr(evaluator, "_run_session", run_session)

        assert evaluator.transcribe(object()) == ("hello", 0.5, None)
        assert len(calls) == 3

    def test_non_retryable_error_is_raised_immediately(
        self, evaluator: AssemblyAIStreamingEvaluator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def run_session(_pcm: bytes) -> tuple[str, float]:
            calls.append(1)
            raise self._stream_error(None)

        monkeypatch.setattr(evaluator, "_run_session", run_session)

        with pytest.raises(RuntimeError, match="closed with None"):
            evaluator.transcribe(object())
        assert len(calls) == 1

    def test_gives_up_after_max_retries(
        self, evaluator: AssemblyAIStreamingEvaluator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def run_session(_pcm: bytes) -> tuple[str, float]:
            calls.append(1)
            raise self._stream_error(4029)

        monkeypatch.setattr(evaluator, "_run_session", run_session)

        with pytest.raises(RuntimeError, match="closed with 4029"):
            evaluator.transcribe(object())
        assert len(calls) == AssemblyAIStreamingEvaluator._MAX_RETRIES + 1


def test_endpoint_evaluator_gets_num_workers() -> None:
    """`ta eval --endpoint -w N` keeps N requests in flight for the server to batch."""
    _, evaluator = _build_tiny_audio_evaluator(
        model="https://pod-8000.proxy.runpod.net",
        endpoint=True,
        streaming=False,
        num_workers=48,
        user_prompt=None,
        local_code=False,
    )
    assert evaluator.num_workers == 48


def test_endpoint_evaluator_bypasses_local_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests go straight to the server, never through a local HTTPS_PROXY."""
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen.update(url=url, **kwargs)
        return httpx.Response(200, json={"text": "hello"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("TINY_AUDIO_API_KEY", "k")
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net")
    text, _, _ = evaluator.transcribe(
        {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}
    )
    assert text == "hello"
    assert seen["trust_env"] is False
    assert seen["headers"]["authorization"] == "Bearer k"
