"""Tests for scripts/eval package.

Note: Audio utility tests (audio_to_wav_bytes, prepare_wav_bytes, TextNormalizer)
are in test_eval_audio.py to avoid duplication.
"""

import base64
import io
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import assemblyai as aai
import httpx
import numpy as np
import numpy.typing as npt
import pytest
import soundfile
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from scripts.eval.cli import _build_tiny_audio_evaluator
from scripts.eval.evaluators.asr import (
    DTYPE_CONFIG_FIELDS,
    AssemblyAIStreamingEvaluator,
    AssemblyAIStreamsEvaluator,
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
        """Return `_resolve_local_runtime` typed as a no-arg (device, dtype) callable."""
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
        """Build a stand-in ASRModel whose `language_model` is a LoRA-wrapped tiny decoder."""

        class Tiny(PreTrainedModel):
            """A minimal PreTrainedModel with one 4x4 linear layer for LoRA to target."""

            config_class = PretrainedConfig

            def __init__(self) -> None:
                """Build the model on a default config with its `lin` layer."""
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
        """Fake a loaded model whose decoder used `loaded_impl`; also return the set-calls log.

        The log records every attention implementation the code under test applies.
        """
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
        """Make the test run as if MPS were available."""
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    @staticmethod
    def _set_sliding_window(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
        """Patch the sliding-window check to report `value` for any decoder."""

        def has_sliding_window(_model_id: str) -> bool:
            return value

        monkeypatch.setattr(
            "tiny_audio.asr_attention._has_sliding_window_attention", has_sliding_window
        )

    @pytest.mark.usefixtures("on_mps")
    def test_corrects_a_stale_eager_decoder_to_sdpa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_sliding_window(monkeypatch, False)
        holder, calls = self._holder("eager")

        _use_sdpa_where_safe(cast(ASRModel, holder))

        assert calls == ["sdpa"]

    @pytest.mark.usefixtures("on_mps")
    def test_leaves_a_sliding_window_model_on_eager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Metal's sdpa returns wrong results for cached decode against that mask."""
        self._set_sliding_window(monkeypatch, True)
        holder, calls = self._holder("eager")

        _use_sdpa_where_safe(cast(ASRModel, holder))

        assert calls == []

    @pytest.mark.usefixtures("on_mps")
    def test_is_a_noop_when_load_already_resolved_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--local-code already gets this right; re-applying must not churn the model."""
        self._set_sliding_window(monkeypatch, False)
        holder, calls = self._holder("sdpa")

        _use_sdpa_where_safe(cast(ASRModel, holder))

        assert calls == []


def _no_pcm(_a: object) -> bytes:
    """Stand in for `_prepare_pcm`, skipping audio conversion."""
    return b""


@pytest.mark.usefixtures("no_retry_backoff")
class TestStreamingRetry:
    """AssemblyAIStreamingEvaluator retries only on transient stream close codes."""

    @pytest.fixture
    def evaluator(self, monkeypatch: pytest.MonkeyPatch) -> AssemblyAIStreamingEvaluator:
        """Provide a streaming evaluator whose PCM preparation is a no-op."""
        evaluator = AssemblyAIStreamingEvaluator(api_key="test")
        monkeypatch.setattr(evaluator, "_prepare_pcm", _no_pcm)
        return evaluator

    @staticmethod
    def _stream_error(code: int | None) -> RuntimeError:
        """Build a stream-closed error carrying `code`, or no code when None."""
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

    def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        seen.update(url=url, **kwargs)
        return httpx.Response(200, json={"text": "hello"}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake_request)
    monkeypatch.setenv("TINY_AUDIO_API_KEY", "k")
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net")
    text, _, _ = evaluator.transcribe(
        {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}
    )
    assert text == "hello"
    assert seen["trust_env"] is False
    assert seen["headers"]["authorization"] == "Bearer k"


@pytest.mark.usefixtures("no_retry_backoff")
def test_endpoint_evaluator_retries_proxy_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy 502 or dropped connection is retried, not scored as an empty prediction."""
    outcomes: list[Any] = [
        httpx.ConnectError("tunnel dropped"),
        502,
        524,
        200,
    ]

    def fake_request(method: str, url: str, **_kwargs: Any) -> httpx.Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, json={"text": "hello"}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake_request)
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net")
    text, _, _ = evaluator.transcribe(
        {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}
    )
    assert text == "hello"
    assert outcomes == []


@pytest.mark.usefixtures("no_retry_backoff")
def test_endpoint_evaluator_does_not_retry_client_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 401 (bad API key) fails at once instead of burning the backoff budget."""
    calls = 0

    def fake_request(method: str, url: str, **_kwargs: Any) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401, json={"error": "unauthorized"}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "request", fake_request)
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net")
    with pytest.raises(httpx.HTTPStatusError):
        evaluator.transcribe({"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000})
    assert calls == 1


def _fake_job_server(
    result: dict[str, Any], polls_running: int, seen: list[tuple[str, str, dict[str, Any]]]
) -> Callable[..., httpx.Response]:
    """A `ta serve` job endpoint: 202 with an id, `polls_running` 202 polls, then `result`."""
    remaining = polls_running

    def fake_request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        nonlocal remaining
        seen.append((method, url, kwargs))
        request = httpx.Request(method, url)
        if method == "POST":
            return httpx.Response(202, json={"id": "j1", "status": "running"}, request=request)
        if remaining:
            remaining -= 1
            return httpx.Response(202, json={"status": "running"}, request=request)
        return httpx.Response(200, json=result, request=request)

    return fake_request


def test_endpoint_evaluator_runs_speaker_datasets_as_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A speaker dataset goes to /jobs with `return_speakers`, is polled, and scores as turns."""
    words = [
        {"word": "hi", "speaker": "SPEAKER_1"},
        {"word": "there", "speaker": "SPEAKER_1"},
        {"word": "hello", "speaker": "SPEAKER_0"},
    ]
    seen: list[tuple[str, str, dict[str, Any]]] = []
    fake = _fake_job_server({"text": "hi there hello", "words": words}, 2, seen)
    monkeypatch.setattr(httpx, "request", fake)
    monkeypatch.setattr(EndpointEvaluator, "POLL_SECONDS", 0.0)
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net/transcribe")
    evaluator.speakers = True
    text, _, _ = evaluator.transcribe(
        {"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}
    )
    assert text == "<SPK_1>hi there<SPK_2>hello"
    assert [(m, u) for m, u, _ in seen] == [
        ("POST", "https://pod-8000.proxy.runpod.net/jobs"),
        *[("GET", "https://pod-8000.proxy.runpod.net/jobs/j1")] * 3,
    ]
    assert seen[0][2]["params"] == {"return_speakers": "true"}


def test_endpoint_evaluator_raises_on_a_diarization_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed diarization is raised, not scored as an unlabelled transcript."""
    seen: list[tuple[str, str, dict[str, Any]]] = []
    fake = _fake_job_server({"text": "hi", "words": [], "diarization_error": "boom"}, 0, seen)
    monkeypatch.setattr(httpx, "request", fake)
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net")
    evaluator.speakers = True
    with pytest.raises(RuntimeError, match="diarization_error"):
        evaluator.transcribe({"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000})


def test_endpoint_evaluator_diarize_decodes_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    """`diarize` runs a `diarize_only` job and decodes the float16 activity it returns."""
    activity = np.array([[0.9, 0.0], [0.25, 0.5]], dtype=np.float32)
    result = {
        "activity": base64.b64encode(activity.astype("<f2").tobytes()).decode(),
        "shape": [2, 2],
        "keep": [0, 1],
    }
    seen: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(httpx, "request", _fake_job_server(result, 0, seen))
    evaluator = EndpointEvaluator(endpoint_url="https://pod-8000.proxy.runpod.net")
    got, keep = evaluator.diarize(b"RIFF")
    np.testing.assert_allclose(got, activity, atol=1e-3)
    assert got.dtype == np.float32
    assert keep == [0, 1]
    assert seen[0][2]["params"] == {"diarize_only": "true"}


def _word(text: str, start_s: float, end_s: float) -> types.SimpleNamespace:
    """Build an AssemblyAI-style word with times converted to milliseconds."""
    return types.SimpleNamespace(text=text, start=int(start_s * 1000), end=int(end_s * 1000))


class _StreamTranscriber:
    """Stands in for `aai.Transcriber`: answers by which half of the stream is unmasked.

    Speaker A talks over 0-1.1 s, speaker B over 0.9-2 s. A's stream hears a
    word in B's region that is really on its own silence ("ghost"), and both
    streams hear "shared" in the overlap.
    """

    def __init__(self) -> None:
        """Start with no recorded streams."""
        self.streams: list[npt.NDArray[np.float32]] = []

    def transcribe(self, data: io.BytesIO) -> types.SimpleNamespace:
        """Record the decoded stream and return A's or B's words, by whether 0.5 s is unmasked."""
        audio = np.asarray(soundfile.read(data, dtype="float32")[0], dtype=np.float32)
        self.streams.append(audio)
        if audio[8000] != 0:  # 0.5 s: only A's stream keeps it
            words = [_word("hello", 0.1, 0.5), _word("shared", 0.95, 1.05)]
            words.append(_word("ghost", 1.5, 1.8))
        else:
            words = [_word("shared", 0.96, 1.04), _word("world", 1.3, 1.7)]
        return types.SimpleNamespace(status=aai.TranscriptStatus.completed, words=words, error=None)


def _streams_evaluator(
    monkeypatch: pytest.MonkeyPatch, activity: npt.NDArray[np.float32], keep: list[int]
) -> tuple[AssemblyAIStreamsEvaluator, _StreamTranscriber]:
    """Build a speaker-mode streams evaluator with a fake transcriber and a fixed diarizer.

    The diarizer returns `activity` and `keep` for any audio.
    """
    evaluator = AssemblyAIStreamsEvaluator(
        api_key="k", diarizer_url="https://pod-8000.proxy.runpod.net", model="universal-3-5-pro"
    )
    fake = _StreamTranscriber()
    evaluator.transcriber = cast(aai.Transcriber, fake)

    def diarize(_wav: bytes) -> tuple[npt.NDArray[np.float32], list[int]]:
        return activity, keep

    monkeypatch.setattr(evaluator.diarizer, "diarize", diarize)
    evaluator.speakers = True
    return evaluator, fake


def test_assemblyai_streams_mask_filter_and_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    """One AssemblyAI call per speaker stream; silence words dropped, the overlap kept once."""
    activity = np.zeros((200, 8), dtype=np.float32)
    activity[:110, 0] = 0.9  # A: 0-1.1 s
    activity[90:, 1] = 0.6  # B: 0.9-2 s
    evaluator, fake = _streams_evaluator(monkeypatch, activity, [0, 1])
    text, _, _ = evaluator.transcribe(
        {"array": np.ones(32000, dtype=np.float32) * 0.5, "sampling_rate": 16000}
    )
    # "shared" goes to A, the more active speaker over it; "ghost" sat on A's silence.
    assert text == "<SPK_1>hello shared<SPK_2>world"
    assert len(fake.streams) == 2
    a_stream = next(s for s in fake.streams if s[8000] != 0)
    assert len(a_stream) == 32000  # full length: timestamps stay on the recording's timeline
    assert not a_stream[int(1.2 * 16000) :].any()  # B's solo region is zeroed


def test_assemblyai_streams_one_speaker_is_unmasked(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one speaker the stream is the whole recording and nothing is filtered."""
    activity = np.zeros((200, 8), dtype=np.float32)
    activity[:110, 0] = 0.9
    evaluator, fake = _streams_evaluator(monkeypatch, activity, [0])
    text, _, _ = evaluator.transcribe(
        {"array": np.ones(32000, dtype=np.float32) * 0.5, "sampling_rate": 16000}
    )
    assert text == "<SPK_1>hello shared ghost"
    assert len(fake.streams) == 1
    assert fake.streams[0].all()
