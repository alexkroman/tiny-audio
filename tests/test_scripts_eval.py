"""Tests for scripts/eval package.

Note: Audio utility tests (audio_to_wav_bytes, prepare_wav_bytes, TextNormalizer)
are in test_eval_audio.py to avoid duplication.
"""

import pytest

from scripts.eval.datasets import (
    DATASET_REGISTRY,
    DatasetConfig,
)
from scripts.eval.evaluators import (
    EvalResult,
    Evaluator,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_basic_config(self):
        """Test creating a basic dataset config."""
        config = DatasetConfig(
            path="test/dataset",
            audio_field="audio",
        )

        assert config.path == "test/dataset"
        assert config.audio_field == "audio"
        assert config.text_field == "text"  # default
        assert config.default_split == "test"  # default


class TestDatasetRegistry:
    """Tests for DATASET_REGISTRY."""

    def test_registry_not_empty(self):
        """Test that the registry contains datasets."""
        assert len(DATASET_REGISTRY) > 0

    def test_loquacious_exists(self):
        """Test that loquacious dataset is in registry."""
        assert "loquacious" in DATASET_REGISTRY
        cfg = DATASET_REGISTRY["loquacious"]
        assert cfg.audio_field == "wav"
        assert cfg.text_field == "text"

    def test_all_configs_have_required_fields(self):
        """Test that all configs have required fields."""
        for name, cfg in DATASET_REGISTRY.items():
            assert cfg.path, f"Missing path for {name}"
            assert cfg.audio_field, f"Missing audio_field for {name}"


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_result(self):
        """Test creating an EvalResult."""
        result = EvalResult(
            prediction="hello world",
            reference="hello world",
            wer=0.0,
            time=1.5,
        )

        assert result.prediction == "hello world"
        assert result.reference == "hello world"
        assert result.wer == 0.0
        assert result.time == 1.5


class TestEvaluatorBase:
    """Tests for base Evaluator class."""

    def test_compute_metrics_empty(self):
        """Test compute_metrics with no results."""
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics()

        assert metrics["wer"] == 0.0
        assert metrics["avg_time"] == 0.0
        assert metrics["num_samples"] == 0

    def test_evaluator_initialization(self):
        """Test Evaluator initialization."""
        evaluator = Evaluator(audio_field="audio", text_field="text")

        assert evaluator.audio_field == "audio"
        assert evaluator.text_field == "text"
        assert evaluator.results == []

    def test_transcribe_not_implemented(self):
        """Test that base transcribe raises NotImplementedError."""
        evaluator = Evaluator()

        with pytest.raises(NotImplementedError):
            evaluator.transcribe(None)


class TestResolveLocalRuntime:
    """Tests for _resolve_local_runtime, the local-model device/dtype policy.

    The regression these guard is silent: the wrong answer here costs 2x memory
    bandwidth on every decoded token and reports nothing, because fp32 is a
    perfectly valid dtype to run in.
    """

    @staticmethod
    def _resolver():
        from scripts.eval.evaluators.asr import _resolve_local_runtime

        return _resolve_local_runtime

    def test_cuda_prefers_bfloat16(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert self._resolver()() == (0, "bfloat16")

    def test_mps_uses_bfloat16_not_float16(self, monkeypatch):
        """bf16, not fp16.

        Measured equal in speed on torch 2.8 / Metal, so fp16 buys nothing
        while giving up the exponent range -- and both submodels of this stack
        are pretrained in bf16.
        """
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert self._resolver()() == ("mps", "bfloat16")

    def test_cpu_stays_float32(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert self._resolver()() == (-1, "float32")

    def test_returns_dtype_as_string_not_torch_dtype(self, monkeypatch):
        """Must be the string ASRConfig.model_dtype wants, not a torch.dtype.

        getattr(torch, config.model_dtype) in ASRModel.__init__ would raise on a
        torch.dtype instance, so this is the contract that makes the override
        land at all.
        """
        import torch

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

    def test_dtype_kwarg_does_not_change_model_dtype(self, tmp_path):
        import torch

        from tiny_audio.asr_config import ASRConfig

        ASRConfig(model_dtype="float32").save_pretrained(tmp_path)
        cfg = ASRConfig.from_pretrained(tmp_path, dtype=torch.bfloat16)
        assert cfg.model_dtype == "float32"

    def test_model_dtype_kwarg_does_change_it(self, tmp_path):
        from tiny_audio.asr_config import ASRConfig

        ASRConfig(model_dtype="float32").save_pretrained(tmp_path)
        cfg = ASRConfig.from_pretrained(tmp_path, model_dtype="bfloat16")
        assert cfg.model_dtype == "bfloat16"


class TestStreamingRetry:
    """AssemblyAIStreamingEvaluator retries only on transient stream close codes."""

    @pytest.fixture
    def evaluator(self, monkeypatch):
        from scripts.eval.evaluators.asr import AssemblyAIStreamingEvaluator

        # tenacity sleeps through time.sleep; skip the backoff in tests.
        monkeypatch.setattr("tenacity.nap.time.sleep", lambda _s: None)
        return AssemblyAIStreamingEvaluator(api_key="test")

    @staticmethod
    def _stream_error(code: int | None) -> RuntimeError:
        exc = RuntimeError(f"closed with {code}")
        if code is not None:
            exc.streaming_code = code  # type: ignore[attr-defined]
        return exc

    def test_transient_close_code_is_retried(self, evaluator, monkeypatch):
        calls = []

        def run_session(_pcm):
            calls.append(1)
            if len(calls) < 3:
                raise self._stream_error(1013)
            return "hello", 0.5

        monkeypatch.setattr(evaluator, "_prepare_pcm", lambda _a: b"")
        monkeypatch.setattr(evaluator, "_run_session", run_session)

        assert evaluator.transcribe(object()) == ("hello", 0.5, None)
        assert len(calls) == 3

    def test_non_retryable_error_is_raised_immediately(self, evaluator, monkeypatch):
        calls = []

        def run_session(_pcm):
            calls.append(1)
            raise self._stream_error(None)

        monkeypatch.setattr(evaluator, "_prepare_pcm", lambda _a: b"")
        monkeypatch.setattr(evaluator, "_run_session", run_session)

        with pytest.raises(RuntimeError, match="closed with None"):
            evaluator.transcribe(object())
        assert len(calls) == 1

    def test_gives_up_after_max_retries(self, evaluator, monkeypatch):
        from scripts.eval.evaluators.asr import AssemblyAIStreamingEvaluator

        calls = []

        def run_session(_pcm):
            calls.append(1)
            raise self._stream_error(4029)

        monkeypatch.setattr(evaluator, "_prepare_pcm", lambda _a: b"")
        monkeypatch.setattr(evaluator, "_run_session", run_session)

        with pytest.raises(RuntimeError, match="closed with 4029"):
            evaluator.transcribe(object())
        assert len(calls) == AssemblyAIStreamingEvaluator._MAX_RETRIES + 1
