"""Inference-time lead-in silence (tiny_audio/asr_processing.prepend_lead_in).

Measured on 500 Peoples clips, paired bootstrap: 20.51% -> 19.28% WER
(delta -1.22, CI [-1.83, -0.64]); CommonVoice +0.30, CI [-0.43, +1.17] (ns).
"""

import numpy as np
import pytest

from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_processing import prepend_lead_in


class TestPrependLeadIn:
    def test_prepends_exactly_the_requested_silence(self):
        audio = np.ones(1600, dtype=np.float32)
        out = prepend_lead_in(audio, 16000, 0.25)
        assert len(out) == 1600 + 4000
        assert np.all(out[:4000] == 0)
        assert np.all(out[4000:] == 1)

    def test_scales_with_sampling_rate(self):
        audio = np.ones(100, dtype=np.float32)
        assert len(prepend_lead_in(audio, 8000, 0.25)) == 100 + 2000

    @pytest.mark.parametrize("seconds", [0.0, -1.0, None])
    def test_disabled_is_an_exact_passthrough(self, seconds):
        audio = np.ones(1600, dtype=np.float32)
        out = prepend_lead_in(audio, 16000, seconds)
        assert out is audio, "disabled must not copy or alter the waveform"

    def test_batch_of_waveforms(self):
        batch = [np.ones(1600, dtype=np.float32), np.ones(800, dtype=np.float32)]
        out = prepend_lead_in(batch, 16000, 0.25)
        assert [len(x) for x in out] == [5600, 4800]

    def test_dtype_is_preserved(self):
        for dt in (np.float32, np.float64):
            out = prepend_lead_in(np.ones(10, dtype=dt), 16000, 0.1)
            assert out.dtype == dt


class TestConfigField:
    def test_default_is_the_measured_value(self):
        assert ASRConfig().inference_lead_in_seconds == 0.25

    def test_can_be_disabled(self):
        assert ASRConfig(inference_lead_in_seconds=0.0).inference_lead_in_seconds == 0.0

    def test_survives_json_round_trip(self):
        import json

        cfg = ASRConfig(inference_lead_in_seconds=0.4)
        restored = ASRConfig.from_dict(json.loads(cfg.to_json_string()))
        assert restored.inference_lead_in_seconds == 0.4


class TestPipelineIntegration:
    """The pipeline must apply a real config value and ignore a stubbed one."""

    def _fake_pipeline(self, lead_in):
        import types

        from tiny_audio.asr_pipeline import ASRPipeline

        pipe = ASRPipeline.__new__(ASRPipeline)
        pipe.model = types.SimpleNamespace(
            config=types.SimpleNamespace(inference_lead_in_seconds=lead_in)
        )
        pipe.feature_extractor = types.SimpleNamespace(sampling_rate=16000)
        return pipe

    def _captured(self, pipe, audio, monkeypatch):
        import transformers

        seen = {}

        def fake_preprocess(self, inputs, **params):
            seen["inputs"] = inputs
            yield {"input_features": None}

        monkeypatch.setattr(
            transformers.AutomaticSpeechRecognitionPipeline, "preprocess", fake_preprocess
        )
        list(pipe.preprocess({"array": audio}))
        return seen["inputs"]

    def test_real_config_value_is_applied(self, monkeypatch):
        audio = np.ones(160, dtype=np.float32)
        got = self._captured(self._fake_pipeline(0.25), audio, monkeypatch)
        assert len(got["raw"]) == 160 + 4000

    def test_zero_is_a_passthrough(self, monkeypatch):
        audio = np.ones(160, dtype=np.float32)
        got = self._captured(self._fake_pipeline(0.0), audio, monkeypatch)
        assert got["raw"] is audio

    def test_non_numeric_config_is_ignored(self, monkeypatch):
        """A MagicMock coerces to 1.0 under `float()` -- a full second of silence."""
        from unittest.mock import MagicMock

        audio = np.ones(160, dtype=np.float32)
        got = self._captured(self._fake_pipeline(MagicMock()), audio, monkeypatch)
        assert got["raw"] is audio
