"""The startup guard that stops a run from silently training without fused CE.

Two `logger.warning` calls already covered this case and both were missed in
practice: the warning scrolls past during model load and the run then trains
normally, just ~25 GiB heavier and at permanent OOM risk. The failure has no
symptom until the batch that does not fit. These tests pin the escalation to
a hard error, and -- just as importantly -- pin the escape hatches, so a mac
smoke run or a deliberate unfused run is never blocked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.train import _require_fused_cross_entropy


def _model(accepts: bool, vocab: int = 248320, name: str = "Qwen3_5ForCausalLM"):
    """Minimal stand-in: the guard reads only these three attributes."""
    lm_cls = type(name, (), {})
    lm = lm_cls()
    lm.config = SimpleNamespace(vocab_size=vocab)
    return SimpleNamespace(_lm_accepts_skip_logits=accepts, language_model=lm)


def _cfg(**training):
    base = {"use_liger": True, "per_device_train_batch_size": 48}
    base.update(training)
    return SimpleNamespace(training=base)


class TestRaisesWhereItCosts:
    def test_raises_on_cuda_with_a_large_vocab(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(RuntimeError, match="fused linear cross-entropy is NOT active"),
        ):
            _require_fused_cross_entropy(_model(False), _cfg())

    def test_the_message_is_actionable(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(RuntimeError) as exc,
        ):
            _require_fused_cross_entropy(_model(False), _cfg())
        msg = str(exc.value)
        assert "poetry install" in msg
        assert "liger-kernel >=0.8.0" in msg
        assert "allow_unfused_ce" in msg
        assert "248,320" in msg


class TestEscapeHatches:
    def test_silent_when_fused_ce_is_active(self):
        with patch("torch.cuda.is_available", return_value=True):
            _require_fused_cross_entropy(_model(True), _cfg())

    def test_silent_off_cuda(self):
        """mac / CPU smoke runs: liger is a linux-only dependency."""
        with patch("torch.cuda.is_available", return_value=False):
            _require_fused_cross_entropy(_model(False), _cfg())

    def test_silent_when_liger_is_deliberately_off(self):
        with patch("torch.cuda.is_available", return_value=True):
            _require_fused_cross_entropy(_model(False), _cfg(use_liger=False))

    def test_allow_unfused_ce_overrides(self):
        with patch("torch.cuda.is_available", return_value=True):
            _require_fused_cross_entropy(_model(False), _cfg(allow_unfused_ce=True))

    def test_silent_on_a_small_vocab(self):
        """The logits tensor is not the dominant term below ~100k."""
        with patch("torch.cuda.is_available", return_value=True):
            _require_fused_cross_entropy(_model(False, vocab=49152), _cfg())
