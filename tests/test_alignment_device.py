"""Device selection and input handling for ForcedAligner.align."""

import numpy as np
import pytest
import torch

from tiny_audio import alignment
from tiny_audio.alignment import ForcedAligner


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_get_device_prefers_cuda_then_mps(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert alignment._get_device() == expected


class TestAlignInputs:
    """`align` accepts numpy or torch audio and resamples off-rate input."""

    LABELS = ("-", "|", "H", "I")

    @pytest.fixture
    def fake_aligner(self, monkeypatch):
        """Patch the singleton with a model emitting a fixed log-prob matrix."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        seen = {}
        num_frames = 6
        # Emit blank, then H, then I, then blanks -- one clear path.
        emission = torch.full((1, num_frames, len(self.LABELS)), -5.0)
        emission[0, :, 0] = -1.0
        emission[0, 1, 2] = 0.0
        emission[0, 3, 3] = 0.0

        def fake_model(waveform):
            seen["waveform"] = waveform
            return emission, None

        dictionary = {c: i for i, c in enumerate(self.LABELS)}
        monkeypatch.setattr(
            ForcedAligner,
            "get_instance",
            classmethod(lambda cls, device="cuda": (fake_model, list(self.LABELS), dictionary)),
        )
        bundle = type("Bundle", (), {"sample_rate": 16000})()
        monkeypatch.setattr(ForcedAligner, "_bundle", bundle)
        return seen

    def test_numpy_input_is_not_mutated_and_is_batched(self, fake_aligner):
        audio = np.zeros(1600, dtype=np.float32)
        words = ForcedAligner.align(audio, "hi")
        assert fake_aligner["waveform"].shape == (1, 1600)
        assert [w["word"] for w in words] == ["hi"]
        assert words[0]["end"] > words[0]["start"] >= 0.0

    def test_torch_input_is_accepted(self, fake_aligner):
        words = ForcedAligner.align(torch.zeros(2, 800), "hi")
        assert fake_aligner["waveform"].shape == (2, 800)
        assert [w["word"] for w in words] == ["hi"]

    def test_off_rate_audio_is_resampled(self, fake_aligner):
        ForcedAligner.align(np.zeros(800, dtype=np.float32), "hi", sample_rate=8000)
        assert fake_aligner["waveform"].shape == (1, 1600)

    def test_unrepresentable_text_returns_nothing(self, fake_aligner):
        assert ForcedAligner.align(np.zeros(160, dtype=np.float32), "123 ...") == []
