"""Tests for QwenForcedAligner with the model and processor faked out."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
import torch

from tiny_audio import alignment
from tiny_audio.alignment import QwenForcedAligner


class FakeProcessor:
    """Times each word as [i, i + 0.5) seconds and records the batches it saw."""

    def __init__(self) -> None:
        self.batches: list[tuple[list[npt.NDArray[np.float32]], list[str]]] = []

    def prepare_forced_aligner_inputs(
        self, audios: list[npt.NDArray[np.float32]], texts: list[str], **_: Any
    ) -> tuple[dict[str, torch.Tensor], list[list[str]]]:
        self.batches.append((audios, texts))
        features = {
            "input_ids": torch.zeros(len(audios), 3, dtype=torch.long),
            "input_features": torch.zeros(len(audios), 4),
        }
        return features, [t.split() for t in texts]

    def decode_forced_alignment(
        self, logits: Any, input_ids: Any, word_lists: list[list[str]], timestamp_id: int
    ) -> list[list[dict[str, float]]]:
        return [
            [{"start_time": float(i), "end_time": i + 0.5} for i in range(len(words))]
            for words in word_lists
        ]


@pytest.fixture
def fake_pair(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, FakeProcessor]:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.config.timestamp_token_id = 7
    processor = FakeProcessor()

    def get_instance(_: object) -> tuple[MagicMock, FakeProcessor]:
        return model, processor

    monkeypatch.setattr(QwenForcedAligner, "get_instance", classmethod(get_instance))
    return model, processor


class TestTo16k:
    def test_flattens_and_casts(self) -> None:
        out = alignment.to_16k(np.ones((2, 8), dtype=np.float64), 16000)
        assert out.dtype == np.float32
        assert out.shape == (16,)

    def test_accepts_tensor_and_resamples(self) -> None:
        out = alignment.to_16k(torch.zeros(8000), 8000)
        assert out.dtype == np.float32
        assert out.shape == (16000,)


class TestAlignChunks:
    def test_keeps_original_words_and_skips_punctuation_only(
        self, fake_pair: tuple[MagicMock, FakeProcessor]
    ) -> None:
        _, processor = fake_pair
        words = QwenForcedAligner.align(np.zeros(1600), "Hello, -- World!")
        assert words == [
            {"word": "Hello,", "start": 0.0, "end": 0.5},
            {"word": "World!", "start": 1.0, "end": 1.5},
        ]
        # The aligner is given the kept words only, with punctuation intact here
        # (the processor itself strips it).
        assert processor.batches[0][1] == ["Hello, World!"]

    def test_feature_dtypes_follow_model(self, fake_pair: tuple[MagicMock, FakeProcessor]) -> None:
        model, _ = fake_pair
        model.dtype = torch.float64
        QwenForcedAligner.align(np.zeros(160), "hi")
        kwargs = model.call_args.kwargs
        assert kwargs["input_features"].dtype == torch.float64
        assert kwargs["input_ids"].dtype == torch.long

    def test_empty_text_never_loads_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_load(_: Any) -> None:
            msg = "model should not load"
            raise AssertionError(msg)

        monkeypatch.setattr(QwenForcedAligner, "get_instance", classmethod(no_load))
        assert QwenForcedAligner.align_chunks([(np.zeros(10), ""), (np.zeros(10), "-- ...")]) == [
            [],
            [],
        ]

    def test_too_long_chunk_raises(self, fake_pair: tuple[MagicMock, FakeProcessor]) -> None:
        too_long = np.zeros(int(16000 * (QwenForcedAligner.MAX_SECONDS + 1)), dtype=np.float32)
        with pytest.raises(ValueError, match="at most 300 s"):
            QwenForcedAligner.align_chunks([(np.zeros(10), "ok"), (too_long, "too long")])

    def test_batches_and_preserves_chunk_order(
        self, fake_pair: tuple[MagicMock, FakeProcessor], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, processor = fake_pair
        monkeypatch.setattr(QwenForcedAligner, "BATCH_SIZE", 2)
        chunks = [(np.zeros(10), f"w{i}") for i in range(3)]
        chunks.insert(1, (np.zeros(10), ""))
        results = QwenForcedAligner.align_chunks(chunks)
        assert [[w["word"] for w in r] for r in results] == [["w0"], [], ["w1"], ["w2"]]
        assert [texts for _, texts in processor.batches] == [["w0", "w1"], ["w2"]]

    def test_missing_transformers_support_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alignment, "_QWEN3_ASR_IMPORT_ERROR", ImportError("too old"))
        with pytest.raises(ImportError, match="too old"):
            QwenForcedAligner.align(np.zeros(10), "hi")


class TestGetInstance:
    @pytest.fixture(autouse=True)
    def reset_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(QwenForcedAligner, "_model", None)
        monkeypatch.setattr(QwenForcedAligner, "_processor", None)

    @pytest.mark.parametrize(("device", "attn"), [("cpu", "sdpa"), ("mps", "eager")])
    def test_loads_once_with_device_specific_attention(
        self, monkeypatch: pytest.MonkeyPatch, device: str, attn: str
    ) -> None:
        model_cls, processor_cls = MagicMock(), MagicMock()
        monkeypatch.setattr(alignment, "Qwen3ASRForTokenClassification", model_cls)
        monkeypatch.setattr(alignment, "Qwen3ASRProcessor", processor_cls)
        monkeypatch.setattr(alignment, "get_device", lambda: torch.device(device))

        first = QwenForcedAligner.get_instance()
        second = QwenForcedAligner.get_instance()

        assert first == second
        model_cls.from_pretrained.assert_called_once_with(
            QwenForcedAligner.MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn
        )
        model = model_cls.from_pretrained.return_value
        model.to.assert_called_once_with(torch.device(device))
        model.eval.assert_called_once()
        processor_cls.from_pretrained.assert_called_once_with(QwenForcedAligner.MODEL_ID)

    def test_import_error_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(alignment, "_QWEN3_ASR_IMPORT_ERROR", ImportError("needs 5.17"))
        with pytest.raises(ImportError, match=r"needs 5\.17"):
            QwenForcedAligner.get_instance()
