"""Forced alignment for word-level timestamps with Qwen3-ForcedAligner."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor

if TYPE_CHECKING:
    from transformers import Qwen3ASRForTokenClassification, Qwen3ASRProcessor
    from transformers.models.qwen3_asr.processing_qwen3_asr import _clean_tokens

    _QWEN3_ASR_IMPORT_ERROR: ImportError | None = None
else:
    # Qwen3-ForcedAligner needs transformers >= 5.17; keep this module importable
    # on older versions and only fail when alignment is actually requested.
    try:
        from transformers import Qwen3ASRForTokenClassification, Qwen3ASRProcessor
        from transformers.models.qwen3_asr.processing_qwen3_asr import _clean_tokens

        _QWEN3_ASR_IMPORT_ERROR = None
    except ImportError as e:
        _QWEN3_ASR_IMPORT_ERROR = e


def _get_device() -> str:
    """Get best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _module_to(module: torch.nn.Module, device: str) -> None:
    """`module.to(device)`, in place.

    `PreTrainedModel.to` is wrapped with `functools.wraps`, which type checkers
    can't bind as a method; typed as `nn.Module` the call resolves. Same method
    at runtime: it moves the module in place and returns it.
    """
    module.to(device)


class QwenForcedAligner:
    """Word timestamps from Qwen3-ForcedAligner-0.6B (transformers >= 5.17).

    Trained for timestamping rather than repurposed from CTC ASR, it aligns
    cased, punctuated text with numbers directly. One forward pass covers up to
    5 minutes of speech, so long recordings are aligned as chunks, each against
    its own transcript (`align_chunks`), batched.

    Returns `[{"word", "start", "end"}]` for every word it can align, in order,
    with the ORIGINAL word (punctuation, casing) as "word". The aligner itself
    sees words with punctuation stripped and drops any that strip to nothing
    ("--"); the split is mirrored word by word so the pairing stays exact.
    """

    MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
    MAX_SECONDS = 300.0
    BATCH_SIZE = 8
    _model: Qwen3ASRForTokenClassification | None = None
    _processor: Qwen3ASRProcessor | None = None

    @classmethod
    def get_instance(cls) -> tuple[Qwen3ASRForTokenClassification, Qwen3ASRProcessor]:
        """Load the aligner model and processor once, then return the cached pair."""
        if cls._model is None or cls._processor is None:
            if _QWEN3_ASR_IMPORT_ERROR is not None:
                raise _QWEN3_ASR_IMPORT_ERROR
            device = _get_device()
            # Batches are padded, and Metal sdpa returns NaN for fully masked rows.
            attn = "eager" if device == "mps" else "sdpa"
            model = Qwen3ASRForTokenClassification.from_pretrained(
                cls.MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn
            )
            _module_to(model, device)
            cls._model = model.eval()
            cls._processor = cast(Qwen3ASRProcessor, AutoProcessor.from_pretrained(cls.MODEL_ID))
        return cls._model, cls._processor

    @staticmethod
    def _to_16k(audio, sample_rate: int) -> np.ndarray:
        """Flatten `audio` to a float32 mono array resampled to 16 kHz."""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sample_rate, 16000
            ).numpy()
        return audio

    @classmethod
    @torch.inference_mode()
    def align_chunks(
        cls,
        chunks: list[tuple[np.ndarray, str]],
        sample_rate: int = 16000,
        language: str = "English",
    ) -> list[list[dict]]:
        """Align each `(audio, text)` pair (each <= 5 min); one word list per pair."""
        if _QWEN3_ASR_IMPORT_ERROR is not None:
            raise _QWEN3_ASR_IMPORT_ERROR
        results: list[list[dict]] = [[] for _ in chunks]
        todo = []
        for i, (raw, text) in enumerate(chunks):
            kept = [w for w in text.split() if _clean_tokens([w])]
            if not kept:
                continue
            audio = cls._to_16k(raw, sample_rate)
            if len(audio) / 16000 > cls.MAX_SECONDS:
                msg = (
                    f"chunk {i} is {len(audio) / 16000:.0f} s; Qwen3-ForcedAligner takes at most "
                    f"{cls.MAX_SECONDS:.0f} s -- cut the audio first"
                )
                raise ValueError(msg)
            todo.append((i, audio, kept))
        if not todo:  # nothing to time: don't load the model
            return results
        model, processor = cls.get_instance()
        for b in range(0, len(todo), cls.BATCH_SIZE):
            batch = todo[b : b + cls.BATCH_SIZE]
            features, word_lists = processor.prepare_forced_aligner_inputs(
                [audio for _, audio, _ in batch],
                [" ".join(kept) for _, _, kept in batch],
                language=language,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            inputs = {
                k: (
                    v.to(model.device, dtype=model.dtype)
                    if v.is_floating_point()
                    else v.to(model.device)
                )
                for k, v in features.items()
            }
            logits = model(**inputs).logits
            timed = processor.decode_forced_alignment(
                logits, inputs["input_ids"], word_lists, model.config.timestamp_token_id
            )
            for (i, _, kept), words in zip(batch, timed, strict=True):
                results[i] = [
                    {"word": w, "start": float(t["start_time"]), "end": float(t["end_time"])}
                    for w, t in zip(kept, words, strict=True)
                ]
        return results

    @classmethod
    def align(cls, audio, text: str, sample_rate: int = 16000, language: str = "English"):
        """Align one clip of at most 5 minutes; `[{"word", "start", "end"}]`."""
        return cls.align_chunks([(audio, text)], sample_rate, language)[0]
