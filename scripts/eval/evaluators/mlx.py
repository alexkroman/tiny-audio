"""`ta eval --mlx`: a tiny_audio checkpoint transcribed with MLX on Apple silicon."""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Unpack

import numpy as np

from scripts.eval.audio import as_16k_array

from .base import Evaluator, EvaluatorOptions, Transcription, console

if TYPE_CHECKING:
    from scripts.mlx_inference import MLXTranscriber


class MLXEvaluator(Evaluator):
    """Transcribe each clip with `MLXTranscriber`: the latest Hub revision, converted once."""

    def __init__(self, model_path: str, **kwargs: Unpack[EvaluatorOptions]) -> None:
        """Load `model_path` (Hub repo or local checkpoint) for MLX; one worker.

        MLX is imported here, not at module level, so `ta eval` and its
        evaluator registry still import on Linux, where MLX does not exist.
        """
        # One model, decoded one clip at a time on the GPU's single command
        # queue: threads would only contend for it.
        kwargs["num_workers"] = 1
        super().__init__(**kwargs)
        transcriber_cls: type[MLXTranscriber] = importlib.import_module(
            "scripts.mlx_inference"
        ).MLXTranscriber
        self.model: MLXTranscriber = transcriber_cls.from_pretrained(model_path)
        revision = f" @ {self.model.revision[:12]}" if self.model.revision else ""
        console.print(f"[dim]MLX model: {model_path}{revision}[/dim]")

    def transcribe(self, audio: object) -> Transcription:
        """One clip at 16 kHz to text; speaker-attributed datasets are not supported."""
        if self.speakers:
            msg = "--mlx transcribes without speakers; drop the speaker-attributed dataset"
            raise NotImplementedError(msg)
        array = np.asarray(as_16k_array(audio), dtype=np.float32)
        start = time.time()
        text = self.model.transcribe(array)
        return text, time.time() - start, None
