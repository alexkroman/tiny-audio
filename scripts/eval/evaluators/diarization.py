"""Diarization evaluator implementations."""

import io
import os
import time

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from scripts.eval.audio import prepare_wav_bytes

from .base import DiarizationResult, setup_assemblyai


class DiarizationEvaluator:
    """Evaluator for speaker diarization using pyannote DER metric."""

    def __init__(
        self,
        audio_field: str = "audio",
        speakers_field: str = "speakers",
        timestamps_start_field: str = "timestamps_start",
        timestamps_end_field: str = "timestamps_end",
        hf_token: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        self.audio_field = audio_field
        self.speakers_field = speakers_field
        self.timestamps_start_field = timestamps_start_field
        self.timestamps_end_field = timestamps_end_field
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.results: list[DiarizationResult] = []
        self.metric = DiarizationErrorRate()

    def _build_reference_annotation(self, sample: dict) -> Annotation:
        """Build pyannote Annotation from dataset sample."""
        annotation = Annotation()
        speakers = sample[self.speakers_field]
        starts = sample[self.timestamps_start_field]
        ends = sample[self.timestamps_end_field]

        for speaker, start, end in zip(speakers, starts, ends):
            annotation[Segment(start, end)] = speaker

        return annotation

    def _build_hypothesis_annotation(self, segments: list[dict]) -> Annotation:
        """Build pyannote Annotation from diarization output."""
        annotation = Annotation()
        for seg in segments:
            annotation[Segment(seg["start"], seg["end"])] = seg["speaker"]
        return annotation

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run diarization on audio and return (segments, inference_time)."""
        import io

        import librosa

        from src.asr_pipeline import SpeakerDiarizer

        # Prepare audio array - handle both decoded and raw bytes formats
        if isinstance(audio, dict):
            if "array" in audio:
                # Already decoded
                audio_array = audio["array"]
                sample_rate = audio.get("sampling_rate", 16000)
            elif "bytes" in audio:
                # Raw bytes - decode with librosa (avoids torchcodec CPU issues)
                audio_array, sample_rate = librosa.load(io.BytesIO(audio["bytes"]), sr=16000)
            else:
                raise ValueError(f"Unsupported audio dict format: {audio.keys()}")
        else:
            raise ValueError(f"Unsupported audio format: {type(audio)}")

        # Ensure float32 dtype (pyannote requires consistent dtype)
        if hasattr(audio_array, "astype"):
            audio_array = audio_array.astype(np.float32)

        start = time.time()
        segments = SpeakerDiarizer.diarize(
            audio_array,
            sample_rate=sample_rate,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            hf_token=self.hf_token,
        )
        elapsed = time.time() - start

        return segments, elapsed

    def evaluate(self, dataset, max_samples: int | None = None) -> list[DiarizationResult]:
        """Run diarization evaluation loop on dataset."""
        self.results = []
        processed = 0

        for sample in dataset:
            processed += 1
            if max_samples and processed > max_samples:
                break

            try:
                # Get reference annotation
                reference = self._build_reference_annotation(sample)

                # Build UEM (evaluation region) from reference extent
                uem = Timeline([reference.get_timeline().extent()])

                # Run diarization
                segments, inference_time = self.diarize(sample[self.audio_field])
                hypothesis = self._build_hypothesis_annotation(segments)

                # Compute DER with detailed components
                details = self.metric(reference, hypothesis, uem=uem, detailed=True)

                # Extract raw values for corpus calculation
                total = details["total"]
                confusion_raw = details["confusion"]
                missed_raw = details["missed detection"]
                false_alarm_raw = details["false alarm"]

                # Compute per-sample percentages
                if total > 0:
                    der = (confusion_raw + missed_raw + false_alarm_raw) / total
                    confusion = confusion_raw / total
                    missed = missed_raw / total
                    false_alarm = false_alarm_raw / total
                else:
                    der = confusion = missed = false_alarm = 0.0

                result = DiarizationResult(
                    der=der * 100,
                    confusion=confusion * 100,
                    missed=missed * 100,
                    false_alarm=false_alarm * 100,
                    time=inference_time,
                    num_speakers_ref=len(set(sample[self.speakers_field])),
                    num_speakers_hyp=len({seg["speaker"] for seg in segments}),
                    total=total,
                    confusion_raw=confusion_raw,
                    missed_raw=missed_raw,
                    false_alarm_raw=false_alarm_raw,
                )
                self.results.append(result)

                print(
                    f"Sample {processed}: DER={result.der:.1f}% "
                    f"(conf={result.confusion:.1f}%, miss={result.missed:.1f}%, fa={result.false_alarm:.1f}%) "
                    f"Time={inference_time:.2f}s "
                    f"Speakers: ref={result.num_speakers_ref}, hyp={result.num_speakers_hyp}"
                )

            except Exception as e:
                print(f"Error on sample {processed}: {e}")
                # Skip failed samples - don't add to results to avoid polluting corpus metrics
                continue

            # Checkpoint every 50 samples
            if processed % 50 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        metrics = self.compute_metrics()
        print(f"\n{'=' * 60}")
        print(
            f"CHECKPOINT @ {sample_count}: DER={metrics['der']:.2f}% "
            f"(conf={metrics['confusion']:.2f}%, miss={metrics['missed']:.2f}%, fa={metrics['false_alarm']:.2f}%)"
        )
        print(f"{'=' * 60}\n")

    def compute_metrics(self) -> dict:
        """Compute final corpus-level metrics."""
        if not self.results:
            return {
                "der": 0.0,
                "confusion": 0.0,
                "missed": 0.0,
                "false_alarm": 0.0,
                "avg_time": 0.0,
                "num_samples": 0,
            }

        # Corpus-level: sum raw values across all samples
        total_duration = sum(r.total for r in self.results)
        total_confusion = sum(r.confusion_raw for r in self.results)
        total_missed = sum(r.missed_raw for r in self.results)
        total_false_alarm = sum(r.false_alarm_raw for r in self.results)

        if total_duration > 0:
            corpus_der = (total_confusion + total_missed + total_false_alarm) / total_duration * 100
            corpus_confusion = total_confusion / total_duration * 100
            corpus_missed = total_missed / total_duration * 100
            corpus_false_alarm = total_false_alarm / total_duration * 100
        else:
            corpus_der = corpus_confusion = corpus_missed = corpus_false_alarm = 0.0

        return {
            "der": corpus_der,
            "confusion": corpus_confusion,
            "missed": corpus_missed,
            "false_alarm": corpus_false_alarm,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }


class AssemblyAIDiarizationEvaluator(DiarizationEvaluator):
    """Evaluator for AssemblyAI speaker diarization."""

    def __init__(self, api_key: str, model: str = "slam_1", **kwargs):
        kwargs.pop("hf_token", None)
        super().__init__(**kwargs)
        self.transcriber = setup_assemblyai(api_key, model, speaker_labels=True)

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run AssemblyAI diarization on audio."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        segments = [
            {"speaker": u.speaker, "start": u.start / 1000.0, "end": u.end / 1000.0}
            for u in (transcript.utterances or [])
        ]
        time.sleep(0.5)
        return segments, elapsed
