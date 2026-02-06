"""Timestamp alignment evaluator implementations."""

import io
import time

from scripts.eval.audio import TextNormalizer, prepare_wav_bytes
from tiny_audio.alignment import ForcedAligner

from .base import AlignmentResult, console, setup_assemblyai


def align_words_to_reference(
    pred_words: list[dict],
    ref_words: list[dict],
    normalizer: TextNormalizer,
) -> list[tuple[dict, dict]]:
    """Align predicted words to reference words using normalized text matching.

    Uses a simple greedy alignment: for each reference word, find the best
    matching predicted word that hasn't been used yet.

    Returns:
        List of (pred_word, ref_word) tuples for successfully aligned words.
    """
    aligned_pairs = []

    # Normalize reference words
    ref_normalized = []
    for rw in ref_words:
        word = rw.get("word", "")
        # Skip <unk> tokens in reference
        if word == "<unk>":
            continue
        norm = normalizer.normalize(word).strip()
        if norm:
            ref_normalized.append((norm, rw))

    # Normalize predicted words
    pred_normalized = []
    for pw in pred_words:
        word = pw.get("word", "")
        norm = normalizer.normalize(word).strip()
        if norm:
            pred_normalized.append((norm, pw))

    # Greedy alignment: for each ref word, find matching pred word
    used_pred_indices = set()

    for ref_norm, ref_word in ref_normalized:
        best_match_idx = None
        best_time_diff = float("inf")

        for i, (pred_norm, pred_word) in enumerate(pred_normalized):
            if i in used_pred_indices:
                continue

            # Check if normalized words match
            if pred_norm == ref_norm:
                # Prefer matches closer in time
                time_diff = abs(pred_word.get("start", 0) - ref_word.get("start", 0))
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match_idx = i

        if best_match_idx is not None:
            used_pred_indices.add(best_match_idx)
            aligned_pairs.append((pred_normalized[best_match_idx][1], ref_word))

    return aligned_pairs


class BaseAlignmentEvaluator:
    """Base class for timestamp alignment evaluators."""

    def __init__(
        self,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        num_workers: int = 1,
    ):
        self.audio_field = audio_field
        self.text_field = text_field
        self.words_field = words_field
        self.num_workers = num_workers
        self.normalizer = TextNormalizer()
        self.results: list[AlignmentResult] = []

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        raise NotImplementedError

    def evaluate(self, dataset, max_samples: int | None = None) -> list[AlignmentResult]:
        """Run timestamp alignment evaluation loop on dataset."""
        self.results = []
        processed = 0

        for sample in dataset:
            ref_text = sample[self.text_field]

            # Skip samples marked as inaudible
            if isinstance(ref_text, str) and "inaudible" in ref_text.lower():
                continue

            processed += 1
            if max_samples and processed > max_samples:
                break

            ref_words = sample[self.words_field]

            try:
                pred_text, pred_words, inference_time = self.transcribe_with_timestamps(
                    sample[self.audio_field]
                )
            except Exception as e:
                console.print(f"[red]Error on sample {processed}: {e}[/red]")
                continue

            # Align predicted words to reference words
            aligned_pairs = align_words_to_reference(pred_words, ref_words, self.normalizer)

            if not aligned_pairs:
                console.print(
                    f"[yellow]Sample {processed}: No words aligned (ref={len(ref_words)}, pred={len(pred_words)})[/yellow]"
                )
                result = AlignmentResult(
                    pred_starts=[],
                    pred_ends=[],
                    ref_starts=[],
                    ref_ends=[],
                    num_aligned_words=0,
                    num_ref_words=len(ref_words),
                    num_pred_words=len(pred_words),
                    time=inference_time,
                    reference_text=ref_text,
                    predicted_text=pred_text,
                )
                self.results.append(result)
                continue

            # Extract timestamps from aligned pairs
            pred_starts = []
            pred_ends = []
            ref_starts = []
            ref_ends = []

            for pred_word, ref_word in aligned_pairs:
                pred_starts.append(pred_word.get("start", 0))
                pred_ends.append(pred_word.get("end", 0))
                ref_starts.append(ref_word.get("start", 0))
                ref_ends.append(ref_word.get("end", 0))

            result = AlignmentResult(
                pred_starts=pred_starts,
                pred_ends=pred_ends,
                ref_starts=ref_starts,
                ref_ends=ref_ends,
                num_aligned_words=len(aligned_pairs),
                num_ref_words=len(ref_words),
                num_pred_words=len(pred_words),
                time=inference_time,
                reference_text=ref_text,
                predicted_text=pred_text,
            )
            self.results.append(result)

            # Compute sample-level MAE for logging
            mae_start = sum(abs(p - r) for p, r in zip(pred_starts, ref_starts)) / len(pred_starts)
            mae_end = sum(abs(p - r) for p, r in zip(pred_ends, ref_ends)) / len(pred_ends)
            mae_combined = (mae_start + mae_end) / 2

            console.print(
                f"Sample {processed}: MAE={mae_combined * 1000:.1f}ms, "
                f"Aligned={len(aligned_pairs)}/{len(ref_words)} words, "
                f"Time={inference_time:.2f}s"
            )

            # Checkpoint every 50 samples
            if processed % 50 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        metrics = self.compute_metrics()
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(
            f"[bold]CHECKPOINT @ {sample_count}:[/bold] MAE={metrics['mae'] * 1000:.1f}ms, "
            f"Alignment Error={metrics['alignment_error'] * 100:.1f}%"
        )
        console.print(f"[bold]{'=' * 60}[/bold]\n")

    def compute_metrics(self) -> dict:
        """Compute final corpus-level metrics."""
        if not self.results:
            return {
                "mae": 0.0,
                "alignment_error": 1.0,
                "mae_adjusted": float("inf"),
                "avg_time": 0.0,
                "num_samples": 0,
            }

        # Collect all predictions and references across samples
        all_pred_times = []
        all_ref_times = []
        total_aligned = 0
        total_ref = 0

        for r in self.results:
            # Combine start and end times for MAE calculation
            all_pred_times.extend(r.pred_starts)
            all_pred_times.extend(r.pred_ends)
            all_ref_times.extend(r.ref_starts)
            all_ref_times.extend(r.ref_ends)
            total_aligned += r.num_aligned_words
            total_ref += r.num_ref_words

        if not all_pred_times:
            return {
                "mae": float("nan"),
                "alignment_error": 1.0,
                "mae_adjusted": float("inf"),
                "avg_time": sum(r.time for r in self.results) / len(self.results),
                "num_samples": len(self.results),
            }

        # Mean absolute error calculation
        mae = sum(abs(p - r) for p, r in zip(all_pred_times, all_ref_times)) / len(all_pred_times)

        # Direct alignment rate calculation
        alignment_rate = total_aligned / total_ref if total_ref > 0 else 0.0
        alignment_error = 1.0 - alignment_rate

        return {
            "mae": mae,
            "alignment_error": alignment_error,
            "alignment_rate": alignment_rate,
            "mae_adjusted": mae / alignment_rate if alignment_rate > 0 else float("inf"),
            "total_aligned_words": total_aligned,
            "total_ref_words": total_ref,
            "avg_time": sum(r.time for r in self.results) / len(self.results),
            "num_samples": len(self.results),
        }


class TimestampAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using local models."""

    def __init__(
        self,
        model_path: str,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        user_prompt: str | None = None,
    ):
        super().__init__(audio_field, text_field, words_field)
        self.user_prompt = user_prompt

        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            trust_remote_code=True,
        )

        # Print generation config
        gen_config = self.pipe.model.generation_config
        console.print(
            f"[dim]Generation config: max_new_tokens={gen_config.max_new_tokens}, "
            f"min_new_tokens={gen_config.min_new_tokens}, "
            f"repetition_penalty={gen_config.repetition_penalty}, "
            f"length_penalty={gen_config.length_penalty}, "
            f"no_repeat_ngram_size={gen_config.no_repeat_ngram_size}[/dim]"
        )

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time).

        Uses local ForcedAligner for timestamps to ensure local code changes are applied.
        """
        # Extract audio array and sample rate
        if isinstance(audio, dict):
            audio_array = audio.get("array", audio.get("raw"))
            sample_rate = audio.get("sampling_rate", 16000)
            audio_input = {"raw": audio_array, "sampling_rate": sample_rate}
        else:
            audio_array = audio
            sample_rate = 16000
            audio_input = audio

        start = time.time()

        # Get transcription without timestamps from pipeline
        if self.user_prompt:
            result = self.pipe(audio_input, user_prompt=self.user_prompt)
        else:
            result = self.pipe(audio_input)

        text = result.get("text", "")

        # Use local ForcedAligner for timestamps
        if text.strip():
            words = ForcedAligner.align(audio_array, text, sample_rate=sample_rate)
        else:
            words = []

        elapsed = time.time() - start

        return text, words, elapsed


class AssemblyAIAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using AssemblyAI."""

    def __init__(
        self,
        api_key: str,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
    ):
        super().__init__(audio_field, text_field, words_field)
        self.transcriber = setup_assemblyai(api_key)

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        transcript = self.transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        words = [
            {"word": w.text, "start": w.start / 1000.0, "end": w.end / 1000.0}
            for w in (transcript.words or [])
        ]
        time.sleep(0.5)
        return transcript.text or "", words, elapsed


class DeepgramAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using Deepgram Nova 3."""

    def __init__(
        self,
        api_key: str,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        num_workers: int = 1,
    ):
        super().__init__(audio_field, text_field, words_field, num_workers)
        from deepgram import DeepgramClient

        self.client = DeepgramClient(api_key=api_key)

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        response = self.client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-3",
            smart_format=True,
        )
        elapsed = time.time() - start

        # Extract transcript text
        channel = response.results.channels[0]
        text = channel.alternatives[0].transcript if channel.alternatives else ""

        # Extract word-level timestamps
        words = []
        if channel.alternatives and channel.alternatives[0].words:
            for w in channel.alternatives[0].words:
                words.append(
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                    }
                )

        time.sleep(0.3)  # Rate limiting
        return text, words, elapsed


class ElevenLabsAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using ElevenLabs Scribe."""

    def __init__(
        self,
        api_key: str,
        model: str = "scribe_v2",
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        num_workers: int = 1,
    ):
        super().__init__(audio_field, text_field, words_field, num_workers)
        from elevenlabs.client import ElevenLabs

        self.client = ElevenLabs(api_key=api_key)
        self.model = model

    def transcribe_with_timestamps(self, audio) -> tuple[str, list[dict], float]:
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        transcription = self.client.speech_to_text.convert(
            file=io.BytesIO(wav_bytes),
            model_id=self.model,
        )
        elapsed = time.time() - start

        # Extract transcript text
        text = getattr(transcription, "text", "") or ""

        # Extract word-level timestamps
        words = []
        api_words = getattr(transcription, "words", []) or []
        for w in api_words:
            word_text = getattr(w, "text", "")
            word_start = getattr(w, "start", None)
            word_end = getattr(w, "end", None)
            word_type = getattr(w, "type", "word")

            # Skip non-word types (spacing, audio_event)
            if word_type != "word" or word_start is None or word_end is None:
                continue

            words.append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                }
            )

        time.sleep(0.3)  # Rate limiting
        return text, words, elapsed
