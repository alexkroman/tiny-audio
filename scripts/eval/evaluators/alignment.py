"""Timestamp alignment evaluator implementations."""

import io
import statistics
import time

from scripts.eval.audio import TextNormalizer, prepare_wav_bytes

from .base import AlignmentResult, console, setup_assemblyai


def align_words_to_reference(
    pred_words: list[dict],
    ref_words: list[dict],
    normalizer: TextNormalizer,
) -> list[tuple[dict, dict]]:
    """Align predicted words to reference words using LCS-based monotonic matching.

    Uses Longest Common Subsequence (LCS) to find the optimal monotonic alignment.
    This maximizes the number of matched words while ensuring they appear in order,
    preventing duplicate words (like "a", "the") from being matched out of order.

    Returns:
        List of (pred_word, ref_word) tuples for successfully aligned words.
    """
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

    n_ref = len(ref_normalized)
    n_pred = len(pred_normalized)

    if n_ref == 0 or n_pred == 0:
        return []

    # Build LCS table
    # dp[i][j] = length of LCS of ref[:i] and pred[:j]
    dp = [[0] * (n_pred + 1) for _ in range(n_ref + 1)]

    for i in range(1, n_ref + 1):
        for j in range(1, n_pred + 1):
            if ref_normalized[i - 1][0] == pred_normalized[j - 1][0]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual alignment
    aligned_pairs = []
    i, j = n_ref, n_pred

    while i > 0 and j > 0:
        if ref_normalized[i - 1][0] == pred_normalized[j - 1][0]:
            aligned_pairs.append((pred_normalized[j - 1][1], ref_normalized[i - 1][1]))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Reverse to get chronological order
    aligned_pairs.reverse()

    return aligned_pairs


class BaseAlignmentEvaluator:
    """Base class for timestamp alignment evaluators."""

    def __init__(
        self,
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        num_workers: int = 1,
        verbose: bool = False,
    ):
        self.audio_field = audio_field
        self.text_field = text_field
        self.words_field = words_field
        self.num_workers = num_workers
        self.verbose = verbose
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

            # Verbose word-by-word alignment logging
            if self.verbose:
                console.print(
                    f"  [dim]{'Word':<20} {'Pred Start':>10} {'Ref Start':>10} {'Δ Start':>10} {'Pred End':>10} {'Ref End':>10} {'Δ End':>10}[/dim]"
                )
                console.print(f"  [dim]{'-' * 92}[/dim]")
                for (pred_word, _ref_word), ps, pe, rs, re in zip(
                    aligned_pairs, pred_starts, pred_ends, ref_starts, ref_ends
                ):
                    word = pred_word.get("word", "")[:20]
                    delta_start = (ps - rs) * 1000  # ms
                    delta_end = (pe - re) * 1000  # ms
                    # Color code: green if <50ms, yellow if <100ms, red otherwise
                    start_color = (
                        "green"
                        if abs(delta_start) < 50
                        else "yellow"
                        if abs(delta_start) < 100
                        else "red"
                    )
                    end_color = (
                        "green"
                        if abs(delta_end) < 50
                        else "yellow"
                        if abs(delta_end) < 100
                        else "red"
                    )
                    console.print(
                        f"  {word:<20} {ps:>10.3f} {rs:>10.3f} [{start_color}]{delta_start:>+10.1f}[/{start_color}] "
                        f"{pe:>10.3f} {re:>10.3f} [{end_color}]{delta_end:>+10.1f}[/{end_color}]"
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
            f"[bold]CHECKPOINT @ {sample_count}:[/bold] Median AE={metrics['mae'] * 1000:.1f}ms"
        )
        console.print(f"[bold]{'=' * 60}[/bold]\n")

    def compute_metrics(self) -> dict:
        """Compute final corpus-level metrics."""
        if not self.results:
            return {
                "mae": 0.0,
                "avg_time": 0.0,
                "num_samples": 0,
            }

        # Collect all predictions and references across samples
        all_pred_times = []
        all_ref_times = []

        for r in self.results:
            # Combine start and end times for MAE calculation
            all_pred_times.extend(r.pred_starts)
            all_pred_times.extend(r.pred_ends)
            all_ref_times.extend(r.ref_starts)
            all_ref_times.extend(r.ref_ends)

        if not all_pred_times:
            return {
                "mae": float("nan"),
                "avg_time": sum(r.time for r in self.results) / len(self.results),
                "num_samples": len(self.results),
            }

        # Median absolute error (robust to outliers)
        errors = [abs(p - r) for p, r in zip(all_pred_times, all_ref_times)]
        median_ae = statistics.median(errors)

        return {
            "mae": median_ae,
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
        verbose: bool = False,
    ):
        super().__init__(audio_field, text_field, words_field, verbose=verbose)
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
        """Transcribe audio and return (text, word_timestamps, inference_time)."""
        # Convert to pipeline-compatible format
        if isinstance(audio, dict) and "array" in audio and "raw" not in audio:
            audio_input = {"raw": audio["array"], "sampling_rate": audio["sampling_rate"]}
        else:
            audio_input = audio

        start = time.time()
        if self.user_prompt:
            result = self.pipe(audio_input, return_timestamps=True, user_prompt=self.user_prompt)
        else:
            result = self.pipe(audio_input, return_timestamps=True)
        elapsed = time.time() - start

        text = result.get("text", "")
        words = result.get("words", [])

        return text, words, elapsed


class AssemblyAIAlignmentEvaluator(BaseAlignmentEvaluator):
    """Evaluator for word-level timestamp alignment accuracy using AssemblyAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "slam_1",
        audio_field: str = "audio",
        text_field: str = "transcript",
        words_field: str = "words",
        verbose: bool = False,
    ):
        super().__init__(audio_field, text_field, words_field, verbose=verbose)
        self.transcriber = setup_assemblyai(api_key, model)

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
        verbose: bool = False,
    ):
        super().__init__(audio_field, text_field, words_field, num_workers, verbose=verbose)
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
        verbose: bool = False,
    ):
        super().__init__(audio_field, text_field, words_field, num_workers, verbose=verbose)
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
