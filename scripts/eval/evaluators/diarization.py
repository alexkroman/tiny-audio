"""Diarization evaluator implementations."""

import io
import os
import time

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from scripts.eval.audio import prepare_wav_bytes

from .base import DiarizationResult, console, setup_assemblyai


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
        num_workers: int = 1,
    ):
        self.audio_field = audio_field
        self.speakers_field = speakers_field
        self.timestamps_start_field = timestamps_start_field
        self.timestamps_end_field = timestamps_end_field
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.num_workers = num_workers
        self.results: list[DiarizationResult] = []
        # Standard collar of 0.25s for boundary tolerance
        self.metric = DiarizationErrorRate(collar=0.25)

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

        from tiny_audio.asr_pipeline import SpeakerDiarizer

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

                console.print(
                    f"Sample {processed}: DER={result.der:.1f}% "
                    f"(conf={result.confusion:.1f}%, miss={result.missed:.1f}%, fa={result.false_alarm:.1f}%) "
                    f"Time={inference_time:.2f}s "
                    f"Speakers: ref={result.num_speakers_ref}, hyp={result.num_speakers_hyp}"
                )

            except Exception as e:
                console.print(f"[red]Error on sample {processed}: {e}[/red]")
                # Skip failed samples - don't add to results to avoid polluting corpus metrics
                continue

            # Checkpoint every 50 samples
            if processed % 50 == 0:
                self._print_checkpoint(processed)

        return self.results

    def _print_checkpoint(self, sample_count: int):
        """Print cumulative metrics checkpoint."""
        metrics = self.compute_metrics()
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(
            f"[bold]CHECKPOINT @ {sample_count}:[/bold] DER={metrics['der']:.2f}% "
            f"(conf={metrics['confusion']:.2f}%, miss={metrics['missed']:.2f}%, fa={metrics['false_alarm']:.2f}%)"
        )
        console.print(f"[bold]{'=' * 60}[/bold]\n")

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


class DeepgramDiarizationEvaluator(DiarizationEvaluator):
    """Evaluator for Deepgram Nova 3 speaker diarization."""

    def __init__(self, api_key: str, **kwargs):
        kwargs.pop("hf_token", None)
        super().__init__(**kwargs)
        from deepgram import DeepgramClient

        self.client = DeepgramClient(api_key=api_key)

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run Deepgram diarization on audio."""
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        response = self.client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-3",
            diarize=True,
            utterances=True,
        )
        elapsed = time.time() - start

        # Extract utterances with speaker info
        utterances = response.results.utterances or []
        segments = [
            {
                "speaker": f"SPEAKER_{u.speaker}",
                "start": u.start,
                "end": u.end,
            }
            for u in utterances
        ]

        time.sleep(0.3)  # Rate limiting
        return segments, elapsed


class LocalDiarizationEvaluator(DiarizationEvaluator):
    """Local diarization evaluator using TEN-VAD + ERes2NetV2 + spectral clustering.

    Pipeline:
    1. TEN-VAD detects speech segments
    2. Sliding window (1.0s, 0.5s overlap) for uniform embedding extraction
    3. ERes2NetV2 extracts speaker embeddings per window
    4. Spectral clustering with eigenvalue gap for auto speaker detection
    5. Post-processing merges adjacent same-speaker windows
    """

    _ten_vad_model = None
    _eres2netv2_model = None
    _clusterer = None

    # Sliding window parameters
    WINDOW_SIZE = 1.0  # seconds
    STEP_SIZE = 0.25  # seconds (75% overlap)

    # GPU Optimization - process multiple windows at once
    BATCH_SIZE = 64

    # VAD hysteresis parameters (segment-level approximation)
    VAD_MIN_DURATION = 0.15  # Remove segments shorter than this (reduces FA)
    VAD_MAX_GAP = 0.40  # Fill gaps shorter than this (reduces Miss)

    def __init__(
        self,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int = 10,
        **kwargs,
    ):
        kwargs.pop("hf_token", None)
        super().__init__(**kwargs)
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    @classmethod
    def _get_ten_vad_model(cls):
        """Lazy-load TEN-VAD model (singleton)."""
        if cls._ten_vad_model is None:
            from ten_vad import TenVad

            console.print("Loading TEN-VAD model...")
            # Balanced threshold
            cls._ten_vad_model = TenVad(hop_size=256, threshold=0.28)
        return cls._ten_vad_model

    _device = None

    @classmethod
    def _get_device(cls):
        """Get the best available device (MPS for MacBook, CUDA, or CPU)."""
        if cls._device is None:
            import torch

            if torch.backends.mps.is_available():
                cls._device = torch.device("mps")
                console.print("Using MPS (Metal) GPU acceleration")
            elif torch.cuda.is_available():
                cls._device = torch.device("cuda")
                console.print("Using CUDA GPU acceleration")
            else:
                cls._device = torch.device("cpu")
                console.print("Using CPU (no GPU available)")
        return cls._device

    @classmethod
    def _get_eres2netv2_model(cls):
        """Lazy-load ERes2NetV2 speaker embedding model (singleton)."""
        if cls._eres2netv2_model is None:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            console.print("Loading ERes2NetV2 speaker embedding model...")
            sv_pipeline = pipeline(
                task=Tasks.speaker_verification, model="iic/speech_eres2netv2_sv_zh-cn_16k-common"
            )
            cls._eres2netv2_model = sv_pipeline.model

            # Move model to GPU if available
            device = cls._get_device()
            cls._eres2netv2_model = cls._eres2netv2_model.to(device)
            # Also set the model's internal device attribute (not updated by .to())
            cls._eres2netv2_model.device = device
            cls._eres2netv2_model.eval()

        return cls._eres2netv2_model

    def _get_clusterer(self):
        """Get speaker clusterer instance."""
        if self._clusterer is None:
            from .clustering import SpeakerClusterer

            self._clusterer = SpeakerClusterer(
                min_num_spks=self.min_speakers,
                max_num_spks=self.max_speakers,
            )
        return self._clusterer

    def _get_speech_segments(self, audio_array: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """Get speech segments using TEN-VAD (frame-by-frame processing)."""
        vad_model = self._get_ten_vad_model()

        # Convert to int16 as required by TEN-VAD
        # CLIP to prevent integer overflow wrapping (values > 1.0 would overflow)
        if audio_array.dtype != np.int16:
            audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_array

        # Process frame by frame (256 samples per frame = 16ms at 16kHz)
        hop_size = 256
        frame_duration = hop_size / sample_rate  # 16ms
        speech_frames = []

        for i in range(0, len(audio_int16) - hop_size, hop_size):
            frame = audio_int16[i : i + hop_size]
            _, is_speech = vad_model.process(frame)
            speech_frames.append(is_speech)

        # Convert frame-level decisions to segments
        segments = []
        in_speech = False
        start_idx = 0

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_idx = i
                in_speech = True
            elif not is_speech and in_speech:
                start_time = start_idx * frame_duration
                end_time = i * frame_duration
                segments.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "start_sample": int(start_time * sample_rate),
                        "end_sample": int(end_time * sample_rate),
                    }
                )
                in_speech = False

        # Handle trailing speech
        if in_speech:
            start_time = start_idx * frame_duration
            end_time = len(speech_frames) * frame_duration
            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "start_sample": int(start_time * sample_rate),
                    "end_sample": int(end_time * sample_rate),
                }
            )

        # Apply hysteresis post-processing
        return self._apply_vad_hysteresis(segments)

    def _apply_vad_hysteresis(self, segments: list[dict]) -> list[dict]:
        """Apply hysteresis-like post-processing to VAD segments.

        This approximates pyannote's hysteresis thresholding at the segment level:
        - Fill short gaps between segments (simulates low offset threshold)
        - Remove very short segments (simulates high onset threshold)

        Args:
            segments: List of VAD segments with start/end times

        Returns:
            Post-processed segments
        """
        if not segments:
            return segments

        # Sort by start time
        segments = sorted(segments, key=lambda x: x["start"])

        # Step 1: Fill short gaps (merge segments that are close together)
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= self.VAD_MAX_GAP:
                # Fill the gap by extending previous segment
                merged[-1]["end"] = seg["end"]
                merged[-1]["end_sample"] = seg["end_sample"]
            else:
                merged.append(seg.copy())

        # Step 2: Remove segments shorter than min_duration
        return [seg for seg in merged if (seg["end"] - seg["start"]) >= self.VAD_MIN_DURATION]

    def _extract_embeddings(
        self, audio_array: np.ndarray, segments: list[dict], sample_rate: int
    ) -> tuple[np.ndarray, list[dict]]:
        """Extract embeddings using GPU-optimized processing.

        Note: The modelscope ERes2NetV2 wrapper doesn't support true batching
        (it collapses batch dimension internally). We optimize by:
        1. Pre-computing all audio chunks (vectorized)
        2. Converting to tensor once (not per-iteration)
        3. Processing on GPU with minimal Python overhead
        """
        import torch

        speaker_model = self._get_eres2netv2_model()
        device = self._get_device()

        window_samples = int(self.WINDOW_SIZE * sample_rate)
        step_samples = int(self.STEP_SIZE * sample_rate)

        # 1. Prepare all audio chunks first (CPU work, vectorized)
        audio_chunks = []
        window_segments = []

        for seg in segments:
            seg_start = seg["start_sample"]
            seg_end = seg["end_sample"]
            seg_len = seg_end - seg_start

            # Generate start indices for this segment
            if seg_len <= window_samples:
                starts = [seg_start]
                ends = [seg_end]
            else:
                # Vectorized generation of sliding windows
                starts = list(range(seg_start, seg_end - window_samples + 1, step_samples))
                ends = [s + window_samples for s in starts]

                # Handle remainder
                if ends and ends[-1] < seg_end and (seg_end - ends[-1]) > (window_samples // 2):
                    starts.append(seg_end - window_samples)
                    ends.append(seg_end)

            for c_start, c_end in zip(starts, ends):
                chunk = audio_array[c_start:c_end]

                # Pad if necessary to ensure uniform size
                if len(chunk) < window_samples:
                    chunk = np.pad(chunk, (0, window_samples - len(chunk)))
                elif len(chunk) > window_samples:
                    chunk = chunk[:window_samples]

                audio_chunks.append(chunk)
                window_segments.append({"start": c_start / sample_rate, "end": c_end / sample_rate})

        if not audio_chunks:
            return np.array([]), []

        # 2. Process embeddings (modelscope model doesn't support true batching)
        embeddings = []
        valid_indices = []

        # Convert all chunks to tensor once (faster than per-iteration conversion)
        all_audio_tensor = torch.from_numpy(np.array(audio_chunks)).float()

        with torch.no_grad():
            for i in range(len(all_audio_tensor)):
                # Get single chunk and add batch dimension
                audio_tensor = all_audio_tensor[i : i + 1].to(device)

                # Forward pass
                embedding = speaker_model.forward(audio_tensor).squeeze(0).cpu().numpy()

                # Handle NaN/inf values
                if not np.isfinite(embedding).all():
                    continue

                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                    embeddings.append(embedding)
                    valid_indices.append(i)

        if embeddings:
            final_embeddings = np.array(embeddings)
            final_window_segments = [window_segments[i] for i in valid_indices]
            return final_embeddings, final_window_segments

        return np.array([]), []

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using spectral clustering with label smoothing."""
        from scipy.ndimage import median_filter

        if embeddings.shape[0] == 0:
            return np.array([])

        clusterer = self._get_clusterer()
        raw_labels = clusterer(embeddings, self.num_speakers)

        # Median filter to smooth out rapid label flickering (A-A-B-A-A -> A-A-A-A-A)
        if len(raw_labels) >= 3:
            return median_filter(raw_labels, size=3)
        return raw_labels

    def _postprocess_segments(self, window_segments: list[dict], labels: np.ndarray) -> list[dict]:
        """Post-process diarization segments using window centers.

        Instead of midpoint cuts on overlapping windows, we trust the window center
        timestamps. Each label applies to its window's center, and we extend segments
        until the label changes. This handles boundaries more naturally with high
        overlap (75%).

        Important: Handles silence gaps correctly - if VAD detected a gap between
        windows, we don't bridge it (which would create false alarms).
        """
        if not window_segments or len(labels) == 0:
            return []

        # Ensure lengths match (in case embedding filtering dropped some)
        if len(window_segments) != len(labels):
            min_len = min(len(window_segments), len(labels))
            window_segments = window_segments[:min_len]
            labels = labels[:min_len]

        labels = self._correct_labels(labels)

        # If windows are further apart than this, don't link them (VAD detected silence)
        # Since step_size is 0.25s, a gap > 0.5s implies a VAD break
        max_link_gap = 0.5

        window_centers = [(ws["start"] + ws["end"]) / 2 for ws in window_segments]

        segments = []
        current_label = int(labels[0])
        segment_start = window_segments[0]["start"]
        last_valid_end = window_segments[0]["end"]

        for i in range(1, len(labels)):
            this_label = int(labels[i])
            prev_end = window_segments[i - 1]["end"]
            curr_start = window_segments[i]["start"]
            gap = curr_start - prev_end

            # Scenario 1: Large gap (Silence detected by VAD)
            if gap > max_link_gap:
                # Close previous segment at its natural end
                segments.append([segment_start, last_valid_end, current_label])
                # Start new segment at current window's start
                segment_start = curr_start
                current_label = this_label

            # Scenario 2: Contiguous speech, label changed
            elif this_label != current_label:
                # Boundary is midpoint between centers
                boundary = (window_centers[i - 1] + window_centers[i]) / 2
                segments.append([segment_start, boundary, current_label])
                segment_start = boundary
                current_label = this_label

            # Scenario 3: Same label, contiguous - continue extending

            # Always update the valid end time
            last_valid_end = window_segments[i]["end"]

        # Close final segment
        segments.append([segment_start, last_valid_end, current_label])

        return [
            {
                "speaker": f"SPEAKER_{seg[2]}",
                "start": round(seg[0], 2),
                "end": round(seg[1], 2),
            }
            for seg in segments
        ]

    def _correct_labels(self, labels: np.ndarray) -> np.ndarray:
        """Re-index labels sequentially starting from 0."""
        id_map = {}
        new_labels = []
        next_id = 0
        for label in labels:
            if label not in id_map:
                id_map[label] = next_id
                next_id += 1
            new_labels.append(id_map[label])
        return np.array(new_labels)

    def _merge_sequential(self, segments: list) -> list:
        """Merge consecutive segments with same speaker."""
        if not segments:
            return []

        result = [segments[0].copy()]
        for i in range(1, len(segments)):
            # Same speaker and no gap -> merge
            if segments[i][2] == result[-1][2] and segments[i][0] <= result[-1][1]:
                result[-1][1] = max(result[-1][1], segments[i][1])
            else:
                result.append(segments[i].copy())
        return result

    def diarize(self, audio) -> tuple[list[dict], float]:
        """Run VAD + speaker embedding + clustering diarization."""
        import librosa

        if isinstance(audio, dict):
            if "array" in audio:
                audio_array = audio["array"]
                sample_rate = audio.get("sampling_rate", 16000)
            elif "bytes" in audio:
                audio_array, sample_rate = librosa.load(io.BytesIO(audio["bytes"]), sr=16000)
            else:
                raise ValueError(f"Unsupported audio dict format: {audio.keys()}")
        else:
            raise ValueError(f"Unsupported audio format: {type(audio)}")

        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        audio_array = audio_array.astype(np.float32)

        start = time.time()

        # Step 1: VAD
        segments = self._get_speech_segments(audio_array, sample_rate)
        if not segments:
            return [], time.time() - start

        # Step 2: Extract embeddings
        embeddings, window_segments = self._extract_embeddings(audio_array, segments, sample_rate)
        if len(embeddings) == 0:
            return [], time.time() - start

        # Step 3: Cluster
        labels = self._cluster_embeddings(embeddings)

        # Step 4: Post-process segments (merge, smooth, distribute overlaps)
        output_segments = self._postprocess_segments(window_segments, labels)

        elapsed = time.time() - start
        output_segments.sort(key=lambda x: x["start"])

        return output_segments, elapsed
