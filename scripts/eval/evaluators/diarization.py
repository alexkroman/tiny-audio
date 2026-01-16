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

    # VAD hysteresis parameters (segment-level approximation)
    VAD_MIN_DURATION = 0.15  # Remove segments shorter than this (reduces FA)
    VAD_MAX_GAP = 0.40  # Fill gaps shorter than this (reduces Miss)

    # VAD segment dilation (collar) - captures breath and unvoiced consonants
    VAD_PAD_ONSET = 0.05  # 50ms before speech
    VAD_PAD_OFFSET = 0.05  # 50ms after speech

    # Internal resolution for voting
    VOTING_RATE = 0.01  # 10ms resolution

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
        filtered = [seg for seg in merged if (seg["end"] - seg["start"]) >= self.VAD_MIN_DURATION]

        # Step 3: Dilate segments (add collar to capture breath and unvoiced consonants)
        # VAD cuts can chop off speaker identity info at segment edges
        for seg in filtered:
            seg["start"] = max(0.0, seg["start"] - self.VAD_PAD_ONSET)
            seg["end"] = seg["end"] + self.VAD_PAD_OFFSET
            seg["start_sample"] = int(seg["start"] * 16000)  # Assuming 16kHz
            seg["end_sample"] = int(seg["end"] * 16000)

        return filtered

    def _extract_embeddings(
        self, audio_array: np.ndarray, segments: list[dict], sample_rate: int
    ) -> tuple[np.ndarray, list[dict]]:
        """Extract speaker embeddings using sliding windows over speech segments."""
        import torch

        speaker_model = self._get_eres2netv2_model()
        device = self._get_device()

        window_samples = int(self.WINDOW_SIZE * sample_rate)
        step_samples = int(self.STEP_SIZE * sample_rate)

        embeddings = []
        window_segments = []

        with torch.no_grad():
            for seg in segments:
                seg_start = seg["start_sample"]
                seg_end = seg["end_sample"]
                seg_len = seg_end - seg_start

                # Generate window positions
                if seg_len <= window_samples:
                    starts = [seg_start]
                    ends = [seg_end]
                else:
                    starts = list(range(seg_start, seg_end - window_samples + 1, step_samples))
                    ends = [s + window_samples for s in starts]

                    # Cover tail if >10% of window remains
                    if ends and ends[-1] < seg_end:
                        remainder = seg_end - ends[-1]
                        if remainder > (window_samples * 0.1):
                            starts.append(seg_end - window_samples)
                            ends.append(seg_end)

                for c_start, c_end in zip(starts, ends):
                    chunk = audio_array[c_start:c_end]

                    # Pad short chunks with reflection (preserves spectral characteristics)
                    if len(chunk) < window_samples:
                        pad_width = window_samples - len(chunk)
                        chunk = np.pad(chunk, (0, pad_width), mode="reflect")

                    # Convert to tensor and process
                    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                    embedding = speaker_model.forward(chunk_tensor).squeeze(0).cpu().numpy()

                    # Validate and normalize
                    if not np.isfinite(embedding).all():
                        continue
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-8:
                        embeddings.append(embedding / norm)
                        window_segments.append(
                            {"start": c_start / sample_rate, "end": c_end / sample_rate}
                        )

        if embeddings:
            return np.array(embeddings), window_segments

        return np.array([]), []

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using spectral clustering."""
        if embeddings.shape[0] == 0:
            return np.array([])

        clusterer = self._get_clusterer()
        return clusterer(embeddings, self.num_speakers)

    def _postprocess_segments(
        self, window_segments: list[dict], labels: np.ndarray, total_duration: float
    ) -> list[dict]:
        """Post-process using Frame-Level Consensus Voting.

        This replaces the midpoint cut logic. We create a timeline at 10ms resolution.
        Every sliding window casts a 'vote' for its predicted speaker across its entire duration.
        The speaker with the most votes at any given 10ms frame wins.
        """
        if not window_segments or len(labels) == 0:
            return []

        # 1. Correct labels to be contiguous 0..N
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        clean_labels = np.array([label_map[lbl] for lbl in labels])
        num_speakers = len(unique_labels)

        if num_speakers == 0:
            return []

        # 2. Create Voting Grid (10ms resolution)
        num_frames = int(np.ceil(total_duration / self.VOTING_RATE)) + 1
        votes = np.zeros((num_frames, num_speakers), dtype=np.float32)

        # 3. Accumulate votes from each window
        for win, label in zip(window_segments, clean_labels):
            start_frame = int(win["start"] / self.VOTING_RATE)
            end_frame = int(win["end"] / self.VOTING_RATE)
            end_frame = min(end_frame, num_frames)
            if start_frame < end_frame:
                votes[start_frame:end_frame, label] += 1.0

        # 4. Determine Winner per Frame
        frame_speakers = np.argmax(votes, axis=1)
        max_votes = np.max(votes, axis=1)

        # 5. Convert Frames back to Segments
        final_segments = []
        current_speaker = -1
        seg_start = 0.0

        for f in range(num_frames):
            speaker = frame_speakers[f]
            score = max_votes[f]

            # Treat zero votes as silence
            if score == 0:
                speaker = -1

            if speaker != current_speaker:
                if current_speaker != -1:
                    final_segments.append(
                        {
                            "speaker": f"SPEAKER_{current_speaker}",
                            "start": seg_start,
                            "end": f * self.VOTING_RATE,
                        }
                    )
                current_speaker = speaker
                seg_start = f * self.VOTING_RATE

        # Close last segment
        if current_speaker != -1:
            final_segments.append(
                {
                    "speaker": f"SPEAKER_{current_speaker}",
                    "start": seg_start,
                    "end": num_frames * self.VOTING_RATE,
                }
            )

        # 6. Merge short segments to reduce flicker
        return self._merge_short_segments(final_segments)

    def _merge_short_segments(self, segments: list[dict], min_dur: float = 0.15) -> list[dict]:
        """Merge extremely short segments into neighbors to reduce DER confusion."""
        if not segments:
            return []

        clean = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            if dur < min_dur:
                # If same speaker as last and close, extend
                if (
                    clean
                    and clean[-1]["speaker"] == seg["speaker"]
                    and seg["start"] - clean[-1]["end"] < 0.1
                ):
                    clean[-1]["end"] = seg["end"]
                continue

            # If same speaker as previous and close, merge
            if (
                clean
                and clean[-1]["speaker"] == seg["speaker"]
                and seg["start"] - clean[-1]["end"] < 0.5
            ):
                clean[-1]["end"] = seg["end"]
            else:
                clean.append(seg)

        return clean

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
        total_duration = len(audio_array) / sample_rate

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

        # Step 4: Consensus voting post-processing
        output_segments = self._postprocess_segments(window_segments, labels, total_duration)

        elapsed = time.time() - start
        return output_segments, elapsed
