"""Speaker diarization using TEN-VAD + ECAPA-TDNN + spectral clustering.

Spectral clustering implementation adapted from FunASR/3D-Speaker:
https://github.com/alibaba-damo-academy/FunASR
MIT License (https://opensource.org/licenses/MIT)
"""

import warnings

import numpy as np
import scipy
import sklearn.metrics.pairwise
import torch
from sklearn.cluster._kmeans import k_means
from sklearn.preprocessing import normalize


def _get_device() -> torch.device:
    """Get best available device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpectralCluster:
    """Spectral clustering using unnormalized Laplacian of affinity matrix.

    Adapted from FunASR/3D-Speaker and SpeechBrain implementations.
    Uses eigenvalue gap to automatically determine number of speakers.
    """

    def __init__(self, min_num_spks: int = 1, max_num_spks: int = 15, pval: float = 0.06):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval

    def __call__(self, embeddings: np.ndarray, oracle_num: int | None = None) -> np.ndarray:
        """Run spectral clustering on embeddings.

        Args:
            embeddings: Speaker embeddings of shape [N, D]
            oracle_num: Optional known number of speakers

        Returns:
            Cluster labels of shape [N]
        """
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(embeddings)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        return self.cluster_embs(emb, num_of_spk)

    def get_sim_mat(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        return sklearn.metrics.pairwise.cosine_similarity(embeddings, embeddings)

    def p_pruning(self, affinity: np.ndarray) -> np.ndarray:
        """Prune low similarity values in affinity matrix (keep top pval fraction)."""
        n = affinity.shape[0]
        pval = max(self.pval, 6.0 / n)
        k_keep = max(1, int(pval * n))

        # Vectorized: find top-k indices per row and zero out the rest
        top_k_idx = np.argpartition(affinity, -k_keep, axis=1)[:, -k_keep:]
        mask = np.zeros_like(affinity, dtype=bool)
        np.put_along_axis(mask, top_k_idx, True, axis=1)
        affinity[~mask] = 0
        return affinity

    def get_laplacian(self, sim_mat: np.ndarray) -> np.ndarray:
        """Compute unnormalized Laplacian matrix."""
        from scipy.sparse.csgraph import laplacian

        np.fill_diagonal(sim_mat, 0)
        return laplacian(sim_mat, normed=False)

    def get_spec_embs(
        self, laplacian: np.ndarray, k_oracle: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Extract spectral embeddings from Laplacian.

        Uses the eigengap heuristic to estimate the number of clusters:
        The number of clusters k is chosen where the gap between consecutive
        eigenvalues is largest, indicating a transition from "cluster" eigenvalues
        (near 0) to "noise" eigenvalues.
        """
        lambdas, eig_vecs = scipy.linalg.eigh(laplacian)

        num_of_spk = k_oracle if k_oracle is not None else self._estimate_num_speakers(lambdas)

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def _estimate_num_speakers(self, lambdas: np.ndarray) -> int:
        """Estimate number of speakers using refined eigengap heuristic.

        For spectral clustering, we look for the largest gap in eigenvalues.
        The eigenvalues corresponding to clusters are close to 0, and there
        should be a significant jump to the remaining eigenvalues.
        """
        # Consider eigenvalues from index 1 to max_num_spks (skip first, it's always ~0)
        # We need gaps between positions, so look at indices 1 to max_num_spks+1
        max_idx = min(self.max_num_spks + 1, len(lambdas))
        relevant_lambdas = lambdas[1:max_idx]  # Skip first eigenvalue

        if len(relevant_lambdas) < 2:
            return self.min_num_spks

        # Compute absolute gaps (not ratios - ratios are unstable near 0)
        gaps = np.diff(relevant_lambdas)

        # Find the largest gap - the index gives us (k-1) since we skipped first
        # Add 1 to convert from gap index to number of speakers
        # Add 1 again because we skipped the first eigenvalue
        max_gap_idx = int(np.argmax(gaps))
        num_of_spk = max_gap_idx + 2  # +1 for gap->count, +1 for skipped eigenvalue

        # Clamp between min and max
        return max(self.min_num_spks, min(num_of_spk, self.max_num_spks))

    def cluster_embs(self, emb: np.ndarray, k: int) -> np.ndarray:
        """Cluster spectral embeddings using k-means."""
        _, labels, _ = k_means(emb, k, n_init=10)
        return labels

    def get_eigen_gaps(self, eig_vals: np.ndarray) -> np.ndarray:
        """Compute gaps between consecutive eigenvalues."""
        return np.diff(eig_vals)


class SpeakerClusterer:
    """Speaker clustering backend using spectral clustering with speaker merging.

    Features:
    - Spectral clustering with eigenvalue gap for auto speaker count detection
    - P-pruning for affinity matrix refinement
    - Post-clustering speaker merging by cosine similarity
    """

    def __init__(
        self,
        min_num_spks: int = 2,
        max_num_spks: int = 10,
        merge_thr: float = 0.90,  # Moderate merging
    ):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.merge_thr = merge_thr
        self._spectral_cluster: SpectralCluster | None = None

    def _get_spectral_cluster(self) -> SpectralCluster:
        """Lazy-load spectral clusterer."""
        if self._spectral_cluster is None:
            self._spectral_cluster = SpectralCluster(
                min_num_spks=self.min_num_spks,
                max_num_spks=self.max_num_spks,
            )
        return self._spectral_cluster

    def __call__(self, embeddings: np.ndarray, num_speakers: int | None = None) -> np.ndarray:
        """Cluster speaker embeddings and return labels.

        Args:
            embeddings: Speaker embeddings of shape [N, D]
            num_speakers: Optional oracle number of speakers

        Returns:
            Cluster labels of shape [N]
        """
        import warnings

        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

        # Handle edge cases
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)
        if embeddings.shape[0] == 1:
            return np.array([0], dtype=int)
        if embeddings.shape[0] < 6:
            return np.zeros(embeddings.shape[0], dtype=int)

        # Normalize embeddings and replace NaN/inf
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        embeddings = normalize(embeddings)

        # Run spectral clustering (suppress numerical warnings)
        spectral = self._get_spectral_cluster()

        # Update min/max for oracle case
        if num_speakers is not None:
            spectral.min_num_spks = num_speakers
            spectral.max_num_spks = num_speakers

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            labels = spectral(embeddings, oracle_num=num_speakers)

        # Reset min/max
        if num_speakers is not None:
            spectral.min_num_spks = self.min_num_spks
            spectral.max_num_spks = self.max_num_spks

        # Merge similar speakers if no oracle
        if num_speakers is None:
            labels = self._merge_by_cos(labels, embeddings, self.merge_thr)

        # Re-index labels sequentially
        _, labels = np.unique(labels, return_inverse=True)

        return labels

    def _merge_by_cos(self, labels: np.ndarray, embs: np.ndarray, cos_thr: float) -> np.ndarray:
        """Merge similar speakers by cosine similarity of centroids."""
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return labels

        # Compute normalized speaker centroids
        centroids = np.array([embs[labels == lbl].mean(0) for lbl in unique_labels])
        centroids = normalize(centroids)

        # Hierarchical clustering with cosine distance
        distances = pdist(centroids, metric="cosine")
        linkage_matrix = linkage(distances, method="average")
        merged_labels = fcluster(linkage_matrix, t=1.0 - cos_thr, criterion="distance") - 1

        # Map original labels to merged labels
        label_map = dict(zip(unique_labels, merged_labels))
        return np.array([label_map[lbl] for lbl in labels])


class LocalSpeakerDiarizer:
    """Local speaker diarization using TEN-VAD + ECAPA-TDNN + spectral clustering.

    Pipeline:
    1. TEN-VAD detects speech segments
    2. Sliding window (1.0s, 75% overlap) for uniform embedding extraction
    3. ECAPA-TDNN extracts speaker embeddings per window
    4. Spectral clustering with eigenvalue gap for auto speaker detection
    5. Frame-level consensus voting for segment reconstruction
    6. Post-processing merges short segments to reduce flicker

    Tunable Parameters (class attributes):
    - WINDOW_SIZE: Embedding extraction window size in seconds
    - STEP_SIZE: Sliding window step size (overlap = WINDOW_SIZE - STEP_SIZE)
    - VAD_THRESHOLD: Speech detection threshold (lower = more sensitive)
    - VAD_MIN_DURATION: Minimum speech segment duration
    - VAD_MAX_GAP: Maximum gap to bridge between segments
    - VAD_PAD_ONSET/OFFSET: Padding added to speech segments
    - VOTING_RATE: Frame resolution for consensus voting
    - MIN_SEGMENT_DURATION: Minimum final segment duration
    - SAME_SPEAKER_GAP: Maximum gap to merge same-speaker segments
    - TAIL_COVERAGE_RATIO: Minimum tail coverage to add extra window
    """

    _ten_vad_model = None
    _ecapa_model = None
    _device = None

    # ==================== TUNABLE PARAMETERS ====================

    # Sliding window for embedding extraction
    WINDOW_SIZE = 0.75  # seconds - shorter window for finer resolution
    STEP_SIZE = 0.15  # seconds (80% overlap for more votes)
    TAIL_COVERAGE_RATIO = 0.1  # Add extra window if tail > this ratio of window

    # VAD hysteresis parameters
    VAD_THRESHOLD = 0.25  # Balanced threshold
    VAD_MIN_DURATION = 0.05  # Minimum speech segment duration (seconds)
    VAD_MAX_GAP = 0.50  # Bridge gaps shorter than this (seconds)
    VAD_PAD_ONSET = 0.05  # Padding at segment start (seconds)
    VAD_PAD_OFFSET = 0.05  # Padding at segment end (seconds)

    # Frame-level voting
    VOTING_RATE = 0.01  # 10ms resolution for consensus voting

    # Post-processing
    MIN_SEGMENT_DURATION = 0.15  # Minimum final segment duration (seconds)
    SHORT_SEGMENT_GAP = 0.1  # Gap threshold for merging short segments
    SAME_SPEAKER_GAP = 0.5  # Gap threshold for merging same-speaker segments

    # ===========================================================

    @classmethod
    def _get_ten_vad_model(cls):
        """Lazy-load TEN-VAD model (singleton)."""
        if cls._ten_vad_model is None:
            from ten_vad import TenVad

            cls._ten_vad_model = TenVad(hop_size=256, threshold=cls.VAD_THRESHOLD)
        return cls._ten_vad_model

    @classmethod
    def _get_device(cls) -> torch.device:
        """Get the best available device."""
        if cls._device is None:
            cls._device = _get_device()
        return cls._device

    @classmethod
    def _get_ecapa_model(cls):
        """Lazy-load ECAPA-TDNN speaker embedding model (singleton)."""
        if cls._ecapa_model is None:
            # Suppress torchaudio deprecation warning from SpeechBrain
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="torchaudio._backend")
                from speechbrain.inference.speaker import EncoderClassifier

                device = cls._get_device()
                cls._ecapa_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": str(device)},
                )

        return cls._ecapa_model

    @classmethod
    def diarize(
        cls,
        audio: np.ndarray | str,
        sample_rate: int = 16000,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int = 10,
        **_kwargs,
    ) -> list[dict]:
        """Run speaker diarization on audio.

        Args:
            audio: Audio waveform as numpy array or path to audio file
            sample_rate: Audio sample rate (default 16000)
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of dicts with 'speaker', 'start', 'end' keys
        """
        # Handle file path input
        if isinstance(audio, str):
            import librosa

            audio, sample_rate = librosa.load(audio, sr=16000)

        # Ensure correct sample rate
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        audio = audio.astype(np.float32)
        total_duration = len(audio) / sample_rate

        # Step 1: VAD (returns segments and raw frame-level decisions)
        segments, vad_frames = cls._get_speech_segments(audio, sample_rate)
        if not segments:
            return []

        # Step 2: Extract embeddings
        embeddings, window_segments = cls._extract_embeddings(audio, segments, sample_rate)
        if len(embeddings) == 0:
            return []

        # Step 3: Cluster
        clusterer = SpeakerClusterer(min_num_spks=min_speakers, max_num_spks=max_speakers)
        labels = clusterer(embeddings, num_speakers)

        # Step 4: Post-process with consensus voting (VAD-aware)
        return cls._postprocess_segments(window_segments, labels, total_duration, vad_frames)

    @classmethod
    def _get_speech_segments(
        cls, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> tuple[list[dict], list[bool]]:
        """Get speech segments using TEN-VAD.

        Returns:
            Tuple of (segments list, vad_frames list of per-frame speech decisions)
        """
        vad_model = cls._get_ten_vad_model()

        # Convert to int16 as required by TEN-VAD
        # Clip to prevent integer overflow
        if audio_array.dtype != np.int16:
            audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            audio_int16 = audio_array

        # Process frame by frame
        hop_size = 256
        frame_duration = hop_size / sample_rate
        speech_frames: list[bool] = []

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

        return cls._apply_vad_hysteresis(segments, sample_rate), speech_frames

    @classmethod
    def _apply_vad_hysteresis(cls, segments: list[dict], sample_rate: int = 16000) -> list[dict]:
        """Apply hysteresis-like post-processing to VAD segments."""
        if not segments:
            return segments

        segments = sorted(segments, key=lambda x: x["start"])

        # Fill short gaps
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap <= cls.VAD_MAX_GAP:
                merged[-1]["end"] = seg["end"]
                merged[-1]["end_sample"] = seg["end_sample"]
            else:
                merged.append(seg.copy())

        # Remove short segments
        filtered = [seg for seg in merged if (seg["end"] - seg["start"]) >= cls.VAD_MIN_DURATION]

        # Dilate segments (add padding)
        for seg in filtered:
            seg["start"] = max(0.0, seg["start"] - cls.VAD_PAD_ONSET)
            seg["end"] = seg["end"] + cls.VAD_PAD_OFFSET
            seg["start_sample"] = int(seg["start"] * sample_rate)
            seg["end_sample"] = int(seg["end"] * sample_rate)

        return filtered

    @classmethod
    def _extract_embeddings(
        cls, audio_array: np.ndarray, segments: list[dict], sample_rate: int
    ) -> tuple[np.ndarray, list[dict]]:
        """Extract speaker embeddings using sliding windows."""
        speaker_model = cls._get_ecapa_model()

        window_samples = int(cls.WINDOW_SIZE * sample_rate)
        step_samples = int(cls.STEP_SIZE * sample_rate)

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

                    # Cover tail if > TAIL_COVERAGE_RATIO of window remains
                    if ends and ends[-1] < seg_end:
                        remainder = seg_end - ends[-1]
                        if remainder > (window_samples * cls.TAIL_COVERAGE_RATIO):
                            starts.append(seg_end - window_samples)
                            ends.append(seg_end)

                for c_start, c_end in zip(starts, ends):
                    chunk = audio_array[c_start:c_end]

                    # Pad short chunks with reflection
                    if len(chunk) < window_samples:
                        pad_width = window_samples - len(chunk)
                        chunk = np.pad(chunk, (0, pad_width), mode="reflect")

                    # Extract embedding using SpeechBrain's encode_batch
                    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                    embedding = (
                        speaker_model.encode_batch(chunk_tensor).squeeze(0).squeeze(0).cpu().numpy()
                    )

                    # Validate embedding
                    if np.isfinite(embedding).all() and np.linalg.norm(embedding) > 1e-8:
                        embeddings.append(embedding)
                        window_segments.append(
                            {
                                "start": c_start / sample_rate,
                                "end": c_end / sample_rate,
                            }
                        )

        # Normalize all embeddings at once
        if embeddings:
            return normalize(np.array(embeddings)), window_segments
        return np.array([]), []

    @classmethod
    def _resample_vad(cls, vad_frames: list[bool], num_frames: int) -> np.ndarray:
        """Resample VAD frame decisions to match voting grid resolution.

        VAD operates at 256 samples / 16000 Hz = 16ms per frame.
        Voting operates at VOTING_RATE (default 10ms) per frame.
        This maps VAD decisions to the finer voting grid.
        """
        if not vad_frames:
            return np.zeros(num_frames, dtype=bool)

        vad_rate = 256 / 16000  # 16ms per VAD frame
        vad_arr = np.array(vad_frames)

        # Vectorized: compute VAD frame indices for each voting frame
        voting_times = np.arange(num_frames) * cls.VOTING_RATE
        vad_indices = np.clip((voting_times / vad_rate).astype(int), 0, len(vad_arr) - 1)
        return vad_arr[vad_indices]

    @classmethod
    def _postprocess_segments(
        cls,
        window_segments: list[dict],
        labels: np.ndarray,
        total_duration: float,
        vad_frames: list[bool],
    ) -> list[dict]:
        """Post-process using frame-level consensus voting with VAD-aware silence."""
        if not window_segments or len(labels) == 0:
            return []

        # Correct labels to be contiguous
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        clean_labels = np.array([label_map[lbl] for lbl in labels])
        num_speakers = len(unique_labels)

        if num_speakers == 0:
            return []

        # Create voting grid
        num_frames = int(np.ceil(total_duration / cls.VOTING_RATE)) + 1
        votes = np.zeros((num_frames, num_speakers), dtype=np.float32)

        # Accumulate votes
        for win, label in zip(window_segments, clean_labels):
            start_frame = int(win["start"] / cls.VOTING_RATE)
            end_frame = int(win["end"] / cls.VOTING_RATE)
            end_frame = min(end_frame, num_frames)
            if start_frame < end_frame:
                votes[start_frame:end_frame, label] += 1.0

        # Determine winner per frame
        frame_speakers = np.argmax(votes, axis=1)
        max_votes = np.max(votes, axis=1)

        # Resample VAD to voting grid resolution for silence-aware voting
        vad_resampled = cls._resample_vad(vad_frames, num_frames)

        # Convert frames to segments
        final_segments = []
        current_speaker = -1
        seg_start = 0.0

        for f in range(num_frames):
            speaker = int(frame_speakers[f])
            score = max_votes[f]

            # Force silence if VAD says no speech OR no votes
            if score == 0 or not vad_resampled[f]:
                speaker = -1

            if speaker != current_speaker:
                if current_speaker != -1:
                    final_segments.append(
                        {
                            "speaker": f"SPEAKER_{current_speaker}",
                            "start": seg_start,
                            "end": f * cls.VOTING_RATE,
                        }
                    )
                current_speaker = speaker
                seg_start = f * cls.VOTING_RATE

        # Close last segment
        if current_speaker != -1:
            final_segments.append(
                {
                    "speaker": f"SPEAKER_{current_speaker}",
                    "start": seg_start,
                    "end": num_frames * cls.VOTING_RATE,
                }
            )

        return cls._merge_short_segments(final_segments)

    @classmethod
    def _merge_short_segments(cls, segments: list[dict]) -> list[dict]:
        """Merge short segments to reduce flicker."""
        if not segments:
            return []

        clean: list[dict] = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            if dur < cls.MIN_SEGMENT_DURATION:
                if (
                    clean
                    and clean[-1]["speaker"] == seg["speaker"]
                    and seg["start"] - clean[-1]["end"] < cls.SHORT_SEGMENT_GAP
                ):
                    clean[-1]["end"] = seg["end"]
                continue

            if (
                clean
                and clean[-1]["speaker"] == seg["speaker"]
                and seg["start"] - clean[-1]["end"] < cls.SAME_SPEAKER_GAP
            ):
                clean[-1]["end"] = seg["end"]
            else:
                clean.append(seg)

        return clean

    @classmethod
    def assign_speakers_to_words(
        cls,
        words: list[dict],
        speaker_segments: list[dict],
    ) -> list[dict]:
        """Assign speaker labels to words based on timestamp overlap.

        Args:
            words: List of word dicts with 'word', 'start', 'end' keys
            speaker_segments: List of speaker dicts with 'speaker', 'start', 'end' keys

        Returns:
            Words list with 'speaker' key added to each word
        """
        for word in words:
            word_mid = (word["start"] + word["end"]) / 2

            # Find the speaker segment that contains this word's midpoint
            best_speaker = None
            for seg in speaker_segments:
                if seg["start"] <= word_mid <= seg["end"]:
                    best_speaker = seg["speaker"]
                    break

            # If no exact match, find closest segment
            if best_speaker is None and speaker_segments:
                min_dist = float("inf")
                for seg in speaker_segments:
                    seg_mid = (seg["start"] + seg["end"]) / 2
                    dist = abs(word_mid - seg_mid)
                    if dist < min_dist:
                        min_dist = dist
                        best_speaker = seg["speaker"]

            word["speaker"] = best_speaker

        return words


class SpeakerDiarizer:
    """Speaker diarization using TEN-VAD + ECAPA-TDNN + spectral clustering.

    Example:
        >>> segments = SpeakerDiarizer.diarize(audio_array)
        >>> for seg in segments:
        ...     print(f"{seg['speaker']}: {seg['start']:.2f} - {seg['end']:.2f}")
    """

    @classmethod
    def diarize(
        cls,
        audio: np.ndarray | str,
        sample_rate: int = 16000,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        **_kwargs,
    ) -> list[dict]:
        """Run speaker diarization on audio.

        Args:
            audio: Audio waveform as numpy array or path to audio file
            sample_rate: Audio sample rate (default 16000)
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of dicts with 'speaker', 'start', 'end' keys
        """
        return LocalSpeakerDiarizer.diarize(
            audio,
            sample_rate=sample_rate,
            num_speakers=num_speakers,
            min_speakers=min_speakers or 2,
            max_speakers=max_speakers or 10,
        )

    @classmethod
    def assign_speakers_to_words(
        cls,
        words: list[dict],
        speaker_segments: list[dict],
    ) -> list[dict]:
        """Assign speaker labels to words based on timestamp overlap."""
        return LocalSpeakerDiarizer.assign_speakers_to_words(words, speaker_segments)
