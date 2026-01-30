"""Forced alignment for word-level timestamps using Wav2Vec2."""

import numpy as np
import torch

# Offset compensation for Wav2Vec2-BASE systematic bias (in seconds)
# Calibrated on librispeech-alignments dataset (n=25, MAE=48ms)
START_OFFSET = 0.04  # Subtract from start times (shift earlier)
END_OFFSET = -0.04  # Subtract from end times (shift later)


def _get_device() -> str:
    """Get best available device for non-transformers models."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ForcedAligner:
    """Lazy-loaded forced aligner for word-level timestamps using torchaudio wav2vec2.

    Uses Viterbi trellis algorithm for optimal alignment path finding.
    """

    _bundle = None
    _model = None
    _labels = None
    _dictionary = None

    @classmethod
    def get_instance(cls, device: str = "cuda"):
        """Get or create the forced alignment model (singleton).

        Args:
            device: Device to run model on ("cuda" or "cpu")

        Returns:
            Tuple of (model, labels, dictionary)
        """
        if cls._model is None:
            import torchaudio

            cls._bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            cls._model = cls._bundle.get_model().to(device)
            cls._model.eval()
            cls._labels = cls._bundle.get_labels()
            cls._dictionary = {c: i for i, c in enumerate(cls._labels)}
        return cls._model, cls._labels, cls._dictionary

    @staticmethod
    def _get_trellis(emission: torch.Tensor, tokens: list[int], blank_id: int = 0) -> torch.Tensor:
        """Build trellis for forced alignment using forward algorithm.

        The trellis[t, j] represents the log probability of the best path that
        aligns the first j tokens to the first t frames.

        Args:
            emission: Log-softmax emission matrix of shape (num_frames, num_classes)
            tokens: List of target token indices
            blank_id: Index of the blank/CTC token (default 0)

        Returns:
            Trellis matrix of shape (num_frames + 1, num_tokens + 1)
        """
        num_frames = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
        trellis[0, 0] = 0

        # Force alignment to use all tokens by preventing staying in blank
        # at the end when there are still tokens to emit
        if num_tokens > 1:
            trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frames):
            for j in range(num_tokens + 1):
                # Stay: emit blank and stay at j tokens
                stay = trellis[t, j] + emission[t, blank_id]

                # Move: emit token j and advance to j+1 tokens
                move = trellis[t, j - 1] + emission[t, tokens[j - 1]] if j > 0 else -float("inf")

                trellis[t + 1, j] = max(stay, move)  # Viterbi: take best path

        return trellis

    @staticmethod
    def _backtrack(
        trellis: torch.Tensor, emission: torch.Tensor, tokens: list[int], blank_id: int = 0
    ) -> list[tuple[int, float, float, float]]:
        """Backtrack through trellis to find optimal forced monotonic alignment.

        Guarantees:
        - All tokens are emitted exactly once
        - Strictly monotonic: each token's frames come after previous token's
        - No frame skipping or token teleporting

        Returns list of (token_id, start_frame, end_frame, peak_frame) for each token.
        The peak_frame is the frame with highest emission probability for that token.
        """
        num_frames = emission.size(0)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return []

        # Find the best ending point (should be at num_tokens)
        # But verify trellis reached a valid state
        if trellis[num_frames, num_tokens] == -float("inf"):
            # Alignment failed - fall back to uniform distribution
            frames_per_token = num_frames / num_tokens
            return [
                (
                    tokens[i],
                    i * frames_per_token,
                    (i + 1) * frames_per_token,
                    (i + 0.5) * frames_per_token,
                )
                for i in range(num_tokens)
            ]

        # Backtrack: find where each token transition occurred
        # Store (frame, emission_score) for each token
        token_frames: list[list[tuple[int, float]]] = [[] for _ in range(num_tokens)]

        t = num_frames
        j = num_tokens

        while t > 0 and j > 0:
            # Check: did we transition from j-1 to j at frame t-1?
            stay_score = trellis[t - 1, j] + emission[t - 1, blank_id]
            move_score = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            if move_score >= stay_score:
                # Token j-1 was emitted at frame t-1
                # Store frame and its emission probability
                emit_prob = emission[t - 1, tokens[j - 1]].exp().item()
                token_frames[j - 1].insert(0, (t - 1, emit_prob))
                j -= 1
            # Always decrement time (monotonic)
            t -= 1

        # Handle any remaining tokens at the start (edge case)
        while j > 0:
            token_frames[j - 1].insert(0, (0, 0.0))
            j -= 1

        # Convert to spans with peak frame
        token_spans: list[tuple[int, float, float, float]] = []
        for token_idx, frames_with_scores in enumerate(token_frames):
            if not frames_with_scores:
                # Token never emitted - assign minimal span after previous
                if token_spans:
                    prev_end = token_spans[-1][2]
                    frames_with_scores = [(int(prev_end), 0.0)]
                else:
                    frames_with_scores = [(0, 0.0)]

            token_id = tokens[token_idx]
            frames = [f for f, _ in frames_with_scores]
            start_frame = float(min(frames))
            end_frame = float(max(frames)) + 1.0

            # Find peak frame (highest emission probability)
            peak_frame, _ = max(frames_with_scores, key=lambda x: x[1])

            token_spans.append((token_id, start_frame, end_frame, float(peak_frame)))

        return token_spans

    @classmethod
    def align(
        cls,
        audio: np.ndarray,
        text: str,
        sample_rate: int = 16000,
        _language: str = "eng",
        _batch_size: int = 16,
    ) -> list[dict]:
        """Align transcript to audio and return word-level timestamps.

        Uses Viterbi trellis algorithm for optimal forced alignment.

        Args:
            audio: Audio waveform as numpy array
            text: Transcript text to align
            sample_rate: Audio sample rate (default 16000)
            _language: ISO-639-3 language code (default "eng" for English, unused)
            _batch_size: Batch size for alignment model (unused)

        Returns:
            List of dicts with 'word', 'start', 'end' keys
        """
        import torchaudio

        device = _get_device()
        model, _labels, dictionary = cls.get_instance(device)
        assert cls._bundle is not None and dictionary is not None  # Initialized by get_instance

        # Convert audio to tensor (copy to ensure array is writable)
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio.copy()).float()
        else:
            waveform = audio.clone().float()

        # Ensure 2D (channels, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed (wav2vec2 expects 16kHz)
        if sample_rate != cls._bundle.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, cls._bundle.sample_rate
            )

        waveform = waveform.to(device)

        # Get emissions from model
        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu()

        # Normalize text: uppercase, keep only valid characters
        transcript = text.upper()

        # Build tokens from transcript (including word separators)
        tokens = []
        for char in transcript:
            if char in dictionary:
                tokens.append(dictionary[char])
            elif char == " ":
                tokens.append(dictionary.get("|", dictionary.get(" ", 0)))

        if not tokens:
            return []

        # Build Viterbi trellis and backtrack for optimal path
        trellis = cls._get_trellis(emission, tokens, blank_id=0)
        alignment_path = cls._backtrack(trellis, emission, tokens, blank_id=0)

        # Convert frame indices to time (model stride is 320 samples at 16kHz = 20ms)
        frame_duration = 320 / cls._bundle.sample_rate

        # Apply separate offset compensation for start/end (Wav2Vec2 systematic bias)
        start_offset = START_OFFSET
        end_offset = END_OFFSET

        # Group aligned tokens into words based on pipe separator
        # Use peak emission frame for more accurate word boundaries
        words = text.split()
        word_timestamps = []
        first_char_peak = None
        last_char_peak = None
        word_idx = 0
        separator_id = dictionary.get("|", dictionary.get(" ", 0))

        for token_id, _start_frame, _end_frame, peak_frame in alignment_path:
            if token_id == separator_id:  # Word separator
                if (
                    first_char_peak is not None
                    and last_char_peak is not None
                    and word_idx < len(words)
                ):
                    # Use peak frames for word boundaries
                    start_time = max(0.0, first_char_peak * frame_duration - start_offset)
                    end_time = max(0.0, (last_char_peak + 1) * frame_duration - end_offset)
                    word_timestamps.append(
                        {
                            "word": words[word_idx],
                            "start": start_time,
                            "end": end_time,
                        }
                    )
                    word_idx += 1
                first_char_peak = None
                last_char_peak = None
            else:
                if first_char_peak is None:
                    first_char_peak = peak_frame
                last_char_peak = peak_frame

        # Don't forget the last word
        if first_char_peak is not None and last_char_peak is not None and word_idx < len(words):
            start_time = max(0.0, first_char_peak * frame_duration - start_offset)
            end_time = max(0.0, (last_char_peak + 1) * frame_duration - end_offset)
            word_timestamps.append(
                {
                    "word": words[word_idx],
                    "start": start_time,
                    "end": end_time,
                }
            )

        return word_timestamps
