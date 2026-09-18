"""Forced alignment for word-level timestamps using Wav2Vec2."""

import numpy as np
import torch


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

        # Each row depends only on the previous row, so the token axis vectorizes.
        # blank[t] is the stay cost; emit[t, j] the cost of emitting tokens[j].
        blank = emission[:, blank_id]
        emit = emission.index_select(1, torch.as_tensor(tokens, dtype=torch.long))

        for t in range(num_frames):
            # j == 0: only staying on blank is reachable.
            trellis[t + 1, 0] = trellis[t, 0] + blank[t]
            if num_tokens:
                # Viterbi over j >= 1: best of staying at j or advancing from j-1.
                trellis[t + 1, 1:] = torch.maximum(
                    trellis[t, 1:] + blank[t], trellis[t, :-1] + emit[t]
                )

        return trellis

    @staticmethod
    def _backtrack(
        trellis: torch.Tensor, emission: torch.Tensor, tokens: list[int], blank_id: int = 0
    ) -> list[tuple[int, float, float]]:
        """Backtrack through trellis to find optimal forced monotonic alignment.

        Guarantees:
        - All tokens are emitted exactly once
        - Strictly monotonic: each token's frames come after previous token's
        - No frame skipping or token teleporting

        Returns list of (token_id, start_frame, end_frame) for each token.
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
                (tokens[i], i * frames_per_token, (i + 1) * frames_per_token)
                for i in range(num_tokens)
            ]

        # Backtrack: find where each token transition occurred
        # path[i] = frame where token i was first emitted
        token_frames: list[list[int]] = [[] for _ in range(num_tokens)]

        t = num_frames
        j = num_tokens

        while t > 0 and j > 0:
            # Check: did we transition from j-1 to j at frame t-1?
            stay_score = trellis[t - 1, j] + emission[t - 1, blank_id]
            move_score = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            if move_score >= stay_score:
                # Token j-1 was emitted at frame t-1
                token_frames[j - 1].append(t - 1)
                j -= 1
            t -= 1

        # Handle any remaining tokens at the start (edge case)
        while j > 0:
            token_frames[j - 1].append(0)
            j -= 1

        # We appended in reverse-time order; restore monotonic order
        for frames in token_frames:
            frames.reverse()

        # Convert to spans
        token_spans: list[tuple[int, float, float]] = []
        for token_idx, emitted_frames in enumerate(token_frames):
            frames = emitted_frames
            if not frames:
                # Token never emitted - assign minimal span after previous
                frames = [int(token_spans[-1][2])] if token_spans else [0]

            token_id = tokens[token_idx]
            start_frame = float(min(frames))
            end_frame = float(max(frames)) + 1.0
            token_spans.append((token_id, start_frame, end_frame))

        return token_spans

    # Offset compensation for Wav2Vec2-BASE systematic bias (in seconds)
    # Calibrated on librispeech-alignments dataset
    START_OFFSET = 0.06  # Subtract from start times (shift earlier)
    END_OFFSET = -0.03  # Add to end times (shift later)

    @staticmethod
    def _tokenize_words(
        text: str, dictionary: dict, blank_id: int = 0
    ) -> tuple[list[str], list[int]]:
        """Split `text` into alignable words and the CTC token sequence for them.

        Returns `(words, tokens)` where `tokens` is each word's character ids
        joined by the separator id, and `words` lists only the words that
        contributed at least one token -- so the token groups the Viterbi path
        produces line up 1:1 with `words`.

        That pairing is the whole point. A word can produce no usable tokens in
        two ways, and both used to leave an empty group behind while still
        emitting the separators on either side of it:

        - every character is outside the label set (digits, "...");
        - every character maps to a control id. In wav2vec2's label set index 0
          is spelled "-", so a hyphen is the *blank* symbol, and "--" became
          two blank target tokens rather than a word.

        Pairing groups against a plain `text.split()` then shifted every later
        word onto the previous word's timing and dropped the last one:
        "hello -- world" returned `hello, --` with `--` carrying WORLD's frames.

        `blank_id` defaults to 0 to match the `blank_id=0` that `align` passes
        to `_get_trellis` and `_backtrack`.
        """
        separator_id = dictionary.get("|", dictionary.get(" ", 0))
        # Neither may appear as a target token: the separator would split the
        # word in two, and the blank is what the trellis emits *between*
        # tokens. Either one desyncs the group-to-word pairing.
        control_ids = {separator_id, blank_id}
        words: list[str] = []
        tokens: list[int] = []
        for word in text.split():
            word_tokens = [
                token_id
                for c in word.upper()
                if (token_id := dictionary.get(c)) is not None and token_id not in control_ids
            ]
            if not word_tokens:
                continue
            if tokens:
                tokens.append(separator_id)
            tokens.extend(word_tokens)
            words.append(word)
        return words, tokens

    @classmethod
    def align(
        cls,
        audio: np.ndarray | torch.Tensor,
        text: str,
        sample_rate: int = 16000,
    ) -> list[dict]:
        """Align transcript to audio and return word-level timestamps.

        Uses Viterbi trellis algorithm for optimal forced alignment.

        Args:
            audio: Audio waveform as a numpy array or torch tensor
            text: Transcript text to align
            sample_rate: Audio sample rate (default 16000)

        Returns:
            List of dicts with 'word', 'start', 'end' keys
        """
        import torchaudio

        device = _get_device()
        model, _labels, dictionary = cls.get_instance(device)
        assert cls._bundle is not None
        assert dictionary is not None

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

        # Tokenize per word so each Viterbi token group maps back to the word
        # it came from; words with no representable characters are dropped.
        words, tokens = cls._tokenize_words(text, dictionary)
        if not tokens:
            return []

        # Build Viterbi trellis and backtrack for optimal path
        trellis = cls._get_trellis(emission, tokens, blank_id=0)
        alignment_path = cls._backtrack(trellis, emission, tokens, blank_id=0)

        # Convert frame indices to time (model stride is 320 samples at 16kHz = 20ms)
        frame_duration = 320 / cls._bundle.sample_rate

        # Apply separate offset compensation for start/end (Wav2Vec2 systematic bias)
        start_offset = cls.START_OFFSET
        end_offset = cls.END_OFFSET

        # Group aligned tokens into words based on pipe separator
        word_timestamps = []
        current_word_start = None
        current_word_end = None
        word_idx = 0
        separator_id = dictionary.get("|", dictionary.get(" ", 0))

        for token_id, start_frame, end_frame in alignment_path:
            if token_id == separator_id:  # Word separator
                if (
                    current_word_start is not None
                    and current_word_end is not None
                    and word_idx < len(words)
                ):
                    start_time = max(0.0, current_word_start * frame_duration - start_offset)
                    end_time = max(0.0, current_word_end * frame_duration - end_offset)
                    word_timestamps.append(
                        {
                            "word": words[word_idx],
                            "start": start_time,
                            "end": end_time,
                        }
                    )
                    word_idx += 1
                current_word_start = None
                current_word_end = None
            else:
                if current_word_start is None:
                    current_word_start = start_frame
                current_word_end = end_frame

        # Don't forget the last word
        if (
            current_word_start is not None
            and current_word_end is not None
            and word_idx < len(words)
        ):
            start_time = max(0.0, current_word_start * frame_duration - start_offset)
            end_time = max(0.0, current_word_end * frame_duration - end_offset)
            word_timestamps.append(
                {
                    "word": words[word_idx],
                    "start": start_time,
                    "end": end_time,
                }
            )

        return word_timestamps
