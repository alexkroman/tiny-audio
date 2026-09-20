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

    The CTC Viterbi search itself is `torchaudio.functional.forced_align`; this
    class owns the model singleton, the word-level tokenization and the
    frame-to-seconds conversion around it.
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
    def _align_tokens(
        emission: torch.Tensor, tokens: list[int], blank_id: int = 0
    ) -> list[tuple[int, float, float]]:
        """Viterbi-align `tokens` to `emission` and return one span per token.

        Delegates to `torchaudio.functional.forced_align` (a C++/CUDA CTC
        Viterbi) and `merge_tokens`, which collapse the per-frame path into
        `(token_id, start_frame, end_frame)` spans with `end_frame` exclusive.

        Guarantees:
        - All tokens are emitted exactly once, in order (strictly monotonic).
        - When no valid path exists -- more tokens than frames, or an emission
          that assigns every path zero probability -- the tokens are spread
          uniformly over the frames instead of raising, so a timestamp request
          degrades to coarse timings rather than failing the transcription.

        Args:
            emission: Log-softmax emission matrix of shape (num_frames, num_classes)
            tokens: Target token indices
            blank_id: Index of the CTC blank (default 0)
        """
        import torchaudio.functional as F  # noqa: N812

        num_frames = emission.size(0)
        num_tokens = len(tokens)
        if num_tokens == 0:
            return []

        def _uniform() -> list[tuple[int, float, float]]:
            frames_per_token = num_frames / num_tokens
            return [
                (tokens[i], i * frames_per_token, (i + 1) * frames_per_token)
                for i in range(num_tokens)
            ]

        log_probs = emission.detach().float().unsqueeze(0)
        targets = torch.tensor([tokens], dtype=torch.int32, device=log_probs.device)
        try:
            aligned, scores = F.forced_align(log_probs, targets, blank=blank_id)
        except RuntimeError:
            # "targets length is too long for CTC": no monotonic path fits.
            return _uniform()
        if not torch.isfinite(scores).all():
            # Every path has -inf log-probability; the returned path is junk.
            return _uniform()

        spans = F.merge_tokens(aligned[0], scores[0].exp(), blank=blank_id)
        if len(spans) != num_tokens:
            return _uniform()
        return [(int(span.token), float(span.start), float(span.end)) for span in spans]

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
        to `_align_tokens`.
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

        alignment_path = cls._align_tokens(emission, tokens, blank_id=0)

        # Convert frame indices to time (model stride is 320 samples at 16kHz = 20ms)
        frame_duration = 320 / cls._bundle.sample_rate

        # Group token spans into words on the separator token. `_tokenize_words`
        # guarantees one non-empty group per entry in `words`, so the groups
        # zip 1:1 with the word list.
        separator_id = dictionary.get("|", dictionary.get(" ", 0))
        groups: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for token_id, start_frame, end_frame in alignment_path:
            if token_id == separator_id:
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append((start_frame, end_frame))
        if current:
            groups.append(current)

        # Apply separate offset compensation for start/end (Wav2Vec2 systematic bias)
        word_timestamps = []
        for word, frames in zip(words, groups):
            start_frame, end_frame = frames[0][0], frames[-1][1]
            word_timestamps.append(
                {
                    "word": word,
                    "start": max(0.0, start_frame * frame_duration - cls.START_OFFSET),
                    "end": max(0.0, end_frame * frame_duration - cls.END_OFFSET),
                }
            )

        return word_timestamps
