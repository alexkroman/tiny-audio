"""ASR pipeline for audio-to-text transcription with optional timestamps and diarization."""

import re
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch
import transformers
from transformers.pipelines.audio_utils import ffmpeg_read

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

    from .alignment import QwenForcedAligner
    from .asr_modeling import ASRModel
    from .asr_processing import prepend_lead_in
    from .diarization import NemotronDiarizer, StreamChunk, masked_audio, pack_spans
else:
    try:
        from .alignment import QwenForcedAligner
        from .asr_modeling import ASRModel
        from .asr_processing import prepend_lead_in
        from .diarization import NemotronDiarizer, StreamChunk, masked_audio, pack_spans
    except ImportError:  # flat layout on the Hub: sibling modules, no package
        from alignment import QwenForcedAligner
        from asr_modeling import ASRModel
        from asr_processing import prepend_lead_in
        from diarization import NemotronDiarizer, StreamChunk, masked_audio, pack_spans

# Re-export for backwards compatibility
__all__ = [
    "ASRPipeline",
    "NemotronDiarizer",
    "QwenForcedAligner",
]

# Audio is transcribed in chunks cut at the quietest point between
# these lengths. The model trained on clips of at most 19 s; 18 leaves room for
# the inference lead-in. Short clips are one chunk, so their text is unchanged.
CHUNK_MAX_S = 18.0
CHUNK_MIN_S = 8.0


def chunk_bounds(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    max_s: float = CHUNK_MAX_S,
    min_s: float = CHUNK_MIN_S,
) -> list[tuple[int, int]]:
    """Sample ranges of at most `max_s`, each cut at the quietest 100 ms frame after `min_s`."""
    frame = int(0.1 * sample_rate)
    bounds: list[tuple[int, int]] = []
    start, n = 0, len(audio)
    while n - start > max_s * sample_rate:
        lo = start + int(min_s * sample_rate)
        hi = start + int(max_s * sample_rate)
        k = (hi - lo) // frame
        rms = np.sqrt(np.mean(np.square(audio[lo : lo + k * frame].reshape(k, frame)), axis=1))
        cut = lo + int(np.argmin(rms)) * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


def stream_chunks(
    audio: npt.NDArray[np.float32],
    spans: list[tuple[int, int]],
    sample_rate: int,
    max_s: float = CHUNK_MAX_S,
) -> list[tuple[int, int]]:
    """Ranges of at most `max_s` over one speaker's `spans`; long turns cut by `chunk_bounds`."""
    pieces: list[tuple[int, int]] = []
    for s, e in spans:
        pieces.extend((s + a, s + b) for a, b in chunk_bounds(audio[s:e], sample_rate, max_s))
    return pack_spans(pieces, int(max_s * sample_rate))


# Below this RMS (-100 dBFS) a chunk is digital silence: exact zeros, as in
# edited or remixed recordings. Given one, the model answers with a memorized
# training sentence ("The film was directed by the same director who directed
# 'The Man with the Moustache'") -- ten such chunks cost 2.3 WER on one AMI
# meeting -- so it is skipped. Quiet real speech sits near -60 dBFS.
SILENCE_RMS = 1e-5


def is_silent(audio: npt.NDArray[np.float32]) -> bool:
    """True for digital silence (or an empty array): nothing for the model to hear."""
    return audio.size == 0 or float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS


_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_MIN_REPEATS = 3
_TRAILING_CHAR_RE = re.compile(rf"(.)\1{{{_MIN_REPEATS - 1},}}$")
_TRAILING_WORD_RE = re.compile(rf"\b(\w+)(?:\s+\1){{{_MIN_REPEATS - 1},}}\s*$", re.IGNORECASE)


class _RawAudio(TypedDict):
    """A waveform and its sampling rate, as `_extract_audio` reads them."""

    array: npt.NDArray[np.float32]
    sampling_rate: int


# `__call__` keywords this pipeline handles itself; the parent never sees them.
_CUSTOM_PARAMS = (
    "return_speakers",
    "num_speakers",
    "min_speakers",
    "max_speakers",
    "hf_token",
    "user_prompt",
)


class _DiarizationParams(TypedDict):
    """Speaker-count hints forwarded to `NemotronDiarizer.top_speakers`."""

    num_speakers: int | None
    max_speakers: int | None


class ASRPipeline(transformers.AutomaticSpeechRecognitionPipeline):
    """ASR Pipeline for audio-to-text transcription."""

    model: ASRModel

    def __init__(self, model: ASRModel, **kwargs: Any) -> None:
        """Initialize ASR pipeline.

        Args:
            model: ASRModel instance for transcription
            **kwargs: Additional arguments (feature_extractor, tokenizer, device)
        """
        feature_extractor: SequenceFeatureExtractor | None = kwargs.pop("feature_extractor", None)
        tokenizer: PreTrainedTokenizerBase | None = kwargs.pop("tokenizer", model.tokenizer)

        if feature_extractor is None:
            feature_extractor = model.get_processor().feature_extractor

        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            # Annotated as the slow `PreTrainedTokenizer`, but any tokenizer works.
            tokenizer=tokenizer,  # pyright: ignore[reportArgumentType]
            **kwargs,
        )

    def _sanitize_parameters(
        self,
        chunk_length_s: float | None = None,
        stride_length_s: float | None = None,
        ignore_warning: bool | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
        return_timestamps: bool | str | None = None,
        return_language: bool | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Intercept our custom parameters before parent class validates them."""
        # Remove our custom parameters so parent doesn't see them.
        # `return_timestamps` means word timestamps here (handled in __call__),
        # not the parent's CTC/Whisper timestamps, so it is dropped too.
        del return_timestamps
        for name in _CUSTOM_PARAMS:
            kwargs.pop(name, None)

        return super()._sanitize_parameters(
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            ignore_warning=ignore_warning,
            decoder_kwargs=decoder_kwargs,
            return_language=return_language,
            **kwargs,
        )

    # The parent annotates `list[dict]`, but one input yields one dict (here and
    # in the parent); a batch the parent handles yields a list. Hence `Any`.
    def __call__(
        self,
        inputs: Any,
        *args: Any,
        num_workers: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Transcribe audio with optional word-level timestamps and speaker diarization.

        Args:
            inputs: Audio input (file path, dict with array/sampling_rate, etc.)
            return_timestamps: If True, return word-level timestamps (Qwen3-ForcedAligner)
            return_speakers: If True, transcribe each Nemotron-3-Diarization speaker on
                audio with the others silenced and label words by speaker (see
                `_transcribe_streams`). Implies return_timestamps.
            user_prompt: Custom transcription prompt (default: "Transcribe: ")
            num_speakers: Exact number of speakers, if known
            max_speakers: Upper bound on the number of speakers
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            Dict with 'text' key, 'words' key if return_timestamps=True,
            and speaker labels on words plus 'speaker_segments' if return_speakers=True
        """
        if args:
            msg = f"ASRPipeline.__call__ takes 1 positional input, got {1 + len(args)}"
            raise TypeError(msg)
        # Not ours: handed back to the parent pipeline's __call__ untouched.
        if num_workers is not None:
            kwargs["num_workers"] = num_workers
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        # Extract our params before super().__call__ (which will also call _sanitize_parameters)
        return_timestamps: bool = kwargs.pop("return_timestamps", False)
        return_speakers: bool = kwargs.pop("return_speakers", False)
        user_prompt: str | None = kwargs.pop("user_prompt", None)
        if kwargs.pop("min_speakers", None) is not None:
            msg = (
                "min_speakers is not supported: Nemotron-3-Diarization decides how many "
                "speakers it hears. Pass num_speakers (exact) or max_speakers instead."
            )
            raise ValueError(msg)
        diarization_params: _DiarizationParams = {
            "num_speakers": kwargs.pop("num_speakers", None),
            "max_speakers": kwargs.pop("max_speakers", None),
        }

        if return_speakers:
            return_timestamps = True

        # Set custom user prompt if provided. The restore below is in a
        # `finally` because this mutates shared model state: without it an
        # exception anywhere in the call leaves the custom prompt in place for
        # every later call on this pipeline, and the eval harness swallows that
        # exception and keeps scoring against the wrong prompt.
        original_prompt: str | None = None
        if user_prompt:
            original_prompt = self.model.TRANSCRIBE_PROMPT
            self.model.TRANSCRIBE_PROMPT = user_prompt

        try:
            if not return_timestamps:
                return self._transcribe_plain(inputs, **kwargs)
            return self._transcribe_timed(
                inputs,
                return_speakers=return_speakers,
                diarization_params=diarization_params,
                **kwargs,
            )
        finally:
            if original_prompt is not None:
                self.model.TRANSCRIBE_PROMPT = original_prompt

    def _transcribe_plain(
        self, inputs: Any, **kwargs: Any
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Transcribe in the same chunks as `_transcribe_timed`, without timing words.

        One call over a whole recording returns only its first
        `max_new_tokens` of text (WER 97% on a 30-min AMI meeting), so plain
        transcription chunks too. A short clip is one chunk: one call, the
        same result as before. Inputs `_extract_audio` can't read (a batch)
        go to the parent untouched.
        """
        audio = self._extract_audio(inputs)
        if audio is None:
            return super().__call__(inputs, **kwargs)
        array, sr = audio["array"], audio["sampling_rate"]

        bounds = chunk_bounds(array, sr)
        if len(bounds) == 1:
            return self._transcribe_chunk(array, sr, **kwargs)
        result: dict[str, Any] = {}
        texts: list[str] = []
        for s, e in bounds:
            out = self._transcribe_chunk(array[s:e], sr, **kwargs)
            texts.append(out["text"])
            # Per-step logprobs (output_scores=True) run on across chunks.
            for key in ("top1_logprob", "top2_logprob"):
                if key in out:
                    result.setdefault(key, []).extend(out[key])
        result["text"] = " ".join(t for t in texts if t)
        return result

    def _transcribe_chunk(
        self, chunk: npt.NDArray[np.float32], sample_rate: int, **kwargs: Any
    ) -> dict[str, Any]:
        """One model call on one chunk; digital silence is "" without calling the model."""
        if is_silent(chunk):
            return {"text": ""}
        # One input in, one dict out; the parent is annotated as always returning a list.
        return cast(
            "dict[str, Any]",
            super().__call__({"raw": chunk, "sampling_rate": sample_rate}, **kwargs),
        )

    def _transcribe_timed(
        self,
        inputs: Any,
        *,
        return_speakers: bool,
        diarization_params: _DiarizationParams,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Transcribe in chunks and time every word; per-speaker streams if requested.

        Chunks of 8-18 s cut at quiet points are transcribed one by one and each
        is aligned against its own transcript (batched), then offset onto the
        recording's timeline -- so audio of any length gets every word timed,
        and the model never sees a clip longer than it trained on. Speakers
        come from `_transcribe_streams`.
        """
        audio = self._extract_audio(inputs)
        if audio is None:
            msg = f"Cannot read audio from {type(inputs).__name__} for timestamps"
            raise ValueError(msg)
        array, sr = audio["array"], audio["sampling_rate"]
        if return_speakers:
            return self._transcribe_streams(array, sr, diarization_params, **kwargs)
        return self._transcribe_aligned(array, sr, **kwargs)

    def _transcribe_aligned(
        self, array: npt.NDArray[np.float32], sr: int, **kwargs: Any
    ) -> dict[str, Any]:
        """`{"text", "words"}` for the whole recording: chunk, transcribe, align, offset."""
        bounds = chunk_bounds(array, sr)
        texts: list[str] = [
            self._transcribe_chunk(array[s:e], sr, **kwargs)["text"] for s, e in bounds
        ]
        result: dict[str, Any] = {"text": " ".join(t for t in texts if t)}

        try:
            aligned = QwenForcedAligner.align_chunks(
                [(array[s:e], text) for (s, e), text in zip(bounds, texts, strict=True)],
                sample_rate=sr,
            )
            result["words"] = [
                {**w, "start": w["start"] + s / sr, "end": w["end"] + s / sr}
                for (s, _), chunk_words in zip(bounds, aligned, strict=True)
                for w in chunk_words
            ]
        except Exception as e:
            result["words"] = []
            result["timestamp_error"] = str(e)
        return result

    def _transcribe_streams(
        self,
        array: npt.NDArray[np.float32],
        sr: int,
        diarization_params: _DiarizationParams,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Per-speaker masked ASR, a zero-shot port of NeMo's `masked_asr`.

        Nemotron runs first, one pass over the whole recording, so speaker ids
        are recording-level. Each speaker is transcribed only where they talk
        (`stream_chunks`), with everyone else zeroed (`masked_audio`); words take
        their stream's speaker, and a word two streams both heard at once is
        kept once (`dedupe_words`). Crosstalk under a speaker's mask is heard.
        With at most one speaker active nothing is masked (today's exact text);
        if diarization fails, the unmasked transcript gets `diarization_error`.
        """
        try:
            activity = NemotronDiarizer.activity(array, sample_rate=sr)
            keep = NemotronDiarizer.top_speakers(activity, **diarization_params)
        except Exception as e:
            result = self._transcribe_aligned(array, sr, **kwargs)
            result["speaker_segments"] = []
            result["diarization_error"] = str(e)
            return result

        # `top_speakers` keeps only columns that are ever active, so every
        # mask has speech -- except its `[0]` fallback when nobody spoke.
        masks = {c: activity[:, c] > NemotronDiarizer.SEGMENT_THRESHOLD for c in keep}
        if len(keep) <= 1:
            result = self._transcribe_aligned(array, sr, **kwargs)
            name = NemotronDiarizer.speaker_names(keep)[keep[0]]
            result["words"] = [{**w, "speaker": name} for w in result["words"]]
        else:
            chunks: list[StreamChunk] = []
            for c in keep:
                spans = NemotronDiarizer.sample_spans(masks[c], sr, len(array))
                chunks.extend(
                    {"col": c, "start": s, "audio": masked_audio(array, spans, s, e)}
                    for s, e in stream_chunks(array, spans, sr)
                )
            texts = [self._transcribe_chunk(ch["audio"], sr, **kwargs)["text"] for ch in chunks]
            result = self._align_streams(chunks, texts, sr, masks, keep)
            if result["words"]:
                result["words"] = NemotronDiarizer.dedupe_words(result["words"], activity, keep)
                result["text"] = " ".join(w["word"] for w in result["words"])
        result["speaker_segments"] = NemotronDiarizer.segments(activity, keep)
        return result

    @staticmethod
    def _align_streams(
        chunks: list[StreamChunk],
        texts: list[str],
        sr: int,
        masks: dict[int, npt.NDArray[np.bool_]],
        keep: list[int],
    ) -> dict[str, Any]:
        """Align every stream chunk in one batch; `{"text", "words"}` merged by time.

        If alignment fails the text is the chunk transcripts by start time and
        `timestamp_error` is set.
        """
        try:
            aligned = QwenForcedAligner.align_chunks(
                [(ch["audio"], text) for ch, text in zip(chunks, texts, strict=True)],
                sample_rate=sr,
            )
        except Exception as e:
            order = sorted(range(len(chunks)), key=lambda k: chunks[k]["start"])
            text = " ".join(texts[k] for k in order if texts[k])
            return {"text": text, "words": [], "timestamp_error": str(e)}
        words = NemotronDiarizer.stream_words(chunks, aligned, sr, masks, keep)
        return {"text": " ".join(w["word"] for w in words), "words": words}

    def _extract_audio(self, inputs: object) -> _RawAudio | None:
        """Extract a float32 audio array and its rate from various input formats."""
        if isinstance(inputs, dict):
            key = "array" if "array" in inputs else "raw" if "raw" in inputs else None
            if key is None:
                return None
            array, sr = inputs[key], inputs.get("sampling_rate", 16000)
        elif isinstance(inputs, (str, bytes)):
            # File path or encoded bytes - decode with ffmpeg (same as HF pipeline)
            data = Path(inputs).read_bytes() if isinstance(inputs, str) else inputs
            array, sr = ffmpeg_read(data, sampling_rate=16000), 16000
        elif isinstance(inputs, np.ndarray):
            array, sr = inputs, 16000
        else:
            return None
        return {"array": np.asarray(array, dtype=np.float32), "sampling_rate": sr}

    def preprocess(self, *args: Any, **preprocess_params: Any) -> Iterator[dict[str, Any]]:
        """Preprocess audio inputs for the model.

        Args:
            inputs: Audio input (dict with array, file path, etc.), the one
                positional argument (see `_single_input`)
            **preprocess_params: Additional preprocessing parameters

        Yields:
            Model input dicts with input_features and attention_mask
        """
        inputs = _single_input(args, preprocess_params, "inputs")
        # Handle dict with "array" key (from datasets)
        if isinstance(inputs, dict) and "array" in inputs:
            assert self.feature_extractor is not None  # always set by __init__
            sample = inputs
            inputs = {
                "raw": sample["array"],
                "sampling_rate": sample.get("sampling_rate", self.feature_extractor.sampling_rate),
            }

        # Inference-only lead-in silence. See prepend_lead_in for the
        # measurement; applied here rather than in ASRProcessor because this
        # pipeline holds the feature extractor directly and never routes
        # through ASRProcessor.__call__, so the two cannot double-apply.
        # Require a genuine number. `float()` on a stand-in config is not safe:
        # a MagicMock coerces to 1.0, which would silently prepend a FULL
        # SECOND of silence to every clip for any caller holding a mock. Same
        # hazard `_load_audio_encoder` documents for encoder_trainable_top_layers.
        raw_lead_in = getattr(self.model.config, "inference_lead_in_seconds", 0.0)
        lead_in = (
            float(raw_lead_in)
            if isinstance(raw_lead_in, (int, float)) and not isinstance(raw_lead_in, bool)
            else 0.0
        )
        if lead_in > 0 and isinstance(inputs, dict) and "raw" in inputs:
            assert self.feature_extractor is not None  # always set by __init__
            padded: dict[str, Any] = dict(inputs)
            padded["raw"] = prepend_lead_in(
                padded["raw"],
                padded.get("sampling_rate", self.feature_extractor.sampling_rate),
                lead_in,
            )
            inputs = padded

        for item in super().preprocess(inputs, **preprocess_params):
            if "is_last" not in item:
                item["is_last"] = True
            yield item

    def _forward(self, *args: Any, **generate_kwargs: Any) -> dict[str, Any]:
        """Run model forward pass to generate transcription.

        Args:
            model_inputs: Dict with input_features and attention_mask, the one
                positional argument (see `_single_input`)
            **generate_kwargs: Generation parameters. Pass ``output_scores=True``
                (and ``return_dict_in_generate=True``, which is then implied) to
                also return per-step top-1 and top-2 log-probabilities — used by
                the eval harness's confidence metric. Backward-compatible: when
                unset, returns just token IDs as before.

        Returns:
            Dict with generated token IDs, and optionally per-step
            ``top1_logprob`` / ``top2_logprob`` tensors when scores were
            requested.
        """
        model_inputs: MutableMapping[str, Any] = _single_input(
            args, generate_kwargs, "model_inputs"
        )
        # Extract audio features and is_last flag
        is_last = model_inputs.pop("is_last", True) if isinstance(model_inputs, dict) else True

        input_features = model_inputs["input_features"].to(self.model.device)
        audio_attention_mask = model_inputs["attention_mask"].to(self.model.device)

        # Opt-in: when output_scores is requested, force return_dict_in_generate
        # so we get a GenerateOutput rather than a bare token tensor.
        want_scores = bool(generate_kwargs.get("output_scores", False))
        if want_scores:
            generate_kwargs.setdefault("return_dict_in_generate", True)

        generate_output = self.model.generate(
            input_features=input_features,
            audio_attention_mask=audio_attention_mask,
            **generate_kwargs,
        )

        # Default (no scores requested): generate returns a tensor of token IDs.
        if torch.is_tensor(generate_output):
            return {"tokens": generate_output, "is_last": is_last}

        # Scores requested: GenerateOutput dict-like with .sequences and .scores.
        # `scores` is a tuple of per-step logits tensors (batch, vocab); convert
        # each to log-probs and take top-2 to produce two short tensors over the
        # generation horizon — kept small (no full vocab) so this is cheap to
        # carry through postprocess.
        sequences = generate_output.sequences
        scores = generate_output.scores
        # This pipeline is single-item: postprocess() also reduces to
        # `tokens[0]`. Taking batch element 0 of each score tensor is
        # therefore consistent with the text we return, but only while the
        # batch really is one clip -- with batch_size > 1 every item would be
        # annotated with sample 0's confidence and nothing downstream could
        # tell. Fail loudly instead of returning a plausible wrong number.
        if sequences.shape[0] > 1:
            msg = (
                f"ASRPipeline received a batch of {sequences.shape[0]} but only "
                "returns one transcript; per-item confidence would be wrong. "
                "Call it with one clip per invocation."
            )
            raise ValueError(msg)
        top1_logprobs: list[float] = []
        top2_logprobs: list[float] = []
        if scores:
            # One batched reduction over every step, then one device transfer.
            logits = torch.stack([step_logits[0] for step_logits in scores]).float()
            steps: list[list[float]] = (
                logits.log_softmax(dim=-1).topk(k=2, dim=-1).values.cpu().tolist()
            )
            for top1, top2 in steps:
                top1_logprobs.append(top1)
                top2_logprobs.append(top2)
        return {
            "tokens": sequences,
            "top1_logprob": top1_logprobs,
            "top2_logprob": top2_logprobs,
            "is_last": is_last,
        }

    def postprocess(
        self,
        model_outputs: dict[str, Any] | list[dict[str, Any]],
        decoder_kwargs: dict[str, Any] | None = None,
        return_timestamps: bool | str | None = None,
        return_language: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert model output tokens to text.

        Args:
            model_outputs: Dict with 'tokens' key containing generated IDs
            **kwargs: Additional postprocessing parameters

        Returns:
            Dict with 'text' key containing transcription
        """
        # Handle list of outputs (from chunking)
        if isinstance(model_outputs, list):
            model_outputs = model_outputs[0] if model_outputs else {}

        tokens = model_outputs.get("tokens")
        if tokens is None:
            return super().postprocess(
                model_outputs,
                decoder_kwargs=decoder_kwargs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                **kwargs,
            )

        if torch.is_tensor(tokens):
            tokens = tokens.cpu()
            if tokens.dim() > 1:
                tokens = tokens[0]

        # Filter out eos tokens that the tokenizer doesn't recognize as special
        # (generation_config.eos_token_id may differ from tokenizer.eos_token_id)
        eos_ids: int | list[int] | None = getattr(
            self.model.generation_config, "eos_token_id", None
        )
        if eos_ids is not None:
            eos_set = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}
            token_ids: list[int] = np.asarray(tokens).tolist()
            tokens = [t for t in token_ids if t not in eos_set]

        assert self.tokenizer is not None  # always set by __init__
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        assert isinstance(decoded, str)  # one sequence in, one string out
        text = decoded.strip()
        # Strip <think>...</think> tags (Qwen3 doesn't respect /no_think prompt)
        if "<think>" in text:
            text = _THINK_TAG_RE.sub("", text).strip()
        text = _truncate_repetitions(text)
        out: dict[str, Any] = {"text": text}
        # Pass through per-step logprobs when _forward captured them (i.e. caller
        # passed output_scores=True). Lets eval harnesses compute confidence
        # stats without re-running the model.
        if "top1_logprob" in model_outputs:
            out["top1_logprob"] = model_outputs["top1_logprob"]
        if "top2_logprob" in model_outputs:
            out["top2_logprob"] = model_outputs["top2_logprob"]
        return out


def _single_input(args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
    """The one input of a pipeline stage, given positionally or as `name=`.

    `preprocess` and `_forward` take `*args` because their two transformers bases
    name that input differently (`inputs`/`model_inputs` in the ASR pipeline,
    `input_`/`input_tensors` in `Pipeline`), and no single name overrides both.
    `Pipeline` always passes it positionally.
    """
    if len(args) == 1 and name not in kwargs:
        return args[0]
    if not args and name in kwargs:
        return kwargs.pop(name)
    msg = f"expected exactly one {name!r} argument, got {len(args)} positional"
    raise TypeError(msg)


def _truncate_repetitions(text: str) -> str:
    """Truncate repeated words/phrases/characters at end of text.

    Detects patterns like:
    - Repeated words: "the the the the" -> "the"
    - Repeated phrases: "i am sorry i am sorry i am sorry" -> "i am sorry"
    - Repeated characters: "444444" -> "4"

    Args:
        text: Input text to process

    Returns:
        Text with trailing repetitions removed
    """
    if not text:
        return text

    text = _TRAILING_CHAR_RE.sub(r"\1", text)
    while _TRAILING_WORD_RE.search(text):
        text = _TRAILING_WORD_RE.sub(r"\1", text)

    # 3. Truncate repeated phrases (2-20 words) at end
    # e.g., "i am sorry i am sorry i am sorry" -> "i am sorry"
    words = text.split()
    if len(words) < _MIN_REPEATS * 2:
        return text

    # Cheap pre-check: trailing window must contain duplicates for any phrase repeat
    # to be possible. set(window) == window means all unique → no repetition.
    window = words[-_MIN_REPEATS * 2 :]
    if len(set(window)) == len(window):
        return text

    for phrase_len in range(2, min(21, len(words) // _MIN_REPEATS + 1)):
        phrase_escaped = re.escape(" ".join(words[-phrase_len:]))
        phrase_pattern = re.compile(
            rf"(^|.*?\s)({phrase_escaped})(?:\s+{phrase_escaped}){{{_MIN_REPEATS - 1},}}\s*$",
            re.IGNORECASE,
        )
        match = phrase_pattern.match(text)
        if match:
            text = (match.group(1) + match.group(2)).strip()
            break

    return text
