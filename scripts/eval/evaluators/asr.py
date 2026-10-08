"""ASR evaluator implementations."""

import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, Unpack, cast

import assemblyai as aai
import numpy as np
import torch
from assemblyai.streaming.v3 import (
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
from deepgram import DeepgramClient, ListenV1Response
from huggingface_hub import InferenceClient
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTuner
from rich.table import Table
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter
from transformers import TextIteratorStreamer, pipeline

from scripts.eval.audio import as_16k_array, is_str_dict, prepare_wav_bytes
from scripts.eval.speaker_metrics import serialize_turns
from tiny_audio.asr_config import ASRConfig
from tiny_audio.asr_modeling import ASRModel, resolve_attn_implementation
from tiny_audio.asr_pipeline import ASRPipeline

from .base import (
    Confidence,
    Evaluator,
    EvaluatorOptions,
    Metrics,
    Transcription,
    console,
    setup_assemblyai,
)

if TYPE_CHECKING:
    from elevenlabs import SpeechToTextChunkResponseModel
    from elevenlabs.client import ElevenLabs

    from tiny_audio.asr_types import GenerativeDecoder

    _ELEVENLABS_AVAILABLE: bool
else:
    # Optional vendor SDK, dev-only on purpose (see [tool.deptry] DEP004).
    try:
        from elevenlabs import SpeechToTextChunkResponseModel
        from elevenlabs.client import ElevenLabs

        _ELEVENLABS_AVAILABLE = True
    except ImportError:
        _ELEVENLABS_AVAILABLE = False


def print_generation_config(model: ASRModel, model_path: str) -> None:
    """Print generation config in a visible format."""
    gen_config = model.generation_config
    table = Table(title=f"Generation config: {model_path}", show_header=False, title_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key in (
        "max_new_tokens",
        "min_new_tokens",
        "num_beams",
        "do_sample",
        "repetition_penalty",
        "length_penalty",
        "no_repeat_ngram_size",
    ):
        table.add_row(key, str(getattr(gen_config, key)))
    console.print(table)


# Every ASRConfig field that names a weight dtype. All three get pinned to the
# same value for inference, because `model_dtype` alone does not reach the
# encoder or the projector: `projector_dtype: float32` is in every experiment
# config and `encoder_dtype: float32` is in the encoder-unfreezing ones
# (granite_qwen_top4 / _encoder / _full_encoder). Both persist into the
# checkpoint's config.json, and `_load_audio_encoder` / `_create_projector`
# prefer them over `model_dtype` (asr_modeling.py `_resolve_dtype` callers).
# Overriding only `model_dtype` therefore left the 473M-param Granite encoder
# in fp32 on every `-top4` eval and the 12.6M projector in fp32 on all of them.
# Those two fields exist to hold fp32 MASTER WEIGHTS for AdamW -- the same
# training-only argument `_resolve_local_runtime` makes for `model_dtype`,
# which had only been half-applied.
DTYPE_CONFIG_FIELDS = ("model_dtype", "projector_dtype", "encoder_dtype")


def _resolve_local_runtime() -> tuple[int | str, str]:
    """Pick the device and weight dtype for a locally loaded ASRModel pipeline.

    Returns `(device, model_dtype)` where model_dtype is the STRING name
    `ASRConfig.model_dtype` expects, not a `torch.dtype`. That distinction is
    the whole point of this function.

    `pipeline(dtype=...)` and `from_pretrained(dtype=...)` do nothing to this
    model class. ASRModel.__init__ does `target_dtype = getattr(torch,
    config.model_dtype)` and then casts both submodules with an explicit
    `.to(dtype=...)` after load (deliberately -- see the comment in
    _load_audio_encoder about loader paths honoring `dtype=` inconsistently),
    so a dtype kwarg is silently overwritten by the saved config. Verified
    against the published granite-qwen config: `dtype=torch.float16` leaves
    `model_dtype` at "float32", while `model_dtype="float16"` takes. Routing it
    through model_kwargs is the only path that lands.

    Why it matters: granite-qwen's config.json carries `model_dtype: float32`,
    which is load-bearing for TRAINING only -- fp32 master weights exist so
    AdamW's 2e-5 decoder step doesn't round away under bf16's ULP (see
    configs/experiments/granite_qwen.yaml). Inference has no optimizer, so it
    was holding 2.27B params in fp32 and reading 9.1 GB of weights per decoded
    token instead of 4.5 GB, on a decode loop that is entirely
    memory-bandwidth bound.

    bfloat16 rather than float16 on MPS. Measured on torch 2.8 / Metal, the
    two are the same speed (0.78 vs 0.79 ms/iter on a LayerNorm + 2048x6144
    matmul + SiLU), so fp16 buys nothing -- while costing the exponent range
    that matters here, since fp16 tops out at 65504 and both the Granite
    encoder and Qwen3.5 were pretrained in bf16. Equal speed, strictly more
    headroom.

    CPU stays fp32: reduced precision there is emulated rather than
    accelerated without AMX, so it is a slowdown, not a win.

    attn_implementation is deliberately NOT set here. ASRModel routes whatever
    the config asks for through resolve_attn_implementation; passing it here
    would be a second, silently diverging copy of that policy -- which is what
    it had become: the streaming evaluator asked for "sdpa" on MPS and got
    eager anyway. The kernel is instead corrected AFTER load, by
    `_use_sdpa_where_safe`, which calls that same helper. See its docstring for
    why post-load is the only placement that works.
    """
    if torch.cuda.is_available():
        return 0, "bfloat16"
    if torch.backends.mps.is_available():
        return "mps", "bfloat16"
    return -1, "float32"


def _build_working_tree_pipeline(
    model_path: str, device: int | str, model_dtype: str
) -> ASRPipeline:
    """Build the pipeline from THIS checkout's tiny_audio, ignoring the checkpoint's copy.

    `save_pretrained` ships the modeling code alongside the weights and points
    `auto_map` / `custom_pipelines` in config.json at it
    (`asr_modeling.ASRModel`, `asr_pipeline.ASRPipeline`). Under
    `trust_remote_code=True` transformers resolves those entries by importing
    the checkpoint's OWN .py files -- from the Hub snapshot, byte-compiled into
    ~/.cache/huggingface/modules/transformers_modules/ -- and never looks at
    the working tree. So `ta eval -m mazesmazes/...` measures the code as it
    was when that checkpoint was pushed, and an edit to asr_modeling.py or
    asr_pipeline.py has no effect on the number until it is pushed. Locally
    saved checkpoints carry the same frozen copy, so `ta eval -m
    checkpoints/checkpoint-2000` is equally stale.

    This path takes only the weights and config from `model_path` and supplies
    every class from the import graph: `ASRConfig.from_pretrained` rather than
    `AutoConfig` (no auto_map lookup), the imported `ASRModel`, and the
    imported `ASRPipeline` in place of the `custom_pipelines` entry.

    dtype goes through the config, not a `dtype=` kwarg -- see
    `_resolve_local_runtime` for why that distinction is load-bearing.
    """
    # PretrainedConfig.from_pretrained leaves PathLike unparameterized.
    config = ASRConfig.from_pretrained(model_path)  # pyright: ignore[reportUnknownMemberType]
    for field in DTYPE_CONFIG_FIELDS:
        setattr(config, field, model_dtype)
    model = ASRModel.from_pretrained(model_path, config=config)
    return ASRPipeline(model=model, device=device)


def _module_eval(module: torch.nn.Module) -> None:
    """`module.eval()`, in place.

    transformers leaves `PreTrainedModel.eval` unannotated; typed as
    `nn.Module` the call resolves. Same method at runtime.
    """
    module.eval()


def _merge_lora_adapters(model: ASRModel) -> bool:
    """Fold LoRA adapters into the base weights for inference. Returns whether it ran.

    `ASRModel.from_pretrained` wraps the decoder in a live `PeftModel` and
    nothing ever unwrapped it, so eval decoded through the adapters as separate
    modules: two extra matmuls per adapted linear per token, and peft builds
    adapter weights in fp32 regardless of the base dtype, so those matmuls ran
    fp32 against a bf16 base. granite_qwen_frozen adapts six projections per
    layer -- mlp gate/up/down and linear_attn in_proj_qkv/in_proj_z/out_proj --
    which is 67.3M fp32 params and tens of thousands of extra Metal dispatches
    over a 64-token decode.

    Merging is a pure win on a decode loop that is launch- and
    bandwidth-bound, and it is not MPS-specific. Measured on an M-series Mac
    (granite-qwen-frozen, bf16, 10s audio, 64 tokens, batch 1, sdpa):
    20.6 -> 36.7 tok/s, a 78% gain.

    NOT bit-exact: the fp32 `B @ A` product is rounded into bf16 base weights,
    where holding the adapters separate kept the correction in fp32 until the
    residual add. Bounded rather than assumed -- paired against the unmerged
    path on 30 librispeech samples (with `_use_sdpa_where_safe`, which has the
    same exposure), all 30 predictions came back byte-identical, WER matched at
    1.5986, and the only metric that moved at all was mean top1/top2 margin,
    5.8474 -> 5.8466. Nothing came close to flipping a token. A larger paired
    run is still the right check before a sub-noise WER delta is reported as
    real.

    Inference only. Training must keep the adapters live -- they are the only
    thing with gradients -- so this belongs in the eval loader and nowhere in
    `asr_modeling`. `config.use_lora` is read only during load, so unwrapping
    afterwards is invisible to the rest of the stack.
    """
    language_model = model.language_model
    if not isinstance(language_model, PeftModel):
        return False
    # A LoRA PeftModel's base_model is its tuner; PeftModel's __getattr__
    # forwards merge_and_unload there, so call it on the tuner directly.
    merged = cast(BaseTuner, language_model.base_model).merge_and_unload()
    model.language_model = cast("GenerativeDecoder", merged)
    return True


def _use_sdpa_where_safe(model: ASRModel) -> None:
    """Re-apply the MPS attention policy after load, overriding the checkpoint's copy.

    This duplicates what `resolve_attn_implementation` already decided at
    load -- deliberately, because on the default eval path that function is not
    the working tree's. `trust_remote_code` resolves the `auto_map` entry by
    importing the checkpoint's OWN `asr_modeling.py` (see
    `_build_working_tree_pipeline`), so the attention policy that runs is
    whatever was published with the weights. Every checkpoint on the Hub
    predates the fix that scoped MPS's eager coercion to models that actually
    build a sliding-window mask, and carries the blanket "MPS -> eager"
    version instead. `--local-code` cannot rescue it either: every published
    checkpoint fails the `projector.output_scale` compatibility guard.

    So the fix shipped and was unreachable. `ta eval` on a Mac printed
    `decoder attn: eager` against Qwen3.5-2B, which declares
    `sliding_window: None` and cannot hit the Metal correctness bug that
    coercion exists for. Cost, same setup as `_merge_lora_adapters`:
    15.8 tok/s eager vs 20.6 tok/s sdpa. Together the two take `ta eval`
    on librispeech from 1.63 to 1.05 s/sample at identical WER.

    Setting it post-load fixes it regardless of which code version loaded, and
    is a no-op under `--local-code` where the resolution was already right.
    The policy itself is imported rather than restated, so this stays one
    source of truth -- it only moves WHEN that policy is consulted.

    Batch 1 only, which the eval harness is: Metal's sdpa returns NaN for a
    fully-masked row, so a left-padded batch decodes as "!!!!!!". `-w N` runs
    N threads each making batch-1 calls, so it stays safe. Anything that
    batches ragged audio on a Mac must not come through here.
    """
    text_model_id = getattr(model.config, "text_model_id", None)
    resolved = resolve_attn_implementation(model.config.attn_implementation, model_id=text_model_id)
    # transformers exposes the active kernel only as the private config field.
    lm_config = model.language_model.config
    current = lm_config._attn_implementation  # pyright: ignore[reportPrivateUsage]
    if resolved is None or resolved == current:
        return
    # transformers leaves the per-submodule dict form unparameterized.
    model.language_model.set_attn_implementation(  # pyright: ignore[reportUnknownMemberType]
        resolved
    )


def _build_local_pipeline(model_path: str, *, local_code: bool = False) -> ASRPipeline:
    """Build the ASR pipeline both local evaluators run on.

    `local_code=True` swaps the checkpoint's bundled modeling code for this
    checkout's; see `_build_working_tree_pipeline`.
    """
    device, model_dtype = _resolve_local_runtime()
    if local_code:
        pipe = _build_working_tree_pipeline(model_path, device, model_dtype)
    else:
        # `custom_pipelines` in the checkpoint's config.json resolves this to
        # the checkpoint's own copy of ASRPipeline (see above), not the
        # generic ASR pipeline the task name alone would suggest.
        pipe = cast(
            ASRPipeline,
            pipeline(
                "automatic-speech-recognition",
                model=model_path,
                trust_remote_code=True,
                device=device,
                model_kwargs=dict.fromkeys(DTYPE_CONFIG_FIELDS, model_dtype),
            ),
        )
    merged = _merge_lora_adapters(pipe.model)
    _use_sdpa_where_safe(pipe.model)
    # Which code ran, and which attention kernel, are both part of what the WER
    # means -- so they are logged next to the device rather than left to infer.
    source = "working tree" if local_code else "checkpoint (trust_remote_code)"
    lm_config = pipe.model.language_model.config
    resolved = lm_config._attn_implementation  # pyright: ignore[reportPrivateUsage]
    console.print(
        f"[dim]Using device: {device}, model_dtype: {model_dtype}, "
        f"code: {source}, decoder attn: {resolved}, "
        f"lora: {'merged' if merged else 'none'}[/dim]"
    )
    return pipe


class LocalEvaluator(Evaluator):
    """Evaluator for local models."""

    def __init__(
        self,
        model_path: str,
        user_prompt: str | None = None,
        local_code: bool = False,
        **kwargs: Unpack[EvaluatorOptions],
    ) -> None:
        # PyTorch's MPS backend is not thread-safe: concurrent threads encode
        # into a shared Metal command encoder and segfault in the AGX driver
        # (setComputePipelineState). `-w 4` on frozen-4 crashed within 3 minutes.
        if kwargs.get("num_workers", 1) > 1 and _resolve_local_runtime()[0] == "mps":
            console.print(
                "[yellow]Warning: LocalEvaluator forces num_workers=1 on MPS "
                "(concurrent Metal kernel encoding segfaults)[/yellow]"
            )
            kwargs["num_workers"] = 1
        super().__init__(**kwargs)
        self.pipe = _build_local_pipeline(model_path, local_code=local_code)
        # Explicit, not incidental. LocalStreamingEvaluator has always done
        # this; this class never did, and got away with it only because
        # nothing in the stack was train/eval-sensitive. A partially unfrozen
        # Granite encoder is: its 16 BatchNorm1d modules normalise with batch
        # statistics in train mode, which cost 25 WER points on Earnings22.
        _module_eval(self.pipe.model)
        self.user_prompt = user_prompt

        print_generation_config(self.pipe.model, model_path)

    def transcribe(self, audio: object) -> Transcription:
        if self.speakers:
            return self._transcribe_speakers(audio)
        start = time.time()
        # Request per-step top-1/top-2 logprobs. The Hub-loaded pipeline may be
        # an older version without scores support, in which case the kwarg is
        # absorbed by generate() but the result dict won't include the new
        # `top1_logprob` / `top2_logprob` fields and we return confidence=None.
        result = self.pipe(audio, user_prompt=self.user_prompt, output_scores=True)
        elapsed = time.time() - start

        text: str = result.get("text", "") if is_str_dict(result) else str(result)
        confidence: Confidence | None = None
        if is_str_dict(result):
            top1 = result.get("top1_logprob")
            top2 = result.get("top2_logprob")
            if top1 and top2 and len(top1) == len(top2):
                n = len(top1)
                mean_top1 = sum(top1) / n
                mean_margin = sum(a - b for a, b in zip(top1, top2, strict=True)) / n
                confidence = {
                    "mean_top1_logprob": mean_top1,
                    "mean_margin": mean_margin,
                    "num_tokens": n,
                }
        return text, elapsed, confidence

    def _transcribe_speakers(self, audio: object) -> Transcription:
        """The pipeline's own diarization (`return_speakers=True`) as `<SPK_n>` turns.

        Whatever the checkpoint's pipeline does for speakers is what gets
        scored -- the same call a user makes. An alignment or diarization
        error is raised, not scored as an unlabelled transcript.
        """
        start = time.time()
        result = self.pipe(audio, user_prompt=self.user_prompt, return_speakers=True)
        elapsed = time.time() - start
        for key in ("timestamp_error", "diarization_error"):
            if result.get(key):
                msg = f"{key}: {result[key]}"
                raise RuntimeError(msg)
        text = serialize_turns((w["speaker"], w["word"]) for w in result["words"])
        return text, elapsed, None


class LocalStreamingEvaluator(Evaluator):
    """Evaluator for local models with streaming metrics (TTFB, processing time)."""

    def __init__(
        self,
        model_path: str,
        user_prompt: str | None = None,
        local_code: bool = False,
        **kwargs: Unpack[EvaluatorOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.pipe = _build_local_pipeline(model_path, local_code=local_code)
        self.model = self.pipe.model
        _module_eval(self.model)
        self.processor = self.model.get_processor()
        self.user_prompt = user_prompt

        # Track timing stats
        self.ttfb_times: list[float] = []
        self.processing_times: list[float] = []

        # Print generation config
        print_generation_config(self.model, model_path)

    def _reset_run_state(self) -> None:
        """Also clear the TTFB / processing accumulators.

        `compute_metrics` averages these, and the CLI reuses one evaluator
        for every dataset in a sweep, so without this the second dataset's
        avg_ttfb would include the first dataset's samples.
        """
        super()._reset_run_state()
        self.ttfb_times = []
        self.processing_times = []

    def transcribe(self, audio: object) -> Transcription:
        audio_array = as_16k_array(audio)

        # Process audio (ASRProcessor handles sampling_rate internally)
        inputs = self.processor(
            audio_array,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(
            device=self.model.device, dtype=self.model.dtype
        )
        audio_attention_mask = inputs["audio_attention_mask"].to(self.model.device)

        # Set up streamer to capture first token time
        streamer = TextIteratorStreamer(
            self.model.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        first_token_time: list[float | None] = [None]
        generation_start: list[float | None] = [None]

        def generate() -> None:
            generation_start[0] = time.time()
            self.model.generate(
                input_features=input_features,
                audio_attention_mask=audio_attention_mask,
                streamer=streamer,
            )

        # Start generation in background thread
        thread = threading.Thread(target=generate)
        thread.start()

        # Collect tokens and measure TTFB
        tokens: list[str] = []
        for text in cast(Iterable[str], streamer):
            if first_token_time[0] is None and text:
                first_token_time[0] = time.time()
            tokens.append(text)

        thread.join()
        processing_end = time.time()

        # Calculate timing metrics
        processing_time = processing_end - generation_start[0] if generation_start[0] else 0
        ttfb = (
            (first_token_time[0] - generation_start[0])
            if first_token_time[0] and generation_start[0]
            else None
        )

        # Store for aggregation
        self.processing_times.append(processing_time)
        if ttfb is not None:
            self.ttfb_times.append(ttfb)

        # Print timing info
        ttfb_str = f"{ttfb * 1000:.0f}ms" if ttfb else "N/A"
        console.print(
            f"  [dim][Streaming] TTFB: {ttfb_str}, Processing: {processing_time * 1000:.0f}ms[/dim]"
        )

        full_text = "".join(tokens).strip()
        return full_text, processing_time, None

    def compute_metrics(self) -> Metrics:
        """Compute final metrics including streaming-specific timing."""
        metrics = super().compute_metrics()
        if self.ttfb_times:
            metrics["avg_ttfb"] = sum(self.ttfb_times) / len(self.ttfb_times)
            metrics["min_ttfb"] = min(self.ttfb_times)
            metrics["max_ttfb"] = max(self.ttfb_times)
        if self.processing_times:
            metrics["avg_processing"] = sum(self.processing_times) / len(self.processing_times)
        return metrics


class _SpeechRecognitionClient(Protocol):
    """The slice of `huggingface_hub.InferenceClient` EndpointEvaluator calls.

    The response type is a dict subclass carrying the endpoint's JSON fields.
    """

    def automatic_speech_recognition(self, audio: bytes, /) -> Mapping[str, Any]: ...


class EndpointEvaluator(Evaluator):
    """Evaluator for HuggingFace Inference Endpoints."""

    def __init__(self, endpoint_url: str, **kwargs: Unpack[EvaluatorOptions]) -> None:
        super().__init__(**kwargs)
        self.client: _SpeechRecognitionClient = InferenceClient(base_url=endpoint_url)

    def transcribe(self, audio: object) -> Transcription:
        wav_bytes = prepare_wav_bytes(audio)

        start = time.time()
        result = self.client.automatic_speech_recognition(wav_bytes)
        elapsed = time.time() - start

        text: str = result.get("text", result.get("transcription", ""))
        return text, elapsed, None


class AssemblyAIEvaluator(Evaluator):
    """Evaluator for AssemblyAI API.

    On a speaker-attributed dataset (`evaluate(..., speakers=True)`) requests
    run with `speaker_labels=True` and the utterances are written as
    `<SPK_1>text<SPK_2>text`, speakers numbered by first appearance -- the
    format the references and cpWER use.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "universal-3-pro",
        base_url: str | None = None,
        **kwargs: Unpack[EvaluatorOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.transcriber = setup_assemblyai(api_key, model, base_url=base_url)
        self.speaker_transcriber = setup_assemblyai(
            api_key, model, speaker_labels=True, base_url=base_url
        )

    def transcribe(self, audio: object) -> Transcription:
        wav_bytes = prepare_wav_bytes(audio)
        transcriber = self.speaker_transcriber if self.speakers else self.transcriber
        start = time.time()
        transcript = transcriber.transcribe(io.BytesIO(wav_bytes))
        elapsed = time.time() - start
        if self.speakers:
            return speaker_text(transcript), elapsed, None
        return transcript.text or "", elapsed, None


def speaker_text(transcript: aai.Transcript) -> str:
    """An AssemblyAI transcript's utterances as `<SPK_n>` text (plain text if none)."""
    utterances = transcript.utterances or []
    if not utterances:
        return transcript.text or ""
    return serialize_turns((u.speaker, u.text) for u in utterances)


class _StreamingError(RuntimeError):
    """A streaming-session failure carrying the server's close code (or None)."""

    def __init__(self, code: object, detail: str) -> None:
        super().__init__(f"Streaming error (code={code}): {detail}")
        self.streaming_code = code


class AssemblyAIStreamingEvaluator(Evaluator):
    """Evaluator for AssemblyAI Streaming API (Universal-Streaming model).

    Opens a fresh websocket per call so parallel workers don't share state.
    A class-level semaphore caps real concurrent streams below the account's
    server-side limit; override via ASSEMBLYAI_STREAM_CONCURRENCY.
    """

    # WebSocket close codes the server uses for transient backpressure
    # (concurrent-stream cap, overload). These deserve a retry rather than
    # bubbling up as a permanent failure.
    _RETRYABLE_CODES = frozenset({1008, 1011, 1013, 4029, 4102})
    _MAX_RETRIES = 6
    _BACKOFF_CAP = 30.0
    _DEFAULT_CONCURRENCY = 8

    _stream_semaphore: threading.Semaphore | None = None
    _semaphore_lock = threading.Lock()

    def __init__(self, api_key: str, **kwargs: Unpack[EvaluatorOptions]) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self._ensure_semaphore()

    @classmethod
    def _ensure_semaphore(cls) -> None:
        if cls._stream_semaphore is not None:
            return
        with cls._semaphore_lock:
            if cls._stream_semaphore is None:
                cap = int(os.environ.get("ASSEMBLYAI_STREAM_CONCURRENCY", cls._DEFAULT_CONCURRENCY))
                cls._stream_semaphore = threading.Semaphore(cap)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        return (
            isinstance(exc, RuntimeError)
            and getattr(exc, "streaming_code", None)
            in AssemblyAIStreamingEvaluator._RETRYABLE_CODES
        )

    def transcribe(self, audio: object) -> Transcription:
        pcm_data = self._prepare_pcm(audio)
        text, elapsed = self._run_session_with_retry(pcm_data)
        return text, elapsed, None

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(_MAX_RETRIES + 1),
        # Exponential backoff with jitter, capped to keep latency bounded.
        wait=wait_exponential_jitter(initial=0.5, max=_BACKOFF_CAP),
        reraise=True,
    )
    def _run_session_with_retry(self, pcm_data: bytes) -> tuple[str, float]:
        assert self._stream_semaphore is not None
        with self._stream_semaphore:
            return self._run_session(pcm_data)

    def _prepare_pcm(self, audio: object) -> bytes:
        audio_array = as_16k_array(audio)

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()
        return (audio_array * 32767).astype(np.int16).tobytes()

    def _run_session(self, pcm_data: bytes) -> tuple[str, float]:
        # Per-call state — closures capture these, so threads cannot stomp on
        # each other's connection or callback buckets.
        transcripts: dict[int, str] = {}
        error_box: list[object] = [None]
        turn_done = threading.Event()

        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        def on_turn(_client: StreamingClient, event: TurnEvent) -> None:
            if event.transcript and event.end_of_turn and event.turn_is_formatted:
                transcripts[event.turn_order] = event.transcript
                turn_done.set()

        def on_error(_client: StreamingClient, err: StreamingError) -> None:
            error_box[0] = err
            turn_done.set()

        def on_terminated(_client: StreamingClient, _event: TerminationEvent) -> None:
            turn_done.set()

        handlers: list[tuple[StreamingEvents, Callable[..., None]]] = [
            (StreamingEvents.Turn, on_turn),
            (StreamingEvents.Error, on_error),
            (StreamingEvents.Termination, on_terminated),
        ]
        for event, handler in handlers:
            # `handler` is annotated as a bare Callable.
            client.on(event, handler)  # pyright: ignore[reportUnknownMemberType]
        client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                speech_model=SpeechModel.universal_streaming_english,
            )
        )

        start_time = time.time()

        chunk_size = 3200  # 100ms of 16kHz 16-bit audio
        for i in range(0, len(pcm_data), chunk_size):
            client.stream(pcm_data[i : i + chunk_size])
            time.sleep(0.02)

        client.disconnect(terminate=True)
        turn_done.wait(timeout=30)

        elapsed = time.time() - start_time

        if error_box[0] is not None:
            err = error_box[0]
            code = getattr(err, "code", None)
            raise _StreamingError(code, str(err) or "no message")

        full_transcript = " ".join(transcripts[k] for k in sorted(transcripts.keys()))
        return full_transcript, elapsed


class DeepgramEvaluator(Evaluator):
    """Evaluator for Deepgram Nova 3 API."""

    def __init__(self, api_key: str, **kwargs: Unpack[EvaluatorOptions]) -> None:
        super().__init__(**kwargs)
        self.client = DeepgramClient(api_key=api_key)

    def transcribe(self, audio: object) -> Transcription:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        response = self.client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-3",
        )
        elapsed = time.time() - start

        # Without a callback URL the API answers synchronously, with the
        # transcript rather than an accepted-for-later acknowledgement.
        alternatives = cast(ListenV1Response, response).results.channels[0].alternatives
        if not alternatives:
            msg = "Deepgram response has no alternatives"
            raise RuntimeError(msg)
        return alternatives[0].transcript or "", elapsed, None


class _SmallestHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Smallest.ai HTTP {status}: {body[:200]}")
        self.status = status


class SmallestEvaluator(Evaluator):
    """Evaluator for the Smallest.ai Pulse pre-recorded STT API.

    Plain urllib rather than the `smallestai` SDK: the endpoint is one POST of
    raw WAV bytes, and stdlib keeps poetry.lock untouched.
    """

    URL = "https://waves-api.smallest.ai/api/v1/pulse/get_text"
    # 429 / 5xx retried here because the base Evaluator turns any exception into
    # an empty prediction -- a rate limit under -w N would read as 100% WER rows.
    _RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str,
        model: str = "pulse",
        language: str = "en",
        **kwargs: Unpack[EvaluatorOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.language = language

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, _SmallestHTTPError):
            return exc.status in SmallestEvaluator._RETRYABLE_STATUSES
        return isinstance(exc, (urllib.error.URLError, TimeoutError))

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1.0, max=30.0),
        reraise=True,
    )
    def _post(self, wav_bytes: bytes) -> dict[str, Any]:
        query = urllib.parse.urlencode({"model": self.model, "language": self.language})
        request = urllib.request.Request(
            f"{self.URL}?{query}",
            data=wav_bytes,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "audio/wav"},
        )
        try:
            # B310 guards file:// schemes; the URL is the fixed https constant above.
            with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
                payload: dict[str, Any] = json.loads(response.read())
                return payload
        except urllib.error.HTTPError as e:
            raise _SmallestHTTPError(e.code, e.read().decode(errors="replace")) from e

    def transcribe(self, audio: object) -> Transcription:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()
        payload = self._post(wav_bytes)
        elapsed = time.time() - start

        if payload.get("status") != "success":
            msg = f"Smallest.ai returned non-success payload: {payload}"
            raise RuntimeError(msg)
        return payload.get("transcription") or "", elapsed, None


class ElevenLabsEvaluator(Evaluator):
    """Evaluator for ElevenLabs Scribe API."""

    def __init__(
        self, api_key: str, model: str = "scribe_v2", **kwargs: Unpack[EvaluatorOptions]
    ) -> None:
        super().__init__(**kwargs)
        if not _ELEVENLABS_AVAILABLE:
            msg = "ElevenLabs backend requires the elevenlabs SDK: pip install elevenlabs"
            raise ImportError(msg)
        self.client = ElevenLabs(api_key=api_key)
        self.model = model

    def transcribe(self, audio: object) -> Transcription:
        wav_bytes = prepare_wav_bytes(audio)
        start = time.time()

        transcription = self.client.speech_to_text.convert(
            file=io.BytesIO(wav_bytes),
            model_id=self.model,
        )
        elapsed = time.time() - start

        # Extract text from transcription response
        text = (
            transcription.text
            if isinstance(transcription, SpeechToTextChunkResponseModel)
            else str(transcription)
        )
        return text or "", elapsed, None
