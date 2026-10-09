"""Load a tiny_audio checkpoint for fast inference: the setup `ta serve` and evals share."""

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTuner
from transformers import AttentionInterface, AttentionMaskInterface, StaticCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

from tiny_audio.alignment import QwenForcedAligner
from tiny_audio.asr_modeling import ASRModel
from tiny_audio.asr_pipeline import CHUNK_MAX_S, ASRPipeline, PreparedChunk, collate_chunks
from tiny_audio.diarization import NemotronDiarizer

if TYPE_CHECKING:
    from tiny_audio.asr_types import GenerativeDecoder

logger = logging.getLogger(__name__)

# Packages behind the fast path of Qwen3.5's gated-delta-rule layers (18 of its
# 24). Without them transformers silently runs a torch reference
# implementation: same numerics, several times slower prefill and decode.
FAST_KERNEL_MODULES = {"causal-conv1d": "causal_conv1d", "flash-linear-attention": "fla"}


# Short static-cache tier for `GraphedDecoder` (see its __init__).
SHORT_BUDGET = 128
SHORT_CACHE_LEN = 416

# Registered attention implementation: sdpa that survives left padding on MPS.
MPS_SAFE_SDPA = "sdpa_mps_safe"


def _mps_safe_sdpa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """sdpa with fully-masked query rows unmasked, so Metal returns no NaN.

    A left-padded batch's pad positions are query rows whose mask is all
    False, and Metal's sdpa answers those with NaN, which the linear-attention
    layers then spread (their padding mask multiplies, and NaN * 0 is NaN).
    Letting such a row attend everywhere makes its output finite garbage
    instead. Real tokens never attend to pad positions (still masked as keys),
    so their outputs are unchanged: 14/14 ragged-batch transcripts matched
    eager byte for byte on an M4 Max, at 1.04-1.15x eager's speed. This is
    transformers' former `_unmask_unattended`, dropped because CUDA's sdpa no
    longer needs it.
    """
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask | ~attention_mask.any(dim=-1, keepdim=True)
        else:  # additive float mask
            dead = torch.isinf(attention_mask).all(dim=-1, keepdim=True)
            attention_mask = attention_mask.masked_fill(dead, 0.0)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register(MPS_SAFE_SDPA, _mps_safe_sdpa)
AttentionMaskInterface.register(MPS_SAFE_SDPA, sdpa_mask)


def missing_fast_kernels() -> list[str]:
    """The fast-path kernel packages that are not installed (pip names)."""
    return [
        pkg for pkg, mod in FAST_KERNEL_MODULES.items() if importlib.util.find_spec(mod) is None
    ]


def merge_lora_adapters(model: ASRModel) -> bool:
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
    thing with gradients -- so this belongs in inference loaders and nowhere in
    `asr_modeling`. `config.use_lora` is read only during load, so unwrapping
    afterwards is invisible to the rest of the stack. The saved checkpoint is
    untouched: this runs on the loaded copy, at every load.
    """
    language_model = model.language_model
    if not isinstance(language_model, PeftModel):
        return False
    # A LoRA PeftModel's base_model is its tuner; PeftModel's __getattr__
    # forwards merge_and_unload there, so call it on the tuner directly.
    merged = cast(BaseTuner, language_model.base_model).merge_and_unload()
    model.language_model = cast("GenerativeDecoder", merged)
    return True


def load_serving_pipeline(
    model_id: str,
    *,
    device: torch.device,
    max_batch_size: int,
    require_fast_kernels: bool = True,
) -> ASRPipeline:
    """The working tree's ASRPipeline over `model_id`, tuned for batched serving.

    Loads with this checkout's code, never the checkpoint's bundled copy
    (`trust_remote_code` would run that), merges LoRA, and preloads the forced
    aligner and diarizer so no request pays their load. On CUDA the fast
    linear-attention kernels are required unless `require_fast_kernels` is
    off: a missing one silently costs several times the latency. On MPS the
    decoder runs `_mps_safe_sdpa`, because stock sdpa returns NaN for
    left-padded rows there (`ASRModel._assert_sdpa_safe_on_mps`).
    """
    if device.type == "cuda":
        missing = missing_fast_kernels()
        if missing and require_fast_kernels:
            msg = (
                f"Fast linear-attention kernels missing: {', '.join(missing)}. "
                "`ta runpod deploy` installs them; pass --allow-slow-kernels to "
                "serve on the torch reference path anyway (several times slower)."
            )
            raise RuntimeError(msg)
        if missing:
            logger.warning("Serving without %s: torch reference path", ", ".join(missing))
        # TF32 for the few fp32 matmuls left (bf16 weights are unaffected).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = ASRModel.from_pretrained(model_id)
    merge_lora_adapters(model)
    # PreTrainedModel.to is functools.wraps'd, which pyright cannot bind as a method.
    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()
    if device.type == "mps":
        model.language_model.set_attn_implementation(MPS_SAFE_SDPA)

    pipe = ASRPipeline(
        model=model,
        feature_extractor=model.feature_extractor,
        tokenizer=model.tokenizer,
        device=device,
    )
    pipe.max_batch_size = max_batch_size
    QwenForcedAligner.get_instance()
    NemotronDiarizer.get_instance()
    return pipe


class GraphedDecoder:
    """`generate_prepared` on CUDA with a static KV cache and CUDA-graphed decode steps.

    Decode on a 4090 is launch-bound: ~24 ms a step at batch 1 and at batch 32
    alike, with the GPU busy 16-27% of the time (~1,200 kernel launches a
    step). transformers compiles the decode step into CUDA graphs
    (`CompileConfig` mode reduce-overhead) whenever the cache is static -- but
    `generate` builds a fresh `StaticCache` every call, and the compiled graph
    guards on the cache's identity, so every call recompiled (~5 s). So the
    caches live here, one per batch-size bucket, reset between calls.

    Graphs replay fixed shapes, so a batch is padded up to the next bucket
    (powers of two up to `max_batch_size`) with copies of its last chunk, whose
    texts are dropped. Measured on an RTX 4090 (18 s AMI clip, 2 chunks):
    batch 1 601 -> 192 ms, batch 32 949 -> 565 ms. Each bucket compiles once,
    30-50 s, so `warm_up` every bucket before serving.
    """

    def __init__(self, pipe: ASRPipeline, max_batch_size: int) -> None:
        self.pipe = pipe
        sizes = [1 << i for i in range(max(1, max_batch_size).bit_length())]
        self.buckets = sorted({*(b for b in sizes if b < max_batch_size), max_batch_size})
        model = pipe.model
        self.eos_ids = set(_as_list(model.generation_config.eos_token_id))
        # Full tier: every token generate may add, plus the longest prompt (an
        # 18 s chunk with lead-in is ~253 tokens) with headroom.
        self.full_budget = int(model.generation_config.max_new_tokens)
        self.full_cache_len = self.full_budget + 384
        # Short tier, the default: attention over the static cache costs per
        # slot, filled or not, so a 416-slot cache decodes a batch-32 step in
        # 10.6 ms against 640's 12.1. Its 128-token budget covers 11,999 of the
        # 12,000 transcripts in the n=1000 x 12-dataset eval (median 15, max
        # 157); a row that hits it unfinished is redone on the full tier.
        # Calibrated on the longest chunk the pipeline cuts (prompt length
        # grows with audio), so any chunk up to that length fits.
        longest = pipe.prepare_chunk(np.zeros(int(CHUNK_MAX_S * 16000), np.float32), 16000)
        self.short_max_frames = int(longest["attention_mask"].shape[-1])
        self.short_enabled = prompt_tokens(pipe, longest) + SHORT_BUDGET <= SHORT_CACHE_LEN
        self._caches: dict[tuple[int, int], StaticCache] = {}
        self.redone = 0  # rows re-decoded on the full tier (for /stats-style logging)

    def bucket(self, n: int) -> int:
        """The smallest bucket that holds `n` chunks."""
        return next(b for b in self.buckets if b >= n)

    def _cache(self, size: int, length: int) -> StaticCache:
        cache = self._caches.get((size, length))
        if cache is None:
            config = self.pipe.model.language_model.config
            cache = self._caches[size, length] = StaticCache(config=config, max_cache_len=length)
        else:
            cache.reset()
        return cache

    def _decode(
        self, prepared: list[PreparedChunk], cache_len: int, budget: int
    ) -> tuple[list[str], list[bool]]:
        """Texts, and whether each row ran out of budget before an end token."""
        size = self.bucket(len(prepared))
        padded = [*prepared, *[prepared[-1]] * (size - len(prepared))]
        batch = collate_chunks(padded)
        model = self.pipe.model
        tokens = model.generate(
            input_features=batch["input_features"].to(model.device),
            audio_attention_mask=batch["attention_mask"].to(model.device),
            past_key_values=self._cache(size, cache_len),
            max_new_tokens=budget,
        )
        assert torch.is_tensor(tokens)  # no output_scores: plain token ids
        rows = tokens[: len(prepared)].cpu()
        texts = [self.pipe.postprocess({"tokens": row})["text"] for row in rows]
        cut = [row.shape[-1] >= budget and not self.eos_ids & set(row.tolist()) for row in rows]
        return texts, cut

    # The caches' tensors are made under generate's inference_mode, and an
    # in-place reset of an inference tensor must happen under it too.
    @torch.inference_mode()
    def __call__(self, prepared: list[PreparedChunk]) -> list[str]:
        longest = max(int(p["attention_mask"].shape[-1]) for p in prepared)
        if not self.short_enabled or longest > self.short_max_frames:
            return self._decode(prepared, self.full_cache_len, self.full_budget)[0]
        texts, cut = self._decode(prepared, SHORT_CACHE_LEN, SHORT_BUDGET)
        redo = [i for i, was_cut in enumerate(cut) if was_cut]
        if redo:
            self.redone += len(redo)
            again, _ = self._decode(
                [prepared[i] for i in redo], self.full_cache_len, self.full_budget
            )
            for i, text in zip(redo, again, strict=True):
                texts[i] = text
        return texts

    @torch.inference_mode()
    def warm_up(self, prepared: PreparedChunk) -> None:
        """Compile every short-tier bucket, and the full tier at batch 1 (the usual redo).

        Larger full-tier buckets compile on first use: redos are rare, and
        compiling all of them would double startup.
        """
        for size in self.buckets:
            if self.short_enabled:
                self._decode([prepared] * size, SHORT_CACHE_LEN, SHORT_BUDGET)
        self._decode([prepared], self.full_cache_len, self.full_budget)


def _as_list(ids: int | list[int] | None) -> list[int]:
    if ids is None:
        return []
    return [ids] if isinstance(ids, int) else list(ids)


def prompt_tokens(pipe: ASRPipeline, prepared: PreparedChunk) -> int:
    """Prompt length (chat template plus audio tokens) the decoder sees for one chunk."""
    batch = collate_chunks([prepared])
    model = pipe.model
    with torch.inference_mode():
        input_ids, _, _ = model._prepare_audio_inputs(  # pyright: ignore[reportPrivateUsage]
            batch["input_features"].to(model.device), batch["attention_mask"].to(model.device)
        )
    return int(input_ids.shape[1])
