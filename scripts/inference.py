"""Load a tiny_audio checkpoint for fast inference: the setup `ta serve` and evals share."""

import functools
import importlib.util
import inspect
import logging
import sys
from types import ModuleType
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
from tiny_audio.asr_pipeline import ASRPipeline, PreparedChunk, collate_chunks
from tiny_audio.asr_processing import CHUNK_MAX_S
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

# Registered attention implementation: sdpa that reads the KV cache once per decode step.
GQA_DECODE_SDPA = "sdpa_gqa_decode"


def _gqa_decode_sdpa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """sdpa for grouped-query attention that skips `repeat_kv` on one-token queries.

    With a mask, transformers' sdpa path copies every KV head out to its query
    heads (`repeat_kv`) before attending, so a decode step reads and rewrites
    the whole cache G-fold. With one query token, the G query heads that share
    a KV head can instead be G query rows against that head -- query head h
    belongs to KV head h // G, as `repeat_kv` maps them -- which is the same
    attention over an un-copied cache. Qwen3.5-2B (8 query heads, 2 KV heads)
    on an RTX 4090, batch 32: attention 1.45 -> 0.24 ms of a 8.1 ms decode
    step. Prefill (more than one query token) takes the stock path.
    """
    batch, heads, q_len, dim = query.shape
    kv_heads = key.shape[1]
    if q_len != 1 or heads == kv_heads:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    out = torch.nn.functional.scaled_dot_product_attention(
        query.reshape(batch, kv_heads, heads // kv_heads, dim),
        key,
        value,
        attn_mask=attention_mask,
        scale=scaling,
    )
    return out.reshape(batch, heads, 1, dim).transpose(1, 2).contiguous(), None


AttentionInterface.register(GQA_DECODE_SDPA, _gqa_decode_sdpa)
AttentionMaskInterface.register(GQA_DECODE_SDPA, sdpa_mask)


def use_fast_kernels_under_compile(module: ModuleType) -> list[str]:
    """Point `module`'s kernel-or-reference wrappers straight at the kernel they found.

    transformers wraps each fast kernel (fla's gated delta rule, causal-conv1d)
    in a function that runs the torch reference instead while exporting, and
    asks `torch.compiler.is_exporting()` -- which dynamo answers True inside any
    `torch.compile` on torch 2.8. So the CUDA-graphed decode step silently ran
    the reference delta rule and depthwise conv, while eager prefill used the
    kernels. Rebinding the module-level names to the resolved implementation
    (keeping the wrapper's kwargs filtering) puts the kernels in the graph.
    Returns the names rebound; none if transformers changes how it wraps them.
    """
    rebound = []
    for name, fn in list(vars(module).items()):
        if not inspect.isfunction(fn) or fn.__closure__ is None:
            continue
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__, strict=True))
        if "is_new_implementation" not in cells or not cells["is_new_implementation"].cell_contents:
            continue
        implementation = cells["implementation"].cell_contents
        params = frozenset(cells["applicable_params"].cell_contents)

        @functools.wraps(fn)
        def direct(
            *args: Any, _impl: Any = implementation, _params: frozenset[str] = params, **kwargs: Any
        ) -> Any:
            return _impl(*args, **{k: v for k, v in kwargs.items() if k in _params})

        setattr(module, name, direct)
        rebound.append(name)
    return rebound


@torch.library.custom_op("tiny_audio::gated_delta_step_", mutates_args={"state"})
def gated_delta_step_(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """fla's fused recurrent gated delta rule, writing the new state over `state`.

    Each kernel program loads its tile of the initial state before storing the
    same tile of the final one, so passing one tensor as both is race-free. A
    custom op with `mutates_args` so inductor sees the mutation: handing the
    triton kernel aliased pointers directly made the compiled graph drop it.
    """
    # CUDA-only dependencies, installed on the pod by `ta runpod deploy`.
    import triton  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    from fla.ops.gated_delta_rule.fused_recurrent import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        fused_recurrent_gated_delta_rule_fwd_kernel,
    )

    _, steps, heads, k_dim = k.shape
    v_heads, v_dim = v.shape[2], v.shape[-1]
    block_v = min(8, triton.next_power_of_2(v_dim))
    out = torch.empty_like(v)
    fused_recurrent_gated_delta_rule_fwd_kernel[
        (triton.cdiv(v_dim, block_v), k.shape[0] * v_heads)
    ](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        beta=beta,
        A_log=None,
        dt_bias=None,
        o=out,
        h0=state,
        ht=state,
        cu_seqlens=None,
        scale=scale,
        T=steps,
        H=heads,
        HV=v_heads,
        K=k_dim,
        V=v_dim,
        BK=triton.next_power_of_2(k_dim),
        BV=block_v,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        APPLY_BETA_SIGMOID=False,
        ALLOW_NEG_EIGVAL=False,
        STATE_V_FIRST=False,
        num_warps=1,
        num_stages=3,
    )
    return out


@gated_delta_step_.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    return torch.empty_like(v)


def _in_place_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    fallback: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The decode-step delta rule, updating the cached state in place when it can.

    The cache's `update_recurrent_state` copies whatever state comes back into
    its static buffer; fla returns a fresh one, so every step copied 18 layers
    of fp32 state (33.5 MB a layer at batch 32: 1.27 ms of a 9.4 ms step).
    Returning the buffer itself makes that copy a no-op. Bit-identical to
    fla's own call.
    """
    if (
        g is None
        or beta is None
        or initial_state is None
        or not output_final_state
        or cu_seqlens is not None
        or initial_state.dtype != torch.float32
        or not initial_state.is_contiguous()
    ):
        return fallback(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
    # The kernel walks raw pointers; q/k/v arrive as views of one fused projection.
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    out = gated_delta_step_(q, k, v, g, beta, initial_state, scale, use_qk_l2norm_in_kernel)
    return out, initial_state


def speed_up_cuda_decode(model: ASRModel) -> list[str]:
    """Install the CUDA decode-step fixes on `model`'s decoder; what was changed.

    Kernels into the compiled graph (`use_fast_kernels_under_compile`), the
    gated delta rule's state updated in place, and grouped-query attention
    without `repeat_kv` (`_gqa_decode_sdpa`). Measured together on an RTX 4090
    (Qwen3.5-2B decoder, batch 32): decode step 9.7 -> 6.8 ms; on 1,200 eval
    clips (12 datasets) 1.31x the throughput at batch 32, with WER inside the
    noise a batch-size change alone makes.
    """
    language_model = model.language_model
    modeling = sys.modules[type(language_model).__module__]
    changed = use_fast_kernels_under_compile(modeling)
    step_name = "torch_recurrent_gated_delta_rule"
    reference = getattr(modeling, step_name, None)
    if step_name in changed and reference is not None:
        setattr(
            modeling, step_name, functools.partial(_in_place_recurrent_step, fallback=reference)
        )
        changed.append("in-place recurrent state")
    language_model.set_attn_implementation(GQA_DECODE_SDPA)
    changed.append(GQA_DECODE_SDPA)
    return changed


def contiguous_lm_head_input(model: ASRModel) -> None:
    """Give the output projection a contiguous input, so prefill runs it as one GEMM.

    generate's prefill keeps only the last position's hidden state, a strided
    `[batch, 1, hidden]` slice, and `matmul` can't fold a strided 3-D input
    into one GEMM: it ran `bmm` against the weight expanded per row -- one
    GEMV per row, each reading Qwen3.5's whole 1 GB embedding matrix. 34 ms
    of a 532 ms batch-32 call on an RTX 4090. Decode inputs are already
    contiguous, so this costs nothing there.
    """
    lm_head = cast("torch.nn.Module | None", model.language_model.get_output_embeddings())
    if lm_head is None:
        return

    def make_contiguous(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return (args[0].contiguous(), *args[1:])

    lm_head.register_forward_pre_hook(make_contiguous)


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
    off: a missing one silently costs several times the latency; with them,
    the decoder gets `speed_up_cuda_decode`. On MPS the decoder runs
    `_mps_safe_sdpa`, because stock sdpa returns NaN for left-padded rows there
    (`ASRModel._assert_sdpa_safe_on_mps`).
    """
    missing = missing_fast_kernels() if device.type == "cuda" else []
    if device.type == "cuda":
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
    contiguous_lm_head_input(model)
    if device.type == "mps":
        model.language_model.set_attn_implementation(MPS_SAFE_SDPA)
    elif device.type == "cuda" and not missing:
        logger.info("Decode speedups: %s", ", ".join(speed_up_cuda_decode(model)))

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

    Graphs replay fixed shapes, so a batch is padded up to the next bucket with
    copies of its last chunk, whose texts are dropped. Measured on an RTX 4090
    (18 s AMI clip, 2 chunks): batch 1 601 -> 192 ms, batch 32 949 -> 565 ms,
    so a padded row costs ~12 ms, mostly encoder and prefill. Buckets are the
    powers of two plus the 1.5x points between them (1, 2, 4, 6, 8, 12, 16,
    24, 32): the served mean batch of 12 padded to 16 under powers of two
    alone. Each bucket compiles once, so `warm_up` every bucket before serving.
    """

    def __init__(self, pipe: ASRPipeline, max_batch_size: int) -> None:
        self.pipe = pipe
        top = max(1, max_batch_size)
        sizes = [1 << i for i in range(top.bit_length())]
        sizes += [3 * s // 2 for s in sizes if s >= 4]
        self.buckets = sorted({*(b for b in sizes if b < top), top})
        # Every bucket is a recompile of the one decode frame, short tier and
        # full, and dynamo's default limit of 8 would quietly run the rest
        # eagerly (no CUDA graph, correct but several times slower).
        dynamo_config = torch._dynamo.config  # pyright: ignore[reportPrivateUsage]
        dynamo_config.recompile_limit = max(
            dynamo_config.recompile_limit, 2 * len(self.buckets) + 4
        )
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
