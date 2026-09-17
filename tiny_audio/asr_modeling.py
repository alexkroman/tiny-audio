import functools
import json
import logging
import math
from pathlib import Path
from threading import Thread
from typing import Iterator, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .asr_config import ASRConfig, compute_encoder_output_length
    from .projectors import PROJECTOR_CLASSES
except ImportError:
    from asr_config import ASRConfig, compute_encoder_output_length  # type: ignore[no-redef]
    from projectors import PROJECTOR_CLASSES  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

# FlashAttention's kernels are compiled for head dimensions up to 256.
FLASH_ATTENTION_MAX_HEAD_DIM = 256

# PyTorch's Metal kernels index tensor storage with 32-bit offsets, so a gather
# into a tensor holding more than INT32_MAX elements wraps around and silently
# returns the wrong rows -- no error, no warning, just corrupt values.
MPS_MAX_TENSOR_ELEMENTS = 2**31 - 1


def mps_unsafe_parameters(model: nn.Module) -> list[tuple[str, int]]:
    """Parameters too large for MPS to index correctly, as (name, numel) pairs.

    Gemma 4 E2B trips this on `embed_tokens_per_layer`: its per-layer embedding
    table is 262144 x (35*256) = 2,348,810,240 elements, 201M past INT32_MAX.
    On MPS the lookup returns garbage, which Gemma then adds into all 35 layers
    at every position. The failure is entirely silent -- the model loads, runs,
    and emits fluent, confident, wrong transcripts. Measured on this stack:
    teacher-forced loss 0.14 on CPU versus 3.80 on MPS from identical weights.

    Returns an empty list for models that are safe, so callers can treat a
    non-empty result as "this model needs `chunk_oversized_embeddings`".
    """
    return [
        (name, param.numel())
        for name, param in model.named_parameters()
        if param.numel() > MPS_MAX_TENSOR_ELEMENTS
    ]


class ChunkedEmbedding(nn.Module):
    """Drop-in `nn.Embedding` that splits its table to stay within 32-bit indexing.

    The overflow lives in the kernel's arithmetic over the weight's *storage*,
    so slicing is not enough -- views share that storage and stay broken. Each
    chunk therefore has to be its own contiguous allocation. Gathering per
    chunk and concatenating on the feature axis reproduces the full gather
    exactly: verified bit-identical to CPU on MPS, where the single-tensor
    lookup was off by up to 3.16.

    Wraps rather than replaces the semantics of `Gemma4TextScaledWordEmbedding`,
    whose forward multiplies the lookup by `embed_scale` -- dropping that would
    shrink every per-layer embedding by sqrt(hidden_size_per_layer_input).
    """

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        # Gemma 4's scaled embedding carries this; a plain nn.Embedding does not.
        self.embed_scale = getattr(embedding, "embed_scale", None)

        num_chunks = math.ceil(embedding.weight.numel() / MPS_MAX_TENSOR_ELEMENTS)
        device = embedding.weight.device
        # Chunk from a CPU copy: carving contiguous pieces out of an
        # already-oversized MPS tensor would run the same overflowing kernel
        # this class exists to avoid.
        source = embedding.weight.detach().cpu()
        self.chunks = nn.ParameterList(
            [
                nn.Parameter(
                    chunk.contiguous().to(device), requires_grad=embedding.weight.requires_grad
                )
                for chunk in torch.chunk(source, num_chunks, dim=1)
            ]
        )
        del source

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = torch.cat([F.embedding(input_ids, chunk) for chunk in self.chunks], dim=-1)
        if self.embed_scale is not None:
            out = out * self.embed_scale.to(out.dtype)
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Reassembled table, for callers that introspect `.weight`.

        Materializes the full tensor, so this is a debugging/serialization
        convenience rather than something to call on a hot path.
        """
        return torch.cat(list(self.chunks), dim=1)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"chunks={len(self.chunks)} (MPS int32-indexing workaround)"
        )


def chunk_oversized_embeddings(root: nn.Module) -> list[str]:
    """Replace every embedding MPS cannot index with a `ChunkedEmbedding`.

    Returns the qualified names that were replaced, empty when nothing needed
    it. Idempotent: `ChunkedEmbedding` is not an `nn.Embedding`, so a second
    pass finds nothing.
    """
    replaced: list[str] = []
    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Embedding) and child.weight.numel() > MPS_MAX_TENSOR_ELEMENTS:
                setattr(module, child_name, ChunkedEmbedding(child))
                replaced.append(f"{module_name}.{child_name}" if module_name else child_name)
    return replaced


def _max_attention_head_dim(text_config) -> Optional[int]:
    """Largest attention head dim across layers, or None if undeterminable.

    Has to cope with heterogeneous configs: Gemma 4 varies head_dim per layer,
    and reading `config.head_dim` on one of those raises
    AmbiguousGlobalPerLayerAttributeError instead of returning a number, so the
    per-layer list must be consulted explicitly.
    """
    per_layer = getattr(text_config, "per_layer_config", None)
    dims: list = []
    if per_layer:
        dims = [getattr(layer, "head_dim", None) for layer in per_layer]
    else:
        try:
            dims = [getattr(text_config, "head_dim", None)]
        except Exception:
            # Raised by the heterogeneity guard; treated as "unknown".
            dims = []
    usable = [d for d in dims if isinstance(d, int) and d > 0]
    return max(usable) if usable else None


def _resolve_attn_implementation(requested: Optional[str]) -> Optional[str]:
    """Coerce flash_attention_2 to sdpa when CUDA isn't available, and avoid sdpa on MPS.

    FA2 is CUDA-only. On MPS/CPU, requesting it either errors at load or
    silently falls back to a slower path; either way the user pays the FA2
    install + import cost for no win. Coerce here so a saved config that
    pins flash_attention_2 still loads on Mac / CPU-only Linux boxes.

    MPS goes further and needs eager. PyTorch's Metal sdpa kernel returns wrong
    results for cached single-token decode against a sliding-window mask, which
    is exactly Gemma 4's layout (`sliding_window=512`, four sliding layers per
    full-attention layer). A full-sequence forward is fine, so the damage shows
    up only during generation: greedy decode with no cache transcribed
    correctly while the identical cached decode produced "Mr. and a". Eager is
    correct there and, at this model size, no slower in practice.
    """
    if torch.backends.mps.is_available() and requested in (None, "sdpa", "flash_attention_2"):
        return "eager"
    if requested != "flash_attention_2":
        return requested
    if not torch.cuda.is_available():
        return "sdpa"
    # CUDA alone isn't enough -- the flash_attn package must actually be
    # importable. It is a source build that routinely fails on RunPod images
    # (its metadata hook imports torch, so any broken torch install takes it
    # down with it). Without this check a CUDA box with no flash-attn raises
    # at from_pretrained instead of quietly using sdpa, which is the same
    # numerics at lower throughput.
    from importlib.util import find_spec

    if find_spec("flash_attn") is None:
        logger.warning(
            "flash_attention_2 requested but flash_attn is not installed; falling back to sdpa."
        )
        return "sdpa"
    return requested


def _gather_audio_embeds(audio_embeds: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
    """Flatten per-sample audio embeddings into a packed tensor.

    For each row i, takes the first ``token_counts[i]`` rows of
    ``audio_embeds[i]`` and concatenates them. If any token count exceeds
    ``audio_embeds.shape[1]``, the deficit is zero-padded.

    Equivalent to a per-sample slice/cat loop but with O(1) host-device
    syncs per call (one ``max().item()``) instead of one per sample.
    """
    _, max_len, _ = audio_embeds.shape
    needed = int(token_counts.max().item())
    if needed > max_len:
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, needed - max_len))
        max_len = needed
    indices = torch.arange(max_len, device=audio_embeds.device).unsqueeze(0)
    mask = indices < token_counts.unsqueeze(1)
    return audio_embeds[mask]


def _patch_gemma_decode_loop(model) -> None:
    """Make a Gemma4ForCausalLM drop `per_layer_inputs` after the first step.

    `Gemma4ForConditionalGeneration` does this itself; the causal LM does not.
    Without it, step 2 of generation hands PLE to forward alongside
    `input_ids` and raises "You cannot specify per_layer_inputs if input_ids is
    provided". Dropping is semantically correct rather than a workaround: PLE
    describes the prompt, and from step 2 on generation feeds a single freshly
    sampled token whose PLE Gemma derives from that token's own id.

    Patched on the instance because subclassing Gemma4ForCausalLM trips
    transformers' experts-implementation check during __init__
    ("does not support setting experts implementation").
    """
    inner_prepare = model.prepare_inputs_for_generation

    # functools.wraps is load-bearing, not cosmetic. Generation's
    # _prepare_model_inputs decides whether a model can accept `inputs_embeds`
    # by testing for it in
    # inspect.signature(prepare_inputs_for_generation).parameters. A bare
    # *args/**kwargs wrapper hides the parameter, and generate() then refuses
    # inputs_embeds with "doesn't have its forwarding implemented" -- fatal
    # here, since audio only ever reaches the decoder as inputs_embeds.
    # `wraps` sets __wrapped__, which inspect.signature follows.
    @functools.wraps(inner_prepare)
    def prepare_inputs_for_generation(*args, is_first_iteration: bool = False, **kwargs):
        model_inputs = inner_prepare(*args, is_first_iteration=is_first_iteration, **kwargs)
        if not is_first_iteration:
            model_inputs.pop("per_layer_inputs", None)
        return model_inputs

    model.prepare_inputs_for_generation = prepare_inputs_for_generation


class ASRModel(PreTrainedModel, GenerationMixin):
    """Audio-to-text model combining an audio encoder, projector, and language model."""

    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False

    TRANSCRIBE_PROMPT = "Transcribe the speech to text"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "ASRModel":
        """Load model from pretrained, handling device placement correctly."""
        from safetensors.torch import load_file
        from transformers.utils.hub import cached_file

        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Set flag to avoid device_map="auto" in sub-model loaders
        cls._is_loading_from_pretrained = True

        try:
            model = cls(config, **kwargs)

            # Load projector weights from safetensors
            subfolder = kwargs.get("subfolder")
            revision = kwargs.get("revision")
            cache_kwargs = {}
            if subfolder:
                cache_kwargs["subfolder"] = subfolder
            if revision:
                cache_kwargs["revision"] = revision

            model_file = cached_file(
                pretrained_model_name_or_path,
                "model.safetensors",
                _raise_exceptions_for_missing_entries=False,
                **cache_kwargs,
            )

            if model_file is not None:
                state_dict = load_file(model_file)
                model.load_state_dict(state_dict, strict=False)

            # Load LoRA adapters if use_lora is enabled
            if getattr(config, "use_lora", False):
                # Check for adapter_config.json (required by PEFT to load adapters)
                adapter_config_file = cached_file(
                    pretrained_model_name_or_path,
                    "adapter_config.json",
                    _raise_exceptions_for_missing_entries=False,
                    **cache_kwargs,
                )
                if adapter_config_file is not None:
                    # Load saved adapter weights using the original repo_id/path
                    # PEFT handles Hub downloads and caching internally
                    from peft import PeftModel

                    model.language_model = PeftModel.from_pretrained(
                        model.language_model,
                        pretrained_model_name_or_path,
                        is_trainable=True,
                        **cache_kwargs,
                    )
                else:
                    # No saved adapters - initialize fresh LLM LoRA for training.
                    # __init__ skips _setup_lora while loading, so call it here.
                    model._setup_lora(config)

            return model
        finally:
            cls._is_loading_from_pretrained = False

    def __init__(self, config: ASRConfig, **kwargs) -> None:
        super().__init__(config)

        self.system_prompt = config.system_prompt
        target_dtype = getattr(torch, config.model_dtype)

        # Audio encoder (frozen)
        self.audio_tower = self._load_audio_encoder(config, target_dtype)

        # Language model (frozen)
        self.language_model = self._load_language_model(config, target_dtype)

        # Initialize tokenizer and special tokens
        self._init_tokenizer(config)

        # Set up generation config with greedy decoding defaults
        self.generation_config = self.language_model.generation_config
        self.generation_config.max_new_tokens = config.max_new_tokens
        self.generation_config.min_new_tokens = config.min_new_tokens
        self.generation_config.num_beams = config.num_beams
        self.generation_config.do_sample = config.do_sample
        # Set sampling params from config (None means use model defaults)
        self.generation_config.temperature = config.temperature
        self.generation_config.top_p = config.top_p
        self.generation_config.top_k = config.top_k
        self.generation_config.use_cache = config.use_cache
        self.generation_config.length_penalty = config.length_penalty
        self.generation_config.repetition_penalty = config.repetition_penalty
        self.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
        # Set EOS tokens, filtering out any that don't exist in the tokenizer.
        # `convert_tokens_to_ids` reports "not in vocab" inconsistently: Qwen-
        # style tokenizers return None, while Gemma's returns unk_token_id.
        # Filtering on None alone left Gemma with eos_token_id=[unk, unk], so
        # generation never stopped and every sample ran to max_new_tokens.
        #
        # The membership test is a name round-trip, NOT `id == unk_token_id`:
        # in GPT-2-lineage tokenizers (SmolLM2, Qwen) unk_token *is*
        # "<|endoftext|>", so comparing ids would discard a legitimate stop
        # token here while fixing Gemma. A token that fell back to unk comes
        # back under unk's name; one that is really in the vocab comes back
        # as itself.
        #
        # The name list alone is not enough, and silently produced garbage on
        # Gemma 4: that template dropped Gemma 2/3's "<end_of_turn>" for a new
        # "<|turn>role\n ... <turn|>\n" scheme, so all three probes missed and
        # eos collapsed to the bare "<eos>" -- a token the template never
        # emits. Training taught the model to close its turn with "<turn|>"
        # (id 106) and generation then ignored it, running every sample to
        # max_new_tokens and burying the transcript under 250 copies of
        # "<turn|>" that `skip_special_tokens` silently swallowed.
        #
        # So derive the real terminator from the template instead of guessing
        # its name: render an assistant turn holding a sentinel and take the
        # first token that follows it. That token is the turn closer by
        # construction, for any decoder, including ones released after this
        # code was written. The name probes stay as a fallback for tokenizers
        # whose template renders no trailing special token.
        eos_ids: list[int] = []
        derived = self._derive_turn_end_token_id()
        if derived is not None:
            eos_ids.append(derived)
        for token in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or self.tokenizer.convert_ids_to_tokens(token_id) != token:
                continue
            if token_id not in eos_ids:
                eos_ids.append(token_id)
        # The chat-template stop tokens above are additions, not replacements:
        # the tokenizer's own EOS is always a valid stop and is the only one
        # left for a decoder using none of those three templates.
        tokenizer_eos = self.tokenizer.eos_token_id
        if tokenizer_eos is not None and tokenizer_eos not in eos_ids:
            eos_ids.append(tokenizer_eos)
        self.generation_config.eos_token_id = eos_ids
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Feature extractor for audio preprocessing
        self.feature_extractor = self._create_feature_extractor(config)

        # Audio projector (trainable unless freeze_projector is set)
        self.projector = self._create_projector(config, target_dtype)

        # Setup LoRA if enabled (Stage 2 fine-tuning)
        # Skip if loading from pretrained - from_pretrained will handle adapter loading
        if getattr(config, "use_lora", False) and not getattr(
            self.__class__, "_is_loading_from_pretrained", False
        ):
            self._setup_lora(config)

        # Freeze projector if specified (for Stage 2 LoRA-only training)
        if getattr(config, "freeze_projector", False):
            self.projector.requires_grad_(False)

        # Freeze the text-vocab embedding table (preserves base Qwen3's
        # token→embedding mapping during joint fine-tune). With
        # tie_word_embeddings=True the same tensor backs lm_head, so this
        # also freezes the output projection. Audio tokens bypass this
        # table — they're scattered into inputs_embeds via masked_scatter
        # at <audio> positions (forward(), below), so the audio path is
        # unaffected. Mirrors Baichuan-Audio's stage-2 policy of training
        # all decoder params except the text embedding and LM head.
        if getattr(config, "freeze_text_embed_tokens", False):
            self.language_model.get_input_embeddings().weight.requires_grad_(False)

        # Freeze Gemma 4's per-layer embedding table. This is a separate
        # vocabulary lookup from embed_tokens (see ASRConfig for why it is
        # worth its own flag), so `freeze_text_embed_tokens` does not reach it.
        if getattr(config, "freeze_text_per_layer_embeddings", False):
            table = self._per_layer_embedding_table()
            if table is None:
                logging.warning(
                    "freeze_text_per_layer_embeddings=True but %s has no per-layer "
                    "embedding table — ignoring. Only Gemma 4 style decoders have one.",
                    config.text_model_id,
                )
            else:
                table.weight.requires_grad_(False)

        # For model parallelism
        self._no_split_modules = getattr(self.language_model, "_no_split_modules", [])

    def _create_feature_extractor(self, config: ASRConfig):
        """Create the appropriate feature extractor for the audio encoder."""
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(config.audio_model_id)
        # Whisper's encoder requires a fixed 3000 mel frames (30s) and the
        # feature extractor pads to that by default — leave it alone. Other
        # encoders (e.g. GLM-ASR) accept variable-length input, so we disable
        # padding to avoid wasting compute on silent frames.
        if "whisper" not in config.audio_model_id.lower():
            feature_extractor.padding = False
        return feature_extractor

    @classmethod
    def _load_audio_encoder(cls, config: ASRConfig, dtype: torch.dtype) -> nn.Module:
        """Load the audio encoder; freeze unless `config.freeze_audio_encoder=False`.

        When unfrozen, the encoder participates in joint training — pair with a
        much lower `encoder_learning_rate` than the projector/decoder LRs
        (encoder is large, sensitive to perturbation, and shouldn't drift far
        from its pretrained features). See `ASRTrainer.create_optimizer` for the
        LR routing.
        """
        encoder_kwargs = {
            "attn_implementation": _resolve_attn_implementation(config.attn_implementation),
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        if "whisper" in config.audio_model_id.lower():
            from transformers import WhisperModel

            full_model = WhisperModel.from_pretrained(config.audio_model_id, **encoder_kwargs)
            encoder = full_model.encoder
            del full_model
        elif "granite-speech" in config.audio_model_id.lower():
            # Granite Speech 5.0 TurboCTC is encoder-only (Conformer blocks +
            # a 16k-BPE CTC head). GraniteSpeech5Encoder.from_pretrained pulls
            # just the encoder out of the GraniteSpeech5ForCTC checkpoint and
            # leaves the CTC head behind.
            #
            # Requires transformers >= 5.16 for the native `granite_speech5`
            # architecture. There is no trust_remote_code fallback: the
            # checkpoint ships modeling .py files but its config.json has no
            # `auto_map`, so Auto* classes can't resolve them on older
            # versions -- they fail with KeyError('granite_speech5_ctc').
            from transformers import GraniteSpeech5Encoder

            # Granite's block-attention implementation has no FA2 kernel and
            # raises ValueError on attn_implementation="flash_attention_2".
            # _resolve_attn_implementation only downgrades FA2 when CUDA is
            # absent, so on a CUDA box the inherited default would hard-fail
            # at load. Pin sdpa regardless of what the config asks for.
            granite_kwargs = {**encoder_kwargs, "attn_implementation": "sdpa"}
            encoder = GraniteSpeech5Encoder.from_pretrained(config.audio_model_id, **granite_kwargs)
        elif "glm" in config.audio_model_id.lower():
            # GLM-ASR stores its encoder at audio_tower (GlmAsrEncoder), but
            # which object owns that attribute depends on the transformers
            # version. Under the 5.x composite-model layout,
            # GlmAsrForConditionalGeneration is a thin wrapper holding a
            # GlmAsrModel at `.model`, and it is GlmAsrModel that owns
            # audio_tower / language_model / multi_modal_projector. Older
            # flat checkpoints hung them off the top-level model. Resolve the
            # owner instead of assuming, so neither layout AttributeErrors
            # at load.
            from transformers import AutoModelForSeq2SeqLM
            from transformers import __version__ as transformers_version

            full_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.audio_model_id, trust_remote_code=True, **encoder_kwargs
            )
            inner = getattr(full_model, "model", None)
            holder = inner if hasattr(inner, "audio_tower") else full_model
            if not hasattr(holder, "audio_tower"):
                raise AttributeError(
                    f"{type(full_model).__name__} exposes no audio_tower at "
                    "`.audio_tower` or `.model.audio_tower`; GLM-ASR encoder "
                    f"extraction needs updating for transformers "
                    f"{transformers_version}."
                )
            encoder = holder.audio_tower
            # Drop the LLM decoder and projector to free their VRAM. These must
            # go through the same owner as audio_tower: assigning None to a name
            # that is not a registered submodule on that object silently creates
            # a plain attribute and frees nothing.
            holder.language_model = None
            holder.multi_modal_projector = None
            del full_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        # Explicit cast: from_pretrained's `dtype=` kwarg is honored
        # inconsistently across loader paths (especially trust_remote_code
        # branches like GLM-ASR), leaving submodules in fp32. FA2's startup
        # then complains "current dype is torch.float32, expected fp16/bf16",
        # and even with sdpa the projector→encoder feed mismatches dtypes.
        # `.to(dtype=...)` after load is idempotent and forces the issue.
        encoder = encoder.to(dtype=dtype)
        if getattr(config, "freeze_audio_encoder", True):
            encoder.requires_grad_(False)
            encoder.train(False)  # equivalent to .eval(); avoids a security hook false-positive
        return encoder

    @classmethod
    def _load_language_model(cls, config: ASRConfig, dtype: torch.dtype) -> PreTrainedModel:
        """Load and freeze the language model."""
        attn_implementation = _resolve_attn_implementation(config.attn_implementation)

        # FlashAttention only has kernels for head_dim <= 256, and it fails at
        # the first forward rather than at load. Gemma 4 E2B trips this: 7 of
        # its 35 layers use head_dim=512 (the other 28 use 256), so FA2 can
        # never run this architecture regardless of flash-attn version. Detect
        # it from the config and fall back instead of dying a step into
        # training.
        if attn_implementation == "flash_attention_2":
            probe = AutoConfig.from_pretrained(config.text_model_id, trust_remote_code=True)
            text_probe = probe.get_text_config() if hasattr(probe, "get_text_config") else probe
            head_dim = _max_attention_head_dim(text_probe)
            if head_dim is not None and head_dim > FLASH_ATTENTION_MAX_HEAD_DIM:
                logger.warning(
                    "%s has max head_dim=%d, above FlashAttention's limit of %d; "
                    "using sdpa for the decoder.",
                    config.text_model_id,
                    head_dim,
                    FLASH_ATTENTION_MAX_HEAD_DIM,
                )
                attn_implementation = "sdpa"

        decoder_kwargs = {
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)

        # Gemma 4 checkpoints are natively multimodal: AutoModelForCausalLM maps
        # `gemma4` to Gemma4ForConditionalGeneration, which is both wasteful
        # (a vision tower + audio tower this model never calls) and unusable
        # for inference. Convert it to the text-only causal LM.
        #
        # Gated on the model id, not on the `.model.language_model` layout:
        # that layout is the generic transformers composite-model shape, shared
        # by GLM-ASR and Llava/Qwen-Omni style checkpoints. Probing for it
        # structurally would funnel those into _gemma_text_only, which imports
        # transformers.models.gemma4 and transplants weights into a
        # Gemma4ForCausalLM shell built from a foreign text_config. The
        # hasattr check is kept as a guard so an already-text-only gemma-4
        # checkpoint passes through untouched.
        gemma_inner = getattr(decoder, "model", None)
        is_gemma4 = "gemma-4" in (config.text_model_id or "").lower()
        if is_gemma4 and gemma_inner is not None and hasattr(gemma_inner, "language_model"):
            decoder = cls._gemma_text_only(decoder)
        # See _load_audio_encoder note: idempotent post-load cast to dodge the
        # FA2 "current dype is fp32" warning when from_pretrained's dtype kwarg
        # isn't fully propagated to every submodule.
        decoder = decoder.to(dtype=dtype)
        decoder.config.use_cache = getattr(config, "use_cache", True)
        if getattr(config, "freeze_language_model", True):
            decoder.requires_grad_(False)
            decoder.train(False)
        return decoder

    @staticmethod
    def _gemma_text_only(loaded: PreTrainedModel) -> PreTrainedModel:
        """Turn a loaded Gemma4ForConditionalGeneration into a text-only CausalLM.

        Two independent problems force this transplant rather than just loading
        the right class directly:

        1. Only `Gemma4ForCausalLM` supports `inputs_embeds` in `.generate()`.
           Its `prepare_inputs_for_generation` accepts the argument; the
           multimodal wrapper's does not, so generation dies with "You passed
           `inputs_embeds` to `.generate()`, but the model class
           Gemma4ForConditionalGeneration doesn't have its forwarding
           implemented." Audio reaches this decoder exclusively as
           inputs_embeds, so the wrapper cannot do inference at all.

        2. `Gemma4ForCausalLM.from_pretrained` cannot read the published
           checkpoint. Its keys are `model.language_model.*` (the multimodal
           layout) while the causal LM expects `model.*`, and gemma4 ships no
           checkpoint conversion mapping. Critically it fails *silently*:
           every tensor is reported MISSING and randomly initialized
           (embedding std lands at initializer_range), producing a model that
           trains without error and learns nothing.

        So the wrapper is loaded for its correct key layout, then its text
        model and lm_head are moved into a shell built on the meta device --
        which allocates nothing, because every real parameter is transplanted.
        Dropping the wrapper also frees the vision and audio towers: the
        returned text-only model holds 4,628,569,344 parameters against the
        checkpoint's 5,123,178,979, so ~494M are released.
        """
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        text_config = loaded.config.get_text_config()
        with torch.device("meta"):
            shell = Gemma4ForCausalLM(text_config)

        shell.model = loaded.model.language_model
        shell.lm_head = loaded.lm_head
        shell.generation_config = loaded.generation_config
        shell.tie_weights()

        # Gemma4ForCausalLM does not drop `per_layer_inputs` between decode
        # steps, while the multimodal wrapper does. Without the drop, step 2
        # of generation passes PLE alongside `input_ids` and forward raises
        # "You cannot specify per_layer_inputs if input_ids is provided". The
        # drop is semantically right, not a workaround: PLE describes the
        # prompt, and from step 2 on generation feeds one freshly sampled
        # token whose PLE Gemma derives itself from that token's id.
        #
        # Patched on the instance rather than via a subclass: subclassing
        # Gemma4ForCausalLM trips transformers' experts-implementation check
        # ("does not support setting experts implementation") during __init__.
        _patch_gemma_decode_loop(shell)

        stranded = [n for n, t in shell.named_parameters() if t.device.type == "meta"]
        stranded += [n for n, t in shell.named_buffers() if t.device.type == "meta"]
        if stranded:
            raise RuntimeError(
                "Gemma text-only conversion left parameters on the meta device "
                f"(would silently produce garbage): {stranded[:5]}"
            )
        return shell

    def _create_projector(self, config: ASRConfig, dtype: torch.dtype) -> nn.Module:
        """Create the trainable audio projector."""
        # Auto-detect dimensions if not specified
        if config.encoder_dim is None:
            enc_cfg = self.audio_tower.config
            config.encoder_dim = getattr(enc_cfg, "hidden_size", None) or getattr(
                enc_cfg, "d_model", None
            )
            if config.encoder_dim is None:
                raise ValueError("Could not auto-detect encoder_dim. Please specify in config.")

        if config.llm_dim is None:
            dec_cfg = self.language_model.config
            # Composite configs (Gemma 4) keep hidden_size on `text_config`;
            # get_text_config() returns self for plain decoders like Qwen3.
            if hasattr(dec_cfg, "get_text_config"):
                dec_cfg = dec_cfg.get_text_config()
            config.llm_dim = getattr(dec_cfg, "hidden_size", None) or getattr(
                dec_cfg, "d_model", None
            )
            if config.llm_dim is None:
                raise ValueError("Could not auto-detect llm_dim. Please specify in config.")

        # Select projector type based on config
        projector_type = getattr(config, "projector_type", "mlp")
        projector_class = PROJECTOR_CLASSES.get(projector_type)
        if projector_class is None:
            raise ValueError(
                f"Unknown projector_type: {projector_type}. "
                f"Valid options: {list(PROJECTOR_CLASSES.keys())}"
            )
        projector = projector_class(config)

        # Move projector to same device as language model (important when using quantization)
        device = next(self.language_model.parameters()).device
        # The projector may run at a higher precision than the frozen stack --
        # it is the only module with optimizer state, so it is the only one
        # that needs master-weight precision. See ASRConfig.projector_dtype.
        proj_name = getattr(config, "projector_dtype", None)
        proj_dtype = getattr(torch, proj_name) if proj_name else dtype
        self._projector_dtype = proj_dtype
        return projector.to(device=device, dtype=proj_dtype)

    def _setup_lora(self, config: ASRConfig):
        """Apply LoRA adapters to the language model for Stage 2 fine-tuning."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)

    def _init_tokenizer(self, config: ASRConfig):
        """Initialize tokenizer with audio token."""
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id, trust_remote_code=True)

        # Set pad token. Prefer a dedicated pad token if the tokenizer has one
        # (e.g. Qwen's <|finetune_right_pad_id|>); otherwise fall back to
        # eos_token, which is the standard pattern for Llama-style tokenizers
        # (SmolLM2, Llama, etc.) that ship without a separate pad token.
        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            if "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
            elif self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Audio placeholder token. Prefer a token the tokenizer already knows
        # (Gemma 4 ships "<|audio|>") so no embedding resize is needed at all.
        # `unk_token_id` is the sentinel convert_tokens_to_ids returns for an
        # unknown token, so it means "not in vocab" rather than a usable id.
        self.audio_token = getattr(config, "audio_token", None) or "<audio>"
        existing_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        already_in_vocab = existing_id is not None and existing_id != self.tokenizer.unk_token_id

        if not already_in_vocab:
            existing_special = getattr(self.tokenizer, "additional_special_tokens", None) or []
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": existing_special + [self.audio_token]}
            )
            # mean_resizing=True initializes the new row at the mean of existing
            # rows so its scale matches the pretrained distribution. The
            # input-side embedding is overwritten via masked_scatter and never
            # seen by the LM, but with tied embeddings (Qwen3-0.6B) this same
            # row is the lm_head column for predicting the audio token; a
            # Gaussian draw at config.initializer_range was visible in
            # early-step logits.
            self.language_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.tokenizer.padding_side = "right"

        # Sync token IDs to configs
        for cfg in [self.config.text_config, self.language_model.config, self.generation_config]:
            if cfg is not None:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def _derive_turn_end_token_id(self) -> Optional[int]:
        """Return the token id the chat template uses to close an assistant turn.

        Renders a throwaway assistant turn whose content is a sentinel, then
        takes the first token after it. Whatever the template appends there is
        the turn terminator by construction -- "<turn|>" on Gemma 4,
        "<|im_end|>" on Qwen/SmolLM2 -- so this stays correct for decoders
        whose template names nobody has hardcoded yet.

        Returns None when the template is missing, the sentinel survives
        rendering (some templates strip or transform content), or the trailing
        token is ordinary text rather than a special token -- callers fall back
        to the hardcoded name probes.
        """
        # Invisible separators keep the sentinel out of any word-level
        # trim/strip the template applies to assistant content.
        sentinel = "⁣turnendprobe⁣"
        try:
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": sentinel},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:  # noqa: BLE001 - any template failure means "no answer"
            return None
        if not isinstance(rendered, str) or sentinel not in rendered:
            return None
        tail_ids = self.tokenizer(rendered.split(sentinel)[-1], add_special_tokens=False)[
            "input_ids"
        ]
        if not tail_ids:
            return None
        # Require a special token: a template that ends the turn with plain
        # text has no stop token to find, and adopting a text token as EOS
        # would truncate real transcripts.
        first = tail_ids[0]
        return first if first in set(self.tokenizer.all_special_ids) else None

    def _apply(self, *args, **kwargs):
        """Repair MPS-unsafe embeddings whenever the model lands on MPS.

        Device placement happens after construction -- `pipeline` builds the
        model, then moves it -- so the check belongs on the move, not in
        __init__. `_apply` is the single funnel every relocation goes through
        (`.to()`, `.cuda()`, `.mps()`, dtype casts), which keeps this off the
        many individual entry points.

        Without it, Gemma 4 E2B on MPS reads the wrong rows out of its 2.35B-
        element per-layer embedding table and transcribes confident nonsense at
        full speed. Chunking keeps the model on the GPU -- falling back to CPU
        would be correct but costs roughly an order of magnitude in latency.
        """
        module = super()._apply(*args, **kwargs)
        try:
            on_mps = any(p.device.type == "mps" for p in module.parameters())
        except StopIteration:  # pragma: no cover - parameterless model
            return module
        if on_mps and (replaced := chunk_oversized_embeddings(module)):
            logger.warning(
                "MPS cannot index %s with 32-bit offsets; split into chunks to "
                "keep the lookup correct on GPU.",
                ", ".join(replaced),
            )
        return module

    def train(self, mode: bool = True):
        """Set train/eval mode, but keep frozen submodules out of train mode.

        HF Trainer calls `model.train()` at the top of every training step, which
        recursively switches every submodule into train mode — re-enabling dropout
        on modules with `requires_grad_(False)`. The frozen encoder (and the LM
        when `freeze_language_model=True`) should always run deterministically;
        train-mode dropout only adds noise that can't improve a frozen network.
        """
        super().train(mode)
        if getattr(self.config, "freeze_audio_encoder", True):
            self.audio_tower.train(False)
        if getattr(self.config, "freeze_language_model", True):
            self.language_model.train(False)
        return self

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func=None,
        every_n_layers: int = 1,
        **kwargs,
    ):
        """Enable/disable gradient checkpointing on the trainable submodules.

        Routes the request to whichever components are actually trainable in
        this run. The LM is always reached (its forward activations are
        needed for backprop to the projector even when its weights are
        frozen). The encoder is reached only when `freeze_audio_encoder` is
        False — when frozen, no gradient flows through it and checkpointing
        would just add recompute cost for no memory savings.

        The signature must track `PreTrainedModel.gradient_checkpointing_enable`,
        which calls this with keyword arguments. transformers 5.16 added
        `every_n_layers`, and because it is passed by keyword the previous
        two-argument override raised TypeError *after* the model and dataset
        had loaded. `**kwargs` absorbs future additions so an upgrade
        degrades to "new option ignored" instead of a crash, and each kwarg is
        filtered against the target's own signature before being forwarded so
        submodules on older signatures still work.

        One name is load-bearing: `value` must never become a parameter here.
        Upstream sniffs for it to detect the pre-4.35 checkpointing format and
        would silently take the legacy `self.apply(...)` path instead.
        """
        import inspect

        forwardable = dict(kwargs)
        forwardable["every_n_layers"] = every_n_layers
        if gradient_checkpointing_func is not None:
            forwardable["gradient_checkpointing_func"] = gradient_checkpointing_func

        # The LLM still stores activations during forward for backprop to projector.
        # Gradient checkpointing trades compute for memory by recomputing activations.
        for submodule in self._gradient_checkpointing_targets():
            setter = getattr(submodule, "_set_gradient_checkpointing", None)
            if setter is not None:
                accepted = inspect.signature(setter).parameters
                passthrough = {k: v for k, v in forwardable.items() if k in accepted}
                setter(enable=enable, **passthrough)
            elif hasattr(submodule, "gradient_checkpointing_enable") and enable:
                submodule.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            elif hasattr(submodule, "gradient_checkpointing_disable") and not enable:
                submodule.gradient_checkpointing_disable()

    def _gradient_checkpointing_targets(self) -> list[nn.Module]:
        """Return the submodules that should respond to gradient_checkpointing
        toggles. Always includes the LM (activations are on the gradient path
        to the projector); includes the encoder only when it's trainable.
        """
        targets: list[nn.Module] = [self.language_model]
        if not getattr(self.config, "freeze_audio_encoder", True):
            targets.append(self.audio_tower)
        return targets

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_output_embeddings(value)

    def get_processor(self):
        """Get the processor for this model."""
        try:
            from .asr_processing import ASRProcessor
        except ImportError:
            from asr_processing import ASRProcessor  # type: ignore[no-redef]

        return ASRProcessor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            projector=self.projector,
            encoder_conv_layers=self.config.encoder_conv_layers,
            audio_token=self.audio_token,
        )

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Save trainable weights: projector, plus the encoder/LM when unfrozen.

        Every module this returns is gated on the same freeze flag that decides
        whether it trains, so a module that receives gradients is always
        serialized. Getting that pairing wrong is silent: training runs to
        completion, checkpoints save without error, and the updates are simply
        absent when the checkpoint is reloaded.

        With LoRA attached, the language_model entries are flattened to plain
        (non-PEFT) HF naming so model.safetensors round-trips through
        ASRModel.from_pretrained — which builds a vanilla base LM, overlays
        these weights, and only then re-attaches PEFT. lora_*/adapter weights
        are skipped here; PEFT serializes them separately as
        adapter_model.safetensors via the save_pretrained path below.

        Note this can't be derived from `requires_grad`: the LoRA branch below
        deliberately re-saves frozen base-layer weights, and
        `freeze_text_embed_tokens` freezes an embedding that must still
        round-trip with the rest of the LM.
        """
        sd = {f"projector.{k}": v for k, v in self.projector.state_dict().items()}
        if not getattr(self.config, "freeze_audio_encoder", True):
            sd.update({f"audio_tower.{k}": v for k, v in self.audio_tower.state_dict().items()})
        if not getattr(self.config, "freeze_language_model", True):
            lm = self.language_model
            if hasattr(lm, "peft_config"):
                for k, v in lm.state_dict().items():
                    if "lora_" in k:
                        continue
                    if k.startswith("base_model.model."):
                        k = k[len("base_model.model.") :]
                    # LoRA layers wrap the original Linear as `<name>.base_layer.<weight|bias>`.
                    k = k.replace(".base_layer.", ".")
                    sd[f"language_model.{k}"] = v
            else:
                sd.update({f"language_model.{k}": v for k, v in lm.state_dict().items()})
        return sd

    def _ple_text_model(self):
        """Locate the submodule owning `get_per_layer_inputs`, or None.

        Only Gemma 4 style decoders have one. The attribute sits on
        Gemma4TextModel, which is `language_model.model.language_model` for the
        multimodal checkpoint and `language_model.model` for a text-only one.
        """
        candidates = (
            getattr(getattr(self.language_model, "model", None), "language_model", None),
            getattr(self.language_model, "model", None),
            self.language_model,
        )
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "get_per_layer_inputs"):
                return candidate
        return None

    def _per_layer_embedding_table(self) -> Optional[nn.Module]:
        """Gemma 4's `embed_tokens_per_layer` lookup table, or None.

        It hangs off the same submodule as `get_per_layer_inputs`, so reuse
        that walk instead of guessing the decoder's layout a second time.
        """
        ple_model = self._ple_text_model()
        if ple_model is None:
            return None
        return getattr(ple_model, "embed_tokens_per_layer", None)

    def _per_layer_kwargs(self, input_ids: Optional[torch.Tensor]) -> dict:
        """Precompute Gemma 4 per-layer embeddings (PLE) from clean `input_ids`.

        Gemma 4 builds a token-identity PLE component by looking `input_ids` up
        in `embed_tokens_per_layer`. It must be computed *before* masked_scatter
        overwrites the audio positions, because Gemma's `inputs_embeds`-only
        path tries to recover ids by reversing the main embedding -- an exact
        row match against `embed_tokens.weight * sqrt(hidden_size)`. Projector
        output matches no row, so that path raises RuntimeError (and would be
        an O(B*T*vocab*hidden) comparison even if it matched).

        Computing it here keeps every text token's PLE row intact and leaves
        audio positions on the audio placeholder's own row, which is exactly
        what Gemma does natively for image/audio placeholder tokens.

        Returns {} for decoders without PLE, so callers can splat it blindly.
        """
        if input_ids is None:
            return {}
        text_model = self._ple_text_model()
        if text_model is None:
            return {}
        if not getattr(text_model.config, "hidden_size_per_layer_input", None):
            return {}
        return {"per_layer_inputs": text_model.get_per_layer_inputs(input_ids, None)}

    def _compute_encoder_output_lengths(
        self,
        audio_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample encoder output lengths using conv layer formulas."""
        return compute_encoder_output_length(
            audio_attention_mask.sum(dim=-1),
            self.config.encoder_conv_layers,
        )

    def _encode_audio(
        self,
        audio_features: torch.Tensor,
        expected_token_counts: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio features and return flattened embeddings matching expected_token_counts.

        Args:
            audio_features: Mel spectrogram features (batch, n_mels, mel_len)
            expected_token_counts: Per-sample audio token counts as int64 tensor (batch,).
            audio_attention_mask: Mel-frame padding mask, forwarded to the
                encoder when `config.encoder_attention_mask` is set.

        Returns:
            Flattened audio embeddings of shape (sum(expected_token_counts), hidden_dim).
        """
        # SpecAugment is applied on the mel input, training-only. Most useful
        # when the encoder is trainable; on the frozen-encoder path it still
        # perturbs the projector's input slightly but with no gradient flowing
        # back to the encoder to leverage the diversity.
        if (
            self.training
            and getattr(self.config, "apply_spec_augment", False)
            and audio_features.numel() > 0
        ):
            audio_features = self._mask_input_features(audio_features, audio_attention_mask)

        # When the encoder is frozen, skip gradient tracking through it — cuts
        # activation memory and matches the prior published recipe's behavior.
        # When trainable, we MUST allow gradients to flow back to encoder
        # params; wrapping in no_grad here would silently zero encoder
        # gradients regardless of requires_grad on its parameters.
        # Conformer encoders (Granite) must see the padding mask -- their block
        # attention otherwise mixes pad frames into real ones. Gated on
        # `encoder_attention_mask` so the Whisper/GLM path stays byte-identical
        # to prior runs; see ASRConfig for the measured impact.
        tower_kwargs = {"input_features": audio_features}
        if audio_attention_mask is not None and getattr(
            self.config, "encoder_attention_mask", False
        ):
            tower_kwargs["attention_mask"] = audio_attention_mask.to(audio_features.device)

        encoder_frozen = getattr(self.config, "freeze_audio_encoder", True)
        if encoder_frozen:
            with torch.no_grad():
                encoder_out = self.audio_tower(**tower_kwargs)
                hidden_states = encoder_out.last_hidden_state
        else:
            encoder_out = self.audio_tower(**tower_kwargs)
            hidden_states = encoder_out.last_hidden_state

        # The encoder may be bf16 while the projector holds fp32 master
        # weights, so align explicitly rather than relying on autocast being
        # active -- eval and MPS paths run outside it.
        proj_dtype = getattr(self, "_projector_dtype", None)
        if proj_dtype is not None and hidden_states.dtype != proj_dtype:
            hidden_states = hidden_states.to(proj_dtype)
        audio_embeds = self.projector(hidden_states)

        token_counts = expected_token_counts.to(device=audio_embeds.device, dtype=torch.long)
        return _gather_audio_embeds(audio_embeds, token_counts)

    def _mask_input_features(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SpecAugment on mel input (pure-torch, vectorized, compile-ready).

        Follows the same semantics as
        `transformers.models.whisper.modeling_whisper.WhisperModel._mask_input_features`
        (wav2vec2-style mask sampling: sample N start positions per sample,
        mask `mask_length` frames forward from each), but reimplemented in
        pure torch so it stays inside the autograd graph without crossing
        the numpy boundary. This avoids inductor codegen failures
        (e.g. the `‘zuf0’ was not declared` error from the prior numpy ->
        torch.tensor round-trip) AND avoids the per-forward host-to-GPU
        sync that the numpy path required.

        One minor semantic divergence vs the upstream helper: this version
        allows mask spans to overlap, while upstream rejects overlapping
        samples. For ASR purposes this is irrelevant — occasional region
        double-coverage has no measurable effect on the regularization
        signal.

        Reads ASRConfig fields by Whisper naming convention: mask_time_prob,
        mask_time_length, mask_time_min_masks, mask_feature_prob,
        mask_feature_length, mask_feature_min_masks.

        Args:
            input_features: (batch, n_mels, mel_len) log-mel features.
            attention_mask: (batch, mel_len) mel-frame padding mask. Like
                upstream, it confines time-axis span count and start
                positions to each sample's unpadded length. Without it,
                spans are drawn uniformly over the padded axis, so under
                `padding="longest"` a 1s clip batched with a 19s clip gets
                roughly 1/19 of its spans in real audio — short utterances
                go effectively unaugmented while long ones take the full
                dose. Whisper is unaffected either way (fixed 3000 frames);
                the variable-length encoders are the ones that need it.
                Pass None to treat the whole axis as valid.

        Returns:
            Same-shape tensor with time-axis and/or feature-axis masks zeroed.
        """
        input_features = input_features.clone()
        config = self.config
        # Whisper/GLM-ASR hand us (batch, n_mels, mel_len); Conformer encoders
        # (Granite, Parakeet, Nemotron) hand us (batch, time, feature_dim).
        # Masking the wrong axis is silent -- it still runs and still zeroes
        # cells -- so the axis order is read from config rather than guessed.
        time_major = getattr(config, "audio_features_time_major", False)
        if time_major:
            batch_size, sequence_length, hidden_size = input_features.size()
            time_dim, feature_dim = 1, 2
        else:
            batch_size, hidden_size, sequence_length = input_features.size()
            time_dim, feature_dim = 2, 1
        device = input_features.device

        valid_lengths = None
        if attention_mask is not None:
            if attention_mask.shape[-1] != sequence_length:
                raise ValueError(
                    "SpecAugment attention_mask time axis "
                    f"({attention_mask.shape[-1]}) does not match input_features "
                    f"({sequence_length}). Masking against the wrong axis is silent, "
                    "so this is raised rather than ignored."
                )
            valid_lengths = attention_mask.to(device).sum(dim=-1)

        if getattr(config, "mask_time_prob", 0.0) > 0:
            mask_time = self._sample_mask_indices(
                batch_size,
                sequence_length,
                mask_prob=config.mask_time_prob,
                mask_length=config.mask_time_length,
                min_masks=config.mask_time_min_masks,
                device=device,
                valid_lengths=valid_lengths,
            )
            # Broadcast (B, T) over the feature axis to mask all bins at
            # masked times: (B, 1, T) feature-major, (B, T, 1) time-major.
            input_features.masked_fill_(mask_time.unsqueeze(feature_dim), 0)

        if getattr(config, "mask_feature_prob", 0.0) > 0:
            mask_feature = self._sample_mask_indices(
                batch_size,
                hidden_size,
                mask_prob=config.mask_feature_prob,
                mask_length=config.mask_feature_length,
                min_masks=config.mask_feature_min_masks,
                device=device,
                valid_lengths=None,  # mel-bin axis is never padded
            )
            # Broadcast (B, F) over the time axis to mask all time steps at
            # masked bins: (B, F, 1) feature-major, (B, 1, F) time-major.
            input_features.masked_fill_(mask_feature.unsqueeze(time_dim), 0)

        return input_features

    @staticmethod
    def _sample_mask_indices(
        batch_size: int,
        axis_length: int,
        mask_prob: float,
        mask_length: int,
        min_masks: int,
        device: torch.device,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized SpecAugment mask sampler — torch.compile-friendly.

        Returns a (batch_size, axis_length) bool tensor where True marks
        a position covered by at least one mask span. Spans may overlap
        (see _mask_input_features docstring on the semantic difference vs
        the upstream Whisper helper).

        `valid_lengths` is a (batch_size,) tensor of unpadded lengths along
        this axis; both the span count and the start positions are derived
        per sample from it, so every span lands inside real content. Pass
        None for an axis with no padding.
        """
        # Worst-case span count, from the full axis. Computed as a Python int
        # from axis_length (an upper bound on any sample's valid length)
        # rather than from valid_lengths.max(), which would need a .item()
        # call and reintroduce the per-forward GPU->host sync this function
        # was rewritten to avoid. Surplus spans are masked off per sample
        # below, so over-allocating here is free apart from memory.
        #
        # Matches the upstream formula (ignoring the epsilon noise term, which
        # only shifts the count by ±1 stochastically — negligible at the
        # default mask_time_prob=0.05 / mask_length=10 setting which gives
        # ~5 spans for a typical 1500-frame mel input).
        max_spans = max(int(mask_prob * axis_length / mask_length + 0.5), min_masks)
        if max_spans == 0:
            return torch.zeros(batch_size, axis_length, device=device, dtype=torch.bool)

        if valid_lengths is None:
            lengths = torch.full((batch_size,), axis_length, device=device, dtype=torch.long)
        else:
            lengths = valid_lengths.to(device=device, dtype=torch.long).clamp(
                min=1, max=axis_length
            )

        # Per-sample span count, so a short clip is not given a long clip's
        # dose of masking. Bounded above by max_spans since lengths <= axis_length.
        num_spans = torch.clamp(
            (mask_prob * lengths.float() / mask_length + 0.5).long(), min=min_masks
        )  # (B,)

        # Sample start positions independently per sample × span, inside that
        # sample's own valid range. Clamp so a span of length mask_length never
        # starts past the end of short content. torch.rand is [0, 1), so the
        # floor below never reaches max_start.
        max_start = torch.clamp(lengths - mask_length + 1, min=1)  # (B,)
        starts = (
            torch.rand(batch_size, max_spans, device=device) * max_start.unsqueeze(1)
        ).long()  # (B, N)

        # Drop the spans a sample didn't earn (its num_spans < max_spans).
        span_active = torch.arange(max_spans, device=device).unsqueeze(0) < num_spans.unsqueeze(1)

        # For each (sample, span, position), True iff position ∈ [start, start+mask_length).
        positions = torch.arange(axis_length, device=device).view(1, 1, -1)  # (1, 1, T)
        starts_b = starts.unsqueeze(-1)  # (B, N, 1)
        span_mask = (positions >= starts_b) & (positions < starts_b + mask_length)
        span_mask &= span_active.unsqueeze(-1)
        # Reduce over the span dim: True if ANY span covers this position.
        return span_mask.any(dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        audio_token_counts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training and inference."""
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Gemma 4 PLE: derived from input_ids, so it must be captured before the
        # masked_scatter below replaces the audio rows. No-op ({}) elsewhere.
        ple_kwargs = self._per_layer_kwargs(input_ids)

        if input_features is not None and input_ids is not None:
            is_audio_token = input_ids == self.audio_token_id
            if audio_token_counts is None:
                audio_token_counts = is_audio_token.sum(dim=-1)
            else:
                audio_token_counts = audio_token_counts.to(
                    device=input_ids.device, dtype=torch.long
                )

            audio_embeds = self._encode_audio(
                input_features, audio_token_counts, audio_attention_mask
            )

            audio_token_mask = is_audio_token.unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device),
                audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        # Forward label_smoothing to the LM's loss_function via **kwargs.
        # transformers.loss.loss_utils.ForCausalLMLoss → fixed_cross_entropy
        # forwards extra kwargs to F.cross_entropy, which accepts label_smoothing.
        # When apply_liger_kernel_to_qwen3() has patched the LM, the smoothing
        # is consumed by liger's fused linear CE (no (B,T,V) materialization).
        # Zeroed on eval so eval/loss is raw CE and comparable to LS=0 runs.
        if labels is not None and self.training and self.config.label_smoothing > 0:
            kwargs.setdefault("label_smoothing", self.config.label_smoothing)

        # Ask for the dataclass output explicitly. Every consumer below and in
        # generate() reads `outputs.loss` / `outputs.logits` by attribute, so
        # this only makes an existing assumption visible -- but it also keeps
        # liger off a deprecated path. Its patched forwards do
        # `return_dict = kwargs.pop("return_dict", None)` and fall back to
        # `self.config.use_return_dict` when that is None, which in
        # transformers 5.x logs "`use_return_dict` is deprecated! Use
        # `return_dict` instead!". Supplying the value means the fallback never
        # runs.
        kwargs.setdefault("return_dict", True)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **ple_kwargs,
            **kwargs,
        )

        if outputs.loss is not None and hasattr(self.projector, "get_aux_loss"):
            aux_loss = self.projector.get_aux_loss()
            if aux_loss is not None and aux_loss.numel() > 0:
                outputs.loss = outputs.loss + aux_loss.to(outputs.loss.device)

        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation, handling audio features for cached decoding."""
        input_features = kwargs.pop("input_features", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = self.language_model.prepare_inputs_for_generation(*args, **kwargs)

        # Only pass audio features on the first generation step (cache_position[0] == 0)
        if cache_position is not None and cache_position[0] == 0 and input_features is not None:
            model_inputs["input_features"] = input_features

        return model_inputs

    def _get_num_audio_tokens(
        self,
        audio_attention_mask: torch.Tensor,
    ) -> int:
        """Calculate number of audio tokens based on actual audio length.

        Uses attention mask to get real audio length, then computes:
        mel_frames -> encoder_frames (via conv formulas) -> projector output tokens
        """
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
        # Use max length for batch (all samples should have same token count for generation)
        encoder_output_len = int(encoder_lengths.max().item())
        return int(self.projector.get_output_length(encoder_output_len))

    def _prepare_audio_inputs(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        system_prompt: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Encode audio and splice it into the decoder's input embeddings.

        Builds the chat prompt when `input_ids` is not supplied. The number of
        `<audio>` placeholders must match the projector's output length exactly
        -- `masked_scatter` relies on it -- so the prompt and the embeddings are
        both derived from `audio_attention_mask` here instead of in each caller.

        Returns:
            Tuple of (input_ids, attention_mask, inputs_embeds). The mask is
            built here only when the prompt is; a caller that supplies
            `input_ids` gets its own `attention_mask` back untouched, including
            None.
        """
        device = input_features.device
        batch_size = input_features.shape[0]

        # Encode audio -> flattened embeddings (no per-sample host sync)
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
        token_counts = self.projector.get_output_length(encoder_lengths).to(torch.long)
        audio_embeds = self._encode_audio(input_features, token_counts, audio_attention_mask)

        # If input_ids not provided, build prompt with correct number of audio tokens
        if input_ids is None:
            num_audio_tokens = self._get_num_audio_tokens(audio_attention_mask)
            audio_placeholder = self.audio_token * num_audio_tokens

            system_prompt = system_prompt or self.system_prompt

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Audio tokens only (instruction-free)
            user_content = audio_placeholder
            if self.TRANSCRIBE_PROMPT:
                user_content += " " + self.TRANSCRIBE_PROMPT
            messages.append({"role": "user", "content": user_content})

            chat_result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,  # Disable Qwen3 thinking mode for ASR
            )
            input_ids = chat_result.input_ids.to(device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.shape[0] == 1 and batch_size > 1:
                input_ids = input_ids.expand(batch_size, -1)

            attention_mask = torch.ones_like(input_ids)

        # Get text embeddings and replace audio tokens with audio embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        audio_token_mask = (input_ids == self.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            audio_token_mask.to(inputs_embeds.device),
            audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        return input_ids, attention_mask, inputs_embeds

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        **generate_kwargs,
    ):
        """Generate transcription from audio input.

        Can be called in two ways:
        1. With input_ids containing <audio> tokens (from processor)
        2. With just audio, and we build the prompt internally
        """
        if input_features is None:
            raise ValueError("input_features required for generation")
        if audio_attention_mask is None:
            raise ValueError("audio_attention_mask required for generation")

        input_ids, attention_mask, inputs_embeds = self._prepare_audio_inputs(
            input_features,
            audio_attention_mask,
            system_prompt=system_prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # transformers v5 deprecates passing generation flags as kwargs when a
        # `generation_config` is also passed — the kwargs get silently dropped.
        # Pull any score-related flags out of generate_kwargs and apply them to
        # a derived generation_config so they actually take effect.
        gen_cfg = self.generation_config
        score_flags = {}
        for flag in ("output_scores", "output_logits", "return_dict_in_generate"):
            if flag in generate_kwargs:
                score_flags[flag] = generate_kwargs.pop(flag)
        if score_flags:
            from copy import copy as _copy

            gen_cfg = _copy(self.generation_config)
            for flag, value in score_flags.items():
                setattr(gen_cfg, flag, value)
            # output_scores requires return_dict_in_generate for HF generate to
            # actually populate .scores on the output object.
            if gen_cfg.output_scores and not gen_cfg.return_dict_in_generate:
                gen_cfg.return_dict_in_generate = True

        # Generate using language model.
        #
        # Default path passes both input_ids and inputs_embeds so
        # repetition_penalty works correctly (it needs input_ids to track which
        # tokens have been used). Gemma 4 forbids that combination outright --
        # "You must specify exactly one of input_ids or inputs_embeds" -- so
        # there we pass embeds plus the PLE precomputed from the clean prompt
        # ids. Gemma's prepare_inputs_for_generation drops per_layer_inputs
        # after the first step, so it is correct to hand it to generate().
        ple_kwargs = self._per_layer_kwargs(input_ids)
        lm_inputs = (
            {"inputs_embeds": inputs_embeds, **ple_kwargs}
            if ple_kwargs
            else {"input_ids": input_ids, "inputs_embeds": inputs_embeds}
        )
        output = self.language_model.generate(
            **lm_inputs,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            **generate_kwargs,
        )

        # With input_ids AND inputs_embeds, generate returns the full sequence
        # (prompt + generated), so the prompt must be stripped. With
        # inputs_embeds alone it already returns only the new tokens, so
        # stripping would eat real output -- hence prompt_len = 0 there.
        # When scores were requested, preserve the GenerateOutput so callers
        # can read .scores; otherwise return the bare tensor for backward
        # compatibility with existing callers.
        prompt_len = input_ids.shape[1] if "input_ids" in lm_inputs else 0
        if isinstance(output, torch.Tensor):
            return output[:, prompt_len:]
        output.sequences = output.sequences[:, prompt_len:]
        return output

    def generate_streaming(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        system_prompt: Optional[str] = None,
        **generate_kwargs,
    ) -> Iterator[str]:
        """Generate transcription with streaming token output.

        Yields partial transcript strings as tokens are generated.
        Reduces time-to-first-word by streaming tokens as they're decoded.

        Args:
            input_features: Mel spectrogram features (batch, n_mels, mel_len)
            audio_attention_mask: Mask for real vs padded mel frames (batch, mel_len)
            system_prompt: Optional system prompt override
            **generate_kwargs: Additional generation arguments

        Yields:
            Partial transcript text as each token is generated
        """
        input_ids, attention_mask, inputs_embeds = self._prepare_audio_inputs(
            input_features, audio_attention_mask, system_prompt=system_prompt
        )

        # Setup streamer for token-by-token output
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Prepare generation kwargs
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "generation_config": self.generation_config,
            "streamer": streamer,
            # Gemma 4 needs PLE precomputed from the prompt ids; {} elsewhere.
            **self._per_layer_kwargs(input_ids),
            **generate_kwargs,
        }

        # Run generation in background thread
        thread = Thread(target=self.language_model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they're generated, filtering out <think>...</think> blocks
        # Start assuming no think block - only filter when we see <think>
        in_think_block = False
        buffer = ""

        for text in streamer:
            buffer += text

            # Check for think block start (in case model outputs think blocks)
            while "<think>" in buffer:
                in_think_block = True
                # Yield any text before <think>
                before_think = buffer.split("<think>")[0]
                if before_think:
                    yield before_think
                buffer = buffer.split("<think>", 1)[-1]

            # Check for think block end
            while in_think_block and "</think>" in buffer:
                in_think_block = False
                buffer = buffer.split("</think>", 1)[-1]

            # Yield text if not in think block
            if not in_think_block and buffer:
                yield buffer
                buffer = ""

        # Yield any remaining buffer
        if buffer and not in_think_block:
            yield buffer

        thread.join()

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs) -> None:
        """Save model, tokenizer, and processor."""
        import shutil

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Update config with actual vocab size. Composite decoder configs
        # (Gemma 4 is multimodal, so Gemma4Config holds text/vision/audio
        # sub-configs) have no top-level `vocab_size` -- reading it raises
        # AttributeError, which used to surface only at the first checkpoint
        # save, i.e. after the run had already spent real training time.
        lm_config = self.language_model.config
        if hasattr(lm_config, "get_text_config"):
            lm_config = lm_config.get_text_config()
        vocab_size = getattr(lm_config, "vocab_size", None)
        if vocab_size is not None:
            self.config.vocab_size = vocab_size
            self.config.text_config.vocab_size = vocab_size

        if hasattr(self.audio_tower.config, "num_mel_bins"):
            self.config.audio_config.num_mel_bins = self.audio_tower.config.num_mel_bins

        # Save model (temporarily remove non-serializable attributes)
        tokenizer = self.tokenizer
        del self.tokenizer

        try:
            super().save_pretrained(save_dir, **kwargs)
        finally:
            self.tokenizer = tokenizer

        # Save tokenizer and feature extractor
        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)

        # Save LoRA adapters if present (creates adapter_model.safetensors and adapter_config.json)
        # Don't save embedding layers - the <audio> token embedding is never used
        # (it's replaced with projected audio embeddings before the LLM sees it)
        if hasattr(self.language_model, "peft_config"):
            self.language_model.save_pretrained(save_dir, save_embedding_layers=False)

            # Clear base_model_name_or_path in adapter_config.json to prevent HF pipeline
            # from redirecting to the base LLM repo (like Qwen) which breaks feature
            # extractor loading for multimodal models. If a repo_id is provided, use that
            # so the model can be loaded directly from the Hub.
            adapter_config_path = save_dir / "adapter_config.json"
            if adapter_config_path.exists():
                with adapter_config_path.open() as f:
                    adapter_config = json.load(f)

                # Use repo_id if available, otherwise clear to prevent redirect.
                # Use empty string instead of None to avoid str(None) -> "None" bug
                # in some transformers/PEFT versions.
                repo_id = (
                    kwargs.get("repo_id")
                    or kwargs.get("push_to_hub_model_id")
                    or getattr(self.config, "pretrained_model_path", None)
                    or ""  # Use empty string instead of None
                )
                adapter_config["base_model_name_or_path"] = repo_id

                with adapter_config_path.open("w") as f:
                    json.dump(adapter_config, f, indent=2)

        # Add processor auto_map to preprocessor_config.json
        config_path = save_dir / "preprocessor_config.json"
        if config_path.exists():
            with config_path.open() as f:
                processor_config = json.load(f)
        else:
            processor_config = {}

        processor_config.update(
            {
                "processor_class": "ASRProcessor",
                "auto_map": {"AutoProcessor": "asr_processing.ASRProcessor"},
            }
        )

        with config_path.open("w") as f:
            json.dump(processor_config, f, indent=2)

        # Copy source files for auto-loading
        src_dir = Path(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)
        # Copy projectors module
        shutil.copy(src_dir / "projectors.py", save_dir / "projectors.py")
        # Copy alignment module
        shutil.copy(src_dir / "alignment.py", save_dir / "alignment.py")
        # Copy diarization module
        shutil.copy(src_dir / "diarization.py", save_dir / "diarization.py")

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        """Push model to HuggingFace Hub, ensuring adapter_config points to repo.

        IMPORTANT: Sets base_model_name_or_path in adapter_config.json to repo_id
        so that transformers pipeline() can load the model correctly. Without this,
        the pipeline tries to load from "None" which fails.
        """
        # Store repo_id in config so save_pretrained can access it
        self.config.pretrained_model_path = repo_id
        # Call parent's push_to_hub
        return super().push_to_hub(repo_id, **kwargs)


# Register with transformers Auto classes
# (AutoConfig.register is handled in asr_config.py at module load.)
AutoModel.register(ASRConfig, ASRModel)
