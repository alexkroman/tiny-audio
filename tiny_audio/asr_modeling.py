import contextlib
import functools
import importlib.util
import inspect
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from copy import deepcopy
from os import PathLike
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Any, Self, cast, overload

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch.nn import functional
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
    TextIteratorStreamer,
    WhisperModel,
)
from transformers import __version__ as transformers_version
from transformers.generation.utils import GenerateOutput, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.hub import (
    cached_file,
)

if TYPE_CHECKING:
    from transformers import GraniteSpeech5Encoder as _GraniteSpeech5Encoder
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForCausalLM as _Gemma4ForCausalLM,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    from .asr_attention import resolve_attn_implementation, resolve_decoder_attn_implementation
    from .asr_config import ASRConfig, compute_encoder_output_length
    from .asr_layers import chunk_oversized_embeddings, unfreeze_encoder_top_layers
    from .asr_processing import (
        DEFAULT_TRANSCRIBE_PROMPT,
        ASRProcessor,
        left_pad_prompt_rows,
        render_audio_prompt,
    )
    from .asr_types import (
        DecoderLoadKwargs,
        GenerativeDecoder,
        HubFileKwargs,
        LoadKwargs,
        LoadStateDictResult,
        OutputLengthProjector,
        StateDictT,
    )
    from .projectors import PROJECTOR_CLASSES, MLPAudioProjector

    # None at runtime on a transformers without the architecture; see below.
    GraniteSpeech5Encoder: type[_GraniteSpeech5Encoder] | None
    Gemma4ForCausalLM: type[_Gemma4ForCausalLM] | None
else:
    try:
        from .asr_attention import resolve_attn_implementation, resolve_decoder_attn_implementation
        from .asr_config import ASRConfig, compute_encoder_output_length
        from .asr_layers import chunk_oversized_embeddings, unfreeze_encoder_top_layers
        from .asr_processing import (
            DEFAULT_TRANSCRIBE_PROMPT,
            ASRProcessor,
            left_pad_prompt_rows,
            render_audio_prompt,
        )
        from .asr_types import LoadStateDictResult, OutputLengthProjector, StateDictT
        from .projectors import PROJECTOR_CLASSES
    except ImportError:  # flat layout on the Hub: sibling modules, no package
        from asr_attention import resolve_attn_implementation, resolve_decoder_attn_implementation
        from asr_config import ASRConfig, compute_encoder_output_length
        from asr_layers import chunk_oversized_embeddings, unfreeze_encoder_top_layers
        from asr_processing import (
            DEFAULT_TRANSCRIBE_PROMPT,
            ASRProcessor,
            left_pad_prompt_rows,
            render_audio_prompt,
        )
        from asr_types import LoadStateDictResult, OutputLengthProjector, StateDictT
        from projectors import PROJECTOR_CLASSES

    # Architectures newer than the transformers floor in pyproject.toml. Kept
    # optional so a checkpoint that needs neither still loads on an older
    # install; the branch that needs one raises at its use site instead.
    try:
        from transformers import GraniteSpeech5Encoder
    except ImportError:  # transformers < 5.16: no native granite_speech5
        GraniteSpeech5Encoder = None
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    except ImportError:  # transformers without the gemma4 architecture
        Gemma4ForCausalLM = None


logger = logging.getLogger(__name__)

# Vocab dimension the embedding table is padded to after adding the audio
# token. 128 is what transformers' own `pad_to_multiple_of` docs recommend for
# tensor cores on sm_75+ and is what Qwen already pads its published tables to.
VOCAB_PAD_MULTIPLE = 128


def _resolve_dtype(name: object, fallback: torch.dtype) -> torch.dtype:
    """Resolve a config dtype name to a `torch.dtype`, falling back when unusable.

    Only a genuine `str` naming a real floating-point dtype is honoured. Stand-in
    config objects hand back a `MagicMock` for every attribute, and
    `getattr(torch, MagicMock())` raises `TypeError: attribute name must be
    string` -- which turned an unset optional field into a hard crash for any
    caller holding a mock or a checkpoint config predating the field. Same
    hazard `encoder_trainable_top_layers` guards against just below.
    """
    if not isinstance(name, str):
        return fallback
    resolved = getattr(torch, name, None)
    return (
        resolved if isinstance(resolved, torch.dtype) and resolved.is_floating_point else fallback
    )


def _gather_audio_embeds(
    audio_embeds: torch.Tensor,
    token_counts: torch.Tensor,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Flatten per-sample audio embeddings into a packed tensor.

    For each row i, takes the first ``token_counts[i]`` rows of
    ``audio_embeds[i]`` and concatenates them. If any token count exceeds
    ``audio_embeds.shape[1]``, the deficit is zero-padded.

    Equivalent to a per-sample slice/cat loop but with O(1) host-device
    syncs per call (one ``max().item()``) instead of one per sample. Callers
    that already have that maximum on the host pass it as ``max_tokens`` so
    the forward pays no sync at all.
    """
    _, max_len, _ = audio_embeds.shape
    needed = int(token_counts.max().item()) if max_tokens is None else max_tokens
    if needed > max_len:
        audio_embeds = functional.pad(audio_embeds, (0, 0, 0, needed - max_len))
        max_len = needed
    indices = torch.arange(max_len, device=audio_embeds.device).unsqueeze(0)
    mask = indices < token_counts.unsqueeze(1)
    return audio_embeds[mask]


def _assert_audio_token_counts(
    audio_embeds: torch.Tensor,
    token_counts: torch.Tensor,
    projector: OutputLengthProjector,
    encoder_valid_lengths: torch.Tensor | None = None,
    max_tokens: int | None = None,
) -> None:
    """Check, per sample, that the projector produced the tokens the prompt expects.

    This must run BEFORE ``_gather_audio_embeds``, which reconciles any
    disagreement by zero-padding a shortfall and truncating an excess. A check
    placed after it can never fire. That reconciliation is also why a mismatch
    is silent rather than loud today: the decoder receives zero vectors in
    place of audio, or the tail of a long utterance is dropped, and
    ``masked_scatter`` is satisfied either way.

    Per-sample rather than the batch total that ``get_placeholder_mask`` checks
    across transformers, because our failure mode is self-cancelling -- one row
    zero-padded by k and another truncated by k leaves the sum unchanged.

    ``encoder_valid_lengths`` comes from the encoder's own downsampled validity
    mask when it returns one (Granite does), making this a genuine cross-check
    against ``compute_encoder_output_length``, which reimplements the same
    arithmetic from ``config.encoder_conv_layers``.
    """
    if encoder_valid_lengths is not None:
        actual = projector.get_output_length(encoder_valid_lengths)
        actual = torch.as_tensor(actual, device=token_counts.device).to(torch.long).reshape(-1)
        if actual.shape == token_counts.shape and not torch.equal(actual, token_counts):
            rows = (actual != token_counts).nonzero().flatten()[:8].tolist()
            msg = (
                "Audio token count mismatch between prompt and projector. Rows "
                f"{rows}: prompt expects {token_counts[rows].tolist()}, encoder+projector "
                f"produced {actual[rows].tolist()}. A wrong `encoder_conv_layers` for this "
                "encoder is the usual cause."
            )
            raise ValueError(msg)

    available = audio_embeds.shape[1]
    if max_tokens is not None:
        needed = max_tokens
    else:
        needed = int(token_counts.max().item()) if token_counts.numel() else 0
    if needed > available:
        msg = (
            f"Projector produced {available} audio frames but the prompt expects up to "
            f"{needed}. Without this check `_gather_audio_embeds` would zero-pad the "
            "deficit, silently feeding the decoder zero vectors in place of audio."
        )
        raise ValueError(msg)


def _splice_audio(
    inputs_embeds: torch.Tensor, is_audio_token: torch.Tensor, audio_embeds: torch.Tensor
) -> torch.Tensor:
    """Scatter `audio_embeds` into `inputs_embeds` at the audio-placeholder positions."""
    dev = inputs_embeds.device
    return inputs_embeds.masked_scatter(
        is_audio_token.unsqueeze(-1).to(dev), audio_embeds.to(dev, dtype=inputs_embeds.dtype)
    )


class TensorNoRepeatNGram(LogitsProcessor):
    """`NoRepeatNGramLogitsProcessor`'s bans, computed as tensor ops on the ids' device.

    The stock processor rebuilds a Python dict of every n-gram in every row
    each step, with a `.tolist()` (a GPU sync) per row: 0.3-0.7 ms of host work
    a step at batch 8-32 against a 6.8 ms CUDA-graphed decode step. Same
    semantics: a token is banned when the last n-1 ids plus it would repeat an
    n-gram already in the row, prompt included.
    """

    def __init__(self, ngram_size: int) -> None:
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n = self.ngram_size
        if input_ids.shape[1] < n:
            return scores
        windows = input_ids.unfold(1, n, 1)  # (batch, positions, n)
        tail = input_ids[:, None, input_ids.shape[1] - n + 1 :]
        repeats = (windows[..., :-1] == tail).all(dim=-1)
        # Non-repeating windows point at a spare column, so every write is True.
        vocab = scores.shape[-1]
        banned = torch.zeros(scores.shape[0], vocab + 1, dtype=torch.bool, device=scores.device)
        banned.scatter_(1, torch.where(repeats, windows[..., -1], vocab), True)
        return cast("torch.FloatTensor", scores.masked_fill(banned[:, :vocab], -float("inf")))


def _patch_gemma_decode_loop(model: "GenerativeDecoder") -> None:
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
    def prepare_inputs_for_generation(
        *args: Any, is_first_iteration: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """Drop stale `per_layer_inputs` on every decode step after the first."""
        model_inputs = inner_prepare(*args, is_first_iteration=is_first_iteration, **kwargs)
        if not is_first_iteration:
            model_inputs.pop("per_layer_inputs", None)
        return model_inputs

    model.prepare_inputs_for_generation = prepare_inputs_for_generation


def _assert_projector_loaded(incompatible_keys: LoadStateDictResult, projector_type: str) -> None:
    """Fail loudly when a checkpoint's projector doesn't match the built one.

    `save_pretrained` serializes `projector.*` (plus the language model when
    it is trainable) and nothing else, so every `projector.*` key should
    round-trip exactly. Encoder keys -- and decoder keys on projector-only
    runs -- are legitimately absent because they are frozen and never saved,
    which is why the load itself runs with `strict=False`.

    That tolerance is dangerous for the projector specifically: it is the one
    component with no pretrained fallback, so a silent key mismatch leaves it
    at its random init and the model loads, generates, and scores like a model
    that was never trained. Checking only the `projector.` prefix keeps the
    frozen-module tolerance while making that failure impossible.
    """
    missing = sorted(k for k in incompatible_keys.missing_keys if k.startswith("projector."))
    unexpected = sorted(k for k in incompatible_keys.unexpected_keys if k.startswith("projector."))
    if not missing and not unexpected:
        return

    detail: list[str] = []
    if missing:
        detail.append(f"missing from the checkpoint: {missing}")
    if unexpected:
        detail.append(f"present in the checkpoint but not in the model: {unexpected}")

    hint = ""
    if unexpected == ["projector.output_scale"] and not missing:
        hint = (
            "\n\nThis checkpoint carries `projector.output_scale`, a fixed output-scale "
            "buffer that has since been removed. Its linears were trained behind that "
            "multiplier, so loading them without it changes the output magnitude ~13x; "
            "install a tiny-audio revision from before the removal to load this "
            "checkpoint as-is, or retrain the projector."
        )
    elif any(k.startswith("projector.norm_2.") for k in unexpected):
        hint = (
            "\n\nThis checkpoint predates the MLP projector layout change. It was "
            "saved as linear_1 -> norm -> act -> linear_2 -> norm_2; the current "
            "layout is input_norm -> linear_1 -> act -> linear_2. The weights are "
            "not interchangeable (the old linears were trained scale-invariant "
            "behind their norms), so the projector has to be retrained -- or "
            "install a tiny-audio revision from before the change to load this "
            "checkpoint as-is."
        )

    raise RuntimeError(
        f"Projector weights do not match the configured projector "
        f"(projector_type={projector_type!r}): " + "; ".join(detail) + hint
    )


def _log_linear_attention_backends(level: int = logging.WARNING) -> None:
    """Say which fast paths Qwen3.5-style hybrid layers will get on this box.

    Logged rather than enforced: a missing package means the torch reference
    path (same numerics, slower), and decoders without linear attention never
    touch these functions at all. `level` is INFO for an inference load
    (`from_pretrained`), where the slower path is not something to act on.
    """
    found = {
        pkg: importlib.util.find_spec(mod) is not None
        for pkg, mod in (("causal-conv1d", "causal_conv1d"), ("flash-linear-attention", "fla"))
    }
    missing = [pkg for pkg, ok in found.items() if not ok]
    if missing:
        logger.log(
            level,
            "Linear-attention fast path incomplete: %s not installed. Qwen3.5-style "
            "hybrid layers will use the torch reference path for those functions "
            "(same numerics, slower). `ta runpod deploy` installs both.",
            ", ".join(missing),
        )
    else:
        logger.info("Linear-attention fast path: causal-conv1d + flash-linear-attention packages.")


class ASRModel(PreTrainedModel, GenerationMixin):
    """Audio-to-text model combining an audio encoder, projector, and language model."""

    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False

    TRANSCRIBE_PROMPT = DEFAULT_TRANSCRIBE_PROMPT

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | None,
        *args: Any,
        **kwargs: Any,
    ) -> "ASRModel":
        """Load model from pretrained, handling device placement correctly."""
        if pretrained_model_name_or_path is None:
            msg = "ASRModel.from_pretrained needs a checkpoint directory or Hub repo id"
            raise ValueError(msg)

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
            cache_kwargs: HubFileKwargs = {}
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
                incompatible = model.load_state_dict(state_dict, strict=False)
                _assert_projector_loaded(incompatible, getattr(config, "projector_type", "?"))

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
                    # PEFT handles Hub downloads and caching internally.
                    #
                    # Adapter weights load onto the base model's device. Left to
                    # PEFT they go straight to "cuda" whenever CUDA reports
                    # available, which ZeroGPU Spaces do at startup without a
                    # GPU attached ("No CUDA GPUs are available").
                    base_device = next(model.language_model.parameters()).device
                    peft = PeftModel.from_pretrained(
                        model.language_model,
                        pretrained_model_name_or_path,
                        is_trainable=True,
                        torch_device=str(base_device),
                        **cache_kwargs,
                    )
                    # See `language_model` in __init__: PEFT forwards the
                    # decoder interface, so callers keep using it unchanged.
                    model.language_model = cast("GenerativeDecoder", peft)
                else:
                    # No saved adapters - initialize fresh LLM LoRA for training.
                    # __init__ skips _setup_lora while loading, so call it here.
                    model._setup_lora(config)

            return model
        finally:
            cls._is_loading_from_pretrained = False

    def __init__(self, config: ASRConfig, **kwargs: Any) -> None:
        """Build encoder, projector, decoder and tokenizer from `config`.

        `**kwargs` are the loader arguments `from_pretrained` forwards (e.g.
        `device_map`); they are intentionally ignored, see `from_pretrained`.
        """
        super().__init__(config)

        # Shadows the class attribute when the config names one, so a run that
        # trained under a specific instruction decodes under the same one.
        prompt = getattr(config, "transcribe_prompt", None)
        if prompt is not None:
            self.TRANSCRIBE_PROMPT = prompt  # pyright: ignore[reportConstantRedefinition]
        target_dtype = getattr(torch, config.model_dtype)

        # Audio encoder (frozen)
        self.audio_tower = self._load_audio_encoder(config, target_dtype)

        # Language model (frozen). Typed as the decoder even with LoRA attached:
        # a PeftModel forwards every attribute it does not define to the
        # decoder it wraps, so the decoder's interface is the right static view.
        self.language_model: GenerativeDecoder = self._load_language_model(config, target_dtype)

        # Does the decoder's forward take `skip_logits`? Only liger's patched
        # `lce_forward` declares it; the stock transformers forward does not.
        # Resolved once here rather than per-step, and read in forward() to
        # skip the lm_head projection on any labelled forward, train or eval
        # (see the comment there). Checked
        # on the class rather than the instance because liger patches the
        # class. scripts/train.py applies liger BEFORE constructing ASRModel,
        # so this sees the patched state.
        self._lm_accepts_skip_logits = (
            "skip_logits" in inspect.signature(type(self.language_model).forward).parameters
        )
        if not self._lm_accepts_skip_logits:
            # Say so once, loudly -- when building a model to train. A
            # `from_pretrained` load is almost always inference, where no
            # loss is computed and this is noise; it stays at INFO there.
            # Without the fused path every labelled
            # forward builds a (B, T, vocab) tensor plus its fp32 upcast and
            # its gradient, which on a large-vocab decoder is the difference
            # between fitting on the card and not -- and the ways to end up
            # here are all quiet: no liger patcher mapped for this
            # text_model_id, liger missing (it is linux-only), or a patcher
            # that patched a different class than the one that got loaded.
            logger.log(
                logging.INFO if type(self)._is_loading_from_pretrained else logging.WARNING,
                "%s's forward does not accept `skip_logits` — liger's fused "
                "linear cross-entropy is NOT active, so every training step "
                "materializes a (batch, seq, %s) logits tensor. Check for an "
                "'Applied liger kernels via ...' line earlier in the log.",
                type(self.language_model).__name__,
                getattr(self.language_model.config, "vocab_size", None) or "vocab",
            )

        # Initialize tokenizer and special tokens
        self._init_tokenizer(config)

        # Set up generation config with greedy decoding defaults
        self.generation_config = self.language_model.generation_config
        self.generation_config.max_new_tokens = config.max_new_tokens
        self.generation_config.use_cache = config.use_cache
        # `generate()` reads the GenerationConfig and nothing else, so every
        # decoding knob ASRConfig owns has to be copied across here. Omitting
        # this line left ASRConfig's 12-gram loop guard inert: the value was
        # set on the config, serialized into config.json, and never consulted.
        self.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.generation_config.eos_token_id = self._resolve_eos_token_ids()
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Feature extractor for audio preprocessing
        self.feature_extractor = self._create_feature_extractor(config)

        # Audio projector (trainable unless freeze_projector is set)
        self.projector = self._create_projector(config, target_dtype)

        # Setup LoRA if enabled (Stage 2 fine-tuning)
        # Skip if loading from pretrained - from_pretrained will handle adapter loading
        if getattr(config, "use_lora", False) and not type(self)._is_loading_from_pretrained:
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

        # For model parallelism
        self._no_split_modules = getattr(self.language_model, "_no_split_modules", [])

    def _create_feature_extractor(self, config: ASRConfig) -> SequenceFeatureExtractor:
        """Create the appropriate feature extractor for the audio encoder."""
        feature_extractor: SequenceFeatureExtractor = AutoFeatureExtractor.from_pretrained(
            config.audio_model_id
        )
        # Whisper's encoder requires a fixed 3000 mel frames (30s) and the
        # feature extractor pads to that by default — leave it alone. Other
        # encoders (e.g. GLM-ASR) accept variable-length input, so we disable
        # padding to avoid wasting compute on silent frames.
        if "whisper" not in config.audio_model_id.lower():
            feature_extractor.padding = False  # pyright: ignore[reportAttributeAccessIssue]
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
        encoder_kwargs: LoadKwargs = {
            "attn_implementation": resolve_attn_implementation(
                config.attn_implementation, config.audio_model_id
            ),
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        encoder: nn.Module
        if "whisper" in config.audio_model_id.lower():
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
            if GraniteSpeech5Encoder is None:
                msg = (
                    "Granite Speech 5.0 encoders need transformers >= 5.16 for the "
                    f"native granite_speech5 architecture; found {transformers_version}."
                )
                raise ImportError(msg)

            # Granite's block-attention implementation has no FA2 kernel and
            # raises ValueError on attn_implementation="flash_attention_2".
            # resolve_attn_implementation only downgrades FA2 when CUDA is
            # absent, so on a CUDA box the inherited default would hard-fail
            # at load. Pin sdpa regardless of what the config asks for.
            granite_kwargs: LoadKwargs = {**encoder_kwargs, "attn_implementation": "sdpa"}
            granite = GraniteSpeech5Encoder
            encoder = granite.from_pretrained(config.audio_model_id, **granite_kwargs)
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
            glm_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.audio_model_id, trust_remote_code=True, **encoder_kwargs
            )
            inner = getattr(glm_model, "model", None)
            holder: Any = (
                inner if inner is not None and hasattr(inner, "audio_tower") else glm_model
            )
            if not hasattr(holder, "audio_tower"):
                msg = (
                    f"{type(glm_model).__name__} exposes no audio_tower at "
                    "`.audio_tower` or `.model.audio_tower`; GLM-ASR encoder "
                    f"extraction needs updating for transformers "
                    f"{transformers_version}."
                )
                raise AttributeError(msg)
            encoder = holder.audio_tower
            # Drop the LLM decoder and projector to free their VRAM. These must
            # go through the same owner as audio_tower: assigning None to a name
            # that is not a registered submodule on that object silently creates
            # a plain attribute and frees nothing.
            holder.language_model = None
            holder.multi_modal_projector = None
            del glm_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        # Explicit cast: from_pretrained's `dtype=` kwarg is honored
        # inconsistently across loader paths (especially trust_remote_code
        # branches like GLM-ASR), leaving submodules in fp32. FA2's startup
        # then complains "current dype is torch.float32, expected fp16/bf16",
        # and even with sdpa the projector→encoder feed mismatches dtypes.
        # `.to(dtype=...)` after load is idempotent and forces the issue.
        # `encoder_dtype` overrides the stack dtype when the encoder has
        # trainable blocks; it must be fp32 for Adam's step to survive
        # rounding. See ASRConfig.encoder_dtype for the arithmetic.
        encoder_dtype = _resolve_dtype(getattr(config, "encoder_dtype", None), dtype)
        # PreTrainedModel.to is functools.wraps'd, which pyright cannot bind as a method.
        encoder = encoder.to(dtype=encoder_dtype)  # pyright: ignore[reportCallIssue]
        if getattr(config, "freeze_audio_encoder", True):
            encoder.requires_grad_(False)
            encoder.train(False)  # equivalent to .eval(); avoids a security hook false-positive
            # Require a genuine positive int. `int()` on a stand-in config
            # object is not safe -- a MagicMock coerces to 1, which would
            # silently unfreeze the encoder's top block in any caller holding
            # a mock or a checkpoint config predating this field.
            top_n = getattr(config, "encoder_trainable_top_layers", 0)
            top_n = top_n if isinstance(top_n, int) and not isinstance(top_n, bool) else 0
            if top_n > 0:
                unfreeze_encoder_top_layers(
                    encoder,
                    top_n,
                    include_post_projections=bool(
                        getattr(config, "encoder_trainable_post_projections", True)
                    ),
                )
        # Deliberately does NOT call `encoder.train(True)` for the partial
        # unfreeze. Construction must not decide train/eval mode: `train()` /
        # `eval()` own that, and `_apply_train_mode` already routes the encoder
        # correctly on every such call. Setting train mode here instead left it
        # sticky for any consumer that never calls either -- which is exactly
        # what LocalEvaluator did, and Granite's encoder carries 16
        # BatchNorm1d modules. In train mode those normalise with per-utterance
        # batch statistics rather than IBM's running statistics: near-harmless
        # on in-domain clean audio, and measured at 12.27% -> 37.44% WER on
        # Earnings22, where the audio is out of distribution and the clips are
        # short enough for batch statistics to be noisy.
        #
        # Note the encoder stays in EVAL mode during training too (the branch
        # in `_apply_train_mode` keyed on `freeze_audio_encoder`, which the
        # partial-unfreeze recipe leaves True). That is correct rather than
        # incidental: frozen BatchNorm statistics are standard practice when
        # fine-tuning a pretrained encoder, gradients still reach the conv
        # weights and BN's own affine parameters through an eval-mode BN, and
        # it keeps the running stats identical to base -- which matters because
        # a partial-unfreeze checkpoint saves only trainable PARAMETERS, not
        # buffers, so drifting running stats would silently not round-trip.
        return encoder

    @classmethod
    def _load_language_model(cls, config: ASRConfig, dtype: torch.dtype) -> "GenerativeDecoder":
        """Load and freeze the language model."""
        decoder_kwargs: DecoderLoadKwargs = {
            "attn_implementation": resolve_decoder_attn_implementation(
                config.attn_implementation, config.text_model_id
            ),
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }

        # Hub kernels are deliberately NOT requested (no `use_kernels=True`).
        # Qwen3.5's linear-attention functions resolve, per transformers'
        # @use_kernel_func_from_hub_with_fallback, in the order
        #   Hub kernel (only if use_kernels) -> installed package -> torch
        # and scripts/deploy/runpod.py installs and verifies the packages
        # (causal-conv1d compiled against the pod's torch, flash-linear-attention
        # + tilelang). Measured on a torch 2.8 A100 pod: with Hub kernels off,
        # causal_conv1d_fn resolves to causal_conv1d.causal_conv1d_interface and
        # chunk_gated_delta_rule to fla.ops.gated_delta_rule.chunk -- the fast
        # path, picked once at import.
        #
        # The Hub route bought nothing on top of that and broke every run from
        # f001a061 on: transformers pins kernels-community/mamba-ssm at a
        # version whose builds start at torch 2.11, the first `model.train()`
        # raised, and a catch-and-retry here disabled Hub dispatch anyway. A
        # re-pin could not hold either, because `PreTrainedModel.train()`
        # re-registers transformers' default mapping on every mode flip.
        if torch.cuda.is_available():
            _log_linear_attention_backends(
                logging.INFO if cls._is_loading_from_pretrained else logging.WARNING
            )

        decoder = cast(
            "GenerativeDecoder",
            AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs),
        )

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
        # `to` returns the module itself, so the cast happens in place.
        # PreTrainedModel.to is functools.wraps'd, which pyright cannot bind as a method.
        decoder.to(dtype=dtype)  # pyright: ignore[reportCallIssue]
        decoder.config.use_cache = getattr(config, "use_cache", True)
        if getattr(config, "freeze_language_model", True):
            decoder.requires_grad_(False)
            decoder.train(False)
        return decoder

    @staticmethod
    def _gemma_text_only(loaded: "GenerativeDecoder") -> "GenerativeDecoder":
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
        if Gemma4ForCausalLM is None:
            msg = f"Gemma 4 decoders need the gemma4 architecture; found {transformers_version}."
            raise ImportError(msg)

        # Only reached for gemma-4 checkpoints, whose text config is Gemma4TextConfig.
        text_config = cast("Gemma4TextConfig", loaded.config.get_text_config())
        with torch.device("meta"):
            shell = Gemma4ForCausalLM(text_config)

        shell.model = cast("Gemma4TextModel", loaded.get_submodule("model.language_model"))
        shell.lm_head = cast(nn.Linear, loaded.get_submodule("lm_head"))
        shell.generation_config = loaded.generation_config
        shell.tie_weights()

        # See _patch_gemma_decode_loop for why PLE is dropped after step 1.
        # Gemma4ForCausalLM is a PreTrainedModel with GenerationMixin, which is
        # all `GenerativeDecoder` describes.
        decoder = cast("GenerativeDecoder", shell)
        _patch_gemma_decode_loop(decoder)

        stranded = [n for n, t in shell.named_parameters() if t.device.type == "meta"]
        stranded += [n for n, t in shell.named_buffers() if t.device.type == "meta"]
        if stranded:
            msg = (
                "Gemma text-only conversion left parameters on the meta device "
                f"(would silently produce garbage): {stranded[:5]}"
            )
            raise RuntimeError(msg)
        return decoder

    def _create_projector(self, config: ASRConfig, dtype: torch.dtype) -> "MLPAudioProjector":
        """Create the trainable audio projector."""
        # Auto-detect dimensions if not specified
        if config.encoder_dim is None:
            enc_cfg = self.audio_tower.config
            config.encoder_dim = getattr(enc_cfg, "hidden_size", None) or getattr(
                enc_cfg, "d_model", None
            )
            if config.encoder_dim is None:
                msg = "Could not auto-detect encoder_dim. Please specify in config."
                raise ValueError(msg)

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
                msg = "Could not auto-detect llm_dim. Please specify in config."
                raise ValueError(msg)

        # Select projector type based on config
        projector_type = getattr(config, "projector_type", "mlp")
        projector_class = PROJECTOR_CLASSES.get(projector_type)
        if projector_class is None:
            msg = (
                f"Unknown projector_type: {projector_type}. "
                f"Valid options: {list(PROJECTOR_CLASSES.keys())}"
            )
            raise ValueError(msg)
        projector = projector_class(config)

        # Move projector to same device as language model (important when using quantization)
        device = next(self.language_model.parameters()).device
        # The projector may run at a higher precision than the frozen stack --
        # it is the only module with optimizer state, so it is the only one
        # that needs master-weight precision. See ASRConfig.projector_dtype.
        proj_dtype = _resolve_dtype(getattr(config, "projector_dtype", None), dtype)
        self._projector_dtype = proj_dtype
        return projector.to(device=device, dtype=proj_dtype)

    def _setup_lora(self, config: ASRConfig) -> None:
        """Apply LoRA adapters to the language model for Stage 2 fine-tuning."""
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            # Per-module overrides for targets whose shape makes the global
            # rank wrong -- see ASRConfig.lora_rank_pattern. PEFT resolves the
            # two patterns independently, so a recipe that sets one and not the
            # other silently changes that module's alpha/r scale; the config
            # field's docstring spells out the arithmetic.
            rank_pattern=dict(getattr(config, "lora_rank_pattern", None) or {}),
            alpha_pattern=dict(getattr(config, "lora_alpha_pattern", None) or {}),
            bias="none",
            task_type="CAUSAL_LM",
        )
        # See `language_model` in __init__ for why the PEFT wrapper is
        # typed as the decoder it wraps.
        self.language_model = cast(
            "GenerativeDecoder", get_peft_model(self.language_model, lora_config)
        )

    def _init_tokenizer(self, config: ASRConfig) -> None:
        """Initialize tokenizer with audio token."""
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.text_model_id, trust_remote_code=True
        )

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
            existing_special: list[str] = (
                getattr(self.tokenizer, "additional_special_tokens", None) or []
            )
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [*existing_special, self.audio_token]}
            )
            # mean_resizing=True initializes the new row at the mean of existing
            # rows so its scale matches the pretrained distribution. The
            # input-side embedding is overwritten via masked_scatter and never
            # seen by the LM, but with tied embeddings (Qwen3-0.6B) this same
            # row is the lm_head column for predicting the audio token; a
            # Gaussian draw at config.initializer_range was visible in
            # early-step logits.
            #
            # pad_to_multiple_of keeps the lm_head GEMM on a tensor-core-
            # friendly vocab dimension, which resizing to len(tokenizer) alone
            # destroys. Qwen ships a table already padded to a multiple of 128
            # and leaves the spare rows unaddressable -- 248,320 allocated
            # against 248,077 addressable on Qwen3.5-2B, 151,936 against
            # 151,669 on Qwen3-0.6B. Resizing to len(tokenizer)+1 shrinks past
            # that padding and lands on 248,078 / 151,670, neither of which is
            # even a multiple of 8. Confirmed in a shipped checkpoint:
            # tiny-audio-next-sat stores embed_tokens as (151670, 1024) against
            # the base model's (151936, 1024), so every Qwen run so far has done
            # the model's largest matmul on an unaligned vocab.
            #
            # The pad rows are mean-initialized like any new row and are tied
            # into lm_head, but the tokenizer cannot emit their ids and no label
            # contains them, so they only ever receive the softmax's negative
            # evidence. The base checkpoints already carry such rows.
            #
            # Note this changes the saved embedding shape, so a checkpoint
            # written before this (with freeze_language_model=False, which is
            # what puts language_model.* in the state dict) will not reload:
            # load_state_dict raises on a size mismatch even with strict=False.
            self.language_model.resize_token_embeddings(
                len(self.tokenizer),
                pad_to_multiple_of=VOCAB_PAD_MULTIPLE,
                mean_resizing=True,
            )

        audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        # The token is in the vocab by now, so a single name maps to one id.
        if not isinstance(audio_token_id, int):
            msg = f"Tokenizer has no id for the audio token {self.audio_token!r}"
            raise TypeError(msg)
        self.audio_token_id = audio_token_id
        self.tokenizer.padding_side = "right"

        # Sync token IDs to configs
        for cfg in [self.config.text_config, self.language_model.config, self.generation_config]:
            if cfg is not None:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def _resolve_eos_token_ids(self) -> list[int]:
        """Collect the ids generation stops on: turn closer, known end tokens, tokenizer EOS."""
        # Candidates that don't exist in the tokenizer are filtered out.
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
            # A single token name maps to a single id (None when unknown).
            if (
                not isinstance(token_id, int)
                or self.tokenizer.convert_ids_to_tokens(token_id) != token
            ):
                continue
            if token_id not in eos_ids:
                eos_ids.append(token_id)
        # The chat-template stop tokens above are additions, not replacements:
        # the tokenizer's own EOS is always a valid stop and is the only one
        # left for a decoder using none of those three templates.
        tokenizer_eos = self.tokenizer.eos_token_id
        if isinstance(tokenizer_eos, int) and tokenizer_eos not in eos_ids:
            eos_ids.append(tokenizer_eos)
        return eos_ids

    def _derive_turn_end_token_id(self) -> int | None:
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
        except Exception:
            return None
        if not isinstance(rendered, str) or sentinel not in rendered:
            return None
        tail_ids: list[int] = self.tokenizer(
            rendered.split(sentinel)[-1], add_special_tokens=False
        )["input_ids"]
        if not tail_ids:
            return None
        # Require a special token: a template that ends the turn with plain
        # text has no stop token to find, and adopting a text token as EOS
        # would truncate real transcripts.
        first = tail_ids[0]
        return first if first in set(self.tokenizer.all_special_ids) else None

    def _apply(self, *args: Any, **kwargs: Any) -> Self:
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
        module: Self = super()._apply(*args, **kwargs)
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

    def train(self, mode: bool = True) -> Self:
        """Set train/eval mode, but keep frozen submodules out of train mode.

        HF Trainer calls `model.train()` at the top of every training step, which
        recursively switches every submodule into train mode — re-enabling dropout
        on modules with `requires_grad_(False)`. The frozen encoder (and the LM
        when `freeze_language_model=True`) should always run deterministically;
        train-mode dropout only adds noise that can't improve a frozen network.
        """
        self._apply_train_mode(mode)
        return self

    def _apply_train_mode(self, mode: bool) -> None:
        """The actual mode switch; named so configs can point at the BN/dropout policy."""
        super().train(mode)
        if getattr(self.config, "freeze_audio_encoder", True):
            self.audio_tower.train(False)
        elif mode:
            # Fully-unfrozen encoder: keep BatchNorm on its pretrained running
            # statistics anyway. Granite Speech 5.0 carries 16 BatchNorm1d and
            # zero Dropout, and for non-Whisper extractors the collator pads to
            # the batch longest (`DataCollator._audio_padding`) with
            # group_by_length off, so clips spanning 0.8-19.0s leave a large
            # pad fraction. Granite's conv module zeroes pad positions and THEN
            # runs BatchNorm1d over the full padded (B, C, T); BN reduces over
            # B*T, so those zeros are counted as data -- deflating the mean and
            # shrinking the variance. At BN's default momentum=0.1 the running
            # stats converge to the polluted values within ~30-50 steps, and
            # every eval and checkpoint after that uses them.
            #
            # This channel is invisible to `encoder_learning_rate`: running
            # stats are buffers, so they carry no gradient and sit in no
            # optimizer group. See the measured 12.27% -> 37.44% Earnings22
            # regression documented in `_load_audio_encoder` for what wrong BN
            # statistics cost on this exact encoder.
            #
            # Pinning the statistics does not freeze the module: gradients
            # still reach the conv weights and BN's own affine weight/bias
            # through an eval-mode BN, and those params remain trainable at
            # `encoder_learning_rate` (routed to the no-decay group, since
            # create_optimizer matches any module whose class name contains
            # "Norm"). This mirrors what the partial-unfreeze path already gets
            # for free via the `freeze_audio_encoder: true` branch above.
            for module in self.audio_tower.modules():
                if isinstance(
                    module,
                    (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
                ):
                    module.eval()
        if getattr(self.config, "freeze_language_model", True):
            self.language_model.train(False)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., Any] | None = None,
        every_n_layers: int = 1,
        **kwargs: Any,
    ) -> None:
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
        forwardable: dict[str, Any] = dict(kwargs)
        forwardable["every_n_layers"] = every_n_layers
        if gradient_checkpointing_func is not None:
            forwardable["gradient_checkpointing_func"] = gradient_checkpointing_func

        # The LLM still stores activations during forward for backprop to projector.
        # Gradient checkpointing trades compute for memory by recomputing activations.
        for submodule in self._gradient_checkpointing_targets():
            setter = getattr(submodule, "_set_gradient_checkpointing", None)
            enable_fn = getattr(submodule, "gradient_checkpointing_enable", None)
            disable_fn = getattr(submodule, "gradient_checkpointing_disable", None)
            if setter is not None:
                accepted = inspect.signature(setter).parameters
                passthrough = {k: v for k, v in forwardable.items() if k in accepted}
                setter(enable=enable, **passthrough)
            elif enable_fn is not None and enable:
                enable_fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            elif disable_fn is not None and not enable:
                disable_fn()

    def _gradient_checkpointing_targets(self) -> list[nn.Module]:
        """Return the submodules that should respond to gradient_checkpointing
        toggles. Always includes the LM (activations are on the gradient path
        to the projector); includes the encoder only when it's trainable.
        """
        targets: list[nn.Module] = [self.language_model]
        if self._encoder_has_trainable_params():
            targets.append(self.audio_tower)
        return targets

    def _encoder_has_trainable_params(self) -> bool:
        """True if any encoder parameter requires grad.

        Keyed on autograd state, NOT on `freeze_audio_encoder`. The documented
        partial-unfreeze recipe is `freeze_audio_encoder: true` PLUS
        `encoder_trainable_top_layers: N` (see ASRConfig), which leaves the flag
        True while `_load_audio_encoder` switches the top N blocks back on.
        Reading the flag instead once wrapped those blocks in no_grad (so the
        run paid ~1.6 GiB of AdamW state for 198M params that could never
        move) and kept them out of the checkpoint.
        """
        return any(p.requires_grad for p in self.audio_tower.parameters())

    def get_input_embeddings(self) -> nn.Module:
        """Return the decoder's token embedding module."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the decoder's token embedding module."""
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module | None:
        """Return the decoder's LM head (None for a decoder without one)."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Replace the decoder's LM head."""
        self.language_model.set_output_embeddings(new_embeddings)

    def get_processor(self) -> ASRProcessor:
        """Get the processor for this model."""
        return ASRProcessor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            projector=self.projector,
            encoder_conv_layers=self.config.encoder_conv_layers,
            audio_token=self.audio_token,
        )

    @overload
    def state_dict(
        self, *, destination: StateDictT, prefix: str = ..., keep_vars: bool = ...
    ) -> StateDictT: ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> dict[str, Any]: ...

    def state_dict(
        self,
        *,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
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
        sd = {
            f"projector.{k}": v for k, v in self.projector.state_dict(keep_vars=keep_vars).items()
        }
        # Keyed on autograd state, not `freeze_audio_encoder`. Under the
        # documented partial-unfreeze recipe that flag stays True while the top
        # N blocks train, so gating on it meant a trained encoder was never
        # written to the checkpoint -- the run did the work and threw it away.
        # Only the trainable tensors are saved: the rest reload from
        # `audio_model_id`, and from_pretrained overlays with strict=False.
        enc_trainable = {n for n, p in self.audio_tower.named_parameters() if p.requires_grad}
        if enc_trainable:
            fully = len(enc_trainable) == sum(1 for _ in self.audio_tower.parameters())
            sd.update(
                {
                    f"audio_tower.{k}": v
                    for k, v in self.audio_tower.state_dict(keep_vars=keep_vars).items()
                    if fully or k in enc_trainable
                }
            )
        if not getattr(self.config, "freeze_language_model", True):
            lm = self.language_model
            if hasattr(lm, "peft_config"):
                for name, v in lm.state_dict(keep_vars=keep_vars).items():
                    if "lora_" in name:
                        continue
                    k = name.removeprefix("base_model.model.")
                    # LoRA layers wrap the original Linear as `<name>.base_layer.<weight|bias>`.
                    k = k.replace(".base_layer.", ".")
                    sd[f"language_model.{k}"] = v
            else:
                sd.update(
                    {
                        f"language_model.{k}": v
                        for k, v in lm.state_dict(keep_vars=keep_vars).items()
                    }
                )
        if destination is None:
            destination = {}
        destination.update({f"{prefix}{k}": v for k, v in sd.items()})
        return destination

    def _ple_text_model(self) -> "Gemma4TextModel | None":
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
                # Checked just above; only Gemma 4 text models define it.
                return cast("Gemma4TextModel", candidate)
        return None

    def _per_layer_kwargs(self, input_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
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

    def _embed_tokens(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Look `input_ids` up in the decoder's token embedding table."""
        embeds: torch.Tensor = self.language_model.get_input_embeddings()(input_ids)
        return embeds

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
        audio_attention_mask: torch.Tensor | None = None,
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

        grad_ctx = (
            contextlib.nullcontext() if self._encoder_has_trainable_params() else torch.no_grad()
        )
        with grad_ctx:
            encoder_out = self.audio_tower(**tower_kwargs)
        hidden_states = encoder_out.last_hidden_state

        # Conformer encoders (Granite) return their own validity mask, halved
        # alongside each subsampling block. That is the encoder's arithmetic
        # rather than our reimplementation of it, so it is the right ground
        # truth for the token-count check below.
        encoder_mask = getattr(encoder_out, "attention_mask", None)
        encoder_valid_lengths = (
            encoder_mask.sum(dim=-1).to(torch.long) if encoder_mask is not None else None
        )

        # The encoder may be bf16 while the projector holds fp32 master
        # weights, so align explicitly rather than relying on autocast being
        # active -- eval and MPS paths run outside it.
        proj_dtype = getattr(self, "_projector_dtype", None)
        if proj_dtype is not None and hidden_states.dtype != proj_dtype:
            hidden_states = hidden_states.to(proj_dtype)
        audio_embeds = self.projector(hidden_states)

        token_counts = expected_token_counts.to(device=audio_embeds.device, dtype=torch.long)
        max_tokens = int(token_counts.max().item()) if token_counts.numel() else 0
        _assert_audio_token_counts(
            audio_embeds, token_counts, self.projector, encoder_valid_lengths, max_tokens
        )
        return _gather_audio_embeds(audio_embeds, token_counts, max_tokens)

    def _mask_input_features(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SpecAugment on mel input (pure-torch, vectorized, compile-ready).

        Follows the same semantics as
        `transformers.models.whisper.modeling_whisper.WhisperModel._mask_input_features`
        (wav2vec2-style mask sampling: sample N start positions per sample,
        mask `mask_length` frames forward from each), but reimplemented in
        pure torch so it stays inside the autograd graph without crossing
        the numpy boundary. This avoids inductor codegen failures
        (e.g. the `'zuf0' was not declared` error from the prior numpy ->
        torch.tensor round-trip) AND avoids the per-forward host-to-GPU
        sync that the numpy path required.

        One minor semantic divergence vs the upstream helper: this version
        allows mask spans to overlap, while upstream rejects overlapping
        samples. For ASR purposes this is irrelevant — occasional region
        double-coverage has no measurable effect on the regularization
        signal.

        Reads ASRConfig fields by Whisper naming convention: mask_time_prob,
        mask_time_length, mask_time_min_masks.

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
            Same-shape tensor with time-axis masks zeroed.
        """
        input_features = input_features.clone()
        config = self.config
        # Whisper/GLM-ASR hand us (batch, n_mels, mel_len); Conformer encoders
        # (Granite, Parakeet, Nemotron) hand us (batch, time, feature_dim).
        # Masking the wrong axis is silent -- it still runs and still zeroes
        # cells -- so the axis order is read from config rather than guessed.
        time_major = getattr(config, "audio_features_time_major", False)
        if time_major:
            batch_size, sequence_length, _ = input_features.size()
            feature_dim = 2
        else:
            batch_size, _, sequence_length = input_features.size()
            feature_dim = 1
        device = input_features.device

        valid_lengths = None
        if attention_mask is not None:
            if attention_mask.shape[-1] != sequence_length:
                msg = (
                    "SpecAugment attention_mask time axis "
                    f"({attention_mask.shape[-1]}) does not match input_features "
                    f"({sequence_length}). Masking against the wrong axis is silent, "
                    "so this is raised rather than ignored."
                )
                raise ValueError(msg)
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

        return input_features

    @staticmethod
    def _sample_mask_indices(
        batch_size: int,
        axis_length: int,
        mask_prob: float,
        mask_length: int,
        min_masks: int,
        device: torch.device,
        valid_lengths: torch.Tensor | None = None,
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

        # Sample start positions independently per sample x span, inside that
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
        input_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        audio_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass for training and inference."""
        if inputs_embeds is None:
            inputs_embeds = self._embed_tokens(input_ids)

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

            inputs_embeds = _splice_audio(inputs_embeds, is_audio_token, audio_embeds)

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

        # Skip the lm_head projection whenever the loss is all that is wanted.
        # Nothing in this repo reads `outputs.logits` on a labelled forward --
        # not the Trainer (no compute_metrics), not the custom metrics -- so
        # the (B, T, V) tensor is pure cost on both train and eval.
        #
        # Requesting it EXPLICITLY on both, rather than leaning on liger's
        # default, because that default is wrong here in both directions:
        #
        #   skip_logits = self.training and labels is not None
        #
        # where `self` is the LANGUAGE MODEL. Two ways that misfires:
        #
        #   eval  -- runs under model.eval(), so the default is False and the
        #            full tensor gets built. granite_qwen.yaml's claim that
        #            "the logits never materialize past liger's fused CE" was
        #            not true as configured: Trainer injects skip_logits only
        #            when `args.use_liger_kernel` is set (trainer.py:3135),
        #            and this repo patches liger by hand and leaves that flag
        #            False. At B=32, T~320, V=248320 that is ~5.1 GiB in bf16
        #            plus the fp32 upcast inside the loss.
        #
        #   train -- `train()` above forces the decoder into eval mode when
        #            `freeze_language_model=True`, so the LM's `training` flag
        #            is False WHILE THE RUN IS TRAINING and the default turns
        #            the fused CE off exactly where it matters most. That is
        #            an OOM, not a slowdown: granite_qwen_lora at B=48, T=557
        #            died on a 12.37 GiB allocation in backward, which is
        #            48*557*248320*2 bytes to the byte. The full-FT recipe
        #            never saw it -- its decoder stayed in train mode, so the
        #            default happened to be right.
        #
        # Done here rather than via `use_liger_kernel: true` because that flag
        # makes transformers call apply_liger_kernel(), which *raises* when
        # liger-kernel is unavailable -- and liger is a linux-only dependency
        # (pyproject.toml), so it would break every mps_smoke run on a mac.
        # Gating on the patched signature is equivalent where it matters and
        # inert everywhere else.
        if labels is not None and self._lm_accepts_skip_logits:
            kwargs.setdefault("skip_logits", True)

        outputs: CausalLMOutputWithPast = self.language_model(
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
        return outputs

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Prepare inputs for generation, handling audio features for cached decoding."""
        input_features = kwargs.pop("input_features", None)
        cache_position = kwargs.get("cache_position")

        model_inputs = self.language_model.prepare_inputs_for_generation(*args, **kwargs)

        # Only pass audio features on the first generation step (cache_position[0] == 0)
        if cache_position is not None and cache_position[0] == 0 and input_features is not None:
            model_inputs["input_features"] = input_features

        return model_inputs

    def _render_audio_prompt(self, num_audio_tokens: int) -> torch.Tensor:
        """Tokenize one chat prompt carrying exactly `num_audio_tokens` placeholders."""
        return render_audio_prompt(
            self.tokenizer, self.audio_token, num_audio_tokens, self.TRANSCRIBE_PROMPT
        )

    def _left_pad_prompt_rows(
        self,
        rows: list[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack per-sample prompt rows into a left-padded batch on `device`.

        See `left_pad_prompt_rows`; pad positions never collide with
        `audio_token_id`, so the masked_scatter below is unaffected.
        """
        input_ids, attention_mask = left_pad_prompt_rows(rows, self.tokenizer)
        return input_ids.to(device), attention_mask.to(device)

    def _prepare_audio_inputs(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
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

        # Encode audio -> flattened embeddings (no per-sample host sync)
        encoder_lengths = self._compute_encoder_output_lengths(audio_attention_mask)
        token_counts = self.projector.get_output_length(encoder_lengths).to(torch.long)
        audio_embeds = self._encode_audio(input_features, token_counts, audio_attention_mask)

        # If input_ids not provided, build one prompt PER SAMPLE. A single
        # prompt sized to the batch max and expanded across rows -- which this
        # used to do -- gives every shorter row more `<audio>` placeholders
        # than the projector produced for it, and `_gather_audio_embeds` then
        # zero-fills the difference. Silent, and it scales with how ragged the
        # batch is, so batch-1 eval never sees it.
        if input_ids is None:
            rows = [self._render_audio_prompt(int(n)) for n in token_counts.tolist()]
            input_ids, attention_mask = self._left_pad_prompt_rows(rows, device)

        # Get text embeddings and replace audio tokens with audio embeddings
        inputs_embeds = _splice_audio(
            self._embed_tokens(input_ids), input_ids == self.audio_token_id, audio_embeds
        )
        return input_ids, attention_mask, inputs_embeds

    def _assert_sdpa_safe_on_mps(self, attention_mask: torch.Tensor | None) -> None:
        """Refuse a left-padded batch when Metal's sdpa would silently NaN it.

        A ragged batch left-pads the shorter prompts (see
        `_left_pad_prompt_rows`), and on MPS the sdpa kernel turns any row with
        a fully-masked position into NaN logits. argmax then falls through to
        token 0, so the row decodes as "!!!!!!" -- at full speed, no warning,
        and for 3 of 4 rows in a mixed-length LibriSpeech batch. Silent wrong
        transcripts are the worst outcome here, so this raises instead; a WER
        computed from them would look merely bad, not broken.

        Batch 1 and uniform-length batches pad nothing and pass. `eager` is
        correct on MPS at any padding, and the bug is Metal-specific, so a CUDA
        box is unaffected either way. `resolve_attn_implementation` documents
        the measurements.
        """
        if attention_mask is None or attention_mask.device.type != "mps":
            return
        if (
            self.language_model.config._attn_implementation  # pyright: ignore[reportPrivateUsage]  # transformers has no public accessor
            != "sdpa"
        ):
            return
        if not bool((attention_mask == 0).any()):
            return
        msg = (
            "Refusing to generate: this is a left-padded batch of "
            f"{attention_mask.shape[0]} on MPS with sdpa attention, where "
            "Metal's kernel returns NaN for the padded rows and every one of "
            "them would decode as '!!!!!!' with no error. Either load with "
            "attn_implementation='eager' (correct at any padding, ~30% slower "
            "to decode), batch only equal-length clips, or generate one clip "
            "at a time."
        )
        raise ValueError(msg)

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate(
        self,
        *args: Any,
        input_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> GenerateOutput | torch.LongTensor:
        """Generate transcription from audio input.

        Can be called in two ways:
        1. With input_ids containing <audio> tokens (from processor)
        2. With just audio, and we build the prompt internally

        The audio arguments are keyword-only; `*args` exists only so the
        signature stays compatible with `GenerationMixin.generate`, whose
        positional parameters mean something else entirely.
        """
        if args:
            msg = (
                "ASRModel.generate takes input_ids, input_features, "
                "audio_attention_mask and attention_mask as keyword arguments"
            )
            raise TypeError(msg)
        if input_features is None:
            msg = "input_features required for generation"
            raise ValueError(msg)
        if audio_attention_mask is None:
            msg = "audio_attention_mask required for generation"
            raise ValueError(msg)

        input_ids, attention_mask, inputs_embeds = self._prepare_audio_inputs(
            input_features,
            audio_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self._assert_sdpa_safe_on_mps(attention_mask)

        # Fold generation flags (max_new_tokens, num_beams, output_scores, ...)
        # into a private copy of generation_config instead of passing them
        # alongside it: transformers v5 still applies such kwargs but warns that
        # combining them with `generation_config` is deprecated. update() returns
        # the kwargs that aren't config fields (streamer, logits_processor, ...),
        # which still go to generate() directly.
        gen_cfg = deepcopy(self.generation_config)
        generate_kwargs = gen_cfg.update(**generate_kwargs)
        # output_scores requires return_dict_in_generate for HF generate to
        # actually populate .scores on the output object.
        if gen_cfg.output_scores:
            gen_cfg.return_dict_in_generate = True
        if gen_cfg.no_repeat_ngram_size:
            processors = LogitsProcessorList(generate_kwargs.pop("logits_processor", None) or [])
            processors.append(TensorNoRepeatNGram(gen_cfg.no_repeat_ngram_size))
            generate_kwargs["logits_processor"] = processors
            # 0, not None: generate refills None from the decoder's own config
            # and would build the stock processor alongside this one.
            gen_cfg.no_repeat_ngram_size = 0

        lm_inputs = self._lm_generate_inputs(input_ids, inputs_embeds)
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
            # Slicing a token-id tensor keeps its int64 dtype.
            return cast("torch.LongTensor", output[:, prompt_len:])
        output.sequences = cast("torch.LongTensor", output.sequences[:, prompt_len:])
        return output

    def _lm_generate_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The prompt arguments for `language_model.generate`.

        Default path passes both input_ids and inputs_embeds so
        repetition_penalty and no_repeat_ngram_size see the prompt (they need
        input_ids to track which tokens have been used). Gemma 4 forbids that
        combination outright -- "You must specify exactly one of input_ids or
        inputs_embeds" -- so there we pass embeds plus the PLE precomputed from
        the clean prompt ids. Gemma's prepare_inputs_for_generation drops
        per_layer_inputs after the first step, so it is correct to hand it to
        generate().
        """
        ple_kwargs = self._per_layer_kwargs(input_ids)
        if ple_kwargs:
            return {"inputs_embeds": inputs_embeds, **ple_kwargs}
        return {"input_ids": input_ids, "inputs_embeds": inputs_embeds}

    def generate_streaming(
        self,
        input_features: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        **generate_kwargs: Any,
    ) -> Iterator[str]:
        """Generate transcription with streaming token output.

        Yields partial transcript strings as tokens are generated.
        Reduces time-to-first-word by streaming tokens as they're decoded.

        Args:
            input_features: Mel spectrogram features (batch, n_mels, mel_len)
            audio_attention_mask: Mask for real vs padded mel frames (batch, mel_len)
            **generate_kwargs: Additional generation arguments

        Yields:
            Partial transcript text as each token is generated
        """
        input_ids, attention_mask, inputs_embeds = self._prepare_audio_inputs(
            input_features, audio_attention_mask
        )

        # Setup streamer for token-by-token output
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Prepare generation kwargs
        # Same prompt arguments as generate(); with input_ids present the
        # streamer's skip_prompt drops the echoed prompt tokens.
        # Generation flags go into a copy of generation_config, as in generate().
        gen_cfg = deepcopy(self.generation_config)
        generate_kwargs = gen_cfg.update(**generate_kwargs)
        gen_kwargs: dict[str, Any] = {
            **self._lm_generate_inputs(input_ids, inputs_embeds),
            "attention_mask": attention_mask,
            "generation_config": gen_cfg,
            "streamer": streamer,
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

    def save_pretrained(
        self, save_directory: str | PathLike[str], *args: Any, **kwargs: Any
    ) -> None:
        """Save model, tokenizer, and processor.

        Extra positional arguments are forwarded to `PreTrainedModel.save_pretrained`.
        """
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

        encoder_config = self.audio_tower.config
        if hasattr(encoder_config, "num_mel_bins"):
            # The default is never used: hasattr just confirmed the attribute.
            # Read through getattr because nn.Module types `.config` as
            # Tensor | Module, which hasattr does not narrow.
            self.config.audio_config.num_mel_bins = getattr(encoder_config, "num_mel_bins", None)

        # Save model (temporarily remove non-serializable attributes)
        tokenizer = self.tokenizer
        del self.tokenizer

        try:
            super().save_pretrained(
                save_dir,
                *args,
                **kwargs,
            )
        finally:
            self.tokenizer = tokenizer

        # Save tokenizer and feature extractor
        self.tokenizer.save_pretrained(save_dir)
        self.feature_extractor.save_pretrained(save_dir)

        # Save LoRA adapters if present (creates adapter_model.safetensors and adapter_config.json)
        # Don't save embedding layers - the <audio> token embedding is never used
        # (it's replaced with projected audio embeddings before the LLM sees it)
        if hasattr(self.language_model, "peft_config"):
            # Override base_model_name_or_path in the saved adapter_config.json to
            # prevent HF pipeline from redirecting to the base LLM repo (like Qwen),
            # which breaks feature extractor loading for multimodal models. If a
            # repo_id is provided, use that so the model can be loaded directly
            # from the Hub. Use empty string instead of None: PeftModel.save_pretrained
            # fills the field with the base LLM's name only when it is None, and an
            # empty string also avoids the str(None) -> "None" bug in some
            # transformers/PEFT versions.
            repo_id = (
                kwargs.get("repo_id")
                or kwargs.get("push_to_hub_model_id")
                or getattr(self.config, "pretrained_model_path", None)
                or ""
            )
            peft_configs = list(self.language_model.peft_config.values())
            # Set on the live configs only for the save, then restore, so the
            # in-memory model keeps its real base model name.
            original_names = [cfg.base_model_name_or_path for cfg in peft_configs]
            try:
                for cfg in peft_configs:
                    cfg.base_model_name_or_path = repo_id
                self.language_model.save_pretrained(save_dir, save_embedding_layers=False)
            finally:
                for cfg, name in zip(peft_configs, original_names, strict=True):
                    cfg.base_model_name_or_path = name

        # Add processor auto_map to preprocessor_config.json
        config_path = save_dir / "preprocessor_config.json"
        processor_config: dict[str, Any]
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
        siblings = ("projectors.py", "alignment.py", "diarization.py")
        for src in [*src_dir.glob("asr_*.py"), *(src_dir / name for name in siblings)]:
            shutil.copy(src, save_dir / src.name)

    def create_or_update_model_card(self, output_dir: str | Path) -> None:
        """Re-apply the PEFT card metadata to `output_dir/README.md`.

        This is a PEFT method, not a transformers one, and it exists here only
        because `Trainer` calls it on the unwrapped model. `create_model_card`
        overwrites the README with its own training summary, and if the card it
        replaced declared `library_name: peft` it then asks the model to put the
        adapter metadata back. On a LoRA run that branch always fires: the
        `language_model.save_pretrained` call above writes the adapter through
        `PeftModel.save_pretrained`, which stamps a peft card into the same
        directory the trainer is about to overwrite. `PreTrainedModel` has no
        such method, so without this delegation the final `trainer.save_model()`
        dies with AttributeError *after* the run has finished training.

        Delegating to the language model is a no-op when there is no adapter
        (a stale peft README left in `output_dir` by an earlier LoRA run in the
        same directory is enough to reach here on a run with LoRA disabled).
        """
        card_fn = getattr(self.language_model, "create_or_update_model_card", None)
        if card_fn is not None:
            card_fn(str(output_dir))

    # typeshed's functools._Wrapped (how PreTrainedModel declares push_to_hub) is
    # not a descriptor, so no method can match it; the signature is the
    # PushToHubMixin one that PreTrainedModel wraps.
    def push_to_hub(  # pyright: ignore[reportIncompatibleVariableOverride]
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        commit_description: str | None = None,
        private: bool | None = None,
        token: bool | str | None = None,
        revision: str | None = None,
        create_pr: bool = False,
        max_shard_size: int | str | None = "50GB",
        tags: list[str] | None = None,
    ) -> str:
        """Push model to HuggingFace Hub, ensuring adapter_config points to repo.

        IMPORTANT: Sets base_model_name_or_path in adapter_config.json to repo_id
        so that transformers pipeline() can load the model correctly. Without this,
        the pipeline tries to load from "None" which fails.
        """
        # Store repo_id in config so save_pretrained can access it
        self.config.pretrained_model_path = repo_id
        # Call parent's push_to_hub
        return PreTrainedModel.push_to_hub(
            self,
            repo_id,
            commit_message=commit_message,
            commit_description=commit_description,
            private=private,
            token=token,
            revision=revision,
            create_pr=create_pr,
            max_shard_size=max_shard_size,
            tags=tags,
        )


# Register with transformers Auto classes
# (AutoConfig.register is handled in asr_config.py at module load.)
AutoModel.register(ASRConfig, ASRModel)
