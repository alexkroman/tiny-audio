"""Estimate the GPU memory and disk a training config will actually need.

Provisioning is the one step where guessing is expensive: too small a GPU and
the run OOMs an hour into dataset prep, too small a container disk and the
weights download dies at 95%. This reads the resolved Hydra config and reports
both, then emits a ready-to-run `runpodctl pod create`.

Everything that can be measured is measured rather than assumed:
  - parameter counts come from safetensors headers over HTTP range requests
    (no weights downloaded),
  - download sizes come from the Hub's file metadata,
  - the projector is instantiated from the real PROJECTOR_CLASSES entry so the
    number can't drift from the implementation.

Activation memory is the one genuine estimate; its formula is printed so the
number can be argued with rather than trusted blindly.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from tenacity import RetryError, retry, retry_if_result, stop_after_delay, wait_fixed

from scripts.deploy import gpu_catalog
from scripts.deploy.hub_sizes import (
    NON_LM_TOWER_PREFIXES,
    hidden_dim,
    lora_trainable_params,
    repo_weight_bytes,
    safetensors_params,
    vocab_table_params,
)
from scripts.deploy.pods import POD_PORTS, create_first_available, ssh_public_key
from scripts.train_config import register_configs
from scripts.utils import get_project_root

if TYPE_CHECKING:
    from transformers import AutoConfig

    from tiny_audio.asr_config import ASRConfig, ConvLayerSpec
    from tiny_audio.projectors import MLPAudioProjector

GIB = 1024**3
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
# AdamW keeps exp_avg + exp_avg_sq per trainable parameter.
OPTIMIZER_STATES = 2
# Headroom for fragmentation, cuBLAS workspaces, NCCL buffers and the CUDA
# context. 1.25 is deliberately modest; raise it if runs OOM near the estimate.
# NOTE: this used to be doing double duty, silently absorbing part of the
# activation undercount described at ACTIVATION_CALIBRATION below. Now that the
# activation term is calibrated against a real run, this is pure safety margin
# again.
OVERHEAD_FACTOR = 1.25
VRAM_RECOMMENDED_KEY = f"recommended (x{OVERHEAD_FACTOR})"
# The analytical per-layer activation formula further down is a FLOOR, not an
# estimate, and this reconciles it with reality.
#
# Measured, from the completed granite_qwen run on an H100 80GB (recorded in
# configs/experiments/granite_qwen.yaml): per_device=32 sat at ~41.4 of the
# 79.6 GiB nvidia-smi reports, against ~5.6 GiB of static for that recipe, so
#     (41.4 - 5.6) / 32          = 1.12 GiB of activation tape per sample
#     1.12 GiB / (320 tok * 24 layers) = 156,587 bytes / token / layer
# The formula below yields 2 * (6*2048 + 3*6144) = 61,440 for the same model.
#     156,587 / 61,440 = 2.55
#
# Where the missing 2.55x comes from, in rough order of size:
#   - Qwen3.5 is HYBRID. 18 of its 24 layers are `linear_attention` (gated
#     delta rule + causal depthwise conv), whose retained state is nothing
#     like the `6*hidden + 3*inter` a standard transformer block implies.
#     The formula has no notion of layer_types at all.
#   - SwiGLU keeps the gate*up elementwise product, a fourth `inter` tensor.
#   - Two RMSNorms per layer each save their input.
#   - LoRA runs add an r-sized activation per targeted linear (186 of them at
#     r=64 for granite_qwen_lora) -- real but only ~0.6% of the gap.
#
# Derived from ONE measurement on ONE recipe, so treat it as calibration, not
# theory. It is conservative for a dense decoder (granite_gemma), which has no
# gated-delta-net state -- over-provisioning there is the cheap direction.
# Re-derive if a run's observed peak diverges: the inputs are all in the
# wandb system metrics plus the static breakdown this script prints.
ACTIVATION_CALIBRATION = 2.55
# DataCollator._MAX_AUDIO_SECONDS -- every batch pads to its own longest row
# and multiasr puts enough mass near the ceiling that the worst case is the
# one to plan for.
MAX_AUDIO_SECONDS = 19.0
# Frames per second entering the encoder's block stack, before
# `encoder_conv_layers` subsampling. Granite's feature extractor emits 100 Hz
# log-mels and stacks adjacent pairs (its input dim is 160 = 80 mels x 2), so
# the stack runs at 50 Hz. Whisper-family encoders are also 50 Hz after their
# stride-2 conv front end, so this holds for both encoders this repo uses.
ENCODER_FRAME_RATE_HZ = 50.0

# Rough size of the installed python env + apt packages on the pod.
ENV_DISK_GIB = 12.0
# datasets keeps two persistent copies of every source, and neither is cleaned
# up during the run: the Hub download lands as parquet in the hub cache under
# HF_HOME, and load_dataset() then writes its own Arrow tables under
# data.dataset_cache_dir. Measured on hf-internal-testing/librispeech_asr_dummy
# (datasets 4.x): 8.99 MB of parquet in the hub cache plus 9.46 MB of Arrow in
# the cache_dir, i.e. 2.05x the download size. Charging the download once is
# how a correctly-sized pod still dies with ENOSPC halfway through prep.
DATASET_DISK_FACTOR = 2.05

# Past this, emit the network-volume provisioning flow instead of a plain
# container disk. RunPod container disks are carved from host-local storage,
# so a ~1 TB+ request is refused on every GPU type and reads as "no capacity"
# rather than as a disk problem.
NETWORK_VOLUME_THRESHOLD_GB = 1024
# `runpodctl network-volume create --size` accepts 1-4000 (GB).
NETWORK_VOLUME_MAX_GB = 4000
# With a volume mounted at /workspace, HF_HOME and HF_DATASETS_CACHE both
# resolve onto it, so the container disk only holds the image and python env.
CONTAINER_DISK_WITH_VOLUME_GB = 50


@dataclass
class Component:
    name: str
    params: int = 0
    download_bytes: int = 0
    trainable: bool = False
    note: str = ""


@dataclass
class Plan:
    components: list[Component] = field(default_factory=list[Component])
    dataset_bytes: int = 0
    dataset_rows: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    warnings: list[str] = field(default_factory=list[str])
    vram: dict[str, float] = field(default_factory=dict[str, float])
    disk: dict[str, float] = field(default_factory=dict[str, float])

    @property
    def trainable_params(self) -> int:
        return sum(c.params for c in self.components if c.trainable)

    @property
    def need_vram_gib(self) -> float:  # overhead-padded VRAM a GPU must have
        return self.vram[VRAM_RECOMMENDED_KEY]

    @property
    def disk_gb(self) -> int:  # disk to request: the estimate + 15% + 5 GB of slack
        return int(self.disk["recommended"] * 1.15) + 5


def _load_cfg(experiment: str, overrides: list[str]) -> DictConfig:
    # Registers the `base_config` structured-config schema the experiment
    # configs compose against (idempotent; train_config also does it on import).
    register_configs()
    configs = get_project_root() / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        return compose(config_name="config", overrides=[f"+experiments={experiment}", *overrides])


def _get(node: DictConfig, key: str, default: Any) -> Any:
    """`node.get(key, default)` that also treats an explicit None as unset.

    The structured-config schema (scripts/train_config.py) declares optional
    keys with a None default, so a plain `.get` returns None instead of the
    fallback for any key the composed config left unset.
    """
    val = node.get(key)
    return default if val is None else val


@dataclass
class _Precision:
    """Bytes per element for each kind of tensor the VRAM estimate charges."""

    weights: int  # frozen master weights (model_dtype)
    trainable: int  # trainable weights, grads and AdamW state (projector_dtype)
    activations: int  # the autocast dtype the activation tape is stored in


def _encoder_trainable_blocks(train: DictConfig, depth: int) -> int:
    """How many encoder blocks receive gradients (all of them, the top N, or none)."""
    if not bool(_get(train, "freeze_audio_encoder", True)):
        return depth
    top_n = int(_get(train, "encoder_trainable_top_layers", 0) or 0)
    return min(top_n, depth)


def _add_encoder(plan: Plan, train: DictConfig, audio_id: str, enc_inner: Any) -> int:
    """Append the encoder's frozen/trainable split; return its trainable block count."""
    enc_params, enc_dtype = safetensors_params(audio_id)
    enc_depth = int(getattr(enc_inner, "num_hidden_layers", 0) or 0)
    enc_frozen_flag = bool(_get(train, "freeze_audio_encoder", True))
    trainable_blocks = _encoder_trainable_blocks(train, enc_depth)

    # A partial unfreeze is trainable too. `freeze_audio_encoder: true` PLUS
    # `encoder_trainable_top_layers: N` is the documented idiom, so reading the
    # flag alone reported 109.76M of trainable encoder as frozen and dropped
    # its gradients and AdamW state (~1.3 GiB) from the estimate entirely.
    # Split it the way the decoder's frozen vocabulary table is split.
    if not enc_frozen_flag:
        enc_trainable_params = enc_params
    elif trainable_blocks and enc_depth:
        # Pro-rata by block. Approximate -- it charges the pre/post-stack
        # projections at the same rate as a block, so for Granite's top-4 it
        # says 118.2M against an actual 109.76M. Errs high, which is the safe
        # direction for sizing a card.
        enc_trainable_params = int(enc_params * trainable_blocks / enc_depth)
    else:
        enc_trainable_params = 0

    plan.components.append(
        Component(
            f"encoder ({audio_id})",
            enc_params - enc_trainable_params,
            repo_weight_bytes(audio_id),
            False,
            f"checkpoint {enc_dtype}",
        )
    )
    if enc_trainable_params:
        enc_top_n = int(_get(train, "encoder_trainable_top_layers", 0) or 0)
        label = "all blocks" if not enc_frozen_flag else f"top {enc_top_n} of {enc_depth} blocks"
        plan.components.append(
            Component(f"  encoder trainable ({label})", enc_trainable_params, 0, True, "")
        )
    return trainable_blocks


def _add_lora(plan: Plan, model: DictConfig, text_id: str) -> None:
    """Append the LoRA adapters' trainable params when `use_lora` is on."""
    # Without this the plan reports a frozen decoder as contributing zero
    # trainable params, which is wrong whenever `use_lora` is on -- it
    # understated the granite_qwen_lora recipe by 67.28M and made AdamW state
    # look 6x smaller than it is. The adapters are freshly initialised, so
    # they add optimizer state but nothing to download.
    if not _get(model, "use_lora", False):
        return
    lora_params = lora_trainable_params(
        text_id,
        int(_get(model, "lora_rank", 8)),
        model.get("lora_target_modules") or ["q_proj", "v_proj"],
    )
    if not lora_params:
        return
    targets = model.get("lora_target_modules")
    label = targets if isinstance(targets, str) else "custom targets"
    plan.components.append(
        Component(
            f"  LoRA adapters (r={_get(model, 'lora_rank', 8)}, {label})",
            lora_params,
            0,  # fresh init, nothing to fetch
            True,
            "trainable; base decoder frozen",
        )
    )


def _add_decoder(plan: Plan, cfg: DictConfig, text_id: str, text_cfg: Any) -> bool:
    """Append the decoder, its LoRA adapters and frozen tables; return whether it trains."""
    train = cfg.training
    dec_params, dec_dtype = safetensors_params(text_id, NON_LM_TOWER_PREFIXES)
    dec_trainable = not _get(train, "freeze_language_model", True)
    # Split the frozen vocabulary table out of the trainable decoder. The
    # freeze flag acts on an individual tensor inside the language model, so a
    # single all-or-nothing `trainable` on one component would charge AdamW
    # state for parameters that never see the optimizer on this recipe.
    frozen_tables: dict[str, int] = {}
    if dec_trainable:
        tables = vocab_table_params(text_cfg)
        if _get(train, "freeze_text_embed_tokens", False) and tables.get("embed_tokens"):
            frozen_tables["embed_tokens"] = tables["embed_tokens"]
    frozen_table_params = sum(frozen_tables.values())

    plan.components.append(
        Component(
            f"decoder ({text_id})",
            dec_params - frozen_table_params,
            repo_weight_bytes(text_id),
            dec_trainable,
            f"checkpoint {dec_dtype}",
        )
    )
    _add_lora(plan, cfg.model, text_id)
    if frozen_table_params:
        plan.components.append(
            Component(
                f"  frozen tables ({', '.join(frozen_tables)})",
                frozen_table_params,
                0,  # already charged by the decoder's repo download above
                False,
                "no optimizer state",
            )
        )
    return dec_trainable


def _add_projector(
    plan: Plan,
    model: DictConfig,
    encoder_dim: int | None,
    llm_dim: int | None,
    audio_id: str,
    text_id: str,
) -> None:
    """Append the projector, sized by instantiating the real PROJECTOR_CLASSES entry."""
    # Lazy, like the transformers import in build_plan.
    asr_config_cls: type[ASRConfig] = importlib.import_module("tiny_audio.asr_config").ASRConfig
    projector_classes: dict[str, type[MLPAudioProjector]] = importlib.import_module(
        "tiny_audio.projectors"
    ).PROJECTOR_CLASSES
    proj_params = 0
    if encoder_dim and llm_dim:
        shim = asr_config_cls(
            audio_model_id=audio_id,
            text_model_id=text_id,
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
            projector_type=str(_get(model, "projector_type", "mlp")),
            projector_pool_stride=int(_get(model, "projector_pool_stride", 4)),
            projector_hidden_dim=model.get("projector_hidden_dim"),
        )
        cls = projector_classes[shim.projector_type]
        # Meta device: shapes without storage or init, all a numel count needs.
        with importlib.import_module("torch").device("meta"):
            proj_params = sum(p.numel() for p in cls(shim).parameters())
    else:
        plan.warnings.append("Could not resolve encoder/llm dims; projector excluded.")
    plan.components.append(
        Component("projector (fresh init)", proj_params, 0, True, f"{encoder_dim}->{llm_dim}")
    )


def _add_datasets(plan: Plan, data: DictConfig) -> None:
    """Record the Hub download size of every dataset the run trains or evals on."""
    entries: Iterable[DictConfig] = data.get("datasets", []) or []
    for entry in entries:
        path = str(entry.get("path"))
        name = entry.get("name")
        if not entry.get("train_splits") and not entry.get("eval_splits"):
            continue
        try:
            size = repo_weight_bytes(path, "dataset", str(name) if name else None)
        except Exception as exc:  # gated repos, renames, network
            plan.warnings.append(f"dataset {path}: size unknown ({type(exc).__name__})")
            continue
        plan.dataset_bytes += size
        plan.dataset_rows.append((f"{path}" + (f":{name}" if name else ""), size))


# Per token per layer: attention q/k/v/o + residual (~6*dim) and the MLP's
# gate/up/down (~3*inter). Coarse but the right order. Scaled by
# ACTIVATION_CALIBRATION -- see its definition; unscaled this term is 2.55x
# under what the granite_qwen run actually used, which is enough to
# recommend a 48 GB card for a job that needs ~42 GiB.
def _tape_bytes(
    batch: int, seq: int, dim: int, inter: int, layers: int, act_bytes_per: int, ckpt: bool
) -> float:
    """Activation tape retained across `layers` transformer-shaped blocks."""
    per_tok_layer = act_bytes_per * (6 * dim + 3 * inter) * ACTIVATION_CALIBRATION
    if ckpt:
        # Only layer boundaries are kept; one layer is recomputed at a time.
        # The boundary term is a plain hidden-sized tensor per layer and is
        # NOT subject to the calibration, which corrects the within-layer
        # tape; only the single recomputed layer carries that.
        return batch * seq * dim * layers * act_bytes_per + batch * seq * per_tok_layer
    return batch * seq * layers * per_tok_layer


def _decoder_activation_bytes(
    train: DictConfig, text_cfg: Any, llm_dim: int, seq_len: int, act_bytes_per: int
) -> float:
    """Activation tape retained across the decoder stack for one micro-batch."""
    batch = int(_get(train, "per_device_train_batch_size", 1))
    layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    inter = int(getattr(text_cfg, "intermediate_size", 0) or 0)
    ckpt = bool(_get(train, "gradient_checkpointing", False))
    return _tape_bytes(batch, seq_len, llm_dim, inter, layers, act_bytes_per, ckpt)


def _encoder_activation_bytes(
    cfg: DictConfig, trainable_blocks: int, encoder_dim: int, act_bytes_per: int
) -> float:
    """Activation tape retained across the encoder's trainable blocks (0 when frozen)."""
    # Encoder activation tape. Zero while the encoder is frozen -- ASRModel.forward
    # runs it under no_grad, so nothing is retained -- and a large term the moment
    # it is not. Missing this is what made granite_qwen_lora plan at 67.94 GiB and
    # then OOM at 79.16 on an 80 GB card: only the static side (+6 GiB) was
    # counted, and the tape is bigger than the static.
    #
    # Granite's 16 conformer blocks run at the post-subsampling rate, so the
    # sequence length here is the ENCODER's, not the decoder's: 19s of audio is
    # ~950 frames at the stacked-mel rate, and encoder_conv_layers halves twice
    # to ~237. Per token per block the tape is dominated by the two half-step
    # feed-forwards (2 x 2 x ff_expansion x d), attention q/k/v/o (4d) plus its
    # score matrix, and the conv module -- the same shape as the decoder term,
    # so it reuses the same empirical calibration.
    # Only the trainable blocks count (see _encoder_trainable_blocks). Autograd
    # retains activations only from the first trainable parameter onward, and
    # the projector sits AFTER the encoder, so a top-N unfreeze retains the top
    # N blocks and nothing below them. Gating on `freeze_audio_encoder` alone
    # reported zero for the partial case; measured peaks say otherwise --
    # frozen 55.4 GiB, top-4 60.6, full ~79.4 at batch 48, and 4/16 of the
    # full increment is 6.0 GiB against an observed 5.2.
    if not (trainable_blocks and encoder_dim):
        return 0.0
    train = cfg.training
    batch = int(_get(train, "per_device_train_batch_size", 1))
    # The encoder's sequence is its own, NOT the decoder's `seq_len`: it is
    # acoustic frames, and there are far more of them than there are text
    # tokens. DataCollator caps audio at 19s and Granite's extractor stacks
    # mel pairs to 50 Hz, so ~950 frames enter the block stack and
    # encoder_conv_layers halves twice to ~237. Using seq_len here instead
    # gave 82 and undercounted the tape ~3x.
    enc_seq = int(MAX_AUDIO_SECONDS * ENCODER_FRAME_RATE_HZ)
    # Same default as ASRConfig: unset means DEFAULT_ENCODER_CONV_LAYERS,
    # not "no subsampling".
    # Lazy, like the transformers/tiny_audio imports in build_plan.
    asr_config = importlib.import_module("tiny_audio.asr_config")
    encoder_output_length: Callable[[int, Sequence[ConvLayerSpec] | None], int] = (
        asr_config.compute_encoder_output_length
    )
    enc_seq = encoder_output_length(enc_seq, cfg.model.get("encoder_conv_layers") or None)
    ckpt = bool(_get(train, "gradient_checkpointing", False))
    return _tape_bytes(
        batch, enc_seq, encoder_dim, 4 * encoder_dim, trainable_blocks, act_bytes_per, ckpt
    )


def _precision(cfg: DictConfig) -> _Precision:
    """Resolve the byte widths of frozen weights, trainable state and activations."""
    train = cfg.training
    model_dtype = str(_get(cfg.model, "model_dtype", "bfloat16"))
    bytes_per = DTYPE_BYTES.get(model_dtype, 2)
    # Trainable params may be held at a different (higher) precision than the
    # frozen stack; see ASRConfig.projector_dtype. Charging everything at the
    # trainable dtype overstated weights by 10.4 GiB on this recipe.
    proj_dtype = str(cfg.model.get("projector_dtype") or model_dtype)
    # ACTIVATIONS ARE AUTOCAST'S DTYPE, NOT THE MASTER WEIGHTS'. Under
    # `bf16: true` every matmul emits bf16, so the tape is 2 B/element even
    # when model_dtype is float32. Charging it at `bytes_per` doubled the term
    # on every fp32-master recipe, and it also silently contradicted
    # ACTIVATION_CALIBRATION, which was fitted against a formula written as
    # `2 * (6*hidden + 3*inter)` -- a hardcoded 2 -- on granite_qwen, itself a
    # model_dtype: float32 run. Fit and use disagreed by exactly 2x, which is
    # why this predicted 101.87 GiB for a run that completed on an 80 GB card
    # and recommended an H200 for granite_qwen_full at ~76% of an H100.
    act_bytes_per = 2 if (train.get("bf16") or train.get("fp16")) else bytes_per
    return _Precision(bytes_per, DTYPE_BYTES.get(proj_dtype, bytes_per), act_bytes_per)


def _cross_entropy_bytes(train: DictConfig, text_cfg: Any, seq_len: int) -> int:
    """Logits memory for the loss; zero when liger fuses lm_head+CE."""
    # Cross-entropy. liger fuses lm_head+softmax+CE into O(B*T*D); without it
    # the (B, T, V) fp32 logits plus a log_softmax copy dominate everything.
    #
    # What makes `use_liger` a sufficient condition is ASRModel.forward
    # requesting `skip_logits=True` on every labelled forward. Leaning on
    # liger's own default instead is what made this term wrong once already:
    # that default reads the DECODER's `training` flag, which is False
    # whenever the decoder is frozen, so granite_qwen_lora planned at 60.10
    # GiB and then OOMed on a 12.37 GiB logits gradient. The term is still
    # optimistic if liger has no patcher for the architecture -- train.py
    # warns in that case, and so does ASRModel.__init__.
    if bool(_get(train, "use_liger", True)):
        return 0
    batch = int(_get(train, "per_device_train_batch_size", 1))
    vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
    return batch * seq_len * vocab * 4 * 2


def _estimate_vram(plan: Plan, prec: _Precision, acts: float, enc_acts: float, logits: int) -> None:
    """Fill `plan.vram` with the static + activation breakdown and the overhead-padded total."""
    total_params = sum(c.params for c in plan.components)
    trainable_params = plan.trainable_params
    frozen_params = total_params - trainable_params
    weights = frozen_params * prec.weights + trainable_params * prec.trainable
    grads = trainable_params * prec.trainable
    optim = trainable_params * prec.trainable * OPTIMIZER_STATES

    # Autocast's bf16 copy of every fp32 weight. torch.autocast caches its
    # casts for the lifetime of the autocast region, and that region ends when
    # the FORWARD ends -- exactly when the activation tape is at its maximum,
    # so this stacks onto the peak rather than the trough. Zero when the master
    # weights are already 2 bytes (nothing to cast) or when autocast is off.
    # Worth 4.41 GiB on granite_qwen_full, and it was simply missing here.
    autocast_cache = total_params * 2 if (prec.activations == 2 and prec.weights != 2) else 0

    subtotal = weights + grads + optim + acts + logits + autocast_cache
    plan.vram = {
        "weights": weights / GIB,
        "gradients": grads / GIB,
        "optimizer (AdamW x2)": optim / GIB,
        "activations (est.)": acts / GIB,
        "  of which encoder": enc_acts / GIB,
        "autocast bf16 weight cache": autocast_cache / GIB,
        "cross-entropy": logits / GIB,
        "subtotal": subtotal / GIB,
        VRAM_RECOMMENDED_KEY: subtotal * OVERHEAD_FACTOR / GIB,
    }


def _add_vram_warnings(
    plan: Plan, train: DictConfig, dec_trainable: bool, logits: int, seq_len: int, text_cfg: Any
) -> None:
    """Warn about known over-counts and about the levers that would shrink the estimate."""
    # Gradients must reach the projector, which sits at the *input* of the
    # decoder, so every decoder layer's activations are retained even though
    # the decoder itself is frozen. Freezing saves optimizer state, not
    # activation memory -- a common and expensive surprise.
    # Two known over-counts, both harmless while the decoder was frozen (the
    # projector was the only trainable module) and both material once it is not.
    # Stated rather than corrected: erring high is the safe direction for pod
    # sizing, but a silently inflated figure invites renting the wrong GPU.
    if dec_trainable:
        plan.warnings.append(
            "Trainable-parameter count is an upper bound: it charges trainable "
            "weights and gradients at projector_dtype even though the decoder "
            "trains at model_dtype. Expect real usage at or below the figure "
            "above. (Multimodal towers the loader discards -- visual./mtp./"
            "audio_tower. -- are no longer counted; see NON_LM_TOWER_PREFIXES.)"
        )

    ckpt = bool(_get(train, "gradient_checkpointing", False))
    if not dec_trainable and plan.trainable_params and not ckpt:
        plan.warnings.append(
            "Decoder is frozen but the projector feeds its input, so activations "
            "are still held for every decoder layer. gradient_checkpointing=true "
            "is the lever if activations dominate."
        )
    if not bool(_get(train, "use_liger", True)):
        batch = int(_get(train, "per_device_train_batch_size", 1))
        vocab = int(getattr(text_cfg, "vocab_size", 0) or 0)
        plan.warnings.append(
            f"use_liger is off: unfused CE over vocab={vocab:,} adds "
            f"{logits / GIB:.1f} GiB at batch={batch}, seq={seq_len}."
        )


def _estimate_disk(plan: Plan, train: DictConfig, trainable_bytes_per: int) -> None:
    """Fill `plan.disk` with weights, datasets, env and retained checkpoints."""
    # A checkpoint is model.safetensors + optimizer.pt, and both scale with the
    # whole *trainable* stack rather than the projector: ASRModel.state_dict
    # serializes the language model too whenever freeze_language_model is false
    # (stage_1), and HF Trainer always writes AdamW's two states per trainable
    # param. save_total_limit copies sit on disk simultaneously, so retention
    # multiplies -- charging one projector-sized checkpoint understated a joint
    # fine-tune by ~500x.
    keep = max(int(_get(train, "save_total_limit", 1) or 1), 1)
    ckpt_each = plan.trainable_params * trainable_bytes_per * (1 + OPTIMIZER_STATES)
    ckpt_bytes = ckpt_each * keep

    weights_bytes = sum(c.download_bytes for c in plan.components)
    datasets_bytes = plan.dataset_bytes * DATASET_DISK_FACTOR
    total = weights_bytes + datasets_bytes + ckpt_bytes
    plan.disk = {
        "model weights": weights_bytes / GIB,
        f"datasets (parquet+arrow x{DATASET_DISK_FACTOR})": datasets_bytes / GIB,
        "python env + apt": ENV_DISK_GIB,
        f"checkpoints ({keep} kept x {_fmt(ckpt_each / GIB)})": ckpt_bytes / GIB,
        "recommended": total / GIB + ENV_DISK_GIB,
    }

    # RunPod container disks are provisioned from the host's local storage and
    # large requests are a common cause of "no instances available"; past ~1 TiB
    # a network volume is the realistic way to satisfy /workspace.
    if plan.disk["recommended"] > 1024:
        plan.warnings.append(
            f"{plan.disk['recommended'] / 1024:.1f} TiB of /workspace is a lot to ask of a "
            "container disk. Attach a network volume (--network-volume-id) or cut the "
            "dataset mix; `datasets` needs room for parquet AND arrow at once."
        )


def build_plan(experiment: str, overrides: list[str], seq_len: int) -> Plan:
    """Resolve a training config and measure its parameter, VRAM and disk footprint."""
    # Imported lazily: transformers + tiny_audio cost several seconds, which
    # every `ta runpod` command would otherwise pay because runpod.py imports
    # this module.
    auto_config: type[AutoConfig] = importlib.import_module("transformers").AutoConfig

    cfg = _load_cfg(experiment, overrides)
    plan = Plan()
    train = cfg.training

    audio_id = str(cfg.model.audio_model_id)
    text_id = str(cfg.model.text_model_id)
    enc_cfg = auto_config.from_pretrained(audio_id)
    enc_inner = getattr(enc_cfg, "encoder_config", None) or enc_cfg
    dec_cfg = auto_config.from_pretrained(text_id)
    text_cfg = dec_cfg.get_text_config() if hasattr(dec_cfg, "get_text_config") else dec_cfg
    encoder_dim = hidden_dim(enc_inner)
    llm_dim = hidden_dim(text_cfg)

    trainable_blocks = _add_encoder(plan, train, audio_id, enc_inner)
    dec_trainable = _add_decoder(plan, cfg, text_id, text_cfg)
    _add_projector(plan, cfg.model, encoder_dim, llm_dim, audio_id, text_id)
    _add_datasets(plan, cfg.data)

    prec = _precision(cfg)
    enc_acts = _encoder_activation_bytes(cfg, trainable_blocks, encoder_dim or 0, prec.activations)
    acts = _decoder_activation_bytes(train, text_cfg, llm_dim or 0, seq_len, prec.activations)
    acts += enc_acts
    logits = _cross_entropy_bytes(train, text_cfg, seq_len)
    _estimate_vram(plan, prec, acts, enc_acts, logits)
    _add_vram_warnings(plan, train, dec_trainable, logits, seq_len, text_cfg)
    _estimate_disk(plan, train, prec.trainable)
    return plan


def _fmt(n: float) -> str:
    return f"{n:,.2f} GiB"


def _print_plan_json(plan: Plan, experiment: str) -> None:
    """Print the plan as machine-readable JSON (`--json`)."""
    print(
        json.dumps(
            {
                "experiment": experiment,
                "vram_gib": plan.vram,
                "disk_gib": plan.disk,
                "components": [
                    {
                        "name": c.name,
                        "params": c.params,
                        "download_gib": c.download_bytes / GIB,
                        "trainable": c.trainable,
                    }
                    for c in plan.components
                ],
                "warnings": plan.warnings,
            },
            indent=2,
        )
    )


def _print_plan_tables(plan: Plan, experiment: str, seq_len: int) -> None:
    """Print the component, dataset, GPU-memory and disk tables plus any warnings."""
    print(f"\n=== Resource plan: +experiments={experiment} (seq_len={seq_len}) ===\n")
    print(f"{'component':<52} {'params':>14} {'download':>11}  train")
    for c in plan.components:
        print(
            f"{c.name[:52]:<52} {c.params:>14,} "
            f"{c.download_bytes / GIB:>10.2f}G  {'yes' if c.trainable else 'no'}"
        )

    if plan.dataset_rows:
        print(f"\n{'dataset':<52} {'download':>11}")
        for name, size in sorted(plan.dataset_rows, key=lambda r: -r[1]):
            print(f"{name[:52]:<52} {size / GIB:>10.2f}G")

    print("\n--- GPU memory ---")
    for k, v in plan.vram.items():
        print(f"  {k:<38} {_fmt(v)}")
    print("\n--- Disk ---")
    for k, v in plan.disk.items():
        print(f"  {k:<38} {_fmt(v)}")

    for w in plan.warnings:
        print(f"\n  ! {w}")


def _print_gpu_choice(
    need_gib: float,
    placeable: list[tuple[int, str]],
    fitting: list[tuple[int, str]],
    needs_volume: bool,
) -> None:
    """Report the chosen GPU, the runners-up, and any excluded for lack of volumes."""
    vram_gib, gpu = placeable[0]
    why = " with network-volume support" if needs_volume else ""
    print(f"  Cheapest listed GPU that fits {need_gib:.0f} GiB{why}: {gpu} ({vram_gib} GB)")
    others = ", ".join(f"{g} ({v} GB)" for v, g in placeable[1:6])
    if others:
        print(f"  Also fit (--gpu to override): {others}")
    skipped = [g for v, g in fitting if (v, g) not in placeable]
    if skipped:
        print(f"  Fit on VRAM but have no volume-capable datacenter: {', '.join(skipped[:6])}")
    print()


def _pick_gpu(plan: Plan, needs_volume: bool) -> str:
    """Choose the cheapest listed GPU that fits the VRAM estimate (and volume needs)."""
    # Pick the GPU from the VRAM estimate rather than defaulting to an H100.
    # This recipe needs ~36 GiB; an 80 GB H100 is roughly 2x the card and
    # several times the price of the smallest part that fits. `gpu_catalog.available_gpus`
    # returns fitting types smallest-first, which approximates cheapest-first
    # (the catalog exposes no price field).
    #
    # When the run needs a network volume, the choice is JOINTLY constrained:
    # the GPU has to exist in a datacenter that also supports volumes. Picking
    # on VRAM alone selects e.g. an A40, whose datacenters have no volume
    # support, and then there is nowhere to put 1.5 TiB.
    need_gib = plan.need_vram_gib
    fitting = gpu_catalog.available_gpus(need_gib)
    placeable = [
        (vram_gib, gpu_id)
        for vram_gib, gpu_id in fitting
        if not needs_volume or gpu_catalog.datacenters_for_gpu(gpu_id, require_network_volume=True)
    ]
    if placeable:
        _print_gpu_choice(need_gib, placeable, fitting, needs_volume)
        return placeable[0][1]
    if fitting:
        vram_gib, gpu = fitting[0]
        print(
            f"  {gpu} ({vram_gib} GB) fits, but no datacenter offering it also\n"
            f"  supports network volumes. Either cut the dataset mix below\n"
            f"  {NETWORK_VOLUME_THRESHOLD_GB} GB, or override --gpu.\n"
        )
        return gpu
    gpu = "NVIDIA H100 80GB HBM3"
    print(f"  No listed GPU reports enough VRAM for {need_gib:.0f} GiB; falling back to {gpu}.\n")
    return gpu


def _volume_datacenter(gpu: str) -> str:
    """List the datacenters offering `gpu` with volume support; return the one to use."""
    dcs = gpu_catalog.datacenters_for_gpu(gpu, require_network_volume=True)
    excluded = [
        d
        for d, _, _ in gpu_catalog.datacenters_for_gpu(gpu)
        if d not in gpu_catalog.NETWORK_VOLUME_DATACENTERS
    ]
    if not dcs:
        print(
            "  No datacenter both offers this GPU and supports network volumes\n"
            "  (or `runpodctl datacenter list` could not be read). Substitute a\n"
            "  datacenter id below after checking `runpodctl datacenter list`.\n"
        )
        return "<DC_ID>"
    print(f"  Datacenters with {gpu} AND network-volume support:")
    for dc_id, loc, stock in dcs:
        print(f"    {dc_id:<12} {loc:<22} stock: {stock or 'unreported'}")
    if excluded:
        print(f"\n  Has the GPU but NO network-volume support, so excluded: {', '.join(excluded)}")
    print(
        "\n  A network volume is pinned to one datacenter and a pod can only\n"
        "  mount a volume in its own, so both commands below use the same id.\n"
        "  If creation is refused, the error lists the currently supported\n"
        "  datacenters -- refresh NETWORK_VOLUME_DATACENTERS in gpu_catalog.py.\n"
    )
    return dcs[0][0]


def _print_volume_provision(plan: Plan, experiment: str, gpu: str, image: str) -> None:
    """Print the network-volume + pod commands for a run too big for a container disk."""
    # Everything that grows lives on /workspace (HF_HOME and
    # HF_DATASETS_CACHE both point there), so a network volume absorbs the
    # whole figure and the container disk only has to hold the image plus
    # the python env. Asking for a >1 TB container disk is the usual cause
    # of "no instances available" on every GPU type -- container disks come
    # from host-local storage.
    vol_gb = min(plan.disk_gb, NETWORK_VOLUME_MAX_GB)
    vol_name = f"tiny-audio-{experiment}"
    capped = vol_gb >= NETWORK_VOLUME_MAX_GB
    dc_id = _volume_datacenter(gpu)

    print(
        f"  # 1. create the volume (size in GB; max {NETWORK_VOLUME_MAX_GB}):\n"
        f"  runpodctl network-volume create --name {vol_name} \\\n"
        f"    --size {vol_gb} --data-center-id {dc_id}\n\n"
        f"  # 2. create the pod in that SAME datacenter, attaching the volume:\n"
        f"  runpodctl pod create --name tiny-audio-{experiment} \\\n"
        f'    --gpu-id "{gpu}" --image {image} \\\n'
        f"    --network-volume-id <VOLUME_ID> --data-center-ids {dc_id} \\\n"
        f"    --container-disk-in-gb {CONTAINER_DISK_WITH_VOLUME_GB} --ports '{POD_PORTS}' \\\n"
        f'    --env "{{\\"SSH_PUBLIC_KEY\\":\\"$(cat ~/.ssh/id_ed25519.pub)\\"}}"\n'
    )
    if capped:
        print(
            f"  ! {plan.disk['recommended'] / 1024:.2f} TiB needed but a RunPod network "
            f"volume\n    caps at {NETWORK_VOLUME_MAX_GB} GB. Cut the dataset mix, or stage "
            "sources\n    across runs -- this will not fit on one volume.\n"
        )
    print(
        f"  Container disk stays at {CONTAINER_DISK_WITH_VOLUME_GB} GB on purpose: with the\n"
        f"  volume mounted at /workspace, weights and datasets land there, and the\n"
        f"  container disk only holds the image and the python env.\n"
    )


def plan_command(
    experiment: str = typer.Option("granite_qwen_frozen", "--experiment", "-e"),
    seq_len: int = typer.Option(
        320,
        "--seq-len",
        help=(
            "Assumed tokens per sample. Default 320 is the measured granite_qwen sequence: "
            "237 audio tokens at the 19s collator ceiling, plus prompt and transcript. "
            "Raise it for recipes with a longer window -- the activation term is linear in this."
        ),
    ),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        help="GPU id for the emitted command (default: cheapest listed GPU that fits)",
    ),
    image: str = typer.Option("runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "--image"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    overrides: Annotated[list[str] | None, typer.Argument(help="Extra Hydra overrides")] = None,
) -> int | None:
    """Estimate GPU memory and disk for a training config, and emit a pod command."""
    plan = build_plan(experiment, list(overrides or []), seq_len)

    if as_json:
        _print_plan_json(plan, experiment)
        return None

    _print_plan_tables(plan, experiment, seq_len)

    disk_gb = plan.disk_gb
    print("\n--- Provision ---")
    needs_volume = disk_gb > NETWORK_VOLUME_THRESHOLD_GB
    if gpu is None:
        gpu = _pick_gpu(plan, needs_volume)

    if needs_volume:
        _print_volume_provision(plan, experiment, gpu, image)
    else:
        print(
            f"  runpodctl pod create --name tiny-audio-{experiment} \\\n"
            f'    --gpu-id "{gpu}" --image {image} \\\n'
            f"    --container-disk-in-gb {disk_gb} --ports '{POD_PORTS}' \\\n"
            f'    --env "{{\\"SSH_PUBLIC_KEY\\":\\"$(cat ~/.ssh/id_ed25519.pub)\\"}}"\n'
        )
    print(f"  Needs a GPU with >= {plan.need_vram_gib:.0f} GiB VRAM.")
    print(
        "  Disk note: the remote training script exports HF_HOME=/workspace/.cache\n"
        "  and HF_DATASETS_CACHE=/workspace/datasets, so the figure above must be\n"
        "  satisfied by whatever backs /workspace -- the network volume when one is\n"
        "  attached (--network-volume-id), otherwise the container disk. Sizing the\n"
        "  container disk while downloads land on a smaller volume, or vice versa,\n"
        "  is the usual way this fails at 95% of a weights pull.\n"
    )
    return 0


def provision_command(
    experiment: str = typer.Option("granite_qwen_frozen", "--experiment", "-e"),
    seq_len: int = typer.Option(320, "--seq-len"),
    name: str | None = typer.Option(
        None, "--name", help="Pod name (default tiny-audio-<experiment>)"
    ),
    image: str = typer.Option("runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404", "--image"),
    max_attempts: int = typer.Option(6, "--max-attempts", help="How many GPU types to try"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan and candidates, create nothing"
    ),
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> str | None:
    """Size a config, then create a pod on the first GPU type that has capacity.

    RunPod's catalog reports `available: true` and `stockStatus: Low` for GPU
    types that still fail to create with "There are no longer any instances
    available with the requested specifications" -- availability is per
    datacenter and racy, so a single hardcoded --gpu-id fails intermittently.
    This walks the fitting GPU types smallest-first until one actually comes up.
    """
    plan = build_plan(experiment, list(overrides or []), seq_len)
    vram = plan.need_vram_gib
    disk = plan.disk_gb
    candidates = gpu_catalog.available_gpus(vram)

    print(f"\n{experiment}: needs >= {vram:.1f} GiB VRAM, {disk} GB disk")
    # Surface the same warnings `plan` prints -- the network-volume one in
    # particular explains a container-disk request that RunPod will refuse to
    # fill, which otherwise reads as plain "no capacity" on every GPU type.
    for w in plan.warnings:
        print(f"  ! {w}")
    if not candidates:
        print("No listed GPU type has enough VRAM. Reduce batch size or enable")
        print("gradient_checkpointing, then re-run.")
        raise typer.Exit(1)
    print(f"candidates (smallest first): {', '.join(g for _, g in candidates[:max_attempts])}\n")

    pubkey = ssh_public_key()

    if dry_run:
        return None

    pod_id = create_first_available(
        name or f"tiny-audio-{experiment}",
        [gpu_id for _, gpu_id in candidates[:max_attempts]],
        image=image,
        disk_gb=disk,
        pubkey=pubkey,
    )
    print("\nNext:")
    print(f"  poetry run ta runpod wait {pod_id}          # prints <ip> <port>")
    print("  poetry run ta runpod deploy <ip> <port>")
    print(
        f"  poetry run ta runpod train <ip> <port> -e {experiment} "
        "--no-attach --session-name run1 -f"
    )
    print(f"  runpodctl pod delete {pod_id}               # when finished\n")
    return pod_id


def wait_command(
    pod_id: str = typer.Argument(..., help="Pod id from `ta runpod up`"),
    timeout_s: int = typer.Option(900, "--timeout", help="Give up after this long"),
) -> None:
    """Block until a pod exposes SSH, then print `<ip> <port>`.

    The endpoint lives at top-level `ssh.ip` / `ssh.port` in the pod JSON.
    `runtime` stays null the whole time on these images, so watching it makes a
    perfectly healthy pod look hung for the 5-10 minutes the image pull takes.
    """

    @retry(
        retry=retry_if_result(lambda ep: ep is None),
        stop=stop_after_delay(timeout_s),
        wait=wait_fixed(15),
    )
    def poll() -> tuple[str, int] | None:
        out = gpu_catalog.runpodctl_json("pod", "get", pod_id)
        try:
            ssh: dict[str, Any] = json.loads(out[out.index("{") :]).get("ssh") or {}
        except (ValueError, AttributeError):
            # No JSON yet (or a partial write) -- same as "not ready", poll again.
            return None
        if ssh.get("ip") and ssh.get("port"):
            return ssh["ip"], ssh["port"]
        return None

    try:
        endpoint = poll()
    except RetryError:
        print(f"Pod {pod_id} exposed no SSH endpoint within {timeout_s}s.")
        raise typer.Exit(1) from None
    # retry_if_result keeps polling while the result is None, so a return
    # without RetryError always carries an endpoint.
    assert endpoint is not None
    ip, port = endpoint
    print(f"{ip} {port}")
