"""Build the Swift SDK's MLX bundle from a trained projector + a converted MLX-LM decoder.

Reads PT GLM-ASR encoder weights from upstream HF, applies the encoder
name remap + Conv1d axis swap, and quantizes Linear/Embedding weights
4-bit with `mlx.core.quantize` directly (no `mlx.nn` modules involved).
The trained projector's `model.safetensors` becomes the bundle's
fp16 `projector.safetensors`. The decoder is mirrored from a local MLX-LM
directory (typically produced by `ta mlx convert-decoder`).

Usage:
    ta mlx build-bundle \\
        --projector mazesmazes/tiny-audio-embedded \\
        --decoder /path/to/mlx-lm-decoder-dir \\
        --output-dir swift/Sources/TinyAudio/Resources/Model
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

MLX_FORMAT_VERSION = 1
DEFAULT_ENCODER_REPO = "zai-org/GLM-ASR-Nano-2512"
AUDIO_TOKEN = "<audio>"
EOS_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")
ENCODER_CONV_LAYERS = [[1, 3, 1], [1, 3, 2]]
HOP_LENGTH = 160

# Suffixes whose `.weight` gets 4-bit quantized (yielding `.weight`/`.scales`/`.biases`
# alongside any pre-existing `.bias`). Mirrors what `mlx.nn.quantize` does when walking
# a module tree built from mlx-audio's WhisperEncoder + nn.LayerNorm — Linear and
# Embedding parameters get quantized, LayerNorm/Conv1d/biases pass through fp16.
_QUANTIZED_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.out_proj.weight",
    ".fc1.weight",
    ".fc2.weight",
    ".embed_positions.weight",
)

_CONV1D_KEYS = ("encoder.conv1.weight", "encoder.conv2.weight")
# mlx-audio's WhisperEncoder builds `embed_positions` (sinusoidal) in fp32. Swift's
# `update(verify: .all)` rejects mismatched dtypes, so quantize from fp32 to keep
# scales/biases fp32 matching the runtime module.
_FP32_KEYS = ("encoder.embed_positions.weight",)


def _remap_pt_encoder_key(k: str) -> str:
    """PT GlmAsrEncoder root keys -> mlx-audio Whisper-style names.

    Keys come from `pt_full.audio_tower.state_dict()` so the `audio_tower.`
    prefix has already been stripped.
    """
    if k.startswith("norm."):
        return k
    new = k.replace("input_layernorm", "self_attn_layer_norm")
    new = new.replace("post_attention_layernorm", "final_layer_norm")
    new = new.replace("mlp.fc1", "fc1")
    new = new.replace("mlp.fc2", "fc2")
    new = new.replace("self_attn.o_proj", "self_attn.out_proj")
    return f"encoder.{new}"


def _is_quantized(key: str) -> bool:
    return any(key.endswith(s) for s in _QUANTIZED_SUFFIXES)


def _sinusoidal_positions(length: int, channels: int, max_timescale: float = 10000.0) -> np.ndarray:
    """Standard Whisper sinusoidal position embedding.

    GLM-ASR's encoder uses RoPE (`use_rope=True`), so `embed_positions` is a
    vestigial `nn.Embedding` that mlx-audio always allocates "for weight loading
    compatibility" but never reads in the forward pass. The Swift runtime still
    expects it as a parameter, so we synthesize valid sinusoidal values whose
    actual content doesn't matter.
    """
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2))
    scaled_time = np.arange(length)[:, None] * inv_timescales[None, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)


def _build_encoder_arrays(
    pt_state_dict: dict,
    *,
    q_bits: int,
    q_group_size: int,
    max_source_positions: int,
    embed_dim: int,
) -> dict[str, mx.array]:
    """Apply name remap, Conv1d axis swap, and quantization. Returns MLX arrays.

    Synthesizes a vestigial `encoder.embed_positions.weight` (sinusoidal) — the
    PT checkpoint doesn't have it because GLM-ASR uses RoPE, but mlx-audio's
    WhisperEncoder always allocates the parameter, and Swift's
    `update(verify: .all)` rejects loads with missing keys.
    """
    import torch

    out: dict[str, mx.array] = {}

    pt_items = list(pt_state_dict.items())
    pt_items.append(
        (
            "embed_positions.weight",
            torch.from_numpy(_sinusoidal_positions(max_source_positions, embed_dim)),
        )
    )

    for pt_key, pt_tensor in pt_items:
        if pt_tensor.dtype == torch.bfloat16:
            pt_tensor = pt_tensor.to(torch.float16)
        np_arr = pt_tensor.detach().cpu().numpy()
        new_key = _remap_pt_encoder_key(pt_key)

        # PT Conv1d weight is [out, in, kernel]; mlx Conv1d wants [out, kernel, in].
        if new_key in _CONV1D_KEYS:
            np_arr = np.swapaxes(np_arr, 1, 2)

        if new_key in _FP32_KEYS:
            np_arr = np_arr.astype(np.float32)

        mx_arr = mx.array(np_arr)
        if _is_quantized(new_key):
            wq, scales, biases = mx.quantize(mx_arr, group_size=q_group_size, bits=q_bits)
            out[new_key] = wq
            out[new_key.replace(".weight", ".scales")] = scales
            out[new_key.replace(".weight", ".biases")] = biases
        else:
            out[new_key] = mx_arr
    return out


def _build_projector_arrays(pt_state_dict: dict) -> dict[str, mx.array]:
    """Strip `projector.` prefix; pass through fp16 (no quantization).

    Trained checkpoints often live in bfloat16 (`model_dtype: bfloat16`), which
    numpy cannot consume — we cast to fp16 so the bundle's projector matches
    Swift's expected dtype.
    """
    import torch

    out: dict[str, mx.array] = {}
    for k, v in pt_state_dict.items():
        if not k.startswith("projector."):
            continue
        if v.dtype == torch.bfloat16:
            v = v.to(torch.float16)
        np_arr = v.detach().cpu().numpy()
        out[k[len("projector.") :]] = mx.array(np_arr)
    return out


def _encoder_config_dict(hf_audio_cfg) -> dict:
    head_dim = getattr(hf_audio_cfg, "head_dim", None) or (
        hf_audio_cfg.hidden_size // hf_audio_cfg.num_attention_heads
    )
    rope_params = getattr(hf_audio_cfg, "rope_parameters", {}) or {}
    return {
        "encoder_dim": int(hf_audio_cfg.hidden_size),
        "n_mels": int(hf_audio_cfg.num_mel_bins),
        "num_layers": int(hf_audio_cfg.num_hidden_layers),
        "num_heads": int(hf_audio_cfg.num_attention_heads),
        "head_dim": int(head_dim),
        "intermediate_size": int(hf_audio_cfg.intermediate_size),
        "rope_theta": float(rope_params.get("rope_theta", 10000.0)),
    }


def _resolve_decoder_dir(decoder: str) -> Path:
    """Treat as a local path if it exists on disk; otherwise snapshot-download."""
    if Path(decoder).is_dir():
        return Path(decoder)
    return Path(snapshot_download(decoder))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _eos_token_ids(tokenizer, candidates: Iterable[str]) -> list[int]:
    seen: set[int] = set()
    unk = tokenizer.unk_token_id
    for s in candidates:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != unk:
            seen.add(int(tid))
    return sorted(seen)


def build_bundle(
    *,
    projector_repo: str,
    encoder_repo: str,
    decoder: str,
    output_dir: Path,
    q_bits: int,
    q_group_size: int,
) -> None:
    import torch
    from safetensors.torch import load_file as load_safetensors_pt
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PT encoder from {encoder_repo}...")
    pt_full = AutoModelForSeq2SeqLM.from_pretrained(
        encoder_repo, trust_remote_code=True, dtype=torch.float16
    )
    pt_encoder_sd = pt_full.audio_tower.state_dict()
    audio_cfg = AutoConfig.from_pretrained(encoder_repo, trust_remote_code=True).audio_config
    del pt_full

    print(f"Quantizing encoder ({q_bits}-bit, group={q_group_size})...")
    enc_arrays = _build_encoder_arrays(
        pt_encoder_sd,
        q_bits=q_bits,
        q_group_size=q_group_size,
        max_source_positions=int(audio_cfg.max_position_embeddings),
        embed_dim=int(audio_cfg.hidden_size),
    )
    mx.save_safetensors(str(output_dir / "encoder.safetensors"), enc_arrays)

    print(f"Loading projector from {projector_repo}...")
    projector_path = Path(snapshot_download(projector_repo))
    proj_sd = load_safetensors_pt(str(projector_path / "model.safetensors"))
    proj_arrays = _build_projector_arrays(proj_sd)
    if not proj_arrays:
        raise SystemExit(
            f"{projector_repo}/model.safetensors has no projector.* keys — "
            f"is this a tiny-audio checkpoint?"
        )
    mx.save_safetensors(str(output_dir / "projector.safetensors"), proj_arrays)

    print(f"Mirroring decoder from {decoder}...")
    decoder_dir = _resolve_decoder_dir(decoder)
    for src_name, dst_name in [
        ("model.safetensors", "decoder.safetensors"),
        ("config.json", "decoder_config.json"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
    ]:
        src = decoder_dir / src_name
        if not src.exists():
            raise SystemExit(f"Decoder dir missing required file: {src}")
        shutil.copy(src, output_dir / dst_name)

    # Transformers v5 stores `chat_template` as a sidecar `chat_template.jinja`
    # rather than inside tokenizer_config.json. Swift's Tokenizers library
    # reads the template from tokenizer_config.json, so inject it back.
    #
    # Also force the `enable_thinking is false` branch always-on. Training
    # (`scripts/train.py:470-476`) does the same in-memory tokenizer patch
    # because TRL's DataCollatorForChatML can't pass `enable_thinking=False`
    # to Qwen3. The model learned to expect the empty `<think>\n\n</think>\n\n`
    # block in every prompt, so Swift inference must emit it too — and Swift's
    # `applyChatTemplate(additionalContext:)` doesn't reliably forward kwargs
    # to the Jinja engine. Hardcoding the conditional to `true` makes the
    # emit unconditional regardless of what Swift forwards.
    chat_template_path = decoder_dir / "chat_template.jinja"
    if chat_template_path.exists():
        bundle_tok_cfg_path = output_dir / "tokenizer_config.json"
        tok_cfg = json.loads(bundle_tok_cfg_path.read_text())
        if "chat_template" not in tok_cfg:
            template = chat_template_path.read_text().replace(
                "enable_thinking is defined and enable_thinking is false",
                "true",
            )
            tok_cfg["chat_template"] = template
            bundle_tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2))

    projector_cfg = json.loads((projector_path / "config.json").read_text())
    pool_stride = int(projector_cfg.get("projector_pool_stride", 4))
    projector_hidden_dim = int(projector_cfg.get("projector_hidden_dim") or 1024)
    encoder_dim = int(projector_cfg.get("encoder_dim") or audio_cfg.hidden_size)
    llm_dim = int(projector_cfg["llm_dim"])

    tokenizer = AutoTokenizer.from_pretrained(str(decoder_dir), trust_remote_code=True)
    if AUDIO_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
    audio_token_id = int(tokenizer.convert_tokens_to_ids(AUDIO_TOKEN))

    bundle_config = {
        "mlx_format_version": MLX_FORMAT_VERSION,
        "encoder": _encoder_config_dict(audio_cfg),
        "projector": {
            "encoder_dim": encoder_dim,
            "llm_dim": llm_dim,
            "hidden_dim": projector_hidden_dim,
            "pool_stride": pool_stride,
        },
        "audio_token": AUDIO_TOKEN,
        "audio_token_id": audio_token_id,
        "eos_token_ids": _eos_token_ids(tokenizer, EOS_TOKEN_STRINGS),
        "encoder_conv_layers": ENCODER_CONV_LAYERS,
        "hop_length": HOP_LENGTH,
    }
    (output_dir / "config.json").write_text(json.dumps(bundle_config, indent=2))

    files_for_manifest = [
        p for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json"
    ]
    manifest = {
        "format_version": MLX_FORMAT_VERSION,
        "files": {
            p.name: {"sha256": _sha256(p), "size": p.stat().st_size} for p in files_for_manifest
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Bundle written to {output_dir}")
