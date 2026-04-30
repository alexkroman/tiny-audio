"""MLXASRModel: end-to-end MLX inference orchestrator for the embedded experiment.

Wires the hand-ported GLM-ASR encoder (R3-2) + MLX MLP projector (R3-3) + an
mlx-lm-loaded Qwen3 4-bit decoder, exposing transcribe() / transcribe_streaming().

Design choices:
- Encoder weights are loaded fp16 from the live `zai-org/GLM-ASR-Nano-2512` PT
  checkpoint, then 4-bit quantized in-place via `mlx.nn.quantize`. This avoids
  shipping a separate MLX-converted encoder (option E1).
- Projector weights come from the trained tiny-audio checkpoint (only projector
  parameters are saved per `ASRModel.state_dict`).
- Decoder is an mlx-lm pre-quantized Qwen3 model so the language head benefits
  from MLX's optimized 4-bit kernels.
- Audio embeddings are spliced into the prompt's embedding stream at the
  positions of the `<audio>` placeholder tokens; prefill is done via the
  decoder's `input_embeddings` argument, then decode steps run normally.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np

from tiny_audio.mlx.convert import (
    MLX_FORMAT_VERSION,
    convert_projector_weights,
    default_cache_root,
    mark_cache_complete,
    safe_repo_id,
)
from tiny_audio.mlx.encoder import (
    GLMASREncoder,
    compute_mel_unpadded,
    encoder_config_from_hf,
)
from tiny_audio.mlx.processor import AUDIO_TOKEN, build_prompt_input_ids
from tiny_audio.mlx.projector import MLXMLPProjector

AudioInput = Union[str, np.ndarray, "list[float]"]

# Decoder repo: pre-quantized Qwen3-0.6B 4-bit, structurally compatible with
# the trained Qwen/Qwen3-0.6B base used for fine-tuning.
_DECODER_MLX_REPO = "Qwen/Qwen3-0.6B-MLX-4bit"

_EOS_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")


def splice_audio_embeds(
    text_embeds: mx.array,
    audio_embeds: mx.array,
    audio_token_positions: np.ndarray,
) -> mx.array:
    """Replace text_embeds rows at audio_token_positions with audio_embeds.

    Args:
        text_embeds: [1, T, D]   - text-embedded prompt sequence.
        audio_embeds: [N, D]     - projected audio frames (one per <audio> token).
        audio_token_positions: 1D int positions where N == len(positions).

    Returns:
        [1, T, D] mx.array with the same dtype as text_embeds.

    Implementation: builds an index lookup `[T]` mapping each prompt position
    to its audio-embed row (or to a sentinel zero row), then a single
    mx.gather + mx.where keeps the whole splice on the GPU. Avoids forcing a
    host sync of `audio_embeds`, which would otherwise materialize the
    encoder + projector lazy graph here instead of chaining into the prefill.
    """
    _, t, d = text_embeds.shape
    positions = np.asarray(audio_token_positions)
    n = positions.shape[0]

    # idx_np[p] = i if p == positions[i] else n (sentinel pointing to a zero
    # row appended to audio_embeds). Built on CPU once per call; tiny.
    idx_np = np.full(t, n, dtype=np.int32)
    idx_np[positions] = np.arange(n, dtype=np.int32)
    idx = mx.array(idx_np)

    # [N+1, D]: append a zero row; positions not in `positions` index into it.
    audio_with_sentinel = mx.concatenate(
        [audio_embeds.astype(text_embeds.dtype), mx.zeros((1, d), dtype=text_embeds.dtype)],
        axis=0,
    )
    audio_full = audio_with_sentinel[idx][None, :, :]  # [1, T, D]

    mask_np = np.zeros((1, t, 1), dtype=np.bool_)
    mask_np[0, positions, 0] = True
    mask = mx.array(mask_np)

    return mx.where(mask, audio_full, text_embeds)


class MLXASRModel:
    """End-to-end MLX inference for the tiny-audio embedded model.

    Construct via `MLXASRModel.from_pretrained(repo_id)`.
    """

    def __init__(
        self,
        encoder: GLMASREncoder,
        projector: MLXMLPProjector,
        decoder,
        tokenizer,
        feature_extractor,
        audio_token_id: int,
        eos_token_ids: set[int],
    ):
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.audio_token_id = audio_token_id
        self.eos_token_ids = eos_token_ids

    # ------------------------------------------------------------------ load

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        *,
        force_reconvert: bool = False,
    ) -> MLXASRModel:
        """Load a tiny-audio embedded checkpoint into an MLX inference pipeline.

        Steps:
          1. Cache management (mark-only; no on-disk weight cache yet).
          2. Snapshot-download the trained checkpoint and read its config.
          3. Validate Rev 3 dims (encoder_dim=1280, llm_dim=1024).
          4. Load the mlx-lm pre-quantized Qwen3 decoder + tokenizer.
          5. Build encoder, copy fp16 weights from live `zai-org/GLM-ASR-Nano-2512`,
             then 4-bit quantize in-place.
          6. Build projector + load trained weights from the snapshot.
          7. Mark cache complete.
        """
        import json

        import torch
        from huggingface_hub import snapshot_download
        from mlx_lm import load as mlx_lm_load
        from safetensors.torch import load_file as load_safetensors_pt
        from transformers import (
            AutoConfig,
            AutoFeatureExtractor,
            AutoModelForSeq2SeqLM,
        )

        # 1. Cache directory - currently a marker-only artifact (full converted
        # weight cache is a future optimization).
        cache_dir = default_cache_root() / safe_repo_id(repo_id_or_path)
        if force_reconvert and cache_dir.exists():
            shutil.rmtree(cache_dir)

        # 2. Download trained checkpoint snapshot and load its config.
        trained_path = Path(snapshot_download(repo_id_or_path))
        with (trained_path / "config.json").open() as f:
            cfg = json.load(f)

        # 3. Validate Rev 3 dims.
        if cfg.get("encoder_dim") != 1280 or cfg.get("llm_dim") != 1024:
            raise ValueError(
                f"MLX inference path expects Rev 3 trained checkpoint "
                f"(encoder_dim=1280, llm_dim=1024). Got encoder_dim="
                f"{cfg.get('encoder_dim')}, llm_dim={cfg.get('llm_dim')} - this "
                f"looks like a Rev 1/2 checkpoint. Train against current "
                f"embedded.yaml (GLM-ASR-Nano-2512 + Qwen3-0.6B) and re-publish."
            )

        audio_model_id = cfg.get("audio_model_id", "zai-org/GLM-ASR-Nano-2512")
        projector_pool_stride = cfg.get("projector_pool_stride", 4)
        projector_hidden_dim = cfg.get("projector_hidden_dim", 1024)

        # 4. Load decoder + tokenizer via mlx-lm. This returns a TokenizerWrapper
        # that delegates to a HF tokenizer underneath.
        decoder, tokenizer = mlx_lm_load(_DECODER_MLX_REPO)
        tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
        audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)

        # 5. Resolve EOS ids to whatever the tokenizer actually recognizes.
        eos_token_ids: set[int] = set()
        for eos_str in _EOS_TOKEN_STRINGS:
            tid = tokenizer.convert_tokens_to_ids(eos_str)
            if tid is not None and tid != tokenizer.unk_token_id:
                eos_token_ids.add(int(tid))
        # Also include the model's own eos_token_id if set.
        if tokenizer.eos_token_id is not None:
            eos_token_ids.add(int(tokenizer.eos_token_id))

        # 6. Build encoder + load weights from live PT GLM-ASR-Nano-2512.
        full_cfg = AutoConfig.from_pretrained(audio_model_id, trust_remote_code=True)
        audio_cfg = full_cfg.audio_config
        enc_cfg = encoder_config_from_hf(audio_cfg)
        encoder = GLMASREncoder(enc_cfg)

        pt_full = AutoModelForSeq2SeqLM.from_pretrained(
            audio_model_id, trust_remote_code=True, dtype=torch.float16
        )
        pt_encoder = pt_full.audio_tower
        pt_encoder.train(False)

        cls._load_pt_encoder_into_mlx(pt_encoder, encoder)

        # Free PT memory before quantization.
        del pt_full, pt_encoder
        import gc

        gc.collect()

        # Quantize encoder to 4-bit in-place. mlx.nn.quantize transforms
        # nn.Linear -> QuantizedLinear and quantizes existing weights.
        import mlx.nn as mlx_nn

        mlx_nn.quantize(encoder, group_size=64, bits=4)

        # 7. Build projector and load trained weights.
        projector = MLXMLPProjector(
            encoder_dim=cfg.get("encoder_dim", 1280),
            llm_dim=cfg.get("llm_dim", 1024),
            hidden_dim=projector_hidden_dim,
            pool_stride=projector_pool_stride,
        )
        pt_state = load_safetensors_pt(str(trained_path / "model.safetensors"))
        flat_mlx = convert_projector_weights(pt_state)
        # Module.update() needs nested-dict params, not dotted-key flat dicts.
        from mlx.utils import tree_unflatten

        projector.update(tree_unflatten(list(flat_mlx.items())))

        # 8. Cache feature extractor (used by every transcribe call) + mark cache.
        feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_id)
        feature_extractor.padding = False
        mark_cache_complete(cache_dir, version=MLX_FORMAT_VERSION)

        return cls(
            encoder=encoder,
            projector=projector,
            decoder=decoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            audio_token_id=int(audio_token_id),
            eos_token_ids=eos_token_ids,
        )

    @staticmethod
    def _load_pt_encoder_into_mlx(pt_encoder, mlx_encoder: GLMASREncoder) -> None:
        """Copy PT GlmAsrEncoder weights into our MLX encoder.

        Same logic as `tests/test_mlx_encoder.py::_copy_pt_encoder_to_mlx`:
        fp16 array conversion, with axis-swap for conv1/conv2 weights to bridge
        PT [out, in, kernel] -> MLX [out, kernel, in].
        """
        import torch
        from mlx.utils import tree_unflatten

        sd = pt_encoder.state_dict()
        flat: list[tuple[str, mx.array]] = []
        for k, v in sd.items():
            arr = v.detach().cpu().to(torch.float16).numpy()
            if k in ("conv1.weight", "conv2.weight"):
                arr = np.swapaxes(arr, 1, 2)
            flat.append((k, mx.array(arr)))
        mlx_encoder.update(tree_unflatten(flat))

    # ------------------------------------------------------------ inference

    def warmup(self, *, audio_seconds: float = 1.0, max_new_tokens: int = 4) -> None:
        """Run one synthetic-audio inference to trigger MLX kernel compilation.

        Combined with `mx.compile(shapeless=True)` on the encoder (in
        from_pretrained), one warmup is enough to JIT all per-shape graphs:
        the encoder compiles once and handles any input length; the decoder's
        prefill / decode shapes (T variable on prefill, [1,1] on decode) are
        also covered by a single trace.
        """
        synthetic = np.zeros(int(16000 * audio_seconds), dtype=np.float32)
        self.transcribe(synthetic, max_new_tokens=max_new_tokens)

    def transcribe(
        self,
        audio: AudioInput,
        *,
        max_new_tokens: int = 256,
        system_prompt: str | None = None,
    ) -> str:
        """Run greedy decoding to completion and return the decoded string."""
        token_ids = list(
            self._iter_token_ids(audio, max_new_tokens=max_new_tokens, system_prompt=system_prompt)
        )
        # Strip trailing EOS tokens before decoding.
        clean = [tid for tid in token_ids if tid not in self.eos_token_ids]
        return self.tokenizer.decode(clean)

    def transcribe_streaming(
        self,
        audio: AudioInput,
        *,
        max_new_tokens: int = 256,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """Greedy decode, yielding incremental string deltas.

        After each step we accumulate ids, decode the prefix, and yield the new
        substring relative to the previous decode. This matches the standard
        streaming-text contract: concatenating all yields == one-shot transcribe().
        """
        accumulated: list[int] = []
        last_text = ""
        for tid in self._iter_token_ids(
            audio, max_new_tokens=max_new_tokens, system_prompt=system_prompt
        ):
            if tid in self.eos_token_ids:
                break
            accumulated.append(tid)
            text = self.tokenizer.decode(accumulated)
            # Detokenization can occasionally rewrite earlier tokens (e.g. mid-byte
            # sequences); when that happens, fall back to yielding the whole new text.
            delta = text[len(last_text) :] if text.startswith(last_text) else text
            if delta:
                yield delta
            last_text = text

    def _iter_token_ids(
        self,
        audio: AudioInput,
        *,
        max_new_tokens: int,
        system_prompt: str | None,
    ) -> Iterator[int]:
        from mlx_lm.models import cache as mlx_cache

        audio_np = self._prepare_audio(audio)
        audio_embeds, num_audio = self._encode_audio(audio_np)

        input_ids_np = build_prompt_input_ids(
            self.tokenizer,
            num_audio_tokens=num_audio,
            system_prompt=system_prompt or "",
        )
        audio_positions = np.where(input_ids_np[0] == self.audio_token_id)[0]
        if len(audio_positions) != num_audio:
            raise RuntimeError(
                f"Prompt has {len(audio_positions)} <audio> placeholders but "
                f"projector emitted {num_audio} audio frames. The chat template "
                f"may have stripped or duplicated placeholders."
            )

        # The audio token id may be out-of-range for the embed table on some
        # tokenizers (when add_special_tokens grows past the model's embedding
        # rows). Replace those positions with id 0 before lookup; we splice the
        # audio embeddings in immediately afterward so the value never matters.
        safe_input_ids_np = input_ids_np.copy()
        safe_input_ids_np[input_ids_np == self.audio_token_id] = 0
        safe_input_ids_mx = mx.array(safe_input_ids_np)

        text_embeds = self.decoder.model.embed_tokens(safe_input_ids_mx)
        prefill_embeds = splice_audio_embeds(text_embeds, audio_embeds, audio_positions)

        # Initialize KV cache and prefill via input_embeddings.
        cache = mlx_cache.make_prompt_cache(self.decoder)
        logits = self.decoder(safe_input_ids_mx, cache=cache, input_embeddings=prefill_embeds)

        # Async-eval pipeline: keep tokens as mx.arrays (no Python int round-trip
        # in the inner loop) and kick off step N+1 computation before syncing
        # step N's result. The .item() sync then overlaps with the next forward
        # pass, hiding the host<-device wait. Same pattern as mlx_lm.generate.
        y = mx.argmax(logits[:, -1:, :], axis=-1)  # [1, 1] mx.array
        mx.async_eval(y)

        for n in range(max_new_tokens):
            if n < max_new_tokens - 1:
                next_logits = self.decoder(y, cache=cache)
                next_y = mx.argmax(next_logits[:, -1:, :], axis=-1)
                mx.async_eval(next_y)

            y_int = int(y.item())  # sync on y; overlaps with next compute above
            yield y_int
            if y_int in self.eos_token_ids:
                return
            if n < max_new_tokens - 1:
                y = next_y

    # ------------------------------------------------------------- helpers

    def _prepare_audio(self, audio) -> np.ndarray:
        # AudioDecoder (lazy torchcodec object emitted by recent `datasets`):
        # call get_all_samples() -> AudioSamples (.data is a torch.Tensor [C, T]).
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            data = samples.data.detach().cpu().numpy()  # [C, T]
            sr = int(samples.sample_rate)
            if data.ndim > 1:
                data = data.mean(axis=0)
            return self._maybe_resample(data, sr)

        # AudioSamples directly (in case datasets returns it pre-decoded).
        if hasattr(audio, "data") and hasattr(audio, "sample_rate"):
            arr = audio.data
            arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return self._maybe_resample(arr, int(audio.sample_rate))

        # HF-dataset-style dict: {"array": np.ndarray, "sampling_rate": int}.
        if isinstance(audio, dict):
            data = audio.get("array", audio.get("raw"))
            if data is None:
                raise ValueError(f"audio dict missing 'array' or 'raw' key; got keys={list(audio)}")
            data = np.asarray(data, dtype=np.float32)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            return self._maybe_resample(data, int(audio.get("sampling_rate", 16000)))

        # File path.
        if isinstance(audio, str):
            import soundfile as sf

            data, sr = sf.read(audio, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=-1)
            return self._maybe_resample(data, sr)

        # Fallback: numpy / list of floats.
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _maybe_resample(data: np.ndarray, sr: int) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if sr != 16000:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data.astype(np.float32)

    def _encode_audio(self, audio_np: np.ndarray) -> tuple[mx.array, int]:
        mel, _ = compute_mel_unpadded(audio_np, feature_extractor=self.feature_extractor)
        enc_out = self.encoder(mel)  # [1, T_enc, encoder_dim]
        proj_out = self.projector(enc_out)  # [1, T_proj, llm_dim]
        # The audio-token count is determined entirely by the projector's output
        # length — no need to recompute via compute_num_audio_tokens here.
        num_audio = proj_out.shape[1]
        return proj_out[0, :num_audio, :], num_audio
