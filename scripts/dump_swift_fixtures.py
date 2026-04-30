# scripts/dump_swift_fixtures.py
"""Capture Python-MLX-path intermediate tensors for Swift parity tests.

Runs MLXASRModel on a fixed librispeech sample and writes:
  - librispeech_sample.wav         (the input audio, copied as-is)
  - reference_mel.bin              (encoder.compute_mel_unpadded output)
  - reference_encoder_out.bin      (GLMASREncoder forward output)
  - reference_projector_out.bin    (MLXMLPProjector output)
  - reference_audio_embeddings.bin (audio embeddings spliced into prompt)
  - reference_prompt_token_ids.json (Qwen3 prompt token IDs)
  - reference_token_ids.json       (first 20 greedy-decoded token IDs)
  - shapes.json                    (shape + dtype manifest for Swift loaders)

Usage:
    python scripts/dump_swift_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf

from tiny_audio.mlx import MLXASRModel
from tiny_audio.mlx.encoder import compute_mel_unpadded
from tiny_audio.mlx.model import splice_audio_embeds
from tiny_audio.mlx.processor import build_prompt_input_ids

REPO_ID = "mazesmazes/tiny-audio-embedded"
FIXTURES_DIR = Path("swift/Tests/TinyAudioTests/Fixtures")


def _save_mx(arr: mx.array, path: Path) -> tuple[list[int], str]:
    np_arr = np.asarray(arr).astype(np.float32)
    np_arr.tofile(path)
    return list(np_arr.shape), str(np_arr.dtype)


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {REPO_ID}...")
    model = MLXASRModel.from_pretrained(REPO_ID)

    print("Loading librispeech sample...")
    src = librosa.ex("libri1")
    audio, sr = sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio[: 16000 * 6].astype(np.float32)  # 6 seconds for fast tests
    sf.write(FIXTURES_DIR / "librispeech_sample.wav", audio, 16000, subtype="PCM_16")

    shapes: dict[str, dict] = {}

    print("Running encoder forward...")
    mel, _ = compute_mel_unpadded(audio, feature_extractor=model.feature_extractor)
    shapes["mel"] = dict(zip(["shape", "dtype"], _save_mx(mel, FIXTURES_DIR / "reference_mel.bin")))

    enc_out = model.encoder(mel)
    shapes["encoder_out"] = dict(
        zip(["shape", "dtype"], _save_mx(enc_out, FIXTURES_DIR / "reference_encoder_out.bin"))
    )

    proj_out = model.projector(enc_out)
    shapes["projector_out"] = dict(
        zip(["shape", "dtype"], _save_mx(proj_out, FIXTURES_DIR / "reference_projector_out.bin"))
    )
    num_audio = proj_out.shape[1]

    print("Building prompt + splicing embeddings...")
    input_ids_np = build_prompt_input_ids(
        model.tokenizer, num_audio_tokens=num_audio, system_prompt=""
    )
    audio_positions = np.where(input_ids_np[0] == model.audio_token_id)[0]

    safe_ids = input_ids_np.copy()
    safe_ids[input_ids_np == model.audio_token_id] = 0
    text_embeds = model.decoder.model.embed_tokens(mx.array(safe_ids))
    spliced = splice_audio_embeds(text_embeds, proj_out[0, :num_audio, :], audio_positions)
    shapes["spliced"] = dict(
        zip(["shape", "dtype"], _save_mx(spliced, FIXTURES_DIR / "reference_audio_embeddings.bin"))
    )

    (FIXTURES_DIR / "reference_prompt_token_ids.json").write_text(
        json.dumps(
            {"input_ids": input_ids_np[0].tolist(), "audio_token_id": int(model.audio_token_id)}
        )
    )

    print("Greedy-decoding 20 tokens...")
    token_ids = []
    for tid in model._iter_token_ids(audio, max_new_tokens=20, system_prompt=None):
        token_ids.append(int(tid))
        if len(token_ids) >= 20:
            break
    (FIXTURES_DIR / "reference_token_ids.json").write_text(json.dumps({"token_ids": token_ids}))

    (FIXTURES_DIR / "shapes.json").write_text(json.dumps(shapes, indent=2))
    print(f"Wrote fixtures to {FIXTURES_DIR}/")


if __name__ == "__main__":
    main()
