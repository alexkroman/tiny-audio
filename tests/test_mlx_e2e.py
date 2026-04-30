"""End-to-end PT-vs-MLX equivalence test on the trained tiny-audio-embedded checkpoint.

The MLX path uses 4-bit quantized encoder + decoder, so transcripts won't be
bit-identical. We assert text is non-empty and matches PT closely (WER < 0.5).
"""

from pathlib import Path

import pytest
import soundfile as sf

mx = pytest.importorskip("mlx.core")

REPO_ID = "mazesmazes/tiny-audio-embedded"
FIXTURE = Path(__file__).parent / "fixtures" / "short_audio.wav"


@pytest.mark.slow
def test_pt_vs_mlx_transcripts_close():
    """Run both PT and MLX paths on the same fixture audio and assert they agree.

    The MLX path uses 4-bit quantization on encoder + decoder while PT uses
    fp16/bf16, so they won't be bit-identical. We use WER < 0.5 as the
    equivalence check (lenient enough for 4-bit quantization effects on clean
    speech, tight enough to catch real wiring bugs).

    Skips with a clear message if the published checkpoint isn't Rev 3 yet.
    """
    if not FIXTURE.exists():
        pytest.skip(f"Fixture missing: {FIXTURE}")

    from transformers import pipeline

    from tiny_audio.mlx import MLXASRModel

    # Load fixture audio
    audio, sr = sf.read(FIXTURE, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    assert sr == 16000, f"fixture must be 16 kHz, got {sr}"

    # PT path: load pipeline and transcribe
    pt_pipe = pipeline("automatic-speech-recognition", model=REPO_ID, trust_remote_code=True)
    pt_pipe.model.generation_config.do_sample = False
    pt_pipe.model.generation_config.num_beams = 1
    pt_text = pt_pipe(audio)["text"].strip().lower()

    # MLX path: load model and transcribe
    try:
        mlx_model = MLXASRModel.from_pretrained(REPO_ID)
    except ValueError as e:
        if "Rev 3" in str(e):
            pytest.skip(f"Checkpoint is not Rev 3 yet: {e}")
        raise

    mlx_text = mlx_model.transcribe(audio, max_new_tokens=128).strip().lower()

    # Validate both are non-empty
    assert pt_text, "PT transcript is empty"
    assert mlx_text, "MLX transcript is empty"

    # Compute WER between the two. 4-bit quantization typically costs <= 2 WER
    # on clean speech, so 0.5 WER is lenient but tight enough to catch bugs.
    import jiwer

    wer = jiwer.wer(pt_text, mlx_text)
    assert wer < 0.5, (
        f"WER between PT and MLX too high: {wer:.3f}\nPT:  {pt_text!r}\nMLX: {mlx_text!r}"
    )
