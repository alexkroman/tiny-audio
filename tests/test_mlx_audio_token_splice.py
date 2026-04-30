"""Splice unit test + slow integration smoke for the MLX orchestrator.

The unit test (`test_splice_audio_embeds_replaces_at_positions`) verifies
the embedding-splice helper in isolation - no model load required, fast.

The slow tests load the published trained checkpoint and run a forward pass.
They are gated behind `-m slow` and additionally skip cleanly if the published
checkpoint isn't a Rev 3 (encoder_dim=1280, llm_dim=1024) artifact.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def test_splice_audio_embeds_replaces_at_positions():
    from tiny_audio.mlx.model import splice_audio_embeds

    rng = np.random.default_rng(0)
    text_embeds = rng.standard_normal((1, 10, 8)).astype(np.float32)
    audio_embeds = rng.standard_normal((3, 8)).astype(np.float32)
    audio_positions = np.array([2, 4, 7])

    out = np.array(
        splice_audio_embeds(
            mx.array(text_embeds),
            mx.array(audio_embeds),
            audio_positions,
        )
    )
    expected = text_embeds.copy()
    expected[0, audio_positions, :] = audio_embeds
    np.testing.assert_array_equal(out, expected)


@pytest.mark.slow
def test_orchestrator_loads_against_rev3_checkpoint():
    """Loads the trained checkpoint and runs transcribe(). Skips with a clear
    message if the published checkpoint isn't Rev 3 yet."""
    from tiny_audio.mlx import MLXASRModel

    try:
        model = MLXASRModel.from_pretrained("mazesmazes/tiny-audio-embedded")
    except ValueError as e:
        if "Rev 3" in str(e):
            pytest.skip(f"No Rev 3 checkpoint published yet: {e}")
        raise

    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000 * 3).astype(np.float32)
    text = model.transcribe(audio, max_new_tokens=10)
    assert isinstance(text, str)


@pytest.mark.slow
def test_orchestrator_streaming_concat_equals_oneshot():
    from tiny_audio.mlx import MLXASRModel

    try:
        model = MLXASRModel.from_pretrained("mazesmazes/tiny-audio-embedded")
    except ValueError as e:
        if "Rev 3" in str(e):
            pytest.skip(f"No Rev 3 checkpoint published yet: {e}")
        raise

    rng = np.random.default_rng(1)
    audio = rng.standard_normal(16000 * 2).astype(np.float32)
    one_shot = model.transcribe(audio, max_new_tokens=15)
    streamed = "".join(model.transcribe_streaming(audio, max_new_tokens=15))
    assert streamed == one_shot


@pytest.mark.slow
def test_audio_token_id_never_in_output():
    """Sharp-edge: <audio> placeholder token id must never appear in the
    decoded output. The trained projector replaces these embedding rows
    before the LM ever sees them, but a poorly trained model could in
    principle still emit `audio_token_id` (151669 on Qwen3) as an output
    token. This test runs N synthetic-audio transcribes and asserts that
    none of the produced token ids are the audio token id.
    """
    from tiny_audio.mlx import MLXASRModel

    try:
        model = MLXASRModel.from_pretrained("mazesmazes/tiny-audio-embedded")
    except ValueError as e:
        if "Rev 3" in str(e):
            pytest.skip(f"No Rev 3 checkpoint published yet: {e}")
        raise

    audio_token_id = model.audio_token_id
    rng = np.random.default_rng(0)
    for seed in range(5):
        audio = rng.standard_normal(16000 * 2).astype(np.float32)
        token_ids = list(model._iter_token_ids(audio, max_new_tokens=20, system_prompt=None))
        assert audio_token_id not in token_ids, (
            f"<audio> token id {audio_token_id} leaked into output for seed {seed}: {token_ids}"
        )
