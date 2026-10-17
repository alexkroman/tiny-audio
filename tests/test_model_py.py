"""Keep the read-first model.py runnable: it is the file the README sends newcomers to.

Runs it end to end on the offline fixtures (random-weight Whisper-tiny and
SmolLM2-135M), so it checks wiring -- shapes, label masking, what trains --
never transcript quality.
"""

import numpy as np
import pytest
import torch

from model import IGNORE, Projector, TinyAudio


@pytest.fixture(scope="module")
def tiny() -> TinyAudio:
    """model.py's TinyAudio built from the fixture checkpoints."""
    torch.manual_seed(0)
    return TinyAudio.from_pretrained("openai/whisper-tiny", "HuggingFaceTB/SmolLM2-135M-Instruct")


def _clip(seconds: float) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(int(16_000 * seconds)).astype(np.float32)


def test_projector_stacks_k_frames_and_drops_the_tail() -> None:
    """Ten frames at k=4 make two audio tokens of the LLM's width."""
    out = Projector(encoder_dim=8, llm_dim=16, k=4)(torch.randn(2, 10, 8))
    assert out.shape == (2, 2, 16)


def test_audio_tokens_scale_with_clip_length_not_whisper_padding(tiny: TinyAudio) -> None:
    """A 1 s clip is 50 encoder frames -> 12 tokens, not 1500 // 4 for the 30 s pad."""
    short, long = tiny.embed_audio([_clip(1.0), _clip(4.0)])
    assert short.shape == (12, tiny.decoder.config.hidden_size)
    assert len(long) == 50


def test_labels_cover_only_the_transcript_and_eos(tiny: TinyAudio) -> None:
    """The prompt and the audio are context, so the loss must ignore them."""
    audio = tiny.embed_audio([_clip(1.0)])[0]
    embeds, labels = tiny.build_prompt(audio, "hello world")
    target = tiny.tokenizer("hello world", add_special_tokens=False).input_ids
    assert labels is not None
    assert len(labels) == len(embeds)
    assert labels[labels != IGNORE].tolist() == [*target, tiny.tokenizer.eos_token_id]


def test_only_the_projector_receives_gradients(tiny: TinyAudio) -> None:
    """A training step must leave the encoder and decoder frozen."""
    tiny.zero_grad()
    loss = tiny([_clip(1.0), _clip(2.5)], ["hello world", "a longer second transcript"])
    loss.backward()
    assert torch.isfinite(loss)
    trained = {name for name, p in tiny.named_parameters() if p.grad is not None}
    assert trained == {name for name, _ in tiny.projector.named_parameters(prefix="projector")}


def test_transcribe_returns_text(tiny: TinyAudio) -> None:
    """Generation from embeddings alone runs and decodes to a string."""
    assert isinstance(tiny.transcribe(_clip(1.0), max_new_tokens=4), str)
