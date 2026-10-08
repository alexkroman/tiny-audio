"""Tests for speaker-ASR long form: context prefixes, chunked decoding, label linking."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from scripts.eval.speaker_metrics import (
    CONTEXT_END,
    parse_turns,
    sa_errors,
    serialize_turns,
    speaker_metrics,
)
from scripts.speaker_asr.context import ContextConfig, build_context_rows, prompt_speakers
from scripts.speaker_asr.longform import (
    SAMPLE_RATE,
    best_span,
    chunk_bounds,
    time_words,
    transcribe_long,
)

# --------------------------------------------------------------------- format


def test_serialize_turns_keeps_pinned_labels_and_numbers_new_voices_after_them():
    turns = [("B", "back again"), ("Z", "new here"), ("A", "hi")]
    assert serialize_turns(turns, {"A": 1, "B": 2}) == "<SPK_2>back again<SPK_3>new here<SPK_1>hi"


def test_parse_turns_ignores_the_context_end_token():
    assert parse_turns(f"<SPK_1>a{CONTEXT_END}<SPK_2>b") == [(1, "a"), (2, "b")]


def test_sawer_compares_labels_as_given_unlike_cpwer():
    ref, swapped = "<SPK_1>a b<SPK_2>c d", "<SPK_2>a b<SPK_1>c d"
    assert sa_errors(ref, ref) == (0, 4)
    assert sa_errors(ref, swapped) == (4, 4)  # every word substituted under the wrong label
    m = speaker_metrics([ref], [swapped], fixed_labels=True)
    assert (m["cpwer"], m["sawer"]) == (0.0, 1.0)
    assert "sawer" not in speaker_metrics([ref], [swapped])


# --------------------------------------------------------------------- prefix order


def test_prompt_speakers_keep_first_appearance_order_and_most_recent_when_capped():
    first = {"A": 0, "B": 1, "C": 2}
    last = {"A": 9, "B": 1, "C": 5}
    assert prompt_speakers(first, last, lambda s: True, 6) == ["A", "B", "C"]
    assert prompt_speakers(first, last, lambda s: True, 2) == ["A", "C"]
    assert prompt_speakers(first, last, lambda s: s != "A", 6) == ["B", "C"]


# --------------------------------------------------------------------- training rows


def _part(i, speaker, offset, dur):
    return {"row": i, "id": f"u{i}", "speaker": speaker, "offset_s": offset, "dur_s": dur}


def _rows():
    """Two consecutive windows of one meeting: A and B, then B returns with new C."""
    w1 = [_part(0, "A", 0.0, 3.0), _part(1, "B", 3.5, 3.0)]
    w2 = [_part(2, "B", 0.0, 2.5), _part(3, "C", 3.0, 1.0)]
    rows = [
        {"key": "m:u2", "group": "m", "parts": json.dumps(w2), "tail_s": 0.0, "duration_s": 4.0},
        {"key": "m:u0", "group": "m", "parts": json.dumps(w1), "tail_s": 0.0, "duration_s": 6.5},
    ]
    starts = {"u0": 10.0, "u1": 13.5, "u2": 20.0, "u3": 23.0}
    texts = {"u0": "hello there all", "u1": "hi how are you", "u2": "me again", "u3": "yes"}
    return rows, starts, texts


def test_first_window_has_no_prefix_and_plain_numbering():
    rows, starts, texts = _rows()
    first, _ = build_context_rows(rows, starts, texts, ContextConfig(carry_prob=0.0))
    assert first["key"] == "m:u0"  # ordered by meeting time, not manifest order
    assert first["prefix"] == CONTEXT_END
    assert first["target"] == "<SPK_1>hello there all<SPK_2>hi how are you"
    assert first["n_memory"] == 0


def test_later_window_lists_known_voices_keeps_their_labels_and_numbers_new_ones():
    rows, starts, texts = _rows()
    _, second = build_context_rows(rows, starts, texts, ContextConfig(carry_prob=0.0))
    assert second["prefix"] == f"<SPK_1>hello there all<SPK_2>hi how are you{CONTEXT_END}"
    # B keeps label 2 even though it speaks first; C is new, so 3.
    assert second["target"] == "<SPK_2>me again<SPK_3>yes"
    assert (second["n_memory"], second["n_speakers"]) == (2, 2)
    parts = json.loads(second["parts"])
    assert [p["id"] for p in parts] == ["u0", "u1", "u2", "u3"]
    # Samples back to back with the gap, then the live window after live_gap_s.
    assert [p["offset_s"] for p in parts] == [0.0, 3.4, 7.1, 10.1]


def test_carry_puts_the_previous_windows_tail_into_the_prefix():
    rows, starts, texts = _rows()
    cfg = ContextConfig(carry_prob=1.0, carry_s=3.0)
    _, second = build_context_rows(rows, starts, texts, cfg)
    assert second["prefix"] == (
        f"<SPK_1>hello there all<SPK_2>hi how are you<SPK_2>hi how are you{CONTEXT_END}"
    )
    assert [p["id"] for p in json.loads(second["parts"])] == ["u0", "u1", "u1", "u2", "u3"]


def test_short_utterances_never_become_voice_samples():
    rows, starts, texts = _rows()
    cfg = ContextConfig(carry_prob=0.0, memory_s=(3.0, 6.0))  # B's 3.0 s ok, A's 3.0 ok
    _, second = build_context_rows(rows, starts, texts, cfg)
    assert second["n_memory"] == 2
    cfg = ContextConfig(carry_prob=0.0, memory_s=(5.0, 6.0))  # nobody qualifies
    _, second = build_context_rows(rows, starts, texts, cfg)
    assert second["prefix"] == CONTEXT_END
    assert second["target"] == "<SPK_1>me again<SPK_2>yes"


# --------------------------------------------------------------------- chunking + timing


def test_chunks_cut_at_the_quietest_point_within_bounds():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.1, 50 * SAMPLE_RATE).astype(np.float32)
    audio[int(17 * SAMPLE_RATE) : int(17.5 * SAMPLE_RATE)] = 0.0  # a pause at 17 s
    bounds = chunk_bounds(audio, max_s=25, min_s=10)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(audio)
    assert 17 * SAMPLE_RATE <= bounds[0][1] <= 17.5 * SAMPLE_RATE
    assert all(e - s <= 25 * SAMPLE_RATE for s, e in bounds)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(bounds))
    assert chunk_bounds(audio[: 5 * SAMPLE_RATE], 25, 10) == [(0, 5 * SAMPLE_RATE)]


def test_time_words_matches_aligned_words_back_to_turns_skipping_unaligned():
    turns = [(1, "it costs 25 dollars"), (2, "okay")]
    aligned = [  # the aligner drops "25"
        {"word": "it", "start": 0.0, "end": 0.2},
        {"word": "costs", "start": 0.2, "end": 0.6},
        {"word": "dollars", "start": 1.0, "end": 1.4},
        {"word": "okay", "start": 2.0, "end": 2.3},
    ]
    timed = time_words(turns, aligned)
    assert [w for w, _, _ in timed[0]] == ["it", "costs", "dollars"]
    assert timed[1] == [("okay", 2.0, 2.3)]


def test_best_span_takes_the_longest_run_within_range():
    words = [(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(10)]
    start, end, text = best_span(words, 2.0, 4.0)
    assert end - start <= 4.0
    assert end - start == pytest.approx(3.8)
    assert text.split()[0] == "w0"
    assert best_span(words[:1], 2.0, 4.0) is None
    assert best_span([("long", 0.0, 9.0)], 2.0, 4.0) is None


# --------------------------------------------------------------------- linking


class _Tokenizer:
    def __init__(self, context: bool):
        self.vocab = {f"<SPK_{i}>": i for i in range(1, 9)}
        if context:
            self.vocab[CONTEXT_END] = 99

    def get_vocab(self):
        return dict(self.vocab)


def _fake_decoder(outputs: list[str], seen: list):
    def transcribe(model, processor, audios, n_speakers, max_new_tokens, prefixes=None, **kw):
        seen.append(prefixes[0] if prefixes else None)
        return [outputs[len(seen) - 1]]

    return transcribe


def _fake_align(audio, text):
    """Words 0.6 s apart from the start of the chunk."""
    return [{"word": w, "start": i * 0.6, "end": i * 0.6 + 0.5} for i, w in enumerate(text.split())]


def test_long_form_links_a_returning_speaker_and_numbers_a_new_one(monkeypatch):
    import scripts.speaker_asr.model as model_mod

    seen: list = []
    outputs = [
        "<SPK_1>one two three four five six seven<SPK_2>eight nine ten eleven twelve thirteen",
        "<SPK_2>back again after the break today<SPK_3>and i am new",
    ]
    monkeypatch.setattr(model_mod, "transcribe_speakers", _fake_decoder(outputs, seen))
    processor = type("P", (), {"tokenizer": _Tokenizer(context=True)})()
    audio = np.random.default_rng(1).normal(0, 0.1, 40 * SAMPLE_RATE).astype(np.float32)
    cfg = ContextConfig(chunk_s=25, min_chunk_s=15, carry_s=3.0)
    result = transcribe_long(None, processor, audio, cfg, align=_fake_align)

    assert len(result.chunks) == 2
    assert seen[0] == CONTEXT_END  # nothing heard yet
    # Second chunk: both speakers' samples, then the carried tail, labelled 1 and 2.
    assert seen[1].startswith("<SPK_1>one two three")
    assert "<SPK_2>eight" in seen[1]
    assert seen[1].endswith(CONTEXT_END)
    speakers = [t.speaker for t in result.turns]
    assert speakers == [1, 2, 3]  # chunk 2's SPK_2 is speaker 2 again; SPK_3 is new
    assert result.turns[1].text.startswith("eight nine")
    assert "back again" in result.turns[1].text
    assert result.text.startswith("<SPK_1>one two")
    assert result.turns[2].start == pytest.approx(result.chunks[1][0] + 6 * 0.6)


def test_without_context_token_every_chunk_gets_new_speakers(monkeypatch):
    import scripts.speaker_asr.model as model_mod

    seen: list = []
    outputs = ["<SPK_1>a b c<SPK_2>d e f", "<SPK_1>g h i"]
    monkeypatch.setattr(model_mod, "transcribe_speakers", _fake_decoder(outputs, seen))
    processor = type("P", (), {"tokenizer": _Tokenizer(context=False)})()
    audio = np.random.default_rng(2).normal(0, 0.1, 40 * SAMPLE_RATE).astype(np.float32)
    result = transcribe_long(
        None, processor, audio, ContextConfig(chunk_s=25, min_chunk_s=15), align=_fake_align
    )
    assert seen == [None, None]  # no prefixes for a model that cannot read them
    assert [t.speaker for t in result.turns] == [1, 2, 3]  # no linking: chunk 2 is new


# --------------------------------------------------------------------- meetings + collator


def test_meeting_parts_keep_real_time_and_crop():
    from scripts.speaker_asr.data import meeting_parts

    meta = pd.DataFrame(
        {
            "id": ["a", "b", "c", "x"],
            "group": ["m", "m", "m", "other"],
            "speaker": ["A", "B", "A", "Z"],
            "start": [100.0, 160.0, 103.0, 0.0],
            "end": [102.0, 162.0, 104.0, 1.0],
            "row": [0, 1, 2, 3],
        }
    )
    parts = meeting_parts(meta, "m")
    assert [(p["id"], p["offset_s"]) for p in parts] == [("a", 0.0), ("c", 3.0), ("b", 60.0)]
    assert [p["id"] for p in meeting_parts(meta, "m", max_s=10)] == ["a", "c"]


def test_collator_supervises_only_after_the_context_prefix(monkeypatch):
    import torch

    from scripts.speaker_asr.data import SpeakerASRCollator
    from scripts.turn_aware.data import TurnAwareCollator

    def fake_parent_call(self, batch):
        assert batch[0]["target"].startswith("<SPK_1>sample")  # prefix + target, joined
        ids = torch.tensor([[5, 6, 7, 99, 8, 9]])
        return {"input_ids": ids, "labels": ids.clone()}

    monkeypatch.setattr(TurnAwareCollator, "__init__", lambda self, processor: None)
    monkeypatch.setattr(TurnAwareCollator, "__call__", fake_parent_call)
    processor = type("P", (), {"tokenizer": _Tokenizer(context=True)})()
    collator = SpeakerASRCollator(processor)
    enc = collator([{"prefix": f"<SPK_1>sample{CONTEXT_END}", "target": "<SPK_1>new"}])
    assert enc["labels"].tolist() == [[-100, -100, -100, -100, 8, 9]]


# --------------------------------------------------------------------- config


def test_context_preset_turns_on_context_with_more_speaker_tokens_and_same_pool():
    from scripts.speaker_asr.config import (
        context_config,
        load_config,
        n_speaker_tokens,
        pool_signature,
    )

    v1, ctx = load_config(["+experiment=v1"]), load_config(["+experiment=context"])
    assert not v1.context.enabled
    assert ctx.context.enabled
    assert (n_speaker_tokens(v1), n_speaker_tokens(ctx)) == (4, 8)
    assert pool_signature(v1) == pool_signature(ctx)  # one published pool serves both
    assert context_config(load_config()) == ContextConfig()
    assert ctx.hub_model_id != v1.hub_model_id
