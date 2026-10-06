"""Tests for the speaker-embedding head: turn bookkeeping, losses, sampler, config."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from scripts.speaker_asr.embedding import (
    ECAPA_DIM,
    EmbeddingConfig,
    MeetingBatchSampler,
    SpeakerHead,
    distill_loss,
    pool_turns,
    supcon_loss,
    turn_metadata,
    turn_spans,
    turn_speakers,
)

SPK = {101, 102, 103}  # <SPK_1..3> ids
END = {200}  # <|im_end|>


def part(pid, speaker, offset, dur=1.0):
    return {"id": pid, "speaker": speaker, "offset_s": offset, "dur_s": dur}


# ----------------------------------------------------------------- turns


def test_turn_speakers_follow_the_target_order_merge_runs_and_skip_wordless_parts():
    parts = [
        part("c", "B", 2.0),
        part("a", "A", 0.0),
        part("b", "A", 1.0),
        part("x", "C", 1.5),
        part("d", "A", 3.0),
    ]
    texts = {"a": "hi", "b": "there", "c": "yes", "d": "ok", "x": ""}  # x has no words
    speakers, ids = turn_speakers(parts, texts)
    assert speakers == ["A", "B", "A"]
    assert ids == [["a", "b"], ["c"], ["d"]]


def test_turn_spans_run_from_each_speaker_token_and_ignore_the_masked_prefix():
    #               prefix (masked)        | target: <S1> w w <S2> w <S1> w <im_end>
    ids = torch.tensor([5, 101, 7, 99, 101, 8, 9, 102, 10, 101, 11, 200])
    labels = torch.tensor([-100, -100, -100, -100, 101, 8, 9, 102, 10, 101, 11, 200])
    assert turn_spans(ids, labels, SPK, END) == [(4, 7), (7, 9), (9, 11)]


def test_turn_metadata_codes_speakers_per_meeting_and_skips_mismatched_rows():
    ids = torch.tensor([[101, 8, 102, 9, 200], [101, 8, 102, 9, 200], [101, 8, 9, 9, 200]])
    labels = ids.clone()
    rows = [
        {"turn_speakers": ["A", "B"], "group": "m1"},
        {"turn_speakers": ["B", "A"], "group": "m1"},  # same people, other window
        {"turn_speakers": ["A", "B"], "group": "m2"},  # 2 turns claimed, 1 speaker token: skipped
    ]
    meta = turn_metadata(ids, labels, SPK, END, rows)
    assert meta["turn_speaker"].tolist() == [0, 1, 1, 0]  # A, B | B, A in meeting m1
    assert meta["turn_group"].tolist() == [0, 0, 0, 0]
    assert meta["turn_index"][2].tolist() == [-1] * 5
    assert meta["turn_index"][0].tolist() == [0, 0, 1, 1, -1]
    assert [k[:2] for k in meta["turn_keys"]] == [
        ("m1", "A"),
        ("m1", "B"),
        ("m1", "B"),
        ("m1", "A"),
    ]
    assert "turn_ecapa" not in meta


def test_turn_metadata_carries_ecapa_targets_when_every_row_has_them():
    ids = torch.tensor([[101, 8, 102, 9, 200]])
    vec = [[1.0] * ECAPA_DIM, [0.5] * ECAPA_DIM]
    meta = turn_metadata(
        ids, ids.clone(), SPK, END, [{"turn_speakers": ["A", "B"], "group": "m", "turn_ecapa": vec}]
    )
    assert meta["turn_ecapa"].shape == (2, ECAPA_DIM)


def test_turn_metadata_is_none_without_usable_rows():
    ids = torch.tensor([[101, 8, 200]])
    assert turn_metadata(ids, ids.clone(), SPK, END, [{"turn_speakers": None}]) is None


def test_pool_turns_averages_each_turns_hidden_states():
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    index = torch.tensor([[0, 0, -1, 1, 1, 1]])
    pooled = pool_turns(hidden, index, 2)
    assert torch.allclose(pooled[0], hidden[0, :2].mean(0))
    assert torch.allclose(pooled[1], hidden[0, 3:].mean(0))


# ----------------------------------------------------------------- losses


def test_supcon_prefers_embeddings_that_separate_speakers():
    torch.manual_seed(0)
    speaker = torch.tensor([0, 0, 1, 1, 2, 2])
    group = torch.zeros(6, dtype=torch.long)
    centers = F.normalize(torch.randn(3, 16), dim=-1)
    good = F.normalize(centers[speaker] + 0.01 * torch.randn(6, 16), dim=-1)
    bad = F.normalize(torch.randn(6, 16), dim=-1)
    assert supcon_loss(good, speaker, group) < supcon_loss(bad, speaker, group)


def test_supcon_draws_negatives_only_from_the_anchors_meeting():
    # Two meetings whose speakers collide across meetings: that must not matter.
    speaker = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    group = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    a, b = (
        F.normalize(torch.tensor([1.0, 0.0]), dim=0),
        F.normalize(torch.tensor([0.0, 1.0]), dim=0),
    )
    emb = torch.stack([a, a, b, b, a, a, b, b])  # meeting 1 reuses meeting 0's directions
    within = supcon_loss(emb, speaker, group, temperature=0.1)
    pooled = supcon_loss(emb, speaker, torch.zeros(8, dtype=torch.long), temperature=0.1)
    assert within < pooled


def test_supcon_is_zero_without_a_positive_and_negative_pair():
    emb = F.normalize(torch.randn(3, 8), dim=-1)
    assert supcon_loss(emb, torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long)) == 0


def test_distill_loss_is_zero_when_aligned():
    t = torch.randn(4, ECAPA_DIM)
    assert distill_loss(t, t) == pytest.approx(0.0, abs=1e-6)
    assert distill_loss(t, -t) == pytest.approx(2.0, abs=1e-6)


def test_head_outputs_unit_vectors_and_an_ecapa_projection_when_distilling():
    head = SpeakerHead(32, dim=8, distill=True)
    z, t = head(torch.randn(5, 32))
    assert z.shape == (5, 8)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert t.shape == (5, ECAPA_DIM)
    assert SpeakerHead(32, dim=8)(torch.randn(2, 32))[1] is None


# ----------------------------------------------------------------- sampler


def test_meeting_sampler_batches_few_meetings_and_covers_every_index_once():
    groups = ["m1"] * 10 + ["m2"] * 10 + ["m3"] * 10 + ["m4"] * 6
    sampler = MeetingBatchSampler(groups, batch_size=8, meetings=2, seed=0)
    order = list(sampler)
    assert sorted(order) == list(range(len(groups)))
    for start in range(0, len(order) - 8, 8):
        assert len({groups[i] for i in order[start : start + 8]}) <= 2
    assert list(sampler) != order  # reshuffled next epoch


# ----------------------------------------------------------------- config


def test_embedding_config_defaults_match_the_base_config():
    from scripts.speaker_asr.config import embedding_config, load_config

    assert embedding_config(load_config()) == EmbeddingConfig()
    assert {f.name for f in fields(EmbeddingConfig)} <= set(load_config().embedding)


def test_embed_preset_warm_starts_from_the_context_checkpoint_with_distillation():
    from scripts.speaker_asr.config import load_config, pool_signature

    context, embed = load_config(["+experiment=context"]), load_config(["+experiment=embed"])
    assert embed.embedding.enabled
    assert not context.embedding.enabled
    assert embed.context.enabled
    assert embed.embedding.distill_weight > 0
    assert embed.model.model_id == context.hub_model_id  # continues the trained labeller
    assert pool_signature(embed) == pool_signature(context)
    assert embed.hub_model_id != context.hub_model_id
