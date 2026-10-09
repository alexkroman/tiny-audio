"""TensorNoRepeatNGram bans exactly what transformers' NoRepeatNGramLogitsProcessor bans."""

import pytest
import torch
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor

from tiny_audio.asr_modeling import TensorNoRepeatNGram

VOCAB = 50


def looping_ids(batch: int, length: int, n: int, seed: int) -> torch.LongTensor:
    """Random rows, some ending mid-way through a repeat of an earlier n-gram."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (batch, length), generator=gen)
    for row in range(0, batch, 2):  # every other row: last n-1 ids copy an earlier span
        start = int(torch.randint(0, length - 2 * n, (1,), generator=gen))
        ids[row, length - n + 1 :] = ids[row, start : start + n - 1]
    ids[1, :5] = 0  # left padding, as in a ragged batch
    return torch.LongTensor(ids)


@pytest.mark.parametrize("n", [2, 3, 12])
@pytest.mark.parametrize("length", [5, 40, 270])
def test_matches_transformers(n: int, length: int) -> None:
    if length < 2 * n + 1:
        ids = torch.LongTensor(torch.randint(0, VOCAB, (4, length)))
    else:
        ids = looping_ids(6, length, n, seed=length * n)
    scores = torch.FloatTensor(torch.randn(ids.shape[0], VOCAB))
    expected = NoRepeatNGramLogitsProcessor(n)(ids, torch.FloatTensor(scores.clone()))
    actual = TensorNoRepeatNGram(n)(ids, torch.FloatTensor(scores.clone()))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    assert torch.equal(actual, expected)


def test_bans_the_token_that_would_repeat() -> None:
    ids = torch.LongTensor([[7, 8, 9, 1, 2, 7, 8]])
    out = TensorNoRepeatNGram(3)(ids, torch.FloatTensor(torch.zeros(1, 10)))
    assert torch.isinf(out[0]).nonzero().flatten().tolist() == [9]
