"""Token-level helpers in scripts.itn."""

from scripts.itn import _loose_tokens, contains_subsequence


def test_loose_tokens_strip_formatting():
    assert _loose_tokens("Call $1,250.00 at 5:30 p.m.!") == ["call", "125000", "at", "530", "pm"]


def test_loose_tokens_drop_punctuation_only_tokens():
    assert _loose_tokens("wait -- what ...") == ["wait", "what"]


def test_empty_needle_never_matches():
    assert contains_subsequence(["a", "b"], []) is False
    assert contains_subsequence([], []) is False


def test_subsequence_must_be_contiguous():
    assert contains_subsequence(["a", "b", "c"], ["b", "c"]) is True
    assert contains_subsequence(["a", "b", "c"], ["a", "c"]) is False


def test_needle_longer_than_haystack():
    assert contains_subsequence(["a"], ["a", "b"]) is False
