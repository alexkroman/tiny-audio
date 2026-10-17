"""Transcript text cleanup: collapse the repetition loops a decoder can fall into.

A language-model decoder that loses the audio sometimes repeats its last
character, word or phrase until it hits the token limit. These functions trim
such a trailing run back to one copy; ASRPipeline applies them to every output.
"""

import re

MIN_REPEATS = 3
LOOSE_MIN_REPEATS = 6
TRAILING_CHAR_RE = re.compile(rf"(.)\1{{{MIN_REPEATS - 1},}}$")
TRAILING_WORD_RE = re.compile(rf"\b(\w+)(?:\s+\1){{{MIN_REPEATS - 1},}}\s*$", re.IGNORECASE)
NON_WORD_RE = re.compile(r"\W+")


def truncate_repetitions(text: str) -> str:
    """Truncate repeated words/phrases/characters at end of text.

    Detects patterns like:
    - Repeated words: "the the the the" -> "the"
    - Repeated phrases: "i am sorry i am sorry i am sorry" -> "i am sorry"
    - Repeated characters: "444444" -> "4"

    Words match ignoring case. A run of `LOOSE_MIN_REPEATS` also matches
    ignoring punctuation, since the model punctuates inside a loop ("Yeah Yeah
    Yeah. Yeah. Yeah,"); shorter punctuated runs are real speech ("No, no,
    no,"), and collapsing those cost 40 errors over 47 rows of the served eval.

    Args:
        text: Input text to process

    Returns:
        Text with trailing repetitions removed
    """
    if not text:
        return text

    text = TRAILING_CHAR_RE.sub(r"\1", text)
    while TRAILING_WORD_RE.search(text):
        text = TRAILING_WORD_RE.sub(r"\1", text)

    # 3. Truncate repeated phrases (2-20 words) at end
    # e.g., "i am sorry i am sorry i am sorry" -> "i am sorry"
    words = text.split()
    if len(words) < MIN_REPEATS * 2:
        return text

    # Cheap pre-check: trailing window must contain duplicates for any phrase repeat
    # to be possible. set(window) == window means all unique → no repetition.
    window = words[-MIN_REPEATS * 2 :]
    if len(set(window)) == len(window):
        return truncate_punctuated_loop(text)

    for phrase_len in range(2, min(21, len(words) // MIN_REPEATS + 1)):
        phrase_escaped = re.escape(" ".join(words[-phrase_len:]))
        phrase_pattern = re.compile(
            rf"(^|.*?\s)({phrase_escaped})(?:\s+{phrase_escaped}){{{MIN_REPEATS - 1},}}\s*$",
            re.IGNORECASE,
        )
        match = phrase_pattern.match(text)
        if match:
            text = (match.group(1) + match.group(2)).strip()
            break

    return truncate_punctuated_loop(text)


def truncate_punctuated_loop(text: str) -> str:
    """Collapse a trailing 1-20 word phrase run `LOOSE_MIN_REPEATS`+ times, punctuation ignored."""
    words = text.split()
    keys = [NON_WORD_RE.sub("", w).lower() for w in words]
    end = len(words)
    # Shortest phrase first, so "Yeah Yeah. Yeah," stops at "Yeah".
    for n in range(1, min(21, end // LOOSE_MIN_REPEATS + 1)):
        phrase = keys[end - n : end]
        reps = 1
        while end - (reps + 1) * n >= 0 and keys[end - (reps + 1) * n : end - reps * n] == phrase:
            reps += 1
        if reps >= LOOSE_MIN_REPEATS:
            return " ".join(words[: end - (reps - 1) * n])
    return text
