"""Tests for _normalize_label - the per-sample label-normalization helper
applied at training time in DataCollator._build_sample.

After the Ultravox-style normalizer rewrite the pipeline is:
  ftfy + NFKC → Gigaspeech punct-tag → marker strip → TEDLIUM bracket strip →
  per-cent spelling canon → whitespace collapse → conditional truecase

Tests below capture both the marker-stripping correctness (deterministic)
and the truecase output (which depends on the truecase library's NLTK-backed
vocab). The truecase-dependent expected values were captured from the live
normalizer on 2026-05-13; if the truecase library updates and these expected
values shift, update with `python -c "from scripts.train import _normalize_label; print(_normalize_label(...))"`.
"""

import pytest

from scripts.train import _has_edge_content_tag, _normalize_label


class TestPercentCanonicalization:
    """The `%` character is PRESERVED. A prior revision rewrote it to
    " percent", which destroyed `%` in 100% of training targets and was the
    measured cause of ~93% of a 66%-vs-94% raw-text ITN gap. Removing it is
    WER-neutral: Whisper's EnglishTextNormalizer maps "105 percent" and
    "105%" to the same string on both reference and hypothesis.
    """

    def test_percent_symbol_is_preserved(self):
        # Lowercase input → truecase fires → sentence-initial cap.
        assert _normalize_label("we grew 25%") == "We grew 25%"

    def test_percent_symbol_mid_sentence(self):
        # Truecase capitalizes acronym-shaped tokens like "q2" → "Q2".
        assert _normalize_label("a 25% margin in q2") == "A 25% margin in Q2"

    def test_decimal_percent(self):
        assert _normalize_label("decreasing 0.4% quarter") == "Decreasing 0.4% quarter"

    def test_per_cent_two_word_form(self):
        assert _normalize_label("we grew 25 per cent") == "We grew 25 percent"

    def test_already_spoken_form_unchanged_except_truecase(self):
        # Same percent canon path; lowercase input gets sentence-initial cap.
        assert _normalize_label("we grew 25 percent") == "We grew 25 percent"


class TestGigaspeechPunctRestoration:
    """Gigaspeech ships punct as <PERIOD>/<COMMA>/<QUESTIONMARK>/<EXCLAMATIONPOINT>.
    The normalizer maps each to real punctuation BEFORE the truecase pass.
    """

    def test_period_at_end(self):
        assert _normalize_label("BOOST PIPES <PERIOD>") == "Boost pipes."

    def test_comma_mid_sentence(self):
        assert _normalize_label("PIPES <COMMA> AND THE FLOWERS") == "Pipes, and the flowers"

    def test_multiple_punct_tags(self):
        assert _normalize_label("HELLO <COMMA> WORLD <PERIOD>") == "Hello, world."

    def test_question_tag(self):
        assert _normalize_label("REALLY <QUESTIONMARK>") == "Really?"

    def test_exclamation_tag(self):
        # Short input (<5 letters total after punct) skips truecase.
        assert _normalize_label("WOW <EXCLAMATIONPOINT>") == "WOW!"


class TestNonSpeechTagStripping:
    """Gigaspeech non-speech segment tags (<SIL>/<MUSIC>/<NOISE>/<OTHER>) are
    stripped — not whole-sample dropped. The empty-label filter at the
    collator catches the entire-label-was-just-a-tag edge case.
    """

    def test_music_stripped(self):
        assert _normalize_label("hello <music> world") == "Hello world"

    def test_noise_at_start(self):
        assert _normalize_label("<noise> hello") == "Hello"

    def test_other_at_end(self):
        assert _normalize_label("hello <other>") == "Hello"

    def test_sil_stripped(self):
        assert _normalize_label("hello <sil> world") == "Hello world"

    def test_tag_alone_becomes_empty(self):
        # Tag-only label → strip leaves nothing → empty string returned.
        # Collator's empty-label filter then drops the row.
        assert _normalize_label("<music>") == ""
        assert _normalize_label("<NOISE>") == ""


class TestGigaspeechEdgeCases:
    def test_punct_tag_at_start_no_leading_space(self):
        # `\s*` in the regex matches zero-width at string start.
        # Resulting `.Boost` is unusual but a literal consequence of the
        # mapping rule; downstream WER scoring normalizes regardless.
        assert _normalize_label("<period>boost") == ".Boost"

    def test_unknown_angle_bracket_token_stripped(self):
        # ASR transcripts never legitimately contain `<word>` tokens, so
        # ALL residual `<...>` (after the Gigaspeech punct map runs) are
        # treated as annotation markers and stripped. Prior whitelist
        # approach silently leaked novel marker variants into training
        # labels, teaching the decoder to emit them as literal tokens.
        assert _normalize_label("<foo> hello") == "Hello"


class TestCombinedNormalization:
    def test_percent_and_gigaspeech_marker(self):
        # Now that `%` survives, no lowercase "percent" is appended, so the
        # text stays ALL-CAPS → upper_frac > 0.9 → truecase FIRES. Under the
        # old rewrite this landed in the mixed-case band and truecase was
        # skipped, yielding "WE GREW 25 percent.".
        assert _normalize_label("WE GREW 25% <PERIOD>") == "We grew 25%."

    def test_marker_then_percent(self):
        assert _normalize_label("HELLO <COMMA> WE GREW 25%") == "Hello, we grew 25%"


class TestHygiene:
    def test_strips_leading_trailing_whitespace(self):
        assert _normalize_label("   hello world   ") == "Hello world"

    def test_collapses_internal_whitespace(self):
        assert _normalize_label("hello    world") == "Hello world"

    def test_empty_input(self):
        assert _normalize_label("") == ""

    def test_only_whitespace(self):
        assert _normalize_label("   ") == ""

    def test_only_marker_remains_after_strip(self):
        # <period> → '.' via Gigaspeech-tag map; whole label becomes just '.'.
        assert _normalize_label("<period>") == "."

    def test_already_cased_passes_through(self):
        # Mixed case (sentence-initial caps + proper nouns) → upper_count > 0
        # → truecase SKIPPED → output preserves existing casing.
        assert (
            _normalize_label("My, what imaginations these children have developed!")
            == "My, what imaginations these children have developed!"
        )


class TestTedliumNormalization:
    """TEDLIUM ships <unk> in 60.8% of train rows (measured over 800 streamed
    rows, 2026-09-18) and [...] editorial brackets in ~0.25%. Both stripped;
    surrounding lowercase prose gets truecased.

    NOTE: _normalize_label still strips <unk> wherever it appears, and these
    tests pin that. Rows whose label STARTS or ENDS with <unk> are dropped
    upstream by the collator (see TestEdgeContentTagFilter and
    _has_edge_content_tag), because stripping an edge <unk> yields a target
    missing its first or last spoken word while the audio retains it. So in
    practice only the mid-sentence case below survives into training.
    """

    def test_unk_at_start(self):
        assert (
            _normalize_label("<unk> i thought i would read poems") == "I thought I would read poems"
        )

    def test_unk_mid_sentence(self):
        assert _normalize_label("hello <unk> world") == "Hello world"

    def test_unk_at_end(self):
        assert (
            _normalize_label("washing my mouth out with soap <unk>")
            == "Washing my mouth out with soap"
        )

    def test_bracket_block_stripped(self):
        assert _normalize_label("she said [ medicine ] and laughed") == "She said and laughed"

    def test_long_bracket_block_stripped(self):
        assert (
            _normalize_label("then [ her face and hands stood out ] she paused")
            == "Then she paused"
        )

    def test_unk_and_bracket_combined(self):
        assert _normalize_label("<unk> hello [ aside ] world") == "Hello world"


class TestEdgeContentTagFilter:
    """<unk>/<foreign>/<overlap> stand in for spoken words the audio still
    contains. At the START of a label, stripping them supervises onset
    truncation, which was the measured root cause of this recipe's Peoples
    Speech regression. The collator drops those rows.

    LEADING ONLY as of 2026-09-20. Trailing tags are kept: the supporting
    measurement was a leading-word-deletion prior (>=1 dropped leading
    reference word on 52/100 Peoples samples), the trailing half was never
    measured on its own, and together they were cutting 41.2% of TEDLIUM --
    the one corpus where the decoder beats the frozen encoder.

    Non-speech tags (<noise>/<music>/<sil>/<laugh>/<breath>) are NOT
    content-bearing — no word was uttered — so they must not trigger the drop.
    Medial content tags also must not trigger it: dropping every <unk> row
    would remove 60.8% of TEDLIUM, the one dataset where the decoder
    measurably beats the frozen encoder.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "<unk> i thought i would read poems",
            "<unk> called dirt",
            "<UNK> case insensitive",
            "  <unk> leading whitespace before tag",
            "<foreign> hola there",
            "<unk> both ends <unk>",
        ],
    )
    def test_edge_content_tag_is_dropped(self, text):
        assert _has_edge_content_tag(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Trailing content tags are KEPT as of 2026-09-20 (leading-only
            # filter). The drop rule's evidence was a *leading*-deletion prior
            # measured on the Peoples eval; the trailing half was never
            # measured separately and cost TEDLIUM 13.9% of its rows.
            "health <unk>",
            "washing my mouth out with soap <unk>",
            "trailing whitespace after tag <unk>   ",
            "and then <overlap>",
            "hello <unk> world",  # medial — kept
            "she said <foreign> in reply",  # medial — kept
            "<noise> hello",  # non-speech at edge — kept
            "hello <music>",  # non-speech at edge — kept
            "<sil> quiet then speech",
            "laughing <laugh> loudly",
            "she said [ medicine ] and laughed",  # bracket, not a tag
            "plain text with no tags",
            "",
        ],
    )
    def test_kept(self, text):
        assert _has_edge_content_tag(text) is False

    def test_none_is_safe(self):
        assert _has_edge_content_tag(None) is False


class TestEdAccNormalization:
    """EdAcc ships <overlap>/<laugh>/<dtmf>/<foreign>/<no-speech>/<lipsmack>
    in ~20% of rows. All stripped; surrounding text gets recased per truecase.
    """

    def test_overlap_marker_stripped(self):
        # Truecase capitalizes "Promo" (recognized as a proper-noun-ish token).
        assert (
            _normalize_label("YOU'RE A BIG PROMO <OVERLAP> YOU'RE THE BIG PROMOTER")
            == "You're a big Promo you're the big promoter"
        )

    def test_laugh_marker_stripped(self):
        # Mid-sentence "Yeah" cap is a known truecase library quirk
        # (sentence-boundary heuristic over-capitalizes interjections).
        assert (
            _normalize_label("ANALYZING THIS CONVERSATION BUT ANYWAY <LAUGH> YEAH")
            == "Analyzing this conversation but anyway Yeah"
        )

    def test_dtmf_marker_stripped(self):
        assert (
            _normalize_label("EVERYBODY IS GOING THERE AND <DTMF> A LITTLE BIT GRIM")
            == "Everybody is going there and a little bit grim"
        )

    def test_foreign_marker_stripped(self):
        assert _normalize_label("HE SAID <FOREIGN> AND LAUGHED") == "He said and laughed"

    def test_no_speech_marker_stripped(self):
        # Hyphenated tag form must be caught literally.
        assert _normalize_label("OKAY <NO-SPEECH> RIGHT") == "Okay right"

    def test_lipsmack_marker_stripped(self):
        # Short total letter count (< 5 after strip) → truecase SKIPPED.
        assert _normalize_label("UM <LIPSMACK> SO") == "UM SO"

    @pytest.mark.parametrize(
        "marker",
        ["overlap", "laugh", "dtmf", "foreign", "no-speech", "lipsmack"],
    )
    def test_lowercase_form_also_stripped(self, marker):
        # Generic `<[^>]+>` strip is case-agnostic — both cases handled.
        assert _normalize_label(f"hello <{marker}> world") == "Hello world"


class TestEarnings22Normalization:
    """Earnings22 ships <clear_throat>/<inaudible>/<crosstalk> in ~3% of rows."""

    def test_clear_throat_marker_stripped(self):
        assert _normalize_label("um <clear_throat> as i was saying") == "Um as I was saying"

    def test_inaudible_marker_stripped(self):
        # "q4" → "Q4" by truecase (recognized acronym shape).
        assert _normalize_label("the revenue <inaudible> in q4") == "The revenue in Q4"

    def test_crosstalk_marker_stripped(self):
        assert _normalize_label("yeah <crosstalk> i agree") == "Yeah I agree"


class TestBodilyNoiseMarkers:
    """Bodily-noise annotation tags surface across corpora (sigh, inhale,
    cough, etc.). Probe of the multiasr mix surfaced these surviving the
    prior whitelist-only stripper. Generic `<[^>]+>` strip catches them
    along with any novel marker a future corpus introduces."""

    @pytest.mark.parametrize(
        "marker",
        ["sigh", "inhale", "exhale", "breath", "cough", "throat", "sniff", "click"],
    )
    def test_bodily_noise_marker_stripped(self, marker):
        assert _normalize_label(f"hello <{marker}> world") == "Hello world"


class TestPerCentBoundary:
    """`per cent` → `percent` must respect word boundaries; prior unbounded
    `text.replace("per cent", "percent")` mangled `per centage` → `percentage`
    and would have mangled `per centimeter` → `percentimeter`."""

    def test_per_centage_preserved(self):
        # Lowercase mono-case → truecase fires → sentence-initial cap, and
        # may also cap "centage" as a perceived proper-noun (truecase
        # library artifact, unrelated to the regex fix). The contract this
        # test enforces is: `per centage` does NOT collapse to `percentage`.
        result = _normalize_label("the per centage was high").lower()
        assert "percentage" not in result
        assert "per centage" in result

    def test_per_centimeter_preserved(self):
        result = _normalize_label("five per centimeter").lower()
        assert "percentimeter" not in result
        assert "per centimeter" in result

    def test_per_cent_still_collapsed_when_word_bounded(self):
        assert _normalize_label("five per cent here") == "Five percent here"


class TestTruecaseArtifactCleanup:
    """Truecase's NLTK tokenizer introduces three classes of artifact in
    its output. The post-truecase cleanup function fixes each."""

    def test_mid_sentence_period_no_leading_space(self):
        # Truecase output `rate . But` → cleaned to `rate. But`. The
        # `<PERIOD>` substitution happens in ~25% of Gigaspeech rows.
        assert (
            _normalize_label("USE A RATE <PERIOD> BUT TODAY IT WORKS")
            == "Use a rate. But today it works"
        )

    def test_sentence_start_after_period_capitalized(self):
        # Truecase may leave the next sentence lowercase after a
        # mid-sentence period (`E T. the Video game`). Post-cleanup caps it.
        assert _normalize_label("E T <PERIOD> THE VIDEO GAME <PERIOD>") == "E T. The Video game."

    def test_em_dash_spaces_restored(self):
        # Truecase collapses ` -- ` → `--`. Post-cleanup restores spacing.
        # Input is mono-case lowercase so truecase fires; the cleanup
        # then re-inserts the em-dash spaces.
        assert _normalize_label("we agreed -- it was fine") == "We agreed -- it was fine"

    def test_gonna_artifact_normalized(self):
        # Truecase mangles `GONNA`/`gonna` → `gonNA` regardless of input case.
        assert _normalize_label("I'M GONNA DO IT NOW") == "I'm gonna do it now"

    def test_wanna_artifact_normalized(self):
        assert _normalize_label("you wanna go home") == "You wanna go home"

    def test_gotta_artifact_normalized(self):
        # Same MidWord-caps family as gonna/wanna — truecase outputs `gotTA`.
        assert _normalize_label("YOU GOTTA DO IT NOW") == "You gotta do it now"


class TestOrphanedNtContraction:
    """TEDLIUM tokenizes ~10% of negation contractions with the apostrophe-t
    split off the verb stem (`didn 't` instead of `didn't`). Truecase then
    treats `'t` as a standalone token and uppercases it, producing
    `didn 'T embrace`. Pre-collapse fixes this before truecase runs."""

    @pytest.mark.parametrize(
        "stem",
        [
            "didn",
            "don",
            "wouldn",
            "wasn",
            "doesn",
            "isn",
            "aren",
            "shouldn",
            "couldn",
            "hasn",
            "haven",
            "hadn",
            "won",
            "weren",
        ],
    )
    def test_orphan_nt_joined(self, stem):
        assert _normalize_label(f"i {stem} 't think so").lower().startswith(f"i {stem}'t")

    def test_does_not_break_correct_form(self):
        # Already-joined `didn't` (no space) must be untouched.
        result = _normalize_label("i didn't think so")
        assert "didn't" in result
        assert "didn 't" not in result
        assert "didn 'T" not in result

    def test_does_not_break_alternate_tokenization(self):
        # TEDLIUM's other tokenization style — `did n't` — joins correctly
        # via truecase's existing contraction vocabulary. Our regex must
        # not interfere.
        result = _normalize_label("they did n't think so")
        assert "didn't" in result

    def test_does_not_match_apostrophe_followed_by_letters(self):
        # `'tis` / `'twas` (archaic) — apostrophe followed by letters that
        # aren't a contraction suffix. Our regex requires `\w+n` before the
        # space; `hark` ends in `k`, so the orphan-n't fix does NOT fire.
        # (Truecase may still upper-case the post-apostrophe letter, but
        # that's pre-existing behavior independent of this fix.)
        result = _normalize_label("hark 'tis the night")
        assert "'tis" in result.lower()  # case-insensitive: regex didn't mangle it

    def test_no_change_when_preceding_word_doesnt_end_in_n(self):
        # The `\w+n` anchor restricts our fix to negation contractions.
        # Forms like `friends ' mothers` (plural possessive) and `it 's`
        # (which truecase handles) stay on the existing code path.
        result = _normalize_label("my friends 's car broke")
        # Verify no n't-style mangling crept in
        assert " 't" not in result.lower()


class TestAdjacentNoWhitespaceMarkers:
    """Markers without surrounding whitespace must not collapse adjacent
    words. The substitute-with-space approach (vs. substitute-with-empty)
    keeps `hello<unk>world` from becoming `helloworld`."""

    def test_angle_tag_no_whitespace(self):
        assert _normalize_label("hello<unk>world") == "Hello world"

    def test_square_bracket_no_whitespace(self):
        assert _normalize_label("abc[laughter]def") == "Abc def"
