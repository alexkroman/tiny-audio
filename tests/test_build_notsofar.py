"""Tests for scripts.eval.build_notsofar: NOTSOFAR transcripts -> references."""

from scripts.eval.build_notsofar import clean_text, meeting_devices, reference_text
from scripts.eval.datasets import DATASET_REGISTRY


def test_clean_text_drops_tags_and_keeps_names_and_fillers() -> None:
    raw = "<ST/> Um, hi <PName>Bob</PName> <FILL/> <UNKNOWN/> there"
    assert clean_text(raw) == "Um, hi Bob there"


def test_reference_sorts_segments_and_merges_turns() -> None:
    transcript = [
        {"speaker_id": "Ron", "start_time": 5.0, "end_time": 6.0, "text": "later"},
        {"speaker_id": "Ann", "start_time": 1.0, "end_time": 2.0, "text": "hello <ST/>"},
        {"speaker_id": "Ann", "start_time": 2.5, "end_time": 3.0, "text": "again"},
        {"speaker_id": "Ron", "start_time": 4.0, "end_time": 4.5, "text": "<FILL/>"},
    ]
    assert reference_text(transcript) == "<SPK_1>hello again<SPK_2>later"


def test_meeting_devices_picks_matching_single_channel_files() -> None:
    root = "benchmark-datasets/eval_set/v/MTG/"
    files = [
        f"{root}MTG_2/sc_rockfall_0/ch0.wav",
        f"{root}MTG_1/sc_plaza_0/ch0.wav",
        f"{root}MTG_1/sc_meetup_0/ch0.wav",
        f"{root}MTG_1/mc_plaza_0/ch0.wav",
        f"{root}MTG_1/close_talk/CT_1.wav",
        f"{root}MTG_1/gt_transcription.json",
    ]
    assert meeting_devices(files, root, "sc_") == {
        "MTG_1": [f"{root}MTG_1/sc_meetup_0/ch0.wav", f"{root}MTG_1/sc_plaza_0/ch0.wav"],
        "MTG_2": [f"{root}MTG_2/sc_rockfall_0/ch0.wav"],
    }


def test_notsofar_test_is_registered_for_verbatim_speaker_scoring() -> None:
    cfg = DATASET_REGISTRY["notsofar-test"]
    assert cfg.path == "mazesmazes/notsofar-test"
    assert cfg.speakers
    assert cfg.verbatim_scoring
