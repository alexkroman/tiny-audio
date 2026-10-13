"""Tests for scripts.eval.build_ami_ma: AMI word files -> MA-tool references."""

import io
import zipfile

from scripts.eval.build_ami_ma import MEETINGS, parse_words, reference_text


def _annotations(files: dict[str, str]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, xml in files.items():
            zf.writestr(name, xml)
    return zipfile.ZipFile(buf)


def _w(text: str, start: float, **attrs: str) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f'<w starttime="{start}" endtime="{start + 0.2}"{extra}>{text}</w>'


A = (
    "<nite:root xmlns:nite='http://nite.sourceforge.net/'>"
    + _w("Um", 1.0)
    + _w(",", 1.2, punc="true")
    + _w("hello", 1.3)
    + "<w>untimed</w>"
    + _w("again", 9.0)
    + "<vocalsound/></nite:root>"
)
B = "<nite:root xmlns:nite='http://nite.sourceforge.net/'>" + _w("Yeah", 2.0) + "</nite:root>"


def test_parse_words_keeps_what_the_ma_tool_keeps() -> None:
    ann = _annotations({"words/X.A.words.xml": A, "words/X.B.words.xml": B})
    words = parse_words(ann, "X")
    # Punctuation, untimed words and non-word elements are dropped; speakers in letter order.
    assert [(w["speaker"], w["text"]) for w in words] == [
        ("A", "Um"),
        ("A", "hello"),
        ("A", "again"),
        ("B", "Yeah"),
    ]


def test_reference_interleaves_speakers_by_time() -> None:
    ann = _annotations({"words/X.A.words.xml": A, "words/X.B.words.xml": B})
    assert reference_text(parse_words(ann, "X")) == "<SPK_1>Um hello<SPK_2>Yeah<SPK_1>again"


def test_meetings_are_the_ma_tool_set() -> None:
    assert len(MEETINGS) == len(set(MEETINGS)) == 24
    assert {m[:6] for m in MEETINGS} == {"EN2002", "ES2004", "ES2014", "IS1009", "TS3003", "TS3007"}
