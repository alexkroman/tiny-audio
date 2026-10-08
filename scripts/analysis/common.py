"""Pieces shared by the `ta analysis` commands: run-directory naming, entity matching."""

import functools
import json
import re
from pathlib import Path
from typing import Annotated, TypedDict

import typer
from rich.console import Console

console = Console()

# Written by the (since removed) `ta analysis extract-entities`: spaCy entities
# per reference, keyed on the normalized reference.
KEYWORDS_FILE = Path("outputs/keywords.json")

# Shared help text: every analysis command reads the same run-directory layout,
# so the flags that select runs are spelled and described identically.
MODEL_ARG_HELP = "Model short name to analyze (text after the last '/', matched exactly)"
OUTPUT_DIR_HELP = "Directory containing `ta eval` results"
OutputDirOption = Annotated[
    Path,
    typer.Option("--output-dir", "-o", exists=True, file_okay=False, help=OUTPUT_DIR_HELP),
]
ExcludeOption = Annotated[
    list[str] | None,
    typer.Option("--exclude", help="Model name pattern to exclude (repeatable)"),
]

# OntoNotes' seven numeric labels. The ITN table scores these spans off the raw
# reference, covers classes NER never labels at all (phone numbers, URLs,
# versions), and separates a formatting miss from a recognition miss -- so the
# entity table leaves them out and reports semantic recall only.
ITN_COVERED_ENTITY_TYPES = frozenset(
    {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}
)


class Entity(TypedDict):
    """One spaCy entity in a reference transcript."""

    text: str
    label: str


def extract_dataset_name(dir_name: str) -> str:
    """Dataset from a `{date}_{time}_{model}[_{endpoint}]_{dataset}` run directory."""
    return dir_name.rsplit("_", 1)[-1]


@functools.lru_cache(maxsize=65536)
def normalize_text(text: str) -> str:
    """Normalize text for entity matching (looser than the eval normalizer)."""
    text = text.lower()
    text = text.replace("%", " percent").replace("per cent", "percent")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_in_text(entity_text: str, text: str) -> bool:
    """Check if entity appears in text (normalized comparison).

    A plain substring test. `normalize_text` collapses whitespace on both
    sides, so a contiguous token-run match is already a substring match; a
    token-level subsequence fallback here could never fire.
    """
    return normalize_text(entity_text) in normalize_text(text)


def load_ref_entities() -> dict[str, list[Entity]]:
    """Reference text -> spaCy entities, keyed on the NORMALIZED reference.

    The key is the normalized form because that is what `results.txt` stores
    first and what the extractor keyed on. Empty when there is no keywords file.
    """
    if not KEYWORDS_FILE.exists():
        return {}
    keywords = json.loads(KEYWORDS_FILE.read_text())
    return {ref["text"]: ref["entities"] for ref in keywords["references"]}


def sort_key(value: str) -> float:
    """Extract numeric sort key from a formatted value like '12.34%' or '123' or '-'."""
    if value == "-":
        return float("inf")  # Put missing values at the end
    try:
        return float(value.rstrip("%"))
    except ValueError:
        return float("inf")


def sort_key_desc(value: str) -> tuple[int, float]:
    """Descending-numeric sort key that still pushes missing values last.

    `-sort_key(value)` does not work for descending order: sort_key maps the
    "-" placeholder to +inf so it lands at the end of an *ascending* sort, and
    negating that sends it to the front instead. The leading flag keeps
    missing values last regardless of direction.
    """
    if value == "-":
        return (1, 0.0)
    try:
        return (0, -float(value.rstrip("%")))
    except ValueError:
        return (1, 0.0)
