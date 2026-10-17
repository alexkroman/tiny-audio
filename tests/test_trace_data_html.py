"""docs/course/examples/trace_data.py renders a page that is valid, escaped HTML.

The course has students run the script and open the page it writes, so the
generator, not just the committed snapshot of its output, has to pass the
HTML lint. Rendering needs only the payload, not the model.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from docs.course.examples import trace_data

# Text the page shows verbatim: a dataset reference and the nearest vocabulary
# tokens, which include markup-like pieces such as `</` and `<|im_end|>`.
REFERENCE = 'Tom & Jerry said "<b>hi</b>"'
TOKENS = "lacking </ yer <|im_end|> & more"


def _stage(rows: int, cols: int, seed: int) -> trace_data.StageOutput:
    """A random rows x cols activation map with its summary stats."""
    values = np.random.default_rng(seed).standard_normal((rows, cols))
    stats: trace_data.TensorStats = {
        "name": "stage",
        "shape": f"(1, {cols}, {rows})",
        "mean": "0.0",
        "std": "1.0",
        "min": "-3.0",
        "max": "3.0",
    }
    return {"values": values.tolist(), "width": cols, "height": rows, "stats": stats}


@pytest.fixture(scope="module")
def page() -> str:
    """The HTML the script writes for a small synthetic trace."""
    rng = np.random.default_rng(0)
    projector = _stage(80, 12, seed=2)
    payload: trace_data.TracePayload = {
        "reference_text": REFERENCE,
        "waveform": rng.standard_normal(1600).tolist(),
        "sampling_rate": 16000,
        "audio_duration": 1.0,
        "spectrogram": {"values": rng.random((16, 40)).tolist(), "width": 40, "height": 16},
        "encoder_output": _stage(80, 12, seed=1),
        "projector_output": {**projector, "nearest_tokens": TOKENS},
    }
    return trace_data.generate_observable_html(payload)


def test_page_starts_with_the_doctype(page: str) -> None:
    assert page.startswith("<!DOCTYPE html>")


def test_model_and_dataset_text_is_escaped(page: str) -> None:
    assert "Tom &amp; Jerry said &quot;&lt;b&gt;hi&lt;/b&gt;&quot;" in page
    assert "lacking &lt;/ yer &lt;|im_end|&gt; &amp; more" in page
    assert "<b>hi</b>" not in page
    assert "<|im_end|>" not in page


def test_page_passes_the_html_lint(page: str, tmp_path: Path) -> None:
    html_file = tmp_path / "data_trace.html"
    html_file.write_text(page)
    lint = subprocess.run(
        [sys.executable, "-m", "djlint", str(html_file), "--lint", "--profile=html"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert lint.returncode == 0, lint.stdout
