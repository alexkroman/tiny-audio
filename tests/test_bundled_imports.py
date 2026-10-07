"""Every file shipped with the checkpoint must load with only the model's requirements.

transformers scans each bundled module's imports (including ones inside
functions, but not inside `try`) before loading the model, and refuses to load
if any is missing. A function-level `import speechbrain` in diarization.py once
took the Space down with that error even though nothing called it.
"""

import sys
from pathlib import Path

import pytest
from transformers.dynamic_module_utils import get_imports

BUNDLED = sorted(p for p in Path("tiny_audio").glob("*.py") if p.name.startswith("asr_")) + [
    Path("tiny_audio") / n for n in ("projectors.py", "alignment.py", "diarization.py")
]
# requirements.txt: transformers, torch, torchaudio, peft (+ their deps safetensors, numpy).
ALLOWED = {"transformers", "torch", "torchaudio", "peft", "safetensors", "numpy"}


@pytest.mark.parametrize("path", BUNDLED, ids=lambda p: p.name)
def test_bundled_file_needs_only_model_requirements(path):
    third_party = {m for m in get_imports(str(path)) if m not in sys.stdlib_module_names}
    assert third_party <= ALLOWED, f"{path.name} needs {sorted(third_party - ALLOWED)}"
