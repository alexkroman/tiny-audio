"""Serve every asset the test suite would download from local fixtures instead.

Importing this module points huggingface_hub at a seeded cache with offline
mode on, and NLTK at a vendored copy of the `punkt_tab` English model that
`truecase` needs (scripts/labels.py otherwise fetches it from GitHub at
import). The tests then run with no network at all -- Claude Code on the web,
sandboxed CI -- and never download ~400 MB on a fresh machine.

The cache is built from `fixtures/hf_hub/<org>--<name>/`, which holds each
repo's real configs and tokenizer files exactly as the Hub serves them. Weight
files are too big to vendor, so the repos in `_RANDOM_WEIGHTS` get a
randomly initialised checkpoint of the real architecture and shape. No test
depends on pretrained quality -- only on shapes, vocab, and chat template.

huggingface_hub and transformers read HF_HUB_CACHE / HF_HUB_OFFLINE, and NLTK
reads NLTK_DATA, once at import, so this must be imported before any of them
(conftest imports it first).
To add a repo, drop its small files into `fixtures/hf_hub/` and, if a test
instantiates its weights, list its architecture in `_RANDOM_WEIGHTS`. The cache
key hashes the fixtures, so edits rebuild it automatically.
"""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures" / "hf_hub"
_NLTK_DATA = Path(__file__).parent / "fixtures" / "nltk_data"

# Repos whose weights a test instantiates -> the class their real checkpoint
# was saved from, so key names match what the loaders expect.
_RANDOM_WEIGHTS = {
    "openai--whisper-tiny": "WhisperForConditionalGeneration",
    "HuggingFaceTB--SmolLM2-135M-Instruct": "LlamaForCausalLM",
}

# Any 40-hex string: offline resolution maps refs/main to snapshots/<hash>.
_REVISION = "0" * 40

# Bump to force a rebuild when the generation logic below changes.
_BUILD_VERSION = b"1"


def _cache_key() -> str:
    digest = hashlib.sha256(_BUILD_VERSION)
    for path in sorted(_FIXTURES.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(_FIXTURES)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _build(staging: Path) -> None:
    # Deferred: transformers imports huggingface_hub, which must not load
    # until the environment below is set.
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    for repo in _FIXTURES.iterdir():
        root = staging / f"models--{repo.name}"
        shutil.copytree(repo, root / "snapshots" / _REVISION)
        (root / "refs").mkdir()
        (root / "refs" / "main").write_text(_REVISION)

    torch.manual_seed(0)
    for name, arch in _RANDOM_WEIGHTS.items():
        snapshot = staging / f"models--{name}" / "snapshots" / _REVISION
        config = transformers.AutoConfig.from_pretrained(snapshot)
        model = getattr(transformers, arch)(config).to(config.dtype or torch.float32)
        # save_pretrained also rewrites config.json; keep only the weights so
        # the snapshot's config stays byte-identical to the Hub's.
        with tempfile.TemporaryDirectory() as scratch:
            model.save_pretrained(scratch)
            shutil.move(Path(scratch) / "model.safetensors", snapshot / "model.safetensors")


def _seed(cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=cache.parent))
    try:
        _build(staging)
        staging.rename(cache)
    except OSError:
        # A concurrent session finished first; its cache is equivalent.
        if not cache.is_dir():
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


assert not {"huggingface_hub", "nltk"} & sys.modules.keys(), (
    "tests/offline_assets.py must be imported before huggingface_hub and nltk, "
    "which read their cache locations only at import time."
)
os.environ["NLTK_DATA"] = str(_NLTK_DATA)
_CACHE = Path(tempfile.gettempdir()) / "tiny-audio-test-hub" / _cache_key()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = str(_CACHE)
if not _CACHE.is_dir():
    _seed(_CACHE)
