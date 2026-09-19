"""Re-pinning the causal-conv1d Hub kernel to a revision built for this torch.

transformers pins kernels-community/mamba-ssm at `version=2`, whose build set
starts at torch 2.11. RunPod images run torch 2.8, so dispatch resolves the
repo and then raises FileNotFoundError, and the run silently drops to the
torch reference implementation for Qwen3.5's causal depthwise conv. Released
tags v0.0.2-v0.0.4 all carry torch28-x86_64 builds; these tests pin the
selection logic and every bail-out path.
"""

from unittest.mock import MagicMock, patch

from scripts.train import _kernel_revision_for, _repin_causal_conv1d_kernel

# Mirrors the real repo layout as of writing.
_FILES = {
    "v0.0.4": [
        "build/torch28-cxx11-cu126-x86_64-linux/k.so",
        "build/torch29-cxx11-cu126-x86_64-linux/k.so",
    ],
    "v0.0.3": [
        "build/torch27-cxx11-cu126-x86_64-linux/k.so",
        "build/torch28-cxx11-cu126-aarch64-linux/k.so",
    ],
    "v0.0.2": ["build/torch25-cxx11-cu126-x86_64-linux/k.so"],
    "main": [
        "build/torch211-cxx11-cu128-x86_64-linux/k.so",
        "build/torch28-cxx11-cu128-x86_64-linux/k.so",
    ],
}


def _api(files=None):
    api = MagicMock()
    api.list_repo_files.side_effect = lambda repo, revision: (files or _FILES)[revision]
    return api


class TestRevisionSelection:
    def test_prefers_the_newest_released_tag_that_matches(self):
        with patch("huggingface_hub.HfApi", return_value=_api()):
            assert _kernel_revision_for("torch28", "x86_64") == "v0.0.4"

    def test_falls_through_to_main_for_a_newer_torch(self):
        with patch("huggingface_hub.HfApi", return_value=_api()):
            assert _kernel_revision_for("torch211", "x86_64") == "main"

    def test_matches_on_cpu_architecture_too(self):
        """v0.0.4 has torch28 only for x86_64, so aarch64 must skip it."""
        with patch("huggingface_hub.HfApi", return_value=_api()):
            assert _kernel_revision_for("torch28", "aarch64") == "v0.0.3"

    def test_returns_none_when_nothing_matches(self):
        with patch("huggingface_hub.HfApi", return_value=_api()):
            assert _kernel_revision_for("torch35", "x86_64") is None

    def test_a_revision_that_fails_to_list_is_skipped(self):
        api = MagicMock()

        def side_effect(repo, revision):
            if revision == "v0.0.4":
                raise OSError("offline")
            return _FILES[revision]

        api.list_repo_files.side_effect = side_effect
        with patch("huggingface_hub.HfApi", return_value=api):
            assert _kernel_revision_for("torch28", "x86_64") == "main"


class TestRepinIsBestEffort:
    def test_noop_without_cuda(self):
        """mac / CPU: nothing dispatches from the Hub, and HfApi is never hit."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("huggingface_hub.HfApi") as api,
        ):
            _repin_causal_conv1d_kernel()
        api.assert_not_called()

    def test_survives_a_hub_failure(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("scripts.train._kernel_revision_for", side_effect=RuntimeError("boom")),
        ):
            _repin_causal_conv1d_kernel()  # must not raise
