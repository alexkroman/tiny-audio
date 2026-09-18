"""Tests for scripts/deploy package.

Focuses on behavior testing rather than existence checks.
"""

from pathlib import Path

import pytest


class TestRunpodCLI:
    """Tests for runpod CLI configuration and utilities."""

    def test_gitignore_aware_file_list_excludes_gitignored_paths(self):
        """File list piped into rsync should honor .gitignore (no __pycache__,
        no .git/, no datasets_cache) and drop the suffix blocklist (no
        .safetensors — Swift bundled weights aren't needed on RunPod)."""
        from scripts.deploy.runpod import _gitignore_aware_file_list
        from scripts.utils import get_project_root

        files = _gitignore_aware_file_list(get_project_root()).splitlines()

        assert files, "expected at least one file from git ls-files"
        assert not any("__pycache__" in f for f in files)
        assert not any(f.startswith(".git/") for f in files)
        assert not any(f.startswith("datasets_cache/") for f in files)
        assert not any(f.endswith(".safetensors") for f in files)
        # Sanity: at least the project's pyproject.toml is in there
        assert "pyproject.toml" in files

    def test_ssh_key_path_is_valid(self):
        """Test that SSH_KEY_PATH points to expected location."""
        from scripts.deploy.runpod import SSH_KEY_PATH

        assert SSH_KEY_PATH is not None
        assert "ssh" in SSH_KEY_PATH.lower()


class TestRunpodConnectionUtils:
    """Tests for runpod connection utilities."""

    def test_get_connection_configures_correctly(self):
        """Test that get_connection returns properly configured Connection."""
        from scripts.deploy.runpod import get_connection

        conn = get_connection("example.com", 22)

        assert conn.host == "example.com"
        assert conn.port == 22
        assert conn.user == "root"

        # Verify SSH key is configured
        key_path = conn.connect_kwargs.get("key_filename", "")
        if isinstance(key_path, list):
            key_path = key_path[0] if key_path else ""
        assert "id_ed25519" in key_path


class TestRemoteDiskPreflight:
    """`ta runpod train` refuses a pod /workspace cannot hold.

    The failure this guards against is expensive and late: ENOSPC arrives
    hours in, as an xet reconstruction error during dataset prep.
    """

    @staticmethod
    def _conn(stdout: str, ok: bool = True):
        from unittest.mock import MagicMock

        conn = MagicMock()
        conn.run.return_value = MagicMock(ok=ok, stdout=stdout)
        return conn

    def test_parses_available_column_from_df(self):
        from scripts.deploy.runpod import _remote_free_gib

        # `df -Pk` columns: Filesystem 1K-blocks Used Available Capacity Mounted
        conn = self._conn("overlay 2113650000 1000000 2097152000 1% /workspace\n")
        free = _remote_free_gib(conn)

        assert free == pytest.approx(2097152000 / 1024**2, rel=1e-6)  # 2000 GiB

    def test_returns_none_when_df_fails(self):
        from scripts.deploy.runpod import _remote_free_gib

        assert _remote_free_gib(self._conn("", ok=False)) is None

    def test_returns_none_on_unparseable_output(self):
        from scripts.deploy.runpod import _remote_free_gib

        assert _remote_free_gib(self._conn("df: /workspace: No such file\n")) is None

    def test_exits_when_pod_is_too_small(self, monkeypatch):
        import typer

        from scripts.deploy import runpod

        monkeypatch.setattr(runpod, "_remote_free_gib", lambda conn, *a, **k: 100.0)
        monkeypatch.setattr(
            "scripts.deploy.plan.build_plan",
            lambda *a, **k: type("P", (), {"disk": {"recommended": 2500.0}})(),
        )

        with pytest.raises(typer.Exit):
            runpod._check_remote_disk(self._conn(""), "stage_1", [])

    def test_proceeds_when_pod_is_big_enough(self, monkeypatch):
        from scripts.deploy import runpod

        monkeypatch.setattr(runpod, "_remote_free_gib", lambda conn, *a, **k: 3000.0)
        monkeypatch.setattr(
            "scripts.deploy.plan.build_plan",
            lambda *a, **k: type("P", (), {"disk": {"recommended": 2500.0}})(),
        )

        # No exception == training is allowed to start.
        runpod._check_remote_disk(self._conn(""), "stage_1", [])

    def test_unreadable_df_does_not_block_training(self, monkeypatch):
        from scripts.deploy import runpod

        monkeypatch.setattr(runpod, "_remote_free_gib", lambda conn, *a, **k: None)

        # A df we cannot read is not evidence the pod is too small; the
        # preflight must not become a new way for `train` to fail.
        runpod._check_remote_disk(self._conn(""), "stage_1", [])

    def test_plan_failure_does_not_block_training(self, monkeypatch):
        from scripts.deploy import runpod

        monkeypatch.setattr(runpod, "_remote_free_gib", lambda conn, *a, **k: 100.0)

        def boom(*a, **k):
            raise RuntimeError("Hub is down")

        monkeypatch.setattr("scripts.deploy.plan.build_plan", boom)

        runpod._check_remote_disk(self._conn(""), "stage_1", [])


class TestDiskPlan:
    """plan.py's disk model must count what actually lands on /workspace."""

    def test_datasets_charged_for_parquet_and_arrow(self):
        from scripts.deploy.plan import DATASET_DISK_FACTOR

        # datasets keeps the Hub parquet download AND the arrow tables it
        # generates from it; measured at 2.05x on librispeech_asr_dummy.
        assert DATASET_DISK_FACTOR > 2.0

    def test_checkpoints_scale_with_trainable_stack_and_retention(self):
        """A joint fine-tune's checkpoints are the decoder + AdamW, times
        save_total_limit -- not one projector-sized file."""
        from scripts.deploy.plan import build_plan

        plan = build_plan(
            "stage_1",
            ["training.save_total_limit=1", "data=librispeech_dummy"],
            512,
        )
        one = next(v for k, v in plan.disk.items() if k.startswith("checkpoints"))

        plan5 = build_plan(
            "stage_1",
            ["training.save_total_limit=5", "data=librispeech_dummy"],
            512,
        )
        five = next(v for k, v in plan5.disk.items() if k.startswith("checkpoints"))

        assert five == pytest.approx(one * 5)
        # stage_1 fine-tunes Qwen3-0.6B, so one checkpoint is GiB-scale.
        assert one > 1.0


class TestBuildTrainingScript:
    """Tests for build_training_script function."""

    @pytest.fixture
    def build_script(self):
        """Get the build_training_script function."""
        from scripts.deploy.runpod import build_training_script

        return build_training_script

    def test_basic_script_structure(self, build_script):
        """Test basic training script has required components."""
        script = build_script(
            experiment="mlp",
            hf_token="test_token",
            wandb_run_id=None,
            wandb_resume=None,
            extra_args=[],
        )

        assert "#!/bin/bash" in script
        assert "mlp" in script
        assert "test_token" in script
        assert "python -m scripts.train" in script

    def test_script_includes_required_env_vars(self, build_script):
        """Test that script includes necessary environment variables."""
        script = build_script(
            experiment="mlp",
            hf_token="token",
            wandb_run_id=None,
            wandb_resume=None,
            extra_args=[],
        )

        assert "TOKENIZERS_PARALLELISM" in script
        assert "HF_TOKEN" in script
        assert "PYTORCH_CUDA_ALLOC_CONF" in script

    def test_script_with_wandb_settings(self, build_script):
        """Test training script includes W&B settings when provided."""
        script = build_script(
            experiment="mosa",
            hf_token="token",
            wandb_run_id="abc123",
            wandb_resume="must",
            extra_args=[],
        )

        assert 'WANDB_RUN_ID="abc123"' in script
        assert 'WANDB_RESUME="must"' in script

    def test_script_with_extra_hydra_args(self, build_script):
        """Test training script includes extra Hydra arguments."""
        script = build_script(
            experiment="mlp",
            hf_token="token",
            wandb_run_id=None,
            wandb_resume=None,
            extra_args=["training.learning_rate=1e-4", "training.batch_size=8"],
        )

        assert "training.learning_rate=1e-4" in script
        assert "training.batch_size=8" in script


class TestHandlerLocal:
    """Tests for local handler testing utilities."""

    def test_find_latest_model_returns_none_for_nonexistent_dir(self, tmp_path: Path):
        """Test find_latest_model returns None when outputs dir doesn't exist."""
        from scripts.deploy.handler_local import find_latest_model

        result = find_latest_model(str(tmp_path / "nonexistent"))
        assert result is None

    def test_find_latest_model_returns_none_for_empty_dir(self, tmp_path: Path):
        """Test find_latest_model returns None when outputs dir is empty."""
        from scripts.deploy.handler_local import find_latest_model

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        result = find_latest_model(str(outputs_dir))
        assert result is None


class TestPackageImports:
    """Tests that all deploy-related packages are properly importable."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "scripts.deploy",
            "scripts.deploy.runpod",
            "scripts.deploy.hf_space",
            "scripts.deploy.handler_local",
            "scripts.hub",
            "scripts.hub.push",
            "scripts.debug",
            "scripts.debug.cli",
        ],
    )
    def test_module_importable(self, module_path):
        """Test that module can be imported without errors."""
        import importlib

        module = importlib.import_module(module_path)
        assert module is not None
