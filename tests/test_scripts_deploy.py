"""Tests for scripts/deploy package.

Focuses on behavior testing rather than existence checks.
"""

from pathlib import Path

import pytest


class TestRunpodCLI:
    """Tests for runpod CLI configuration and utilities."""

    def test_rsync_excludes_contains_expected_patterns(self):
        """Test that RSYNC_EXCLUDES contains necessary patterns."""
        from scripts.deploy.runpod import RSYNC_EXCLUDES

        assert isinstance(RSYNC_EXCLUDES, list)
        assert "__pycache__" in RSYNC_EXCLUDES
        assert ".git" in RSYNC_EXCLUDES

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
            "scripts.debug.check_mosa",
        ],
    )
    def test_module_importable(self, module_path):
        """Test that module can be imported without errors."""
        import importlib

        module = importlib.import_module(module_path)
        assert module is not None
