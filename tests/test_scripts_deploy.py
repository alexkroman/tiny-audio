"""Tests for scripts/deploy package."""

from pathlib import Path


class TestRunpodCLI:
    """Tests for runpod CLI structure."""

    def test_app_exists(self):
        """Test that the typer app is properly configured."""
        from scripts.deploy.runpod import app

        assert app is not None

    def test_cli_entry_point(self):
        """Test that cli entry point exists."""
        from scripts.deploy.runpod import cli

        assert callable(cli)

    def test_rsync_excludes_defined(self):
        """Test that RSYNC_EXCLUDES list is defined."""
        from scripts.deploy.runpod import RSYNC_EXCLUDES

        assert isinstance(RSYNC_EXCLUDES, list)
        assert len(RSYNC_EXCLUDES) > 0
        assert "__pycache__" in RSYNC_EXCLUDES
        assert ".git" in RSYNC_EXCLUDES

    def test_ssh_key_path_defined(self):
        """Test that SSH_KEY_PATH is defined."""
        from scripts.deploy.runpod import SSH_KEY_PATH

        assert SSH_KEY_PATH is not None
        assert "ssh" in SSH_KEY_PATH.lower()


class TestRunpodConnectionUtils:
    """Tests for runpod connection utilities."""

    def test_get_connection_returns_connection(self):
        """Test that get_connection returns a Fabric Connection."""
        from scripts.deploy.runpod import get_connection

        conn = get_connection("example.com", 22)

        assert conn.host == "example.com"
        assert conn.port == 22
        assert conn.user == "root"

    def test_get_connection_uses_ssh_key(self):
        """Test that connection uses SSH key."""
        from scripts.deploy.runpod import get_connection

        conn = get_connection("example.com", 22)

        key_path = conn.connect_kwargs.get("key_filename", "")
        # key_filename can be a string or list
        if isinstance(key_path, list):
            key_path = key_path[0] if key_path else ""
        assert "id_ed25519" in key_path


class TestBuildTrainingScript:
    """Tests for build_training_script function."""

    def test_basic_script_generation(self):
        """Test basic training script generation."""
        from scripts.deploy.runpod import build_training_script

        script = build_training_script(
            experiment="mlp",
            hf_token="test_token",
            wandb_run_id=None,
            wandb_resume=None,
            extra_args=[],
        )

        assert "#!/bin/bash" in script
        assert "mlp" in script
        assert "test_token" in script
        assert "accelerate launch" in script

    def test_script_with_wandb(self):
        """Test training script with W&B settings."""
        from scripts.deploy.runpod import build_training_script

        script = build_training_script(
            experiment="mosa",
            hf_token="token",
            wandb_run_id="abc123",
            wandb_resume="must",
            extra_args=[],
        )

        assert 'WANDB_RUN_ID="abc123"' in script
        assert 'WANDB_RESUME="must"' in script

    def test_script_with_extra_args(self):
        """Test training script with extra Hydra args."""
        from scripts.deploy.runpod import build_training_script

        script = build_training_script(
            experiment="mlp",
            hf_token="token",
            wandb_run_id=None,
            wandb_resume=None,
            extra_args=["training.learning_rate=1e-4", "training.batch_size=8"],
        )

        assert "training.learning_rate=1e-4" in script
        assert "training.batch_size=8" in script

    def test_script_includes_env_vars(self):
        """Test that script includes necessary environment variables."""
        from scripts.deploy.runpod import build_training_script

        script = build_training_script(
            experiment="mlp",
            hf_token="token",
            wandb_run_id=None,
            wandb_resume=None,
            extra_args=[],
        )

        assert "TOKENIZERS_PARALLELISM" in script
        assert "HF_TOKEN" in script
        assert "PYTORCH_CUDA_ALLOC_CONF" in script


class TestHfSpaceCLI:
    """Tests for HF Space deployment CLI."""

    def test_main_exists(self):
        """Test that main entry point exists."""
        from scripts.deploy.hf_space import main

        assert callable(main)

    def test_deploy_to_space_exists(self):
        """Test that deploy_to_space function exists."""
        from scripts.deploy.hf_space import deploy_to_space

        assert callable(deploy_to_space)

    def test_run_command_exists(self):
        """Test that run_command function exists."""
        from scripts.deploy.hf_space import run_command

        assert callable(run_command)


class TestHandlerLocalCLI:
    """Tests for local handler testing CLI."""

    def test_main_exists(self):
        """Test that main entry point exists."""
        from scripts.deploy.handler_local import main

        assert callable(main)

    def test_find_latest_model_exists(self):
        """Test that find_latest_model function exists."""
        from scripts.deploy.handler_local import find_latest_model

        assert callable(find_latest_model)

    def test_find_test_audio_exists(self):
        """Test that find_test_audio function exists."""
        from scripts.deploy.handler_local import find_test_audio

        assert callable(find_test_audio)

    def test_find_latest_model_no_outputs(self, tmp_path: Path):
        """Test find_latest_model when outputs dir doesn't exist."""
        from scripts.deploy.handler_local import find_latest_model

        result = find_latest_model(str(tmp_path / "nonexistent"))

        assert result is None

    def test_find_latest_model_empty_dir(self, tmp_path: Path):
        """Test find_latest_model when outputs dir is empty."""
        from scripts.deploy.handler_local import find_latest_model

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        result = find_latest_model(str(outputs_dir))

        assert result is None


class TestHubPush:
    """Tests for hub/push.py module."""

    def test_main_exists(self):
        """Test that main entry point exists."""
        from scripts.hub.push import main

        assert callable(main)


class TestDeployPackageImports:
    """Tests for deploy package imports."""

    def test_deploy_package_importable(self):
        """Test that deploy package is importable."""
        import scripts.deploy

        assert scripts.deploy is not None

    def test_runpod_importable(self):
        """Test that runpod module is importable."""
        from scripts.deploy import runpod

        assert runpod is not None

    def test_hf_space_importable(self):
        """Test that hf_space module is importable."""
        from scripts.deploy import hf_space

        assert hf_space is not None

    def test_handler_local_importable(self):
        """Test that handler_local module is importable."""
        from scripts.deploy import handler_local

        assert handler_local is not None


class TestHubPackageImports:
    """Tests for hub package imports."""

    def test_hub_package_importable(self):
        """Test that hub package is importable."""
        import scripts.hub

        assert scripts.hub is not None

    def test_push_importable(self):
        """Test that push module is importable."""
        from scripts.hub import push

        assert push is not None


class TestDebugPackageImports:
    """Tests for debug package imports."""

    def test_debug_package_importable(self):
        """Test that debug package is importable."""
        import scripts.debug

        assert scripts.debug is not None

    def test_check_mosa_importable(self):
        """Test that check_mosa module is importable."""
        from scripts.debug import check_mosa

        assert check_mosa is not None
