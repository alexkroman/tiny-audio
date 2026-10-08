"""Tests for scripts/deploy package.

Focuses on behavior testing rather than existence checks.
"""

import importlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import typer

from scripts.deploy import plan as plan_module
from scripts.deploy import runpod
from scripts.deploy.handler_local import find_latest_model
from scripts.deploy.plan import DATASET_DISK_FACTOR, build_plan, wait_command
from scripts.deploy.remote_scripts import build_eval_script, build_training_script
from scripts.deploy.runpod import (
    SSH_CONNECT_ATTEMPTS,
    SSH_KEY_PATH,
    _gitignore_aware_file_list,
    _remote_free_gib,
    get_connection,
)
from scripts.utils import get_project_root


class TestRunpodCLI:
    """Tests for runpod CLI configuration and utilities."""

    def test_gitignore_aware_file_list_excludes_gitignored_paths(self) -> None:
        """File list piped into rsync should honor .gitignore (no __pycache__,
        no .git/, no datasets_cache) and drop the suffix blocklist (no
        .safetensors — Swift bundled weights aren't needed on RunPod)."""
        files = _gitignore_aware_file_list(get_project_root()).splitlines()

        assert files, "expected at least one file from git ls-files"
        assert not any("__pycache__" in f for f in files)
        assert not any(f.startswith(".git/") for f in files)
        assert not any(f.startswith("datasets_cache/") for f in files)
        assert not any(f.endswith(".safetensors") for f in files)
        # Sanity: at least the project's pyproject.toml is in there
        assert "pyproject.toml" in files

    def test_ssh_key_path_is_valid(self) -> None:
        """Test that SSH_KEY_PATH points to expected location."""
        assert SSH_KEY_PATH is not None
        assert "ssh" in SSH_KEY_PATH.lower()


class TestRunpodConnectionUtils:
    """Tests for runpod connection utilities."""

    def test_get_connection_configures_correctly(self) -> None:
        """Test that get_connection returns properly configured Connection."""
        conn = get_connection("example.com", 22)

        assert conn.host == "example.com"
        assert conn.port == 22
        assert conn.user == "root"

        # Verify SSH key is configured
        assert conn.connect_kwargs is not None
        key_path: str | list[str] = conn.connect_kwargs.get("key_filename", "")
        if isinstance(key_path, list):
            key_path = key_path[0] if key_path else ""
        assert "id_ed25519" in key_path


def _free_gib(value: float | None) -> Callable[..., float | None]:
    def free_gib(conn: object, *a: object, **k: object) -> float | None:
        return value

    return free_gib


def _plan_recommending_2500(*a: object, **k: object) -> object:
    return type("P", (), {"disk": {"recommended": 2500.0}})()


class TestRemoteDiskPreflight:
    """`ta runpod train` refuses a pod /workspace cannot hold.

    The failure this guards against is expensive and late: ENOSPC arrives
    hours in, as an xet reconstruction error during dataset prep.
    """

    @staticmethod
    def _conn(stdout: str, ok: bool = True) -> MagicMock:
        conn = MagicMock()
        conn.run.return_value = MagicMock(ok=ok, stdout=stdout)
        return conn

    def test_parses_available_column_from_df(self) -> None:
        # `df -Pk` columns: Filesystem 1K-blocks Used Available Capacity Mounted
        conn = self._conn("overlay 2113650000 1000000 2097152000 1% /workspace\n")
        free = _remote_free_gib(conn)

        assert free == pytest.approx(2097152000 / 1024**2, rel=1e-6)  # 2000 GiB

    def test_returns_none_when_df_fails(self) -> None:
        assert _remote_free_gib(self._conn("", ok=False)) is None

    def test_returns_none_on_unparseable_output(self) -> None:
        assert _remote_free_gib(self._conn("df: /workspace: No such file\n")) is None

    def test_exits_when_pod_is_too_small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runpod, "_remote_free_gib", _free_gib(100.0))
        monkeypatch.setattr("scripts.deploy.plan.build_plan", _plan_recommending_2500)

        with pytest.raises(typer.Exit):
            runpod._check_remote_disk(self._conn(""), "stage_1", [])

    def test_proceeds_when_pod_is_big_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runpod, "_remote_free_gib", _free_gib(3000.0))
        monkeypatch.setattr("scripts.deploy.plan.build_plan", _plan_recommending_2500)

        # No exception == training is allowed to start.
        runpod._check_remote_disk(self._conn(""), "stage_1", [])

    def test_unreadable_df_does_not_block_training(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runpod, "_remote_free_gib", _free_gib(None))

        # A df we cannot read is not evidence the pod is too small; the
        # preflight must not become a new way for `train` to fail.
        runpod._check_remote_disk(self._conn(""), "stage_1", [])

    def test_plan_failure_does_not_block_training(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runpod, "_remote_free_gib", _free_gib(100.0))

        def boom(*a: object, **k: object) -> None:
            msg = "Hub is down"
            raise RuntimeError(msg)

        monkeypatch.setattr("scripts.deploy.plan.build_plan", boom)

        runpod._check_remote_disk(self._conn(""), "stage_1", [])


class TestDiskPlan:
    """plan.py's disk model must count what actually lands on /workspace."""

    def test_datasets_charged_for_parquet_and_arrow(self) -> None:
        # datasets keeps the Hub parquet download AND the arrow tables it
        # generates from it; measured at 2.05x on librispeech_asr_dummy.
        assert DATASET_DISK_FACTOR > 2.0

    def test_checkpoints_scale_with_trainable_stack_and_retention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A joint fine-tune's checkpoints are the decoder + AdamW, times
        save_total_limit -- not one projector-sized file."""
        # The planner reads parameter counts and repo sizes from the live Hub
        # API; stand in approximate real numbers so the test runs offline.
        params = {"zai-org/GLM-ASR-Nano-2512": 640_000_000, "Qwen/Qwen3-0.6B": 596_049_920}

        def fake_params(repo: str, *_: object) -> tuple[int, str]:
            return params[repo], "BF16"

        def fake_weight_bytes(repo: str, *_: object) -> int:
            return 2 * params.get(repo, 0)

        monkeypatch.setattr(plan_module, "safetensors_params", fake_params)
        monkeypatch.setattr(plan_module, "repo_weight_bytes", fake_weight_bytes)

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
    def build_script(self) -> Callable[..., str]:
        """Get the build_training_script function."""
        return build_training_script

    def test_basic_script_structure(self, build_script: Callable[..., str]) -> None:
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

    def test_script_includes_required_env_vars(self, build_script: Callable[..., str]) -> None:
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

    def test_script_with_wandb_settings(self, build_script: Callable[..., str]) -> None:
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

    def test_script_with_extra_hydra_args(self, build_script: Callable[..., str]) -> None:
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

    def test_find_latest_model_returns_none_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test find_latest_model returns None when outputs dir doesn't exist."""
        result = find_latest_model(str(tmp_path / "nonexistent"))
        assert result is None

    def test_find_latest_model_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """Test find_latest_model returns None when outputs dir is empty."""
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
    def test_module_importable(self, module_path: str) -> None:
        """Test that module can be imported without errors."""
        module = importlib.import_module(module_path)
        assert module is not None


def test_generic_training_script_runs_scripts_train() -> None:
    script = build_training_script("granite_qwen", "token", None, None, [])
    assert "python -m scripts.train +experiments=granite_qwen" in script


class TestDeployRetries:
    """`wait` polling and the SSH probe retry through tenacity."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # tenacity sleeps through time.sleep; skip the backoff in tests.
        def no_sleep(_s: float) -> None:
            return None

        monkeypatch.setattr("tenacity.nap.time.sleep", no_sleep)

    def test_wait_polls_until_ssh_endpoint_appears(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        outputs = iter(["", '{"ssh": null}', '{"ssh": {"ip": "1.2.3.4", "port": 22022}}'])

        def fake_run(*a: object, **k: object) -> SimpleNamespace:
            return SimpleNamespace(stdout=next(outputs))

        monkeypatch.setattr(subprocess, "run", fake_run)
        wait_command(pod_id="pod", timeout_s=900)
        assert capsys.readouterr().out.strip() == "1.2.3.4 22022"

    def test_wait_gives_up_after_timeout(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fake_run(*a: object, **k: object) -> SimpleNamespace:
            return SimpleNamespace(stdout="{}")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(typer.Exit):
            wait_command(pod_id="pod", timeout_s=0)
        assert "no SSH endpoint within 0s" in capsys.readouterr().out

    def test_ssh_probe_retries_then_succeeds(self) -> None:
        conn = MagicMock()
        conn.run.side_effect = [OSError("refused"), None]
        assert runpod.test_connection(conn) is True
        assert conn.run.call_count == 2

    def test_ssh_probe_gives_up(self) -> None:
        conn = MagicMock()
        conn.run.side_effect = OSError("refused")
        assert runpod.test_connection(conn) is False
        assert conn.run.call_count == SSH_CONNECT_ATTEMPTS


class TestBuildEvalScript:
    """build_eval_script turns eval options into the remote shell script."""

    def _build(self, **overrides: object) -> str:
        kwargs: dict[str, object] = {
            "hf_token": "hf_x",
            "model": "me/model",
            "datasets": ["loquacious", "earnings22"],
            "max_samples": 100,
            "assemblyai_api_key": "aai_key",
            "assemblyai_model": "best",
            "num_workers": 4,
            "streaming": True,
            "extra_args": ["--foo", "bar"],
        }
        kwargs.update(overrides)
        return build_eval_script(**kwargs)  # type: ignore[arg-type]

    def test_all_options(self) -> None:
        script = self._build()
        assert "python -m scripts.eval.cli" in script
        assert "--model me/model" in script
        assert "--datasets loquacious earnings22" in script
        assert "--max-samples 100" in script
        assert "--assemblyai-model best" in script
        assert "--num-workers 4" in script
        assert "--streaming" in script
        assert "--foo bar" in script
        assert 'export ASSEMBLYAI_API_KEY="aai_key"' in script
        assert "pip install modelscope" in script
        assert "Evaluation Completed Successfully" in script

    def test_optional_flags_omitted(self) -> None:
        script = self._build(
            datasets=[],
            max_samples=None,
            assemblyai_api_key=None,
            num_workers=1,
            streaming=False,
            extra_args=None,
        )
        for flag in ("--datasets", "--max-samples", "--num-workers", "--streaming"):
            assert flag not in script
        assert "ASSEMBLYAI_API_KEY" not in script
        assert "--assemblyai-model best" in script
