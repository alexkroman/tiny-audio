"""Tests for the `plan` / `provision` commands in scripts/deploy/plan.py.

`build_plan` (Hub lookups) and the RunPod catalog are stubbed, so these cover
the GPU / datacenter choice and the commands that get printed or run.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from scripts.deploy import gpu_catalog
from scripts.deploy import plan as plan_module
from scripts.deploy.plan import (
    GIB,
    NETWORK_VOLUME_MAX_GB,
    Component,
    Plan,
    plan_command,
    provision_command,
)

IMAGE = "img:1"


def _plan(disk_gib: float = 100.0, vram_gib: float = 30.0) -> Plan:
    return Plan(
        components=[
            Component("encoder", params=1_000, download_bytes=2 * GIB),
            Component("projector", params=10, trainable=True),
        ],
        dataset_rows=[("small", GIB), ("big", 5 * GIB)],
        warnings=["heads up"],
        vram={"total": vram_gib / 1.25, "recommended (x1.25)": vram_gib},
        disk={"total": disk_gib, "recommended": disk_gib},
    )


@pytest.fixture
def stub_catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mutable catalog: `gpus` for available_gpus, `dcs` for datacenters_for_gpu."""
    state: dict[str, Any] = {
        "gpus": [(48, "A40"), (80, "H100")],
        "dcs": {"H100": [("EU-RO-1", "Romania", "Low"), ("AP-IN-1", "India", "High")]},
    }

    def datacenters_for_gpu(gpu: str, require_network_volume: bool = False) -> list[Any]:
        rows = state["dcs"].get(gpu, [])
        if require_network_volume:
            rows = [r for r in rows if r[0] in gpu_catalog.NETWORK_VOLUME_DATACENTERS]
        return rows

    def available_gpus(_min_vram_gib: float) -> list[tuple[int, str]]:
        return state["gpus"]

    monkeypatch.setattr(gpu_catalog, "available_gpus", available_gpus)
    monkeypatch.setattr(gpu_catalog, "datacenters_for_gpu", datacenters_for_gpu)
    return state


def _use_plan(monkeypatch: pytest.MonkeyPatch, plan: Plan) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    def fake_build_plan(*args: Any) -> Plan:
        calls.append(args)
        return plan

    monkeypatch.setattr(plan_module, "build_plan", fake_build_plan)
    return calls


def _run_plan(**kwargs: Any) -> int | None:
    args: dict[str, Any] = {
        "experiment": "exp",
        "seq_len": 320,
        "gpu": None,
        "image": IMAGE,
        "as_json": False,
        "overrides": None,
    }
    args.update(kwargs)
    return plan_command(**args)


class TestPlanCommand:
    def test_json_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls = _use_plan(monkeypatch, _plan())
        assert _run_plan(as_json=True, overrides=["a=1"]) is None
        assert calls == [("exp", ["a=1"], 320)]
        out = json.loads(capsys.readouterr().out)
        assert out["experiment"] == "exp"
        assert out["components"][0] == {
            "name": "encoder",
            "params": 1_000,
            "download_gib": 2.0,
            "trainable": False,
        }
        assert out["warnings"] == ["heads up"]

    def test_small_disk_uses_container_disk_and_picks_smallest_gpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        stub_catalog: dict[str, Any],
    ) -> None:
        _use_plan(monkeypatch, _plan(disk_gib=100))
        assert _run_plan() == 0
        out = capsys.readouterr().out
        assert "=== Resource plan: +experiments=exp (seq_len=320) ===" in out
        # Datasets are listed largest first.
        assert out.index("big") < out.index("small")
        assert "! heads up" in out
        assert "Cheapest listed GPU that fits 30 GiB: A40 (48 GB)" in out
        assert "Also fit (--gpu to override): H100 (80 GB)" in out
        assert '--gpu-id "A40"' in out
        assert f"--container-disk-in-gb {int(100 * 1.15) + 5}" in out
        assert "network-volume create" not in out

    def test_explicit_gpu_skips_catalog(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _use_plan(monkeypatch, _plan())

        def no_catalog(_: float) -> None:
            msg = "catalog should not be queried"
            raise AssertionError(msg)

        monkeypatch.setattr(gpu_catalog, "available_gpus", no_catalog)
        assert _run_plan(gpu="L40S") == 0
        assert '--gpu-id "L40S"' in capsys.readouterr().out

    def test_large_disk_needs_volume_in_supported_datacenter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        stub_catalog: dict[str, Any],
    ) -> None:
        _use_plan(monkeypatch, _plan(disk_gib=1500))
        assert _run_plan() == 0
        out = capsys.readouterr().out
        # A40 has no volume-capable datacenter, so H100 wins.
        assert "with network-volume support: H100 (80 GB)" in out
        assert "Fit on VRAM but have no volume-capable datacenter: A40" in out
        assert "--data-center-id EU-RO-1" in out
        assert "--data-center-ids EU-RO-1" in out
        assert "NO network-volume support, so excluded: AP-IN-1" in out
        assert f"--size {int(1500 * 1.15) + 5}" in out
        assert "caps at" not in out

    def test_oversized_volume_is_capped_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        stub_catalog: dict[str, Any],
    ) -> None:
        _use_plan(monkeypatch, _plan(disk_gib=5000))
        _run_plan(gpu="NVIDIA H100 80GB HBM3")
        out = capsys.readouterr().out
        assert f"--size {NETWORK_VOLUME_MAX_GB}" in out
        assert "this will not fit on one volume" in out

    def test_volume_without_any_datacenter_uses_placeholder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        stub_catalog: dict[str, Any],
    ) -> None:
        stub_catalog["dcs"] = {}
        _use_plan(monkeypatch, _plan(disk_gib=1500))
        _run_plan()
        out = capsys.readouterr().out
        assert "no datacenter offering it also\n  supports network volumes" in out
        assert "--data-center-id <DC_ID>" in out

    def test_no_gpu_fits_falls_back_to_h100(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        stub_catalog: dict[str, Any],
    ) -> None:
        stub_catalog["gpus"] = []
        _use_plan(monkeypatch, _plan(vram_gib=500))
        _run_plan()
        out = capsys.readouterr().out
        assert "falling back to NVIDIA H100 80GB HBM3" in out


class TestProvisionCommand:
    @pytest.fixture
    def home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".ssh").mkdir()
        (tmp_path / ".ssh" / "id_ed25519.pub").write_text("ssh-ed25519 AAAA me\n")
        return tmp_path

    def _run(self, **kwargs: Any) -> str | None:
        args: dict[str, Any] = {
            "experiment": "exp",
            "seq_len": 320,
            "name": None,
            "image": IMAGE,
            "max_attempts": 6,
            "dry_run": False,
            "overrides": None,
        }
        args.update(kwargs)
        return provision_command(**args)

    def _stub_create(
        self, monkeypatch: pytest.MonkeyPatch, responses: list[str]
    ) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_: Any) -> SimpleNamespace:
            calls.append(args)
            return SimpleNamespace(stdout=responses[len(calls) - 1], stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_no_candidates_exits(
        self, monkeypatch: pytest.MonkeyPatch, stub_catalog: dict[str, Any], home: Path
    ) -> None:
        stub_catalog["gpus"] = []
        _use_plan(monkeypatch, _plan())
        with pytest.raises(typer.Exit):
            self._run()

    def test_missing_pubkey_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_catalog: dict[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        _use_plan(monkeypatch, _plan())
        with pytest.raises(typer.Exit):
            self._run()
        assert "id_ed25519.pub" in capsys.readouterr().out

    def test_dry_run_creates_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_catalog: dict[str, Any],
        home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _use_plan(monkeypatch, _plan())
        calls = self._stub_create(monkeypatch, [])
        assert self._run(dry_run=True, max_attempts=1) is None
        assert calls == []
        out = capsys.readouterr().out
        assert "candidates (smallest first): A40\n" in out
        assert "! heads up" in out

    def test_walks_candidates_until_one_is_created(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub_catalog: dict[str, Any],
        home: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        stub_catalog["gpus"] = [(24, "L4"), (48, "A40"), (80, "H100")]
        _use_plan(monkeypatch, _plan())
        calls = self._stub_create(
            monkeypatch,
            [
                "There are no longer any instances available",
                "garbage without json",
                'warning\n{"id": "pod123"}',
            ],
        )
        assert self._run(name="custom") == "pod123"
        assert [c[c.index("--gpu-id") + 1] for c in calls] == ["L4", "A40", "H100"]
        last = calls[-1]
        assert last[last.index("--name") + 1] == "custom"
        assert last[last.index("--container-disk-in-gb") + 1] == str(int(100 * 1.15) + 5)
        assert json.loads(last[last.index("--env") + 1]) == {
            "SSH_PUBLIC_KEY": "ssh-ed25519 AAAA me"
        }
        out = capsys.readouterr().out
        assert "no capacity" in out
        assert "unexpected response" in out
        assert "runpodctl pod delete pod123" in out

    def test_all_out_of_capacity_exits(
        self, monkeypatch: pytest.MonkeyPatch, stub_catalog: dict[str, Any], home: Path
    ) -> None:
        _use_plan(monkeypatch, _plan())
        calls = self._stub_create(monkeypatch, ['{"error": "x"}', '{"error": "y"}'])
        with pytest.raises(typer.Exit):
            self._run()
        assert len(calls) == 2
