"""Tests for scripts/deploy/gpu_catalog.py (runpodctl calls are stubbed)."""

import json
import subprocess
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.deploy import gpu_catalog


@pytest.fixture(autouse=True)
def clear_catalog_cache() -> Iterator[None]:
    gpu_catalog.datacenter_catalog.cache_clear()
    yield
    gpu_catalog.datacenter_catalog.cache_clear()


def _stub_runpodctl(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


class TestRunpodctlJson:
    def test_appends_json_output_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _stub_runpodctl(monkeypatch, "[]")
        assert gpu_catalog.runpodctl_json("gpu", "list") == "[]"
        assert calls == [["runpodctl", "gpu", "list", "-o", "json"]]


class TestAvailableGpus:
    def test_filters_unavailable_and_small_then_sorts_and_dedupes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog = [
            {"gpuId": "H100", "memoryInGb": 80, "available": True},
            {"gpuId": "A40", "memoryInGb": 48, "available": True},
            {"gpuId": "A40", "memoryInGb": 48, "available": True},
            {"gpuId": "L4", "memoryInGb": 24, "available": True},
            {"gpuId": "B200", "memoryInGb": 180, "available": False},
            {"gpuId": "NoMem", "available": True},
        ]
        # A warning banner before the JSON must be skipped.
        _stub_runpodctl(monkeypatch, "update available!\n" + json.dumps(catalog))
        assert gpu_catalog.available_gpus(40) == [(48, "A40"), (80, "H100")]


class TestDatacenterCatalog:
    def test_parses_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _stub_runpodctl(monkeypatch, json.dumps([{"id": "US-TX-3"}]))
        assert gpu_catalog.datacenter_catalog() == ({"id": "US-TX-3"},)
        assert gpu_catalog.datacenter_catalog() == ({"id": "US-TX-3"},)
        assert len(calls) == 1

    def test_unparseable_output_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_runpodctl(monkeypatch, "runpodctl: command not found")
        assert gpu_catalog.datacenter_catalog() == ()

    def test_missing_cli_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_: Any, **__: Any) -> None:
            raise FileNotFoundError("runpodctl")

        monkeypatch.setattr(subprocess, "run", boom)
        assert gpu_catalog.datacenter_catalog() == ()


class TestDatacentersForGpu:
    CATALOG = (
        {
            "id": "AP-IN-1",  # no network volume support
            "location": "India",
            "gpuAvailability": [{"gpuId": "H100", "stockStatus": "High"}],
        },
        {
            "id": "US-TX-3",
            "location": "Texas",
            "gpuAvailability": [{"gpuId": "H100", "stockStatus": ""}],
        },
        {
            "id": "EU-RO-1",
            "location": "Romania",
            "gpuAvailability": [
                {"gpuId": "H100", "stockStatus": "Low"},
                {"gpuId": "A40", "stockStatus": "High"},
            ],
        },
        {"id": "US-CA-2", "gpuAvailability": [{"gpuId": "H100"}]},
        {"id": "EU-NL-1", "location": "NL"},
    )

    @pytest.fixture(autouse=True)
    def catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_catalog, "datacenter_catalog", lambda: self.CATALOG)

    def test_orders_by_stock_then_id(self) -> None:
        assert gpu_catalog.datacenters_for_gpu("H100") == [
            ("AP-IN-1", "India", "High"),
            ("EU-RO-1", "Romania", "Low"),
            ("US-CA-2", "?", ""),
            ("US-TX-3", "Texas", ""),
        ]

    def test_network_volume_filter_drops_unsupported(self) -> None:
        ids = [dc for dc, _, _ in gpu_catalog.datacenters_for_gpu("H100", True)]
        assert ids == ["EU-RO-1", "US-CA-2", "US-TX-3"]

    def test_unknown_gpu_is_empty(self) -> None:
        assert gpu_catalog.datacenters_for_gpu("TPU") == []

    def test_unknown_stock_status_sorts_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        catalog = (
            {"id": "A", "gpuAvailability": [{"gpuId": "X", "stockStatus": "Weird"}]},
            {"id": "B", "gpuAvailability": [{"gpuId": "X", "stockStatus": ""}]},
        )
        monkeypatch.setattr(gpu_catalog, "datacenter_catalog", lambda: catalog)
        assert [dc for dc, _, _ in gpu_catalog.datacenters_for_gpu("X")] == ["B", "A"]
