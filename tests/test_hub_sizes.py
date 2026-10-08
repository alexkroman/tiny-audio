"""Hub-metadata size readers for the deploy planner (scripts/deploy/hub_sizes.py)."""

from types import SimpleNamespace

import pytest

from scripts.deploy import hub_sizes
from scripts.deploy.hub_sizes import (
    NON_LM_TOWER_PREFIXES,
    hidden_dim,
    lora_trainable_params,
    repo_weight_bytes,
    safetensors_params,
    vocab_table_params,
)


def _tensor(dtype: str, shape: tuple[int, ...]) -> SimpleNamespace:
    count = 1
    for dim in shape:
        count *= dim
    return SimpleNamespace(dtype=dtype, shape=list(shape), parameter_count=count)


# A decoder checkpoint with a vision tower, an int buffer, and a tied embedding.
_TENSORS = {
    "model.embed_tokens.weight": _tensor("BF16", (100, 8)),
    "model.layers.0.self_attn.q_proj.weight": _tensor("BF16", (8, 8)),
    "model.layers.0.mlp.up_proj.weight": _tensor("BF16", (16, 8)),
    "model.layers.0.input_layernorm.weight": _tensor("BF16", (8,)),
    "model.visual.layers.0.proj.weight": _tensor("BF16", (4, 4)),
    "model.rotary.inv_freq": _tensor("I64", (4,)),
    "lm_head.weight": _tensor("BF16", (100, 8)),
}


@pytest.fixture
def fake_metadata(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve _TENSORS as the safetensors header of every repo; record the ids asked for."""
    seen: list[str] = []
    totals: dict[str, int] = {}
    for info in _TENSORS.values():
        totals[info.dtype] = totals.get(info.dtype, 0) + info.parameter_count
    meta = SimpleNamespace(
        parameter_count=totals,
        files_metadata={"model.safetensors": SimpleNamespace(tensors=_TENSORS)},
    )

    def get_metadata(repo_id: str) -> SimpleNamespace:
        seen.append(repo_id)
        return meta

    monkeypatch.setattr(hub_sizes, "get_safetensors_metadata", get_metadata)
    return seen


def test_safetensors_params_counts_every_float_tensor(fake_metadata: list[str]) -> None:
    total, dominant = safetensors_params("org/decoder")
    # Everything except the I64 rotary buffer.
    assert total == 800 + 64 + 128 + 8 + 16 + 800
    assert dominant == "BF16"
    assert fake_metadata == ["org/decoder"]


def test_safetensors_params_drops_unloaded_towers(fake_metadata: list[str]) -> None:
    total, _ = safetensors_params("org/decoder", NON_LM_TOWER_PREFIXES)
    assert total == 800 + 64 + 128 + 8 + 800


def test_lora_all_linear_adapts_body_linears_only(fake_metadata: list[str]) -> None:
    # q_proj (8x8) and up_proj (16x8); not embed, lm_head, norms or the vision tower.
    assert lora_trainable_params("org/decoder", 4, "all-linear") == 4 * (8 + 8) + 4 * (16 + 8)


def test_lora_named_targets_match_module_suffix(fake_metadata: list[str]) -> None:
    assert lora_trainable_params("org/decoder", 2, ["q_proj"]) == 2 * (8 + 8)
    assert lora_trainable_params("org/decoder", 2, None) == 0


def test_vocab_table_params_reports_only_present_tables() -> None:
    plain = SimpleNamespace(vocab_size=100, hidden_size=8)
    assert vocab_table_params(plain) == {"embed_tokens": 800}
    gemma = SimpleNamespace(
        vocab_size=100,
        hidden_size=8,
        vocab_size_per_layer_input=50,
        hidden_size_per_layer_input=2,
        num_hidden_layers=3,
    )
    assert vocab_table_params(gemma) == {"embed_tokens": 800, "embed_tokens_per_layer": 300}
    assert vocab_table_params(SimpleNamespace()) == {}


def _info(*files: tuple[str, int | None]) -> SimpleNamespace:
    return SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files]
    )


def test_repo_weight_bytes_sums_model_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def model_info(repo_id: str, files_metadata: bool) -> SimpleNamespace:
        calls.append((repo_id, files_metadata))
        return _info(("a.safetensors", 10), ("b.safetensors", 5), ("empty", 0), ("none", None))

    monkeypatch.setattr(hub_sizes, "model_info", model_info)
    assert repo_weight_bytes("org/model") == 15
    assert calls == [("org/model", True)]


def test_repo_weight_bytes_scopes_dataset_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def dataset_info(repo_id: str, files_metadata: bool) -> SimpleNamespace:
        return _info(("small/0.parquet", 3), ("Large/0.parquet", 100), ("Large/1.parquet", 50))

    monkeypatch.setattr(hub_sizes, "dataset_info", dataset_info)
    assert repo_weight_bytes("org/ds", "dataset", "large") == 150
    # A config name that matches no file falls back to the whole repo.
    assert repo_weight_bytes("org/ds", "dataset", "missing") == 153


def test_hidden_dim_prefers_hidden_size_then_d_model() -> None:
    assert hidden_dim(SimpleNamespace(hidden_size=8, d_model=4)) == 8
    assert hidden_dim(SimpleNamespace(d_model=4)) == 4
    assert hidden_dim(SimpleNamespace()) is None
