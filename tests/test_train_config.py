"""Tests for the Hydra structured-config schema in scripts/train_config.py."""

import inspect
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from scripts import train as train_module
from scripts.train import TRAINING_MODEL_PARAMS
from scripts.train_config import ModelConfig, TrainingConfig
from tiny_audio.asr_config import ASRConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DATA_CONFIGS = sorted(p.stem for p in (CONFIGS / "data").glob("*.yaml"))

# Keys TrainingConfig declares for scripts/train.py itself rather than for
# TrainingArguments (TRAINING_MODEL_PARAMS go to ASRConfig, the rest are popped
# or read directly in main()).
TRAIN_SCRIPT_KEYS = {
    "use_liger",
    "allow_unfused_ce",
    "wandb_project",
    "decoder_learning_rate",
    "projector_weight_decay",
    "encoder_learning_rate",
    "encoder_weight_decay",
    "use_cache",
    "init_from",
    *TRAINING_MODEL_PARAMS,
}


def _compose(overrides: Sequence[str] = ()) -> DictConfig:
    """Compose the repo's Hydra ``config`` with ``overrides`` applied."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_model_fields_are_asrconfig_params() -> None:
    params = set(inspect.signature(ASRConfig.__init__).parameters)
    missing = {f.name for f in fields(ModelConfig)} - params
    assert not missing, f"ModelConfig fields not accepted by ASRConfig: {missing}"


def test_model_fields_default_to_unset() -> None:
    # ASRConfig owns the model defaults; None lets train.py fall through to them.
    assert all(f.default is None for f in fields(ModelConfig))


def test_training_model_params_default_to_unset() -> None:
    # A non-None default here would silently override the `model:` block.
    defaults = {f.name: f.default for f in fields(TrainingConfig)}
    for param in TRAINING_MODEL_PARAMS:
        assert param in defaults, f"{param} missing from TrainingConfig"
        assert defaults[param] is None, param


def test_training_fields_are_known() -> None:
    valid = {f.name for f in fields(TrainingArguments)} | TRAIN_SCRIPT_KEYS
    unknown = {f.name for f in fields(TrainingConfig)} - valid
    assert not unknown, f"TrainingConfig fields TrainingArguments would drop: {unknown}"


def test_default_config_composes() -> None:
    cfg = _compose()
    assert OmegaConf.get_type(cfg.training) is TrainingConfig
    assert cfg.data.datasets, "default data group should list datasets"
    assert isinstance(cfg.training.learning_rate, float)


def test_default_config_is_qwen3_asr_encoder_and_smollm3() -> None:
    cfg = _compose()
    assert cfg.model.audio_model_id == "Qwen/Qwen3-ASR-1.7B-hf"
    # ~13 Hz straight out of the encoder, so no frame stacking.
    assert cfg.model.encoder_conv_layers == [[1, 3, 2], [1, 3, 2], [1, 3, 2]]
    assert cfg.model.projector_pool_stride == 1
    # 30 s clips in training, 28 s chunks at inference: chunks stay under it.
    assert cfg.data.max_audio_seconds == 30.0
    assert cfg.model.inference_chunk_seconds == 28.0
    assert cfg.model.text_model_id == "HuggingFaceTB/SmolLM3-3B"
    assert cfg.model.use_lora is True
    assert cfg.training.freeze_audio_encoder is True
    assert cfg.training.freeze_language_model is True


def test_tiny_preset_trains_the_published_model_at_laptop_size() -> None:
    """`-e tiny` keeps the Granite encoder its laptop numbers were measured with.

    Only the encoder settings, the decoder, data, steps and hardware differ:
    the projector width and the LoRA setup are the default's.
    """
    default, tiny = _compose(), _compose(["+experiments=tiny"])
    assert tiny.model.text_model_id == "Qwen/Qwen3.5-0.8B"
    assert tiny.model.audio_model_id == "ibm-granite/granite-speech-5.0-470m-turboctc-nc"
    assert tiny.model.projector_pool_stride == 1
    # 19 s training clips, so 18 s inference chunks (the default recipe: 30 / 28).
    assert tiny.data.max_audio_seconds == 19.0
    assert tiny.model.inference_chunk_seconds == 18.0
    encoder_keys = (
        "audio_model_id",
        "text_model_id",
        "encoder_conv_layers",
        "projector_pool_stride",
        "inference_chunk_seconds",
    )
    blank = dict.fromkeys(encoder_keys)
    assert {**tiny.model, **blank} == {**default.model, **blank}
    for key in ("freeze_audio_encoder", "freeze_language_model", "learning_rate"):
        assert tiny.training[key] == default.training[key], key
    [source] = tiny.data.datasets
    assert source.data_files == {
        "train": "hf://datasets/openslr/librispeech_asr/clean/validation/*.parquet"
    }
    # Nothing leaves the machine, and nothing CUDA-only is required.
    assert tiny.training.push_to_hub is False
    assert tiny.training.report_to == "none"
    assert tiny.training.optim == "adamw_torch"
    assert tiny.training.bf16 is False


@pytest.mark.parametrize("data", DATA_CONFIGS)
def test_data_config_composes(data: str) -> None:
    cfg = _compose([f"data={data}"])
    # Typed nodes: every dataset entry validated against DatasetConfig.
    for entry in cfg.data.datasets:
        assert entry.path


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigCompositionException):
        _compose(["training.lerning_rate=1e-4"])


def test_wrong_type_rejected() -> None:
    with pytest.raises(ConfigCompositionException):
        _compose(["training.seed=not-a-number"])


def test_append_syntax_adds_undeclared_key() -> None:
    cfg = _compose(["+training.warmup_ratio=0.05"])
    assert cfg.training.warmup_ratio == 0.05


class _ReachedModelBuildError(Exception):
    """Raised in place of ASRModel: main() got past its startup checks."""


class TestHubTokenPreflight:
    """main() refuses push_to_hub only when huggingface_hub has no token at all."""

    @staticmethod
    def _run_main(monkeypatch: pytest.MonkeyPatch, token: str | None, *overrides: str) -> None:
        """Run main() with push_to_hub on and Hub token ``token``.

        ASRModel is replaced so getting past the preflight raises _ReachedModelBuildError.
        """

        def no_model(*_a: object, **_k: object) -> None:
            raise _ReachedModelBuildError

        monkeypatch.setattr(train_module, "get_token", lambda: token)
        monkeypatch.setattr(train_module, "ASRModel", no_model)
        cfg = _compose(
            [
                "training.push_to_hub=true",
                "training.hub_model_id=me/model",
                "training.report_to=none",
                "training.use_liger=false",
                # The offline-fixture model pair, so startup never reaches the Hub.
                "model.audio_model_id=zai-org/GLM-ASR-Nano-2512",
                "model.text_model_id=Qwen/Qwen3-0.6B",
                *overrides,
            ]
        )
        # Bypass @hydra.main's argv parsing; the undecorated body takes the cfg.
        train_module.main.__wrapped__(cfg)

    def test_no_token_anywhere_fails_before_model_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="no Hugging Face token"):
            self._run_main(monkeypatch, None)

    def test_cached_login_is_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`hf auth login` with no HF_TOKEN in the env used to be rejected."""
        with pytest.raises(_ReachedModelBuildError):
            self._run_main(monkeypatch, "hf_cached")

    def test_explicit_hub_token_is_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(_ReachedModelBuildError):
            self._run_main(monkeypatch, None, "training.hub_token=hf_cfg")


class TestResolvedDict:
    """train._resolved_dict: a config node as a plain dict with string keys."""

    def test_mapping_resolves_with_string_keys(self) -> None:
        node = OmegaConf.create({"a": 1, "b": "${a}"})
        assert train_module._resolved_dict(node, "test") == {"a": 1, "b": 1}

    def test_a_non_mapping_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="test config must be a mapping"):
            # A list where a mapping belongs, as a mistyped YAML file would give.
            train_module._resolved_dict(cast(DictConfig, OmegaConf.create([1, 2])), "test")


class TestApplyLigerKernels:
    """Liger patching falls back to stock kernels, with a warning, when it can't apply."""

    @staticmethod
    def _cfg() -> DictConfig:
        return OmegaConf.create({"model": {"text_model_id": "Qwen/Qwen3.5-0.8B"}})

    @pytest.fixture(autouse=True)
    def _on_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The tests below are about a CUDA machine; without one liger is skipped.
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def test_skipped_without_cuda(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """liger is Triton (CUDA-only): a CPU or Mac run never calls the patcher."""
        calls: list[str] = []
        liger = SimpleNamespace(apply_liger_kernel_to_qwen3_5=lambda: calls.append("patched"))
        monkeypatch.setattr(train_module, "_liger_transformers", liger)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        caplog.set_level("INFO")
        train_module._apply_liger_kernels(self._cfg())
        assert calls == []
        assert "No CUDA device" in caplog.text

    def test_missing_liger_warns_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(train_module, "_liger_transformers", None)
        monkeypatch.setattr(train_module, "_liger_import_error", None)
        train_module._apply_liger_kernels(self._cfg())
        assert "liger_kernel.transformers did not import" in caplog.text

    def test_a_patcher_that_fails_warns_with_its_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken() -> None:
            msg = "no Qwen3_5 class"
            raise AttributeError(msg)

        liger = SimpleNamespace(apply_liger_kernel_to_qwen3_5=broken)
        monkeypatch.setattr(train_module, "_liger_transformers", liger)
        train_module._apply_liger_kernels(self._cfg())
        assert "no Qwen3_5 class" in caplog.text

    def test_a_working_patcher_is_applied(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list[str] = []
        liger = SimpleNamespace(apply_liger_kernel_to_qwen3_5=lambda: calls.append("patched"))
        monkeypatch.setattr(train_module, "_liger_transformers", liger)
        caplog.set_level("INFO")
        train_module._apply_liger_kernels(self._cfg())
        assert calls == ["patched"]
        assert "Applied liger kernels" in caplog.text

    @pytest.mark.parametrize(
        ("text_model_id", "patcher"),
        [
            ("Qwen/Qwen3.5-2B", "apply_liger_kernel_to_qwen3_5"),
            ("Qwen/Qwen3-1.7B-Base", "apply_liger_kernel_to_qwen3"),
            ("HuggingFaceTB/SmolLM3-3B", "apply_liger_kernel_to_smollm3"),
        ],
    )
    def test_each_decoder_gets_its_own_patcher(
        self, monkeypatch: pytest.MonkeyPatch, text_model_id: str, patcher: str
    ) -> None:
        """Qwen3.5 ids also contain "qwen3"; they must not fall through to the Qwen3 patcher."""
        calls: list[str] = []
        names = (
            "apply_liger_kernel_to_qwen3_5",
            "apply_liger_kernel_to_qwen3",
            "apply_liger_kernel_to_smollm3",
        )
        liger = SimpleNamespace(**{n: (lambda n=n: calls.append(n)) for n in names})
        monkeypatch.setattr(train_module, "_liger_transformers", liger)
        cfg = OmegaConf.create({"model": {"text_model_id": text_model_id}})
        train_module._apply_liger_kernels(cfg)
        assert calls == [patcher]


def test_default_mix_degrades_and_reweights_under_one_prompt_but_tiny_does_not() -> None:
    """The ten-corpus default carries the URGENT-matched augmentation, source
    weights and a single training prompt; the free laptop run uses none of them."""
    default, tiny = _compose(), _compose(["+experiments=tiny"])
    assert default.data.noise_prob > 0
    assert set(default.data.noise_corpus) == {"philgzl/dns5", "philgzl/wham"}
    assert set(default.data.reverb_corpus) == {"philgzl/dns5"}
    assert default.data.clip_prob > 0
    assert default.data.bandlimit_prob > 0
    assert default.data.single_prompt is True
    assert not default.data.verbatim_sources
    assert "speechcolab/gigaspeech:m" not in default.data.source_weights  # back at x1
    assert tiny.data.noise_prob == 0
    assert not tiny.data.noise_corpus
    assert not tiny.data.verbatim_sources
    assert not tiny.data.single_prompt
    assert not tiny.data.source_weights


def test_default_augmentation_leaves_about_half_the_rows_clean() -> None:
    """8 of the 9 leaderboard sets are clean audio. Degrading ~83% of rows (the
    2026-10-18 run) made clean conversational audio hallucinate."""
    d = _compose().data
    post_noise = d.bandlimit_prob + d.clip_prob + d.phone_prob
    clean = (1 - d.reverb_prob) * (1 - d.noise_prob) * (1 - post_noise)
    assert 0.4 <= clean <= 0.6


def test_nptel_stays_out_of_the_mix() -> None:
    """Removed 2026-10-20 after it made the Monsoon column worse (configs/data/multiasr.yaml)."""
    assert not [d for d in _compose().data.datasets if "NPTEL" in str(d.get("data_files"))]


def test_edacc_is_upweighted_three_times() -> None:
    assert _compose().data.source_weights["edinburghcstr/edacc"] == 3.0
