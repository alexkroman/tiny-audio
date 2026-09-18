"""Tests for scripts/eval/datasets.py - dataset configuration and registry."""

from scripts.eval.datasets import (
    DATASET_REGISTRY,
    DatasetConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = DatasetConfig(
            path="test/path",
            audio_field="audio",
        )
        assert config.path == "test/path"
        assert config.audio_field == "audio"
        assert config.text_field == "text"  # Default
        assert config.default_split == "test"  # Default

    def test_create_full_config(self):
        """Test creating config with all fields."""
        config = DatasetConfig(
            path="test/path",
            audio_field="wav",
            text_field="transcript",
            config="en",
            default_split="validation",
        )
        assert config.text_field == "transcript"
        assert config.config == "en"
        assert config.default_split == "validation"


class TestDatasetRegistry:
    """Tests for the dataset registry."""

    def test_registry_not_empty(self):
        """Test that registry contains datasets."""
        assert len(DATASET_REGISTRY) > 0

    def test_required_asr_datasets(self):
        """Test that common ASR datasets are registered."""
        required_datasets = [
            "loquacious",
            "earnings22",
            "ami",
            "librispeech",
            "librispeech-other",
            "tedlium",
            "commonvoice",
            "gigaspeech",
            "peoples",
        ]
        for name in required_datasets:
            assert name in DATASET_REGISTRY, f"Missing dataset: {name}"

    def test_all_configs_have_required_fields(self):
        """Test that all configs have required fields."""
        for name, config in DATASET_REGISTRY.items():
            assert config.path, f"Missing path for {name}"
            assert config.audio_field, f"Missing audio_field for {name}"
            assert config.text_field, f"Missing text_field for {name}"

    def test_loquacious_config(self):
        """Test specific config for loquacious dataset."""
        config = DATASET_REGISTRY["loquacious"]
        assert config.path == "speechbrain/LoquaciousSet"
        assert config.config == "small"
        assert config.audio_field == "wav"
        assert config.text_field == "text"

    def test_earnings22_config(self):
        """Test specific config for earnings22 dataset."""
        config = DATASET_REGISTRY["earnings22"]
        assert config.path == "sanchit-gandhi/earnings22_robust_split"
        assert config.audio_field == "audio"
        assert config.text_field == "sentence"

    def test_ami_config(self):
        """Test specific config for AMI dataset."""
        config = DATASET_REGISTRY["ami"]
        assert config.path == "edinburghcstr/ami"
        assert config.config == "ihm"


class TestDatasetValidation:
    """Tests for dataset configuration validation."""

    def test_default_splits_are_valid(self):
        """Test that default splits are reasonable values."""
        valid_splits = {"test", "validation", "dev", "train"}
        for name, config in DATASET_REGISTRY.items():
            err = f"Invalid split '{config.default_split}' for {name}"
            assert config.default_split in valid_splits, err
