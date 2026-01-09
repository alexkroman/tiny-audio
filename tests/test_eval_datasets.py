"""Tests for scripts/eval/datasets.py - dataset configuration and registry."""

from scripts.eval.datasets import (
    ALIGNMENT_DATASETS,
    DATASET_REGISTRY,
    DIARIZATION_DATASETS,
    DatasetConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = DatasetConfig(
            name="test",
            path="test/path",
            audio_field="audio",
        )
        assert config.name == "test"
        assert config.path == "test/path"
        assert config.audio_field == "audio"
        assert config.text_field == "text"  # Default
        assert config.default_split == "test"  # Default

    def test_create_full_config(self):
        """Test creating config with all fields."""
        config = DatasetConfig(
            name="test",
            path="test/path",
            audio_field="wav",
            text_field="transcript",
            config="en",
            default_split="validation",
            weight=0.5,
            speakers_field="speakers",
            timestamps_start_field="start",
            timestamps_end_field="end",
            words_field="words",
        )
        assert config.text_field == "transcript"
        assert config.config == "en"
        assert config.default_split == "validation"
        assert config.weight == 0.5
        assert config.speakers_field == "speakers"
        assert config.words_field == "words"


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
            assert config.name == name, f"Config name mismatch for {name}"
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


class TestDiarizationDatasets:
    """Tests for diarization dataset categorization."""

    def test_diarization_datasets_not_empty(self):
        """Test that diarization datasets set is not empty."""
        assert len(DIARIZATION_DATASETS) > 0

    def test_callhome_is_diarization(self):
        """Test that callhome is categorized as diarization."""
        assert "callhome" in DIARIZATION_DATASETS

    def test_diarization_configs_have_speaker_fields(self):
        """Test that diarization datasets have speaker fields."""
        for name in DIARIZATION_DATASETS:
            config = DATASET_REGISTRY[name]
            assert config.speakers_field is not None, f"Missing speakers_field for {name}"
            assert config.timestamps_start_field is not None, (
                f"Missing timestamps_start_field for {name}"
            )
            assert config.timestamps_end_field is not None, (
                f"Missing timestamps_end_field for {name}"
            )

    def test_asr_datasets_not_in_diarization(self):
        """Test that ASR datasets are not in diarization set."""
        asr_datasets = ["loquacious", "earnings22", "ami", "librispeech"]
        for name in asr_datasets:
            assert name not in DIARIZATION_DATASETS


class TestAlignmentDatasets:
    """Tests for alignment dataset categorization."""

    def test_alignment_datasets_not_empty(self):
        """Test that alignment datasets set is not empty."""
        assert len(ALIGNMENT_DATASETS) > 0

    def test_librispeech_alignments_is_alignment(self):
        """Test that librispeech-alignments is categorized as alignment."""
        assert "librispeech-alignments" in ALIGNMENT_DATASETS

    def test_alignment_configs_have_words_field(self):
        """Test that alignment datasets have words_field."""
        for name in ALIGNMENT_DATASETS:
            config = DATASET_REGISTRY[name]
            assert config.words_field is not None, f"Missing words_field for {name}"

    def test_no_overlap_diarization_alignment(self):
        """Test that diarization and alignment sets don't overlap."""
        overlap = DIARIZATION_DATASETS & ALIGNMENT_DATASETS
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"


class TestDatasetValidation:
    """Tests for dataset configuration validation."""

    def test_all_datasets_in_exactly_one_category(self):
        """Test that specialized datasets are properly categorized."""
        for name in DATASET_REGISTRY:
            in_diarization = name in DIARIZATION_DATASETS
            in_alignment = name in ALIGNMENT_DATASETS
            # Can be in at most one specialized category
            assert not (in_diarization and in_alignment), f"{name} in multiple categories"

    def test_default_splits_are_valid(self):
        """Test that default splits are reasonable values."""
        valid_splits = {"test", "validation", "dev", "train", "data", "dev_clean"}
        for name, config in DATASET_REGISTRY.items():
            assert config.default_split in valid_splits, (
                f"Invalid split '{config.default_split}' for {name}"
            )

    def test_weights_are_positive(self):
        """Test that all weights are positive."""
        for name, config in DATASET_REGISTRY.items():
            assert config.weight > 0, f"Non-positive weight for {name}"
