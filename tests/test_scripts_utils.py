"""Tests for scripts/utils.py shared utilities."""

import tempfile
from pathlib import Path

import pytest

from scripts.utils import find_model_dirs, get_project_root, parse_results_file


class TestParseResultsFile:
    """Tests for parse_results_file function."""

    def test_parse_single_sample(self, tmp_path: Path):
        """Test parsing a results file with a single sample."""
        results_file = tmp_path / "results.txt"
        results_file.write_text(
            "Sample 1 - WER: 12.50%\n"
            "Ground Truth: hello world\n"
            "Prediction: hello word\n"
            + "-" * 80
        )

        samples = parse_results_file(results_file)

        assert len(samples) == 1
        assert samples[0]["sample_num"] == 1
        assert samples[0]["ground_truth"] == "hello world"
        assert samples[0]["prediction"] == "hello word"
        assert samples[0]["wer"] == 12.50
        assert samples[0]["word_count"] == 2

    def test_parse_multiple_samples(self, tmp_path: Path):
        """Test parsing a results file with multiple samples."""
        results_file = tmp_path / "results.txt"
        results_file.write_text(
            "Sample 1 - WER: 0.00%\n"
            "Ground Truth: the quick brown fox\n"
            "Prediction: the quick brown fox\n"
            + "-" * 80 + "\n"
            "Sample 2 - WER: 25.00%\n"
            "Ground Truth: jumps over the lazy dog\n"
            "Prediction: jumps over the lazy cat\n"
            + "-" * 80
        )

        samples = parse_results_file(results_file)

        assert len(samples) == 2
        assert samples[0]["wer"] == 0.0
        assert samples[0]["word_count"] == 4
        assert samples[1]["wer"] == 25.0
        assert samples[1]["word_count"] == 5

    def test_parse_empty_file(self, tmp_path: Path):
        """Test parsing an empty results file."""
        results_file = tmp_path / "results.txt"
        results_file.write_text("")

        samples = parse_results_file(results_file)

        assert len(samples) == 0

    def test_parse_malformed_block(self, tmp_path: Path):
        """Test that malformed blocks are skipped."""
        results_file = tmp_path / "results.txt"
        results_file.write_text(
            "Some random text\n"
            + "-" * 80 + "\n"
            "Sample 1 - WER: 10.00%\n"
            "Ground Truth: valid sample\n"
            "Prediction: valid sample\n"
            + "-" * 80
        )

        samples = parse_results_file(results_file)

        assert len(samples) == 1
        assert samples[0]["sample_num"] == 1

    def test_parse_100_percent_wer(self, tmp_path: Path):
        """Test parsing a sample with 100% WER."""
        results_file = tmp_path / "results.txt"
        results_file.write_text(
            "Sample 1 - WER: 100.00%\n"
            "Ground Truth: completely different\n"
            "Prediction: nothing matches here at all\n"
            + "-" * 80
        )

        samples = parse_results_file(results_file)

        assert len(samples) == 1
        assert samples[0]["wer"] == 100.0


class TestFindModelDirs:
    """Tests for find_model_dirs function."""

    def test_find_matching_dirs(self, tmp_path: Path):
        """Test finding directories that match a pattern."""
        # Create test directories
        (tmp_path / "20240101_tiny-audio_librispeech").mkdir()
        (tmp_path / "20240102_tiny-audio_commonvoice").mkdir()
        (tmp_path / "20240103_whisper_librispeech").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 2
        assert all("tiny-audio" in d.name for d in dirs)

    def test_find_with_exclude(self, tmp_path: Path):
        """Test finding directories with exclusion patterns."""
        (tmp_path / "20240101_tiny-audio_librispeech").mkdir()
        (tmp_path / "20240102_tiny-audio_moe_librispeech").mkdir()
        (tmp_path / "20240103_tiny-audio_mosa_librispeech").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio", exclude=["moe", "mosa"])

        assert len(dirs) == 1
        assert "moe" not in dirs[0].name
        assert "mosa" not in dirs[0].name

    def test_find_case_insensitive(self, tmp_path: Path):
        """Test that pattern matching is case-insensitive."""
        (tmp_path / "Tiny-Audio_test").mkdir()
        (tmp_path / "TINY-AUDIO_test2").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 2

    def test_find_no_matches(self, tmp_path: Path):
        """Test when no directories match."""
        (tmp_path / "whisper_librispeech").mkdir()
        (tmp_path / "wav2vec_commonvoice").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 0

    def test_find_ignores_files(self, tmp_path: Path):
        """Test that regular files are ignored."""
        (tmp_path / "tiny-audio_results.txt").write_text("test")
        (tmp_path / "tiny-audio_dir").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 1
        assert dirs[0].is_dir()

    def test_find_returns_sorted(self, tmp_path: Path):
        """Test that results are sorted."""
        (tmp_path / "c_tiny-audio").mkdir()
        (tmp_path / "a_tiny-audio").mkdir()
        (tmp_path / "b_tiny-audio").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert [d.name for d in dirs] == ["a_tiny-audio", "b_tiny-audio", "c_tiny-audio"]


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path(self):
        """Test that get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_expected_files(self):
        """Test that the project root contains expected files."""
        root = get_project_root()
        # Should contain pyproject.toml at minimum
        assert (root / "pyproject.toml").exists()

    def test_root_contains_scripts_dir(self):
        """Test that the project root contains scripts directory."""
        root = get_project_root()
        assert (root / "scripts").is_dir()
