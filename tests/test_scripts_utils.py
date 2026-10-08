"""Tests for scripts/utils.py shared utilities."""

from pathlib import Path

from scripts.utils import find_model_dirs, get_project_root


class TestFindModelDirs:
    """Tests for find_model_dirs function."""

    def test_find_matching_dirs(self, tmp_path: Path):
        """Test finding directories that match a pattern."""
        # Create test directories with expected format: timestamp_time_model_dataset
        (tmp_path / "20240101_120000_tiny-audio_librispeech").mkdir()
        (tmp_path / "20240102_130000_tiny-audio_commonvoice").mkdir()
        (tmp_path / "20240103_140000_whisper_librispeech").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 2
        assert all("tiny-audio" in d.name for d in dirs)

    def test_find_with_exclude(self, tmp_path: Path):
        """Test finding directories with exclusion patterns."""
        (tmp_path / "20240101_120000_tiny-audio_librispeech").mkdir()
        (tmp_path / "20240102_130000_tiny-audio-moe_librispeech").mkdir()
        (tmp_path / "20240103_140000_tiny-audio-mosa_librispeech").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio", exclude=["moe", "mosa"])

        assert len(dirs) == 1
        assert "moe" not in dirs[0].name
        assert "mosa" not in dirs[0].name

    def test_find_case_insensitive(self, tmp_path: Path):
        """Test that pattern matching is case-insensitive."""
        (tmp_path / "20240101_120000_Tiny-Audio_test").mkdir()
        (tmp_path / "20240102_130000_TINY-AUDIO_test2").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 2

    def test_empty_pattern_matches_every_model(self, tmp_path: Path):
        """`extract-entities` defaults to an empty pattern meaning "all models"."""
        (tmp_path / "20240101_120000_tiny-audio_librispeech").mkdir()
        (tmp_path / "20240102_130000_whisper_commonvoice").mkdir()

        dirs = find_model_dirs(tmp_path, "")

        assert len(dirs) == 2

    def test_empty_pattern_still_honors_exclude(self, tmp_path: Path):
        (tmp_path / "20240101_120000_tiny-audio_librispeech").mkdir()
        (tmp_path / "20240102_130000_whisper_commonvoice").mkdir()

        dirs = find_model_dirs(tmp_path, "", exclude=["whisper"])

        assert len(dirs) == 1
        assert "tiny-audio" in dirs[0].name

    def test_find_no_matches(self, tmp_path: Path):
        """Test when no directories match."""
        (tmp_path / "20240101_120000_whisper_librispeech").mkdir()
        (tmp_path / "20240102_130000_wav2vec_commonvoice").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 0

    def test_find_ignores_files(self, tmp_path: Path):
        """Test that regular files are ignored."""
        (tmp_path / "tiny-audio_results.txt").write_text("test")
        (tmp_path / "20240101_120000_tiny-audio_dir").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 1
        assert dirs[0].is_dir()

    def test_find_returns_sorted(self, tmp_path: Path):
        """Test that results are sorted."""
        (tmp_path / "20240103_140000_tiny-audio_c").mkdir()
        (tmp_path / "20240101_120000_tiny-audio_a").mkdir()
        (tmp_path / "20240102_130000_tiny-audio_b").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio")

        assert len(dirs) == 3
        # Sorted by name (which sorts by timestamp)
        assert "20240101" in dirs[0].name
        assert "20240102" in dirs[1].name
        assert "20240103" in dirs[2].name


class TestFindModelDirsLatest:
    """Tests for find_model_dirs(latest=True)."""

    def test_latest_keeps_one_run_per_model_and_dataset(self, tmp_path: Path):
        """An empty pattern spans models, so the key must include the model."""
        (tmp_path / "20240101_120000_tiny-audio_ami").mkdir()
        (tmp_path / "20240105_120000_tiny-audio_ami").mkdir()
        (tmp_path / "20240103_120000_whisper_ami").mkdir()

        dirs = find_model_dirs(tmp_path, "", latest=True)

        names = sorted(d.name for d in dirs)
        assert names == ["20240103_120000_whisper_ami", "20240105_120000_tiny-audio_ami"]

    def test_latest_treats_suffixed_runs_as_separate_evaluations(self, tmp_path: Path):
        """`_mcq` is a suffix, not the dataset -- two MCQ datasets must both survive."""
        (tmp_path / "20240101_120000_tiny-audio_ami_mcq").mkdir()
        (tmp_path / "20240102_120000_tiny-audio_earnings22_mcq").mkdir()
        (tmp_path / "20240103_120000_tiny-audio_ami").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio", latest=True)

        assert len(dirs) == 3

    def test_latest_picks_the_newest_of_identical_runs(self, tmp_path: Path):
        (tmp_path / "20240101_120000_tiny-audio_ami").mkdir()
        (tmp_path / "20240202_120000_tiny-audio_ami").mkdir()

        dirs = find_model_dirs(tmp_path, "tiny-audio", latest=True)

        assert [d.name for d in dirs] == ["20240202_120000_tiny-audio_ami"]


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
