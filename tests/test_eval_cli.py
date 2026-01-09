"""Tests for scripts/eval/cli.py - CLI utilities and result saving."""

import tempfile

import pytest

from scripts.eval.cli import save_results
from scripts.eval.evaluators.base import EvalResult


class TestSaveResults:
    """Tests for save_results function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return [
            EvalResult(
                prediction="hello world",
                reference="hello world",
                wer=0.0,
                time=1.0,
            ),
            EvalResult(
                prediction="test prediction",
                reference="test reference",
                wer=50.0,
                time=0.5,
            ),
        ]

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics."""
        return {
            "wer": 25.0,
            "avg_time": 0.75,
            "num_samples": 2,
        }

    def test_creates_output_directory(self, sample_results, sample_metrics):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test-model",
                dataset_name="test-dataset",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
            )
            assert result_dir.exists()
            assert result_dir.is_dir()

    def test_creates_results_file(self, sample_results, sample_metrics):
        """Test that results.txt is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test-model",
                dataset_name="test-dataset",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
            )
            results_file = result_dir / "results.txt"
            assert results_file.exists()

            content = results_file.read_text()
            assert "Sample 1" in content
            assert "Sample 2" in content
            assert "WER:" in content

    def test_creates_metrics_file(self, sample_results, sample_metrics):
        """Test that metrics.txt is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test-model",
                dataset_name="test-dataset",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
            )
            metrics_file = result_dir / "metrics.txt"
            assert metrics_file.exists()

            content = metrics_file.read_text()
            assert "Model: test-model" in content
            assert "Dataset: test-dataset" in content
            assert "wer:" in content

    def test_directory_name_format(self, sample_results, sample_metrics):
        """Test that directory name follows expected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test-model",
                dataset_name="test-dataset",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
            )
            # Format: timestamp_model_dataset
            dir_name = result_dir.name
            parts = dir_name.split("_")
            assert len(parts) >= 3
            # First part should be timestamp (YYYYMMDD_HHMMSS)
            assert parts[0].isdigit()
            assert len(parts[0]) == 8  # YYYYMMDD

    def test_model_name_slash_replacement(self, sample_results, sample_metrics):
        """Test that slashes in model name are replaced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="org/model-name",
                dataset_name="test-dataset",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
            )
            assert "/" not in result_dir.name
            assert "org_model-name" in result_dir.name


class TestSaveResultsWithBaseUrl:
    """Tests for save_results with base_url parameter."""

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return [
            EvalResult(prediction="test", reference="test", wer=0.0, time=1.0),
        ]

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics."""
        return {"wer": 0.0, "avg_time": 1.0, "num_samples": 1}

    def test_sandbox_url_in_directory_name(self, sample_results, sample_metrics):
        """Test that sandbox URL is included in directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="assemblyai",
                dataset_name="ami",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
                base_url="https://api.sandbox013.assemblyai-labs.com",
            )
            assert "sandbox013" in result_dir.name

    def test_base_url_in_metrics_file(self, sample_results, sample_metrics):
        """Test that base_url is written to metrics file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_url = "https://api.sandbox013.assemblyai-labs.com"
            result_dir = save_results(
                model_name="assemblyai",
                dataset_name="ami",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
                base_url=base_url,
            )
            metrics_file = result_dir / "metrics.txt"
            content = metrics_file.read_text()
            assert f"Base URL: {base_url}" in content

    def test_no_base_url_no_suffix(self, sample_results, sample_metrics):
        """Test that no suffix is added when base_url is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="assemblyai",
                dataset_name="ami",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
                base_url=None,
            )
            assert "sandbox" not in result_dir.name

    def test_no_base_url_not_in_metrics(self, sample_results, sample_metrics):
        """Test that Base URL line is not in metrics when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="assemblyai",
                dataset_name="ami",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
                base_url=None,
            )
            metrics_file = result_dir / "metrics.txt"
            content = metrics_file.read_text()
            assert "Base URL:" not in content

    def test_different_sandbox_numbers(self, sample_results, sample_metrics):
        """Test extraction of different sandbox numbers."""
        sandbox_urls = [
            ("https://api.sandbox001.assemblyai-labs.com", "sandbox001"),
            ("https://api.sandbox123.assemblyai-labs.com", "sandbox123"),
            ("https://api.sandbox999.assemblyai-labs.com", "sandbox999"),
        ]
        for url, expected in sandbox_urls:
            with tempfile.TemporaryDirectory() as tmpdir:
                result_dir = save_results(
                    model_name="assemblyai",
                    dataset_name="test",
                    results=sample_results,
                    metrics=sample_metrics,
                    output_dir=tmpdir,
                    base_url=url,
                )
                assert expected in result_dir.name, f"Expected {expected} in {result_dir.name}"

    def test_non_sandbox_url_fallback(self, sample_results, sample_metrics):
        """Test fallback for non-sandbox URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="assemblyai",
                dataset_name="ami",
                results=sample_results,
                metrics=sample_metrics,
                output_dir=tmpdir,
                base_url="https://custom.example.com/api",
            )
            # Should use first part of hostname as fallback
            assert "custom" in result_dir.name


class TestResultsContent:
    """Tests for content of saved files."""

    def test_wer_formatting(self):
        """Test that WER is properly formatted in results."""
        results = [
            EvalResult(prediction="a", reference="b", wer=33.333333, time=1.0),
        ]
        metrics = {"wer": 33.33, "avg_time": 1.0, "num_samples": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test",
                dataset_name="test",
                results=results,
                metrics=metrics,
                output_dir=tmpdir,
            )
            content = (result_dir / "results.txt").read_text()
            # Should be formatted to 2 decimal places
            assert "33.33%" in content

    def test_metrics_float_formatting(self):
        """Test that float metrics are formatted to 4 decimal places."""
        results = [EvalResult(prediction="a", reference="a", wer=0.0, time=1.0)]
        metrics = {"wer": 12.345678, "avg_time": 0.123456, "num_samples": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test",
                dataset_name="test",
                results=results,
                metrics=metrics,
                output_dir=tmpdir,
            )
            content = (result_dir / "metrics.txt").read_text()
            assert "12.3457" in content  # Rounded to 4 decimal places
            assert "0.1235" in content

    def test_non_float_metrics(self):
        """Test that non-float metrics are written as-is."""
        results = [EvalResult(prediction="a", reference="a", wer=0.0, time=1.0)]
        metrics = {"wer": 0.0, "avg_time": 1.0, "num_samples": 42, "model_type": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = save_results(
                model_name="test",
                dataset_name="test",
                results=results,
                metrics=metrics,
                output_dir=tmpdir,
            )
            content = (result_dir / "metrics.txt").read_text()
            assert "num_samples: 42" in content
            assert "model_type: test" in content
