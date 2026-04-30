"""Tiny test for MLXEvaluator wiring."""

import pytest

mx = pytest.importorskip("mlx.core")


def test_mlx_evaluator_class_exists_and_imports():
    """Smoke: class is importable from the public eval module."""
    from scripts.eval.evaluators import MLXEvaluator

    assert MLXEvaluator is not None
    # Class signature: __init__(self, model_path: str, **kwargs)
    import inspect

    sig = inspect.signature(MLXEvaluator.__init__)
    assert "model_path" in sig.parameters


def test_cli_mlx_url_prefix_routing(tmp_path):
    """The mlx:// prefix should be parsed and routed in cli.py."""
    from pathlib import Path

    cli_path = Path(__file__).parent.parent / "scripts" / "eval" / "cli.py"
    src = cli_path.read_text()
    assert "mlx://" in src, "cli.py must have mlx:// routing branch"
    assert "MLXEvaluator" in src, "cli.py must instantiate MLXEvaluator"
