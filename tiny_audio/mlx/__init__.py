"""MLX inference path for the embedded experiment.

Only available on Apple Silicon. Install with `pip install tiny-audio[mlx]`.
"""

from tiny_audio.mlx.model import MLXASRModel

__all__ = ["MLXASRModel"]
