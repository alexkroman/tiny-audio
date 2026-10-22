from typing import Any, Self

from mlx import nn
from mlx.core import array

class ModelArgs:
    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> Self: ...

class Qwen3Model(nn.Module):
    embed_tokens: nn.Embedding

class Model(nn.Module):
    model: Qwen3Model
    def __init__(self, args: ModelArgs) -> None: ...
    def __call__(
        self, inputs: array, cache: Any = ..., input_embeddings: array | None = ...
    ) -> array: ...
