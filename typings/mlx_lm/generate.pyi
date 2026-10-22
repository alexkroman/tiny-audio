from collections.abc import Callable, Iterator
from typing import Any

from mlx.core import array

def generate_step(
    prompt: array,
    model: Any,
    *,
    max_tokens: int = ...,
    logits_processors: list[Callable[[array, array], array]] | None = ...,
    input_embeddings: array | None = ...,
) -> Iterator[tuple[array, array]]: ...
