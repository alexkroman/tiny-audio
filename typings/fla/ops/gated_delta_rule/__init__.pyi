from typing import Any

import torch

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]: ...
