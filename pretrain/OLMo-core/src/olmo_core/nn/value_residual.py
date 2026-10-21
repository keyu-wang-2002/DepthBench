"""RWKV-7 value residual learning with a LoRA-parameterized mixing gate."""

import math
from typing import Optional

import torch
import torch.nn as nn


def default_value_residual_lora_rank(d_model: int) -> int:
    """
    The RWKV-7 suggestion for the value-residual LoRA rank (``D_MV_LORA``):
    ``max(32, round(1.7 * sqrt(d_model) / 32) * 32)``.
    """
    return max(32, int(round(1.7 * math.sqrt(d_model) / 32)) * 32)


class ValueResidualMix(nn.Module):
    """
    Token-dependent interpolation weights towards the first layer's values,
    ``sigmoid(v0 + (x @ v1) @ v2)``, as in RWKV-7.

    ``v0`` starts at 0.73, ``v1`` at zero, and ``v2`` is
    orthogonally initialized with gain 0.1. The initial mixing
    weight is ``sigmoid(0.73) ~= 0.675``.

    :param d_model: The input (model) dimensionality.
    :param v_dim: The value dimensionality, ``n_kv_heads * head_dim``.
    :param lora_rank: The rank of the LoRA. Defaults to
        :func:`default_value_residual_lora_rank`.
    """

    def __init__(
        self,
        *,
        d_model: int,
        v_dim: int,
        lora_rank: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.lora_rank = lora_rank or default_value_residual_lora_rank(d_model)
        self.v0 = nn.Parameter(torch.empty(v_dim, dtype=dtype, device=init_device))
        self.v1 = nn.Parameter(torch.empty(d_model, self.lora_rank, dtype=dtype, device=init_device))
        self.v2 = nn.Parameter(torch.empty(self.lora_rank, v_dim, dtype=dtype, device=init_device))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.v0.is_meta:
            return
        nn.init.constant_(self.v0, 0.73)
        nn.init.zeros_(self.v1)

        v2 = torch.empty_like(self.v2, dtype=torch.float32)
        nn.init.orthogonal_(v2, gain=0.1)
        self.v2.copy_(v2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The attention input of shape ``(batch_size, seq_len, d_model)``.

        :returns: Mixing weights of shape ``(batch_size, seq_len, v_dim)``.
        """
        return torch.sigmoid(self.v0 + (x @ self.v1) @ self.v2)
