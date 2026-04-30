from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from olmo_core.distributed.utils import get_local_tensor


@dataclass
class RunningMoments:
    sum: torch.Tensor
    sq_sum: torch.Tensor
    abs_sum: torch.Tensor
    count: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> "RunningMoments":
        kwargs = {"device": device, "dtype": torch.float32}
        return cls(
            sum=torch.zeros([], **kwargs),
            sq_sum=torch.zeros([], **kwargs),
            abs_sum=torch.zeros([], **kwargs),
            count=torch.zeros([], **kwargs),
        )

    def update(self, tensor: torch.Tensor) -> None:
        local_tensor = get_local_tensor(tensor.detach()).float()
        if local_tensor.numel() == 0:
            return

        flat = local_tensor.reshape(-1)
        self.sum += flat.sum()
        self.sq_sum += torch.square(flat).sum()
        self.abs_sum += flat.abs().sum()
        self.count += torch.tensor(float(flat.numel()), device=flat.device, dtype=flat.dtype)


class LayerStatsCollector:
    """
    Collect per-layer hidden-state statistics for the current training step.
    """

    def __init__(self):
        self._enabled = False
        self._stats: Dict[str, Dict[str, RunningMoments]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_step(self) -> None:
        self._stats = {}
        self._enabled = True

    def finish_step(self) -> None:
        self._enabled = False

    def reset(self) -> None:
        self._stats = {}
        self._enabled = False

    def pop_step_stats(self) -> Dict[str, Dict[str, RunningMoments]]:
        stats = self._stats
        self._stats = {}
        return stats

    def observe_hidden_state(self, layer_name: str, hidden_state: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return hidden_state

        self._update(layer_name, "forward", hidden_state)
        if hidden_state.requires_grad:
            hidden_state.register_hook(self._make_backward_hook(layer_name))
        return hidden_state

    def _make_backward_hook(self, layer_name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if self._enabled:
                self._update(layer_name, "backward", grad)
            return grad

        return hook

    def _update(self, layer_name: str, direction: str, tensor: torch.Tensor) -> None:
        if tensor.numel() <= 1:
            return

        layer_stats = self._stats.setdefault(layer_name, {})
        if direction not in layer_stats:
            layer_stats[direction] = RunningMoments.empty(tensor.device)
        layer_stats[direction].update(tensor)
