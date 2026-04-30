from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from olmo_core.train.common import ReduceType

from ...nn.transformer.layer_stats import LayerStatsCollector
from ..train_module import TransformerTrainModule
from .callback import Callback


@dataclass
class LayerStatsMonitorCallback(Callback):
    """
    Records per-layer hidden-state statistics for transformer blocks.
    """

    enabled: bool = True
    interval: int = 1

    _collector: LayerStatsCollector = field(default_factory=LayerStatsCollector, repr=False)
    _should_monitor_step: bool = field(default=False, repr=False)

    def post_attach(self):
        if not self.enabled:
            return
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")
        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise ValueError(f"{type(self).__name__} only works with the TransformerTrainModule.")

        self.trainer.train_module.model.set_layer_stats_collector(self._collector)

    def pre_train(self):
        self._collector.reset()
        self._should_monitor_step = False

    def pre_step(self, batch: Dict[str, object]):
        del batch
        if not self.enabled:
            return

        self._should_monitor_step = self.step % self.interval == 0
        if self._should_monitor_step:
            self._collector.start_step()
        else:
            self._collector.reset()

    def pre_optim_step(self):
        if not self.enabled or not self._should_monitor_step:
            return

        for layer_name, direction_stats in self._collector.pop_step_stats().items():
            for direction, stats in direction_stats.items():
                prefix = f"layer_stats_raw/{layer_name}/{direction}"
                self.trainer.record_metric(f"{prefix}/sum", stats.sum, ReduceType.sum, namespace="train")
                self.trainer.record_metric(
                    f"{prefix}/sq_sum", stats.sq_sum, ReduceType.sum, namespace="train"
                )
                self.trainer.record_metric(
                    f"{prefix}/abs_sum", stats.abs_sum, ReduceType.sum, namespace="train"
                )
                self.trainer.record_metric(
                    f"{prefix}/count", stats.count, ReduceType.sum, namespace="train"
                )

        self._collector.finish_step()

    def post_train_batch(self):
        self._collector.finish_step()
        self._should_monitor_step = False

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        del step
        raw_prefix = "train/layer_stats_raw/"
        grouped: Dict[tuple[str, str], Dict[str, float]] = {}

        for key in list(metrics.keys()):
            if not key.startswith(raw_prefix):
                continue

            suffix = key[len(raw_prefix) :]
            parts = suffix.split("/")
            if len(parts) != 3:
                continue

            layer_name, direction, stat_name = parts
            grouped.setdefault((layer_name, direction), {})[stat_name] = metrics.pop(key)

        for (layer_name, direction), raw_stats in grouped.items():
            count = raw_stats.get("count", 0.0)
            if count <= 0:
                continue

            sum_ = raw_stats["sum"]
            sq_sum = raw_stats["sq_sum"]
            abs_sum = raw_stats["abs_sum"]

            mean = sum_ / count
            variance = max((sq_sum / count) - (mean * mean), 0.0)
            magnitude = abs_sum / count
            norm = math.sqrt(max(sq_sum, 0.0))

            metric_prefix = f"train/layer_stats/{layer_name}/{direction}"
            metrics[f"{metric_prefix}/mean"] = mean
            metrics[f"{metric_prefix}/variance"] = variance
            metrics[f"{metric_prefix}/magnitude"] = magnitude
            metrics[f"{metric_prefix}/norm"] = norm

    def close(self):
        self._collector.reset()
        if isinstance(self.trainer.train_module, TransformerTrainModule):
            self.trainer.train_module.model.set_layer_stats_collector(None)
