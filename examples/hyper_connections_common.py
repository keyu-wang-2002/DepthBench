"""Shared configuration helpers for HC and mHC training entries."""

from olmo_core.nn.transformer import HyperConnectionsConfig
from olmo_core.optim import OptimGroupOverride


def disable_static_routing_weight_decay(
    config, hc_config: HyperConnectionsConfig
) -> None:
    if not hc_config.disable_static_weight_decay:
        return

    patterns = []
    for connector in ("attention_hyper_connection", "feed_forward_hyper_connection"):
        patterns.extend(hc_config.static_parameter_patterns(f"blocks.*.{connector}"))

    overrides = list(config.train_module.optim.group_overrides or [])
    overrides.append(OptimGroupOverride(params=patterns, opts={"weight_decay": 0.0}))
    config.train_module.optim.group_overrides = overrides
