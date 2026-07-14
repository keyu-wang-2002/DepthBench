#!/usr/bin/env python3
"""Train a pre-norm Llama backbone with Hyper-Connections."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base
from hyper_connections_common import disable_static_routing_weight_decay
from olmo_core.nn.transformer import (
    HyperConnectionsConfig,
    HyperConnectionsKind,
    TransformerBlockType,
)


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    config = base.build_config(args, overrides)
    hc_config = HyperConnectionsConfig(
        kind=HyperConnectionsKind.hc,
        num_residual_streams=4,
        gating_factor_init=0.01,
    )
    config.model.block.name = TransformerBlockType.hc
    config.model.block.hyper_connections = hc_config
    disable_static_routing_weight_decay(config, hc_config)
    return config


def main() -> None:
    parser = base.build_parser()
    args, overrides = parser.parse_known_args()
    config = build_config(args, overrides)
    base.prepare_training_environment()
    try:
        base.train(config)
    finally:
        base.teardown_training_environment()


if __name__ == "__main__":
    main()
