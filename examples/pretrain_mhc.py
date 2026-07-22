#!/usr/bin/env python3
"""Train a pre-norm Llama backbone with manifold-constrained Hyper-Connections."""

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
    mhc_config = HyperConnectionsConfig(
        kind=HyperConnectionsKind(args.mhc_backend),
        num_residual_streams=4,
        tanh=False,
        gating_factor_init=0.01,
        sinkhorn_iters=20,
        collapse="auto",
    )
    config.model.block.name = TransformerBlockType.mhc
    config.model.block.hyper_connections = mhc_config
    disable_static_routing_weight_decay(config, mhc_config)
    return config


def main() -> None:
    parser = base.build_parser()
    parser.add_argument(
        "--mhc-backend",
        choices=("mhc", "mhc_static", "liger_mhc"),
        default="liger_mhc",
        help="mHC routing backend (default: fused Liger implementation)",
    )
    args, overrides = parser.parse_known_args()
    config = build_config(args, overrides)
    base.prepare_training_environment()
    try:
        base.train(config)
    finally:
        base.teardown_training_environment()


if __name__ == "__main__":
    main()
