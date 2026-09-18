#!/usr/bin/env python3
"""Train a pre-norm Llama backbone with manifold-constrained Hyper-Connections."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base
from hyper_connections_common import disable_static_routing_weight_decay
from olmo_core.config import DType
from olmo_core.nn.transformer import (
    HyperConnectionsConfig,
    HyperConnectionsKind,
    TransformerBlockType,
)


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    config = base.build_config(args, [])
    mhc_config = HyperConnectionsConfig(
        kind=HyperConnectionsKind(args.mhc_backend),
        num_residual_streams=4,
        tanh=False,
        gating_factor_init=0.01,
        sinkhorn_iters=20,
        disable_static_weight_decay=True,
        scale_output_init_by_sqrt_n=True,
        liger_phi_dtype=DType.bfloat16,
        liger_allow_fp32=False,
        liger_rms_eps=1e-6,
        liger_pre_eps=0.0,
        liger_sinkhorn_eps=1e-6,
        liger_post_mult=2.0,
        collapse="auto",
    )
    config.model.block.name = TransformerBlockType.mhc
    config.model.block.hyper_connections = mhc_config
    # Apply routing overrides after the method config exists, then derive optimizer groups.
    config = config.merge(overrides)
    mhc_config = config.model.block.hyper_connections
    if config.model.block.name != TransformerBlockType.mhc or mhc_config is None:
        raise ValueError("pretrain_mhc.py requires an mHC block and routing config")
    if mhc_config.kind not in {
        HyperConnectionsKind.mhc,
        HyperConnectionsKind.mhc_static,
        HyperConnectionsKind.liger_mhc,
    }:
        raise ValueError("pretrain_mhc.py requires an mHC routing backend")
    disable_static_routing_weight_decay(config, mhc_config)
    return config


def build_parser():
    parser = base.build_parser()
    parser.add_argument(
        "--mhc-backend",
        choices=("mhc", "mhc_static", "liger_mhc"),
        default="liger_mhc",
        help="mHC routing backend (default: fused Liger with gap-8 initialization)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args()
    config = build_config(args, overrides)
    base.prepare_training_environment()
    try:
        base.train(config)
    finally:
        base.teardown_training_environment()


if __name__ == "__main__":
    main()
