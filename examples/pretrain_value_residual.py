#!/usr/bin/env python3
"""Train a pre-norm Llama backbone with the RWKV-7 value residual (LoRA mixing gate)."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base
from olmo_core.nn.transformer import TransformerBlockType
from olmo_core.optim import OptimGroupOverride


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    config = base.build_config(args, overrides)
    config.model.block.name = TransformerBlockType.value_residual
    # RWKV-7 only decays ".weight" matrices, so the gate's v0/v1/v2 get no weight decay.
    group_overrides = list(config.train_module.optim.group_overrides or [])
    group_overrides.append(
        OptimGroupOverride(params=["blocks.*.value_residual.*"], opts={"weight_decay": 0.0})
    )
    config.train_module.optim.group_overrides = group_overrides
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
