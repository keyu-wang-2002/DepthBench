#!/usr/bin/env python3
"""Train the DeepNorm model."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base
from olmo_core.nn.layer_norm import LayerNormType
from olmo_core.nn.transformer import InitMethod, TransformerBlockType


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    config = base.build_config(args, overrides)
    config.model.block.name = TransformerBlockType.deepnorm
    config.model.block.layer_norm.name = LayerNormType.default
    config.model.init_method = InitMethod.deepnorm
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
