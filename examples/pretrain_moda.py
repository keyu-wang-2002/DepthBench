#!/usr/bin/env python3
"""Train a pre-norm Llama backbone with Mixture-of-Depths Attention."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base
from moda_common import configure_moda


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    config = base.build_config(args, overrides)
    configure_moda(config, post_norm=False)
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
