#!/usr/bin/env python3
"""Train the default Pre-LN model."""

from __future__ import annotations

from typing import List

import pretrain_llama_base as base


def build_config(args, overrides: List[str]) -> base.ExperimentConfig:
    return base.build_config(args, overrides)


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
