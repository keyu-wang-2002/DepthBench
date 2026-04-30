#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLMO_CORE_SRC = PROJECT_ROOT / "pretrain" / "OLMo-core" / "src"
if str(OLMO_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(OLMO_CORE_SRC))

from examples.huggingface.convert_checkpoint_to_hf import convert_checkpoint_to_hf  # noqa: E402
from olmo_core.config import DType  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DepthBench OLMo-core checkpoint into Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to a single checkpoint directory, e.g. /path/to/step2600",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to save the converted Hugging Face checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=None,
        help="Optional Hugging Face tokenizer ID to save together with the model.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Override max_position_embeddings in the converted config.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Precision used when saving the Hugging Face model.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the logits check between OLMo-core and Hugging Face models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        experiment_config = json.load(f)

    if "model" not in experiment_config:
        raise KeyError(f"'model' section missing in {config_path}")
    if "dataset" not in experiment_config or "tokenizer" not in experiment_config["dataset"]:
        raise KeyError(f"'dataset.tokenizer' section missing in {config_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    convert_checkpoint_to_hf(
        original_checkpoint_path=checkpoint_dir,
        output_path=args.output_dir,
        transformer_config_dict=experiment_config["model"],
        tokenizer_config_dict=experiment_config["dataset"]["tokenizer"],
        dtype=DType(args.dtype),
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
        validate=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
