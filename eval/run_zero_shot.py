from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import lm_eval

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from olmo_lm import OLMoNativeLM


DEFAULT_ZERO_SHOT_TASKS = [
    "openbookqa",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "social_iqa",
    "piqa",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot lm-eval-harness tasks on a native OLMo-core checkpoint."
    )
    parser.add_argument("checkpoint_dir", help="Checkpoint step directory or model_and_optim dir")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_ZERO_SHOT_TASKS,
        help=(
            "lm-eval task names. Defaults to the 7 zero-shot MC tasks: "
            "openbookqa winogrande arc_challenge arc_easy hellaswag social_iqa piqa"
        ),
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path or HF id")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional maximum context length override",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Optional model dtype override at load time",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["torch", "flash_2", "flash_3"],
        default=None,
        help="Optional attention backend override",
    )
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save the raw lm-eval JSON result",
    )
    return parser.parse_args()


def _print_summary(results: dict[str, Any]) -> None:
    task_results = results.get("results", {})
    print(json.dumps(task_results, indent=2, ensure_ascii=False))


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in obj]
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    return str(obj)


def main() -> None:
    args = parse_args()

    lm = OLMoNativeLM.build(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_to_jsonable(results), indent=2, ensure_ascii=False)
        )

    _print_summary(results)


if __name__ == "__main__":
    main()
