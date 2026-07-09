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

DEFAULT_ZERO_SHOT_TASKS = [
    "openbookqa",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "social_iqa",
    "piqa",
]

BBH_REASONING_ZERO_SHOT_TASKS = [
    "bbh_zeroshot_tracking_shuffled_objects_three_objects",
    "bbh_zeroshot_tracking_shuffled_objects_five_objects",
    "bbh_zeroshot_tracking_shuffled_objects_seven_objects",
    "bbh_zeroshot_logical_deduction_three_objects",
    "bbh_zeroshot_logical_deduction_five_objects",
    "bbh_zeroshot_logical_deduction_seven_objects",
    "bbh_zeroshot_boolean_expressions",
    "bbh_zeroshot_multistep_arithmetic_two",
    "bbh_zeroshot_web_of_lies",
    "bbh_zeroshot_navigate",
]

TASK_SETS = {
    "commonsense": DEFAULT_ZERO_SHOT_TASKS,
    "bbh-reasoning": BBH_REASONING_ZERO_SHOT_TASKS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot lm-eval-harness tasks on a native OLMo-core checkpoint."
    )
    parser.add_argument("checkpoint_dir", help="Checkpoint step directory or model_and_optim dir")
    parser.add_argument(
        "--task-set",
        choices=sorted(TASK_SETS),
        default="commonsense",
        help=(
            "Named task set used when --tasks is omitted. "
            "'commonsense' is the original 7-task suite; 'bbh-reasoning' is the "
            "recommended 10-task BBH reasoning suite."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=(
            "Explicit lm-eval task names. Overrides --task-set when provided."
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
        "--num-fewshot",
        type=int,
        default=None,
        help="Few-shot count. By default, honor each lm-eval task's configured value.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Include prompts, generations, and per-sample scores in the output JSON.",
    )
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
    tasks = args.tasks if args.tasks is not None else TASK_SETS[args.task_set]

    from olmo_lm import OLMoNativeLM

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
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=args.log_samples,
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
