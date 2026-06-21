#!/usr/bin/env python3
"""Standalone completion-only NLL evaluation for the Coding / STEM / Math suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


DEFAULT_TASKS = (
    "mbpp",        # Coding
    "humaneval",   # Coding
    "sciq",        # STEM
    "gpqa_diamond",# STEM
    "gsm8k",       # Math
    "math500",     # Math
)

TASK_CATEGORIES = {
    "coding": ("mbpp", "humaneval"),
    "stem": ("sciq", "gpqa_diamond"),
    "math": ("gsm8k", "math500"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone completion-only NLL evaluation on six tasks."
    )
    parser.add_argument(
        "checkpoint_dir", help="Checkpoint step directory or model_and_optim dir"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_TASKS,
        default=list(DEFAULT_TASKS),
        help="Subset of NLL tasks to evaluate",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path or HF id")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument(
        "--max-length", type=int, default=None, help="Optional maximum context length override"
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
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional per-task sample limit"
    )
    parser.add_argument("--output-path", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--save-samples", action="store_true", help="Save per-sample NLL records"
    )
    return parser.parse_args()


def _ensure_leading_separator(prompt: str, target: str) -> str:
    if not target:
        return target
    if prompt and not prompt[-1].isspace() and not target[0].isspace():
        return " " + target
    return target


def _format_mbpp(doc: dict[str, Any]) -> tuple[str, str]:
    tests = list(doc.get("test_list") or [])
    while len(tests) < 3:
        tests.append("")
    prompt = (
        "You are an expert Python programmer, and here is your task: "
        f"{doc['text']} Your code should pass these tests:\n\n"
        f"{tests[0]}\n{tests[1]}\n{tests[2]}\n[BEGIN]\n"
    )
    response = str(doc["code"])
    if not response.endswith("\n"):
        response += "\n"
    return prompt, response


def _format_humaneval(doc: dict[str, Any]) -> tuple[str, str]:
    # HumanEval's prompt already ends at the function body indentation.
    return str(doc["prompt"]), str(doc["canonical_solution"])


def _format_sciq(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Question: {doc['question']}\nAnswer:"
    response = f"{doc['correct_answer']}\nExplanation: {doc['support']}"
    return prompt, _ensure_leading_separator(prompt, response)


def _format_gpqa(doc: dict[str, Any]) -> tuple[str, str]:
    choices = [
        str(doc["Correct Answer"]).strip(),
        str(doc["Incorrect Answer 1"]).strip(),
        str(doc["Incorrect Answer 2"]).strip(),
        str(doc["Incorrect Answer 3"]).strip(),
    ]
    record_id = str(doc.get("Record ID") or doc["Question"])
    seed = int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:8], "big")
    random.Random(seed).shuffle(choices)
    rendered_choices = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    )
    prompt = (
        f"Question: {str(doc['Question']).strip()}\n"
        f"Choices:\n{rendered_choices}\n"
        "Answer:"
    )
    response = str(doc["Correct Answer"]).strip()
    explanation = str(doc.get("Explanation") or "").strip()
    if explanation:
        response += f"\nExplanation: {explanation}"
    return prompt, _ensure_leading_separator(prompt, response)


def _format_gsm8k(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Question: {doc['question']}\nAnswer:"
    response = str(doc["answer"])
    return prompt, _ensure_leading_separator(prompt, response)


def _format_math500(doc: dict[str, Any]) -> tuple[str, str]:
    prompt = f"Problem:\n{doc['problem']}\n\nSolution:"
    response = str(doc.get("solution") or doc.get("answer"))
    return prompt, _ensure_leading_separator(prompt, response)


def _load_task_docs(task: str) -> list[dict[str, Any]]:
    if task == "mbpp":
        # Parquet-backed mirror used by lm-eval; compatible with new datasets versions.
        return list(load_dataset("google-research-datasets/mbpp", "full", split="test"))
    if task == "humaneval":
        return list(load_dataset("openai/openai_humaneval", split="test"))
    if task == "sciq":
        return list(load_dataset("allenai/sciq", split="test"))
    if task == "gpqa_diamond":
        return list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
    if task == "gsm8k":
        return list(load_dataset("openai/gsm8k", "main", split="test"))
    if task == "math500":
        return list(load_dataset("HuggingFaceH4/MATH-500", "default", split="test"))
    raise ValueError(f"Unsupported task: {task}")


def _format_example(task: str, doc: dict[str, Any]) -> tuple[str, str]:
    if task == "mbpp":
        return _format_mbpp(doc)
    if task == "humaneval":
        return _format_humaneval(doc)
    if task == "sciq":
        return _format_sciq(doc)
    if task == "gpqa_diamond":
        return _format_gpqa(doc)
    if task == "gsm8k":
        return _format_gsm8k(doc)
    if task == "math500":
        return _format_math500(doc)
    raise ValueError(f"Unsupported task: {task}")


def _count_continuation_tokens(lm: Any, prompt: str, target: str) -> int:
    if prompt:
        _, continuation_enc = lm._encode_pair(prompt, target)
        return len(continuation_enc)
    return len(lm.tok_encode(target, add_special_tokens=False))


def _evaluate_task(
    lm: Any,
    task: str,
    *,
    limit: int | None = None,
    save_samples: bool = False,
) -> dict[str, Any]:
    from lm_eval.api.instance import Instance

    docs = _load_task_docs(task)
    if limit is not None:
        docs = docs[:limit]

    prompts_and_targets = [_format_example(task, doc) for doc in docs]
    instances = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(prompt, target),
            idx=index,
        )
        for index, (prompt, target) in enumerate(prompts_and_targets)
    ]
    scored = lm.loglikelihood(instances, disable_tqdm=False)

    total_logprob = 0.0
    total_tokens = 0
    samples: list[dict[str, Any]] = []
    for index, ((prompt, target), (logprob, is_greedy)) in enumerate(
        zip(prompts_and_targets, scored, strict=True)
    ):
        continuation_tokens = _count_continuation_tokens(lm, prompt, target)
        sample_nll = -float(logprob)
        total_logprob += float(logprob)
        total_tokens += continuation_tokens
        if save_samples:
            samples.append(
                {
                    "index": index,
                    "num_target_tokens": continuation_tokens,
                    "nll_sum": sample_nll,
                    "nll_mean": sample_nll / continuation_tokens
                    if continuation_tokens
                    else math.nan,
                    "is_greedy": bool(is_greedy),
                }
            )

    nll_sum = -total_logprob
    result = {
        "dataset": task,
        "num_examples": len(docs),
        "num_target_tokens": total_tokens,
        "nll_sum": nll_sum,
        "nll_mean": nll_sum / total_tokens if total_tokens else math.nan,
        "perplexity": math.exp(nll_sum / total_tokens) if total_tokens else math.nan,
        "completion_only": True,
    }
    if save_samples:
        result["samples"] = samples
    return result


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in obj]
    return str(obj)


def main() -> None:
    args = parse_args()

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
    task_results = {
        task: _evaluate_task(
            lm,
            task,
            limit=args.limit,
            save_samples=args.save_samples,
        )
        for task in args.tasks
    }
    total_nll = sum(result["nll_sum"] for result in task_results.values())
    total_tokens = sum(result["num_target_tokens"] for result in task_results.values())
    output = {
        "checkpoint_dir": args.checkpoint_dir,
        "tasks": task_results,
        "aggregate": {
            "num_target_tokens": total_tokens,
            "nll_sum": total_nll,
            "nll_mean": total_nll / total_tokens if total_tokens else math.nan,
            "perplexity": math.exp(total_nll / total_tokens)
            if total_tokens
            else math.nan,
            "completion_only": True,
        },
    }

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_to_jsonable(output), indent=2, ensure_ascii=False)
        )
    print(json.dumps(_to_jsonable(output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
