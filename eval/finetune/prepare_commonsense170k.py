#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import numpy as np


def format_instruction_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def load_tokenizer(tokenizer_name_or_path: str):
    tokenizer_path = Path(tokenizer_name_or_path)
    
    if tokenizer_path.is_file():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        class TokenizerWrapper:
            def __init__(self, inner):
                self.inner = inner
                self.vocab_size = inner.get_vocab_size()
                self.eos_token_id = inner.token_to_id("<|endoftext|>")

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                del add_special_tokens
                return self.inner.encode(text).ids

            def get_vocab_size(self) -> int:
                return self.inner.get_vocab_size()

            def token_to_id(self, token: str):
                return self.inner.token_to_id(token)

            def save_pretrained(self, output_dir: str) -> None:
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "tokenizer.json").write_text(
                    tokenizer_path.read_text(encoding="utf-8"), encoding="utf-8"
                )

        return TokenizerWrapper(tokenizer)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    return tokenizer


def choose_token_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def choose_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    raise ValueError("Unable to determine tokenizer vocab size.")


def choose_eos_token_id(tokenizer) -> int:
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    if hasattr(tokenizer, "token_to_id"):
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is not None:
            return int(eos_id)
    raise ValueError("Unable to determine tokenizer eos_token_id.")


def encode_example(
    tokenizer,
    *,
    instruction: str,
    input_text: str,
    output_text: str,
    eos_token_id: int,
    max_seq_len: int,
) -> Tuple[List[int], List[bool], dict]:
    prompt = format_instruction_prompt(instruction=instruction, input_text=input_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_text = (output_text or "").strip()
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    token_ids = list(prompt_ids) + list(response_ids) + [eos_token_id]
    label_mask = [False] * len(prompt_ids) + [True] * (len(response_ids) + 1)

    truncated = False
    if len(token_ids) > max_seq_len:
        if max_seq_len < 2:
            raise ValueError("--max-seq-len must be at least 2 to keep one supervised token and EOS.")
        prefix_limit = max_seq_len - 1
        token_ids = token_ids[:prefix_limit] + [eos_token_id]
        label_mask = label_mask[:prefix_limit] + [True]
        truncated = True
        if not any(label_mask[:-1]):
            raise ValueError(
                "Example was truncated before the response started. Increase --max-seq-len."
            )

    stats = {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids) + 1,
        "total_tokens": len(token_ids),
        "truncated": truncated,
        "supervised_tokens": int(sum(label_mask)),
    }
    return token_ids, label_mask, stats


def flush_part(
    token_buffer: Sequence[int],
    mask_buffer: Sequence[bool],
    *,
    output_dir: Path,
    token_dtype: np.dtype,
    part_idx: int,
) -> None:
    token_array = np.asarray(token_buffer, dtype=token_dtype)
    mask_array = np.asarray(mask_buffer, dtype=np.bool_)

    token_path = output_dir / f"token_ids_part_{part_idx:06d}.npy"
    mask_path = output_dir / f"labels_mask_part_{part_idx:06d}.npy"

    token_memmap = np.memmap(token_path, dtype=token_dtype, mode="w+", shape=token_array.shape)
    token_memmap[:] = token_array
    token_memmap.flush()

    mask_memmap = np.memmap(mask_path, dtype=np.bool_, mode="w+", shape=mask_array.shape)
    mask_memmap[:] = mask_array
    mask_memmap.flush()


def iter_records(dataset) -> Iterable[dict]:
    for row in dataset:
        yield {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
        }


def clear_previous_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("token_ids_part_*.npy"):
        path.unlink()
    for path in output_dir.glob("labels_mask_part_*.npy"):
        path.unlink()
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()
    tokenizer_dir = output_dir / "tokenizer"
    if tokenizer_dir.is_dir():
        shutil.rmtree(tokenizer_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare zwhe99/commonsense_170k for OLMo-core SFT using a plain instruction template.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-name", type=str, default="zwhe99/commonsense_170k")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--part-size", type=int, default=1_000_000)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser

def main() -> None:
    args = build_parser().parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Preparing Commonsense170K requires the 'datasets' package."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(output_dir)

    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    vocab_size = choose_vocab_size(tokenizer)
    eos_token_id = choose_eos_token_id(tokenizer)
    token_dtype = choose_token_dtype(vocab_size)

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    token_buffer: List[int] = []
    mask_buffer: List[bool] = []
    part_idx = 0

    stats = {
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "num_examples": 0,
        "num_truncated_examples": 0,
        "total_tokens": 0,
        "total_supervised_tokens": 0,
        "max_total_tokens": 0,
        "max_prompt_tokens": 0,
        "max_response_tokens": 0,
        "max_seq_len": args.max_seq_len,
        "token_dtype": str(token_dtype),
        "eos_token_id": eos_token_id,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "template": "instruction_plain",
    }

    for record in iter_records(dataset):
        token_ids, label_mask, example_stats = encode_example(
            tokenizer,
            instruction=record["instruction"],
            input_text=record["input"],
            output_text=record["output"],
            eos_token_id=eos_token_id,
            max_seq_len=args.max_seq_len,
        )
        token_buffer.extend(token_ids)
        mask_buffer.extend(label_mask)

        stats["num_examples"] += 1
        stats["num_truncated_examples"] += int(example_stats["truncated"])
        stats["total_tokens"] += example_stats["total_tokens"]
        stats["total_supervised_tokens"] += example_stats["supervised_tokens"]
        stats["max_total_tokens"] = max(stats["max_total_tokens"], example_stats["total_tokens"])
        stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], example_stats["prompt_tokens"])
        stats["max_response_tokens"] = max(
            stats["max_response_tokens"], example_stats["response_tokens"]
        )

        if len(token_buffer) >= args.part_size:
            flush_part(
                token_buffer,
                mask_buffer,
                output_dir=output_dir,
                token_dtype=token_dtype,
                part_idx=part_idx,
            )
            token_buffer.clear()
            mask_buffer.clear()
            part_idx += 1

    if token_buffer:
        flush_part(
            token_buffer,
            mask_buffer,
            output_dir=output_dir,
            token_dtype=token_dtype,
            part_idx=part_idx,
        )

    tokenizer_dir = output_dir / "tokenizer"
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(tokenizer_dir))

    (output_dir / "metadata.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
