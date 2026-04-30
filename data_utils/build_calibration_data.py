#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from tokenize_from_pretrain_datasets import iter_parquet_text_batches, load_hf_tokenizer


log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/fast/wangk/data/calibration"
DEFAULT_TOKENIZER_PATH = (
    "/home/wangk/DepthBench/pretrain/OLMo-core/src/olmo_core/data/tokenizers/"
    "allenai_gpt-neox-olmo-dolma-v1_5.json"
)
DEFAULT_FINEWEB_LOCAL_GLOB = "/fast/wangk/data/fineweb-edu/100BT/*.parquet"
TARGET_TOTAL_TOKENS = 256 * 1024


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    text_field: str
    description: Optional[str] = None
    hf_name: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None
    parquet_glob: Optional[str] = None


@dataclass
class SampleRecord:
    sample_id: int
    source: str
    source_document_index: int
    start_token: int
    end_token: int
    num_tokens: int
    text: str


SOURCE_PRESETS = {
    "fineweb_local": SourceSpec(
        name="fineweb_local",
        kind="parquet",
        parquet_glob=DEFAULT_FINEWEB_LOCAL_GLOB,
        text_field="text",
        description="Local FineWeb-edu parquet shards under /fast/wangk/data/fineweb-edu/100BT.",
    ),
    "fineweb": SourceSpec(
        name="fineweb",
        kind="hf",
        hf_name="HuggingFaceFW/fineweb-edu",
        hf_config="sample-10BT",
        hf_split="train",
        text_field="text",
        description="FineWeb-Edu sample served directly through Hugging Face datasets.",
    ),
    "c4": SourceSpec(
        name="c4",
        kind="hf",
        hf_name="allenai/c4",
        hf_config="en",
        hf_split="train",
        text_field="text",
        description="Official C4 English split served directly through Hugging Face datasets.",
    ),
    "dolma": SourceSpec(
        name="dolma",
        kind="hf",
        hf_name="emozilla/dolma-v1_7-30B",
        hf_split="train",
        text_field="text",
        description=(
            "Parquet mirror of a Dolma v1.7 sample. Used because recent datasets releases no longer "
            "load the official allenai/dolma dataset script directly."
        ),
    ),
    "dolma_official": SourceSpec(
        name="dolma_official",
        kind="hf",
        hf_name="allenai/dolma",
        hf_split="train",
        text_field="text",
        description=(
            "Official Dolma dataset script. This may fail on recent datasets versions unless you pre-download "
            "Dolma and set up local loading as described in the dataset card."
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build calibration text corpora for DepthBench analysis metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_PRESETS.keys()),
        default=None,
        help="Calibration source preset. Repeat to build multiple corpora.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="calibration")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocab-size", type=int, default=50280)
    parser.add_argument("--bos-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=50279)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--target-total-tokens", type=int, default=TARGET_TOTAL_TOKENS)
    parser.add_argument(
        "--sample-length-mode",
        type=str,
        choices=("fixed", "uniform"),
        default="fixed",
        help="Use a fixed token length or sample uniformly from a range.",
    )
    parser.add_argument("--sample-length", type=int, default=512)
    parser.add_argument("--min-sample-length", type=int, default=128)
    parser.add_argument("--max-sample-length", type=int, default=1024)
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=20000,
        help="Trim extremely long raw documents before tokenization for speed.",
    )
    parser.add_argument(
        "--hf-buffer-size",
        type=int,
        default=10000,
        help="Shuffle buffer size for streaming HF datasets.",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    parser.add_argument("--shuffle-samples", action="store_true", help="Shuffle collected samples before writing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_length_mode == "fixed" and args.sample_length <= 0:
        raise ValueError("--sample-length must be positive")
    if args.sample_length_mode == "uniform":
        if args.min_sample_length <= 0 or args.max_sample_length <= 0:
            raise ValueError("--min-sample-length and --max-sample-length must be positive")
        if args.min_sample_length > args.max_sample_length:
            raise ValueError("--min-sample-length must be <= --max-sample-length")
    if args.target_total_tokens <= 0:
        raise ValueError("--target-total-tokens must be positive")


def resolve_sources(args: argparse.Namespace) -> List[SourceSpec]:
    source_names = args.source if args.source else ["fineweb_local", "c4", "dolma"]
    return [SOURCE_PRESETS[name] for name in source_names]


def build_tokenizer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        bos_token_id=args.bos_token_id,
    )


def iter_hf_texts(source: SourceSpec, seed: int, buffer_size: int) -> Iterator[str]:
    try:
        dataset = load_dataset(
            source.hf_name,
            source.hf_config,
            split=source.hf_split,
            streaming=True,
        )
    except RuntimeError as exc:
        if source.name == "dolma_official" and "dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "The official 'allenai/dolma' loader now relies on a dataset script that your installed "
                "'datasets' version does not support. Use '--source dolma' for the parquet mirror, or "
                "pre-download Dolma locally and add a parquet-based preset."
            ) from exc
        raise
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    for example in dataset:
        value = example.get(source.text_field, "")
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text.strip():
            yield text


def iter_parquet_texts(source: SourceSpec, batch_size: int, rng: random.Random) -> Iterator[str]:
    parquet_paths = sorted(Path(path) for path in glob(source.parquet_glob))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files matched source preset {source.name}: {source.parquet_glob}")

    rng.shuffle(parquet_paths)
    for parquet_path in parquet_paths:
        for batch in iter_parquet_text_batches(
            parquet_path,
            text_field=source.text_field,
            batch_size=batch_size,
        ):
            rng.shuffle(batch)
            for text in batch:
                if text.strip():
                    yield text


def iter_source_texts(
    source: SourceSpec,
    *,
    seed: int,
    hf_buffer_size: int,
    parquet_batch_size: int,
    rng: random.Random,
) -> Iterator[str]:
    if source.kind == "hf":
        return iter_hf_texts(source, seed=seed, buffer_size=hf_buffer_size)
    if source.kind == "parquet":
        return iter_parquet_texts(source, batch_size=parquet_batch_size, rng=rng)
    raise ValueError(f"Unsupported source kind: {source.kind}")


def choose_sample_length(args: argparse.Namespace, rng: random.Random) -> int:
    if args.sample_length_mode == "fixed":
        return args.sample_length
    return rng.randint(args.min_sample_length, args.max_sample_length)


def get_min_allowed_length(args: argparse.Namespace) -> int:
    if args.sample_length_mode == "fixed":
        return args.sample_length
    return args.min_sample_length


def normalize_text_for_output(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sample_from_source(
    *,
    source: SourceSpec,
    token_budget: int,
    tokenizer,
    args: argparse.Namespace,
    seed: int,
) -> List[SampleRecord]:
    rng = random.Random(seed)
    source_iter = iter_source_texts(
        source,
        seed=seed,
        hf_buffer_size=args.hf_buffer_size,
        parquet_batch_size=args.parquet_batch_size,
        rng=rng,
    )

    samples: List[SampleRecord] = []
    consumed_tokens = 0
    document_index = 0
    progress = tqdm(total=token_budget, desc=f"{source.name} tokens", unit="tok", dynamic_ncols=True)

    for raw_text in source_iter:
        if consumed_tokens >= token_budget:
            break

        document_index += 1
        if args.max_doc_chars and len(raw_text) > args.max_doc_chars:
            raw_text = raw_text[: args.max_doc_chars]

        encoded = tokenizer(
            raw_text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if not encoded:
            continue

        target_length = choose_sample_length(args, rng)
        if len(encoded) < target_length:
            continue

        remaining = token_budget - consumed_tokens
        if remaining < get_min_allowed_length(args):
            break
        actual_length = min(target_length, remaining)
        if len(encoded) < actual_length:
            continue

        max_start = len(encoded) - actual_length
        start = rng.randint(0, max_start) if max_start > 0 else 0
        end = start + actual_length
        window = encoded[start:end]
        decoded = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        normalized = normalize_text_for_output(decoded)
        if not normalized:
            continue

        samples.append(
            SampleRecord(
                sample_id=len(samples),
                source=source.name,
                source_document_index=document_index,
                start_token=start,
                end_token=end,
                num_tokens=len(window),
                text=normalized,
            )
        )
        consumed_tokens += len(window)
        progress.update(len(window))

    progress.close()
    log.info(
        "Collected %s samples from %s totaling %s tokens",
        f"{len(samples):,}",
        source.name,
        f"{consumed_tokens:,}",
    )
    return samples


def rebalance_token_budgets(total_tokens: int, num_sources: int) -> List[int]:
    base = total_tokens // num_sources
    remainder = total_tokens % num_sources
    return [base + (1 if idx < remainder else 0) for idx in range(num_sources)]


def write_outputs(
    output_dir: Path,
    output_prefix: str,
    samples: List[SampleRecord],
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / f"{output_prefix}.txt"
    jsonl_path = output_dir / f"{output_prefix}.jsonl"
    summary_path = output_dir / f"{output_prefix}.summary.json"

    text_path.write_text("".join(sample.text + "\n" for sample in samples), encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log.info("Wrote text file: %s", text_path)
    log.info("Wrote jsonl file: %s", jsonl_path)
    log.info("Wrote summary file: %s", summary_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    validate_args(args)

    output_dir = Path(args.output_dir)
    sources = resolve_sources(args)
    tokenizer = load_hf_tokenizer(build_tokenizer_args(args))

    log.info("Tokenizer: %s", args.tokenizer_name_or_path)
    log.info("Sources: %s", ", ".join(source.name for source in sources))
    log.info("Target total tokens: %s", f"{args.target_total_tokens:,}")
    for source in sources:
        if source.description:
            log.info("Source %s: %s", source.name, source.description)

    token_budgets = rebalance_token_budgets(args.target_total_tokens, len(sources))
    all_samples: List[SampleRecord] = []

    for source_idx, (source, token_budget) in enumerate(zip(sources, token_budgets)):
        source_samples = sample_from_source(
            source=source,
            token_budget=token_budget,
            tokenizer=tokenizer,
            args=args,
            seed=args.seed + source_idx,
        )
        all_samples.extend(source_samples)

    if args.shuffle_samples:
        rng = random.Random(args.seed)
        rng.shuffle(all_samples)

    for idx, sample in enumerate(all_samples):
        sample.sample_id = idx

    total_tokens = sum(sample.num_tokens for sample in all_samples)
    source_stats = {}
    for source in sources:
        source_samples = [sample for sample in all_samples if sample.source == source.name]
        source_stats[source.name] = {
            "num_samples": len(source_samples),
            "num_tokens": sum(sample.num_tokens for sample in source_samples),
        }

    metadata = {
        "output_prefix": args.output_prefix,
        "sources": [source.name for source in sources],
        "target_total_tokens": args.target_total_tokens,
        "actual_total_tokens": total_tokens,
        "num_samples": len(all_samples),
        "sample_length_mode": args.sample_length_mode,
        "sample_length": args.sample_length,
        "min_sample_length": args.min_sample_length,
        "max_sample_length": args.max_sample_length,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "shuffle_samples": args.shuffle_samples,
        "seed": args.seed,
        "source_stats": source_stats,
    }

    write_outputs(output_dir, args.output_prefix, all_samples, metadata)


if __name__ == "__main__":
    main()
