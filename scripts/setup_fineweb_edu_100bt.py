#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "fineweb-edu"


def default_token_file() -> Path:
    candidates = (
        REPO_ROOT / "hf_token.txt",
        REPO_ROOT.parent / "hf_token.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return REPO_ROOT.parent / "hf_token.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the FineWeb-Edu sample/100BT parquet shards and arrange them for DepthBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token-file", type=Path, default=default_token_file())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--eval-shard", type=str, default="013_00008.parquet")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
        help="Optional extra allow pattern. Defaults to the full sample/100BT subset.",
    )
    return parser.parse_args()


def ensure_download(data_root: Path, token: str, allow_patterns: list[str] | None) -> Path:
    patterns = allow_patterns or ["sample/100BT/*.parquet"]
    cache_dir = data_root / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        token=token,
        allow_patterns=patterns,
        cache_dir=str(cache_dir),
        local_dir=str(data_root),
    )
    downloaded_dir = data_root / "sample" / "100BT"
    if not downloaded_dir.is_dir():
        raise FileNotFoundError(f"Expected downloaded shards under {downloaded_dir}")
    return downloaded_dir


def move_if_needed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    shutil.move(str(source), str(destination))


def main() -> None:
    args = parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Empty Hugging Face token in {args.token_file}")

    data_root = args.data_root.resolve()
    train_dir = data_root / "100BT"
    eval_dir = data_root / "eval"

    downloaded_dir = ensure_download(data_root, token, args.allow_patterns)

    for parquet_path in sorted(downloaded_dir.glob("*.parquet")):
        if parquet_path.name == args.eval_shard:
            move_if_needed(parquet_path, eval_dir / f"eval_{parquet_path.name}")
        else:
            move_if_needed(parquet_path, train_dir / parquet_path.name)

    if downloaded_dir.exists() and not any(downloaded_dir.iterdir()):
        downloaded_dir.rmdir()

    sample_dir = data_root / "sample"
    if sample_dir.exists() and not any(sample_dir.iterdir()):
        sample_dir.rmdir()

    train_count = len(list(train_dir.glob("*.parquet")))
    eval_count = len(list(eval_dir.glob("*.parquet")))
    print(f"Train shards: {train_count}")
    print(f"Eval shards: {eval_count}")
    print(f"Train dir: {train_dir}")
    print(f"Eval dir: {eval_dir}")


if __name__ == "__main__":
    main()
