#!/usr/bin/env python3
"""Export version pins and source provenance without reading credentials."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depthbench-repo", type=Path, default=HERE.parent)
    parser.add_argument("--moda-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Empty destination; never overwrite the captured snapshot")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Use an empty output directory for a new snapshot")
    versions = {}
    excluded = []
    excluded_names = {"ai2-olmo-core", "flash-linear-attention", "fla-core"}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        canonical_name = name.lower().replace("_", "-")
        direct_url = dist.read_text("direct_url.json")
        if direct_url or canonical_name in excluded_names:
            excluded_names.add(canonical_name)
            excluded.append(dict(name=name, version=dist.version,
                                 reason="Local/source install; preserve the source checkout separately"))
            continue
        versions[canonical_name] = dist.version
    # Editable .dist-info and source-tree .egg-info may both be discoverable.
    for name in excluded_names:
        versions.pop(name, None)
    if versions.get("torch") != "2.8.0+cu128" or versions.get("triton") != "3.4.0":
        raise RuntimeError("Run this exporter with the validated cu128 Python environment")
    header = ["# Generated from the validated Python 3.10.12 training environment.",
              "# Linux x86_64. OLMo-core and FLA profiles are installed separately; see README.md.",
              "--extra-index-url https://download.pytorch.org/whl/cu128", ""]
    sources = {}
    selected = {
        "DepthBench": (args.depthbench_repo, [
            "pretrain/OLMo-core/src/olmo_core/nn/transformer/block.py",
            "pretrain/OLMo-core/src/olmo_core/nn/hyper_connections.py"]),
        "MoDA": (args.moda_repo, ["libs/moda_triton/fla/ops/moda/moda_v17.py"]),
    }
    for name, (repo, files) in selected.items():
        sources[name] = dict(commit=git(repo, "rev-parse", "HEAD"),
                             branch=git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                             tracked_changes=git(repo, "diff", "--stat"),
                             file_sha256={file: hashlib.sha256((repo / file).read_bytes()).hexdigest()
                                          for file in files})
    patch = subprocess.check_output(
        ["git", "-C", str(args.moda_repo), "diff", "--binary", "HEAD", "--",
         "libs/moda_triton/fla/ops/moda/moda_v17.py"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "requirements-cu128.lock.txt").write_text(
        "\n".join(header + [f"{name}=={version}" for name, version in sorted(versions.items())]) + "\n")
    (output / "moda-v17-local.patch").write_bytes(patch)
    (output / "source-manifest.json").write_text(json.dumps(dict(
        captured_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        platform="linux-x86_64", package_count=len(versions), excluded_source_installs=excluded,
        sources=sources, moda_patch_sha256=hashlib.sha256(patch).hexdigest()), indent=2) + "\n")
    print(f"Exported {len(versions)} package pins, source manifest, and MoDA kernel patch")


if __name__ == "__main__":
    main()
