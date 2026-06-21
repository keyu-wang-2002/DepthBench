#!/usr/bin/env python3
"""Plot per-layer token effective rank for all DepthBench aspect-ratio variants.

Usage:
    python scripts/david/plot_token_collapse.py
    python scripts/david/plot_token_collapse.py \
        --results-dir /fast/dmartinez/depthbench/analysis/token_collapse \
        --output-dir  scripts/david/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


LAYER_ORDER = ["16l", "20l", "24l", "26l", "28l", "30l", "32l",
               "16l-attnres", "20l-attnres", "24l-attnres", "26l-attnres",
               "28l-attnres", "30l-attnres", "32l-attnres",
               "16l-lns-lr1e2", "20l-lns-lr1e2", "24l-lns-lr1e2", "26l-lns-lr1e2",
               "28l-lns-lr1e2", "30l-lns-lr1e2", "32l-lns-lr1e2",
               "16l-preln-lr2e3", "20l-preln-lr2e3", "24l-preln-lr2e3", "26l-preln-lr2e3",
               "28l-preln-lr2e3", "30l-preln-lr2e3", "32l-preln-lr2e3"]
LAYER_LABELS = {
    "16l": "L16",
    "20l": "L20",
    "24l": "L24",
    "26l": "L26",
    "28l": "L28",
    "30l": "L30",
    "32l": "L32",
    "16l-attnres": "L16-attnres",
    "20l-attnres": "L20-attnres",
    "24l-attnres": "L24-attnres",
    "26l-attnres": "L26-attnres",
    "28l-attnres": "L28-attnres",
    "30l-attnres": "L30-attnres",
    "32l-attnres": "L32-attnres",
    "16l-lns-lr1e2": "L16-lns-lr1e2",
    "20l-lns-lr1e2": "L20-lns-lr1e2",
    "24l-lns-lr1e2": "L24-lns-lr1e2",
    "26l-lns-lr1e2": "L26-lns-lr1e2",
    "28l-lns-lr1e2": "L28-lns-lr1e2",
    "30l-lns-lr1e2": "L30-lns-lr1e2",
    "32l-lns-lr1e2": "L32-lns-lr1e2",
    "16l-preln-lr2e3": "L16-preln-lr2e3",
    "20l-preln-lr2e3": "L20-preln-lr2e3",
    "24l-preln-lr2e3": "L24-preln-lr2e3",
    "26l-preln-lr2e3": "L26-preln-lr2e3",
    "28l-preln-lr2e3": "L28-preln-lr2e3",
    "30l-preln-lr2e3": "L30-preln-lr2e3",
    "32l-preln-lr2e3": "L32-preln-lr2e3",
}

METRICS = [
    ("stable_rank_per_layer",  "Stable rank  ||H||_F² / ||H||_2²"),
    ("erank_per_layer",        "Eff. rank  exp(H(p))"),
    ("l1_l2_ratio_per_layer",  "L1/L2 sval ratio  ||s||₁/||s||₂"),
]


def load_results(results_dir: Path) -> dict[str, dict]:
    """Returns {tag: results_dict} for all available variants."""
    data = {}
    for tag in LAYER_ORDER:
        path = results_dir / tag / "results.json"
        if not path.exists():
            print(f"[WARN] no results for {tag}, skipping")
            continue
        with open(path) as f:
            data[tag] = json.load(f)
    return data


def extract_method(tag: str) -> str:
    """Extract method from tag (base/attnres/lns-lr1e2/preln-lr2e3)."""
    if "attnres" in tag:
        return "attnres"
    elif "lns-lr1e2" in tag:
        return "lns-lr1e2"
    elif "preln-lr2e3" in tag:
        return "preln-lr2e3"
    else:
        return "base"


def plot_per_layer_on_ax(ax, data: dict[str, dict], metric_key: str, ylabel: str, log_y: bool = False):
    """Draw a per-layer plot on an existing axis."""
    cmap = matplotlib.colormaps["tab10"]
    tags = [t for t in LAYER_ORDER if t in data]
    
    # Group by method
    methods = {}
    for tag in tags:
        m = extract_method(tag)
        if m not in methods:
            methods[m] = []
        methods[m].append(tag)
    
    # Plot one line per method
    for color_idx, (method, method_tags) in enumerate(sorted(methods.items())):
        for tag in sorted(method_tags):
            values = np.array(data[tag].get(metric_key, []))
            if len(values) == 0:
                continue
            label = LAYER_LABELS[tag] if tag == method_tags[0] else None
            ax.plot(np.arange(len(values)), values, marker="o", markersize=3,
                    linewidth=1.8, label=label, color=cmap(color_idx), alpha=0.7)

    ax.set_xlabel("Layer index", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(f"{ylabel} — base vs attnres vs lns-lr1e2" + (" (log y)" if log_y else ""), fontsize=13, pad=8)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(True, alpha=0.3)


def plot_per_layer(data: dict[str, dict], metric_key: str, ylabel: str, output_path: Path, log_y: bool = False):
    """One line per method (all depths same color per method)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_per_layer_on_ax(ax, data, metric_key, ylabel, log_y=log_y)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_layer_combined(data: dict[str, dict], output_path: Path):
    """Combined 2x3 figure: linear row on top, log-y row on bottom."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=False)
    for col_idx, (metric_key, ylabel) in enumerate(METRICS):
        plot_per_layer_on_ax(axes[0, col_idx], data, metric_key, ylabel, log_y=False)
        plot_per_layer_on_ax(axes[1, col_idx], data, metric_key, ylabel, log_y=True)
        axes[0, col_idx].set_title(f"{ylabel} — linear y", fontsize=12, pad=6)
        axes[1, col_idx].set_title(f"{ylabel} — log y", fontsize=12, pad=6)
        axes[0, col_idx].legend(fontsize=8, ncol=3)
        axes[1, col_idx].legend(fontsize=8, ncol=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_layer_bar(data: dict[str, dict], metric_key: str, ylabel: str, output_path: Path):
    """Bar chart of final-layer value per variant, grouped by method."""
    tags = [t for t in LAYER_ORDER if t in data]
    cmap = matplotlib.colormaps["tab10"]
    
    # Group by method
    methods = {}
    for tag in tags:
        m = extract_method(tag)
        if m not in methods:
            methods[m] = []
        methods[m].append(tag)
    
    # Collect labels and finals
    all_labels = []
    all_finals = []
    all_colors = []
    
    for color_idx, (method, method_tags) in enumerate(sorted(methods.items())):
        for tag in sorted(method_tags):
            if metric_key in data[tag]:
                all_labels.append(LAYER_LABELS[tag])
                all_finals.append(data[tag][metric_key][-1])
                all_colors.append(cmap(color_idx))

    if not all_finals:
        return
        
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(all_labels, all_finals, color=all_colors, width=0.6, edgecolor="white")
    ax.set_xlabel("Model variant", fontsize=12)
    ax.set_ylabel(f"Final-layer {ylabel}", fontsize=11)
    ax.set_title(f"Final-layer {ylabel} — base vs attnres vs lns-lr1e2", fontsize=12, pad=8)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="/fast/dmartinez/depthbench/analysis/token_collapse",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/david/plots",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(results_dir)
    if not data:
        print("No results found — jobs may still be running.")
        return

    for metric_key, ylabel in METRICS:
        stem = metric_key.replace("_per_layer", "")
        plot_per_layer(data, metric_key, ylabel, output_dir / f"{stem}_per_layer.png")
        plot_per_layer(data, metric_key, ylabel, output_dir / f"{stem}_per_layer_logy.png", log_y=True)
        plot_final_layer_bar(data, metric_key, ylabel, output_dir / f"{stem}_final_layer.png")

    plot_per_layer_combined(data, output_dir / "combined_per_layer_6panel.png")

    # Summary table
    print(f"\n{'Model':<6}  {'stable_rank':>12}  {'erank':>10}  {'l1_l2':>10}")
    print("-" * 45)
    for tag in LAYER_ORDER:
        if tag not in data:
            continue
        d = data[tag]
        sr = d.get("stable_rank_per_layer", [float("nan")])[-1]
        er = d.get("erank_per_layer",       [float("nan")])[-1]
        l1 = d.get("l1_l2_ratio_per_layer", [float("nan")])[-1]
        print(f"{LAYER_LABELS[tag]:<6}  {sr:>12.3f}  {er:>10.3f}  {l1:>10.3f}")


if __name__ == "__main__":
    main()
