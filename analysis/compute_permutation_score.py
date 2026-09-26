"""Pairwise weight-swap (permutation) score.

P(i, j) = |L_swap(i, j) - L| / L, where L is the next-token loss of the original model and
L_swap(i, j) the loss after exchanging the learned weights of blocks i and j while the
architecture stays fixed. Higher scores mean less interchangeable, more position-specific
blocks. What a block's weights comprise per architecture is defined in `depth_probes.py`.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tqdm import tqdm

from analysis_utils import add_model_args, get_decoder_layers, load_eval_input_ids, load_model_and_tokenizer, write_json
from depth_probes import describe_architecture, mean_lm_loss, swapped_blocks


def compute_permutation_scores(
    model,
    input_ids,
    layer_pairs: list[tuple[int, int]],
    *,
    micro_batch_size: int,
    device: str,
    dtype,
) -> tuple[float, dict[tuple[int, int], float]]:
    loss_kwargs = dict(micro_batch_size=micro_batch_size, device=device, dtype=dtype)
    baseline_loss = mean_lm_loss(model, input_ids, **loss_kwargs)
    scores = {}
    for i, j in tqdm(layer_pairs, desc="Permutation score"):
        with swapped_blocks(model, i, j):
            swapped_loss = mean_lm_loss(model, input_ids, **loss_kwargs)
        scores[(i, j)] = abs(swapped_loss - baseline_loss) / baseline_loss
    return baseline_loss, scores


def plot_permutation_scores(scores: dict[tuple[int, int], float], num_layers: int, output_path: Path) -> None:
    matrix = np.full((num_layers, num_layers), np.nan)
    for (i, j), score in scores.items():
        matrix[i, j] = score
    mask = np.tril(np.ones_like(matrix, dtype=bool)) | np.isnan(matrix)
    finite = matrix[~mask]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    # Shift the neutral colour downward so mid-to-high values appear warmer.
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vmin + 0.4 * (vmax - vmin), vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("permutation_score", ["#0b3c78", "#f7f3ec", "#7b001c"])
    cmap.set_bad(color="white")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        matrix,
        cmap=cmap,
        norm=norm,
        mask=mask,
        cbar_kws={"label": "Permutation Score"},
        xticklabels=range(num_layers),
        yticklabels=range(num_layers),
        linewidths=0.3,
        linecolor=(1.0, 1.0, 1.0, 0.45),
        square=True,
    )
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.title("Permutation Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_layer_pairs(num_layers: int, anchor_layers: list[int] | None) -> list[tuple[int, int]]:
    if not anchor_layers:
        return [(i, j) for i in range(num_layers) for j in range(i + 1, num_layers)]
    pairs = {tuple(sorted((anchor, other))) for anchor in anchor_layers for other in range(num_layers) if other != anchor}
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser(description="Compute pairwise weight-swap permutation scores")
    add_model_args(parser, num_samples=12, seq_length=256)
    parser.add_argument(
        "--anchor-layer",
        type=int,
        action="append",
        default=None,
        help="Only test pairs containing this layer; can be repeated. Defaults to all pairs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, dtype, checkpoint_dir = load_model_and_tokenizer(
        args.model_path, args.device, args.dtype, args.tokenizer_id, args.max_sequence_length
    )
    input_ids, sample_info = load_eval_input_ids(args, tokenizer)
    architecture = describe_architecture(model)
    num_layers = len(get_decoder_layers(model))
    print(f"Loaded {checkpoint_dir}: {architecture}")

    layer_pairs = build_layer_pairs(num_layers, args.anchor_layer)
    baseline_loss, scores = compute_permutation_scores(
        model,
        input_ids,
        layer_pairs,
        micro_batch_size=args.micro_batch_size,
        device=device,
        dtype=dtype,
    )
    plot_permutation_scores(scores, num_layers, output_dir / "permutation_scores_heatmap.png")

    layer_scores = [[score for pair, score in scores.items() if layer in pair] for layer in range(num_layers)]
    layer_mean = [float(np.mean(values)) if values else float("nan") for values in layer_scores]
    write_json(
        output_dir / "results.json",
        {
            "score_type": "permutation_score",
            "definition": "P(i, j) = |L_swap(i, j) - L| / L with L the mean next-token loss",
            "model_path": args.model_path,
            "resolved_model_path": checkpoint_dir,
            "architecture": architecture,
            **sample_info,
            "baseline_loss": baseline_loss,
            "num_layers": num_layers,
            "tested_layer_pairs": [list(pair) for pair in layer_pairs],
            "mean_permutation_score": float(np.mean(list(scores.values()))),
            "layer_mean_permutation_scores": layer_mean,
            "permutation_scores": {f"{i}_{j}": score for (i, j), score in scores.items()},
        },
    )
    print(f"Baseline loss {baseline_loss:.4f}, mean permutation score {np.mean(list(scores.values())):.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
