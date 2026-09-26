"""Angular distance between depth states.

d(i, j) = mean_tokens arccos(cos(z_i, z_j)) / pi for all pairs of depth states i < j,
where z_0 is the embedding output and z_l the state after block l-1. The depth states of
each architecture family are defined in `depth_probes.py`.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from analysis_utils import (
    add_model_args,
    autocast_context,
    iter_micro_batches,
    load_eval_input_ids,
    load_model_and_tokenizer,
    write_json,
)
from depth_probes import describe_architecture, forward_depth_states


@torch.no_grad()
def compute_angular_distance_matrix(
    model,
    input_ids: torch.Tensor,
    *,
    micro_batch_size: int,
    device: str,
    dtype: torch.dtype,
    include_embeddings: bool = True,
) -> np.ndarray:
    angle_sums = None
    num_tokens = 0
    num_batches = (input_ids.shape[0] + micro_batch_size - 1) // micro_batch_size
    for batch in tqdm(iter_micro_batches(input_ids, micro_batch_size, device), total=num_batches, desc="Angular distance"):
        with autocast_context(device, dtype):
            _, states = forward_depth_states(model, batch)
        if not include_embeddings:
            states = states[1:]
        units = torch.stack([F.normalize(state.reshape(-1, state.shape[-1]).float(), dim=-1) for state in states])
        if angle_sums is None:
            angle_sums = torch.zeros(len(states), len(states), dtype=torch.float64, device=units.device)
        for i in range(len(states) - 1):
            cosine = (units[i + 1 :] * units[i]).sum(dim=-1).clamp(-1.0, 1.0)
            angle_sums[i, i + 1 :] += torch.acos(cosine).sum(dim=-1).double()
        num_tokens += units.shape[1]

    matrix = (angle_sums / (num_tokens * np.pi)).cpu().numpy().astype(np.float32)
    matrix[np.tril_indices_from(matrix)] = np.nan
    return matrix


def convert_to_subsequent_layer_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row n-1 holds d(l, l+n) for every start layer l."""
    num_states = matrix.shape[0]
    subsequent = np.full((num_states - 1, num_states - 1), np.nan, dtype=np.float32)
    for i in range(num_states):
        for j in range(i + 1, num_states):
            subsequent[j - i - 1, i] = matrix[i, j]
    return subsequent


def plot_angular_distance_heatmap(matrix: np.ndarray, output_path: Path, title: str = "Angular Distance") -> dict:
    display = convert_to_subsequent_layer_matrix(matrix)[::-1]
    mask = np.isnan(display)
    finite = display[~mask]
    vmin, vmax = float(finite.min()), float(finite.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    num_rows, num_cols = display.shape
    plt.figure(figsize=(max(7.5, num_cols * 0.48 + 1.5), max(5.0, num_rows * 0.48 + 1.0)))
    sns.heatmap(
        display,
        mask=mask,
        cmap="viridis_r",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"shrink": 0.9},
        square=True,
    )
    plt.xlabel(r"Layer Index $\ell$", fontsize=16, fontweight="bold")
    plt.ylabel(r"Subsequent $n^{th}$ Layer", fontsize=16, fontweight="bold")
    plt.title(title, fontsize=22, pad=12)
    x_ticks = list(range(0, num_cols, max(1, num_cols // 4)))
    plt.xticks([pos + 0.5 for pos in x_ticks], [str(pos) for pos in x_ticks], rotation=0, fontsize=12)
    y_ticks = list(range(0, num_rows, max(1, num_rows // 4)))
    plt.yticks([pos + 0.5 for pos in y_ticks], [str(num_rows - pos) for pos in y_ticks], rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return {"colorbar_min": vmin, "colorbar_max": vmax}


def summarize_pairs(matrix: np.ndarray, *, smallest: bool, top_k: int = 10) -> list[dict]:
    pairs = [
        (i, j, float(matrix[i, j]))
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[1])
        if np.isfinite(matrix[i, j])
    ]
    pairs.sort(key=lambda item: item[2], reverse=not smallest)
    return [{"layer_i": i, "layer_j": j, "value": value} for i, j, value in pairs[:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Compute layer-wise angular distance between depth states")
    add_model_args(parser, num_samples=128, seq_length=512)
    parser.add_argument(
        "--exclude-embeddings",
        action="store_true",
        help="Drop z_0 (embedding output); by default it is included",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, dtype, checkpoint_dir = load_model_and_tokenizer(
        args.model_path, args.device, args.dtype, args.tokenizer_id, args.max_sequence_length
    )
    input_ids, sample_info = load_eval_input_ids(args, tokenizer)
    architecture = describe_architecture(model)
    print(f"Loaded {checkpoint_dir}: {architecture}")

    matrix = compute_angular_distance_matrix(
        model,
        input_ids,
        micro_batch_size=args.micro_batch_size,
        device=device,
        dtype=dtype,
        include_embeddings=not args.exclude_embeddings,
    )
    np.save(output_dir / "angular_distance_matrix.npy", matrix)
    np.save(output_dir / "angular_distance_subsequent_matrix.npy", convert_to_subsequent_layer_matrix(matrix))
    colorbar = plot_angular_distance_heatmap(matrix, output_dir / "angular_distance_heatmap.png")

    write_json(
        output_dir / "results.json",
        {
            "score_type": "angular_distance",
            "definition": "d(i, j) = mean_tokens arccos(cos(z_i, z_j)) / pi over depth states i < j",
            "model_path": args.model_path,
            "resolved_model_path": checkpoint_dir,
            "architecture": architecture,
            **sample_info,
            "exclude_embeddings": args.exclude_embeddings,
            "hidden_state_start_index": 1 if args.exclude_embeddings else 0,
            **colorbar,
            "smallest_angular_distances": summarize_pairs(matrix, smallest=True),
            "largest_angular_distances": summarize_pairs(matrix, smallest=False),
            "angular_distance_matrix": np.nan_to_num(matrix, nan=-1.0).tolist(),
        },
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
