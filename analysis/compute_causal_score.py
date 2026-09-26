"""Layer-skip causal score.

For every skipped block s and downstream block l > s:

    C(s, l) = mean_tokens ||u_l^{skip s} - u_l||_2 / max(||u_l||_2, eps),   u_l = z_{l+1} - z_l,

i.e. how much removing block s changes the update written by a later block. Depth states
and the per-architecture skip rule are defined in `depth_probes.py`.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tqdm import tqdm

from analysis_utils import (
    add_model_args,
    autocast_context,
    get_decoder_layers,
    iter_micro_batches,
    load_eval_input_ids,
    load_model_and_tokenizer,
    write_json,
)
from depth_probes import describe_architecture, forward_depth_states, skip_block


def _updates(states: list[torch.Tensor]) -> torch.Tensor:
    """Stack u_l = z_{l+1} - z_l as (num_layers, tokens, width) in float32."""
    flat = torch.stack([state.reshape(-1, state.shape[-1]).float() for state in states])
    return flat[1:] - flat[:-1]


@torch.no_grad()
def compute_causal_score_matrix(
    model,
    input_ids: torch.Tensor,
    *,
    micro_batch_size: int,
    device: str,
    dtype: torch.dtype,
    eps: float = 1e-8,
) -> np.ndarray:
    num_layers = len(get_decoder_layers(model))
    ratio_sums = torch.zeros(num_layers, num_layers, dtype=torch.float64)
    num_tokens = 0
    num_batches = (input_ids.shape[0] + micro_batch_size - 1) // micro_batch_size
    for batch in tqdm(iter_micro_batches(input_ids, micro_batch_size, device), total=num_batches, desc="Causal score"):
        with autocast_context(device, dtype):
            _, states = forward_depth_states(model, batch)
        baseline = _updates(states)
        baseline_norm = torch.linalg.vector_norm(baseline, dim=-1).clamp_min(eps)
        for skipped in range(num_layers - 1):
            with skip_block(model, skipped), autocast_context(device, dtype):
                _, skipped_states = forward_depth_states(model, batch)
            downstream = slice(skipped + 1, num_layers)
            change = torch.linalg.vector_norm(_updates(skipped_states)[downstream] - baseline[downstream], dim=-1)
            ratio_sums[skipped, downstream] += (change / baseline_norm[downstream]).sum(dim=-1).double().cpu()
        num_tokens += baseline.shape[1]

    matrix = (ratio_sums / num_tokens).numpy().astype(np.float32)
    matrix[np.tril_indices_from(matrix)] = np.nan
    return matrix


def plot_causal_scores(matrix: np.ndarray, output_path: Path, *, title: str = "Causal Score") -> dict:
    mask = np.tril(np.ones_like(matrix, dtype=bool))
    finite = matrix[np.isfinite(matrix)]
    vmin = 0.0
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    if np.isclose(vmax, vmin):
        vmax = vmin + 1e-6
    vcenter = vmin + 0.4 * (vmax - vmin)
    cmap = LinearSegmentedColormap.from_list("causal_score", ["#ffffff", "#9bd4e5", "#0072b2", "#08306b"])
    cmap.set_bad(color="white")

    num_layers = matrix.shape[0]
    plt.figure(figsize=(max(7.5, num_layers * 0.34 + 1.8), max(6.5, num_layers * 0.34 + 1.4)))
    sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
        linewidths=0.25,
        linecolor=(1, 1, 1, 0.35),
        square=True,
        cbar_kws={"label": "Causal score"},
    )
    ticks = list(range(0, num_layers, max(1, num_layers // 4)))
    plt.xticks([pos + 0.5 for pos in ticks], [str(pos) for pos in ticks], rotation=0)
    plt.yticks([pos + 0.5 for pos in ticks], [str(pos) for pos in ticks], rotation=0)
    plt.xlabel("Affected Layer", fontsize=13, fontweight="bold")
    plt.ylabel("Removed Layer", fontsize=13, fontweight="bold")
    plt.title(title, fontsize=18, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return {"colorbar_min": vmin, "colorbar_center": vcenter, "colorbar_max": vmax}


def upper_triangle(matrix: np.ndarray) -> list[tuple[int, int, float]]:
    return [
        (i, j, float(matrix[i, j]))
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[1])
        if np.isfinite(matrix[i, j])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the layer-skip causal score heatmap")
    add_model_args(parser, num_samples=16, seq_length=256)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, dtype, checkpoint_dir = load_model_and_tokenizer(
        args.model_path, args.device, args.dtype, args.tokenizer_id, args.max_sequence_length
    )
    input_ids, sample_info = load_eval_input_ids(args, tokenizer)
    architecture = describe_architecture(model)
    print(f"Loaded {checkpoint_dir}: {architecture}")

    matrix = compute_causal_score_matrix(
        model,
        input_ids,
        micro_batch_size=args.micro_batch_size,
        device=device,
        dtype=dtype,
        eps=args.eps,
    )
    np.save(output_dir / "causal_score_matrix.npy", matrix)
    colorbar = plot_causal_scores(matrix, output_dir / "causal_score_heatmap.png")

    entries = upper_triangle(matrix)
    values = np.asarray([value for _, _, value in entries])
    top = sorted(entries, key=lambda item: item[2], reverse=True)[:10]
    write_json(
        output_dir / "results.json",
        {
            "score_type": "causal_score",
            "definition": (
                "C(s, l) = mean_tokens ||u_l^{skip s} - u_l||_2 / max(||u_l||_2, eps) for l > s, "
                "where u_l = z_{l+1} - z_l"
            ),
            "model_path": args.model_path,
            "resolved_model_path": checkpoint_dir,
            "architecture": architecture,
            **sample_info,
            "eps": args.eps,
            **colorbar,
            "mean_causal_score": float(values.mean()) if values.size else float("nan"),
            "max_causal_score": float(values.max()) if values.size else float("nan"),
            "top_causal_scores": [
                {"skipped_layer": i, "observed_update_layer": j, "value": value} for i, j, value in top
            ],
            "causal_scores": {f"{i}_{j}": value for i, j, value in entries},
        },
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
