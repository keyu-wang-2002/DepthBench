"""compute_token_collapse.py — per-layer singular-value metrics of the token matrix H^(l).

For each layer l, H^(l) in R^{T x D} (no centering). Three metrics from one SVD:

    stable_rank(H)  = ||H||_F^2 / ||H||_2^2  =  sum(s^2) / s_max^2
    erank(H)        = exp(-sum_i p_i log p_i),  p_i = s_i^2 / sum(s^2)   [Roy & Vetterli 2007]
    l1_l2_ratio(H)  = ||s||_1 / ||s||_2  =  sum(s) / sqrt(sum(s^2))

All three go to 1 when tokens collapse to a single direction, and to
min(T, D) when tokens are maximally spread.

Usage:
    python analysis/compute_token_collapse.py \\
        --model_path /path/to/checkpoint/step1000 \\
        --output_dir ./token_collapse_results \\
        [--num_samples 256] [--seq_length 512] [--device auto] [--dtype auto]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure analysis_utils is importable when run from repo root or from
# the analysis/ directory.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from analysis_utils import (
    autocast_context,
    build_sample_batch,
    get_decoder_layers,
    load_model_and_tokenizer,
    model_forward,
    normalize_layer_output,
)


# ---------------------------------------------------------------------------
# Hidden state collection (mirrors compute_angular_distance.py)
# ---------------------------------------------------------------------------

def collect_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    micro_batch_size: int = 4,
) -> List[torch.Tensor]:
    """Returns one tensor per layer (including embedding), shape (B, T, D), on CPU."""
    model.eval()
    batch_size = input_ids.shape[0]
    mb = min(micro_batch_size, batch_size)
    n_mb = (batch_size + mb - 1) // mb
    accum: list[torch.Tensor | None] = []

    with torch.no_grad():
        for i in range(n_mb):
            ids = input_ids[i * mb : (i + 1) * mb].to(device)
            mask = attention_mask[i * mb : (i + 1) * mb].to(device)
            with autocast_context(device, model_dtype):
                out = model_forward(
                    model,
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            for j, hs in enumerate(out.hidden_states):
                val = normalize_layer_output(hs)
                if val is None:
                    continue
                val = val.detach().cpu()
                if i == 0:
                    accum.append(val)
                else:
                    accum[j] = torch.cat([accum[j], val], dim=0)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return [h for h in accum if h is not None]


# ---------------------------------------------------------------------------
# Effective rank per layer
# ---------------------------------------------------------------------------

def compute_sval_metrics(H: torch.Tensor, device: str = "cpu") -> dict:
    """Compute stable_rank, erank, and l1_l2_ratio from one SVD of H in R^{T x D}.

    stable_rank  = sum(s^2) / s_max^2          (Frobenius^2 / spectral^2)
    erank        = exp(-sum p_i log p_i),  p_i = s_i^2 / sum(s^2)   [Roy & Vetterli 2007]
    l1_l2_ratio  = sum(s) / sqrt(sum(s^2))     (L1/L2 norm of svals)

    All equal 1 at full collapse and min(T,D) at maximum spread.
    No centering — operates on H directly.
    """
    H_f = H.to(device=device, dtype=torch.float32)
    s = torch.linalg.svdvals(H_f)               # (min(T, D),)
    s2 = s ** 2
    total_s2 = s2.sum()
    if total_s2 < 1e-12:
        return {"stable_rank": 1.0, "erank": 1.0, "l1_l2_ratio": 1.0}

    # stable rank
    stable_rank = float(total_s2 / s2[0])

    # entropy-based effective rank
    p = (s2 / total_s2).clamp(min=1e-12)
    erank = float(torch.exp(-(p * torch.log(p)).sum()))

    # L1/L2 ratio of singular values
    l1_l2_ratio = float(s.sum() / total_s2.sqrt())

    return {"stable_rank": stable_rank, "erank": erank, "l1_l2_ratio": l1_l2_ratio}


def compute_per_layer_metrics(
    hidden_states: List[torch.Tensor],
    attention_mask: torch.Tensor,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Returns dict of arrays, each shape (num_layers,), averaged over samples.

    Keys: 'stable_rank', 'erank', 'l1_l2_ratio'.
    Each sample's token matrix is the (T_valid, D) slice of non-padding tokens.
    Hidden states live on CPU; each slice is moved to `device` for the SVD.
    """
    accum: dict[str, list[float]] = {"stable_rank": [], "erank": [], "l1_l2_ratio": []}
    valid_mask = attention_mask.bool().cpu()  # (B, T)

    for layer_hs in tqdm(hidden_states, desc="Computing metrics per layer"):
        # layer_hs: (B, T, D) on CPU
        B = layer_hs.shape[0]
        sample: dict[str, list[float]] = {k: [] for k in accum}
        for b in range(B):
            valid_tok = valid_mask[b]        # (T,)
            H = layer_hs[b][valid_tok]       # (T_valid, D)
            if H.shape[0] < 2:
                continue
            m = compute_sval_metrics(H, device=device)
            for k, v in m.items():
                sample[k].append(v)
        for k in accum:
            accum[k].append(float(np.mean(sample[k])) if sample[k] else float("nan"))

    return {k: np.array(v, dtype=np.float32) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_METRIC_META = {
    "stable_rank":  ("Stable rank  ||H||_F² / ||H||_2²",  "tab:blue"),
    "erank":        ("Eff. rank  exp(H(p))",               "tab:orange"),
    "l1_l2_ratio":  ("L1/L2 sval ratio  ||s||₁/||s||₂",   "tab:green"),
}


def plot_metrics(
    metrics: dict[str, np.ndarray],
    output_path: Path,
    title: str = "Token representation metrics per layer",
):
    n = len(metrics)
    n_layers = len(next(iter(metrics.values())))
    layers = np.arange(n_layers)
    w = max(6, n_layers * 0.35 + 2)

    fig, axes = plt.subplots(1, n, figsize=(w * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, metrics.items()):
        label, color = _METRIC_META.get(key, (key, "gray"))
        ax.plot(layers, values, marker="o", linewidth=1.8, markersize=4, color=color)
        ax.set_xlabel("Layer index", fontsize=12)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, pad=6)
        ax.grid(True, alpha=0.35)
        ax.set_xticks(layers[::max(1, n_layers // 8)])

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute per-layer token effective rank")
    parser.add_argument("--model_path", type=str, required=True, help="OLMo-core checkpoint dir")
    parser.add_argument("--output_dir", type=str, default="./token_collapse_results")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-embeddings", action="store_true",
                        help="Exclude embedding layer (layer 0) from plot/results")
    parser.add_argument("--tokenizer-id", type=str, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, model_dtype, resolved_path = load_model_and_tokenizer(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
    )
    decoder_layers = get_decoder_layers(model)
    print(f"Loaded model from: {resolved_path}")
    print(f"Model has {len(decoder_layers)} decoder layers")

    sample_data = build_sample_batch(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        text_file=args.text_file,
        prompts=args.prompt,
    )

    hidden_states = collect_hidden_states(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
        model_dtype=model_dtype,
    )

    analyzed = hidden_states[1:] if args.exclude_embeddings else hidden_states
    start_idx = 1 if args.exclude_embeddings else 0

    metrics = compute_per_layer_metrics(analyzed, sample_data["attention_mask"], device=device)

    for key, values in metrics.items():
        np.save(output_dir / f"{key}_per_layer.npy", values)

    # keep legacy name for erank so downstream code still works
    np.save(output_dir / "erank_per_layer.npy", metrics["erank"])

    plot_metrics(
        metrics,
        output_dir / "metrics_per_layer.png",
        title=f"Token metrics — {Path(args.model_path).name}",
    )

    print(f"\n{'Layer':>6}  {'stable_rank':>12}  {'erank':>10}  {'l1_l2_ratio':>12}")
    print("-" * 48)
    for i in range(len(metrics["erank"])):
        print(f"  {start_idx + i:3d}    "
              f"{metrics['stable_rank'][i]:>12.3f}  "
              f"{metrics['erank'][i]:>10.3f}  "
              f"{metrics['l1_l2_ratio'][i]:>12.3f}")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_path,
        "num_decoder_layers": len(decoder_layers),
        "num_analyzed_layers": len(analyzed),
        "exclude_embeddings": args.exclude_embeddings,
        "hidden_state_start_index": start_idx,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "seed": args.seed,
    }
    for key, values in metrics.items():
        results[f"{key}_per_layer"] = values.tolist()
        results[f"{key}_mean"]       = float(np.nanmean(values))
        results[f"{key}_min"]        = float(np.nanmin(values))
        results[f"{key}_max"]        = float(np.nanmax(values))
        results[f"{key}_final_layer"]= float(values[-1]) if len(values) else None
    # legacy key
    results["erank_per_layer"] = metrics["erank"].tolist()

    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
