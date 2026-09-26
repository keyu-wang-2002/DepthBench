"""Logit lens: decode every depth state with the final norm and LM head.

For each depth state l = 0..L the probe distribution is p_l = softmax(W_LM Norm(r_l)), where
r_l is the single-stream state the next consumer reads (see `depth_probes.py`). Reported
per layer, averaged over all next-token positions:

    early_exit_ce  cross entropy of p_l against the next token
    kl_to_final    KL(p_final || p_l), p_final being the model's actual output distribution
    top5_overlap   |top5(p_l) ∩ top5(p_final)| / 5
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
from depth_probes import decode_readout, describe_architecture, forward_readout_states, next_token_loss_sum

METRICS = ("early_exit_ce", "kl_to_final", "top5_overlap")


@torch.no_grad()
def compute_logit_lens(
    model,
    input_ids: torch.Tensor,
    *,
    micro_batch_size: int,
    device: str,
    dtype: torch.dtype,
    top_k: int = 5,
) -> tuple[float, list[dict[str, float]]]:
    num_states = len(get_decoder_layers(model)) + 1
    sums = np.zeros((num_states, len(METRICS)), dtype=np.float64)
    final_loss_sum, num_tokens = 0.0, 0
    num_batches = (input_ids.shape[0] + micro_batch_size - 1) // micro_batch_size
    for batch in tqdm(iter_micro_batches(input_ids, micro_batch_size, device), total=num_batches, desc="Logit lens"):
        with autocast_context(device, dtype):
            final_logits, readouts = forward_readout_states(model, batch)
        loss_sum, batch_tokens = next_token_loss_sum(final_logits, batch)
        final_loss_sum += loss_sum
        num_tokens += batch_tokens

        targets = batch[:, 1:].reshape(-1)
        final_log_probs = F.log_softmax(final_logits[:, :-1].float(), dim=-1).flatten(0, 1)
        final_probs = final_log_probs.exp()
        final_top = final_log_probs.topk(top_k, dim=-1).indices
        for idx, readout in enumerate(readouts):
            with autocast_context(device, dtype):
                logits = decode_readout(model, readout)
            log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1).flatten(0, 1)
            probe_top = log_probs.topk(top_k, dim=-1).indices
            sums[idx, 0] += F.nll_loss(log_probs, targets, reduction="sum").item()
            sums[idx, 1] += (final_probs * (final_log_probs - log_probs)).sum().item()
            overlap = (probe_top.unsqueeze(-1) == final_top.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
            sums[idx, 2] += (overlap / top_k).sum().item()
            del logits, log_probs

    means = sums / num_tokens
    rows = [
        {"layer": idx, "position": "emb" if idx == 0 else f"block_{idx - 1:02d}", **dict(zip(METRICS, map(float, row)))}
        for idx, row in enumerate(means)
    ]
    return final_loss_sum / num_tokens, rows


def plot_logit_lens(rows: list[dict], final_loss: float, output_path: Path, title: str) -> None:
    layers = [row["layer"] for row in rows]
    labels = {
        "early_exit_ce": "Early-exit CE",
        "kl_to_final": r"KL($p_{\mathrm{final}} \,\|\, p_\ell$)",
        "top5_overlap": "Top-5 overlap with final",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, metric in zip(axes, METRICS):
        ax.plot(layers, [row[metric] for row in rows], marker="o", markersize=3, linewidth=1.6)
        if metric == "early_exit_ce":
            ax.axhline(final_loss, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="final")
            ax.legend(fontsize=8)
        ax.set_xlabel(r"Depth state $\ell$")
        ax.set_ylabel(labels[metric])
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute logit-lens metrics for every depth state")
    add_model_args(parser, num_samples=32, seq_length=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, dtype, checkpoint_dir = load_model_and_tokenizer(
        args.model_path, args.device, args.dtype, args.tokenizer_id, args.max_sequence_length
    )
    input_ids, sample_info = load_eval_input_ids(args, tokenizer)
    architecture = describe_architecture(model)
    print(f"Loaded {checkpoint_dir}: {architecture}")

    final_loss, rows = compute_logit_lens(
        model, input_ids, micro_batch_size=args.micro_batch_size, device=device, dtype=dtype
    )
    with (output_dir / "logit_lens.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_logit_lens(rows, final_loss, output_dir / "logit_lens.png", title=f"Logit lens ({architecture['family']})")

    write_json(
        output_dir / "results.json",
        {
            "score_type": "logit_lens",
            "definition": (
                "p_l = softmax(W_LM Norm(r_l)) with r_l the state read by the next consumer of block l-1; "
                "early_exit_ce = CE(p_l, next token), kl_to_final = KL(p_final || p_l), "
                "top5_overlap = |top5(p_l) & top5(p_final)| / 5"
            ),
            "model_path": args.model_path,
            "resolved_model_path": checkpoint_dir,
            "architecture": architecture,
            **sample_info,
            "final_loss": final_loss,
            "layers": rows,
        },
    )
    print(f"Final loss {final_loss:.4f}; results saved to {output_dir}")


if __name__ == "__main__":
    main()
