import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from analysis_utils import (
    build_sample_batch,
    get_decoder_layers,
    is_moe_model,
    load_model_and_tokenizer,
    normalize_layer_output,
    replace_decoder_layer,
)
from layer_score_utils import collect_layer_input_output_pairs, compute_mean_loss


def fit_linear_mapping(inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = inputs.float()
    outputs = outputs.float()
    inputs_with_bias = torch.cat([inputs, torch.ones(inputs.shape[0], 1, dtype=inputs.dtype)], dim=1)
    params = torch.linalg.lstsq(inputs_with_bias, outputs).solution
    weights = params[:-1, :]
    bias = params[-1, :]
    return weights, bias


class LinearDecoderLayer(nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        hidden_size = weights.shape[0]
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        with torch.no_grad():
            self.linear.weight.copy_(weights.T.to(dtype=dtype))
            self.linear.bias.copy_(bias.to(dtype=dtype))
        self.to(device)
        self.eval()

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return self.linear(normalize_layer_output(hidden_states))


def compute_loss_with_layer_replacement(
    model,
    layer_idx: int,
    replacement_layer: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
) -> float:
    with replace_decoder_layer(model, layer_idx, replacement_layer):
        return compute_mean_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
            micro_batch_size_limit=2,
        )


def compute_layer_usefulness_linear_approximation(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    baseline_loss: float,
    num_layers: int,
) -> Dict[int, Dict[str, float]]:
    print("Computing layer usefulness via linear approximation...")
    layer_metrics = {}

    for layer_idx in tqdm(range(num_layers), desc="Computing layer usefulness"):
        inputs, outputs = collect_layer_input_output_pairs(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
            layer_idx=layer_idx,
            micro_batch_size_limit=2,
        )
        print(
            f"Layer {layer_idx}: collected {inputs.shape[0]} samples, input_dim={inputs.shape[1]}, output_dim={outputs.shape[1]}"
        )
        weights, bias = fit_linear_mapping(inputs, outputs)
        replacement_layer = LinearDecoderLayer(
            weights,
            bias,
            device=device,
            dtype=model_dtype,
        )
        linear_loss = compute_loss_with_layer_replacement(
            model=model,
            layer_idx=layer_idx,
            replacement_layer=replacement_layer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
        )

        loss_increase = linear_loss - baseline_loss
        loss_ratio = linear_loss / baseline_loss if baseline_loss > 0 else float("inf")
        layer_metrics[layer_idx] = {
            "loss_increase": float(loss_increase),
            "loss_ratio": float(loss_ratio),
            "linear_loss": float(linear_loss),
            "significant_increase": bool(loss_ratio > 1.1),
        }

    return layer_metrics


def compute_global_usefulness_score(layer_metrics: Dict[int, Dict[str, float]], num_layers: int) -> Dict[str, float]:
    significant_count = sum(1 for metrics in layer_metrics.values() if metrics["significant_increase"])
    return {
        "global_usefulness_score": float(significant_count / num_layers if num_layers > 0 else 0.0),
        "num_significant_layers": int(significant_count),
        "total_layers": int(num_layers),
        "mean_loss_increase": float(np.mean([m["loss_increase"] for m in layer_metrics.values()])),
        "mean_loss_ratio": float(np.mean([m["loss_ratio"] for m in layer_metrics.values()])),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute layer usefulness via linear approximation")
    parser.add_argument("--model_path", type=str, required=True, help="OLMo-core checkpoint dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./layer_usefulness_linear_results",
        help="Directory to save results",
    )
    parser.add_argument("--num_samples", type=int, default=16, help="Number of token chunks to evaluate")
    parser.add_argument("--seq_length", type=int, default=256, help="Token chunk length")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto/cpu/cuda/cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Load dtype for the model",
    )
    parser.add_argument("--text-file", type=str, default=None, help="Optional UTF-8 text file used to build samples")
    parser.add_argument("--prompt", action="append", default=None, help="Optional prompt text; can be repeated")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--tokenizer-id", type=str, default=None, help="Optional tokenizer override")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Optional tokenizer model_max_length override",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Layer Usefulness via Linear Approximation")
    print("=" * 60)

    model, tokenizer, device, model_dtype, resolved_model_path = load_model_and_tokenizer(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
    )
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    print(f"Loaded model from: {resolved_model_path}")
    print(f"Number of layers: {num_layers}")
    if is_moe_model(model):
        print("Warning: this approximation is mainly designed for dense decoder layers.")

    sample_data = build_sample_batch(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        text_file=args.text_file,
        prompts=args.prompt,
    )

    print("Computing baseline loss...")
    baseline_loss = compute_mean_loss(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
        model_dtype=model_dtype,
        micro_batch_size_limit=2,
    )
    print(f"\nBaseline loss: {baseline_loss:.4f}")

    layer_metrics = compute_layer_usefulness_linear_approximation(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
        model_dtype=model_dtype,
        baseline_loss=baseline_loss,
        num_layers=num_layers,
    )
    global_metrics = compute_global_usefulness_score(layer_metrics, num_layers)

    print("\n" + "=" * 60)
    print("Global Usefulness Score")
    print("=" * 60)
    print(f"Global Usefulness Score: {global_metrics['global_usefulness_score']:.4f}")
    print(
        f"Layers with >10% loss increase: {global_metrics['num_significant_layers']}/{global_metrics['total_layers']}"
    )
    print(f"Mean Loss Increase: {global_metrics['mean_loss_increase']:.4f}")
    print(f"Mean Loss Ratio: {global_metrics['mean_loss_ratio']:.4f}")

    print("\n" + "=" * 60)
    print("Layer-wise Results")
    print("=" * 60)
    print(f"{'Layer':<10} {'Loss Increase':<15} {'Loss Ratio':<15} {'Significant':<15}")
    print("-" * 60)
    for layer_idx in sorted(layer_metrics):
        metrics = layer_metrics[layer_idx]
        significant = "Yes" if metrics["significant_increase"] else "No"
        print(f"{layer_idx:<10} {metrics['loss_increase']:<15.4f} {metrics['loss_ratio']:<15.4f} {significant:<15}")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "baseline_loss": float(baseline_loss),
        "num_layers": num_layers,
        "global_metrics": global_metrics,
        "layer_metrics": layer_metrics,
    }

    output_file = output_dir / "layer_usefulness_linear.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    csv_file = output_dir / "layer_usefulness_linear.csv"
    with csv_file.open("w", encoding="utf-8") as f:
        f.write("layer,loss_increase,loss_ratio,linear_loss,significant_increase\n")
        for layer_idx in sorted(layer_metrics):
            metrics = layer_metrics[layer_idx]
            f.write(
                f"{layer_idx},{metrics['loss_increase']},{metrics['loss_ratio']},"
                f"{metrics['linear_loss']},{metrics['significant_increase']}\n"
            )
    print(f"CSV saved to: {csv_file}")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
