import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from analysis_utils import (
    autocast_context,
    build_sample_batch,
    get_decoder_layers,
    get_model_backend,
    is_moe_model,
    load_model_and_tokenizer,
    model_forward,
    normalize_layer_output,
    replace_decoder_layer,
)


def compute_baseline_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
) -> float:
    print("Computing baseline loss...")
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(2, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
    total_loss = 0.0

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            micro_labels = input_ids[start_idx:end_idx].to(device)

            with autocast_context(device, model_dtype):
                outputs = model_forward(
                    model,
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    labels=micro_labels,
                    use_cache=False,
                )

            total_loss += outputs.loss.item() * (end_idx - start_idx)

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return total_loss / batch_size


def collect_layer_input_output(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(2, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

            with autocast_context(device, model_dtype):
                outputs = model_forward(
                    model,
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            layer_input = normalize_layer_output(outputs.hidden_states[layer_idx])
            layer_output = normalize_layer_output(outputs.hidden_states[layer_idx + 1])
            if layer_input is None or layer_output is None:
                continue

            all_inputs.append(layer_input.flatten(0, 1).float().cpu())
            all_outputs.append(layer_output.flatten(0, 1).float().cpu())

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    inputs = torch.cat(all_inputs, dim=0)
    outputs = torch.cat(all_outputs, dim=0)
    print(
        f"Layer {layer_idx}: collected {inputs.shape[0]} samples, input_dim={inputs.shape[1]}, output_dim={outputs.shape[1]}"
    )
    return inputs, outputs


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
        return_tuple: bool,
    ):
        super().__init__()
        hidden_size = weights.shape[0]
        self.return_tuple = return_tuple
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)
        with torch.no_grad():
            self.linear.weight.copy_(weights.T.to(dtype=dtype))
            self.linear.bias.copy_(bias.to(dtype=dtype))
        self.to(device)
        self.eval()

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = normalize_layer_output(hidden_states)
        output = self.linear(hidden_states)
        if not self.return_tuple:
            return output
        outputs = [output]
        if kwargs.get("output_attentions", False):
            outputs.append(None)
        return tuple(outputs)


def compute_loss_with_layer_replacement(
    model,
    layer_idx: int,
    replacement_layer: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
) -> float:
    batch_size = input_ids.shape[0]
    micro_batch_size = min(2, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
    loss_accum = 0.0

    with replace_decoder_layer(model, layer_idx, replacement_layer):
        with torch.no_grad():
            for mb_idx in range(num_micro_batches):
                start_idx = mb_idx * micro_batch_size
                end_idx = min(start_idx + micro_batch_size, batch_size)

                micro_input_ids = input_ids[start_idx:end_idx].to(device)
                micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
                micro_labels = input_ids[start_idx:end_idx].to(device)

                with autocast_context(device, model_dtype):
                    outputs = model_forward(
                        model,
                        input_ids=micro_input_ids,
                        attention_mask=micro_attention_mask,
                        labels=micro_labels,
                        use_cache=False,
                    )

                loss_accum += outputs.loss.item() * (end_idx - start_idx)

                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    return loss_accum / batch_size


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
        inputs, outputs = collect_layer_input_output(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
            layer_idx=layer_idx,
        )
        weights, bias = fit_linear_mapping(inputs, outputs)
        replacement_layer = LinearDecoderLayer(
            weights,
            bias,
            device=device,
            dtype=model_dtype,
            return_tuple=get_model_backend(model) != "olmo_core",
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
    parser.add_argument("--model_path", type=str, required=True, help="HF model dir or OLMo checkpoint dir")
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
        help="Load dtype for the HF model",
    )
    parser.add_argument("--text-file", type=str, default=None, help="Optional UTF-8 text file used to build samples")
    parser.add_argument("--prompt", action="append", default=None, help="Optional prompt text; can be repeated")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--tokenizer-id", type=str, default=None, help="Optional tokenizer ID used during conversion")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Optional max_position_embeddings override during conversion",
    )
    parser.add_argument(
        "--skip-conversion-validation",
        action="store_true",
        help="Skip logits validation when auto-converting OLMo checkpoints to HF",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        default="auto",
        choices=["auto", "olmo_core", "hf"],
        help="Model loading backend. 'auto' prefers native OLMo-core checkpoints.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Layer Usefulness via Linear Approximation")
    print("=" * 60)

    model, tokenizer, device, model_dtype, resolved_model_path, was_converted = load_model_and_tokenizer(
        model_path=args.model_path,
        output_dir=output_dir,
        device=args.device,
        dtype=args.dtype,
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
        skip_conversion_validation=args.skip_conversion_validation,
        model_backend=args.model_backend,
    )
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    print(f"Loaded model from: {resolved_model_path}")
    print(f"Number of layers: {num_layers}")
    if was_converted:
        print("Input checkpoint was auto-converted to Hugging Face format for analysis.")
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

    baseline_loss = compute_baseline_loss(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
        model_dtype=model_dtype,
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
        "was_converted": was_converted,
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
