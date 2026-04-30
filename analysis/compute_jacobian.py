import argparse
import json
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from analysis_utils import (
    build_sample_batch,
    get_decoder_layers,
    get_hidden_size,
    load_model_and_tokenizer,
    model_forward,
    normalize_layer_output,
)


def compute_mean_layer_jacobian(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    layer_idx: int,
) -> torch.Tensor:
    model.eval()
    layers = get_decoder_layers(model)
    hidden_size = get_hidden_size(model)

    captured_input = {"tensor": None}

    def forward_hook(module, module_input, module_output):
        del module_output
        layer_input = module_input[0] if isinstance(module_input, tuple) else module_input
        captured_input["tensor"] = layer_input

    hook_handle = layers[layer_idx].register_forward_hook(forward_hook)

    batch_size = input_ids.shape[0]
    micro_batch_size = min(2, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    jacobian_sum = torch.zeros(hidden_size, hidden_size)
    total_positions = 0

    with torch.enable_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            captured_input["tensor"] = None

            outputs = model_forward(
                model,
                input_ids=micro_input_ids,
                attention_mask=micro_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            layer_input = captured_input["tensor"]
            layer_output = normalize_layer_output(outputs.hidden_states[layer_idx + 1])

            if layer_input is None or layer_output is None:
                continue

            position_count = layer_output.shape[0] * layer_output.shape[1]
            total_positions += position_count

            rows = []
            for hidden_idx in range(hidden_size):
                grad_output = torch.zeros_like(layer_output)
                grad_output[..., hidden_idx] = 1.0
                grad = torch.autograd.grad(
                    layer_output,
                    layer_input,
                    grad_outputs=grad_output,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                rows.append(grad.mean(dim=(0, 1)).detach().cpu())

            jacobian_sum += torch.stack(rows, dim=1) * position_count

            del outputs
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    hook_handle.remove()

    if total_positions == 0:
        return torch.zeros(hidden_size, hidden_size)

    return jacobian_sum / total_positions


def compute_jacobian_deviation_norms(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> List[float]:
    num_layers = len(get_decoder_layers(model))
    norms = []

    for layer_idx in tqdm(range(num_layers), desc="Computing ||J-I||_F"):
        jacobian = compute_mean_layer_jacobian(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            layer_idx=layer_idx,
        )
        hidden_size = jacobian.shape[0]
        identity = torch.eye(hidden_size, dtype=jacobian.dtype)
        norm = torch.norm(jacobian - identity, p="fro").item() / math.sqrt(hidden_size * hidden_size)
        norms.append(float(norm))

    return norms


def plot_jacobian_norms(norms: List[float], output_path: Path):
    if len(norms) <= 1:
        layers = list(range(len(norms)))
        plotted_norms = norms
    else:
        layers = list(range(1, len(norms)))
        plotted_norms = norms[1:]

    plt.figure(figsize=(9, 5.5))
    plt.plot(layers, plotted_norms, marker="o", linewidth=2, markersize=5)
    plt.xlabel("Layer Index", fontsize=13)
    plt.ylabel(r"$\|J - I\|_F / d$", fontsize=13)
    plt.title("Layer-wise Normalized Jacobian Deviation from Identity", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute layer-wise ||J-I||_F")
    parser.add_argument("--model_path", type=str, required=True, help="HF model dir or OLMo checkpoint dir")
    parser.add_argument("--output_dir", type=str, default="./jacobian_results", help="Directory to save outputs")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of token chunks to evaluate")
    parser.add_argument("--seq_length", type=int, default=512, help="Token chunk length")
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

    model, tokenizer, device, _, resolved_model_path, was_converted = load_model_and_tokenizer(
        model_path=args.model_path,
        output_dir=output_dir,
        device=args.device,
        dtype=args.dtype,
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
        skip_conversion_validation=args.skip_conversion_validation,
        model_backend=args.model_backend,
    )

    sample_data = build_sample_batch(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        text_file=args.text_file,
        prompts=args.prompt,
    )

    jacobian_norms = compute_jacobian_deviation_norms(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
    )

    plot_jacobian_norms(jacobian_norms, output_dir / "jacobian_norms.png")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "was_converted": was_converted,
        "num_layers": len(jacobian_norms),
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "metric": "||J - I||_F / d",
        "plot_excludes_layer_0": len(jacobian_norms) > 1,
        "jacobian_norms": jacobian_norms,
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Loaded model from: {resolved_model_path}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
