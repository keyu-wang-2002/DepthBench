import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from analysis_utils import (
    autocast_context,
    build_sample_batch,
    get_decoder_layers,
    load_model_and_tokenizer,
    model_forward,
)


def extract_hidden_tensor(output):
    if isinstance(output, tuple):
        return output[0] if output else None
    return output


def collect_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
) -> List[torch.Tensor]:
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
    hidden_states_accum: list[torch.Tensor | None] = []

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

            if mb_idx == 0:
                for hidden_state in outputs.hidden_states:
                    value = extract_hidden_tensor(hidden_state)
                    hidden_states_accum.append(value.detach().cpu() if value is not None else None)
            else:
                for idx, hidden_state in enumerate(outputs.hidden_states):
                    value = extract_hidden_tensor(hidden_state)
                    if value is None:
                        continue
                    value = value.detach().cpu()
                    if hidden_states_accum[idx] is None:
                        hidden_states_accum[idx] = value
                    else:
                        hidden_states_accum[idx] = torch.cat([hidden_states_accum[idx], value], dim=0)

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    first_valid = next(hidden for hidden in hidden_states_accum if hidden is not None)
    return [hidden if hidden is not None else torch.zeros_like(first_valid) for hidden in hidden_states_accum]


def compute_angular_distance_matrix(
    layer_hidden_states: List[torch.Tensor],
    attention_mask: torch.Tensor,
) -> np.ndarray:
    num_layers = len(layer_hidden_states)
    angular_distance = np.full((num_layers, num_layers), np.nan, dtype=np.float32)
    valid_positions = attention_mask.reshape(-1).bool().cpu()
    flattened_layers = [hidden.reshape(-1, hidden.shape[-1]).float()[valid_positions] for hidden in layer_hidden_states]

    for layer_idx in tqdm(range(num_layers), desc="Computing angular distance"):
        hidden_i = flattened_layers[layer_idx]
        for subsequent_idx in range(layer_idx + 1, num_layers):
            hidden_j = flattened_layers[subsequent_idx]
            cosine = F.cosine_similarity(hidden_i, hidden_j, dim=-1).clamp(-1.0, 1.0)
            mean_angle = torch.acos(cosine).mean().item() / np.pi
            angular_distance[layer_idx, subsequent_idx] = mean_angle

    return angular_distance


def convert_to_subsequent_layer_matrix(matrix: np.ndarray) -> np.ndarray:
    num_layers = matrix.shape[0]
    subsequent_matrix = np.full((num_layers - 1, num_layers - 1), np.nan, dtype=np.float32)

    for layer_idx in range(num_layers):
        for subsequent_idx in range(layer_idx + 1, num_layers):
            subsequent_n = subsequent_idx - layer_idx
            subsequent_matrix[subsequent_n - 1, layer_idx] = matrix[layer_idx, subsequent_idx]

    return subsequent_matrix


def plot_angular_distance_heatmap(matrix: np.ndarray, output_path: Path, title: str = "Angular Distance"):
    display_matrix = convert_to_subsequent_layer_matrix(matrix)[::-1]
    mask = np.isnan(display_matrix)
    finite_values = display_matrix[~mask]
    if finite_values.size == 0:
        raise ValueError("Angular distance matrix does not contain any finite values.")

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    num_rows, num_cols = display_matrix.shape
    fig_width = max(7.5, num_cols * 0.48 + 1.5)
    fig_height = max(5.0, num_rows * 0.48 + 1.0)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        display_matrix,
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

    x_tick_step = max(1, num_cols // 4)
    x_tick_positions = list(range(0, num_cols, x_tick_step))
    plt.xticks(
        [pos + 0.5 for pos in x_tick_positions],
        [str(pos) for pos in x_tick_positions],
        rotation=0,
        fontsize=12,
    )

    y_tick_step = max(1, num_rows // 4)
    y_tick_positions = list(range(0, num_rows, y_tick_step))
    plt.yticks(
        [pos + 0.5 for pos in y_tick_positions],
        [str(num_rows - pos) for pos in y_tick_positions],
        rotation=0,
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")

    return {"colorbar_min": vmin, "colorbar_max": vmax}


def summarize_pairs(matrix: np.ndarray, smallest: bool, top_k: int):
    values = []
    num_layers = matrix.shape[0]

    for layer_idx in range(num_layers):
        for subsequent_idx in range(layer_idx + 1, num_layers):
            value = matrix[layer_idx, subsequent_idx]
            if np.isnan(value):
                continue
            values.append((layer_idx, subsequent_idx, float(value)))

    values.sort(key=lambda item: item[2], reverse=not smallest)
    selected = values[:top_k]
    return [
        {"layer_i": layer_idx, "layer_j": subsequent_idx, "value": value}
        for layer_idx, subsequent_idx, value in selected
    ]


def main():
    parser = argparse.ArgumentParser(description="Compute and visualize layer-wise angular distance")
    parser.add_argument("--model_path", type=str, required=True, help="HF model dir or OLMo checkpoint dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./angular_distance_results",
        help="Directory to save outputs",
    )
    parser.add_argument("--num_samples", type=int, default=1024, help="Number of token chunks to evaluate")
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
    parser.add_argument(
        "--exclude-embeddings",
        action="store_true",
        help="Exclude embedding hidden state. By default x^0 (embedding output) is included.",
    )
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

    decoder_layers = get_decoder_layers(model)
    print(f"Loaded model from: {resolved_model_path}")
    print(f"Model has {len(decoder_layers)} decoder layers")
    if was_converted:
        print("Input checkpoint was auto-converted to Hugging Face format for analysis.")

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

    if args.exclude_embeddings:
        analyzed_hidden_states = hidden_states[1:]
        hidden_state_start_index = 1
    else:
        analyzed_hidden_states = hidden_states
        hidden_state_start_index = 0

    angular_distance_matrix = compute_angular_distance_matrix(
        analyzed_hidden_states,
        sample_data["attention_mask"],
    )
    subsequent_matrix = convert_to_subsequent_layer_matrix(angular_distance_matrix)

    np.save(output_dir / "angular_distance_matrix.npy", angular_distance_matrix)
    np.save(output_dir / "angular_distance_subsequent_matrix.npy", subsequent_matrix)

    colorbar_range = plot_angular_distance_heatmap(
        angular_distance_matrix,
        output_dir / "angular_distance_heatmap.png",
    )

    smallest_distances = summarize_pairs(angular_distance_matrix, smallest=True, top_k=10)
    largest_distances = summarize_pairs(angular_distance_matrix, smallest=False, top_k=10)

    print("\nTop 10 smallest angular distances:")
    for item in smallest_distances:
        print(f"  Layers ({item['layer_i']}, {item['layer_j']}): {item['value']:.4f}")

    print("\nTop 10 largest angular distances:")
    for item in largest_distances:
        print(f"  Layers ({item['layer_i']}, {item['layer_j']}): {item['value']:.4f}")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "was_converted": was_converted,
        "num_decoder_layers": len(decoder_layers),
        "num_analyzed_layers": len(analyzed_hidden_states),
        "exclude_embeddings": args.exclude_embeddings,
        "hidden_state_start_index": hidden_state_start_index,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "uses_attention_mask_filtering": True,
        "colorbar_min": colorbar_range["colorbar_min"],
        "colorbar_max": colorbar_range["colorbar_max"],
        "smallest_angular_distances": smallest_distances,
        "largest_angular_distances": largest_distances,
        "angular_distance_matrix": np.nan_to_num(angular_distance_matrix, nan=-1.0).tolist(),
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
