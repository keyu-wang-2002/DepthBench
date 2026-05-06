"""
Pairwise layer weight-swap score: keep the architecture and layer positions fixed,
swap only the learned weights of two layers, and measure the relative change in next-token loss. 
Higher scores indicate less interchangeable, more position-specific layer weights.
"""

import argparse
import json
from pathlib import Path

from analysis_utils import build_sample_batch, get_decoder_layers, load_model_and_tokenizer
from layer_score_utils import (
    compute_all_permutation_scores,
    compute_global_permutation_score,
    plot_permutation_scores,
)


def build_layer_pairs(args, num_layers: int):
    layer_pairs = []

    if not (args.skip_layer_1_with_next or args.skip_layer_2_with_rest or args.skip_layer_3_with_rest):
        for layer1_idx in range(num_layers):
            for layer2_idx in range(layer1_idx + 1, num_layers):
                layer_pairs.append((layer1_idx, layer2_idx))
        return layer_pairs

    if args.skip_layer_1_with_next and 1 < num_layers - 1:
        layer_pairs.append((1, 2))

    if args.skip_layer_2_with_rest:
        for layer2_idx in range(2, num_layers):
            if layer2_idx != 2:
                layer_pairs.append((2, layer2_idx))

    if args.skip_layer_3_with_rest:
        for layer2_idx in range(3, num_layers):
            if layer2_idx != 3:
                layer_pairs.append((3, layer2_idx))

    return layer_pairs


def main():
    parser = argparse.ArgumentParser(description="Compute permutation layer scores")
    parser.add_argument("--model_path", type=str, required=True, help="OLMo-core checkpoint dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./permutation_score_results",
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
    parser.add_argument("--skip_layer_1_with_next", action="store_true", help="Only test layer 1 with layer 2")
    parser.add_argument("--skip_layer_2_with_rest", action="store_true", help="Only test layer 2 with layers 2-n")
    parser.add_argument("--skip_layer_3_with_rest", action="store_true", help="Only test layer 3 with layers 3-n")
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

    model, tokenizer, device, model_dtype, resolved_model_path = load_model_and_tokenizer(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
    )
    num_layers = len(get_decoder_layers(model))

    print(f"Loaded model from: {resolved_model_path}")
    print(f"Model has {num_layers} layers")

    sample_data = build_sample_batch(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        text_file=args.text_file,
        prompts=args.prompt,
    )

    layer_pairs = build_layer_pairs(args, num_layers)
    print(f"Testing {len(layer_pairs)} layer pairs...")

    baseline_loss, permutation_scores = compute_all_permutation_scores(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
        model_dtype=model_dtype,
        num_layers=num_layers,
        layer_pairs=layer_pairs,
    )
    global_scores = compute_global_permutation_score(permutation_scores, num_layers)

    plot_permutation_scores(permutation_scores, num_layers, output_dir / "permutation_scores_heatmap.png")

    print("\nPermutation scores:")
    for (layer1_idx, layer2_idx), score in sorted(permutation_scores.items()):
        print(f"  Layers ({layer1_idx}, {layer2_idx}): {score:.4f}")

    print("\nGlobal permutation scores:")
    print(f"  Mean Score: {global_scores['global_score_mean']:.4f}")
    print(f"  Normalized Score: {global_scores['global_score_normalized']:.4f}")
    print(f"  Weighted Score: {global_scores['global_score_weighted']:.4f}")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "baseline_loss": float(baseline_loss),
        "num_layers": num_layers,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "tested_layer_pairs": [[layer1_idx, layer2_idx] for layer1_idx, layer2_idx in layer_pairs],
        "permutation_scores": {f"{layer1_idx}_{layer2_idx}": score for (layer1_idx, layer2_idx), score in permutation_scores.items()},
        "global_permutation_scores": global_scores,
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
