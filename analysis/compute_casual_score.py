import argparse
import json
from pathlib import Path

import numpy as np

from analysis_utils import build_sample_batch, get_decoder_layers, load_model_and_tokenizer
from layer_score_utils import compute_all_causal_effects, plot_causal_effects


def compute_global_causal_score(causal_effect_matrix: np.ndarray) -> tuple[float, list[float]]:
    num_layers = causal_effect_matrix.shape[0]
    if num_layers == 0:
        return 0.0, []

    layer_scores = []
    for skipped_layer in range(num_layers - 1):
        future_scores = causal_effect_matrix[skipped_layer, skipped_layer + 1 :]
        if future_scores.size == 0:
            layer_scores.append(0.0)
        else:
            layer_scores.append(float(np.mean(future_scores)))

    global_score = (1.0 / np.sqrt(num_layers)) * (1.0 / num_layers) * float(np.sum(layer_scores))
    return global_score, layer_scores


def main():
    parser = argparse.ArgumentParser(description="Compute causal layer scores")
    parser.add_argument("--model_path", type=str, required=True, help="HF model dir or OLMo checkpoint dir")
    parser.add_argument("--output_dir", type=str, default="./causal_score_results", help="Directory to save results")
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
    num_layers = len(get_decoder_layers(model))

    print(f"Loaded model from: {resolved_model_path}")
    print(f"Model has {num_layers} layers")
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

    causal_effect_matrix = compute_all_causal_effects(
        model=model,
        input_ids=sample_data["input_ids"],
        attention_mask=sample_data["attention_mask"],
        device=device,
        model_dtype=model_dtype,
        num_layers=num_layers,
    )

    np.save(output_dir / "causal_effect_matrix.npy", causal_effect_matrix)
    plot_causal_effects(causal_effect_matrix, output_dir / "causal_effects_heatmap.png")

    global_causal_score, truncated_layer_scores = compute_global_causal_score(causal_effect_matrix)
    avg_effect_per_layer = truncated_layer_scores + [0.0] if num_layers > 0 else []

    print("\nAverage causal effect per skipped layer:")
    for layer_idx, avg_effect in enumerate(avg_effect_per_layer):
        print(f"  Layer {layer_idx}: {avg_effect:.4f}")

    print(f"\nGlobal causal score: {global_causal_score:.6f}")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "was_converted": was_converted,
        "num_layers": num_layers,
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "global_causal_score": global_causal_score,
        "avg_effect_per_layer": avg_effect_per_layer,
        "causal_effect_matrix": causal_effect_matrix.tolist(),
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
