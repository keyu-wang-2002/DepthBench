import argparse
import json
import math
from pathlib import Path
from typing import Any, List

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


def _detach_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_nested(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_nested(item) for key, item in value.items()}
    return value


def estimate_backward_passes(
    num_layers: int,
    hidden_size: int,
    num_micro_batches: int,
    vjp_chunk_size: int,
) -> int:
    chunk_size = hidden_size if vjp_chunk_size <= 0 else vjp_chunk_size
    return num_layers * num_micro_batches * math.ceil(hidden_size / chunk_size)


def _capture_all_layer_calls(
    model,
    layers,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    captured = [{"input": None, "kwargs": None} for _ in range(len(layers))]
    hook_handles = []

    def make_forward_hook(layer_idx: int):
        def forward_hook(module, module_input, module_kwargs, module_output):
            del module, module_output
            layer_input = module_input[0] if isinstance(module_input, tuple) else module_input
            captured[layer_idx]["input"] = layer_input.detach()
            captured[layer_idx]["kwargs"] = _detach_nested(module_kwargs)

        return forward_hook

    for layer_idx, layer in enumerate(layers):
        hook_handles.append(layer.register_forward_hook(make_forward_hook(layer_idx), with_kwargs=True))

    try:
        with torch.no_grad():
            outputs = model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        del outputs
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()

    missing_layers = [idx for idx, item in enumerate(captured) if item["input"] is None]
    if missing_layers:
        raise RuntimeError(f"Failed to capture layer inputs for layers: {missing_layers}")

    return [(item["input"], item["kwargs"] or {}) for item in captured]


def _compute_captured_layer_jacobian(
    target_layer,
    layer_idx: int,
    layer_input: torch.Tensor,
    layer_kwargs: dict,
    hidden_size: int,
    vjp_chunk_size: int,
) -> torch.Tensor:
    def layer_fn(layer_input_arg: torch.Tensor) -> torch.Tensor:
        layer_output = normalize_layer_output(target_layer(layer_input_arg, **layer_kwargs))
        if layer_output is None:
            raise RuntimeError(f"Layer {layer_idx} returned no tensor output.")
        return layer_output

    _, vjp_fn = torch.func.vjp(layer_fn, layer_input)
    eye = torch.eye(hidden_size, dtype=layer_input.dtype, device=layer_input.device)
    rows = []

    chunk_size = hidden_size if vjp_chunk_size <= 0 else min(vjp_chunk_size, hidden_size)
    for start in range(0, hidden_size, chunk_size):
        basis = eye[start : start + chunk_size]
        cotangent_shape = (basis.shape[0],) + (1,) * (layer_input.ndim - 1) + (hidden_size,)
        cotangents = basis.view(cotangent_shape).expand(-1, *layer_input.shape)

        if basis.shape[0] == 1:
            chunk_grads = vjp_fn(cotangents[0])[0].unsqueeze(0)
        else:
            chunk_grads = torch.vmap(lambda cotangent: vjp_fn(cotangent)[0])(cotangents)

        position_dims = tuple(range(1, chunk_grads.ndim - 1))
        rows.append(chunk_grads.mean(dim=position_dims).detach().float().cpu())
        del cotangents, chunk_grads

    jacobian = torch.cat(rows, dim=0)
    layer_jacobian = jacobian.detach().float().cpu().T

    del eye, rows, jacobian
    return layer_jacobian


def compute_jacobian_deviation_norms(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    micro_batch_size: int,
    vjp_chunk_size: int,
) -> List[float]:
    layers = get_decoder_layers(model)
    num_layers = len(layers)
    hidden_size = get_hidden_size(model)
    batch_size = input_ids.shape[0]
    micro_batch_size = min(max(1, micro_batch_size), batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    jacobian_sums = [torch.zeros(hidden_size, hidden_size) for _ in range(num_layers)]
    total_positions = [0 for _ in range(num_layers)]

    for mb_idx in tqdm(range(num_micro_batches), desc="Computing exact fast Jacobians"):
        start_idx = mb_idx * micro_batch_size
        end_idx = min(start_idx + micro_batch_size, batch_size)

        micro_input_ids = input_ids[start_idx:end_idx].to(device)
        micro_attention_mask = attention_mask[start_idx:end_idx].to(device)

        captured_layer_calls = _capture_all_layer_calls(
            model=model,
            layers=layers,
            input_ids=micro_input_ids,
            attention_mask=micro_attention_mask,
        )

        for layer_idx, (layer_input, layer_kwargs) in enumerate(captured_layer_calls):
            position_count = layer_input.shape[0] * layer_input.shape[1]
            total_positions[layer_idx] += position_count
            layer_jacobian = _compute_captured_layer_jacobian(
                target_layer=layers[layer_idx],
                layer_idx=layer_idx,
                layer_input=layer_input,
                layer_kwargs=layer_kwargs,
                hidden_size=hidden_size,
                vjp_chunk_size=vjp_chunk_size,
            )
            jacobian_sums[layer_idx] += layer_jacobian * position_count
            del layer_input, layer_kwargs, layer_jacobian

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        del captured_layer_calls

    norms = []
    for layer_idx, jacobian_sum in enumerate(jacobian_sums):
        if total_positions[layer_idx] == 0:
            jacobian = torch.zeros(hidden_size, hidden_size)
        else:
            jacobian = jacobian_sum / total_positions[layer_idx]
        hidden_size = jacobian.shape[0]
        identity = torch.eye(hidden_size, dtype=jacobian.dtype)
        norm = torch.norm(jacobian - identity, p="fro").item() / math.sqrt(hidden_size)
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
    plt.ylabel(r"$\|J - I\|_F / \sqrt{d}$", fontsize=13)
    plt.title("Layer-wise Normalized Jacobian Deviation from Identity", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute exact layer-wise ||J-I||_F faster with torch.func VJPs batched by vmap"
    )
    parser.add_argument("--model_path", type=str, required=True, help="OLMo-core checkpoint dir")
    parser.add_argument("--output_dir", type=str, default="./jacobian_results_fast", help="Directory to save outputs")
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
    parser.add_argument("--tokenizer-id", type=str, default=None, help="Optional tokenizer override")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Optional tokenizer model_max_length override",
    )
    parser.add_argument(
        "--max-hidden-size-without-force",
        type=int,
        default=1024,
        help="Refuse to run the full Jacobian computation above this hidden size unless --force-large-run is set.",
    )
    parser.add_argument(
        "--force-large-run",
        action="store_true",
        help="Allow the expensive exact Jacobian computation for large hidden sizes.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Microbatch size for layer-input capture and exact VJP computation.",
    )
    parser.add_argument(
        "--vjp-chunk-size",
        type=int,
        default=1024,
        help="Number of output hidden dimensions to differentiate in one batched VJP. Use <=0 for all.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, _, resolved_model_path = load_model_and_tokenizer(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        tokenizer_id=args.tokenizer_id,
        max_sequence_length=args.max_sequence_length,
    )
    num_layers = len(get_decoder_layers(model))
    hidden_size = get_hidden_size(model)
    micro_batch_size = min(max(1, args.micro_batch_size), args.num_samples)
    num_micro_batches = (args.num_samples + micro_batch_size - 1) // micro_batch_size
    estimated_batched_vjp_calls = estimate_backward_passes(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_micro_batches=num_micro_batches,
        vjp_chunk_size=args.vjp_chunk_size,
    )

    if hidden_size > args.max_hidden_size_without_force and not args.force_large_run:
        raise ValueError(
            "Jacobian analysis is intentionally blocked for large models by default. "
            f"hidden_size={hidden_size} exceeds --max-hidden-size-without-force={args.max_hidden_size_without_force}. "
            "Re-run with --force-large-run if you really want the full exact computation."
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
        micro_batch_size=args.micro_batch_size,
        vjp_chunk_size=args.vjp_chunk_size,
    )

    plot_jacobian_norms(jacobian_norms, output_dir / "jacobian_norms.png")

    results = {
        "model_path": args.model_path,
        "resolved_model_path": resolved_model_path,
        "num_layers": len(jacobian_norms),
        "num_samples": args.num_samples,
        "seq_length": args.seq_length,
        "metric": "||J - I||_F / sqrt(d)",
        "exact": True,
        "method": "torch.func.vjp batched with torch.vmap over output hidden dimensions",
        "micro_batch_size": args.micro_batch_size,
        "vjp_chunk_size": args.vjp_chunk_size,
        "plot_excludes_layer_0": len(jacobian_norms) > 1,
        "jacobian_norms": jacobian_norms,
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Loaded model from: {resolved_model_path}")
    print(
        "Exact fast Jacobian run summary: "
        f"layers={num_layers}, hidden_size={hidden_size}, "
        f"micro_batches={num_micro_batches}, vjp_chunk_size={args.vjp_chunk_size}, "
        f"estimated_batched_vjp_calls={estimated_batched_vjp_calls}"
    )
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
