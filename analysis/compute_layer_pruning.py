"""Single-layer pruning: remove one block at a time and measure the change in performance.

Two evaluation signals are supported:

* LM loss (default): mean next-token loss on the sampled token windows;
  delta = L_pruned - L_baseline (positive = pruning hurts).
* A zero-shot lm-eval task (`--task arc_easy`, ...): delta = score_pruned - score_baseline
  (negative = pruning hurts).

The per-architecture skip rule is `depth_probes.skip_block`, shared with the causal score.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from analysis_utils import (
    PROJECT_ROOT,
    add_model_args,
    get_decoder_layers,
    install_runtime_fallbacks,
    json_ready,
    load_eval_input_ids,
    load_experiment_config,
    load_model_and_tokenizer,
    resolve_device,
    write_json,
)
from depth_probes import describe_architecture, mean_lm_loss, skip_block


def _task_metric(results: dict[str, Any], task: str, metric: str | None) -> tuple[str, float]:
    task_results = results["results"][task]
    if metric is not None:
        for key, value in task_results.items():
            if key == metric or key.split(",", 1)[0] == metric:
                return key, float(value)
        raise KeyError(f"Metric '{metric}' not found for task '{task}'. Available: {sorted(task_results)}")
    for preferred in ("acc_norm,none", "acc,none", "exact_match,none"):
        if preferred in task_results:
            return preferred, float(task_results[preferred])
    raise KeyError(f"No default metric found for task '{task}'. Available: {sorted(task_results)}")


class TaskEvaluator:
    """Zero-shot lm-eval evaluation of an OLMo-core checkpoint through `eval/olmo_lm.py`."""

    def __init__(self, args: argparse.Namespace):
        sys.path.insert(0, str(PROJECT_ROOT / "eval"))
        import lm_eval
        from olmo_core.nn.transformer.config import TransformerConfig
        from olmo_lm import OLMoNativeLM

        device = resolve_device(args.device)
        install_runtime_fallbacks(device)
        checkpoint_dir, config = load_experiment_config(args.model_path)
        dtype = args.dtype if args.dtype != "auto" else ("bfloat16" if device.startswith("cuda") else "float32")
        self.lm = OLMoNativeLM.build(
            checkpoint_dir,
            tokenizer_name_or_path=args.tokenizer_id or config["dataset"]["tokenizer"]["identifier"],
            device=device,
            batch_size=args.eval_batch_size,
            dtype=dtype,
            attention_backend="torch",
            transformer_config=TransformerConfig.from_dict(config["model"]),
        )
        self.model = self.lm.generation_module.model
        self.checkpoint_dir = str(checkpoint_dir)
        self.lm_eval = lm_eval
        self.args = args

    def __call__(self) -> tuple[str, float]:
        cache_hook = getattr(self.lm, "cache_hook", None)
        for attr in ("cache_dict", "cache", "_cache"):
            cache = getattr(cache_hook, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()
        results = self.lm_eval.simple_evaluate(
            model=self.lm,
            tasks=[self.args.task],
            num_fewshot=0,
            batch_size=self.args.eval_batch_size,
            limit=self.args.limit,
            log_samples=False,
        )
        return _task_metric(results, self.args.task, self.args.metric)


class LossEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.model, tokenizer, self.device, self.dtype, self.checkpoint_dir = load_model_and_tokenizer(
            args.model_path, args.device, args.dtype, args.tokenizer_id, args.max_sequence_length
        )
        self.input_ids, self.sample_info = load_eval_input_ids(args, tokenizer)
        self.micro_batch_size = args.micro_batch_size

    def __call__(self) -> tuple[str, float]:
        loss = mean_lm_loss(
            self.model, self.input_ids, micro_batch_size=self.micro_batch_size, device=self.device, dtype=self.dtype
        )
        return "lm_loss", loss


def plot_pruning(results: dict, output_path: Path) -> None:
    layers = sorted(results["layer_metrics"], key=int)
    deltas = [results["layer_metrics"][layer]["delta"] for layer in layers]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.32 * len(layers) + 2.0), 3.6))
    ax.bar([int(layer) for layer in layers], deltas, width=0.78, color="#5b8cc0", edgecolor="#2d5a88", linewidth=0.6)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Pruned layer $\ell$")
    ax.set_ylabel(f"Δ {results['metric']} (pruned − baseline)")
    ax.set_title(f"Single-layer pruning ({results['architecture']['family']})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune one layer at a time and measure the performance change")
    add_model_args(parser, num_samples=16, seq_length=256)
    parser.add_argument("--task", type=str, default=None, help="lm-eval task (e.g. arc_easy); default: LM loss")
    parser.add_argument("--metric", type=str, default=None, help="lm-eval metric; defaults to acc_norm/acc")
    parser.add_argument("--limit", type=float, default=None, help="lm-eval example limit")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="lm-eval batch size")
    parser.add_argument("--layer-index", type=int, action="append", default=None, help="Layer to prune; repeatable")
    parser.add_argument("--resume", action="store_true", help="Reuse results already in the output file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "layer_pruning.json"

    evaluate = TaskEvaluator(args) if args.task else LossEvaluator(args)
    model = evaluate.model
    num_layers = len(get_decoder_layers(model))
    layer_indices = sorted(set(args.layer_index)) if args.layer_index else list(range(num_layers))
    if any(idx < 0 or idx >= num_layers for idx in layer_indices):
        raise ValueError(f"Layer indices must lie in [0, {num_layers}); got {layer_indices}")

    results = json.loads(result_path.read_text(encoding="utf-8")) if args.resume and result_path.exists() else {}
    results.update(
        score_type="layer_pruning",
        signal=args.task or "lm_loss",
        model_path=args.model_path,
        resolved_model_path=evaluate.checkpoint_dir,
        architecture=describe_architecture(model),
        num_layers=num_layers,
        **getattr(evaluate, "sample_info", {"limit": args.limit}),
    )
    results.setdefault("layer_metrics", {})
    if "baseline" not in results:
        metric, score = evaluate()
        results.update(metric=metric, baseline=score)
        write_json(result_path, results)
    print(f"Baseline {results['metric']} = {results['baseline']:.4f}")

    for layer_idx in tqdm(layer_indices, desc="Layer pruning"):
        if str(layer_idx) in results["layer_metrics"]:
            continue
        with skip_block(model, layer_idx):
            _, score = evaluate()
        results["layer_metrics"][str(layer_idx)] = {"score": score, "delta": score - results["baseline"]}
        write_json(result_path, results)

    with (output_dir / "layer_pruning.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "metric", "baseline", "pruned", "delta"])
        for layer in sorted(results["layer_metrics"], key=int):
            row = results["layer_metrics"][layer]
            writer.writerow([layer, results["metric"], results["baseline"], row["score"], row["delta"]])
    plot_pruning(json_ready(results), output_dir / "layer_pruning.png")
    deltas = np.asarray([row["delta"] for row in results["layer_metrics"].values()])
    print(f"Mean delta {deltas.mean():.4f}; results saved to {output_dir}")


if __name__ == "__main__":
    main()
