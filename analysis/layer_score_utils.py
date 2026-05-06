from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from tqdm import tqdm

from analysis_utils import (
    autocast_context,
    get_decoder_layers,
    model_forward,
    normalize_layer_output,
    replace_decoder_layer,
    restore_state_dict,
    state_dict_clone,
)


class IdentityDecoderLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return normalize_layer_output(hidden_states)


def compute_mean_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    *,
    micro_batch_size_limit: int = 4,
) -> float:
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(micro_batch_size_limit, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
    total_loss = 0.0

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            micro_labels = F.pad(micro_input_ids[:, 1:], (0, 1), value=-100)
            
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


def collect_layer_input_output_pairs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    layer_idx: int,
    *,
    micro_batch_size_limit: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(micro_batch_size_limit, batch_size)
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
    return inputs, outputs


def compute_baseline_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
) -> Tuple[float, List[torch.Tensor], float]:
    model.eval()

    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    hidden_states_accum = []
    loss_accum = 0.0

    with torch.no_grad():
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            micro_input_ids = input_ids[start_idx:end_idx].to(device)
            micro_attention_mask = attention_mask[start_idx:end_idx].to(device)
            micro_labels = F.pad(micro_input_ids[:, 1:], (0, 1), value=-100)

            with autocast_context(device, model_dtype):
                outputs = model_forward(
                    model,
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                    labels=micro_labels,
                    output_hidden_states=True,
                    use_cache=False,
                )

            loss_accum += outputs.loss.item() * (end_idx - start_idx)

            if mb_idx == 0:
                for hidden_state in outputs.hidden_states:
                    value = normalize_layer_output(hidden_state)
                    hidden_states_accum.append(value.detach().cpu() if value is not None else None)
            else:
                for idx, hidden_state in enumerate(outputs.hidden_states):
                    value = normalize_layer_output(hidden_state)
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
    all_hidden_states = [hidden if hidden is not None else torch.zeros_like(first_valid) for hidden in hidden_states_accum]
    loss = loss_accum / batch_size
    return loss, all_hidden_states, math.exp(loss)


def compute_causal_effect_on_future(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    baseline_hidden_states: List[torch.Tensor],
    layer_to_skip: int,
) -> Dict[int, float]:
    model.eval()
    modified_hidden_states_accum = []
    batch_size = input_ids.shape[0]
    micro_batch_size = min(4, batch_size)
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    with replace_decoder_layer(
        model,
        layer_to_skip,
        IdentityDecoderLayer(),
    ):
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
                        value = normalize_layer_output(hidden_state)
                        modified_hidden_states_accum.append(value.detach().cpu() if value is not None else None)
                else:
                    for idx, hidden_state in enumerate(outputs.hidden_states):
                        value = normalize_layer_output(hidden_state)
                        if value is None:
                            continue
                        value = value.detach().cpu()
                        if modified_hidden_states_accum[idx] is None:
                            modified_hidden_states_accum[idx] = value
                        else:
                            modified_hidden_states_accum[idx] = torch.cat(
                                [modified_hidden_states_accum[idx], value],
                                dim=0,
                            )

                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    modified_hidden_states = []
    for idx, hidden_state in enumerate(modified_hidden_states_accum):
        if hidden_state is None:
            modified_hidden_states.append(torch.zeros_like(baseline_hidden_states[idx]))
        else:
            modified_hidden_states.append(hidden_state)

    causal_effects = {}
    max_layer = min(len(baseline_hidden_states), len(modified_hidden_states)) - 1
    for future_layer in range(layer_to_skip + 1, max_layer):
        baseline_diff = baseline_hidden_states[future_layer + 1] - baseline_hidden_states[future_layer]
        modified_diff = modified_hidden_states[future_layer + 1] - modified_hidden_states[future_layer]

        baseline_norm = torch.norm(baseline_diff, p=2, dim=-1).mean().item()
        relative_change = torch.norm(modified_diff - baseline_diff, p=2, dim=-1).mean().item()
        causal_effects[future_layer] = relative_change / baseline_norm if baseline_norm > 1e-8 else 0.0

    return causal_effects


def compute_all_causal_effects(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    num_layers: int,
) -> np.ndarray:
    print("Computing causal effects...")
    _, baseline_hidden_states, _ = compute_baseline_forward(model, input_ids, attention_mask, device, model_dtype)
    causal_effect_matrix = np.zeros((num_layers, num_layers))

    for source_layer in tqdm(range(num_layers), desc="Computing causal effects"):
        causal_effects = compute_causal_effect_on_future(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
            baseline_hidden_states=baseline_hidden_states,
            layer_to_skip=source_layer,
        )
        for future_layer, effect in causal_effects.items():
            causal_effect_matrix[source_layer, future_layer] = effect

    return causal_effect_matrix


def plot_causal_effects(causal_effect_matrix: np.ndarray, output_path: Path):
    mask = np.tril(np.ones_like(causal_effect_matrix, dtype=bool))
    masked_matrix = np.ma.array(causal_effect_matrix, mask=mask)

    cmap = sns.light_palette("#1186cf", as_cmap=True)
    cmap.set_bad(color="white")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        masked_matrix,
        cmap=cmap,
        mask=mask,
        cbar_kws={"label": "Causal Effect"},
        xticklabels=range(causal_effect_matrix.shape[1]),
        yticklabels=range(causal_effect_matrix.shape[0]),
        linewidths=0.3,
        linecolor=(1.0, 1.0, 1.0, 0.45),
        square=True,
    )
    plt.xlabel("Affected Layer")
    plt.ylabel("Removed Layer")
    plt.title("Causal Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved causal effects plot to {output_path}")


def swap_layer_weights(model, layer1_idx: int, layer2_idx: int):
    layers = get_decoder_layers(model)
    layer1 = layers[layer1_idx]
    layer2 = layers[layer2_idx]
    state1 = state_dict_clone(layer1)
    state2 = state_dict_clone(layer2)
    restore_state_dict(layer1, state2)
    restore_state_dict(layer2, state1)


def compute_permutation_score_for_pair(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    baseline_loss: float,
    layer1_idx: int,
    layer2_idx: int,
) -> float:
    model.eval()
    layers = get_decoder_layers(model)
    original_state_1 = state_dict_clone(layers[layer1_idx])
    original_state_2 = state_dict_clone(layers[layer2_idx])

    swap_layer_weights(model, layer1_idx, layer2_idx)

    try:
        swapped_loss = compute_mean_loss(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
        )
    finally:
        restore_state_dict(layers[layer1_idx], original_state_1)
        restore_state_dict(layers[layer2_idx], original_state_2)

    return abs(baseline_loss - swapped_loss) / baseline_loss if baseline_loss > 1e-8 else 0.0


def compute_all_permutation_scores(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    model_dtype: torch.dtype,
    num_layers: int,
    layer_pairs: List[Tuple[int, int]],
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    print("Computing permutation scores...")
    baseline_loss, _, _ = compute_baseline_forward(model, input_ids, attention_mask, device, model_dtype)
    permutation_scores = {}

    for layer1_idx, layer2_idx in tqdm(layer_pairs, desc="Computing permutation scores"):
        permutation_scores[(layer1_idx, layer2_idx)] = compute_permutation_score_for_pair(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            model_dtype=model_dtype,
            baseline_loss=baseline_loss,
            layer1_idx=layer1_idx,
            layer2_idx=layer2_idx,
        )

    return baseline_loss, permutation_scores


def compute_global_permutation_score(
    permutation_scores: Dict[Tuple[int, int], float],
    num_layers: int,
) -> Dict[str, float | List[float]]:
    if not permutation_scores:
        return {
            "global_score_mean": 0.0,
            "global_score_normalized": 0.0,
            "global_score_weighted": 0.0,
            "layer_avg_scores": [],
        }

    scores = list(permutation_scores.values())
    global_score_mean = float(np.mean(scores))
    max_pairs = num_layers * (num_layers - 1) / 2
    global_score_normalized = float(np.sum(scores) / max_pairs if max_pairs > 0 else 0.0)

    layer_counts = np.zeros(num_layers)
    layer_sums = np.zeros(num_layers)
    layer_weights = np.array([0.9**idx for idx in range(num_layers)])

    for (layer1_idx, layer2_idx), score in permutation_scores.items():
        layer_counts[layer1_idx] += 1
        layer_sums[layer1_idx] += score * layer_weights[layer1_idx]
        layer_counts[layer2_idx] += 1
        layer_sums[layer2_idx] += score * layer_weights[layer2_idx]

    layer_avg_scores = [
        float(layer_sums[idx] / layer_counts[idx]) if layer_counts[idx] > 0 else 0.0
        for idx in range(num_layers)
    ]
    global_score_weighted = float(np.mean(layer_avg_scores) if layer_avg_scores else 0.0)

    return {
        "global_score_mean": global_score_mean,
        "global_score_normalized": global_score_normalized,
        "global_score_weighted": global_score_weighted,
        "layer_avg_scores": layer_avg_scores,
    }


def plot_permutation_scores(permutation_scores: Dict[Tuple[int, int], float], num_layers: int, output_path: Path):
    score_matrix = np.zeros((num_layers, num_layers))
    score_matrix[:] = np.nan
    for (layer1_idx, layer2_idx), score in permutation_scores.items():
        score_matrix[layer1_idx, layer2_idx] = score

    mask = np.tril(np.ones_like(score_matrix, dtype=bool))
    valid_scores = score_matrix[~mask]
    finite_scores = valid_scores[np.isfinite(valid_scores)]

    if finite_scores.size == 0:
        vmin, vmax = 0.0, 1.0
        vcenter = 0.5
    else:
        vmin = float(np.min(finite_scores))
        vmax = float(np.max(finite_scores))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        # Shift the neutral boundary downward so mid-to-high values appear warmer.
        vcenter = vmin + 0.4 * (vmax - vmin)

    cmap = LinearSegmentedColormap.from_list(
        "permutation_score",
        ["#0b3c78", "#f7f3ec", "#7b001c"],
    )
    cmap.set_bad(color="white")
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        score_matrix,
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": "Permutation Score"},
        xticklabels=range(num_layers),
        yticklabels=range(num_layers),
        mask=mask,
        linewidths=0.3,
        linecolor=(1.0, 1.0, 1.0, 0.45),
        square=True,
    )
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.title("Permutation Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved permutation scores plot to {output_path}")
