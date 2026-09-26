"""Loss and layer input/output helpers used by `compute_usefulness_score.py`."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from analysis_utils import autocast_context, model_forward, normalize_layer_output


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
