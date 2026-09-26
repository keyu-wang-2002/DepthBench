"""Architecture-aware depth probes shared by the DepthBench analysis scripts.

Every metric is phrased over *depth states* ``z_0, ..., z_L`` (``z_0`` is the embedding
output, ``z_{l+1}`` is the state produced by block ``l``) and over two interventions on
single blocks: ``skip_block`` (remove block ``l``) and ``swap_blocks`` (exchange the learned
weights of blocks ``i`` and ``j`` while keeping the architecture fixed). The supported
architecture families only differ in how these are realised:

residual
    Pre-LN, Sandwich/Peri-LN, LNS, DeepNorm, KEEL and MoDA. ``z_l`` is the block output
    and a skipped block is the identity. A skipped MoDA block writes no depth-KV entries,
    so its slots in the depth cache stay zero.
hyper-connections
    HC and mHC. ``z_l`` concatenates the ``n`` residual streams of the block output into
    one ``n * d`` vector per token. The logit-lens readout collapses the streams with the
    model's own output reduction, which matches the stream mean up to the scale removed
    by the final RMSNorm.
attnres
    Full AttnRes (``attnres_block_size=1``) and Block AttnRes. ``z_l`` is the depth-mixed
    state at every true AttnRes boundary (block ``l``'s attention-side mix, or the final
    mix for ``l = L``) and the plain block output inside a block group; with Full AttnRes
    every position is a boundary. A skipped block adds nothing to the source bank and,
    when its next consumer is a boundary mix, that mix becomes the identity on its newest
    source ("next-output" rule). A swap exchanges each block's core weights together with
    the depth-mixing parameters that produce its right boundary. The logit-lens readout
    of ``z_l`` is the mix computed by the next consumer (block ``l``'s attention-side mix
    or the final mix), decoded with the final norm and LM head.
"""

from __future__ import annotations

import types
from contextlib import ExitStack, contextmanager
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis_utils import autocast_context, get_decoder_layers, iter_micro_batches, normalize_layer_output

RESIDUAL = "residual"
HYPER_CONNECTIONS = "hyper-connections"
ATTNRES = "attnres"


def model_family(model) -> str:
    if getattr(model, "_attnres_enabled", False):
        return ATTNRES
    if getattr(model, "_hyper_connection_enabled", False):
        return HYPER_CONNECTIONS
    return RESIDUAL


def _num_streams(model) -> int:
    return int(getattr(model, "_hyper_connection_num_streams", 1))


def _boundary_state_indices(model) -> list[int]:
    """Depth-state indices whose AttnRes state is a true boundary mix."""
    layers = get_decoder_layers(model)
    indices = [idx for idx in range(1, len(layers)) if layers[idx].attnres_is_attn_boundary]
    return [*indices, len(layers)]


def describe_architecture(model) -> dict:
    layers = get_decoder_layers(model)
    family = model_family(model)
    info = {
        "family": family,
        "block_class": type(layers[0]).__name__,
        "num_layers": len(layers),
    }
    if family == HYPER_CONNECTIONS:
        info["num_residual_streams"] = _num_streams(model)
    if family == ATTNRES:
        info["attnres_block_size"] = int(layers[0].attnres_block_size)
        info["attnres_boundary_state_indices"] = _boundary_state_indices(model)
    return info


# ---------------------------------------------------------------------------
# AttnRes depth-mixing interception
# ---------------------------------------------------------------------------


@contextmanager
def _intercept_fused_attnres(interceptor: Callable) -> Iterator[None]:
    """Route every `fused_attnres(**kwargs)` call through `interceptor(original, kwargs)`.

    AttnRes blocks and the model import `fused_attnres` from the kernel module at call time,
    so patching the module attribute reaches every depth-mixing call. Interceptors nest.
    """
    import olmo_core.kernels.attnres as attnres_kernel

    original = attnres_kernel.fused_attnres

    def wrapped(*args, **kwargs):
        if args:
            raise TypeError("fused_attnres is expected to be called with keyword arguments only")
        return interceptor(original, kwargs)

    attnres_kernel.fused_attnres = wrapped
    try:
        yield
    finally:
        attnres_kernel.fused_attnres = original


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (x_float * weight.float()).to(x.dtype)


def _mix_queries(model, state_indices: list[int]) -> dict[int, int]:
    """Map id(depth-mixing query parameter) -> depth-state index it produces."""
    layers = get_decoder_layers(model)
    queries = {}
    for idx in state_indices:
        query = model.res_proj.weight if idx == len(layers) else layers[idx].attn_res_proj.weight
        queries[id(query)] = idx
    return queries


# ---------------------------------------------------------------------------
# Depth states
# ---------------------------------------------------------------------------


def _forward_with_states(model, input_ids: torch.Tensor, mix_state_indices: list[int]):
    """Run the model once and return (logits, [z_0, ..., z_L]).

    Block outputs are captured with hooks; for AttnRes the states listed in
    `mix_state_indices` are replaced by the raw (pre-norm) depth mix that produces them.
    """
    layers = get_decoder_layers(model)
    states: list[torch.Tensor | None] = [None] * (len(layers) + 1)
    handles = []

    def embedding_hook(_module, args):
        states[0] = normalize_layer_output(args[0])

    handles.append(layers[0].register_forward_pre_hook(embedding_hook))
    for idx, layer in enumerate(layers):
        def output_hook(_module, _args, output, idx=idx):
            states[idx + 1] = normalize_layer_output(output)

        handles.append(layer.register_forward_hook(output_hook))

    mixes: dict[int, torch.Tensor] = {}
    with ExitStack() as stack:
        if model_family(model) == ATTNRES and mix_state_indices:
            queries = _mix_queries(model, mix_state_indices)

            def capture(original, kwargs):
                output = original(**kwargs)
                idx = queries.get(id(kwargs["query"]))
                if idx is not None:
                    if kwargs.get("output_rms_weight") is None:
                        mixes[idx] = output
                    else:
                        mixes[idx] = original(**{**kwargs, "output_rms_weight": None})
                return output

            stack.enter_context(_intercept_fused_attnres(capture))
        try:
            logits = model(input_ids=input_ids)
        finally:
            for handle in handles:
                handle.remove()

    for idx, mix in mixes.items():
        states[idx] = mix
    if any(state is None for state in states):
        raise RuntimeError("Failed to capture every depth state")
    return logits, states


def forward_depth_states(model, input_ids: torch.Tensor):
    """Depth states used for geometry (angular distance, causal score): (logits, states).

    Each state has shape (batch, seq, width); width is n*d for HC/mHC.
    """
    family = model_family(model)
    mix_indices = _boundary_state_indices(model) if family == ATTNRES else []
    logits, states = _forward_with_states(model, input_ids, mix_indices)
    if family == HYPER_CONNECTIONS:
        from olmo_core.nn.hyper_connections import _reshape_to_streams

        states = [_reshape_to_streams(state, _num_streams(model)).flatten(-2) for state in states]
    return logits, states


def forward_readout_states(model, input_ids: torch.Tensor):
    """Single-stream states the next consumer reads, for the logit lens: (logits, states).

    Decode a state with `decode_readout`.
    """
    family = model_family(model)
    mix_indices = list(range(1, len(get_decoder_layers(model)) + 1)) if family == ATTNRES else []
    logits, states = _forward_with_states(model, input_ids, mix_indices)
    if family == HYPER_CONNECTIONS:
        states = [model.reduce_residual_streams(state) for state in states]
    return logits, states


def decode_readout(model, state: torch.Tensor) -> torch.Tensor:
    """Final norm + LM head, applied exactly as the model applies it to its own output."""
    if model_family(model) == ATTNRES and model.lm_head.norm is not None:
        # AttnRes fuses the final RMSNorm into the last depth mix (using the mixer's eps).
        normed = _rms_norm(state, model.lm_head.norm.weight, model.res_norm.eps)
        return model.lm_head(normed, skip_norm=True)
    return model.lm_head(state)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


def _next_boundary_query(model, layer_idx: int) -> nn.Parameter | None:
    """Query of the boundary depth mix that consumes block `layer_idx`'s output, if any."""
    layers = get_decoder_layers(model)
    if layer_idx + 1 == len(layers):
        return model.res_proj.weight
    next_layer = layers[layer_idx + 1]
    return next_layer.attn_res_proj.weight if next_layer.attnres_is_attn_boundary else None


@contextmanager
def skip_block(model, layer_idx: int) -> Iterator[None]:
    """Remove block `layer_idx` from the forward pass (see module docstring per family)."""
    layer = get_decoder_layers(model)[layer_idx]
    family = model_family(model)

    if family == ATTNRES:

        def skipped_forward(self, x, *args, attnres_states=None, **kwargs):
            if attnres_states is None:
                return x, [x]
            if hasattr(self, "attn_res_proj"):
                # Still form the block's input mix so that its left boundary z_l is observable.
                from olmo_core.kernels.attnres import fused_attnres

                fused_attnres(
                    query=self.attn_res_proj.weight,
                    residuals=[*attnres_states, x],
                    rms_weight=self.attn_res_norm.weight,
                    output_rms_weight=None,
                    rms_eps=self.attn_res_norm.eps,
                )
            return x, attnres_states

    else:

        def skipped_forward(self, x, *args, **kwargs):
            # MoDA passes (hidden, depth_k, depth_v) and HC passes folded streams through unchanged.
            return x

    with ExitStack() as stack:
        consumer_query = _next_boundary_query(model, layer_idx) if family == ATTNRES else None
        if consumer_query is not None:

            def identity_mix(original, kwargs):
                if kwargs["query"] is not consumer_query:
                    return original(**kwargs)
                newest = kwargs["residuals"][-1]
                weight = kwargs.get("output_rms_weight")
                return newest if weight is None else _rms_norm(newest, weight, kwargs.get("rms_eps", 1e-6))

            stack.enter_context(_intercept_fused_attnres(identity_mix))
        layer.forward = types.MethodType(skipped_forward, layer)
        try:
            yield
        finally:
            del layer.forward


def _is_attnres_input_mixer(name: str) -> bool:
    return name.startswith("attn_res_proj.") or name.startswith("attn_res_norm.")


def _swap_pairs(model, i: int, j: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layers = get_decoder_layers(model)
    family = model_family(model)
    state_i = layers[i].state_dict()
    state_j = layers[j].state_dict()
    pairs = []
    for name, tensor_i in state_i.items():
        if family == ATTNRES and _is_attnres_input_mixer(name):
            continue
        tensor_j = state_j.get(name)
        # Structurally different keys (e.g. KEEL block 0 has no post-attention norm) stay in place.
        if tensor_j is not None and tensor_j.shape == tensor_i.shape:
            pairs.append((tensor_i, tensor_j))

    if family == ATTNRES:
        def output_boundary(idx: int) -> list[torch.Tensor]:
            if idx + 1 == len(layers):
                return [model.res_proj.weight, model.res_norm.weight]
            nxt = layers[idx + 1]
            return [nxt.attn_res_proj.weight, nxt.attn_res_norm.weight]

        pairs.extend(zip(output_boundary(i), output_boundary(j)))
    return pairs


@torch.no_grad()
def swap_blocks(model, i: int, j: int) -> None:
    """Exchange the learned weights of blocks `i` and `j` in place. Calling it twice restores."""
    for tensor_a, tensor_b in _swap_pairs(model, i, j):
        tmp = tensor_a.detach().clone()
        tensor_a.data.copy_(tensor_b.data)
        tensor_b.data.copy_(tmp)
    layers = get_decoder_layers(model)
    # Position-derived scalars (LNS `ln_scale`) travel with the weights.
    if hasattr(layers[i], "ln_scale") and hasattr(layers[j], "ln_scale"):
        layers[i].ln_scale, layers[j].ln_scale = layers[j].ln_scale, layers[i].ln_scale


@contextmanager
def swapped_blocks(model, i: int, j: int) -> Iterator[None]:
    swap_blocks(model, i, j)
    try:
        yield
    finally:
        swap_blocks(model, i, j)


# ---------------------------------------------------------------------------
# Language-modeling loss
# ---------------------------------------------------------------------------


def next_token_loss_sum(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[float, int]:
    """Summed next-token cross entropy over all positions and the number of predicted tokens."""
    targets = input_ids[:, 1:]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="sum",
    )
    return float(loss), targets.numel()


@torch.no_grad()
def mean_lm_loss(model, input_ids: torch.Tensor, *, micro_batch_size: int, device: str, dtype: torch.dtype) -> float:
    total, count = 0.0, 0
    for batch in iter_micro_batches(input_ids, micro_batch_size, device):
        with autocast_context(device, dtype):
            logits = model(input_ids=batch)
        loss_sum, num_tokens = next_token_loss_sum(logits, batch)
        total += loss_sum
        count += num_tokens
    return total / count
