import math
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

__all__ = [
    "HyperConnection",
    "HyperConnectionStreamExpand",
    "HyperConnectionStreamReduce",
    "sinkhorn_log",
]


def _rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 20, tau: float = 1.0) -> torch.Tensor:
    """
    Project logits onto the Birkhoff polytope with log-domain Sinkhorn iterations.
    """
    n = logits.shape[-1]
    z = logits.float() / tau
    log_marginal = -math.log(n)

    u = torch.zeros_like(z[..., 0])
    v = torch.zeros_like(z[..., 0, :])

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(z + u.unsqueeze(-1), dim=-2)

    projected = torch.exp(z + u.unsqueeze(-1) + v.unsqueeze(-2)) * n
    return projected.to(dtype=logits.dtype)


def _reshape_to_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected rank >= 2 tensor, got shape {tuple(x.shape)}")
    if x.shape[0] % num_streams != 0:
        raise ValueError(
            f"Batch dimension {x.shape[0]} is not divisible by num_streams={num_streams}"
        )

    batch = x.shape[0] // num_streams
    leading = x.shape[1:-1]
    d_model = x.shape[-1]
    return x.reshape(batch, num_streams, *leading, d_model).movedim(1, -2)


def _flatten_from_streams(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"Expected stream tensor rank >= 3, got shape {tuple(x.shape)}")

    batch = x.shape[0]
    num_streams = x.shape[-2]
    leading = x.shape[1:-2]
    d_model = x.shape[-1]
    return x.movedim(-2, 1).reshape(batch * num_streams, *leading, d_model)


class HyperConnectionStreamExpand(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        return x.unsqueeze(1).expand(-1, self.num_streams, *x.shape[1:]).reshape(
            x.shape[0] * self.num_streams, *x.shape[1:]
        )


class HyperConnectionStreamReduce(nn.Module):
    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        batch = x.shape[0] // self.num_streams
        return x.reshape(batch, self.num_streams, *x.shape[1:]).sum(dim=1)


class HyperConnection(nn.Module):
    """
    Residual backend for Hyper-Connections (HC) and Manifold-Constrained Hyper-Connections (mHC).

    The transformer keeps residual streams in the batch dimension with shape
    ``(batch_size * num_streams, seq_len, d_model)``.
    """

    def __init__(
        self,
        *,
        kind: str,
        num_residual_streams: int,
        dim: int,
        branch: Optional[nn.Module] = None,
        layer_index: Optional[int] = None,
        tanh: bool = True,
        gating_factor_init: float = 0.01,
        sinkhorn_iters: int = 20,
        sinkhorn_tau: float = 1.0,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if num_residual_streams < 1:
            raise ValueError("'num_residual_streams' must be >= 1")
        if kind not in {"hc", "mhc", "mhc_static"}:
            raise ValueError(f"Unsupported hyper-connection kind: {kind}")

        self.kind = kind
        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.selected_stream = (layer_index or 0) % num_residual_streams
        self.gating_factor_init = gating_factor_init
        self.bias_mag = 8.0
        self.activation = nn.Tanh() if tanh else nn.Identity()
        self.dropout = nn.Identity()
        if kind == "hc":
            static_alpha = torch.zeros(
                num_residual_streams,
                num_residual_streams + 1,
                device=init_device,
                dtype=dtype,
            )
            static_alpha[self.selected_stream, 0] = 1.0
            static_alpha[:, 1:] = torch.eye(
                num_residual_streams, device=init_device, dtype=dtype
            )
            self.static_alpha = nn.Parameter(static_alpha)
            self.dynamic_alpha_proj = nn.Parameter(
                torch.zeros(dim, num_residual_streams + 1, device=init_device, dtype=dtype)
            )
            self.dynamic_alpha_scale = nn.Parameter(
                torch.tensor(gating_factor_init, device=init_device, dtype=dtype)
            )

            self.static_beta = nn.Parameter(
                torch.ones(num_residual_streams, device=init_device, dtype=dtype)
            )
            self.dynamic_beta_proj = nn.Parameter(
                torch.zeros(dim, device=init_device, dtype=dtype)
            )
            self.dynamic_beta_scale = nn.Parameter(
                torch.tensor(gating_factor_init, device=init_device, dtype=dtype)
            )
        elif kind == "mhc":
            flat_dim = dim * num_residual_streams

            pre_bias = torch.full((num_residual_streams,), -self.bias_mag, device=init_device, dtype=dtype)
            pre_bias[self.selected_stream] = self.bias_mag
            self.pre_bias = nn.Parameter(pre_bias)
            self.post_bias = nn.Parameter(
                torch.zeros(num_residual_streams, device=init_device, dtype=dtype)
            )

            residual_bias = torch.full(
                (num_residual_streams, num_residual_streams),
                -self.bias_mag,
                device=init_device,
                dtype=dtype,
            )
            residual_bias.fill_diagonal_(self.bias_mag)
            self.residual_bias = nn.Parameter(residual_bias)

            self.pre_dynamic_proj = nn.Parameter(
                torch.zeros(flat_dim, num_residual_streams, device=init_device, dtype=dtype)
            )
            self.post_dynamic_proj = nn.Parameter(
                torch.zeros(flat_dim, num_residual_streams, device=init_device, dtype=dtype)
            )
            self.residual_dynamic_proj = nn.Parameter(
                torch.zeros(
                    flat_dim,
                    num_residual_streams * num_residual_streams,
                    device=init_device,
                    dtype=dtype,
                )
            )

            self.pre_gate = nn.Parameter(
                torch.tensor(gating_factor_init, device=init_device, dtype=dtype)
            )
            self.post_gate = nn.Parameter(
                torch.tensor(gating_factor_init, device=init_device, dtype=dtype)
            )
            self.residual_gate = nn.Parameter(
                torch.tensor(gating_factor_init, device=init_device, dtype=dtype)
            )
        else:
            h_res_logits = torch.full(
                (num_residual_streams, num_residual_streams),
                -self.bias_mag,
                device=init_device,
                dtype=dtype,
            )
            h_res_logits.fill_diagonal_(0.0)
            self.H_res_logits = nn.Parameter(h_res_logits)

            h_pre_logits = torch.full(
                (num_residual_streams,), -self.bias_mag, device=init_device, dtype=dtype
            )
            h_pre_logits[self.selected_stream] = 0.0
            self.H_pre_logits = nn.Parameter(h_pre_logits)
            self.H_post_logits = nn.Parameter(
                torch.zeros(num_residual_streams, device=init_device, dtype=dtype)
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.kind == "hc":
            self.static_alpha.zero_()
            self.static_alpha[self.selected_stream, 0] = 1.0
            self.static_alpha[:, 1:] = torch.eye(
                self.num_residual_streams,
                device=self.static_alpha.device,
                dtype=self.static_alpha.dtype,
            )
            self.dynamic_alpha_proj.zero_()
            self.dynamic_alpha_scale.fill_(self.gating_factor_init)
            self.static_beta.fill_(1.0)
            self.dynamic_beta_proj.zero_()
            self.dynamic_beta_scale.fill_(self.gating_factor_init)
            return

        if self.kind == "mhc_static":
            self.H_res_logits.fill_(-self.bias_mag)
            self.H_res_logits.fill_diagonal_(0.0)
            self.H_pre_logits.fill_(-self.bias_mag)
            self.H_pre_logits[self.selected_stream] = 0.0
            self.H_post_logits.zero_()
            return

        self.pre_bias.fill_(-self.bias_mag)
        self.pre_bias[self.selected_stream] = self.bias_mag
        self.post_bias.zero_()
        self.residual_bias.fill_(-self.bias_mag)
        self.residual_bias.fill_diagonal_(self.bias_mag)
        self.pre_dynamic_proj.zero_()
        self.post_dynamic_proj.zero_()
        self.residual_dynamic_proj.zero_()
        self.pre_gate.fill_(self.gating_factor_init)
        self.post_gate.fill_(self.gating_factor_init)
        self.residual_gate.fill_(self.gating_factor_init)

    def _forward_hc(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        streams = _reshape_to_streams(residuals, self.num_residual_streams)
        normed = _rms_norm(streams)

        dynamic_alpha = self.dynamic_alpha_scale * self.activation(
            torch.matmul(normed, self.dynamic_alpha_proj)
        )
        alpha = dynamic_alpha + self.static_alpha
        mixed = torch.einsum("...st,...sd->...td", alpha, streams)
        branch_input = mixed[..., 0, :]

        branch_output = self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)

        beta = self.static_beta + self.dynamic_beta_scale * self.activation(
            torch.matmul(normed, self.dynamic_beta_proj)
        )
        branch_to_streams = branch_output.unsqueeze(-2) * beta.unsqueeze(-1)
        # HC uses stream mixing only to form the branch input. The depth connection adds the
        # branch output back to the original residual streams, matching the reference implementation.
        output = self.dropout(residuals + _flatten_from_streams(branch_to_streams))

        return tree_unflatten((output, *rest), tree_spec)

    def _forward_mhc(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        streams = _reshape_to_streams(residuals, self.num_residual_streams)
        streams_fp32 = streams.float()
        flat_streams = streams_fp32.reshape(*streams.shape[:-2], -1)
        normed = _rms_norm(flat_streams)

        pre_tilde = self.pre_gate.float() * self.activation(
            torch.matmul(normed, self.pre_dynamic_proj.float())
        ) + self.pre_bias.float()
        post_tilde = self.post_gate.float() * self.activation(
            torch.matmul(normed, self.post_dynamic_proj.float())
        ) + self.post_bias.float()
        residual_tilde = self.residual_gate.float() * self.activation(
            torch.matmul(normed, self.residual_dynamic_proj.float())
        ).reshape(*normed.shape[:-1], self.num_residual_streams, self.num_residual_streams) + self.residual_bias.float()

        h_pre = torch.sigmoid(pre_tilde)
        h_post = 2.0 * torch.sigmoid(post_tilde)
        h_res = sinkhorn_log(residual_tilde, num_iters=self.sinkhorn_iters)

        branch_input = torch.einsum("...s,...sd->...d", h_pre, streams_fp32).to(streams.dtype)
        mixed_residuals = torch.einsum("...st,...sd->...td", h_res, streams_fp32)

        branch_output = self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)

        branch_to_streams = branch_output.float().unsqueeze(-2) * h_post.unsqueeze(-1)
        output = self.dropout(
            _flatten_from_streams(mixed_residuals + branch_to_streams).to(residuals.dtype)
        )

        return tree_unflatten((output, *rest), tree_spec)

    def _forward_mhc_static(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        streams = _reshape_to_streams(residuals, self.num_residual_streams)

        h_res = sinkhorn_log(
            self.H_res_logits,
            num_iters=self.sinkhorn_iters,
            tau=self.sinkhorn_tau,
        ).to(dtype=streams.dtype)
        h_pre = torch.softmax(self.H_pre_logits.float(), dim=-1).to(dtype=streams.dtype)
        h_post = torch.softmax(self.H_post_logits.float(), dim=-1).to(dtype=streams.dtype)

        mixed_residuals = torch.einsum("st,...sd->...td", h_res, streams)
        branch_input = torch.einsum("s,...sd->...d", h_pre, streams)

        branch_output = self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)

        branch_to_streams = branch_output.unsqueeze(-2) * h_post.to(dtype=branch_output.dtype).view(
            *((1,) * (branch_output.ndim - 1)),
            self.num_residual_streams,
            1,
        )
        output = self.dropout(_flatten_from_streams(mixed_residuals + branch_to_streams))

        return tree_unflatten((output, *rest), tree_spec)

    def forward(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.kind == "hc":
            return self._forward_hc(residuals, *args, **kwargs)
        if self.kind == "mhc_static":
            return self._forward_mhc_static(residuals, *args, **kwargs)
        return self._forward_mhc(residuals, *args, **kwargs)
