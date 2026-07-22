import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

__all__ = [
    "HyperConnection",
    "HyperConnectionStreamExpand",
    "HyperConnectionStreamReduce",
    "LigerHyperConnection",
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
        return (
            x.unsqueeze(1)
            .expand(-1, self.num_streams, *x.shape[1:])
            .reshape(x.shape[0] * self.num_streams, *x.shape[1:])
        )


class HyperConnectionStreamReduce(nn.Module):
    def __init__(self, num_streams: int, mode: str = "sum"):
        super().__init__()
        self.num_streams = num_streams
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unsupported stream reduce mode: {mode}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        batch = x.shape[0] // self.num_streams
        x = x.reshape(batch, self.num_streams, *x.shape[1:])
        if self.mode == "mean":
            return x.mean(dim=1)
        return x.sum(dim=1)


def _require_liger_mhc():
    try:
        from liger_kernel.transformers.functional import (  # type: ignore
            liger_mhc_coeffs,
            liger_mhc_post_res,
            liger_mhc_pre,
        )
    except Exception as e:
        raise ImportError(
            "Liger-backed mHC requires liger-kernel>=0.8.0 with Liger mHC functional "
            "APIs. Install it with `pip install 'liger-kernel>=0.8.0'` or from the "
            "LinkedIn Liger-Kernel source tree."
        ) from e

    return liger_mhc_coeffs, liger_mhc_pre, liger_mhc_post_res


def _check_liger_mhc_runtime(tmax: int):
    if tmax < 8:
        return

    try:
        from packaging.version import Version
        import triton
    except Exception as e:
        raise RuntimeError(
            "Liger-backed mHC with sinkhorn_iters>=8 requires a working Triton "
            "runtime. Use an environment with torch>=2.8 and triton>=3.4 for "
            "the paper-default sinkhorn_iters=20."
        ) from e

    if Version(triton.__version__) < Version("3.4.0"):
        raise RuntimeError(
            f"Liger-backed mHC sinkhorn_iters={tmax} is not usable with "
            f"triton {triton.__version__}: the coefficient kernel can hang at "
            "the paper-default tmax=20. Use the newer torch>=2.8 / triton>=3.4 "
            "environment, or lower sinkhorn_iters only for debugging."
        )


class LigerHyperConnection(nn.Module):
    """
    Liger-backed mHC residual backend.

    The transformer keeps residual streams folded into the batch dimension with shape
    ``(batch_size * num_streams, seq_len, d_model)``. Liger's kernels operate on
    ``(..., num_streams, d_model)``, so this module reshapes only around the fused
    coefficient/pre/post-res kernels.
    """

    def __init__(
        self,
        *,
        num_residual_streams: int,
        dim: int,
        branch: Optional[nn.Module] = None,
        gating_factor_init: float = 0.01,
        sinkhorn_iters: int = 20,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        phi_dtype: torch.dtype = torch.bfloat16,
        allow_fp32: bool = False,
        rms_eps: float = 1e-6,
        pre_eps: float = 0.0,
        sinkhorn_eps: float = 1e-6,
        post_mult: float = 2.0,
    ):
        super().__init__()

        if num_residual_streams < 1:
            raise ValueError("'num_residual_streams' must be >= 1")

        self.liger_mhc_coeffs, self.liger_mhc_pre, self.liger_mhc_post_res = _require_liger_mhc()
        _check_liger_mhc_runtime(sinkhorn_iters)

        self.kind = "liger_mhc"
        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.dim = dim
        self.tmax = int(sinkhorn_iters)
        self.gating_factor_init = float(gating_factor_init)
        self.allow_fp32 = bool(allow_fp32)
        self.rms_eps = float(rms_eps)
        self.pre_eps = float(pre_eps)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.post_mult = float(post_mult)
        self.dropout = nn.Identity()

        k = num_residual_streams * dim
        m = num_residual_streams * num_residual_streams + 2 * num_residual_streams
        self.phi = nn.Parameter(torch.empty(k, m, device=init_device, dtype=phi_dtype))
        self.b = nn.Parameter(torch.empty(m, device=init_device, dtype=torch.float32))
        self.alpha_pre = nn.Parameter(torch.empty((), device=init_device, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.empty((), device=init_device, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.empty((), device=init_device, dtype=torch.float32))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.phi.normal_(mean=0.0, std=0.02)
        self.b.zero_()
        self.alpha_pre.fill_(self.gating_factor_init)
        self.alpha_post.fill_(self.gating_factor_init)
        self.alpha_res.fill_(self.gating_factor_init)

    def _branch_param_dtype(self, fallback: torch.dtype) -> torch.dtype:
        if self.branch is None:
            return fallback
        for param in self.branch.parameters(recurse=True):
            return param.dtype
        return fallback

    def forward(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        streams = _reshape_to_streams(residuals, self.num_residual_streams).contiguous()
        if streams.shape[-2] != self.num_residual_streams or streams.shape[-1] != self.dim:
            raise ValueError(
                f"Expected Liger mHC streams with shape [..., {self.num_residual_streams}, "
                f"{self.dim}], got {tuple(streams.shape)}"
            )

        h_pre, h_post, h_res = self.liger_mhc_coeffs(
            streams,
            self.phi,
            self.b,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            allow_fp32=self.allow_fp32,
            tmax=self.tmax,
            rms_eps=self.rms_eps,
            pre_eps=self.pre_eps,
            sinkhorn_eps=self.sinkhorn_eps,
            post_mult=self.post_mult,
        )
        branch_input = self.liger_mhc_pre(streams, h_pre)
        branch_dtype = self._branch_param_dtype(branch_input.dtype)
        if branch_input.dtype != branch_dtype:
            branch_input = branch_input.to(dtype=branch_dtype)

        branch_output = (
            self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        )
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)
        output = self.liger_mhc_post_res(
            streams,
            branch_output.to(dtype=streams.dtype),
            h_post,
            h_res,
        )
        output = self.dropout(_flatten_from_streams(output).to(dtype=residuals.dtype))

        return tree_unflatten((output, *rest), tree_spec)


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
            static_alpha[:, 1:] = torch.eye(num_residual_streams, device=init_device, dtype=dtype)
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
            self.dynamic_beta_proj = nn.Parameter(torch.zeros(dim, device=init_device, dtype=dtype))
            self.dynamic_beta_scale = nn.Parameter(
                torch.tensor(gating_factor_init, device=init_device, dtype=dtype)
            )
        elif kind == "mhc":
            flat_dim = dim * num_residual_streams

            pre_bias = torch.full(
                (num_residual_streams,), -self.bias_mag, device=init_device, dtype=dtype
            )
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

        branch_output = (
            self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        )
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)

        beta = self.static_beta + self.dynamic_beta_scale * self.activation(
            torch.matmul(normed, self.dynamic_beta_proj)
        )
        branch_to_streams = branch_output.unsqueeze(-2) * beta.unsqueeze(-1)
        # The remaining width-mixed columns are A_r^T H in the paper's depth connection.
        output_streams = mixed[..., 1:, :] + branch_to_streams
        output = self.dropout(_flatten_from_streams(output_streams))

        return tree_unflatten((output, *rest), tree_spec)

    def _forward_mhc(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        streams = _reshape_to_streams(residuals, self.num_residual_streams)
        streams_fp32 = streams.float()
        flat_streams = streams_fp32.reshape(*streams.shape[:-2], -1)
        normed = _rms_norm(flat_streams)

        pre_tilde = (
            self.pre_gate.float()
            * self.activation(torch.matmul(normed, self.pre_dynamic_proj.float()))
            + self.pre_bias.float()
        )
        post_tilde = (
            self.post_gate.float()
            * self.activation(torch.matmul(normed, self.post_dynamic_proj.float()))
            + self.post_bias.float()
        )
        residual_tilde = (
            self.residual_gate.float()
            * self.activation(torch.matmul(normed, self.residual_dynamic_proj.float())).reshape(
                *normed.shape[:-1], self.num_residual_streams, self.num_residual_streams
            )
            + self.residual_bias.float()
        )

        h_pre = torch.sigmoid(pre_tilde)
        h_post = 2.0 * torch.sigmoid(post_tilde)
        h_res = sinkhorn_log(
            residual_tilde,
            num_iters=self.sinkhorn_iters,
            tau=self.sinkhorn_tau,
        )

        branch_input = torch.einsum("...s,...sd->...d", h_pre, streams_fp32).to(streams.dtype)
        mixed_residuals = torch.einsum("...st,...sd->...td", h_res, streams_fp32)

        branch_output = (
            self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        )
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)

        branch_to_streams = branch_output.float().unsqueeze(-2) * h_post.unsqueeze(-1)
        output = self.dropout(
            _flatten_from_streams(mixed_residuals + branch_to_streams).to(residuals.dtype)
        )

        return tree_unflatten((output, *rest), tree_spec)

    def _forward_mhc_static(
        self, residuals: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
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

        branch_output = (
            self.branch(branch_input, *args, **kwargs) if self.branch is not None else branch_input
        )
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
