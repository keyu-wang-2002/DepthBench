from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError

from .attention import Attention
from .feed_forward import FeedForward

if TYPE_CHECKING:
    from .transformer.config import MoDAConfig


def _load_parallel_moda(backend: str):
    try:
        if backend == "v17":
            from fla.ops.moda import parallel_moda_v17 as parallel_moda
        else:
            from fla.ops.moda import parallel_moda
    except Exception as exc:
        raise ImportError(
            "MoDA blocks require the official MoDA Triton kernels. Install them with "
            "`pip install -e /path/to/MoDA/libs/moda_triton` "
            "inside the training environment."
        ) from exc
    return parallel_moda


class _WriteMoDADepthSlotKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buf_k, buf_v, k_data, v_data, slot_idx: int, max_depth: int):
        B, T, L, H, D = buf_k.shape
        Vdim = buf_v.shape[-1]
        if L != max_depth:
            raise RuntimeError("MoDA depth cache depth dim must equal max_depth")
        if buf_v.shape != (B, T, max_depth, H, Vdim):
            raise RuntimeError("MoDA V depth cache shape mismatch")
        ctx.slot_idx = slot_idx
        # Bypass the version counter; backward explicitly routes this slot's gradient.
        buf_k.data[:, :, slot_idx].copy_(k_data.detach())
        buf_v.data[:, :, slot_idx].copy_(v_data.detach())
        return buf_k, buf_v

    @staticmethod
    def backward(ctx, grad_buf_k, grad_buf_v):
        slot_grad_k = grad_buf_k[:, :, ctx.slot_idx].contiguous()
        slot_grad_v = grad_buf_v[:, :, ctx.slot_idx].contiguous()
        return grad_buf_k, grad_buf_v, slot_grad_k, slot_grad_v, None, None


@torch.compiler.disable
def _write_moda_depth_slot(buf_k, buf_v, k_data, v_data, slot: int, max_depth: int):
    if slot + 1 >= max_depth:
        return buf_k, buf_v
    return _WriteMoDADepthSlotKV.apply(buf_k, buf_v, k_data, v_data, slot, max_depth)


class MoDAAttention(nn.Module):
    """
    OLMo Attention wrapper that replaces SDPA with official MoDA depth attention
    after the first block while reusing the baseline Q/K/V/O projections.
    """

    def __init__(
        self,
        attention: Attention,
        *,
        moda: MoDAConfig,
        block_idx: int,
        n_layers: int,
    ):
        super().__init__()
        if not isinstance(attention, Attention):
            raise OLMoConfigurationError("MoDA currently supports only default Attention blocks")
        if attention.window_size is not None:
            raise OLMoConfigurationError("MoDA does not support sliding-window attention")
        self.attention = attention
        self.moda = moda
        self.block_idx = block_idx
        self.n_layers = n_layers
        self.max_depth = 2 * n_layers
        self._parallel_moda = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            attention = self._modules.get("attention")
            if attention is not None and hasattr(attention, name):
                return getattr(attention, name)
            raise

    @property
    def n_heads(self) -> int:
        return self.attention.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.attention.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.attention.head_dim

    def init_weights(self, *args, **kwargs) -> None:
        self.attention.init_weights(*args, **kwargs)

    def _kernel(self):
        if self._parallel_moda is None:
            self._parallel_moda = _load_parallel_moda(self.moda.backend)
        return self._parallel_moda

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.attention.w_q(x)
        k = self.attention.w_k(x)
        v = self.attention.w_v(x)

        if self.attention.clip_qkv is not None:
            q.clamp_(min=-self.attention.clip_qkv, max=self.attention.clip_qkv)
            k.clamp_(min=-self.attention.clip_qkv, max=self.attention.clip_qkv)
            v.clamp_(min=-self.attention.clip_qkv, max=self.attention.clip_qkv)

        raw_k = k

        if not self.attention.use_head_qk_norm:
            if self.attention.q_norm is not None:
                q = self.attention.q_norm(q)
            if self.attention.k_norm is not None:
                k = self.attention.k_norm(k)

        q = q.view(B, T, -1, self.head_dim)
        k = k.view(B, T, -1, self.head_dim)
        v = v.view(B, T, -1, self.head_dim)
        raw_k = raw_k.view(B, T, -1, self.head_dim)

        if self.attention.use_head_qk_norm:
            if self.attention.q_norm is not None:
                q = self.attention.q_norm(q)
            if self.attention.k_norm is not None:
                k = self.attention.k_norm(k)

        return q, k, v, raw_k

    def normalize_cache_k(self, k: torch.Tensor) -> torch.Tensor:
        if self.attention.k_norm is None:
            return k
        if self.attention.use_head_qk_norm:
            return self.attention.k_norm(k)
        B, T, H, D = k.shape
        return self.attention.k_norm(k.reshape(B, T, H * D)).view(B, T, H, D)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention.rope is None:
            return q, k
        if self.attention.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
            raise RuntimeError(
                "MoDA does not support context parallelism without pre-sharded RoPE buffers"
            )
        return self.attention.rope(
            q,
            k,
            head_first=False,
            start_pos=None,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
        )

    def _apply_gate(self, x: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        if self.attention.gate is None:
            return att
        assert self.attention.w_g is not None
        g = self.attention.w_g(x)
        if self.attention.gate.full_precision:
            g = g.float()
        gate_values = torch.sigmoid(g).to(att.dtype)
        if self.attention.gate.granularity == "headwise":
            return att * gate_values.unsqueeze(-1)
        B, T, _, _ = att.shape
        return (att.view(B, T, -1) * gate_values).view_as(att)

    @torch.compiler.disable
    def _extract_cached_kv(self, buf_k: torch.Tensor, buf_v: torch.Tensor, current_depth: int):
        if current_depth <= 0:
            return None, None
        B, T, max_depth, h_kv, d = buf_k.shape
        if current_depth > max_depth:
            raise RuntimeError(
                f"MoDA current depth {current_depth} exceeds max depth {max_depth}"
            )

        # Keep the preallocated max-depth layout. Slicing before reshape copies
        # every populated slot because the depth stride remains max_depth,
        # defeating the v17 kernel's O(1)-allocation cache design.
        cached_k = buf_k.reshape(B, T * max_depth, h_kv, d)
        cached_v = buf_v.reshape(B, T * max_depth, h_kv, d)
        return cached_k, cached_v

    def _depth_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        *,
        current_depth: int,
    ) -> torch.Tensor:
        B, T, h_q, d = q.shape
        h_kv = self.n_kv_heads
        if h_q % h_kv != 0:
            raise RuntimeError(f"MoDA requires n_heads % n_kv_heads == 0, got {h_q} and {h_kv}")
        group = h_q // h_kv
        q_moda = (
            q.reshape(B, T, h_kv, group, d).permute(0, 1, 3, 2, 4).reshape(B, T * group, h_kv, d)
        )
        scale = self.moda.attention_scale
        if scale is None:
            scale = 1.0 / math.sqrt(d)

        cached_k = cached_k.to(q_moda.dtype)
        cached_v = cached_v.to(q_moda.dtype)
        y = self._kernel()(
            q_moda,
            k.to(q_moda.dtype),
            v.to(q_moda.dtype),
            cached_k=cached_k,
            cached_v=cached_v,
            scale=scale,
            moda_group_num=group,
            is_causal=True,
            current_depth=current_depth,
            depth_bs=self.moda.depth_bs,
            depth_warps=self.moda.depth_warps,
        )
        return y.view(B, T, group, h_kv, d).permute(0, 1, 3, 2, 4).reshape(B, T, h_q, d)

    def forward(
        self,
        x: torch.Tensor,
        *,
        buf_k: torch.Tensor,
        buf_v: torch.Tensor,
        current_depth: int,
        slot: int,
        max_depth: int,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del max_depth
        B, T, C = x.shape
        q, k, v, raw_k = self._project_qkv(x)

        if self.moda.cache_post_norm_k and self.attention.k_norm is not None:
            k_for_cache = k
        else:
            k_for_cache = raw_k
        v_for_cache = v

        q, k = self._apply_rope(q, k, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis)

        if current_depth == 0:
            att = self.attention.sdpa(
                q,
                k,
                v,
                cu_doc_lens=cu_doc_lens,
                cu_doc_lens_q=cu_doc_lens_q,
                cu_doc_lens_k=cu_doc_lens_k,
                max_doc_len=max_doc_len,
                max_doc_len_q=max_doc_len_q,
                max_doc_len_k=max_doc_len_k,
                local_k_slice=local_k_slice,
                cache_leftpad=cache_leftpad,
            )
        else:
            unsupported = [
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
                cache_leftpad,
            ]
            if any(value is not None for value in unsupported):
                raise NotImplementedError("MoDA depth attention does not support document masks")
            cached_k, cached_v = self._extract_cached_kv(buf_k, buf_v, current_depth)
            assert cached_k is not None and cached_v is not None
            att = self._depth_attention(q, k, v, cached_k, cached_v, current_depth=current_depth)

        att = self._apply_gate(x, att)
        out = self.attention.w_out(att.reshape(B, T, -1))
        buf_k, buf_v = _write_moda_depth_slot(
            buf_k, buf_v, k_for_cache, v_for_cache, slot, self.max_depth
        )
        return out, buf_k, buf_v

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.attention.num_flops_per_token(seq_len)


class MoDAFeedForwardDepthCache(nn.Module):
    def __init__(
        self,
        feed_forward: FeedForward,
        *,
        d_model: int,
        n_kv_heads: int,
        head_dim: int,
        bias: bool,
        moda: MoDAConfig,
        skip_kv_proj: bool,
        init_device: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.feed_forward = feed_forward
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.moda = moda
        self.kv_proj = (
            None
            if skip_kv_proj or not moda.extra_ffn_kv_proj
            else nn.Linear(
                d_model,
                2 * n_kv_heads * head_dim,
                bias=bias,
                dtype=dtype,
                device=init_device,
            )
        )

    @property
    def w1(self):
        return self.feed_forward.w1

    @property
    def w2(self):
        return self.feed_forward.w2

    @property
    def w3(self):
        return self.feed_forward.w3

    def init_depth_kv_weights(
        self,
        *,
        init_method,
        d_model: int,
        std: float,
        generator: Optional[torch.Generator] = None,
    ):
        if self.kv_proj is None:
            return
        from .transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            std = self.kv_proj.in_features**-0.5
        elif init_method == InitMethod.normalized:
            std = d_model**-0.5
        init_linear(self.kv_proj, std=std, generator=generator)

    def forward(
        self,
        x: torch.Tensor,
        *,
        buf_k: torch.Tensor,
        buf_v: torch.Tensor,
        slot: int,
        max_depth: int,
        k_norm_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kv_proj is not None and slot + 1 < max_depth:
            B, T, _ = x.shape
            k, v = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim).unbind(dim=2)
            if self.moda.cache_post_norm_k and k_norm_fn is not None:
                k = k_norm_fn(k)
            buf_k, buf_v = _write_moda_depth_slot(buf_k, buf_v, k, v, slot, max_depth)
        return self.feed_forward(x), buf_k, buf_v

    def apply_tp(self, *args, **kwargs):
        raise NotImplementedError("Tensor parallelism is not implemented for MoDA FFN")

    def num_flops_per_token(self, seq_len: int) -> int:
        flops = self.feed_forward.num_flops_per_token(seq_len)
        if self.kv_proj is not None:
            flops += 6 * sum(p.numel() for p in self.kv_proj.parameters())
        return flops
