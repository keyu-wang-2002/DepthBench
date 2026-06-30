import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.ops import attach_auxiliary_loss

from ..attention import Attention
from ..attention.base import SequenceMixerConfig
from ..attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForward, FeedForwardConfig
from ..functional import l2_normalize
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig, MoERouter
from ..moe.parallel_mlp import ParallelMLPBase
from ..residual_stream import ResidualStream
from .config import HyperConnectionsConfig, MoDAConfig, TransformerDataParallelWrappingStrategy

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType


class TransformerBlockBase(nn.Module):
    """
    Base class for transformer block implementations.
    """

    def __init__(self, *, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self._uses_hyper_connections = False
        self.hyper_connection_num_streams = 1
        self.hyper_connection_scale_output_init = False

    @property
    def is_moe(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the block on the input ``x``.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        """
        raise NotImplementedError

    def apply_pp(self, pp_mesh: DeviceMesh):
        del pp_mesh

    @abstractmethod
    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        raise NotImplementedError

    def apply_compile(self):
        self.compile(fullgraph=False)

    @abstractmethod
    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def num_flops_per_token(self, seq_len: int) -> int:
        raise NotImplementedError


class PreNormAttentionBranch(nn.Module):
    def __init__(self, norm: nn.Module, attention: nn.Module):
        super().__init__()
        self.norm = norm
        self.attention = attention

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.attention(self.norm(x), **kwargs)


class PreNormFeedForwardBranch(nn.Module):
    def __init__(self, norm: nn.Module, feed_forward: nn.Module):
        super().__init__()
        self.norm = norm
        self.feed_forward = feed_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.norm(x))


def _load_parallel_moda(backend: str):
    try:
        if backend == "v17":
            from fla.ops.moda import parallel_moda_v17 as parallel_moda
        else:
            from fla.ops.moda import parallel_moda
    except Exception as exc:
        raise ImportError(
            "MoDA blocks require the official MoDA Triton kernels. Install them with "
            "`pip install -e /lustre/fast/fast/wliu/yy/DepthBench_workspace/MoDA/libs/moda_triton` "
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

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.attention.w_q(x)
        k = self.attention.w_k(x)
        v = self.attention.w_v(x)

        if self.attention.clip_qkv is not None:
            q.clamp_(min=-self.attention.clip_qkv, max=self.attention.clip_qkv)
            k.clamp_(min=-self.attention.clip_qkv, max=self.attention.clip_qkv)
            v.clamp_(min=-self.attention.clip_qkv, max=self.attention.clip_qkv)

        if not self.attention.use_head_qk_norm:
            if self.attention.q_norm is not None:
                q = self.attention.q_norm(q)
            if self.attention.k_norm is not None:
                k = self.attention.k_norm(k)

        q = q.view(B, T, -1, self.head_dim)
        k = k.view(B, T, -1, self.head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.attention.use_head_qk_norm:
            if self.attention.q_norm is not None:
                q = self.attention.q_norm(q)
            if self.attention.k_norm is not None:
                k = self.attention.k_norm(k)

        return q, k, v

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
        B, T, _, h_kv, d = buf_k.shape
        cached_k = buf_k[:, :, :current_depth].reshape(B, T * current_depth, h_kv, d)
        cached_v = buf_v[:, :, :current_depth].reshape(B, T * current_depth, h_kv, d)
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
            q.reshape(B, T, h_kv, group, d)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, T * group, h_kv, d)
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
        return (
            y.view(B, T, group, h_kv, d)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, T, h_q, d)
        )

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
        q, k, v = self._project_qkv(x)

        if self.moda.cache_post_norm_k and self.attention.k_norm is not None:
            k_for_cache = k
        else:
            k_for_cache = self.attention.w_k(x).view(B, T, -1, self.head_dim)
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
        from .init import InitMethod, init_linear

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


class MoDABlockMixin:
    d_model: int
    block_idx: int
    n_layers: int
    max_depth: int
    attention: MoDAAttention

    def _unpack_moda_state(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            return x
        B, T, _ = x.shape
        buf_shape = (B, T, self.max_depth, self.attention.n_kv_heads, self.attention.head_dim)
        buf_k = torch.zeros(buf_shape, dtype=x.dtype, device=x.device)
        buf_v = torch.zeros_like(buf_k)
        return x, buf_k, buf_v

    def _pack_moda_state(
        self, x: torch.Tensor, buf_k: torch.Tensor, buf_v: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.block_idx == self.n_layers - 1:
            return x
        return x, buf_k, buf_v

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        del tp_mesh, input_layout, float8_enabled
        raise NotImplementedError("Tensor/sequence parallelism is not implemented for MoDA blocks")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not implemented for MoDA blocks")

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        del prefetch_factor, wrapping_strategy
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


class TransformerBlock(TransformerBlockBase):
    """
    A typical "Llama-style" transformer block implementation.

    :param d_model: The model dimensionality.
    :param block_idx: The index/position of the block within the model. Ranges from 0 to ``n_layers - 1``.
    :param sequence_mixer: The sequence mixer module config (e.g. attention, recurrent, convolution, etc.).
    :param feed_forward: The feed forward module config.
    :param layer_norm: The layer norm config for both the attention LN and the feed forward LN.
    :param dropout: Dropout probability.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        hyper_connections: Optional[HyperConnectionsConfig] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx

        # NOTE: The `self.attention` naming is kept for backwards compatibility with old checkpoints.
        # `self.attention` could contain any `SequenceMixer` implementation, such as a `GatedDeltaNet`.
        # Generally it's ok to think of these as "attention" modules at the block level.
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)

        if hyper_connections is None:
            self.attention_residual_stream = ResidualStream(
                alpha=attention_residual_alpha, dropout=dropout
            )
            self.feed_forward_residual_stream = ResidualStream(
                alpha=feed_forward_residual_alpha, dropout=dropout
            )
            self.attention_hyper_connection = None
            self.feed_forward_hyper_connection = None
        else:
            connector_dtype = next(self.attention.parameters()).dtype
            self.attention_hyper_connection = hyper_connections.build(
                dim=d_model,
                branch=PreNormAttentionBranch(self.attention_norm, self.attention),
                layer_index=block_idx * 2,
                init_device=init_device,
                dtype=connector_dtype,
            )
            self.feed_forward_hyper_connection = hyper_connections.build(
                dim=d_model,
                branch=PreNormFeedForwardBranch(self.feed_forward_norm, self.feed_forward),
                layer_index=block_idx * 2 + 1,
                init_device=init_device,
                dtype=connector_dtype,
            )
            self.attention_residual_stream = None
            self.feed_forward_residual_stream = None
            self._uses_hyper_connections = True
            self.hyper_connection_num_streams = hyper_connections.num_residual_streams
            self.hyper_connection_scale_output_init = hyper_connections.scale_output_init_by_sqrt_n
            self.hyper_connection_reduce_mode = hyper_connections.reduce_mode

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        if self._uses_hyper_connections:
            assert self.attention_hyper_connection is not None
            assert self.feed_forward_hyper_connection is not None
            h = self.attention_hyper_connection(x, **kwargs)
            return self.feed_forward_hyper_connection(h)

        assert self.attention_residual_stream is not None
        assert self.feed_forward_residual_stream is not None
        h = self.attention_residual_stream(x, self.attention(self.attention_norm(x), **kwargs))
        return self.feed_forward_residual_stream(h, self.feed_forward(self.feed_forward_norm(h)))

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        if self._uses_hyper_connections:
            raise NotImplementedError("Tensor parallelism is not implemented for HC / mHC blocks")

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )
        parallelize_module(
            self.attention_residual_stream.dropout,
            device_mesh=tp_mesh,
            parallelize_plan=SequenceParallel(),
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )
        parallelize_module(
            self.feed_forward_residual_stream.dropout,
            device_mesh=tp_mesh,
            parallelize_plan=SequenceParallel(),
        )

        self.feed_forward.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_mlp = cast(FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        attn_flops = self.attention.num_flops_per_token(seq_len)
        ff_flops = self.feed_forward.num_flops_per_token(seq_len)
        return attn_flops + ff_flops


class PostNormTransformerBlock(TransformerBlock):
    """
    Dense post-norm baseline block.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        hyper_connections: Optional[HyperConnectionsConfig] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        if hyper_connections is not None:
            raise NotImplementedError("PostNormTransformerBlock does not support HC / mHC")
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            layer_norm=layer_norm,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
            hyper_connections=None,
            init_device=init_device,
            cache=cache,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        assert self.attention_residual_stream is not None
        assert self.feed_forward_residual_stream is not None
        h = self.attention_norm(
            self.attention_residual_stream(x, self.attention(x, **kwargs))
        )
        return self.feed_forward_norm(
            self.feed_forward_residual_stream(h, self.feed_forward(h))
        )


class MoDATransformerBlock(MoDABlockMixin, TransformerBlockBase):
    """
    Pre-norm Llama-style block with MoDA replacing the attention kernel.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        moda: MoDAConfig,
        skip_ffn_kv: bool = False,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        hyper_connections: Optional[HyperConnectionsConfig] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        if hyper_connections is not None:
            raise NotImplementedError("MoDA blocks do not support HC / mHC residual routing")
        if layer_norm is None:
            raise OLMoConfigurationError("MoDA blocks require layer_norm")
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx
        self.max_depth = 2 * n_layers

        attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        if not isinstance(attention, Attention):
            raise OLMoConfigurationError("MoDA requires AttentionConfig(name='default')")
        self.attention = MoDAAttention(
            attention, moda=moda, block_idx=block_idx, n_layers=n_layers
        )
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        base_feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward = MoDAFeedForwardDepthCache(
            base_feed_forward,
            d_model=d_model,
            n_kv_heads=self.attention.n_kv_heads,
            head_dim=self.attention.head_dim,
            bias=bool(getattr(attention.w_k, "bias", None) is not None),
            moda=moda,
            skip_kv_proj=skip_ffn_kv,
            init_device=init_device,
            dtype=next(base_feed_forward.parameters()).dtype,
        )
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.attention_residual_stream = ResidualStream(
            alpha=attention_residual_alpha, dropout=dropout
        )
        self.feed_forward_residual_stream = ResidualStream(
            alpha=feed_forward_residual_alpha, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del loss_div_factor
        x, buf_k, buf_v = self._unpack_moda_state(x)
        attn_slot = 2 * self.block_idx
        attn_out, buf_k, buf_v = self.attention(
            self.attention_norm(x),
            buf_k=buf_k,
            buf_v=buf_v,
            current_depth=attn_slot,
            slot=attn_slot,
            max_depth=self.max_depth,
            **kwargs,
        )
        h = self.attention_residual_stream(x, attn_out)
        mlp_slot = 2 * self.block_idx + 1
        ffn_in = self.feed_forward_norm(h)
        ffn_out, buf_k, buf_v = self.feed_forward(
            ffn_in,
            buf_k=buf_k,
            buf_v=buf_v,
            slot=mlp_slot,
            max_depth=self.max_depth,
            k_norm_fn=self.attention.normalize_cache_k,
        )
        out = self.feed_forward_residual_stream(h, ffn_out)
        return self._pack_moda_state(out, buf_k, buf_v)

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.attention.num_flops_per_token(seq_len) + self.feed_forward.num_flops_per_token(
            seq_len
        )


class PostNormMoDATransformerBlock(MoDATransformerBlock):
    """
    Paper-aligned post-norm MoDA block.
    """

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del loss_div_factor
        x, buf_k, buf_v = self._unpack_moda_state(x)
        attn_slot = 2 * self.block_idx
        attn_out, buf_k, buf_v = self.attention(
            x,
            buf_k=buf_k,
            buf_v=buf_v,
            current_depth=attn_slot,
            slot=attn_slot,
            max_depth=self.max_depth,
            **kwargs,
        )
        h = self.attention_norm(self.attention_residual_stream(x, attn_out))
        mlp_slot = 2 * self.block_idx + 1
        ffn_out, buf_k, buf_v = self.feed_forward(
            h,
            buf_k=buf_k,
            buf_v=buf_v,
            slot=mlp_slot,
            max_depth=self.max_depth,
            k_norm_fn=self.attention.normalize_cache_k,
        )
        out = self.feed_forward_norm(self.feed_forward_residual_stream(h, ffn_out))
        return self._pack_moda_state(out, buf_k, buf_v)


class LayerNormScaledTransformerBlock(TransformerBlock):
    """
    A variant of ``TransformerBlock`` that applies
    `LayerNorm Scaling (LNS) <https://github.com/lmsdss/LayerNorm-Scaling>`_.

    Each LayerNorm output is multiplied by ``1 / sqrt(layer_id)`` where ``layer_id`` is the
    1-based position of the block inside the transformer. Keeping this logic in a dedicated
    subclass ensures that the vanilla ``TransformerBlock`` remains simple and easy to reason
    about.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        hyper_connections: Optional[HyperConnectionsConfig] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        if hyper_connections is not None:
            raise NotImplementedError(
                "LayerNormScaledTransformerBlock does not support HC / mHC residual routing"
            )
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            layer_norm=layer_norm,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
            hyper_connections=hyper_connections,
            init_device=init_device,
            cache=cache,
        )

        # LayerNorm scaling factor 1/sqrt(layer_id), where layer_id is 1-based.
        ln_scale_value = 1.0 / math.sqrt(block_idx + 1)
        self.register_buffer("ln_scale", torch.tensor(ln_scale_value, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        if self._uses_hyper_connections:
            raise NotImplementedError(
                "LayerNormScaledTransformerBlock does not support HC / mHC residual routing"
            )
        scale = self.ln_scale.to(dtype=x.dtype, device=x.device)
        h = self.attention_residual_stream(
            x, self.attention(self.attention_norm(x) * scale, **kwargs)
        )
        return self.feed_forward_residual_stream(
            h, self.feed_forward(self.feed_forward_norm(h) * scale)
        )


class ReorderedNormTransformerBlock(TransformerBlock):
    """
    Like :class:`TransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the output
    of the feed-forward instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        if self._uses_hyper_connections:
            raise NotImplementedError(
                "ReorderedNormTransformerBlock does not support HC / mHC residual routing"
            )
        h = self.attention_residual_stream(x, self.attention_norm(self.attention(x, **kwargs)))
        return self.feed_forward_residual_stream(h, self.feed_forward_norm(self.feed_forward(h)))


class PeriNormTransformerBlock(TransformerBlock):
    """
    A transformer block in the style of `Peri-LN <https://arxiv.org/pdf/2502.02732>`_.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        hyper_connections: Optional[HyperConnectionsConfig] = None,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        if hyper_connections is not None:
            raise NotImplementedError(
                "PeriNormTransformerBlock does not support HC / mHC residual routing"
            )
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            layer_norm=layer_norm,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
            hyper_connections=hyper_connections,
            init_device=init_device,
            cache=cache,
        )
        self.post_attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.post_feed_forward_norm = layer_norm.build(d_model, init_device=init_device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        if self._uses_hyper_connections:
            raise NotImplementedError(
                "PeriNormTransformerBlock does not support HC / mHC residual routing"
            )
        h = self.attention_residual_stream(
            x, self.post_attention_norm(self.attention(self.attention_norm(x), **kwargs))
        )
        return self.feed_forward_residual_stream(
            h, self.post_feed_forward_norm(self.feed_forward(self.feed_forward_norm(h)))
        )

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        super().apply_tp(tp_mesh, input_layout=input_layout, float8_enabled=float8_enabled)
        parallelize_module(
            self.post_feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )
        parallelize_module(
            self.post_attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )


@beta_feature
class NormalizedTransformerBlock(TransformerBlockBase):
    """
    An nGPT block implementation to be used with the :class:`~olmo_core.nn.attention.NormalizedAttention`
    attention type and :class:`~olmo_core.nn.feed_forward.NormalizedFeedForward` feed-forward type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx

        # NOTE: The `self.attention` naming is kept for backwards compatibility with old checkpoints.
        # `self.attention` could contain any `SequenceMixer` implementation, such as a `GatedDeltaNet`.
        # Generally it's ok to think of these as "attention" modules at the block level.
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.attn_alpha = nn.Parameter(
            torch.empty(d_model, dtype=torch.float32, device=init_device)
        )

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / math.sqrt(d_model)
        self.mlp_alpha = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.attn_alpha)
        nn.init.ones_(self.mlp_alpha)
        with torch.no_grad():
            self.attn_alpha.mul_(self.attn_alpha_init_scaling)
            self.mlp_alpha.mul_(self.mlp_alpha_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del loss_div_factor
        h = l2_normalize(
            torch.lerp(
                x,
                l2_normalize(self.attention(x, **kwargs)),
                (
                    self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
                ).abs(),
            )
        )

        return l2_normalize(
            torch.lerp(
                h,
                l2_normalize(self.feed_forward(h)),
                (self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)).abs(),
            )
        )

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        del tp_mesh, input_layout, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer block variant"
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
            fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)

        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

        if (
            wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained
            and prefetch_factor > 0
        ):
            cast(FSDPModule, self).set_modules_to_forward_prefetch(
                [cast(FSDPModule, self.attention)]
            )
            cast(FSDPModule, self.attention).set_modules_to_forward_prefetch(
                [cast(FSDPModule, self.feed_forward)]
            )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if hasattr(self.attention, "normalize_matrices"):
            self.attention.normalize_matrices()  # type: ignore

        if hasattr(self.feed_forward, "normalize_matrices"):
            self.feed_forward.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def num_flops_per_token(self, seq_len: int) -> int:
        attn_flops = self.attention.num_flops_per_token(seq_len)
        ff_flops = self.feed_forward.num_flops_per_token(seq_len)
        return attn_flops + ff_flops


@beta_feature
class MoETransformerBlock(TransformerBlockBase):
    """
    Like :class:`TransformerBlock` except that the dense :class:`~olmo_core.nn.feed_forward.FeedForward`
    module is replaced with a mixture-of-experts (MoE).
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward_moe: MoEConfig,
        layer_norm: LayerNormConfig,
        dropout: float = 0.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(n_layers=n_layers)
        self.d_model = d_model
        self.block_idx = block_idx

        # NOTE: The `self.attention` naming is kept for backwards compatibility with old checkpoints.
        # `self.attention` could contain any `SequenceMixer` implementation, such as a `GatedDeltaNet`.
        # Generally it's ok to think of these as "attention" modules at the block level.
        self.attention = sequence_mixer.build(
            d_model, layer_idx=block_idx, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.attention_norm = layer_norm.build(d_model, init_device=init_device)
        self.feed_forward_moe = feed_forward_moe.build(
            d_model=d_model, n_layers=n_layers, init_device=init_device, cache=cache
        )
        self.feed_forward_norm = layer_norm.build(d_model, init_device=init_device)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._ep_enabled = False
        self._tp_enabled = False

    @property
    def is_moe(self) -> bool:
        return True

    @property
    def router(self) -> MoERouter:
        return self.feed_forward_moe.router

    @property
    def shared_mlp(self) -> Optional[FeedForward]:
        return self.feed_forward_moe.shared_mlp

    @property
    def experts(self) -> ParallelMLPBase:
        return self.feed_forward_moe.experts

    @property
    def top_k(self) -> int:
        return self.feed_forward_moe.top_k

    @property
    def ep_enabled(self) -> bool:
        return self._ep_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        return self.feed_forward_moe.compute_metrics(reset=reset)

    def reset_metrics(self):
        self.feed_forward_moe.reset_metrics()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(
            self.feed_forward_moe(self.feed_forward_norm(h), loss_div_factor=loss_div_factor)
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        self.feed_forward_moe.apply_pp(pp_mesh)

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        self.feed_forward_moe.apply_ep(ep_mesh, **kwargs)
        self._ep_enabled = True

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        parallelize_module(
            self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.attention.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

        self.feed_forward_moe.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())

        self._tp_enabled = True

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        self.attention.apply_cp(cp_mesh, ring=ring, uly=uly)
        self.feed_forward_moe.apply_cp(cp_mesh)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_moe = cast(
                FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def num_flops_per_token(self, seq_len: int) -> int:
        attn_flops = self.attention.num_flops_per_token(seq_len)
        moe_flops = self.feed_forward_moe.num_flops_per_token(seq_len)
        return attn_flops + moe_flops


@beta_feature
class MoEReorderedNormTransformerBlock(MoETransformerBlock):
    """
    Like :class:`MoETransformerBlock` except that the attention norm is applied on the output
    of attention instead of the input, and likewise the feed-forward norm is applied on the
    output of the feed-forward MoE instead of the input.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(
            self.feed_forward_norm(self.feed_forward_moe(h, loss_div_factor=loss_div_factor))
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            fsdp_moe = cast(
                FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
            )
            fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_moe])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEHybridTransformerBlockBase(MoETransformerBlock):
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        layer_norm: LayerNormConfig,
        feed_forward: FeedForwardConfig,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            layer_norm=layer_norm,
            init_device=init_device,
            **kwargs,
        )
        self.feed_forward = feed_forward.build(d_model=d_model, init_device=init_device)
        self.feed_forward_moe_norm = layer_norm.build(d_model, init_device=init_device)
        self._use_combined_forward: Optional[bool] = None

    @property
    def use_combined_forward(self) -> bool:
        if self._use_combined_forward is not None:
            return self._use_combined_forward
        elif not self.ep_enabled and not self.tp_enabled:
            return False
        else:
            return True

    @use_combined_forward.setter
    def use_combined_forward(self, should_use: bool):
        if should_use and not (self.tp_enabled or self.ep_enabled):
            raise RuntimeError(
                "combined forward can only be used when expert parallelism is enabled"
            )
        self._use_combined_forward = should_use

    @abstractmethod
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.use_combined_forward:
            return self.sparse_forward(x, loss_div_factor=loss_div_factor) + self.dense_forward(
                x, **kwargs
            )
        else:
            # NOTE: alternatively could do something like this, but even with an extra stream it's
            # not as fast as the hand-crafted 'combined_forward()'.
            # stream = get_or_init_stream()
            # stream.wait_stream(torch.cuda.default_stream())
            # h_sparse = self._fwd_sparse(x)
            # with torch.cuda.stream(stream):
            #     h_dense = self._fwd_dense(x, **kwargs)
            # torch.cuda.default_stream().wait_stream(stream)
            # return h_sparse + h_dense
            return self.combined_forward(x, loss_div_factor=loss_div_factor, **kwargs)

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        super().apply_tp(tp_mesh, input_layout=input_layout, float8_enabled=float8_enabled)

        self.feed_forward.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=float8_enabled,
        )

        parallelize_module(
            self.feed_forward_moe_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        from torch.distributed.fsdp import MixedPrecisionPolicy

        # Force router to be full-precision.
        fsdp_router = cast(
            FSDPModule,
            fully_shard(
                self.feed_forward_moe.router,
                mesh=dp_mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32),
            ),
        )

        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            if not self.use_combined_forward:
                fsdp_att = cast(
                    FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_mlp = cast(
                    FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_moe = cast(
                    FSDPModule, fully_shard(self.feed_forward_moe, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))
                if prefetch_factor > 0:
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_router, fsdp_moe, fsdp_att])
                    fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
            else:
                fsdp_att = cast(
                    FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
                )
                fsdp_mlp = cast(
                    FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
                )
                #  fsdp_moe = cast(
                #      FSDPModule,
                #      fully_shard(self.feed_forward_moe.experts.mlp, mesh=dp_mesh, **fsdp_kwargs),
                #  )
                fsdp_shared_mlp = (
                    None
                    if self.feed_forward_moe.shared_mlp is None
                    else cast(
                        FSDPModule,
                        fully_shard(self.feed_forward_moe.shared_mlp, mesh=dp_mesh, **fsdp_kwargs),
                    )
                )
                fsdp_root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **fsdp_kwargs))

                if prefetch_factor > 0:
                    #  fsdp_root.set_modules_to_forward_prefetch([fsdp_att, fsdp_moe])
                    fsdp_root.set_modules_to_forward_prefetch([fsdp_att, fsdp_router])
                    if fsdp_shared_mlp is not None:
                        fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp, fsdp_shared_mlp])
                    else:
                        fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


@beta_feature
class MoEHybridTransformerBlock(MoEHybridTransformerBlockBase):
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))
        return h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.dropout(
            self.feed_forward_moe(self.feed_forward_moe_norm(x), loss_div_factor=loss_div_factor)
        )

    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: this follows the same code path as the MoE's forward pass, except that we run
        # dense operations while we wait on expert parallel all-to-all comms.
        B, _, D = x.shape

        x_moe = get_local_tensor(self.feed_forward_moe_norm(x))

        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = self.router(
            x_moe, loss_div_factor=loss_div_factor
        )

        if router_aux_loss is not None:
            x_moe = attach_auxiliary_loss(x_moe, router_aux_loss)

        # shape: (batch_size * seq_len, d_model)
        x_moe = x_moe.view(-1, D)
        # shape: (batch_size * top_k,)
        expert_weights = expert_weights.flatten()
        # shape: (batch_size * top_k,)
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.experts.indices_and_bins(
                expert_indices, batch_size_per_expert
            )

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            handle,
        ) = self.experts.permute_and_all_to_all(
            x_moe,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        # Compute attention while all-to-all is in progress.
        h = x + self.dropout(self.attention(self.attention_norm(x), **kwargs))

        # Maybe compute MoE shared out while all-to-all is in progress.
        moe_shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            # NOTE: -1 on seq dim in case of TP
            moe_shared_out = self.shared_mlp(x_moe.view(B, -1, D))

        handle.wait()
        parallel_x = self.experts.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x_moe, handle = self.experts.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        # Compute feed-forward while all-to-all is in progress.
        h = h + self.dropout(self.feed_forward(self.feed_forward_norm(h)))

        handle.wait()
        x_moe = self.experts.unpermute(
            x_moe,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        ).view(B, -1, D)

        if moe_shared_out is not None:
            moe_shared_out = moe_shared_out / (self.top_k + 1)
            x_moe = moe_shared_out.add(x_moe, alpha=self.top_k / (self.top_k + 1))

        return h + self.dropout(x_moe)


@beta_feature
class MoEHybridReorderedNormTransformerBlock(MoEHybridTransformerBlockBase):
    def dense_forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))
        return h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))

    def sparse_forward(
        self, x: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.dropout(
            self.feed_forward_moe_norm(self.feed_forward_moe(x, loss_div_factor=loss_div_factor))
        )

    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: this follows the same code path as the MoE's forward pass, except that we run
        # dense operations while we wait on expert parallel all-to-all comms.
        B, _, D = x.shape

        x_moe = get_local_tensor(x)

        expert_weights, expert_indices, batch_size_per_expert, router_aux_loss = self.router(
            x_moe, loss_div_factor=loss_div_factor
        )

        if router_aux_loss is not None:
            x_moe = attach_auxiliary_loss(x_moe, router_aux_loss)

        # shape: (batch_size * seq_len, d_model)
        x_moe = x_moe.view(-1, D)
        # shape: (batch_size * seq_len * top_k,)
        expert_weights = get_local_tensor(expert_weights).flatten()
        # shape: (batch_size * seq_len * top_k,)
        expert_indices = get_local_tensor(expert_indices).flatten()

        with torch.no_grad():
            indices, bin_ids, bins = self.experts.indices_and_bins(
                expert_indices, batch_size_per_expert
            )

        (
            parallel_x,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            parallel_batch_size_per_expert,
            recv_counts,
            send_counts,
            expert_capacity,
            handle,
        ) = self.experts.permute_and_all_to_all(
            x_moe,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
            batch_size_per_expert=batch_size_per_expert,
        )

        # Compute attention while all-to-all is in progress.
        h = x + self.dropout(self.attention_norm(self.attention(x, **kwargs)))

        # Maybe compute MoE shared out while all-to-all is in progress.
        moe_shared_out: Optional[torch.Tensor] = None
        if self.shared_mlp is not None:
            # NOTE: -1 on seq dim in case of TP
            moe_shared_out = self.shared_mlp(x_moe.view(B, -1, D))

        handle.wait()
        parallel_x = self.experts.compute_local_experts(
            parallel_x,
            parallel_indices=parallel_indices,
            parallel_bin_ids=parallel_bin_ids,
            parallel_bins=parallel_bins,
            parallel_batch_size_per_expert=parallel_batch_size_per_expert,
            expert_capacity=expert_capacity,
        )

        x_moe, handle = self.experts.reverse_all_to_all(
            parallel_x, send_counts=send_counts, recv_counts=recv_counts
        )

        # Compute feed-forward while all-to-all is in progress.
        h = h + self.dropout(self.feed_forward_norm(self.feed_forward(h)))

        handle.wait()
        x_moe = self.experts.unpermute(
            x_moe,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            indices=indices,
            bin_ids=bin_ids,
            bins=bins,
        ).view(B, -1, D)

        if moe_shared_out is not None:
            moe_shared_out = moe_shared_out / (self.top_k + 1)
            x_moe = moe_shared_out.add(x_moe, alpha=self.top_k / (self.top_k + 1))

        return h + self.dropout(self.feed_forward_moe_norm(x_moe))
