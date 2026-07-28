"""Shared configuration helpers for MoDA training entries."""

from dataclasses import replace

from olmo_core.nn.transformer import (
    MoDAConfig,
    TransformerBlockConfig,
    TransformerBlockType,
)


def configure_moda(config, *, post_norm: bool) -> None:
    if not isinstance(config.model.block, TransformerBlockConfig):
        raise TypeError("MoDA examples require a single dense TransformerBlockConfig")
    if config.model.block.hyper_connections is not None:
        raise ValueError("MoDA experiments cannot be combined with HC or mHC")

    moda = MoDAConfig(
        backend="v17",
        cache_post_norm_k=True,
        extra_ffn_kv_proj=True,
        extra_attn_kv_proj=False,
        depth_bs=64,
        depth_warps=4,
    )
    block_type = (
        TransformerBlockType.post_norm_moda if post_norm else TransformerBlockType.moda
    )
    config.model.block.name = block_type
    config.model.block.moda = moda

    last_idx = config.model.n_layers - 1
    overrides = dict(config.model.block_overrides or {})
    last_block = overrides.get(last_idx, config.model.block)
    overrides[last_idx] = replace(
        last_block,
        name=block_type,
        moda=moda,
        moda_skip_ffn_kv=True,
    )
    config.model.block_overrides = overrides
