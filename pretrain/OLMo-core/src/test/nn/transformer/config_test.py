import json

from cached_path import cached_path

from olmo_core.generate.generation_module.transformer.generation_module import (
    _set_attention_backend,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer.config import (
    MoDAConfig,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

OLMO3_7B_CHECKPOINT = "https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage1/step0"


def test_load_olmo3_7b_config():
    """Verify that old checkpoint configs with a single block (not a dict) still load correctly."""
    config_path = cached_path(f"{OLMO3_7B_CHECKPOINT}/config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = TransformerConfig.from_dict(config_dict["model"])

    assert config.d_model == 4096
    assert config.n_layers == 32
    assert config.vocab_size == 100352
    assert isinstance(config.block, TransformerBlockConfig)
    assert config.block.name == "reordered_norm"

    # Round-trip through as_config_dict / from_dict should be lossless.
    roundtripped = TransformerConfig.from_dict(config.as_config_dict())
    assert roundtripped.as_config_dict() == config.as_config_dict()


def test_roundtrip_with_moda_block_config():
    config = TransformerConfig.llama_like(
        d_model=128,
        vocab_size=32000,
        n_layers=2,
        n_heads=8,
        n_kv_heads=8,
        block_name=TransformerBlockType.post_norm_moda,
        moda=MoDAConfig(backend="v17", depth_bs=64, depth_warps=4),
    )

    roundtripped = TransformerConfig.from_dict(config.as_config_dict())

    assert roundtripped.as_config_dict() == config.as_config_dict()


def test_attention_backend_override_does_not_change_moda_backend():
    config = TransformerConfig.llama_like(
        d_model=128,
        vocab_size=32000,
        n_layers=2,
        n_heads=8,
        block_name=TransformerBlockType.moda,
        moda=MoDAConfig(backend="v17"),
    )

    _set_attention_backend(config, AttentionBackendName.torch)

    assert config.block.sequence_mixer.backend == AttentionBackendName.torch
    assert config.block.moda.backend == "v17"
