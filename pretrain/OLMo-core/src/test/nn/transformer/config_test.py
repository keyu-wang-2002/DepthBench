import json

from cached_path import cached_path

from olmo_core.nn.transformer.config import (
    HyperConnectionsConfig,
    HyperConnectionsKind,
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


def test_roundtrip_with_hyper_connections():
    config = TransformerConfig.llama_like(
        d_model=128,
        vocab_size=32000,
        n_layers=2,
        n_heads=8,
        block_name=TransformerBlockType.mhc,
    )
    config.block.hyper_connections = HyperConnectionsConfig(
        kind="liger_mhc",
        num_residual_streams=4,
        gating_factor_init=0.01,
        sinkhorn_iters=20,
        liger_phi_dtype="bfloat16",
    )

    roundtripped = TransformerConfig.from_dict(config.as_config_dict())

    assert roundtripped.as_config_dict() == config.as_config_dict()


def test_hyper_connection_activation_defaults_match_paper_parameterizations():
    assert HyperConnectionsConfig(kind="hc").use_tanh
    assert not HyperConnectionsConfig(kind="mhc").use_tanh
    assert not HyperConnectionsConfig(kind="liger_mhc").use_tanh
    assert HyperConnectionsConfig(kind="mhc", tanh=True).use_tanh


def test_legacy_default_block_with_hyper_connections_is_migrated():
    config = TransformerConfig.llama_like(
        d_model=128,
        vocab_size=32000,
        n_layers=2,
        n_heads=8,
    )
    config_dict = config.as_config_dict()
    config_dict["block"]["hyper_connections"] = HyperConnectionsConfig(
        kind="liger_mhc"
    ).as_config_dict()

    migrated = TransformerConfig.from_dict(config_dict)

    assert migrated.block.name == TransformerBlockType.mhc
    assert migrated.block.hyper_connections.kind == HyperConnectionsKind.liger_mhc
