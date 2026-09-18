import pytest
import torch

import olmo_core.nn.hyper_connections as hyper_connections
from olmo_core.config import DType
from olmo_core.nn.transformer import HyperConnectionsConfig, TransformerBlockType, TransformerConfig


@pytest.mark.parametrize("backend", ["mhc", "liger_mhc"])
@pytest.mark.parametrize("init_device", ["cpu", "meta"])
def test_mhc_model_reset_preserves_gap8_and_sublayer_cycle(backend, init_device, monkeypatch):
    monkeypatch.setattr(hyper_connections, "_require_liger_mhc", lambda: (None, None, None))
    monkeypatch.setattr(hyper_connections, "_check_liger_mhc_runtime", lambda _: None)
    config = TransformerConfig.llama_like(
        d_model=32,
        vocab_size=64,
        n_layers=3,
        n_heads=2,
        use_flash=False,
        block_name=TransformerBlockType.mhc,
    )
    config.block.hyper_connections = HyperConnectionsConfig(kind=backend)
    config = TransformerConfig.from_dict(config.as_config_dict())
    model = config.build(init_device=init_device)
    for _ in range(2):
        model.init_weights(max_seq_len=16, device=torch.device("cpu"))
        for index, block in enumerate(model.blocks.values()):
            assert not hasattr(block, "attention_residual_stream")
            assert not hasattr(block, "feed_forward_residual_stream")
            for sub, connector in enumerate(
                (block.attention_hyper_connection, block.feed_forward_hyper_connection)
            ):
                selected = (2 * index + sub) % 4
                assert connector.selected_stream == selected
                if backend == "liger_mhc":
                    assert torch.count_nonzero(connector.phi) == 0
                    pre = connector.b[:4]
                    post = connector.b[4:8]
                    residual = connector.b[8:].view(4, 4)
                else:
                    assert torch.count_nonzero(connector.residual_dynamic_proj) == 0
                    pre, post = connector.pre_bias, connector.post_bias
                    residual = connector.residual_bias
                expected_pre = torch.full((4,), -8.0)
                expected_pre[selected] = 8.0
                torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
                torch.testing.assert_close(post, torch.zeros(4), rtol=0, atol=0)
                torch.testing.assert_close(residual, -8.0 * (1 - torch.eye(4)), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger mHC requires CUDA")
def test_liger_gap8_tiny_training_and_checkpoint_roundtrip():
    pytest.importorskip("liger_kernel")
    torch.manual_seed(42)
    config = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=2,
        n_heads=2,
        use_flash=False,
        dtype=DType.bfloat16,
        block_name=TransformerBlockType.mhc,
    )
    config.block.hyper_connections = HyperConnectionsConfig(kind="liger_mhc")
    model = config.build(init_device="meta")
    model.init_weights(max_seq_len=16, device=torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95))
    tokens = torch.randint(0, 128, (2, 16), device="cuda")
    labels = tokens.roll(-1, 1)
    head_shapes = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda _module, inputs: head_shapes.append(inputs[0].shape)
    )
    try:
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens, labels=labels).loss
            loss.backward()
            assert torch.isfinite(loss)
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
            optimizer.step()
    finally:
        hook.remove()
    assert set(head_shapes) == {torch.Size((2, 16, 64))}
    restored = config.build(init_device="meta")
    restored.init_weights(max_seq_len=16, device=torch.device("cuda"))
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
