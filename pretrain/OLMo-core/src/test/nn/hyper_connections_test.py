import pytest
import torch
import torch.nn as nn

from olmo_core.nn.hyper_connections import (
    HyperConnection,
    HyperConnectionStreamExpand,
    HyperConnectionStreamReduce,
)


@pytest.mark.parametrize("kind", ["hc", "mhc", "mhc_static"])
def test_hyper_connection_folded_stream_shape_and_grads(kind: str):
    connector = HyperConnection(
        kind=kind,
        num_residual_streams=4,
        dim=16,
        branch=nn.Linear(16, 16, bias=False),
        sinkhorn_iters=4,
    )
    x = torch.randn(8, 5, 16, requires_grad=True)

    output = connector(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in connector.parameters())


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_stream_expand_reduce(mode: str):
    x = torch.randn(2, 5, 16)
    expanded = HyperConnectionStreamExpand(4)(x)
    reduced = HyperConnectionStreamReduce(4, mode=mode)(expanded)

    assert expanded.shape == (8, 5, 16)
    torch.testing.assert_close(reduced, x if mode == "mean" else 4 * x)
