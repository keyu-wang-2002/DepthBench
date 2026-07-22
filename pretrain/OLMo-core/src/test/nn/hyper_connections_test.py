import pytest
import torch
import torch.nn as nn

from olmo_core.nn.hyper_connections import (
    HyperConnection,
    HyperConnectionStreamExpand,
    HyperConnectionStreamReduce,
    sinkhorn_log,
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


def test_hc_matches_paper_width_and_depth_connection_equations():
    connector = HyperConnection(
        kind="hc",
        num_residual_streams=2,
        dim=3,
        branch=nn.Linear(3, 3, bias=False),
    )
    with torch.no_grad():
        connector.branch.weight.copy_(2.0 * torch.eye(3))
        connector.static_alpha.copy_(
            torch.tensor(
                [
                    [1.5, 0.2, -0.3],
                    [-0.5, 0.7, 1.1],
                ]
            )
        )
        connector.static_beta.copy_(torch.tensor([0.4, -1.2]))
        connector.dynamic_alpha_proj.zero_()
        connector.dynamic_beta_proj.zero_()

    streams = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
                [[2.0, -3.0, 1.0], [0.5, 2.5, -2.0]],
            ]
        ]
    )
    folded = streams.movedim(-2, 1).reshape(2, 2, 3)

    output = connector(folded)

    mixed = torch.einsum("...st,...sd->...td", connector.static_alpha, streams)
    branch_output = 2.0 * mixed[..., 0, :]
    expected_streams = mixed[..., 1:, :] + branch_output.unsqueeze(-2) * connector.static_beta.view(
        1, 1, 2, 1
    )
    expected = expected_streams.movedim(-2, 1).reshape_as(output)
    torch.testing.assert_close(output, expected)


def test_mhc_matches_paper_parameterization_and_propagation_equations():
    connector = HyperConnection(
        kind="mhc",
        num_residual_streams=2,
        dim=3,
        branch=nn.Linear(3, 3, bias=False),
        tanh=False,
        sinkhorn_iters=7,
        sinkhorn_tau=0.7,
    )
    with torch.no_grad():
        connector.branch.weight.copy_(1.5 * torch.eye(3))
        connector.pre_dynamic_proj.copy_(
            torch.arange(12, dtype=torch.float32).reshape(6, 2) / 20.0
        )
        connector.post_dynamic_proj.copy_(
            torch.arange(12, dtype=torch.float32).reshape(6, 2).flip(0) / 25.0
        )
        connector.residual_dynamic_proj.copy_(
            torch.arange(24, dtype=torch.float32).reshape(6, 4) / 30.0
        )
        connector.pre_bias.copy_(torch.tensor([-0.4, 0.2]))
        connector.post_bias.copy_(torch.tensor([0.1, -0.3]))
        connector.residual_bias.copy_(torch.tensor([[0.5, -0.2], [0.1, 0.3]]))
        connector.pre_gate.fill_(0.3)
        connector.post_gate.fill_(0.2)
        connector.residual_gate.fill_(0.4)

    streams = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
                [[2.0, -3.0, 1.0], [0.5, 2.5, -2.0]],
            ]
        ]
    )
    folded = streams.movedim(-2, 1).reshape(2, 2, 3)

    output = connector(folded)

    flat = streams.reshape(1, 2, -1)
    normed = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + 1e-8)
    pre_logits = 0.3 * (normed @ connector.pre_dynamic_proj) + connector.pre_bias
    post_logits = 0.2 * (normed @ connector.post_dynamic_proj) + connector.post_bias
    residual_logits = (
        0.4 * (normed @ connector.residual_dynamic_proj).reshape(1, 2, 2, 2)
        + connector.residual_bias
    )
    h_pre = torch.sigmoid(pre_logits)
    h_post = 2.0 * torch.sigmoid(post_logits)
    h_res = sinkhorn_log(residual_logits, num_iters=7, tau=0.7)
    branch_input = torch.einsum("...s,...sd->...d", h_pre, streams)
    branch_output = 1.5 * branch_input
    expected_streams = torch.einsum("...st,...sd->...td", h_res, streams)
    expected_streams = expected_streams + branch_output.unsqueeze(-2) * h_post.unsqueeze(-1)
    expected = expected_streams.movedim(-2, 1).reshape_as(output)

    torch.testing.assert_close(output, expected)
