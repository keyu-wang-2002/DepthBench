import math

import pytest
import torch
import torch.nn as nn

import olmo_core.nn.hyper_connections as hyper_connections
from olmo_core.nn.hyper_connections import (
    HyperConnection,
    HyperConnectionStreamExpand,
    HyperConnectionStreamReduce,
    LigerHyperConnection,
    sinkhorn_log,
)


@pytest.fixture
def liger_init_only(monkeypatch):
    # Initializers and state-dict loading must be testable without CUDA or Liger.
    monkeypatch.setattr(hyper_connections, "_require_liger_mhc", lambda: (None, None, None))
    monkeypatch.setattr(hyper_connections, "_check_liger_mhc_runtime", lambda _: None)


@pytest.mark.parametrize("kind", ["mhc", "liger_mhc"])
@pytest.mark.parametrize("num_streams", [1, 2, 4, 8])
@pytest.mark.parametrize("layer_index", [0, 1, 5])
def test_mhc_gap8_initialization(kind, num_streams, layer_index, liger_init_only):
    kwargs = dict(num_residual_streams=num_streams, dim=8, layer_index=layer_index)
    connector = (
        LigerHyperConnection(**kwargs)
        if kind == "liger_mhc"
        else HyperConnection(kind=kind, tanh=False, **kwargs)
    )
    # Verify both construction and reinitialization after arbitrary prior weights.
    for reset in (False, True):
        if reset:
            with torch.no_grad():
                for parameter in connector.parameters():
                    parameter.fill_(0.25)
            connector.reset_parameters()
        if kind == "liger_mhc":
            assert connector.phi.dtype == torch.bfloat16
            assert connector.b.dtype == torch.float32
            assert torch.count_nonzero(connector.phi) == 0
            pre = connector.b[:num_streams]
            post = connector.b[num_streams : 2 * num_streams]
            residual = connector.b[2 * num_streams :].view(num_streams, num_streams)
            gates = (connector.alpha_pre, connector.alpha_post, connector.alpha_res)
        else:
            for projection in (
                connector.pre_dynamic_proj,
                connector.post_dynamic_proj,
                connector.residual_dynamic_proj,
            ):
                assert torch.count_nonzero(projection) == 0
            pre, post, residual = connector.pre_bias, connector.post_bias, connector.residual_bias
            gates = (connector.pre_gate, connector.post_gate, connector.residual_gate)

        expected_pre = torch.full((num_streams,), -8.0)
        expected_pre[layer_index % num_streams] = 8.0
        identity = torch.eye(num_streams)
        torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
        torch.testing.assert_close(post, torch.zeros_like(post), rtol=0, atol=0)
        torch.testing.assert_close(residual, -8.0 * (1 - identity), rtol=0, atol=0)
        for gate in gates:
            torch.testing.assert_close(gate, gate.new_tensor(0.01), rtol=0, atol=0)
        expected_residual = (identity + math.exp(-8) * (1 - identity)) / (
            1 + (num_streams - 1) * math.exp(-8)
        )
        torch.testing.assert_close(sinkhorn_log(residual), expected_residual)


def test_liger_mhc_checkpoint_parameters_override_initialization(liger_init_only):
    connector = LigerHyperConnection(num_residual_streams=4, dim=8, layer_index=1)
    # Loading a complete old/trained checkpoint must not reapply the initializer.
    state = {name: torch.full_like(value, 0.25) for name, value in connector.state_dict().items()}
    assert set(state) == {"phi", "b", "alpha_pre", "alpha_post", "alpha_res"}
    restored = LigerHyperConnection(num_residual_streams=4, dim=8, layer_index=3)
    restored.load_state_dict(state, strict=True)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


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
        connector.pre_dynamic_proj.copy_(torch.arange(12, dtype=torch.float32).reshape(6, 2) / 20.0)
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
