"""Exercise the actual mHC entry and every backbone shipped in DepthBench."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import olmo_core.nn.hyper_connections as hyper_connections
from olmo_core.config import DType
from olmo_core.nn.transformer import HyperConnectionsKind, TransformerBlockType

ROOT = Path(__file__).resolve().parents[6]
MODEL_CONFIGS = sorted((ROOT / "configs").glob("llama_*"))
SHAPE_CONFIGS = sorted((ROOT / "configs").glob("llama_400m_*.json"))


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    monkeypatch.chdir(ROOT)
    import pretrain_mhc

    monkeypatch.setattr(hyper_connections, "_require_liger_mhc", lambda: (None, None, None))
    monkeypatch.setattr(hyper_connections, "_check_liger_mhc_runtime", lambda _: None)
    return pretrain_mhc


@pytest.fixture
def args(entry, tmp_path):
    data = tmp_path / "tokens.npy"
    np.save(data, np.zeros(4096, dtype=np.uint32))
    return entry.build_parser().parse_args(
        [
            f"--train-data-glob={data}",
            f"--eval-data-glob={data}",
            f"--save-folder={tmp_path / 'ckpt'}",
            "--wandb-project=",
            "--learning-rate=2e-3",
            "--global-train-batch-size=512",
            "--device-train-microbatch-size=8",
            "--sequence-length=2048",
            "--max-steps=7600",
            "--warmup-steps=760",
            "--save-interval=3000",
            "--eval-interval=7600",
        ]
    )


@pytest.mark.parametrize("model_path", MODEL_CONFIGS, ids=lambda p: p.name)
@pytest.mark.parametrize("backend", ["liger_mhc", "mhc"])
def test_all_backbones_use_mhc_config_and_sublayer_selection(entry, args, model_path, backend):
    args.model_config = str(model_path)
    args.mhc_backend = backend
    config = entry.build_config(args, [])
    raw = json.loads(model_path.read_text())
    routing = config.model.block.hyper_connections
    assert config.model.block.name == TransformerBlockType.mhc
    assert routing.kind == HyperConnectionsKind(backend)
    assert routing.num_residual_streams == 4
    assert not routing.use_tanh
    assert routing.gating_factor_init == 0.01
    assert routing.sinkhorn_iters == 20
    assert routing.liger_phi_dtype == DType.bfloat16
    assert not routing.liger_allow_fp32
    assert routing.liger_rms_eps == routing.liger_sinkhorn_eps == 1e-6
    assert routing.liger_pre_eps == 0.0
    assert routing.liger_post_mult == 2.0
    assert routing.scale_output_init_by_sqrt_n
    assert routing.reduce_mode == ("mean" if backend == "liger_mhc" else "sum")

    model = config.model.build(init_device="meta")
    assert model.d_model == raw["hidden_size"]
    assert len(model.blocks) == raw["num_hidden_layers"]
    assert config.model.block.feed_forward.hidden_size == raw["intermediate_size"]
    attention = config.model.block.sequence_mixer
    assert attention.n_heads == raw["num_attention_heads"]
    assert attention.n_kv_heads == raw["num_key_value_heads"]
    expected_head_dim = raw.get("head_dim", raw["hidden_size"] // attention.n_heads)
    for index, block in enumerate(model.blocks.values()):
        assert block.attention.head_dim == expected_head_dim
        assert not hasattr(block, "attention_residual_stream")
        assert not hasattr(block, "feed_forward_residual_stream")
        for sub, connector in enumerate(
            (block.attention_hyper_connection, block.feed_forward_hyper_connection)
        ):
            assert connector.kind == backend
            assert connector.selected_stream == (2 * index + sub) % 4
    assert config.data_loader.global_batch_size == 512 * 2048
    assert config.train_module.rank_microbatch_size == 8 * 2048
    assert config.train_module.optim.lr == 2e-3
    assert config.train_module.optim.betas == (0.9, 0.95)
    assert config.train_module.max_grad_norm == 1.0
    assert config.train_module.scheduler.warmup == 760
    assert config.trainer.max_duration.value == 7600
    assert config.trainer.callbacks["lm_evaluator"].eval_interval == 7600
    assert config.trainer.callbacks["checkpointer"].save_interval == 3000
    expected_patterns = {
        pattern
        for branch in ("attention_hyper_connection", "feed_forward_hyper_connection")
        for pattern in routing.static_parameter_patterns(f"blocks.*.{branch}")
    }
    assert any(
        set(group.params) == expected_patterns and group.opts == {"weight_decay": 0.0}
        for group in config.train_module.optim.group_overrides
    )


def test_routing_overrides_are_not_discarded(entry, args):
    config = entry.build_config(
        args,
        [
            "model.block.hyper_connections.num_residual_streams=2",
            "model.block.hyper_connections.sinkhorn_iters=12",
            "model.block.hyper_connections.disable_static_weight_decay=false",
            "train_module.optim.lr=0.001",
        ],
    )
    routing = config.model.block.hyper_connections
    assert routing.num_residual_streams == 2
    assert routing.sinkhorn_iters == 12
    assert not config.train_module.optim.group_overrides
    assert config.train_module.optim.lr == 0.001


def test_static_backend_remains_an_explicit_ablation(entry, args):
    args.mhc_backend = "mhc_static"
    config = entry.build_config(args, [])
    assert config.model.block.hyper_connections.kind == HyperConnectionsKind.mhc_static
    assert config.model.block.hyper_connections.reduce_mode == "sum"


@pytest.fixture
def dry_run_env(tmp_path):
    # Capture shell arguments without starting torchrun, CUDA, or a training process.
    launcher = tmp_path / "torchrun"
    launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    launcher.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=f"{tmp_path}:{env.get('PATH', '')}",
        LEARNING_RATE="2e-3",
        LR_TAG="2e3",
        NPROC_PER_NODE="4",
        GLOBAL_BATCH_SIZE="512",
        DEVICE_MICROBATCH_SIZE="8",
        MAX_STEPS="7600",
        RUN_NAME="",
        MHC_BACKEND="liger_mhc",
        SAVE_ROOT="ckpt",
    )
    return env


@pytest.mark.parametrize("model_path", SHAPE_CONFIGS, ids=lambda p: p.stem)
@pytest.mark.parametrize("method", ["hc", "mhc"])
def test_shape_runner_covers_every_400m_config(model_path, method, dry_run_env):
    shape = model_path.stem.removeprefix("llama_400m_")
    dry_run_env.update(METHOD=method, SHAPE=shape)
    result = subprocess.run(
        ["bash", "examples/pretrain_hyper_connections_shape.sh"],
        cwd=ROOT,
        env=dry_run_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"--model-config=configs/{model_path.name}" in result.stdout
    assert "--global-train-batch-size=512" in result.stdout
    assert "--device-train-microbatch-size=8" in result.stdout
    if method == "mhc":
        assert "--mhc-backend=liger_mhc" in result.stdout
        assert f"--run_name=pretrain-mhc-liger_mhc-400M-{shape}-lr2e3-gap8" in result.stdout
    else:
        assert f"--run_name=pretrain-hc-400M-{shape}-lr2e3" in result.stdout


@pytest.mark.parametrize("backend", ["mhc", "mhc_static"])
def test_shape_runner_labels_native_backends(backend, dry_run_env):
    dry_run_env.update(METHOD="mhc", SHAPE="L24", MHC_BACKEND=backend)
    result = subprocess.run(
        ["bash", "examples/pretrain_hyper_connections_shape.sh"],
        cwd=ROOT,
        env=dry_run_env,
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = "-gap8" if backend == "mhc" else ""
    assert f"--mhc-backend={backend}" in result.stdout
    assert f"--run_name=pretrain-mhc-{backend}-400M-L24-lr2e3{suffix}" in result.stdout.splitlines()
    if backend == "mhc_static":
        assert "gap8" not in result.stdout


def test_400m_example_uses_a_new_gap8_run_directory(dry_run_env):
    result = subprocess.run(
        ["bash", "examples/pretrain_400m.sh"],
        cwd=ROOT,
        env=dry_run_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--run_name=pretrain-mhc-400M-lr2e-3-gap8" in result.stdout
    assert "--save-folder=ckpt/depthbench/pretrain-mhc-400M-lr2e-3-gap8" in result.stdout
