# Liger-backed mHC

DepthBench supports a Liger Kernel implementation of Manifold-Constrained
Hyper-Connections through:

```json
{
  "hyper_connections": {
    "kind": "liger_mhc",
    "num_residual_streams": 4,
    "gating_factor_init": 0.01,
    "sinkhorn_iters": 20,
    "liger_phi_dtype": "bfloat16",
    "collapse": "auto"
  }
}
```

## Requirements

Install a Liger Kernel release that includes the mHC functional APIs:

```bash
pip install "liger-kernel>=0.8.0"
```

For the paper-default `sinkhorn_iters=20`, use a runtime with
`torch>=2.8` and `triton>=3.4`. On H100, `liger-kernel==0.8.0` with
`torch 2.6.0` / `triton 3.2.0` was observed to hang in
`liger_mhc_coeffs` even for tiny inputs, while `torch 2.8.0` /
`triton 3.4.0` completed the same kernel and the OLMo-core smoke test.

The backend imports Liger lazily. Baseline, HC, PyTorch `mhc`, and `mhc_static`
do not require Liger to be installed.

## Architecture Rules

`kind="liger_mhc"` reuses the existing OLMo-core hyper-connection path. It
wraps the pre-norm attention and FFN branches and does not create or apply the
standard `ResidualStream` modules, so there is no double residual path.

Internally the model keeps the existing folded stream layout
`[batch * streams, seq, dim]`. The Liger connector reshapes to
`[batch, seq, streams, dim]` only around the fused Liger coefficient,
pre-aggregation, and post-residual kernels, then folds the output back.

For Liger mHC, `collapse="auto"` means mean collapse over streams before the
final LM-head norm. Existing HC and PyTorch mHC backends keep their previous
sum-collapse behavior.

## Unsupported In This Pass

Tensor parallelism, sequence parallelism, context parallelism, and MoE are not
implemented for Liger mHC. The existing hyper-connection path already raises
`NotImplementedError` for tensor/context parallel usage.

## Smoke Test And Benchmark

Run a tiny CUDA smoke benchmark:

```bash
PYTHONPATH=pretrain/OLMo-core/src python scripts/benchmark_liger_mhc.py \
  --batch-size 2 --seq-len 128 --steps 20 --warmup 5
```

The script reports environment information, tokens/s, and peak CUDA memory for
baseline, PyTorch mHC, and Liger mHC when the required CUDA/Liger dependencies
are available.

## Example Training Commands

The PR includes minimal configs and shell launchers without cluster-specific
submit files:

```bash
examples/pretrain_llama_130M_hc.sh
examples/pretrain_llama_130M_mhc.sh
VENV_DIR=.venv_liger examples/pretrain_llama_350M_liger_mhc_linear.sh
```

For Liger mHC on shared filesystems, put Triton/TorchInductor caches on local
storage if available:

```bash
export TRITON_CACHE_DIR=/tmp/$USER-triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER-torchinductor-cache
export XDG_CACHE_HOME=/tmp/$USER-xdg-cache
```

This avoids transient network-filesystem failures during Triton kernel
compilation, such as `OSError: [Errno 116] Stale file handle`.
