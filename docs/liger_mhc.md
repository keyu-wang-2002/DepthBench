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

## 350M Liger mHC Experiments

The current 350M Liger mHC runs use the shared launcher:

```bash
scripts/run_mhc_linear_350m_8gpu.sh
```

The launcher is parameterized through environment variables so the Condor
submit files capture the exact experiment settings:

- `VENV_DIR=/fast/wliu/yy/DepthBench_workspace/DepthBench/.venv_b200`
- `NPROC_PER_NODE=4`
- `DEVICE_TRAIN_MICROBATCH_SIZE=4`
- `GLOBAL_TRAIN_BATCH_SIZE=512`
- `MAX_STEPS=7600`
- `WARMUP_STEPS=760`
- `MODEL_CONFIG=configs/llama_350M_liger_mhc_linear_*.json`
- `LEARNING_RATE` sweep: `5e-3`, `2e-3`, `1e-3`, `5e-4`

The shape-sweep submit file covers the full 7-shape sweep:

```bash
condor_submit_bid 35 scripts/condor_train_350m_liger_mhc_linear_shape_sweep_4gpu_h100.sub
```

It uses 4 H100 GPUs per job and the following configs:

| Layers | Hidden | Intermediate | Heads | Config |
| --- | ---: | ---: | ---: | --- |
| 16 | 1216 | 3248 | 16 | `configs/llama_350M_liger_mhc_linear_l16_h1216_i3248.json` |
| 20 | 1120 | 2992 | 16 | `configs/llama_350M_liger_mhc_linear_l20_h1120_i2992.json` |
| 24 | 1024 | 2736 | 16 | `configs/llama_350M_liger_mhc_linear_l24_h1024_i2736.json` |
| 26 | 992 | 2656 | 16 | `configs/llama_350M_liger_mhc_linear_l26_h992_i2656.json` |
| 28 | 960 | 2560 | 16 | `configs/llama_350M_liger_mhc_linear_l28_h960_i2560.json` |
| 30 | 928 | 2480 | 16 | `configs/llama_350M_liger_mhc_linear_l30_h928_i2480.json` |
| 32 | 896 | 2400 | 16 | `configs/llama_350M_liger_mhc_linear_l32_h896_i2400.json` |

The edge-shape submit file starts only the 16-layer and 32-layer sweeps:

```bash
condor_submit_bid 35 scripts/condor_train_350m_liger_mhc_linear_edge_shapes_4gpu_h100.sub
```

If Triton compilation fails on network storage with `OSError: [Errno 116]
Stale file handle`, use the retry submit file, which sets local `/tmp` cache
directories for Triton/TorchInductor:

```bash
condor_submit_bid 35 scripts/condor_train_350m_liger_mhc_linear_l32_lr1e3_retry_4gpu_h100.sub
```
