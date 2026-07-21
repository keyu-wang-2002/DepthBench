# HC and mHC transformer blocks

DepthBench registers Hyper-Connections (HC) and manifold-constrained
Hyper-Connections (mHC) as independent transformer block types. The standard
pre-norm block is unchanged. In the HC/mHC blocks, each connector wraps only the
pre-normalized attention or feed-forward branch and replaces OLMo-core's
`ResidualStream`; there is no second residual addition around the connector.

## Training entries

Use `examples/pretrain_hc.py` for HC and `examples/pretrain_mhc.py` for mHC. Both
default to four residual streams. The mHC entry uses the paper routing settings
`alpha=0.01` and 20 Sinkhorn iterations with the fused Liger backend:

```bash
pip install 'liger-kernel>=0.8.0'
torchrun --nproc_per_node=4 examples/pretrain_mhc.py \
  --model-config=configs/llama_400m_L24.json \
  --mhc-backend=liger_mhc \
  <common training arguments>
```

`--mhc-backend=mhc_static` selects the dependency-free static PyTorch backend,
while `--mhc-backend=mhc` selects the experimental input-dependent PyTorch
backend. Liger mHC requires a CUDA environment with compatible recent PyTorch
and Triton versions.

For the current 400M-tier shape experiments, use the repo-relative runner. It
defaults to 7,600 steps, global batch size 512, checkpoints every 3,000 steps,
and a single full evaluation at the final step:

```bash
METHOD=hc SHAPE=L16 LEARNING_RATE=2e-3 \
  bash examples/pretrain_hyper_connections_shape.sh

METHOD=mhc SHAPE=L24 LEARNING_RATE=2e-3 MHC_BACKEND=liger_mhc \
  bash examples/pretrain_hyper_connections_shape.sh
```

Override `DATA_ROOT`, `SAVE_ROOT`, `NPROC_PER_NODE`, and
`DEVICE_MICROBATCH_SIZE` for the local cluster layout. The runner supports the
L16, L20, L24, L26, L28, L30, and L32 configs.

The Liger backend matches the paper's per-sublayer routing equations, but the
default DepthBench experiment is not a bit-for-bit reproduction of the paper's
system implementation. It keeps `phi` in BF16 for Tensor Core throughput,
whereas the paper specifies TF32, and it applies the existing
`scale_output_init_by_sqrt_n` initialization option. The paper does not specify
the final multi-stream readout; DepthBench currently uses mean collapse. The
later DeepSeek TileKernels release contains a learned, input-dependent mHC head,
which is not implemented here. Treat `mhc_static` as a historical static
ablation, not as the paper's dynamic mHC method.

Internally streams are folded into the batch dimension as `[B * HC, T, C]` and
collapsed to `[B, T, C]` before the final LM-head normalization/projection.
Tensor, sequence, and context parallelism and MoE combinations are not supported
for these block types. DDP and FSDP data parallel training are supported.
