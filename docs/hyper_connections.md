# HC and mHC transformer blocks

DepthBench registers Hyper-Connections (HC) and manifold-constrained
Hyper-Connections (mHC) as independent transformer block types. The standard
pre-norm block is unchanged. In the HC/mHC blocks, each connector wraps only the
pre-normalized attention or feed-forward branch and replaces OLMo-core's
`ResidualStream`; there is no second residual addition around the connector.

## Training entries

Use `examples/pretrain_hc.py` for HC and `examples/pretrain_mhc.py` for mHC. Both
default to four residual streams. The mHC entry uses the paper routing settings
`alpha=0.01` and 20 Sinkhorn iterations with the fused Liger backend and
DepthBench's final gap-8 initialization:

```bash
pip install 'liger-kernel>=0.8.0'
torchrun --nproc_per_node=4 examples/pretrain_mhc.py \
  --model-config=configs/llama_400m_L24.json \
  --mhc-backend=liger_mhc \
  <common training arguments>
```

`--mhc-backend=mhc_static` selects the dependency-free static PyTorch backend,
while `--mhc-backend=mhc` selects the experimental input-dependent PyTorch
backend. For the default 20 Sinkhorn iterations, use a CUDA environment with
PyTorch >= 2.8 and Triton >= 3.4; older Triton versions can hang during kernel
compilation and are rejected by the runtime check.

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
L16, L20, L24, L26, L28, L30, L32, L36, L42, L50, L56, L62, and L70 configs.
Its dynamic mHC run names include the backend and `gap8` so new runs do not automatically
resume a directory created with the old initialization. An explicit `RUN_NAME`
still takes precedence.

All files under `configs/` describe method-independent backbone shapes. Use
`pretrain_mhc.py --model-config=...` to apply this same routing recipe to any of
them, including `configs/llama_1600m_L28`, without changing the Pre-LN/HC shapes.
Training arguments such as learning rate, token budget and global batch remain
independent of the initialization change. Routing dotlist overrides are applied
after installing the mHC block, before deriving the optimizer's no-decay groups.

## Initialization and checkpoint compatibility

The two dynamic backends (`liger_mhc` and `mhc`) now share the final experimental
initialization. It is a fixed initialization, not a selectable legacy mode:

| Parameter | Initial value |
| --- | --- |
| Dynamic routing projections (`phi` for Liger) | Zero |
| Pre-routing logits | Selected stream `+8`, all others `-8` |
| Selected stream | `sublayer_index % num_residual_streams` |
| Post-routing logits | Zero, giving `2 * sigmoid(0) = 1` |
| Residual logits | Diagonal `0`, off-diagonal `-8` |
| Routing gains | `0.01` |

Attention uses sublayer index `2 * block_idx`; the FFN uses `2 * block_idx + 1`.
For four streams, the initial Sinkhorn-projected residual matrix has diagonal
approximately `0.998995` and off-diagonal approximately `0.000335`, not an exact
identity. Liger keeps `phi` in BF16, biases/gains in FP32, 20 Sinkhorn iterations,
`rms_eps=sinkhorn_eps=1e-6`, `pre_eps=0`, and `post_mult=2`. Biases/gains are excluded
from weight decay while `phi` remains decayed. Final mean collapse and the
existing `1/sqrt(num_streams)` branch-output initialization scale are unchanged.

There is no change to forward equations, parameter names/shapes, or checkpoint
loading. Loading a complete old checkpoint overwrites the initialized routing
weights; the new initializer must not run after loading. Starting from an old
config without loading its weights now uses gap-8 initialization. To reproduce
an old run from scratch, use its historical code revision. No legacy initializer
is retained here. HC and the `mhc_static` ablation are unchanged.

Matching initialization does not make the native PyTorch and Liger backends
numerically interchangeable: they retain their existing precision, normalization
epsilon and collapse behavior (`sum` for native PyTorch, `mean` for Liger).
Use `liger_mhc` for the standard DepthBench mHC experiment recipe.

CPU regressions cover initializer/reset values, sublayer selection after meta
initialization, all shipped backbone configurations and example shell commands.
An optional CUDA test runs 20 tiny-model training steps and a checkpoint roundtrip:

```bash
PYTHONPATH=pretrain/OLMo-core/src pytest \
  pretrain/OLMo-core/src/test/nn/hyper_connections_test.py \
  pretrain/OLMo-core/src/test/nn/transformer/mhc_initialization_test.py \
  pretrain/OLMo-core/src/test/nn/transformer/mhc_experiments_test.py
```

## Architecture notes

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
