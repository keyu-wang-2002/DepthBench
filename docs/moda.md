# Mixture-of-Depths Attention

DepthBench exposes MoDA as dedicated dense transformer block types rather than a
hyper-connection backend:

- `post_norm`: post-norm residual baseline.
- `moda`: pre-norm MoDA for comparison against the standard pre-norm baseline.
- `post_norm_moda`: paper-aligned post-norm MoDA.

The MoDA blocks reuse each attention layer's Q/K/V/O projections, add FFN K/V
projections for the depth cache, and call the official Triton depth-attention
kernel after the first attention sublayer. The final FFN K/V projection is
omitted because no later layer reads it.

## Install

Clone the official repository and install its kernel package into the training
environment. Apply DepthBench's compatibility patch before installing when the
target is H100 or a model uses a non-64 head dimension (for example L16's 76 or
L20's 70):

```bash
git clone https://github.com/hustvl/MoDA.git
git -C ./MoDA apply ../DepthBench/patches/moda_v17_h100_head_dims.patch
pip install -e ./MoDA/libs/moda_triton
```

The patch adds explicit H100 dispatch, makes the Triton tiles cover head
dimensions 70 and 76, preserves explicit tuning overrides, and removes a
per-call unsupported-GPU print that otherwise floods H100 training logs. It is
based on official MoDA commit `ba872a3`.

Then use `examples/pretrain_moda.py`, `examples/pretrain_postnorm.py`, or
`examples/pretrain_postnorm_moda.py` with a normal shape config such as
`configs/llama_400m_L24.json`.

The current 400M-tier sweep can be launched with the repo-relative runner. It
defaults to 7,600 steps, global batch size 512, checkpoints every 3,000 steps,
and a single full evaluation at the final step:

```bash
VARIANT=prenorm_moda SHAPE=L16 LEARNING_RATE=2e-3 \
  bash examples/pretrain_moda_shape.sh

VARIANT=postnorm_moda SHAPE=L24 LEARNING_RATE=1e-3 \
  bash examples/pretrain_moda_shape.sh
```

`VARIANT=postnorm_baseline` selects the matching baseline. Override
`DATA_ROOT`, `SAVE_ROOT`, `NPROC_PER_NODE`, and
`DEVICE_MICROBATCH_SIZE` for the local cluster layout.

The initial integration supports dense DDP/FSDP training. Tensor, sequence, and
context parallelism, MoE, packed-document masks, and inference K/V caches are
not supported.

## Cache Layout And H100 Validation

DepthBench preallocates each block's depth K/V cache as
`[B, T, max_depth, H, D]` and passes a zero-copy
`[B, T * max_depth, H, D]` view to the v17 kernel. The kernel uses
`current_depth` to ignore unfilled slots. Do not slice the depth dimension
before flattening it: that layout is non-contiguous and forces a new packed
cache allocation at every layer.

The patched kernel was validated with BF16 forward and backward on H100 for
head dimensions 64, 70, and 76. A full-block comparison used batch size 1 and
sequence length 2048 on the L16, L20, and L24 400M-tier shapes:

| Shape | Baseline tokens/s | Legacy MoDA tokens/s | Full-cache MoDA tokens/s | Full-cache speedup | Peak memory saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| L16, head dim 76 | 40,819 | 20,039 | 22,866 | 1.14x | 1.93 GiB |
| L20, head dim 70 | 29,072 | 12,567 | 12,834 | 1.02x | 2.91 GiB |
| L24, head dim 64 | 35,713 | 12,835 | 15,883 | 1.24x | 3.94 GiB |

All gradients were finite and legacy/full-cache losses agreed within `1e-4`.
MoDA remains slower than the baseline because its depth attention and extra FFN
K/V projections are real additional work; the cache fix removes avoidable
packing overhead but does not eliminate that algorithmic cost.
