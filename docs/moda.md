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
environment:

```bash
git clone https://github.com/hustvl/MoDA.git
pip install -e ./MoDA/libs/moda_triton
```

Then use `examples/pretrain_moda.py`, `examples/pretrain_postnorm.py`, or
`examples/pretrain_postnorm_moda.py` with a normal shape config such as
`configs/llama_400m_L24.json`.

The initial integration supports dense DDP/FSDP training. Tensor, sequence, and
context parallelism, MoE, packed-document masks, and inference K/V caches are
not supported. The official v17 kernel uses a depth tile size of 64; validate
non-standard head dimensions on the target GPU before launching a full sweep.
