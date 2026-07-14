# HC and mHC transformer blocks

DepthBench registers Hyper-Connections (HC) and manifold-constrained
Hyper-Connections (mHC) as independent transformer block types. The standard
pre-norm block is unchanged. In the HC/mHC blocks, each connector wraps only the
pre-normalized attention or feed-forward branch and replaces OLMo-core's
`ResidualStream`; there is no second residual addition around the connector.

## Training entries

Use `examples/pretrain_hc.py` for HC and `examples/pretrain_mhc.py` for mHC. Both
default to four residual streams. The mHC entry uses the paper-aligned routing
scale `alpha=0.01`, 20 Sinkhorn iterations, and the fused Liger backend:

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

Internally streams are folded into the batch dimension as `[B * HC, T, C]` and
collapsed to `[B, T, C]` before the final LM-head normalization/projection.
Tensor, sequence, and context parallelism and MoE combinations are not supported
for these block types. DDP and FSDP data parallel training are supported.
