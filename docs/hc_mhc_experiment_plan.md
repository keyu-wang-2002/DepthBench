# HC / mHC Experiment Plan

This repo now supports two residual-routing variants that keep the same backbone shape as
`examples/pretrain_llama_130M_base.sh`:

- `HC`: dynamic Hyper-Connections
- `mHC`: manifold-constrained Hyper-Connections

## Paper-aligned settings

The implementation follows the existing DepthBench training recipe unless a paper explicitly changes
the residual-routing setup.

### Shared baseline controls

- Same backbone dimensions as the baseline LLaMA config
- Same tokenizer, data paths, sequence length, batch size, steps, LR, warmup, eval/save cadence
- Same optimizer family: AdamW with betas `(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.1`

### HC alignment

The HC default is aligned to the ICLR 2025 Hyper-Connections paper:

- Dynamic HC is used instead of static HC
- Expansion rate `n = 4`
- `tanh` is enabled on the dynamic routing branch
- Static HC parameters do **not** receive weight decay
- Dynamic HC parameters still receive weight decay
- Attention output projections and FFN output projections are scaled by `1 / sqrt(n)` at init
  to keep the summed residual-stream variance aligned with the baseline

Relevant paper notes:

- The paper keeps the baseline training configuration and replaces only the residual routing
- Dynamic routing weights are zero-initialized so the model starts from a pre-norm-residual-like
  routing pattern
- The paper reports best dense-model ablations with dynamic HC at `n = 4`

### mHC alignment

The mHC default is aligned to the mHC paper comparison setting rather than the toy NanoGPT config
from the public reference repo.

- Expansion rate `n = 4`
- Gating-factor init `alpha = 0.01`
- Sinkhorn iterations `t_max = 20`
- `H_res` is projected to a doubly stochastic matrix with Sinkhorn
- `H_pre` uses `sigmoid`
- `H_post` uses `2 * sigmoid`
- Static constrained-routing biases do **not** receive weight decay
- Dynamic routing projections still receive weight decay
- Attention output projections and FFN output projections are scaled by `1 / sqrt(n)` at init

Important note:

- The public reference repo (`tokenbender/mHC-manifold-constrained-hyper-connections`) uses a
  simplified NanoGPT implementation and its toy configs set `sinkhorn_iters = 10`
- This repo intentionally uses `20` because the mHC paper reports `t_max = 20` in the comparison
  hyper-parameter table and explicitly mentions `20` iterations in its stability analysis

## Commands

Run the baseline:

```bash
cd DepthBench/examples
bash pretrain_llama_130M_base.sh
```

Run HC:

```bash
cd DepthBench/examples
bash pretrain_llama_130M_hc.sh
```

Run mHC:

```bash
cd DepthBench/examples
bash pretrain_llama_130M_mhc.sh
```

## Reuse on other backbone sizes

The new residual backends are driven by the model JSON. To reuse them on another backbone script,
either:

- point `MODEL_CONFIG` to a new JSON that copies the target backbone and adds a `hyper_connections`
  block, or
- add the same `hyper_connections` block to another backbone JSON under `configs/`

For direct comparison runs, keep the non-residual training settings identical to the baseline script.
