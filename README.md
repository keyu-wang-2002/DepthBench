<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img src="assets/banner.svg" alt="DepthBench" height="60">
</picture>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg?style=flat-square)](#citation)
[![hf_model](https://img.shields.io/badge/-Checkpoints-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/aspect-ratio-scaling)
[![Built on OLMo-core](https://img.shields.io/badge/built_on-OLMo--core-f0529c.svg?style=flat-square)](https://github.com/allenai/OLMo-core)
[![Python 3.10](https://img.shields.io/badge/python-3.10-3776ab.svg?logo=python&logoColor=white&style=flat-square)](environment/README.md)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8_cu128-ee4c2c.svg?logo=pytorch&logoColor=white&style=flat-square)](environment/README.md)

</div>

<p>
  🧱 DepthBench is a controlled testbed for <b>measuring how residual connections enable more computational depth</b>.
  It trains a broad range of residual-connection and normalization architectures (Pre-LN, Peri-LN, LNS,
  DeepNorm, KEEL, depth-μP, CompleteP, HC, mHC, MoDA and AttnRes), sweeps aspect ratios (width / depth) under the same data, token budget and fixed parameter count, and ships a set of architecture-aware probes
  (angular distance, causal score, permutation score, logit lens, layer pruning) that run
  directly on native OLMo-core checkpoints.
</p>

--------

* [News](#news)
* [Models](#models)
* [Checkpoints](#checkpoints)
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Training](#training)
  * [Model Configs](#model-configs)
  * [Launching a Run](#launching-a-run)
  * [Layer Statistics](#layer-statistics)
* [Depth Analysis](#depth-analysis)
  * [Depth Metrics](#depth-metrics)
* [Evaluation](#evaluation)
* [Repository Structure](#repository-structure)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## News

- [2026-09] 🚀 Initial public release of DepthBench: 11 depth architectures, 25 model shapes from 200M to 1.6B, and native-checkpoint depth analysis.
- [2026-09] 🤗 160+ pre-trained checkpoints are available on the Hugging Face Hub at [aspect-ratio-scaling](https://huggingface.co/aspect-ratio-scaling).

## Models

Each architecture has its own training entrypoint in [`examples/`](examples). All of them share one
Llama-style backbone, optimizer and data pipeline ([`examples/pretrain_llama_base.py`](examples/pretrain_llama_base.py)),
so the residual/normalization scheme is the only thing that changes.
Block implementations live in [`olmo_core/nn/transformer/block.py`](pretrain/OLMo-core/src/olmo_core/nn/transformer/block.py).

| Year | Model | Paper | |
| :---: | :---: | :--- | :--- |
| 2020 | Pre-LN | [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | [code](examples/pretrain_preln.py) |
| 2022 | DeepNorm | [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) | [code](examples/pretrain_deepnorm.py) |
| 2023 | Depth-μP | [Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks](https://arxiv.org/abs/2310.02244) | [code](examples/pretrain_preln_mup.py) |
| 2024 | HC | [Hyper-Connections](https://arxiv.org/abs/2409.19606) | [code](examples/pretrain_hc.py) · [docs](docs/hyper_connections.md) |
| 2025 | Peri-LN (Sandwich-LN) | [Peri-LN: Revisiting Normalization Layer in the Transformer Architecture](https://arxiv.org/abs/2502.02732) | [code](examples/pretrain_periln.py) |
| 2025 | LNS | [The Curse of Depth in Large Language Models](https://arxiv.org/abs/2502.05795) | [code](examples/pretrain_lns.py) |
| 2025 | CompleteP | [Don't be lazy: CompleteP enables compute-efficient deep transformers](https://arxiv.org/abs/2505.01618) | [code](examples/pretrain_preln_mup.py) |
| 2025 | mHC | [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | [code](examples/pretrain_mhc.py) · [docs](docs/hyper_connections.md) |
| 2026 | KEEL | [Post-LayerNorm Is Back: Stable, ExpressivE, and Deep](https://arxiv.org/abs/2601.19895) | [code](examples/pretrain_keel.py) |
| 2026 | Full / Block AttnRes | [Attention Residuals](https://arxiv.org/abs/2603.15031) | [code](examples/pretrain_attnres.py) |
| 2026 | MoDA (pre-/post-norm) | [Mixture-of-Depths Attention](https://arxiv.org/abs/2603.15619) | [code](examples/pretrain_moda.py) · [docs](docs/moda.md) |

## Checkpoints

We release 160+ pre-trained checkpoints on the Hugging Face Hub under
[🤗 aspect-ratio-scaling](https://huggingface.co/aspect-ratio-scaling).
They cover most architectures above at a broad range of width-depth aspect ratios.

| Tier | Shapes (depth-width) | Architectures |
|---|---|---|
| 400M | L16-d1216 · L20-d1120 · L24-d1024 · L28-d960 · L32-d896 | all |
| 400M deep | L36-d864 · L42-d800 · L70-d640 | HC, mHC, Full AttnRes |
| 1.6B | L28-d2048 · L40-d1728 · L54-d1504 | Pre-LN, HC, Full AttnRes |
| 300M backbone | L16-d1248 · L42-d768 · L56-d672 · L70-d608 | HC, mHC |
| 200M | L12-d896 · L18-d768 · L24-d672 | Pre-LN, HC, mHC, Full / Block AttnRes |
| 300M | L14-d1056 · L21-d896 · L28-d800 | Pre-LN, HC, mHC, Full / Block AttnRes |
| 500M | L17-d1344 · L26-d1120 · L34-d992 | Pre-LN, HC, mHC, Full / Block AttnRes |

Repositories are named `<arch>-lr<lr>-llama-<size>-L<layers>-pretrain`, e.g.
[`preln-lr2e-3-llama-400M-L24-pretrain`](https://huggingface.co/aspect-ratio-scaling/preln-lr2e-3-llama-400M-L24-pretrain)
(Pre-LN uses `base` at 200M / 300M / 500M).
Each repository is a **raw OLMo-core distributed checkpoint**, not a `transformers` export. Most hold
`step*/` directories (initial, intermediate and final steps, each with its `config.json`) plus the
tokenizer; others store the final checkpoint (`config.json` + `model_and_optim/`) directly at the repository
root. Every analysis and evaluation script in this repository loads either layout directly:

```bash
huggingface-cli download aspect-ratio-scaling/preln-lr2e-3-llama-400M-L24-pretrain \
  --local-dir ckpt/hf/preln-400M-L24

# a run directory resolves to its latest step
python analysis/compute_angular_distance.py --model_path ckpt/hf/preln-400M-L24 \
  --output_dir results/angular --token-data-glob "data/fineweb-edu/pre-tokenize/eval/*.npy"
python eval/run_zero_shot.py ckpt/hf/preln-400M-L24/step7600 --device cuda:0 --batch-size 32
```

For step-directory repositories, add `--include "step7600/*" "tokenizer/*"` to download only the final step.
`step7600` is the final step at 400M; the other tiers end at `step3800` (200M), `step5700` (300M),
`step9500` (500M) and `step30400` (1.6B).

## Installation

DepthBench targets Linux x86_64 with NVIDIA GPUs (experiments were run on B200/H100/A100). The reference environment
is Python 3.10, PyTorch 2.8.0 + CUDA 12.8, Triton 3.4 and Liger Kernel 0.8. It uses a **modified OLMo-core**,
vendored in [`pretrain/OLMo-core`](pretrain/OLMo-core). Do not install `ai2-olmo-core` from PyPI.

```bash
git clone https://github.com/keyu-wang-2002/DepthBench.git
cd DepthBench
bash environment/install.sh          # creates .venv-cu128 with pinned dependencies
source .venv-cu128/bin/activate
```

The installer builds a fresh venv, installs the local OLMo-core, puts official FLA 0.4.1 in an isolated
overlay for AttnRes, and runs `pip check`. It covers Pre-LN, Peri-LN, LNS, DeepNorm, KEEL,
depth-μP/CompleteP, HC and mHC out of the box. A Conda recipe is also available in
[`environment/README.md`](environment/README.md).

> [!IMPORTANT]
> **MoDA** and **AttnRes** need two different, mutually incompatible `fla` packages. Choose one per process
> with `PYTHONPATH` before launching. See [Select Dependencies](environment/README.md#select-dependencies)
> and [MoDA Kernel](environment/README.md#moda-kernel).

All commands below are run from the repository root.

## Data Preparation

All models are pre-trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(`sample/100BT`) with the GPT-NeoX/OLMo tokenizer bundled in OLMo-core.

**1. Download** the parquet shards from [`sample/100BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT)
into `data/fineweb-edu/100BT/`, and hold out one shard for evaluation:

```bash
mkdir -p data/fineweb-edu/eval
mv data/fineweb-edu/100BT/013_00008.parquet data/fineweb-edu/eval/eval_013_00008.parquet
```

**2. Pre-tokenize** into `.npy` token shards. The training set can be split across several processes
with `--train-worker-id` / `--train-num-workers`:

```bash
python data_utils/tokenize_from_pretrain_datasets.py \
  --train-parquet-glob "data/fineweb-edu/100BT/*.parquet" \
  --eval-parquet-path "data/fineweb-edu/eval/eval_013_00008.parquet" \
  --output-dir "data/fineweb-edu/pre-tokenize" \
  --text-field "text" \
  --tokenizer-name-or-path "pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
  --vocab-size 50280 --eos-token-id 50279 --pad-token-id 1 \
  --batch-size 4096 \
  --write-doc-indices --skip-existing \
  --train-worker-id 0 --train-num-workers 1
```

The result is `data/fineweb-edu/pre-tokenize/{train,eval}/*.npy`, which is the default data location for
every training script.

## Training

### Model Configs

Model shapes are plain Llama-style JSON files in [`configs/`](configs). They include a 400M series that
sweeps depth from 16 to 70 layers at fixed total size, a 300M series at fixed backbone size, a
200M–500M depth-scaling ladder, and 1.6B shapes based on Qwen3-1.7B. Each series keeps parameter count
within ±3%.

| Tier | Example config | Layers | Hidden | Tokens | Steps |
|---|---|---:|---:|---:|---:|
| 200M | `llama_200m_L18.json` | 12 – 24 | 672 – 896 | 4B | 3.8k |
| 300M | `llama_300m_L21.json` | 14 – 28 | 800 – 1056 | 6B | 5.7k |
| 400M | `llama_400m_L24.json` | 16 – 70 | 640 – 1216 | 8B | 7.6k |
| 500M | `llama_500m_L26.json` | 17 – 34 | 992 – 1344 | 10B | 9.5k |
| 1.6B | `llama_1600m_L28.json` | 28 – 54 | 1504 – 2048 | 32B | 30.4k |

See [`configs/README.md`](configs/README.md) for every shape, including parameter counts and the design rules.

### Launching a Run

Every entrypoint takes the same arguments. A minimal 400M Pre-LN run on 8 GPUs:

```bash
torchrun --nproc_per_node=8 examples/pretrain_preln.py \
  --run_name=pretrain-preln-400M-lr2e-3 \
  --model-config=configs/llama_400m_L24.json \
  --max-steps=7600 --warmup-steps=760 \
  --global-train-batch-size=512 --device-train-microbatch-size=16 \
  --learning-rate=2e-3 \
  --save-folder=ckpt/depthbench/pretrain-preln-400M-lr2e-3
```

To train a different architecture, swap the entrypoint (see [Models](#models)).

<details>
<summary>Architecture-specific options</summary>

| Architecture | Entrypoint | Extra arguments |
|---|---|---|
| Pre-LN / Peri-LN / LNS / DeepNorm / KEEL | `pretrain_{preln,periln,lns,deepnorm,keel}.py` | none |
| Depth-μP / CompleteP | `pretrain_preln_mup.py` | `--parameterization {depth-muP,completeP} --scaling-axis {depth-only,depth-width} --base-depth 24 --base-width 1024` |
| HC | `pretrain_hc.py` | none (4 residual streams) |
| mHC | `pretrain_mhc.py` | `--mhc-backend {liger_mhc,mhc_static,mhc}` (default: fused Liger) |
| Full AttnRes | `pretrain_attnres.py` | none (`model.block.attnres_block_size=1` is the default) |
| Block AttnRes | `pretrain_attnres.py` | `model.block.attnres_block_size=<N>` |
| MoDA | `pretrain_moda.py` / `pretrain_postnorm_moda.py` | none (needs the MoDA dependency selection) |

Any other field of the OLMo-core experiment config can be overridden with trailing
`dotted.key=value` arguments, as in the AttnRes rows above.

</details>

<details>
<summary>Common arguments</summary>

| Argument | Default | Description |
|---|---|---|
| `--model-config` | `configs/llama_400m_L24.json` | Model shape |
| `--train-data-glob` / `--eval-data-glob` | `data/fineweb-edu/pre-tokenize/{train,eval}/*.npy` | Pre-tokenized data |
| `--max-steps` / `--warmup-steps` | `7600` / `760` | Cosine schedule with linear warmup |
| `--global-train-batch-size` | `512` | Sequences per optimizer step |
| `--device-train-microbatch-size` | `16` | Per-GPU micro-batch; lower it for deep or wide shapes |
| `--learning-rate` | `1e-3` | Peak AdamW learning rate (`2e-3` in our 400M runs) |
| `--max-grad-norm` | `1.0` | Set ≤ 0 to disable clipping |
| `--eval-interval` / `--save-interval` | `200` / `10000` | In steps |
| `--save-folder` | – | Checkpoint directory |
| `--load-path` / `--load-trainer-state` | – | Resume or initialize from a checkpoint |
| `--wandb-project` / `--wandb-entity` | `depthbench` / none | Set `--wandb-project ""` to disable W&B |

</details>

Ready-to-run scripts:

- [`examples/pretrain_400m.sh`](examples/pretrain_400m.sh) — every architecture at the 400M base shape.
- [`examples/pretrain_hyper_connections_shape.sh`](examples/pretrain_hyper_connections_shape.sh) — HC/mHC over the 400M depth sweep (`METHOD=hc|mhc SHAPE=L16 ...`).
- [`examples/pretrain_moda_shape.sh`](examples/pretrain_moda_shape.sh) — MoDA and its post-norm baseline over the depth sweep (`VARIANT=prenorm_moda SHAPE=L16 ...`).

### Layer Statistics

Pass `--enable-layer-stats` to log per-block statistics of the hidden state (`forward`) and of its
gradient (`backward`): `mean`, `variance`, `magnitude` (= `abs().mean()`) and `norm` (L2).
`--layer-stats-interval N` logs every N steps. The metrics go to W&B, where regex panels keep them readable:

```text
^train/layer_stats/block_\d+/forward/norm$
^train/layer_stats/block_\d+/backward/norm$
```

## Depth Analysis

All analysis scripts in [`analysis/`](analysis) load **native OLMo-core checkpoints**, so no Hugging Face
conversion is needed. They detect the architecture automatically.

### Depth Metrics

Each metric is defined over *depth states* `z_0, ..., z_L` (`z_0` is the embedding output) and two
single-block interventions, *skip* and *swap*. [`analysis/depth_probes.py`](analysis/depth_probes.py)
implements these consistently for all three architecture families:

| Family | Depth state `z_l` | Skipping block `l` | Swapping blocks `i, j` | Logit-lens readout |
|---|---|---|---|---|
| Residual (Pre-LN, Peri-LN, LNS, DeepNorm, KEEL, MoDA) | block output | identity (MoDA writes no depth-KV slots) | all learned block weights (LNS `ln_scale` travels with them) | `z_l` |
| HC / mHC | all residual streams, concatenated | identity on the streams | block weights incl. its hyper-connection parameters | stream sum (= mean after the final RMSNorm) |
| Full / Block AttnRes | depth mix at AttnRes boundaries, block output inside a block group | no source-bank entry; the next boundary mix becomes the identity on its newest source | core weights + the depth-mixing parameters producing its right boundary | the next consumer's depth mix |

Every script takes `--model_path` (a `step*` checkpoint, or a run directory whose latest step is used) and
`--output_dir`, and samples `--num_samples` windows of `--seq_length` tokens from `--token-data-glob`
(or `--text-file` / `--prompt`). Each run writes `results.json`, a figure and the raw arrays.

```bash
EVAL_GLOB="data/fineweb-edu/pre-tokenize/eval/*.npy"
CKPT=ckpt/depthbench/pretrain-preln-400M-lr2e-3

# Angular distance   d(i,j) = mean_t arccos cos(z_i, z_j) / pi
python analysis/compute_angular_distance.py --model_path $CKPT --output_dir results/angular \
  --token-data-glob "$EVAL_GLOB" --num_samples 128 --seq_length 512

# Causal score       C(s,l) = mean_t ||u_l^{skip s} - u_l|| / ||u_l||,  u_l = z_{l+1} - z_l
python analysis/compute_causal_score.py --model_path $CKPT --output_dir results/causal \
  --token-data-glob "$EVAL_GLOB" --num_samples 16 --seq_length 256

# Permutation score  P(i,j) = |L_swap(i,j) - L| / L
python analysis/compute_permutation_score.py --model_path $CKPT --output_dir results/permutation \
  --token-data-glob "$EVAL_GLOB" --num_samples 12 --seq_length 256

# Logit lens: early-exit CE, KL(p_final || p_l) and top-5 overlap per depth state
python analysis/compute_logit_lens.py --model_path $CKPT --output_dir results/logit_lens \
  --token-data-glob "$EVAL_GLOB" --num_samples 32 --seq_length 512

# Single-layer pruning, on LM loss ...
python analysis/compute_layer_pruning.py --model_path $CKPT --output_dir results/pruning_loss \
  --token-data-glob "$EVAL_GLOB" --num_samples 16 --seq_length 256
# ... or on a zero-shot lm-eval task (resumable per layer)
python analysis/compute_layer_pruning.py --model_path $CKPT --output_dir results/pruning_arc_easy \
  --task arc_easy --resume
```

`--micro_batch_size` trades memory for speed. On CPU, the Triton kernels (AttnRes, Liger mHC) fall back to
PyTorch automatically. On GPU, set `DEPTHBENCH_USE_LIGER_MHC_FALLBACK=1` to force the PyTorch mHC path.

## Evaluation

[`eval/olmo_lm.py`](eval/olmo_lm.py) wraps a native OLMo-core checkpoint as an
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) model (`lm-eval==0.4.12` is in the pinned environment).

**Zero-shot commonsense.** Covers OpenBookQA, WinoGrande, ARC-Challenge, ARC-Easy, HellaSwag, Social IQa and PIQA:

```bash
python eval/run_zero_shot.py ckpt/depthbench/pretrain-preln-400M-lr2e-3/step7600 \
  --device cuda:0 --batch-size 32 --attention-backend torch \
  --output-path results/zero_shot.json
```

**Completion NLL on coding / STEM / math.** Covers MBPP, HumanEval, SciQ, GPQA-Diamond, GSM8K and MATH-500:

```bash
python eval/run_nll_eval_math_coding_stem.py ckpt/depthbench/pretrain-preln-400M-lr2e-3/step7600 \
  --device cuda:0 --batch-size 8 --output-path results/nll.json
```

## Repository Structure

```text
DepthBench/
├── configs/            # model shapes (Llama-style JSON), see configs/README.md
├── examples/           # training entrypoints, one per architecture, and launch scripts
├── analysis/           # depth probes and metrics on native OLMo-core checkpoints
├── eval/               # lm-eval adapter, zero-shot and NLL evaluation
├── data_utils/         # FineWeb-Edu pre-tokenization, calibration-set builder
├── config_utils/       # JSON config → OLMo-core model/tokenizer config
├── environment/        # pinned CUDA 12.8 environment, installer, MoDA kernel patch
├── docs/               # implementation notes for HC/mHC and MoDA
├── assets/             # logo
└── pretrain/OLMo-core/ # modified OLMo-core with all DepthBench block types
```

## Citation

If you find DepthBench useful, please cite:

```bibtex
@misc{depthbench2026,
  title  = {{DepthBench}: Measuring How Residual Connections Enable More Computational Depth},
  author = {Wang, Keyu and Huang, Yangyi and Kang, Jiale and Gonz{\'a}lez-Mart{\'\i}nez, David and Liu, Weiyang and Liu, Shiwei},
  year   = {2026},
  note   = {TODO: arXiv link},
  url    = {https://github.com/keyu-wang-2002/DepthBench}
}
```

## Acknowledgements

DepthBench is built on [OLMo-core](https://github.com/allenai/OLMo-core) (Apache-2.0). We thank the
authors of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[Liger Kernel](https://github.com/linkedin/Liger-Kernel), [MoDA](https://github.com/hustvl/MoDA) and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their open-source kernels
and tools, and the authors of every architecture listed in [Models](#models).
