# DepthBench



## Environment Setup

```bash
python -m venv depthbench
source depthbench/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

cd pretrain/OLMo-core
python -m pip install -e ".[wandb,transformers]"
python -m pip install datasets pyarrow cached_path
pip install torch transformers numpy tqdm matplotlib seaborn

cd eval/lm-evaluation-harness
python -m pip install -e .
```

## Data Preparation

### 1. Download FineWeb-Edu

Download the FineWeb-Edu parquet shards: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT

split evaluation data:
```bash
mv data/100BT/013_00008.parquet data/eval/eval_013_00008.parquet
```

### 2. Pre-tokenize the dataset

We use gpt-neox tokenizer

```bash
python data_utils/tokenize_from_pretrain_datasets.py \
  --train-parquet-glob "data/fineweb-edu/100BT/*.parquet" \
  --eval-parquet-path "data/fineweb-edu/eval/eval_013_00008.parquet" \
  --output-dir "data/fineweb-edu/pre-tokenize" \
  --text-field "text" \
  --tokenizer-name-or-path "./pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
  --vocab-size 50280 \
  --eos-token-id 50279 \
  --pad-token-id 1 \
  --batch-size 4096 \
  --progress-log-interval-docs 8192 \
  --write-doc-indices \
  --skip-existing \
  --skip-summary \
  --train-worker-id 2 \
  --train-num-workers 32
```

After pre-tokenization, the expected output layout is:

```text
data/fineweb-edu/pre-tokenize/train/*.npy
data/fineweb-edu/pre-tokenize/eval/*.npy
```

## Model Config

The following model configs are currently available under: [`./configs`](./configs)


| Size | Hidden | Intermediate | Heads | Layers | Data Volume | Batch Size | Sequence Length | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 400M | 1024 | 2736 | 16 | 24 | 8B | 512 | 2048 | 7.6k |
| 1B | 2048 | 5504 | 16 | 28 | ? | ? | 2048 | ? |

### 400M Aspect-Ratio Variants （Fix Total Size）

| Layers | Hidden | Intermediate | Heads | head_dim | Aspect Ratio | Backbone Size  | Total Size | Total Diff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1216 | 3248 | 16 | 76 | 76.00 | 284M | 407M | +0.28% |
| 20 | 1120 | 2992 | 16 | 70 | 56.00 | 301M |  414M | +2.14% |
| 24 | 1024 | 2736 | 16 | 64 | 42.67 | 302M |  405M | 0.00% |
| 26 | 992 | 2656 | 16 | 62 | 38.15 | 308M |  408M | +0.56% |
| 28 | 960 | 2560 | 16 | 60 | 34.29 | 310M |  406M | +0.21% |
| 30 | 928 | 2480 | 16 | 58 | 30.93 | 311M |  404M | -0.38% |
| 32 | 896 | 2400 | 16 | 56 | 28.00 | 309M |  399M | -1.49% |
| 36 | 864 | 2304 | 16 | 54 | 24.00 | 323M |  409M | +0.99% |
| 42 | 800 | 2144 | 16 | 50 | 19.05 | 324M |  404M | -0.31% |
| 50 | 736 | 1968 | 16 | 46 | 14.72 | 326M |  400M | -1.41% |
| 56 | 704 | 1888 | 16 | 44 | 12.57 | 334M |  405M | -0.05% |
| 62 | 672 | 1792 | 16 | 42 | 10.84 | 336M |  404M | -0.44% |
| 70 | 640 | 1712 | 16 | 40 | 9.14 | 345M |  409M | +0.94% |



Design Principle
```
heads = 16
head_dim is oven
hidden = heads * head_dim
intermediate = ceil(hidden * 8/3, multiple=16)
Keep total size 405M +- 3%
```

### 300M Backbone Aspect-Ratio Variants （Fix Backbone Size）

| Layers | Hidden | Intermediate | Heads | head_dim | Aspect Ratio | Backbone Size | Backbone Diff | Total Size | 
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | 
| 16 | 1248 | 3328 | 16 | 78 | 78.00 | 299M | -1.11% | 425M | 
| 24 (*) | 1024 | 2736 | 16 | 64 | 42.67 | 302M | +0.00% | 405M | 
| 32 (*) | 896 | 2400 | 16 | 56 | 28.00 | 309M | +2.26% | 399M | 
| 42 | 768 | 2048 | 16 | 48 | 18.29 | 297M | -1.69% | 375M | 
| 56 | 672 | 1792 | 16 | 42 | 12.00 | 304M | +0.37% | 371M | 
| 70 | 608 | 1632 | 16 | 38 | 8.69 | 312M | +3.15% | 373M | 

Design Principle
```
heads = 16
head_dim is oven
hidden = heads * head_dim
intermediate = ceil(hidden * 8/3, multiple=16)
Keep backbone size 302M +- 3%
```

## Training Script

Example:

```bash
cd ./examples
bash pretrain_400m.sh
```

Note: DepthBench now supports per-layer monitoring of hidden-state statistics during pretraining. For each transformer block, we record statistics for both `forward`, the block output hidden state,
and `backward`, the activation gradient on the same hidden state.

For both directions, the following statistics are logged: `mean`, `variance`, `magnitude = abs().mean()`, `norm = l2_norm`

This adds two CLI flags:

- `--enable-layer-stats`
- `--layer-stats-interval 1` means record every step. Set it to a larger value to reduce logging overhead.

These metrics are automatically logged to W&B when W&B is enabled. W&B may create many charts because every block and every statistic is logged separately. A convenient way to view them is to create multi-metric panels with regex, for example:

```text
^train/layer_stats/block_\d+/forward/norm$
^train/layer_stats/block_\d+/backward/norm$
```



## Analysis

This directory contains scripts for running DepthBench analysis metrics directly on native `OLMo-core` checkpoints.

Build calibration text first:

```bash
python data_utils/build_calibration_data.py \
  --source fineweb_local \
  --source c4 \
  --source dolma \
  --output-dir data/calibration \
  --output-prefix calibration \
  --tokenizer-name-or-path ./pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
  --target-total-tokens 262144 \
  --sample-length-mode fixed \
  --sample-length 512 \
  --shuffle-samples
```

This writes `data/calibration/calibration.txt`, `data/calibration/calibration.jsonl`, and
`data/calibration/calibration.summary.json`. Use the `.txt` file in the analysis commands below.

Common options:

- `--source`, repeat to mix multiple corpora; defaults to `fineweb_local`, `c4`, and `dolma`
- `--sample-length-mode {fixed,uniform}`, choose fixed-length or variable-length text windows
- `--sample-length`, token length for fixed mode
- `--min-sample-length` and `--max-sample-length`, token-length range for uniform mode
- `--target-total-tokens`, total token budget across all sources
- `--shuffle-samples`, shuffle collected samples before writing the output files


Run angular distance:

```bash
python analysis/compute_angular_distance.py \
  --model_path ckpt/path/to/ckpt \
  --output_dir analysis/results/angular_distance \
  --text-file data/calibration/calibration.txt \
  --num_samples 1024 \
  --seq_length 512
```

Run Jacobian analysis:

```bash
python analysis/compute_jacobian.py \
  --model_path ckpt/path/to/ckpt \
  --output_dir analysis/results/jacobian \
  --text-file data/calibration/calibration.txt \
  --num_samples 128 \
  --seq_length 512
```

Run causal score:

```bash
python analysis/compute_causal_score.py \
  --model_path ckpt/path/to/ckpt \
  --output_dir analysis/results/causal_score \
  --text-file data/calibration/calibration.txt \
  --num_samples 128 \
  --seq_length 512
```

Run permutation score:

```bash
python analysis/compute_permutation_score.py \
  --model_path ckpt/path/to/ckpt \
  --output_dir analysis/results/permutation_score \
  --text-file data/calibration/calibration.txt \
  --num_samples 128 \
  --seq_length 512
```

Run usefulness score:

```bash
python analysis/compute_usefulness_score.py \
  --model_path ckpt/path/to/ckpt \
  --output_dir analysis/results/usefulness_score \
  --text-file data/calibration/calibration.txt \
  --num_samples 1024 \
  --seq_length 512
```

## Downstream Evaluation

### Supervised Finetuning

This repo includes a minimal OLMo-core based SFT pipeline for finetuning a pre-train checkpoint
on `Commonsense170K`.

The main entrypoints are:

- [`eval/finetune/prepare_commonsense170k.py`](./eval/finetune/prepare_commonsense170k.py) for dataset preparation
- [`eval/finetune/sft_llama_base.py`](./eval/finetune/sft_llama_base.py) for training

Prepare the dataset with:

```bash
python3 eval/finetune/prepare_commonsense170k.py \
  --output-dir data/commonsense-170k-olmocore \
  --tokenizer-name-or-path ./pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
  --max-seq-len 512 \
  --part-size 1000000 \
  --seed 42
```

Common options:

- `--dataset-name`, default `zwhe99/commonsense_170k`
- `--dataset-split`, default `train`
- `--max-samples`, use a subset for quick debugging

The dataset is written as flat `token_ids_part_*.npy` and `labels_mask_part_*.npy` files. The
supervised loss is applied only on the response span and the final EOS token.

The prompt template is:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

If `input` is non-empty, an additional `### Input:` section is inserted.

Launch SFT with:

```bash
torchrun --nproc_per_node=8 eval/finetune/sft_llama_base.py \
  --run-name llama-1B-commonsense170k-sft \
  --model-config ./configs/llama_1B_backbone.json \
  --pretrain-checkpoint ckpt/depthbench/pretrain-llama-1B-lr5e-4 \
  --dataset-dir data/commonsense-170k-olmocore \
  --save-folder ckpt/depthbench/llama-1B-commonsense170k-sft \
  --tokenizer-name-or-path ./pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
  --learning-rate 5e-5 \
  --dataset-layout padded \
  --disable-compile
```

Common options:

- `--sequence-length`, default `512`
- `--epochs`, default `3`
- `--global-train-batch-size`, default `128`
- `--device-train-microbatch-size`, default `16`
- `--dataset-layout {packed,padded}`, default `padded`

Typically, we use about 1/10 of the pre-train learning rate for SFT, or do a small sweep around
that value.

### Zero-shot Evaluation

This repo supports zero-shot downstream evaluation directly from native `OLMo-core` checkpoints,
without converting the checkpoint to Hugging Face format first.

The current zero-shot task set is:

- `openbookqa`
- `winogrande`
- `arc_challenge`
- `arc_easy`
- `hellaswag`
- `social_iqa`
- `piqa`

The entrypoint is [`eval/run_zero_shot.py`](./eval/run_zero_shot.py), which uses:

- [`eval/olmo_lm.py`](./eval/olmo_lm.py) as the adapter from native `OLMo-core` checkpoints to `lm-eval-harness`
- the vendored [`eval/lm-evaluation-harness`](./eval/lm-evaluation-harness) task definitions

Run zero-shot evaluation with:

```bash
python eval/run_zero_shot.py \
  /path/to/checkpoint/step2600 \
  --device cuda:0 \
  --batch-size 32 \
  --attention-backend torch \
  --output-path /path/to/output/zero_shot_results.json
```
