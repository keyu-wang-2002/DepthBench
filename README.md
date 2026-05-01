# DepthBench



## Environment Setup

```bash
python -m venv depthbench
source depthbench/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

cd DepthBench/pretrain/OLMo-core
python -m pip install -e ".[wandb,transformers]"
python -m pip install datasets pyarrow
pip install torch transformers numpy tqdm matplotlib seaborn
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
| 60M | 512 | 1376 | 8 | 8 | 1.5B | 512 | 2048 | 1.4k |
| 130M | 768 | 2048 | 12 | 12 | 2.7B | 512 | 2048 | 2.6k |
| 250M | 896 | 2560 | 14 | 16 | 5.2B | 512 | 2048 | 5.0k |
| 350M | 1024 | 2736 | 16 | 24 | 7.3B | 512 | 2048 | 7.0k |
| 1B | 2048 | 5461 | 32 | 24 | 21.0B | 512 | 2048 | 20.0k |

### 350M Aspect-Ratio Variants

The following 350M-family configs keep `#heads` and `intermediate_size/d_model = 8/3` fixed while varying only `d_model` and `n_layer` to probe depth/width scaling at roughly the same parameter budget. The training configs follow the standard backbone.

| Tier | Hidden | Intermediate | Heads | Layers | Head Dim | Aspect Ratio (`d_model / n_layer`) |
|---|---:|---:|---:|---:|---:|---:|
| Ultra shallow-350M | 2240 | 5984 | 16 | 4 | 140 | 560.00 |
| Very shallow-350M | 1680 | 4480 | 16 | 8 | 105 | 210.00 |
| Shallow-350M | 1232 | 3296 | 16 | 16 | 77 | 77.00 |
| Standard-350M | 1024 | 2736 | 16 | 24 | 64 | 42.67 |
| Deeper-350M | 848 | 2272 | 16 | 36 | 53 | 23.56 |
| Deep-350M | 752 | 2016 | 16 | 46 | 47 | 16.35 |
| Very deep-350M | 688 | 1840 | 16 | 56 | 43 | 12.29 |
| Extreme deep-350M | 512 | 1376 | 16 | 104 | 32 | 4.92 |

## Training Script

Example:

```bash
cd ./examples
bash pretrain_llama_130M_base.sh
bash pretrain_llama_250M_base.sh
bash pretrain_llama_350M_base.sh
bash pretrain_llama_1B_base.sh
```

Note: DepthBench now supports per-layer monitoring of hidden-state statistics during pretraining. For each transformer block, we record statistics for:

- `forward`: the block output hidden state
- `backward`: the activation gradient on the same hidden state

For both directions, the following statistics are logged:

- `mean`
- `variance`
- `magnitude = abs().mean()`
- `norm = l2_norm`

This adds two CLI flags:

- `--enable-layer-stats`
- `--layer-stats-interval 1` means record every step. Set it to a larger value to reduce logging overhead.

Metric names follow this pattern:

```text
train/layer_stats/block_00/forward/mean
train/layer_stats/block_00/forward/variance
train/layer_stats/block_00/forward/magnitude
train/layer_stats/block_00/forward/norm
train/layer_stats/block_00/backward/mean
...
```

These metrics are automatically logged to W&B when W&B is enabled. W&B may create many charts because every block and every statistic is logged separately. A convenient way to view them is to create multi-metric panels with regex, for example:

```text
^train/layer_stats/block_\d+/forward/norm$
^train/layer_stats/block_\d+/backward/norm$
```

Add swanlab: TODO


## Analysis

This directory contains scripts for running DepthBench analysis metrics on either:

- a native `OLMo-core` checkpoint via `--model-backend olmo_core`
- a Hugging Face model directory via `--model-backend hf`

I recommend to directly use `OLMo-core` checkpoint via `--model-backend olmo_core`.  There are some potential risks when first converting `OLMo-core` checkpoint to `hf` checkpoint and then using `--model-backend hf` for analysis, refering issue: https://github.com/pUmpKin-Co/SparsityAndCoD/issues/2


Run angular distance:

```bash
python "${PROJECT_ROOT}/analysis/compute_angular_distance.py" \
  --model_path "${CHECKPOINT_DIR}" \
  --model-backend olmo_core \
  --output_dir "${RUN_ROOT}/angular_distance" \
  --text-file "${CALIBRATION_TEXT}" \
  --num_samples 1024 \
  --seq_length 512
```

Run Jacobian analysis:

```bash
python "${PROJECT_ROOT}/analysis/compute_jacobian.py" \
  --model_path "${CHECKPOINT_DIR}" \
  --model-backend olmo_core \
  --output_dir "${RUN_ROOT}/jacobian" \
  --text-file "${CALIBRATION_TEXT}" \
  --num_samples 128 \
  --seq_length 512
```

Run causal score:

```bash
python "${PROJECT_ROOT}/analysis/compute_casual_score.py" \
  --model_path "${CHECKPOINT_DIR}" \
  --model-backend olmo_core \
  --output_dir "${RUN_ROOT}/casual_score" \
  --text-file "${CALIBRATION_TEXT}" \
  --num_samples 128 \
  --seq_length 512
```

Run permutation score:

```bash
python "${PROJECT_ROOT}/analysis/compute_permutation_score.py" \
  --model_path "${CHECKPOINT_DIR}" \
  --model-backend olmo_core \
  --output_dir "${RUN_ROOT}/permutation_score" \
  --text-file "${CALIBRATION_TEXT}" \
  --num_samples 128 \
  --seq_length 512
```

Run usefulness score:

```bash
python "${PROJECT_ROOT}/analysis/compute_usefulness_score.py" \
  --model_path "${CHECKPOINT_DIR}" \
  --model-backend olmo_core \
  --output_dir "${RUN_ROOT}/usefulness_score" \
  --text-file "${CALIBRATION_TEXT}" \
  --num_samples 1024 \
  --seq_length 512
```

TODO: add downstream 
