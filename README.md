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


## Training Script

Example:

```bash
cd ./examples
bash pretrain_llama_130M_base.sh
bash pretrain_llama_250M_base.sh
bash pretrain_llama_350M_base.sh
bash pretrain_llama_1B_base.sh
```

