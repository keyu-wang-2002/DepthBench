#!/bin/bash

set -euo pipefail

source $HOME/depth.sh

cd ${lspace}

source .venv/bin/activate

torchrun --nproc_per_node=4 --master_port=35100 --master_addr=localhost "examples/pretrain_llama_base.py" \
    --run_name=pretrain-llama-350M-lr3e-4 \
    --model-config=${lspace}"/configs/llama_350M_attnres.json" \
    --tokenizer-name-or-path=${lspace}/"pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=250 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=3e-4 \
    --warmup-steps=25 \
    --eval-interval=50 \
    --save-interval=250 \
    --save-folder="${wspace}/results/debug/attnres/pretrain-llama-350M-lr3e-4" \
    --wandb-project="depthbench"
