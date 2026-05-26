#!/bin/bash

torchrun --nproc_per_node=4 --master_port=35100 --master_addr=localhost "examples/pretrain_llama_base.py" \
    --run_name=pretrain-llama-350M-lr3e-4 \
    --model-config="configs/llama_350M_attnres.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=3e-4 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-llama-350M-lr3e-4" \
    --enable-layer-stats \
    --layer-stats-interval 1 \
    --wandb-project="depthbench"
