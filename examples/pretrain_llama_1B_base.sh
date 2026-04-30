#!/bin/bash
source depthbench/bin/activate

torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "pretrain_llama_base.py" \
    --run_name=pretrain-llama-1B-lr3e-4 \
    --model-config="../configs/llama_1B_backbone.json" \
    --tokenizer-name-or-path="../pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=20000 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=3e-4 \
    --warmup-steps=2000 \
    --eval-interval=200 \
    --save-interval=5000 \
    --save-folder="../ckpt/depthbench/pretrain-llama-1B-lr3e-4" \
    --enable-layer-stats \
    --layer-stats-interval 1 
