#!/bin/bash
source /fast/wangk/virtual_env/depthbench/bin/activate

export WANDB_API_KEY="e440e1ebca8f7e09447c3a5b1f7003f8a007691f"

torchrun --nproc_per_node=8 --master_port=35101 --master_addr=localhost "/home/wangk/DepthBench/examples/pretrain_llama_base.py" \
    --run_name=pretrain-llama-130M-lr1e-3 \
    --model-config="/home/wangk/DepthBench/configs/llama_130M_backbone.json" \
    --tokenizer-name-or-path="/home/wangk/DepthBench/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=2600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=1e-3 \
    --warmup-steps=260 \
    --eval-interval=200 \
    --save-interval=1000 \
    --save-folder="/fast/wangk/ckpt/depthbench/pretrain-llama-130M-lr1e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1 
