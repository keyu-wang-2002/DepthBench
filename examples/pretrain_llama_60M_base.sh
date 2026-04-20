#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost examples/pretrain_llama_60M_base.py \
    --learning-rate=5e-4 \
    --save_folder=workspace/checkpoint/llama_60M_base
