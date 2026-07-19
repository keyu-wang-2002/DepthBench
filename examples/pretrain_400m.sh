#!/bin/bash

# Pre-norm MoDA
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_moda.py" \
    --run_name=pretrain-prenorm-moda-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 --max-steps=7600 --global-train-batch-size=512 \
    --device-train-microbatch-size=4 --learning-rate=2e-3 --warmup-steps=760 \
    --eval-interval=7600 --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-prenorm-moda-400M-lr2e-3"

# Post-norm baseline and paper-aligned post-norm MoDA use the same common arguments.
torchrun --nproc_per_node=8 --master_port=35101 --master_addr=localhost "examples/pretrain_postnorm.py" \
    --run_name=pretrain-postnorm-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 --max-steps=7600 --global-train-batch-size=512 \
    --device-train-microbatch-size=16 --learning-rate=2e-3 --warmup-steps=760 \
    --eval-interval=7600 --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-postnorm-400M-lr2e-3"

torchrun --nproc_per_node=8 --master_port=35102 --master_addr=localhost "examples/pretrain_postnorm_moda.py" \
    --run_name=pretrain-postnorm-moda-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 --max-steps=7600 --global-train-batch-size=512 \
    --device-train-microbatch-size=4 --learning-rate=2e-3 --warmup-steps=760 \
    --eval-interval=7600 --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-postnorm-moda-400M-lr2e-3"

# Pre-LN
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_preln.py" \
    --run_name=pretrain-preln-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-preln-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# Pre-LN + depth-only depth-muP
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_preln_mup.py" \
    --parameterization=depth-muP \
    --scaling-axis=depth-only \
    --base-depth=24 \
    --base-width=1024 \
    --run_name=pretrain-preln-depth-muP-depth-only-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-preln-depth-muP-depth-only-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# Pre-LN + depth-width depth-muP
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_preln_mup.py" \
    --parameterization=depth-muP \
    --scaling-axis=depth-width \
    --base-depth=24 \
    --base-width=1024 \
    --run_name=pretrain-preln-depth-muP-depth-width-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-preln-depth-muP-depth-width-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# Pre-LN + depth-only CompleteP
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_preln_mup.py" \
    --parameterization=completeP \
    --scaling-axis=depth-only \
    --base-depth=24 \
    --base-width=1024 \
    --run_name=pretrain-preln-completeP-depth-only-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-preln-completeP-depth-only-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# Pre-LN + depth-width CompleteP
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_preln_mup.py" \
    --parameterization=completeP \
    --scaling-axis=depth-width \
    --base-depth=24 \
    --base-width=1024 \
    --run_name=pretrain-preln-completeP-depth-width-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-preln-completeP-depth-width-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# Peri-LN (Sandwich-LN)
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_periln.py" \
    --run_name=pretrain-periln-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-periln-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# LNS
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_lns.py" \
    --run_name=pretrain-lns-400M-lr1e-2 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=1e-2 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-lns-400M-lr1e-2" \
    --enable-layer-stats \
    --layer-stats-interval 1


# DeepNorm
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_deepnorm.py" \
    --run_name=pretrain-deepnorm-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-deepnorm-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1


# AttnRes
torchrun --nproc_per_node=8 --master_port=35100 --master_addr=localhost "examples/pretrain_attnres.py" \
    --run_name=pretrain-attnres-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 \
    --max-steps=7600 \
    --global-train-batch-size=512 \
    --device-train-microbatch-size=16 \
    --learning-rate=2e-3 \
    --warmup-steps=760 \
    --eval-interval=200 \
    --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-attnres-400M-lr2e-3" \
    --enable-layer-stats \
    --layer-stats-interval 1 \
    model.block.attnres_block_size=1
