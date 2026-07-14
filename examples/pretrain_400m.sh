#!/bin/bash

# Hyper-Connections (HC)
torchrun --nproc_per_node=4 --master_port=35100 --master_addr=localhost "examples/pretrain_hc.py" \
    --run_name=pretrain-hc-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 --max-steps=7600 --global-train-batch-size=512 \
    --device-train-microbatch-size=8 --learning-rate=2e-3 --warmup-steps=760 \
    --eval-interval=200 --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-hc-400M-lr2e-3"

# Manifold-constrained Hyper-Connections (fused Liger backend)
torchrun --nproc_per_node=4 --master_port=35101 --master_addr=localhost "examples/pretrain_mhc.py" \
    --mhc-backend=liger_mhc \
    --run_name=pretrain-mhc-400M-lr2e-3 \
    --model-config="configs/llama_400m_L24.json" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --seed=42 --max-steps=7600 --global-train-batch-size=512 \
    --device-train-microbatch-size=8 --learning-rate=2e-3 --warmup-steps=760 \
    --eval-interval=200 --save-interval=3000 \
    --save-folder="ckpt/depthbench/pretrain-mhc-400M-lr2e-3"

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
