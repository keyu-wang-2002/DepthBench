#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/scripts/load_local_env.sh" ]]; then
    source "${REPO_ROOT}/scripts/load_local_env.sh"
fi

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

torchrun \
    --nproc_per_node="${NPROC_PER_NODE:-1}" \
    --master_port="${MASTER_PORT:-35109}" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    "${SCRIPT_DIR}/pretrain_llama_base.py" \
    --run_name="${RUN_NAME:-debug-llama-60M-single-gpu}" \
    --model-config="${MODEL_CONFIG:-${REPO_ROOT}/configs/llama_60M_backbone.json}" \
    --tokenizer-name-or-path="${TOKENIZER_PATH:-${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json}" \
    --train-data-glob="${TRAIN_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/train/*.npy}" \
    --eval-data-glob="${EVAL_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/eval/*.npy}" \
    --sequence-length="${SEQUENCE_LENGTH:-256}" \
    --seed="${SEED:-42}" \
    --max-steps="${MAX_STEPS:-5}" \
    --global-train-batch-size="${GLOBAL_TRAIN_BATCH_SIZE:-8}" \
    --device-train-microbatch-size="${DEVICE_TRAIN_MICROBATCH_SIZE:-1}" \
    --data-loader-num-workers="${DATA_LOADER_NUM_WORKERS:-2}" \
    --learning-rate="${LEARNING_RATE:-1e-3}" \
    --warmup-steps="${WARMUP_STEPS:-1}" \
    --eval-interval="${EVAL_INTERVAL:-5}" \
    --eval-max-batches="${EVAL_MAX_BATCHES:-1}" \
    --save-interval="${SAVE_INTERVAL:-5}" \
    --save-folder="${SAVE_FOLDER:-${REPO_ROOT}/workspace/debug-llama-60M-single-gpu}" \
    --wandb-project="${WANDB_PROJECT:-}" \
    --wandb-entity="${WANDB_ENTITY:-}" \
    --swanlab-project="${SWANLAB_PROJECT:-}" \
    --swanlab-workspace="${SWANLAB_WORKSPACE:-}" \
    --swanlab-group="${SWANLAB_GROUP:-}" \
    --swanlab-description="${SWANLAB_DESCRIPTION:-}" \
    --swanlab-mode="${SWANLAB_MODE:-}" \
    --swanlab-tags ${SWANLAB_TAGS:-} \
    --enable-layer-stats \
    --layer-stats-interval "${LAYER_STATS_INTERVAL:-1}" \
    "$@"
