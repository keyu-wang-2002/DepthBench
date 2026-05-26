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
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --master_port="${MASTER_PORT:-35100}" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    "${SCRIPT_DIR}/pretrain_llama_base.py" \
    --run_name="${RUN_NAME:-pretrain-llama-350M-mhc-s4-lr3e-4}" \
    --model-config="${MODEL_CONFIG:-${REPO_ROOT}/configs/llama_350M_mhc.json}" \
    --tokenizer-name-or-path="${TOKENIZER_PATH:-${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json}" \
    --train-data-glob="${TRAIN_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/train/*.npy}" \
    --eval-data-glob="${EVAL_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/eval/*.npy}" \
    --seed="${SEED:-42}" \
    --max-steps="${MAX_STEPS:-7600}" \
    --global-train-batch-size="${GLOBAL_TRAIN_BATCH_SIZE:-512}" \
    --device-train-microbatch-size="${DEVICE_TRAIN_MICROBATCH_SIZE:-4}" \
    --learning-rate="${LEARNING_RATE:-3e-4}" \
    --warmup-steps="${WARMUP_STEPS:-760}" \
    --eval-interval="${EVAL_INTERVAL:-200}" \
    --save-interval="${SAVE_INTERVAL:-3000}" \
    --save-folder="${SAVE_FOLDER:-${REPO_ROOT}/ckpt/depthbench/pretrain-llama-350M-mhc-s4-lr3e-4}" \
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
