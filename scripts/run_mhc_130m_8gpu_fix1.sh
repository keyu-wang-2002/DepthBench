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

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export RUN_NAME="${RUN_NAME:-pretrain-llama-130M-mhc-s4-lr1e-3-mb8-fix1}"
export SAVE_FOLDER="${SAVE_FOLDER:-${REPO_ROOT}/ckpt/depthbench/pretrain-llama-130M-mhc-s4-lr1e-3-mb8-fix1}"
export SWANLAB_GROUP="${SWANLAB_GROUP:-mhc}"
export SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-DepthBench 130M mHC rerun with fixed residual init and microbatch 8}"
export SWANLAB_TAGS="${SWANLAB_TAGS:-130M mhc 8gpu fineweb-edu-100bt manifold-constrained-hyper-connections mb8 fix1}"

echo "Starting 130M mHC training rerun"
echo "REPO_ROOT=${REPO_ROOT}"
echo "RUN_NAME=${RUN_NAME}"
echo "SAVE_FOLDER=${SAVE_FOLDER}"
echo "SWANLAB_PROJECT=${SWANLAB_PROJECT:-}"
echo "SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}"
echo "SWANLAB_GROUP=${SWANLAB_GROUP:-}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"

python - <<'PY'
import os
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
print("SWANLAB enabled:", bool(os.environ.get("SWANLAB_PROJECT")))
PY

cd "${REPO_ROOT}/examples"
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-35101}" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    "${REPO_ROOT}/examples/pretrain_llama_base.py" \
    --run_name="${RUN_NAME}" \
    --model-config="${MODEL_CONFIG:-${REPO_ROOT}/configs/llama_130M_mhc.json}" \
    --tokenizer-name-or-path="${TOKENIZER_PATH:-${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json}" \
    --train-data-glob="${TRAIN_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/train/*.npy}" \
    --eval-data-glob="${EVAL_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/eval/*.npy}" \
    --seed="${SEED:-42}" \
    --max-steps="${MAX_STEPS:-2600}" \
    --global-train-batch-size="${GLOBAL_TRAIN_BATCH_SIZE:-512}" \
    --device-train-microbatch-size="${DEVICE_TRAIN_MICROBATCH_SIZE:-8}" \
    --learning-rate="${LEARNING_RATE:-1e-3}" \
    --warmup-steps="${WARMUP_STEPS:-260}" \
    --eval-interval="${EVAL_INTERVAL:-200}" \
    --save-interval="${SAVE_INTERVAL:-1000}" \
    --save-folder="${SAVE_FOLDER}" \
    --wandb-project="${WANDB_PROJECT:-}" \
    --wandb-entity="${WANDB_ENTITY:-}" \
    --swanlab-project="${SWANLAB_PROJECT:-}" \
    --swanlab-workspace="${SWANLAB_WORKSPACE:-}" \
    --swanlab-group="${SWANLAB_GROUP:-}" \
    --swanlab-description="${SWANLAB_DESCRIPTION:-}" \
    --swanlab-mode="${SWANLAB_MODE:-}" \
    --swanlab-tags ${SWANLAB_TAGS:-} \
    --enable-layer-stats \
    --layer-stats-interval "${LAYER_STATS_INTERVAL:-1}"
