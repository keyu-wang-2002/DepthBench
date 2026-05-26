#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.venv/bin/activate"

export TRAIN_DATA_GLOB="${TRAIN_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize-debug/train/*.npy}"
export EVAL_DATA_GLOB="${EVAL_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize-debug/eval/*.npy}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export RUN_NAME="${RUN_NAME:-debug-llama-60M-single-gpu-condor}"
export SAVE_FOLDER="${SAVE_FOLDER:-${REPO_ROOT}/workspace/debug-llama-60M-single-gpu-condor}"

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0))"

bash "${REPO_ROOT}/examples/pretrain_llama_60M_minimal_debug.sh"
