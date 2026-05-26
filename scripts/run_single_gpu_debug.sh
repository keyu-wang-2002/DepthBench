#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${SCRIPT_DIR}/install_env.sh"

SETUP_ARGS=()
if [[ "${FULL_DATASET:-0}" == "1" ]]; then
    :
else
    SETUP_ARGS+=(
        --allow-pattern "sample/100BT/${DEBUG_TRAIN_SHARD:-000_00000.parquet}"
        --allow-pattern "sample/100BT/${DEBUG_EVAL_SHARD:-013_00008.parquet}"
    )
    export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize-debug}"
    export LIMIT_TRAIN_FILES="${LIMIT_TRAIN_FILES:-1}"
    export MAX_DOCUMENTS_PER_FILE="${MAX_DOCUMENTS_PER_FILE:-4096}"
    export TRAIN_DATA_GLOB="${TRAIN_DATA_GLOB:-${OUTPUT_DIR}/train/*.npy}"
    export EVAL_DATA_GLOB="${EVAL_DATA_GLOB:-${OUTPUT_DIR}/eval/*.npy}"
fi

python "${SCRIPT_DIR}/setup_fineweb_edu_100bt.py" "${SETUP_ARGS[@]}"
bash "${SCRIPT_DIR}/tokenize_fineweb_edu_100bt.sh"
bash "${REPO_ROOT}/examples/pretrain_llama_60M_minimal_debug.sh"
