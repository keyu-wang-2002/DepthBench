#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${OMP_NUM_THREADS:-4}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

WORKER_ID="${1:?worker id is required}"
NUM_WORKERS="${2:?num workers is required}"

export TRAIN_WORKER_ID="${WORKER_ID}"
export TRAIN_NUM_WORKERS="${NUM_WORKERS}"

EXTRA_ARGS=()
if [[ "${WORKER_ID}" != "0" ]]; then
    EXTRA_ARGS+=(--skip-eval)
fi

echo "Starting FineWeb-Edu tokenization worker ${WORKER_ID}/${NUM_WORKERS}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAIN_WORKER_ID=${TRAIN_WORKER_ID}"
echo "TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS}"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/tokenize_fineweb_edu_100bt.sh" "${EXTRA_ARGS[@]}"
