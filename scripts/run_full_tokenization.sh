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

echo "Starting full FineWeb-Edu 100BT tokenization"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/tokenize_fineweb_edu_100bt.sh"
