#!/usr/bin/env bash
# run_token_collapse.sh — per-layer token effective rank for a single DepthBench checkpoint.
#
# Usage:
#   bash scripts/david/run_token_collapse.sh <cfg_tag> <ckpt_path> <job_tag>

set -euo pipefail

CFG_TAG="${1:?missing cfg_tag (e.g. 16l)}"
CKPT_PATH="${2:?missing ckpt_path}"
JOB_TAG="${3:-local}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate depthbench

export HF_HOME="${HF_HOME:-/home/dmartinez/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}/analysis:${REPO_ROOT}/data_utils:${PYTHONPATH:-}"

CALIBRATION_TXT="/fast/dmartinez/depthbench/data/calibration/calibration.txt"
OUTPUT_DIR="/fast/dmartinez/depthbench/analysis/token_collapse/${CFG_TAG}"
mkdir -p "${OUTPUT_DIR}"

# Build calibration data if not already present (only runs once).
if [[ ! -f "${CALIBRATION_TXT}" ]]; then
  echo "Calibration data not found — building now..."
  mkdir -p "$(dirname "${CALIBRATION_TXT}")"
  python data_utils/build_calibration_data.py \
    --source c4 \
    --source dolma \
    --output-dir "$(dirname "${CALIBRATION_TXT}")" \
    --output-prefix calibration \
    --tokenizer-name-or-path "${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --target-total-tokens 262144 \
    --sample-length-mode fixed \
    --sample-length 512 \
    --shuffle-samples
  echo "Calibration data built at $(dirname "${CALIBRATION_TXT}")"
fi

echo "============================================================"
echo "  DepthBench token collapse (effective rank)"
echo "  cfg_tag:   ${CFG_TAG}"
echo "  ckpt:      ${CKPT_PATH}"
echo "  output:    ${OUTPUT_DIR}"
echo "  calib:     ${CALIBRATION_TXT}"
echo "============================================================"

TOKENIZER_PATH="${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"

python analysis/compute_token_collapse.py \
  --model_path "${CKPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --text-file "${CALIBRATION_TXT}" \
  --num_samples 1024 \
  --seq_length 512 \
  --device cuda:0 \
  --dtype auto \
  --tokenizer-id "${TOKENIZER_PATH}"

echo "Done ${CFG_TAG} ${JOB_TAG}"
