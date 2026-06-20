#!/usr/bin/env bash
# run_nll_eval.sh — NLL evaluation (GSM8K, MATH-500, MBPP) for a single DepthBench checkpoint.
#
# Usage:
#   bash scripts/david/run_nll_eval.sh <cfg_tag> <ckpt_path> <job_tag>
#
# Example:
#   bash scripts/david/run_nll_eval.sh 16l \
#     /fast/dmartinez/depthbench/runs/pretrain-llama_350M_16l_keel-lr2e-3-s42-17309939_0/step7600 \
#     local

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

OUTPUT_DIR="${REPO_ROOT}/scripts/david/results_nll/${CFG_TAG}"
mkdir -p "${OUTPUT_DIR}"

OUTPUT_PATH="${OUTPUT_DIR}/results_${JOB_TAG}.json"

echo "============================================================"
echo "  DepthBench NLL eval"
echo "  cfg_tag:   ${CFG_TAG}"
echo "  ckpt:      ${CKPT_PATH}"
echo "  output:    ${OUTPUT_PATH}"
echo "============================================================"

python eval/run_nll_eval.py \
  "${CKPT_PATH}" \
  --device cuda:0 \
  --batch-size 8 \
  --attention-backend torch \
  --output-path "${OUTPUT_PATH}"

echo "Done ${CFG_TAG} ${JOB_TAG}"
