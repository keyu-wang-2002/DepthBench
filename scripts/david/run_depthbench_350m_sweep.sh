#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/david/run_depthbench_350m_sweep.sh <lr> <cfg_json> <seed> <job_tag>
# Example:
#   ./scripts/david/run_depthbench_350m_sweep.sh 3e-4 configs/llama_350M_24l_backbone.json 42 1234_0

LR="${1:?missing lr}"
CFG="${2:?missing cfg path}"
SEED="${3:?missing seed}"
JOB_TAG="${4:-local}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate depthbench

# Keep compatibility with original example script defaults.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-35100}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"

CFG_BASENAME="$(basename "$CFG" .json)"
RUN_NAME="pretrain-${CFG_BASENAME}-lr${LR}-s${SEED}-${JOB_TAG}"
SAVE_FOLDER="/fast/dmartinez/depthbench/runs/${RUN_NAME}"

echo "[DepthBench] Starting run"
echo "  lr:        $LR"
echo "  cfg:       $CFG"
echo "  seed:      $SEED"
echo "  run_name:  $RUN_NAME"
echo "  save:      $SAVE_FOLDER"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  --master_addr="$MASTER_ADDR" \
  scripts/david/pretrain_llama_base.py \
  --run_name="$RUN_NAME" \
  --model-config="$CFG" \
  --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
  --seed="$SEED" \
  --max-steps=7600 \
  --global-train-batch-size=512 \
  --device-train-microbatch-size=16 \
  --learning-rate="$LR" \
  --warmup-steps=760 \
  --eval-interval=400 \
  --save-interval=30000000 \
  --save-folder="$SAVE_FOLDER" \
