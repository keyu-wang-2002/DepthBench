#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/scripts/load_local_env.sh" ]]; then
    source "${REPO_ROOT}/scripts/load_local_env.sh"
fi

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
fi

MODA_KERNEL_DIR="${MODA_KERNEL_DIR:-/lustre/fast/fast/wliu/yy/DepthBench_workspace/MoDA/libs/moda_triton}"
if ! python -m pip show flash-linear-attention >/dev/null 2>&1; then
    echo "Installing MoDA Triton kernels from ${MODA_KERNEL_DIR}"
    python -m pip install -e "${MODA_KERNEL_DIR}"
fi

JOB_USER="${USER:-${LOGNAME:-wliu}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${JOB_USER}/xdg-cache-moda-${CONDOR_JOB_ID:-manual}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${JOB_USER}/torchinductor-moda-${CONDOR_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${JOB_USER}/triton-moda-${CONDOR_JOB_ID:-manual}}"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

METHOD="${METHOD:-prenorm_moda}"
LR_TAG="${LR_TAG:-lr2e3}"

case "${METHOD}" in
    prenorm_moda)
        DEFAULT_MODEL_CONFIG="${REPO_ROOT}/configs/llama_350M_moda_prenorm_l24_h1024_i2736.json"
        DEFAULT_GROUP="moda_prenorm"
        ;;
    postnorm_baseline)
        DEFAULT_MODEL_CONFIG="${REPO_ROOT}/configs/llama_350M_postnorm_l24_h1024_i2736.json"
        DEFAULT_GROUP="postnorm_baseline"
        ;;
    postnorm_moda)
        DEFAULT_MODEL_CONFIG="${REPO_ROOT}/configs/llama_350M_moda_postnorm_l24_h1024_i2736.json"
        DEFAULT_GROUP="moda_postnorm"
        ;;
    *)
        echo "Unknown METHOD=${METHOD}; expected prenorm_moda, postnorm_baseline, or postnorm_moda" >&2
        exit 2
        ;;
esac

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MODEL_CONFIG="${MODEL_CONFIG:-${DEFAULT_MODEL_CONFIG}}"
export LEARNING_RATE="${LEARNING_RATE:-2e-3}"
export DEVICE_TRAIN_MICROBATCH_SIZE="${DEVICE_TRAIN_MICROBATCH_SIZE:-2}"
export RUN_NAME="${RUN_NAME:-pretrain-llama-350M-${METHOD}-l24-${LR_TAG}}"
export SAVE_FOLDER="${SAVE_FOLDER:-${REPO_ROOT}/ckpt/depthbench/${RUN_NAME}}"
export SWANLAB_GROUP="${SWANLAB_GROUP:-${DEFAULT_GROUP}}"
export SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-DepthBench 350M L24 ${METHOD} on FineWeb-Edu 100BT pre-tokenized data}"
export SWANLAB_TAGS="${SWANLAB_TAGS:-350M L24 ${METHOD} moda h100 4gpu fineweb-edu-100bt}"

echo "Starting 350M L24 MoDA-family training"
echo "REPO_ROOT=${REPO_ROOT}"
echo "METHOD=${METHOD}"
echo "RUN_NAME=${RUN_NAME}"
echo "SAVE_FOLDER=${SAVE_FOLDER}"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "VENV_DIR=${VENV_DIR}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "DEVICE_TRAIN_MICROBATCH_SIZE=${DEVICE_TRAIN_MICROBATCH_SIZE}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT:-35100}"

python - <<'PY'
import subprocess
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
kernel = subprocess.run(
    ["python", "-m", "pip", "show", "flash-linear-attention"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print("MoDA kernel package available:", kernel.returncode == 0)
PY

cd "${REPO_ROOT}/examples"
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-35100}" \
    --master_addr="${MASTER_ADDR:-localhost}" \
    "${REPO_ROOT}/examples/pretrain_llama_base.py" \
    --run_name="${RUN_NAME}" \
    --model-config="${MODEL_CONFIG}" \
    --tokenizer-name-or-path="${TOKENIZER_PATH:-${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json}" \
    --train-data-glob="${TRAIN_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/train/*.npy}" \
    --eval-data-glob="${EVAL_DATA_GLOB:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize/eval/*.npy}" \
    --seed="${SEED:-42}" \
    --max-steps="${MAX_STEPS:-7600}" \
    --global-train-batch-size="${GLOBAL_TRAIN_BATCH_SIZE:-512}" \
    --device-train-microbatch-size="${DEVICE_TRAIN_MICROBATCH_SIZE}" \
    --learning-rate="${LEARNING_RATE}" \
    --warmup-steps="${WARMUP_STEPS:-760}" \
    --eval-interval="${EVAL_INTERVAL:-200}" \
    --save-interval="${SAVE_INTERVAL:-3000}" \
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
