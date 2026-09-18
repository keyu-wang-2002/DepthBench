#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:-hc}"
SHAPE="${SHAPE:-L24}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
LR_TAG="${LR_TAG:-${LEARNING_RATE//-/}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
MAX_STEPS="${MAX_STEPS:-7600}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${MAX_STEPS}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-3000}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-data/fineweb-edu/pre-tokenize}"
SAVE_ROOT="${SAVE_ROOT:-ckpt/depthbench}"

case "${SHAPE}" in
    L16|L20|L24)
        DEFAULT_DEVICE_MICROBATCH_SIZE=8
        ;;
    L26|L28|L30|L32|L36|L42|L50|L56|L62|L70)
        DEFAULT_DEVICE_MICROBATCH_SIZE=4
        ;;
    *)
        echo "Unsupported SHAPE=${SHAPE}; expected a 400M shape in configs/ (L16 through L70)" >&2
        exit 2
        ;;
esac

case "${METHOD}" in
    hc)
        ENTRYPOINT=examples/pretrain_hc.py
        EXTRA_ARGS=()
        DEFAULT_RUN_NAME="pretrain-${METHOD}-400M-${SHAPE}-lr${LR_TAG}"
        ;;
    mhc)
        ENTRYPOINT=examples/pretrain_mhc.py
        EXTRA_ARGS=("--mhc-backend=${MHC_BACKEND:-liger_mhc}")
        DEFAULT_RUN_NAME="pretrain-${METHOD}-${MHC_BACKEND:-liger_mhc}-400M-${SHAPE}-lr${LR_TAG}"
        if [[ "${MHC_BACKEND:-liger_mhc}" != "mhc_static" ]]; then
            DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}-gap8"
        fi
        ;;
    *)
        echo "Unsupported METHOD=${METHOD}; expected hc or mhc" >&2
        exit 2
        ;;
esac

DEVICE_MICROBATCH_SIZE="${DEVICE_MICROBATCH_SIZE:-${DEFAULT_DEVICE_MICROBATCH_SIZE}}"
if ((GLOBAL_BATCH_SIZE % (NPROC_PER_NODE * DEVICE_MICROBATCH_SIZE) != 0)); then
    echo "Global batch must be divisible by world size * device microbatch size" >&2
    exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (NPROC_PER_NODE * DEVICE_MICROBATCH_SIZE)))

MODEL_CONFIG="configs/llama_400m_${SHAPE}.json"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

echo "method=${METHOD} shape=${SHAPE} learning_rate=${LEARNING_RATE}"
echo "world_size=${NPROC_PER_NODE} global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "device_microbatch_size=${DEVICE_MICROBATCH_SIZE} gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
echo "eval_interval=${EVAL_INTERVAL} save_interval=${SAVE_INTERVAL}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr=localhost \
    --master_port="${MASTER_PORT:-35100}" \
    "${ENTRYPOINT}" \
    "${EXTRA_ARGS[@]}" \
    --run_name="${RUN_NAME}" \
    --model-config="${MODEL_CONFIG}" \
    --tokenizer-name-or-path="pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json" \
    --train-data-glob="${DATA_ROOT}/train/*.npy" \
    --eval-data-glob="${DATA_ROOT}/eval/*.npy" \
    --seed="${SEED}" \
    --sequence-length=2048 \
    --max-steps="${MAX_STEPS}" \
    --global-train-batch-size="${GLOBAL_BATCH_SIZE}" \
    --device-train-microbatch-size="${DEVICE_MICROBATCH_SIZE}" \
    --learning-rate="${LEARNING_RATE}" \
    --warmup-steps=$((MAX_STEPS / 10)) \
    --max-grad-norm=1.0 \
    --eval-interval="${EVAL_INTERVAL}" \
    --save-interval="${SAVE_INTERVAL}" \
    --save-folder="${SAVE_ROOT}/${RUN_NAME}" \
    --wandb-project="${WANDB_PROJECT:-}"
