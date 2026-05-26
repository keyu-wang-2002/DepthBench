#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

TOKENIZE_ARGS=(
    --train-parquet-glob "${TRAIN_PARQUET_GLOB:-${REPO_ROOT}/data/fineweb-edu/100BT/*.parquet}"
    --eval-parquet-path "${EVAL_PARQUET_PATH:-${REPO_ROOT}/data/fineweb-edu/eval/eval_013_00008.parquet}"
    --output-dir "${OUTPUT_DIR:-${REPO_ROOT}/data/fineweb-edu/pre-tokenize}"
    --text-field "${TEXT_FIELD:-text}"
    --tokenizer-name-or-path "${TOKENIZER_PATH:-${REPO_ROOT}/pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json}"
    --vocab-size "${VOCAB_SIZE:-50280}"
    --eos-token-id "${EOS_TOKEN_ID:-50279}"
    --pad-token-id "${PAD_TOKEN_ID:-1}"
    --batch-size "${BATCH_SIZE:-4096}"
    --progress-log-interval-docs "${PROGRESS_LOG_INTERVAL_DOCS:-8192}"
    --train-worker-id "${TRAIN_WORKER_ID:-0}"
    --train-num-workers "${TRAIN_NUM_WORKERS:-1}"
    --write-doc-indices
    --skip-existing
    --skip-summary
)

if [[ -n "${LIMIT_TRAIN_FILES:-}" ]]; then
    TOKENIZE_ARGS+=(--limit-train-files "${LIMIT_TRAIN_FILES}")
fi

if [[ -n "${MAX_DOCUMENTS_PER_FILE:-}" ]]; then
    TOKENIZE_ARGS+=(--max-documents-per-file "${MAX_DOCUMENTS_PER_FILE}")
fi

python "${REPO_ROOT}/data_utils/tokenize_from_pretrain_datasets.py" "${TOKENIZE_ARGS[@]}" "$@"
