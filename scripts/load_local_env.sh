#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

read_token_file() {
    local token_path="$1"
    if [[ -f "${token_path}" ]]; then
        tr -d '\r\n' < "${token_path}"
    fi
}

if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_VALUE="$(read_token_file "${REPO_ROOT}/hf_token.txt")"
    if [[ -z "${HF_TOKEN_VALUE}" ]]; then
        HF_TOKEN_VALUE="$(read_token_file "${WORKSPACE_ROOT}/hf_token.txt")"
    fi
    if [[ -n "${HF_TOKEN_VALUE}" ]]; then
        export HF_TOKEN="${HF_TOKEN_VALUE}"
        export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN_VALUE}"
    fi
    unset HF_TOKEN_VALUE
fi

export SWANLAB_PROJECT="${SWANLAB_PROJECT:-depth_scaling}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-depth_scaling}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
    SWANLAB_API_KEY_VALUE="$(read_token_file "${REPO_ROOT}/swanlab_token.txt")"
    if [[ -z "${SWANLAB_API_KEY_VALUE}" ]]; then
        SWANLAB_API_KEY_VALUE="$(read_token_file "${WORKSPACE_ROOT}/swanlab_token.txt")"
    fi
    if [[ -n "${SWANLAB_API_KEY_VALUE}" ]]; then
        export SWANLAB_API_KEY="${SWANLAB_API_KEY_VALUE}"
    fi
    unset SWANLAB_API_KEY_VALUE
fi

unset -f read_token_file
