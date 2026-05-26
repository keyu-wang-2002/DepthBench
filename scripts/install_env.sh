#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.pip-cache}"
PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in /usr/bin/python3.10 python3.10 python3 python; do
        if [[ -x "${candidate}" ]]; then
            PYTHON_BIN="${candidate}"
            break
        fi
        if command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "${candidate}")"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: could not find a usable Python interpreter." >&2
    exit 1
fi

echo "Using Python interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
export TMPDIR
export PIP_CACHE_DIR
export PIP_NO_CACHE_DIR

if [[ -d "${VENV_DIR}" && ! -f "${VENV_DIR}/bin/activate" ]]; then
    rm -rf "${VENV_DIR}"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if python -c "import datasets, matplotlib, olmo_core, pyarrow, seaborn, torch, transformers, wandb" >/dev/null 2>&1; then
    echo "Existing environment is ready at ${VENV_DIR}"
    exit 0
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e "${REPO_ROOT}/pretrain/OLMo-core[transformers,swanlab,wandb]"
python -m pip install datasets pyarrow huggingface_hub hf_xet matplotlib seaborn
