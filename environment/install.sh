#!/usr/bin/env bash
# Create a NEW environment; never modify an environment used by running jobs.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# != 2 ]]; then
    echo "Usage: PYTHON_BIN=python3.10 bash $0 /path/new-venv /path/DepthBench" >&2
    exit 2
fi
VENV=$1
REPO=$2
PYTHON_BIN=${PYTHON_BIN:-python3.10}
if [[ -e "${VENV}" ]]; then
    echo "Refusing to modify existing environment: ${VENV}" >&2
    exit 1
fi
if [[ ! -f "${REPO}/pretrain/OLMo-core/pyproject.toml" ]]; then
    echo "Not a DepthBench checkout: ${REPO}" >&2
    exit 1
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON_BIN}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --disable-pip-version-check -r "${HERE}/requirements-cu128.lock.txt"
"${VENV}/bin/python" -m pip install --no-deps --no-build-isolation -e "${REPO}/pretrain/OLMo-core"
"${VENV}/bin/python" -m pip install --no-deps --no-compile \
    --target "${VENV}/profiles/attnres-fla-0.4.1" -r "${HERE}/requirements-attnres.txt"
"${VENV}/bin/python" -m pip check
echo "Installed. Activate ${VENV}/bin/activate and select dependencies as described in ${HERE}/README.md."
echo "MoDA additionally requires the pinned source checkout and included kernel patch."
