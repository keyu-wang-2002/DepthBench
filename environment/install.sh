#!/usr/bin/env bash
# Create a NEW environment; never modify an environment used by running jobs.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
    printf '%s\n' \
        "Usage: [PYTHON_BIN=python3.10] bash $0 [VENV_PATH [REPO_PATH]]" \
        "Defaults: repository containing this script; VENV_PATH=REPO_PATH/.venv-cu128" \
        "Installs pinned cu128 packages, local OLMo-core and an isolated AttnRes overlay." \
        "MoDA is NOT installed; see environment/README.md for separate setup."
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -gt 2 ]]; then
    usage >&2
    exit 2
fi
REPO=${2:-$(cd -- "${HERE}/.." && pwd)}
VENV=${1:-${REPO}/.venv-cu128}
PYTHON_BIN=${PYTHON_BIN:-python3.10}
if [[ -e "${VENV}" || -L "${VENV}" ]]; then
    echo "Refusing to modify existing environment: ${VENV}" >&2
    exit 1
fi
if [[ ! -f "${REPO}/pretrain/OLMo-core/pyproject.toml" ]]; then
    echo "Not a DepthBench checkout: ${REPO}" >&2
    exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python 3.10 is required. Set PYTHON_BIN=/path/to/python3.10." >&2
    exit 1
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[1/5] Creating environment: ${VENV}"
"${PYTHON_BIN}" -m venv "${VENV}"
echo "[2/5] Installing pinned cu128 packages (including Liger mHC)"
"${VENV}/bin/python" -m pip install --disable-pip-version-check -r "${HERE}/requirements-cu128.lock.txt"
echo "[3/5] Installing OLMo-core from ${REPO}"
"${VENV}/bin/python" -m pip install --no-deps --no-build-isolation -e "${REPO}/pretrain/OLMo-core"
echo "[4/5] Installing isolated AttnRes FLA dependencies"
"${VENV}/bin/python" -m pip install --no-deps --no-compile \
    --target "${VENV}/profiles/attnres-fla-0.4.1" -r "${HERE}/requirements-attnres.txt"
echo "[5/5] Checking installed dependencies"
"${VENV}/bin/python" -m pip check
printf '\nInstalled. Activate with: source %q\n' "${VENV}/bin/activate"
echo "MoDA was not installed. See ${HERE}/README.md for separate MoDA setup and dependency selection."
