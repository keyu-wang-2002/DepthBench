#!/usr/bin/env bash
# Usage: source activate.sh PROFILE REPO VENV [FLA_SOURCE_OR_OVERLAY]
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file; do not execute it: source $0 PROFILE REPO VENV [DEPS]" >&2
    exit 2
fi

_depthbench_activate() {
    if [[ $# -lt 3 || $# -gt 4 ]]; then
        echo "Usage: source activate.sh {base|moda|attnres} REPO VENV [DEPS]" >&2
        return 2
    fi
    local profile=$1 repo=$2 venv=$3 deps=${4:-}
    [[ -d "${repo}/pretrain/OLMo-core/src/olmo_core" && -x "${venv}/bin/python" ]] || {
        echo "Missing DepthBench source or environment" >&2; return 1;
    }
    repo=$(cd -- "${repo}" && pwd)
    venv=$(cd -- "${venv}" && pwd)
    case "${profile}" in
        base) [[ -z "${deps}" ]] || { echo "base does not use a FLA overlay" >&2; return 1; } ;;
        moda)
            [[ -n "${deps}" && -f "${deps}/fla/ops/moda/moda_v17.py" ]] || {
                echo "moda requires the patched MoDA/libs/moda_triton directory" >&2; return 1;
            }
            ;;
        attnres)
            deps=${deps:-${venv}/profiles/attnres-fla-0.4.1}
            [[ -f "${deps}/fla/ops/utils/op.py" ]] || {
                echo "attnres requires its FLA 0.4.1 overlay, not the MoDA fork" >&2; return 1;
            }
            ;;
        *) echo "Unknown profile: ${profile}" >&2; return 2 ;;
    esac
    if [[ -n "${deps}" ]]; then
        deps=$(cd -- "${deps}" && pwd)
    fi
    export DEPTHBENCH_PROFILE=${profile} DEPTHBENCH_REPO=${repo} DEPTHBENCH_VENV=${venv}
    export DEPTHBENCH_FLA_ROOT=${deps} VIRTUAL_ENV=${venv}
    # Replace, rather than append, old source paths to avoid stale editable/FLA imports.
    export PYTHONPATH="${repo}/pretrain/OLMo-core/src:${repo}:${repo}/examples"
    [[ -z "${deps}" ]] || export PYTHONPATH="${deps}:${PYTHONPATH}"
    export PATH="${venv}/bin:${PATH}"
    if [[ -n "${DEPTHBENCH_TOOLCHAIN_BIN:-}" ]]; then
        export PATH="${DEPTHBENCH_TOOLCHAIN_BIN}:${PATH}"
    fi
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    export CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++}
    local cache=${DEPTHBENCH_CACHE_ROOT:-${TMPDIR:-/tmp}/depthbench-${USER:-user}-${profile}}
    export TORCHINDUCTOR_CACHE_DIR=${cache}/inductor TRITON_CACHE_DIR=${cache}/triton
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" || return
    hash -r
    printf 'DepthBench profile=%s\nPython=%s/bin/python\nSource=%s\nFLA=%s\n' \
        "${profile}" "${venv}" "${repo}" "${deps:-none}"
}
if _depthbench_activate "$@"; then
    unset -f _depthbench_activate
else
    unset -f _depthbench_activate
    return 1
fi
