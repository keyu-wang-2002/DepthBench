# Residual Training Environment

Linux x86_64 environment captured from the HC, mHC, MoDA and AttnRes experiments
on 2026-09-24:

| Component | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch / CUDA wheel runtime | 2.8.0+cu128 / 12.8 |
| Triton | 3.4.0 |
| Liger Kernel | 0.8.0 |
| Transformers / Tokenizers | 5.7.0 / 0.22.2 |
| NumPy | 2.2.6 |
| W&B | 0.26.1 |

The requirements lock includes the captured registry dependencies. Install
DepthBench's modified OLMo-core from this repository, not PyPI. A compatible
NVIDIA driver, gcc/g++ and linker are system prerequisites. Standalone Flash
Attention and other optional backends are not included.

## Install

From the repository root, install with one command:

```bash
bash environment/install.sh
```

This detects the repository automatically and creates `.venv-cu128` inside it.
Python 3.10 must already be available; set `PYTHON_BIN=/path/to/python3.10` if
needed. **MoDA is not installed by default**; follow the separate section below.

Use an allocated CPU node for installation. The installer creates a new venv,
installs local OLMo-core and an isolated AttnRes FLA overlay, and runs `pip check`.
It prints progress and refuses to modify an existing environment. Activate it
after installation:

```bash
source .venv-cu128/bin/activate
```

To choose a different destination (and optionally a different source checkout):

```bash
bash environment/install.sh /path/new-depthbench-venv
# Existing two-path usage remains supported:
bash environment/install.sh /path/new-depthbench-venv /path/DepthBench
```

Alternatively, use Conda from the directory containing the requirements files:

```bash
cd environment
conda env create -f environment.yml
conda activate depthbench-residuals-cu128
python -m pip install --no-deps --no-build-isolation -e /path/DepthBench/pretrain/OLMo-core
python -m pip install --no-deps --no-compile \
  --target "$CONDA_PREFIX/profiles/attnres-fla-0.4.1" -r requirements-attnres.txt
python -m pip check
```

Use an empty overlay directory. Do not install OLMo-core's `[all]` extra, which
can replace this Torch/FLA stack. The legacy Torch 2.6 / CUDA 11.8 environment is
not interchangeable with this one; Liger mHC previously hung on H100 with that
older stack and 20 Sinkhorn iterations.

## MoDA Kernel

MoDA needs its fork of FLA plus the included H100 dispatch and head-dimension
tiling fixes. For a new clone:

```bash
git clone https://github.com/hustvl/MoDA.git /path/MoDA
git -C /path/MoDA checkout ba872a347c2b085ac618c8692de9abd0247a8f4a
git -C /path/MoDA apply --check /path/DepthBench/environment/moda-v17-local.patch
git -C /path/MoDA apply /path/DepthBench/environment/moda-v17-local.patch
```

Do not apply the patch twice to an existing patched checkout.

## Select Dependencies

MoDA's fork and official FLA both provide `fla`, but are not interchangeable:
MoDA needs `fla.ops.moda`; AttnRes needs the official FLA 0.4.1 APIs. Keep them
separate and select exactly one provider per process. Do not globally install
both or append an old FLA path to the paths below.

After activating the venv or Conda environment, set the repository paths:

```bash
REPO=/path/DepthBench
ENV_ROOT="${VIRTUAL_ENV:-${CONDA_PREFIX:?Activate the environment first}}"
BASE_PYTHONPATH="$REPO/pretrain/OLMo-core/src:$REPO:$REPO/examples"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

Then choose **one** of the following before launching training.

HC, Liger mHC, or standard/normalization baselines:
```bash
export PYTHONPATH="$BASE_PYTHONPATH"
```

MoDA, using the patched source:
```bash
export PYTHONPATH="/path/MoDA/libs/moda_triton:$BASE_PYTHONPATH"
```

Full / Block AttnRes, using the isolated official FLA overlay:
```bash
export PYTHONPATH="$ENV_ROOT/profiles/attnres-fla-0.4.1:$BASE_PYTHONPATH"
```

Launch a fresh Python process after selecting dependencies. Training settings
and the choice of model implementation remain controlled by the experiment.

## Validation

The captured environment passed `pip check`, all 161 installed package pins,
and imports for all three dependency selections against main at `4ef3b20`.
The MoDA patch reproduced the kernel used by the experiments. No fresh
environment rebuild or new GPU training run was performed for this bundle;
revalidate method-specific kernels when changing runtime, GPU or model shape.
