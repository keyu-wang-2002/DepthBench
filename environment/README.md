# DepthBench Residual Training Environment

This bundle captures the working Linux x86_64 / Python 3.10.12 **cu128**
environment on 2026-09-24. It does not change any existing environment or jobs.

## Core Versions

| Component | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch / CUDA wheel runtime | 2.8.0+cu128 / 12.8 |
| Triton | 3.4.0 |
| Liger Kernel | 0.8.0 |
| Transformers / Tokenizers | 5.7.0 / 0.22.2 |
| NumPy | 2.2.6 |
| W&B | 0.26.1 |
| Local OLMo-core | 2.5.0 plus DepthBench changes; use the correct source checkout |

`requirements-cu128.lock.txt` pins the installed registry packages, including
CUDA libraries, data/evaluation tools, and transitive dependencies. It excludes
editable/source packages deliberately. **Installing PyPI OLMo-core is not a
replacement for DepthBench's modified source.** This is a captured version lock,
not a wheel archive or a pip `--require-hashes` lock.

## Method Profiles

| Profile | Methods | Additional source/dependencies |
|---|---|---|
| `base` | Pre-LN/Post-LN, normalization variants, HC, Liger mHC | DepthBench source, Liger 0.8.0 |
| `moda` | PreNorm/PostNorm MoDA | Patched `MoDA/libs/moda_triton` as the `fla` provider |
| `attnres` | Full and Block AttnRes | Separate `flash-linear-attention==0.4.1` + `fla-core==0.4.1` overlay |

MoDA's fork and official FLA both provide a package named **`fla`**. The official
0.4.1 overlay does not contain `fla.ops.moda`, while MoDA's older fork lacks APIs
used by the current AttnRes kernel. Do not pip-install them over one another or
try to combine their files. Select one provider per process with the launcher.
This is dependency isolation, not a mixed MoDA/HC architecture.

The old `.venv` is **torch2.6.0+cu118 / Triton3.2.0**, not interchangeable with
`.venv_b200`. In particular, Liger mHC with 20 Sinkhorn iterations previously
hung with the old stack on H100. Use this cu128 stack for reproducing current
HC/mHC runs; do not silently switch an old experiment's runtime on resume.

No standalone `flash-attn`, FA3/FA4, torchao, or causal-conv1d package is included:
they are not installed in the captured base environment. Our standard attention
training uses the PyTorch SDPA path; MoDA/AttnRes/mHC add their own kernels.
An explicit `flash_2`/`flash_3`/`flash_4` backend needs separate installation and
validation. `lm-eval` is captured for the existing custom adapter, not every
optional HF/vLLM evaluation backend.

## Recreate Without Conda

Run installation on an allocated CPU node, not a busy login node. This installs
into a **new** venv and refuses to modify an existing one:

```bash
PYTHON_BIN=python3.10 bash environment/install.sh \
  /path/new-depthbench-venv /path/DepthBench
```

The installer also creates the AttnRes overlay inside the new environment.
MoDA source is handled below. The Python executable, a suitable NVIDIA driver,
gcc/g++, and a working system linker are system prerequisites, not pip packages.
The driver must support this CUDA12.8/PyTorch build and the target GPU.
Do not copy old shebangs, absolute editable-install paths, or linker symlinks
pointing into another user's home directory.

## Conda Alternative

Run from the directory containing the requirements file:

```bash
cd environment
conda env create -f environment.yml
conda activate depthbench-residuals-cu128
python -m pip install --no-deps --no-build-isolation -e /path/DepthBench/pretrain/OLMo-core
python -m pip install --no-deps --no-compile \
  --target "$CONDA_PREFIX/profiles/attnres-fla-0.4.1" -r requirements-attnres.txt
python -m pip check
```

Use an empty overlay directory. Do not install OLMo-core's `[all]` extra: it
selects additional backends and can replace the validated Torch/FLA stack.

## Preserve The MoDA Kernel Patch

The current MoDA kernel has local H100 dispatch and head-dimension tile fixes.
A clean upstream clone is **not** equivalent to the code used for our runs.
For a new clone only:

```bash
git clone https://github.com/hustvl/MoDA.git /path/MoDA
git -C /path/MoDA checkout ba872a347c2b085ac618c8692de9abd0247a8f4a
git -C /path/MoDA apply --check /path/environment/moda-v17-local.patch
git -C /path/MoDA apply /path/environment/moda-v17-local.patch
```

Do not apply the patch twice to the existing workspace checkout. Its exact
post-patch hash is recorded in `source-manifest.json` and checked by the import
test. The manifest also records the DepthBench worktree commits and key source
hashes. Keep the intended source snapshot: an environment file cannot reproduce
uncommitted architecture changes or choose the right mHC initialization for you.

## Select And Verify

Run each profile in a fresh Python process. The activation script replaces
`PYTHONPATH` so an old FLA overlay cannot silently shadow the selected provider.

```bash
# HC / mHC / standard and normalization baselines
source environment/activate.sh base /path/DepthBench /path/new-depthbench-venv
python environment/check_imports.py

# MoDA, using the patched source (not a global editable FLA installation)
source environment/activate.sh moda /path/DepthBench /path/new-depthbench-venv \
  /path/MoDA/libs/moda_triton
python environment/check_imports.py

# Full / Block AttnRes, using the isolated overlay installed above
source environment/activate.sh attnres /path/DepthBench /path/new-depthbench-venv
python environment/check_imports.py
```

Use the source checkout matching the experiment you want to reproduce. If an
AttnRes overlay already exists elsewhere, pass it as the fourth activation
argument. No training setting is changed by activation.

The helper limits CPU/compile workers and uses writable per-user caches. Set
`DEPTHBENCH_TOOLCHAIN_BIN` to a directory containing a known-good `ld` if the
site linker requires an override; no account-specific linker path is embedded.

## Validation Scope

The captured base passes `pip check`. Import/version/source-path checks can run
without a GPU. Login-node CUDA/FLA warnings are not evidence that the training
node has an incompatible GPU driver. On an allocated GPU, add `--cuda` for a
CUDA BF16 matmul/backward check. This basic check is **not** a residual-kernel
parity test or a full training reproduction.

Prior experiment validation includes mHC CUDA smoke tests and production
training, MoDA patched-kernel training/benchmarks, and AttnRes H100
forward/backward parity plus compiled L34 training. Those experiment logs are
not distributed with this bundle; see `VALIDATION.md` for the checks performed
when capturing the environment.
Do not infer that every method, GPU, head dimension, or TP/CP/MoE combination
is supported just because the environment imports successfully. Re-run the
method-specific GPU smoke test when changing runtime, GPU, source or shape.

No credentials, data, checkpoints, or W&B/HF login state are included here.

## Export A New Snapshot

To record a later validated runtime, run the exporter with that runtime's
Python. Supply the source paths explicitly and an empty output directory:

```bash
python environment/export_snapshot.py --depthbench-repo /path/DepthBench \
  --moda-repo /path/MoDA --output-dir /path/new-environment-snapshot
```

It exports package pins, source provenance and the MoDA kernel diff against
that checkout's HEAD. The checked-in manifest records the historical source
worktrees at capture time, not the current repository revision. Review source
revisions, patches and installation instructions together when updating it.
