# Validation On 2026-09-24

Reference interpreter: `DepthBench/.venv_b200/bin/python`.
Reference source: `DepthBench_main_latest`, plus the selected FLA source profile.

| Check | Result |
|---|---|
| Existing base environment `pip check` | Passed; no broken requirements |
| Locked requirements, `pip install --dry-run --no-index` | Passed; all 161 pins already satisfied |
| Shell syntax, `install.sh` and `activate.sh` | Passed |
| Base profile import/version/source-origin checks | Passed |
| MoDA profile imports and patched-kernel SHA256 | Passed |
| AttnRes profile imports and both FLA 0.4.1 distributions | Passed |
| Apply exported MoDA patch to the pinned clean upstream file | Passed; byte-identical kernel SHA256 |

For the environment PR, all three profile import checks were repeated against
the clean source checkout based on upstream `main` at `4ef3b20`, rather than the
historical worktrees above. Python/YAML parsing, the 161 installed package pins,
and the exported patch hash also passed. No training code is changed by this PR.

These checks do not modify the installed packages. No fresh Conda/venv rebuild,
CUDA kernel compilation, GPU allocation, training step, or checkpoint evaluation
was performed for this environment-documentation task. GPU training evidence
comes from the earlier experiments, not these import checks.

The login host has no usable matching CUDA driver. Its CUDA/FLA warnings during
imports are expected; use a compute node for `check_imports.py --cuda` and the
method-specific GPU tests.

The snapshot intentionally keeps Python3.10.12 rather than silently upgrading
the runtime used for old experiments. Some dependencies recommend Python3.11
or warn about upcoming Python3.10 support changes. A Python upgrade should be
validated as a separate environment, not applied to active/reproducibility runs.
