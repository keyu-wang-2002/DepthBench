#!/usr/bin/env python3
"""Verify the selected runtime's versions/import origins; no GPU job is submitted."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Require a visible CUDA GPU and test matmul/backward")
    args = parser.parse_args()
    profile = os.environ.get("DEPTHBENCH_PROFILE")
    if profile not in {"base", "moda", "attnres"}:
        raise RuntimeError("First source environment/activate.sh with the intended profile")
    expected = {"torch": "2.8.0+cu128", "triton": "3.4.0", "liger-kernel": "0.8.0",
                "transformers": "5.7.0", "numpy": "2.2.6"}
    versions = {name: metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"Runtime differs from validated cu128 pins: {versions}")
    import torch
    import olmo_core
    from olmo_core.nn.transformer.config import TransformerBlockType
    from olmo_core.nn.transformer import block
    from liger_kernel.transformers.functional import liger_mhc_coeffs, liger_mhc_pre, liger_mhc_post_res

    repo = Path(os.environ["DEPTHBENCH_REPO"]).resolve()
    if not Path(olmo_core.__file__).resolve().is_relative_to(repo):
        raise RuntimeError(f"Wrong OLMo-core source imported: {olmo_core.__file__}")
    assert all(callable(op) for op in [liger_mhc_coeffs, liger_mhc_pre, liger_mhc_post_res])
    report = dict(profile=profile, versions=versions, olmo_core=olmo_core.__file__,
                  block_source=block.__file__, block_types=[str(x) for x in TransformerBlockType])
    if profile != "base":
        import fla
        deps = Path(os.environ["DEPTHBENCH_FLA_ROOT"]).resolve()
        if not Path(fla.__file__).resolve().is_relative_to(deps):
            raise RuntimeError(f"FLA namespace shadowed by another installation: {fla.__file__}")
        report["fla_source"] = fla.__file__
        if profile == "moda":
            from fla.ops.moda import parallel_moda_v17
            assert callable(parallel_moda_v17)
            manifest = json.loads((Path(__file__).parent / "source-manifest.json").read_text())
            expected_hash = manifest["sources"]["MoDA"]["file_sha256"]["libs/moda_triton/fla/ops/moda/moda_v17.py"]
            digest = hashlib.sha256((deps / "fla/ops/moda/moda_v17.py").read_bytes()).hexdigest()
            if digest != expected_hash:
                raise RuntimeError("MoDA v17 differs from the captured, head-dimension-corrected kernel")
            report["moda_kernel_sha256"] = digest
        else:
            from fla.ops.utils.op import exp
            from fla.utils import autotune_cache_kwargs, input_guard
            from olmo_core.kernels.attnres import fused_attnres
            assert callable(exp) and callable(input_guard) and callable(fused_attnres)
            assert isinstance(autotune_cache_kwargs, dict)
            for name in ("flash-linear-attention", "fla-core"):
                if metadata.version(name) != "0.4.1":
                    raise RuntimeError(f"Expected {name} 0.4.1")
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("No usable CUDA GPU/driver. Run --cuda on an allocated GPU node")
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        (x @ x.T).float().square().mean().backward()
        assert torch.isfinite(x.grad).all()
        torch.cuda.synchronize()
        report["cuda"] = dict(gpu=torch.cuda.get_device_name(), runtime=torch.version.cuda,
                              capability=torch.cuda.get_device_capability(), basic_backward="passed")
    else:
        report["cuda"] = "not tested; imports do not establish kernel correctness or training parity"
    print("ENVIRONMENT_IMPORTS_PASSED " + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
