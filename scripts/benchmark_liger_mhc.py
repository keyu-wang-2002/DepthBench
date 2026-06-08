#!/usr/bin/env python
import argparse
import importlib.util
import time


def build_config(variant: str, args: argparse.Namespace):
    from olmo_core.config import DType
    from olmo_core.nn.transformer import HyperConnectionsConfig, TransformerConfig

    hyper_connections = None
    if variant == "pytorch_mhc":
        hyper_connections = HyperConnectionsConfig(
            kind="mhc",
            num_residual_streams=args.streams,
            tanh=False,
            gating_factor_init=0.01,
            sinkhorn_iters=20,
        )
    elif variant == "liger_mhc":
        hyper_connections = HyperConnectionsConfig(
            kind="liger_mhc",
            num_residual_streams=args.streams,
            tanh=False,
            gating_factor_init=0.01,
            sinkhorn_iters=20,
            liger_phi_dtype=DType.bfloat16,
            collapse="auto",
        )

    return TransformerConfig.llama_like(
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        n_layers=args.layers,
        n_heads=args.heads,
        fused_ops=False,
        dtype=DType.bfloat16,
        hyper_connections=hyper_connections,
    )


def run_variant(variant: str, args: argparse.Namespace) -> None:
    import torch

    if variant == "liger_mhc" and importlib.util.find_spec("liger_kernel") is None:
        print(f"{variant}: skipped, liger_kernel is not installed")
        return

    device = torch.device("cuda")
    config = build_config(variant, args)
    model = config.build(init_device="cuda")
    model.init_weights(device=device, max_seq_len=args.seq_len)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )

    def step() -> torch.Tensor:
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        optim.step()
        return output.loss.detach()

    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = args.batch_size * args.seq_len * args.steps
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"{variant}: loss={loss.item():.4f} tokens/s={tokens / elapsed:.1f} "
        f"peak_mem_gb={peak_gb:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tiny baseline/mHC/Liger mHC models.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "pytorch_mhc", "liger_mhc"],
        choices=["baseline", "pytorch_mhc", "liger_mhc"],
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(f"torch={torch.__version__}")
    try:
        import triton

        print(f"triton={triton.__version__}")
    except Exception:
        print("triton=unavailable")
    print(f"gpu={torch.cuda.get_device_name()}")

    for variant in args.variants:
        run_variant(variant, args)


if __name__ == "__main__":
    main()
