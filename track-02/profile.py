"""
Profiling helper for the Triton kernel.

Two modes:
    python profile.py codegen [--shape decode|prefill]
        Print PTX and resource usage (registers, shared memory) for the kernel
        compiled at a representative shape.

    python profile.py timeline [--shape decode|prefill]
        Run a short torch.profiler trace and print top kernels by CUDA time.

This file is a development aid only — never imported by submission.py.
"""
import argparse
import torch

from submission import hc_split_sinkhorn as hc_ours
from workloads import make_inputs, HC_MULT, SINKHORN_ITERS, EPS, MIXES_DIM


SHAPES = {
    "decode":   (64, 1,    MIXES_DIM),
    "prefill":  (64, 1024, MIXES_DIM),
}


def codegen_info(shape):
    """Compile the kernel and dump PTX + register/shared-memory usage."""
    mixes, scale, base = make_inputs(shape)

    # Warm up to trigger JIT compile
    _ = hc_ours(mixes, scale, base, HC_MULT, SINKHORN_ITERS, EPS)

    # Walk the cache to print compile artifacts
    try:
        import triton
        # Triton 3.x stores compiled kernels per JITFunction; you'll need to
        # adapt this to inspect the *specific* kernel object once submission.py
        # exposes it (e.g. via a module-level _hc_split_sinkhorn_kernel).
        print("Triton version:", triton.__version__)
        print("(Add introspection here once submission.py exposes the kernel object.)")
    except ImportError:
        print("triton not installed")


def timeline(shape, n_iters=50):
    mixes, scale, base = make_inputs(shape)

    # Warm up
    for _ in range(10):
        hc_ours(mixes, scale, base, HC_MULT, SINKHORN_ITERS, EPS)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(n_iters):
            hc_ours(mixes, scale, base, HC_MULT, SINKHORN_ITERS, EPS)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["codegen", "timeline"])
    ap.add_argument("--shape", choices=list(SHAPES.keys()), default="prefill")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    shape = SHAPES[args.shape]
    print(f"Profiling shape={shape} ({args.shape})\n")

    if args.mode == "codegen":
        codegen_info(shape)
    else:
        timeline(shape)


if __name__ == "__main__":
    main()
