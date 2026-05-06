"""
Profiling helper for the Triton kernel.
Run:
    python profile.py timeline [--shape decode|prefill]
"""
import argparse
import torch

from submission import sparse_attn
from workloads import make_inputs


SHAPES = {
    "decode":  (64,  1,    640,   640),    # decode-64x1-640-640
    "prefill": (16,  4096, 4096,  128),    # prefill-16x4096-4096-128
}


def timeline(shape, n_iters=50):
    args = make_inputs(*shape)
    for _ in range(10):
        sparse_attn(*args)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(n_iters):
            sparse_attn(*args)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["timeline"])
    ap.add_argument("--shape", choices=list(SHAPES.keys()), default="prefill")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    shape = SHAPES[args.shape]
    print(f"Profiling shape (b, m, kv_len, topk) = {shape}\n")
    timeline(shape)


if __name__ == "__main__":
    main()
