"""
Benchmark: submission.sparse_attn vs reference, across all 10 workloads.

Run:
    python bench.py
"""
import math
import torch
import triton

from reference.reference import sparse_attn as ref_attn
from submission import sparse_attn as our_attn
from workloads import WORKLOADS, make_inputs


def bench(fn, args, warmup=10, rep=50):
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


def fmt_ms(ms):
    if ms < 1:
        return f"{ms*1000:7.2f}μs"
    return f"{ms:7.3f}ms"


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    print(f"{'workload':<32} {'shape':<28} {'ref':>10} {'ours':>10} {'speedup':>8}")
    print("-" * 92)

    speedups = []
    for label, kind, b, m, kv_len, topk in WORKLOADS:
        args = make_inputs(b, m, kv_len, topk)
        ref_ms  = bench(ref_attn, args)
        our_ms  = bench(our_attn, args)
        spd = ref_ms / our_ms
        speedups.append(spd)
        shape = f"b={b} m={m} kv={kv_len} k={topk}"
        print(f"{label:<32} {shape:<28} "
              f"{fmt_ms(ref_ms):>10} {fmt_ms(our_ms):>10} {spd:7.2f}x")

    print("-" * 92)
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print(f"GEOMEAN speedup across {len(speedups)} workloads: {geo:.2f}x")


if __name__ == "__main__":
    main()
