"""
Benchmark: submission.hc_split_sinkhorn vs reference, across the 4 workloads.

Run:
    python bench.py

Reports decode + prefill latency per workload, plus a frequency-weighted score
(per-workload calls/ms) so you can compare changes that help one regime at
the cost of another.
"""
import torch
import triton

from reference.reference import hc_split_sinkhorn as hc_ref
from submission import hc_split_sinkhorn as hc_ours
from workloads import WORKLOADS, make_inputs, HC_MULT, SINKHORN_ITERS, EPS


def bench(fn, shape, warmup=25, rep=100, seed=0):
    """Returns median latency in ms."""
    mixes, scale, base = make_inputs(shape, seed=seed)
    return triton.testing.do_bench(
        lambda: fn(mixes, scale, base, HC_MULT, SINKHORN_ITERS, EPS),
        warmup=warmup, rep=rep,
    )


def fmt_ms(ms):
    return f"{ms*1000:7.2f}μs" if ms < 1 else f"{ms:7.3f}ms"


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    rows = []
    print(f"{'workload':<10} {'stage':<8} {'shape':<22} {'ref':>10} {'ours':>10} {'speedup':>8}")
    print("-" * 72)

    for name, decode_shape, df, prefill_shape, pf in WORKLOADS:
        ref_d = bench(hc_ref,  decode_shape)
        our_d = bench(hc_ours, decode_shape)
        ref_p = bench(hc_ref,  prefill_shape)
        our_p = bench(hc_ours, prefill_shape)

        print(f"{name:<10} {'decode':<8} {str(decode_shape):<22} "
              f"{fmt_ms(ref_d):>10} {fmt_ms(our_d):>10} {ref_d/our_d:7.2f}x")
        print(f"{name:<10} {'prefill':<8} {str(prefill_shape):<22} "
              f"{fmt_ms(ref_p):>10} {fmt_ms(our_p):>10} {ref_p/our_p:7.2f}x")

        # Frequency-weighted average latency for this workload
        ref_avg = df * ref_d + pf * ref_p
        our_avg = df * our_d + pf * our_p
        rows.append((name, df, pf, ref_d, our_d, ref_p, our_p, ref_avg, our_avg))
        print(f"{name:<10} {'weighted':<8} {f'd={df:.3f} p={pf:.3f}':<22} "
              f"{fmt_ms(ref_avg):>10} {fmt_ms(our_avg):>10} {ref_avg/our_avg:7.2f}x")
        print()

    # Overall geometric-mean speedup of the weighted averages
    import math
    geomean = math.exp(sum(math.log(r[7]/r[8]) for r in rows) / len(rows))
    print("-" * 72)
    print(f"GEOMEAN weighted speedup across {len(rows)} workloads: {geomean:.2f}x")


if __name__ == "__main__":
    main()
