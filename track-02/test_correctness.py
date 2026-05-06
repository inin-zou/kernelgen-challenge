"""
Numerical correctness test: submission.hc_split_sinkhorn vs reference.

Run:
    python test_correctness.py

Tolerances are tight by default. Tighten them as the kernel matures.
"""
import sys
import torch

from reference.reference import hc_split_sinkhorn as hc_ref
from submission import hc_split_sinkhorn as hc_ours
from workloads import all_shapes, make_inputs, HC_MULT, SINKHORN_ITERS, EPS

ATOL = 1e-5
RTOL = 1e-4


def check_one(label, shape, seed=0):
    mixes, scale, base = make_inputs(shape, seed=seed)

    pre_r, post_r, comb_r = hc_ref(mixes, scale, base, HC_MULT, SINKHORN_ITERS, EPS)
    pre_o, post_o, comb_o = hc_ours(mixes, scale, base, HC_MULT, SINKHORN_ITERS, EPS)

    failed = []
    for name, ref, ours in [("pre", pre_r, pre_o),
                            ("post", post_r, post_o),
                            ("comb", comb_r, comb_o)]:
        diff = (ref - ours).abs().max().item()
        ok = torch.allclose(ours, ref, atol=ATOL, rtol=RTOL)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {name:5s}  max|Δ| = {diff:.3e}  shape={tuple(ref.shape)}")
        if not ok:
            failed.append((name, diff))

    return failed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping (Triton requires CUDA).")
        sys.exit(0)

    all_failed = []
    for label, shape in all_shapes():
        print(f"\n=== {label}  shape={shape} ===")
        failed = check_one(label, shape)
        if failed:
            all_failed.append((label, failed))

    print("\n" + "=" * 60)
    if all_failed:
        print(f"FAILED on {len(all_failed)} workload(s):")
        for label, failed in all_failed:
            for name, diff in failed:
                print(f"  {label}  {name}: max|Δ| = {diff:.3e}")
        sys.exit(1)
    else:
        print("ALL WORKLOADS PASSED ✓")


if __name__ == "__main__":
    main()
