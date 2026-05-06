"""
Numerical correctness test: submission.sparse_attn vs reference.

Run:
    python test_correctness.py

bf16 has ~3 decimal digits of precision; tolerances reflect that.
"""
import sys
import torch

from reference.reference import sparse_attn as ref_attn
from submission import sparse_attn as our_attn
from workloads import all_workloads, make_inputs


# bf16 precision is loose; einsum reductions over d=512 amplify error
ATOL = 5e-2
RTOL = 5e-2


def check_one(label, b, m, kv_len, topk, seed=0):
    q, kv, sink, idx, scale = make_inputs(b, m, kv_len, topk, seed=seed)

    o_ref  = ref_attn(q, kv, sink, idx, scale)
    o_ours = our_attn(q, kv, sink, idx, scale)

    diff = (o_ref.float() - o_ours.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(o_ours.float(), o_ref.float(), atol=ATOL, rtol=RTOL)
    flag = "✓" if ok else "✗"
    print(f"  {flag} {label:35s}  max|Δ|={max_diff:.3e}  mean|Δ|={mean_diff:.3e}")
    return ok, max_diff


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping (Triton requires CUDA-like).")
        sys.exit(0)

    failed = []
    for label, b, m, kv_len, topk in all_workloads():
        ok, _ = check_one(label, b, m, kv_len, topk)
        if not ok:
            failed.append(label)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED on {len(failed)} workload(s):")
        for label in failed:
            print(f"  {label}")
        sys.exit(1)
    else:
        print("ALL WORKLOADS PASSED ✓")


if __name__ == "__main__":
    main()
