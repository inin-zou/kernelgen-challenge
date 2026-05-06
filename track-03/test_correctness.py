"""Correctness vs reference, atol=5e-2 (bf16 contest convention).

Cannot run on Mac (no Triton). Ships to platform if we want a side-channel
correctness check there.
"""
import torch
import pytest

from submission import fused_moe as fused_moe_sub
from reference.reference import fused_moe as fused_moe_ref
from workloads import WORKLOADS


def _make_inputs(M, K, N, E, topk, dtype=torch.bfloat16, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(M, K, dtype=dtype, generator=g, device=device)
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, generator=g, device=device) * 0.02
    w2 = torch.randn(E, K, N, dtype=dtype, generator=g, device=device) * 0.02
    score = torch.randn(M, E, dtype=dtype, generator=g, device=device)
    return hidden, w1, w2, score


@pytest.mark.parametrize(
    "workload",
    WORKLOADS,
    ids=lambda w: f"{w['label']}_M{w['M']}_E{w['E']}_topk{w['topk']}",
)
def test_against_reference(workload):
    M, K, N, E, topk = workload["M"], workload["K"], workload["N"], workload["E"], workload["topk"]
    hidden, w1, w2, score = _make_inputs(M, K, N, E, topk)

    out_sub = fused_moe_sub(hidden, w1, w2, score, topk)
    out_ref = fused_moe_ref(hidden, w1, w2, score, topk)

    torch.testing.assert_close(out_sub, out_ref, atol=5e-2, rtol=5e-2)
