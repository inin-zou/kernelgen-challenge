import torch
import pytest
from reference.reference import fused_moe as fused_moe_ref


def _make_inputs(M, K, N, E, topk, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(M, K, dtype=dtype, generator=g)
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, generator=g) * 0.02
    w2 = torch.randn(E, K, N, dtype=dtype, generator=g) * 0.02
    score = torch.randn(M, E, dtype=dtype, generator=g)
    return hidden, w1, w2, score


def test_output_shape_decode_small():
    hidden, w1, w2, score = _make_inputs(M=1, K=2048, N=1024, E=8, topk=2)
    out = fused_moe_ref(hidden, w1, w2, score, topk=2)
    assert out.shape == (1, 2048)
    assert out.dtype == torch.bfloat16


def test_output_shape_prefill_64expert():
    hidden, w1, w2, score = _make_inputs(M=64, K=2048, N=1024, E=64, topk=6)
    out = fused_moe_ref(hidden, w1, w2, score, topk=6)
    assert out.shape == (64, 2048)
    assert out.dtype == torch.bfloat16


def test_topk_subset_invariance():
    """Forcing only the top-k experts via score scattering still yields finite output."""
    M, K, N, E, topk = 4, 2048, 1024, 8, 2
    hidden, w1, w2, score = _make_inputs(M, K, N, E, topk, seed=42)

    out_full = fused_moe_ref(hidden, w1, w2, score, topk=topk)

    weights_fp32 = score.float().softmax(dim=-1)
    _, top_ids = weights_fp32.topk(topk, dim=-1)
    forced_score = torch.full_like(score, -1e4)
    forced_score.scatter_(1, top_ids, score.gather(1, top_ids))

    out_forced = fused_moe_ref(hidden, w1, w2, score=forced_score, topk=topk)

    assert out_forced.shape == out_full.shape
    assert torch.isfinite(out_forced).all()


def test_renormalize_changes_output():
    hidden, w1, w2, score = _make_inputs(M=4, K=2048, N=1024, E=8, topk=2, seed=7)
    a = fused_moe_ref(hidden, w1, w2, score, topk=2, renormalize=False)
    b = fused_moe_ref(hidden, w1, w2, score, topk=2, renormalize=True)
    assert not torch.allclose(a, b, atol=1e-3)
