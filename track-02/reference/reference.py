"""
PyTorch reference implementation of hc_split_sinkhorn.

Frozen — do not modify. This is the numerical oracle for correctness tests.
Source: kernelgen-challenge/track-02/requirements.md
"""
import torch
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """
    Args:
        mixes:          [b, s, (2 + hc_mult) * hc_mult]   — fp32
        hc_scale:       [3]                                 — fp32
        hc_base:        [(2 + hc_mult) * hc_mult]           — fp32
        hc_mult:        int, number of streams
        sinkhorn_iters: int, Sinkhorn iterations
        eps:            float, numerical stability

    Returns:
        pre:    [b, s, hc_mult]            — sigmoid + eps
        post:   [b, s, hc_mult]            — 2 * sigmoid
        comb:   [b, s, hc_mult, hc_mult]   — doubly-stochastic via Sinkhorn
    """
    b, s, _ = mixes.shape
    hc = hc_mult

    pre_raw  = mixes[..., :hc]
    post_raw = mixes[..., hc:2 * hc]
    comb_raw = mixes[..., 2 * hc:].reshape(b, s, hc, hc)

    pre  = torch.sigmoid(pre_raw  * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc:2 * hc])
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)

    comb = F.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb
