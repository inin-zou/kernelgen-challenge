"""Frozen torch oracle for track-03 fused MoE.

Lifted verbatim from track-03/requirements.md. submission.py must match this
within atol=5e-2 (bf16 contest tolerance).
"""
import torch
import torch.nn.functional as F


def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[-1]
    dtype = hidden_states.dtype

    topk_weights = score.softmax(dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    final_out = torch.zeros(M, K, device=hidden_states.device, dtype=dtype)

    for expert_idx in range(E):
        mask = (topk_ids == expert_idx)
        token_weights = (topk_weights * mask).sum(dim=-1, keepdim=True)

        x = F.linear(hidden_states, w1[expert_idx])
        gate = F.silu(x[:, :N])
        x = x[:, N:] * gate
        x = F.linear(x, w2[expert_idx])

        final_out += x * token_weights

    return final_out
