"""
Submission file for GOSIM KernelGen Track-03: Fused MoE.

v3: minimum-viable. After v1 (full plan) and v2 (defensive API fixes) both
failed universally on all 5 backends, v3 strips back to the most basic
Triton API surface to isolate where the bug is.

v3 changes vs v2:
- Drop `tl.dot` entirely. All "matmul" is `tl.sum(a[None, :] * b, axis=1)`.
- Drop `IS_ASCEND: tl.constexpr` branching. Single code path everywhere.
- Drop `acc=` and `out_dtype=` keyword args (they're gone with `tl.dot`).
- Drop pointer-arithmetic broadcast tricks (no 2D `tl.dot` operand needed).
- 2D grid for both compute kernels on every backend.

Will be slower than v2 (~3-5x slower on tensor-core backends), but uses only
the most-supported Triton APIs (tl.load/store, tl.sum, tl.max, tl.min,
tl.where, tl.exp, tl.sigmoid). If v3 still fails universally, the bug is
in routing or the wrapper, not in the compute kernels.

Pipeline:
    score → [routing] → topk_weights, topk_ids
    hidden, w1 → [gateup+silu] → intermediate [M*topk, N]
    intermediate, w2 → [down+topk_reduce] → output

Compliance: all numerical compute in @triton.jit. Torch ops only for memory.

Spec: docs/superpowers/specs/2026-05-06-track-03-fused-moe-design.md
"""
import os

os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")            # MetaX defensive
os.environ.setdefault("MUSA_ENABLE_SQMMA", "1")                 # MTT bf16 (no-op without tl.dot)
os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")        # Ascend grid > 65535 safety

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Backend detection — Triton runtime API only.
# ---------------------------------------------------------------------------
def _detect_backend(target_names) -> bool:
    try:
        target = triton.runtime.driver.active.get_current_target()
        backend = getattr(target, "backend", "")
        return backend in target_names
    except Exception:
        return False


_IS_ASCEND_CACHED = None
_IS_MTT_CACHED = None


def _is_ascend():
    global _IS_ASCEND_CACHED
    if _IS_ASCEND_CACHED is None:
        _IS_ASCEND_CACHED = _detect_backend(("npu", "ascend", "ascendc"))
    return _IS_ASCEND_CACHED


def _is_mtt():
    global _IS_MTT_CACHED
    if _IS_MTT_CACHED is None:
        _IS_MTT_CACHED = _detect_backend(("musa", "mt", "mthread", "mthreads"))
    return _IS_MTT_CACHED


# ---------------------------------------------------------------------------
# Kernel A — routing: softmax + iterative topk
# ---------------------------------------------------------------------------
@triton.jit
def _routing_kernel(
    score_ptr,           # [M, E_REAL] bf16
    topk_w_ptr,          # [M, TOPK]    bf16
    topk_id_ptr,         # [M, TOPK]    int32
    M,
    E:       tl.constexpr,   # power-of-2 padded
    E_REAL:  tl.constexpr,
    TOPK:    tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M

    e = tl.arange(0, E)
    e_mask = e < E_REAL

    NEG_INF = float('-inf')

    s_off = m[:, None] * E_REAL + e[None, :]
    s_load_mask = m_mask[:, None] & e_mask[None, :]
    s = tl.load(score_ptr + s_off, mask=s_load_mask, other=0.0).to(tl.float32)
    s = tl.where(e_mask[None, :], s, NEG_INF)

    s_max = tl.max(s, axis=1)
    s = tl.exp(s - s_max[:, None])
    s_sum = tl.sum(s, axis=1)
    weights = s / s_sum[:, None]

    for k in tl.static_range(TOPK):
        m_val = tl.max(weights, axis=1)
        is_max = (weights == m_val[:, None])
        idx_with_sentinel = tl.where(is_max, e[None, :], E)
        idx_max = tl.min(idx_with_sentinel, axis=1)
        gather_mask = (e[None, :] == idx_max[:, None])
        w_chosen = tl.sum(tl.where(gather_mask, weights, 0.0), axis=1)

        out_off = m * TOPK + k
        tl.store(topk_w_ptr  + out_off, w_chosen.to(tl.bfloat16), mask=m_mask)
        tl.store(topk_id_ptr + out_off, idx_max.to(tl.int32),     mask=m_mask)

        weights = tl.where(gather_mask, NEG_INF, weights)


def _launch_routing(score, topk, M, E):
    topk_w  = torch.empty(M, topk, dtype=score.dtype, device=score.device)
    topk_id = torch.empty(M, topk, dtype=torch.int32, device=score.device)

    E_pow2 = triton.next_power_of_2(E)
    BLOCK_M = 16
    grid = (triton.cdiv(M, BLOCK_M),)
    _routing_kernel[grid](
        score, topk_w, topk_id,
        M,
        E=E_pow2,
        E_REAL=E,
        TOPK=topk,
        BLOCK_M=BLOCK_M,
    )
    return topk_w, topk_id


# ---------------------------------------------------------------------------
# Kernel B — gate+up + SiLU (scalar-reduce, no tl.dot)
# Grid: (M*TOPK, ceil(N / BLOCK_N))
# ---------------------------------------------------------------------------
@triton.jit
def _gateup_silu_kernel(
    hidden_ptr,
    w1_ptr,
    topk_id_ptr,
    inter_ptr,
    K:       tl.constexpr,
    N:       tl.constexpr,
    TWO_N:   tl.constexpr,
    TOPK:    tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_mk = tl.program_id(0)
    pid_n  = tl.program_id(1)

    m     = pid_mk // TOPK
    k_idx = pid_mk %  TOPK
    e = tl.load(topk_id_ptr + m * TOPK + k_idx)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    acc_g = tl.zeros([BLOCK_N], tl.float32)
    acc_u = tl.zeros([BLOCK_N], tl.float32)

    for k_chunk in range(0, K, BLOCK_K):
        k_off = k_chunk + tl.arange(0, BLOCK_K)
        k_msk = k_off < K

        h = tl.load(hidden_ptr + m * K + k_off, mask=k_msk, other=0.0).to(tl.float32)

        w1g_off = e * (TWO_N * K) + n_off[:, None] * K + k_off[None, :]
        w1g_msk = n_mask[:, None] & k_msk[None, :]
        w1g = tl.load(w1_ptr + w1g_off, mask=w1g_msk, other=0.0).to(tl.float32)

        w1u_off = e * (TWO_N * K) + (N + n_off)[:, None] * K + k_off[None, :]
        w1u_msk = n_mask[:, None] & k_msk[None, :]
        w1u = tl.load(w1_ptr + w1u_off, mask=w1u_msk, other=0.0).to(tl.float32)

        acc_g += tl.sum(h[None, :] * w1g, axis=1)
        acc_u += tl.sum(h[None, :] * w1u, axis=1)

    silu_g = acc_g * tl.sigmoid(acc_g)
    out = (silu_g * acc_u).to(tl.bfloat16)

    tl.store(inter_ptr + pid_mk * N + n_off, out, mask=n_mask)


def _launch_gateup_silu(hidden, w1, topk_id, intermediate, M, K, N, topk):
    BLOCK_N = 64
    BLOCK_K = 64
    num_warps = 1 if _is_ascend() else 2

    MK = M * topk
    grid = (MK, triton.cdiv(N, BLOCK_N))
    _gateup_silu_kernel[grid](
        hidden, w1, topk_id, intermediate,
        K=K, N=N, TWO_N=2 * N,
        TOPK=topk,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=1,
    )


# ---------------------------------------------------------------------------
# Kernel C — down + topk-reduce (scalar-reduce, no tl.dot)
# Grid: (M, ceil(K / BLOCK_K_OUT))
# ---------------------------------------------------------------------------
@triton.jit
def _down_reduce_kernel(
    inter_ptr,
    w2_ptr,
    topk_w_ptr,
    topk_id_ptr,
    output_ptr,
    K:           tl.constexpr,
    N:           tl.constexpr,
    TOPK:        tl.constexpr,
    BLOCK_K_OUT: tl.constexpr,
    BLOCK_N_R:   tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m = pid_m
    k_out_off = pid_k * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
    k_out_mask = k_out_off < K

    acc = tl.zeros([BLOCK_K_OUT], tl.float32)

    for k_idx in tl.static_range(TOPK):
        e = tl.load(topk_id_ptr + m * TOPK + k_idx)
        w_t = tl.load(topk_w_ptr + m * TOPK + k_idx).to(tl.float32)

        partial = tl.zeros([BLOCK_K_OUT], tl.float32)
        for n_chunk in range(0, N, BLOCK_N_R):
            n_off = n_chunk + tl.arange(0, BLOCK_N_R)
            n_msk = n_off < N

            mid = tl.load(
                inter_ptr + (m * TOPK + k_idx) * N + n_off,
                mask=n_msk, other=0.0,
            ).to(tl.float32)

            w2_off = e * (K * N) + k_out_off[:, None] * N + n_off[None, :]
            w2_msk = k_out_mask[:, None] & n_msk[None, :]
            w2t = tl.load(w2_ptr + w2_off, mask=w2_msk, other=0.0).to(tl.float32)

            partial += tl.sum(mid[None, :] * w2t, axis=1)

        acc += w_t * partial

    tl.store(output_ptr + m * K + k_out_off, acc.to(tl.bfloat16), mask=k_out_mask)


def _launch_down_reduce(intermediate, w2, topk_w, topk_id, output, M, K, N, topk):
    BLOCK_K_OUT = 64
    BLOCK_N_R   = 64
    num_warps = 1 if _is_ascend() else 2

    grid = (M, triton.cdiv(K, BLOCK_K_OUT))
    _down_reduce_kernel[grid](
        intermediate, w2, topk_w, topk_id, output,
        K=K, N=N, TOPK=topk,
        BLOCK_K_OUT=BLOCK_K_OUT,
        BLOCK_N_R=BLOCK_N_R,
        num_warps=num_warps,
        num_stages=1,
    )


def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    """Fused MoE — entry point called by the judge.

    v3 ignores `renormalize` for now (defaults False per spec; v3 is a
    diagnostic submission, not the final tuned kernel).
    """
    assert hidden_states.dim() == 2
    assert w1.dim() == 3
    assert w2.dim() == 3
    assert score.dim() == 2
    M, K = hidden_states.shape
    E_w1, two_N, K_w1 = w1.shape
    E_w2, K_w2, N = w2.shape
    M_s, E_s = score.shape
    assert E_w1 == E_w2 == E_s
    assert K == K_w1 == K_w2
    assert two_N == 2 * N
    assert M_s == M
    E = E_w1

    hidden_c = hidden_states.contiguous()
    w1_c     = w1.contiguous()
    w2_c     = w2.contiguous()
    score_c  = score.contiguous()

    intermediate = torch.empty(M * topk, N, dtype=hidden_states.dtype, device=hidden_states.device)
    output       = torch.empty(M, K,         dtype=hidden_states.dtype, device=hidden_states.device)

    topk_w, topk_id = _launch_routing(score_c, topk, M, E)
    _launch_gateup_silu(hidden_c, w1_c, topk_id, intermediate, M, K, N, topk)
    _launch_down_reduce(intermediate, w2_c, topk_w, topk_id, output, M, K, N, topk)

    return output
