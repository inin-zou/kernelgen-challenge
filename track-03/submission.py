"""
Submission file for GOSIM KernelGen Track-03: Fused MoE.

v1: hybrid (option C) — one universal @triton.jit pipeline with per-backend
grid shape and BLOCK_* constants. Three kernels, no atomics, no permutation,
no autotune.

Pipeline:
    score → [routing] → topk_weights, topk_ids
    hidden, w1 → [gateup+silu] → intermediate [M*topk, N]
    intermediate, w2 → [down+topk_reduce] → output

Compliance: all numerical compute in @triton.jit. Torch ops only for memory.
Backend detection via Triton runtime API only (no host-side CUDA queries).

Spec: docs/superpowers/specs/2026-05-06-track-03-fused-moe-design.md
"""
import os

os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")  # MetaX defensive
os.environ.setdefault("MUSA_ENABLE_SQMMA", "1")       # MTT bf16 tensor core

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


_IS_METAX_CACHED = None
_IS_ASCEND_CACHED = None
_IS_MTT_CACHED = None
_IS_HYGON_CACHED = None
_IS_ILUVATAR_CACHED = None


def _is_metax():
    global _IS_METAX_CACHED
    if _IS_METAX_CACHED is None:
        _IS_METAX_CACHED = _detect_backend(("maca",))
    return _IS_METAX_CACHED


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


def _is_hygon():
    global _IS_HYGON_CACHED
    if _IS_HYGON_CACHED is None:
        _IS_HYGON_CACHED = _detect_backend(("hip",))
    return _IS_HYGON_CACHED


def _is_iluvatar():
    global _IS_ILUVATAR_CACHED
    if _IS_ILUVATAR_CACHED is None:
        _IS_ILUVATAR_CACHED = _detect_backend(("cuda",)) and not _is_mtt()
    return _IS_ILUVATAR_CACHED


# ---------------------------------------------------------------------------
# Per-backend block constants (seeds from the v1 design doc).
# ---------------------------------------------------------------------------
_CFG_GATEUP_ASCEND   = dict(BLOCK_MK=1,  BLOCK_N=128, BLOCK_K=128, num_warps=2, num_stages=1)
_CFG_GATEUP_HYGON    = dict(BLOCK_MK=16, BLOCK_N=128, BLOCK_K=64,  num_warps=4, num_stages=1)
_CFG_GATEUP_ILUVATAR = dict(BLOCK_MK=16, BLOCK_N=128, BLOCK_K=64,  num_warps=2, num_stages=1)
_CFG_GATEUP_MTT      = dict(BLOCK_MK=32, BLOCK_N=128, BLOCK_K=64,  num_warps=8, num_stages=1)
_CFG_GATEUP_METAX    = dict(BLOCK_MK=16, BLOCK_N=64,  BLOCK_K=64,  num_warps=2, num_stages=1)

_CFG_DOWN_ASCEND   = dict(BLOCK_MK=1,  BLOCK_K_OUT=128, BLOCK_N_R=128, num_warps=2, num_stages=1)
_CFG_DOWN_HYGON    = dict(BLOCK_MK=16, BLOCK_K_OUT=128, BLOCK_N_R=64,  num_warps=4, num_stages=1)
_CFG_DOWN_ILUVATAR = dict(BLOCK_MK=16, BLOCK_K_OUT=128, BLOCK_N_R=64,  num_warps=2, num_stages=1)
_CFG_DOWN_MTT      = dict(BLOCK_MK=32, BLOCK_K_OUT=128, BLOCK_N_R=64,  num_warps=8, num_stages=1)
_CFG_DOWN_METAX    = dict(BLOCK_MK=16, BLOCK_K_OUT=64,  BLOCK_N_R=64,  num_warps=2, num_stages=1)


def _gateup_cfg():
    if _is_ascend():   return _CFG_GATEUP_ASCEND
    if _is_mtt():      return _CFG_GATEUP_MTT
    if _is_metax():    return _CFG_GATEUP_METAX
    if _is_hygon():    return _CFG_GATEUP_HYGON
    return _CFG_GATEUP_ILUVATAR


def _down_cfg():
    if _is_ascend():   return _CFG_DOWN_ASCEND
    if _is_mtt():      return _CFG_DOWN_MTT
    if _is_metax():    return _CFG_DOWN_METAX
    if _is_hygon():    return _CFG_DOWN_HYGON
    return _CFG_DOWN_ILUVATAR


# ---------------------------------------------------------------------------
# Kernel A — routing: softmax + iterative topk
# Inputs : score [M, E] (bf16)
# Outputs: topk_w [M, topk] (bf16), topk_id [M, topk] (int32)
# Grid   : (ceil(M / BLOCK_M),)
# ---------------------------------------------------------------------------
@triton.jit
def _routing_kernel(
    score_ptr,           # [M, E_REAL]
    topk_w_ptr,          # [M, TOPK]
    topk_id_ptr,         # [M, TOPK]
    M,
    RENORMALIZE: tl.constexpr,
    E:      tl.constexpr,   # power-of-2 padded width
    E_REAL: tl.constexpr,   # actual number of experts
    TOPK:    tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M

    e = tl.arange(0, E)
    e_mask = e < E_REAL

    NEG_INF = float('-inf')

    # Load score with -inf padding for both row and column edges.
    s_off = m[:, None] * E_REAL + e[None, :]
    s_load_mask = m_mask[:, None] & e_mask[None, :]
    s = tl.load(score_ptr + s_off, mask=s_load_mask, other=NEG_INF).to(tl.float32)
    s = tl.where(e_mask[None, :], s, NEG_INF)

    # softmax over the expert dim
    s_max = tl.max(s, axis=1)
    s = tl.exp(s - s_max[:, None])
    s_sum = tl.sum(s, axis=1)
    weights = s / s_sum[:, None]              # [BLOCK_M, E] fp32

    # iterative topk: argmax → store → mask the chosen index with -inf
    sum_topk = tl.zeros([BLOCK_M], tl.float32)
    for k in tl.static_range(TOPK):
        idx_max = tl.argmax(weights, axis=1)                # [BLOCK_M]
        gather_mask = (e[None, :] == idx_max[:, None])      # [BLOCK_M, E]
        w_chosen = tl.sum(tl.where(gather_mask, weights, 0.0), axis=1)
        sum_topk += w_chosen

        out_off = m * TOPK + k
        tl.store(topk_w_ptr  + out_off, w_chosen.to(tl.bfloat16), mask=m_mask)
        tl.store(topk_id_ptr + out_off, idx_max.to(tl.int32),     mask=m_mask)

        weights = tl.where(gather_mask, NEG_INF, weights)

    # optional renormalize: divide stored weights by their row-sum
    if RENORMALIZE:
        for k in tl.static_range(TOPK):
            w_off = m * TOPK + k
            w = tl.load(topk_w_ptr + w_off, mask=m_mask, other=0.0).to(tl.float32)
            w = w / sum_topk
            tl.store(topk_w_ptr + w_off, w.to(tl.bfloat16), mask=m_mask)


def _launch_routing(score, topk, renormalize, M, E):
    """Allocate outputs and launch the routing kernel."""
    topk_w  = torch.empty(M, topk, dtype=score.dtype, device=score.device)
    topk_id = torch.empty(M, topk, dtype=torch.int32, device=score.device)

    E_pow2 = triton.next_power_of_2(E)
    BLOCK_M = 16

    grid = (triton.cdiv(M, BLOCK_M),)
    _routing_kernel[grid](
        score, topk_w, topk_id,
        M,
        RENORMALIZE=bool(renormalize),
        E=E_pow2,
        E_REAL=E,
        TOPK=topk,
        BLOCK_M=BLOCK_M,
    )
    return topk_w, topk_id


# ---------------------------------------------------------------------------
# Kernel B — gate+up + SiLU
# Inputs : hidden [M, K] bf16, w1 [E, 2N, K] bf16, topk_id [M, TOPK] int32
# Outputs: intermediate [M*TOPK, N] bf16
# Tensor-core grid : (M*TOPK, ceil(N / BLOCK_N))
# Ascend grid      : (ceil(N / BLOCK_N),)  — inner loop over M*TOPK
# ---------------------------------------------------------------------------
@triton.jit
def _gateup_silu_kernel(
    hidden_ptr,          # [M, K]
    w1_ptr,              # [E, 2N, K]
    topk_id_ptr,         # [M, TOPK]
    inter_ptr,           # [M*TOPK, N]
    M, MK,               # MK = M * TOPK
    K:     tl.constexpr,
    N:     tl.constexpr,
    TWO_N: tl.constexpr,
    TOPK:  tl.constexpr,
    BLOCK_MK: tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_K:  tl.constexpr,
    IS_ASCEND: tl.constexpr,
):
    if IS_ASCEND:
        # 1D grid, one program per N tile. Inner runtime loop over MK pairs.
        pid_n = tl.program_id(0)
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_off < N

        for mk in range(MK):
            m = mk // TOPK
            k_idx = mk % TOPK
            e = tl.load(topk_id_ptr + m * TOPK + k_idx)

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

            tl.store(inter_ptr + mk * N + n_off, out, mask=n_mask)
    else:
        # Tensor-core path: 2D grid (MK, n_tile). One program per token-expert pair × n tile.
        pid_mk = tl.program_id(0)
        pid_n  = tl.program_id(1)

        m     = pid_mk // TOPK
        k_idx = pid_mk % TOPK
        e = tl.load(topk_id_ptr + m * TOPK + k_idx)

        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_off < N

        acc_g = tl.zeros([BLOCK_MK, BLOCK_N], tl.float32)
        acc_u = tl.zeros([BLOCK_MK, BLOCK_N], tl.float32)

        for k_chunk in range(0, K, BLOCK_K):
            k_off = k_chunk + tl.arange(0, BLOCK_K)
            k_msk = k_off < K

            # Hidden tile: replicate row [m,:] across all BLOCK_MK lanes (only lane 0 is real).
            h_row = tl.load(hidden_ptr + m * K + k_off, mask=k_msk, other=0.0)
            h_tile = tl.broadcast_to(h_row[None, :], (BLOCK_MK, BLOCK_K))

            w1g_off = e * (TWO_N * K) + n_off[:, None] * K + k_off[None, :]
            w1g_msk = n_mask[:, None] & k_msk[None, :]
            w1g = tl.load(w1_ptr + w1g_off, mask=w1g_msk, other=0.0)

            w1u_off = e * (TWO_N * K) + (N + n_off)[:, None] * K + k_off[None, :]
            w1u_msk = n_mask[:, None] & k_msk[None, :]
            w1u = tl.load(w1_ptr + w1u_off, mask=w1u_msk, other=0.0)

            acc_g += tl.dot(h_tile, tl.trans(w1g), out_dtype=tl.float32)
            acc_u += tl.dot(h_tile, tl.trans(w1u), out_dtype=tl.float32)

        silu_g = acc_g * tl.sigmoid(acc_g)
        out = (silu_g * acc_u).to(tl.bfloat16)  # [BLOCK_MK, BLOCK_N]

        out_row0 = out[0, :]
        tl.store(inter_ptr + pid_mk * N + n_off, out_row0, mask=n_mask)


def _launch_gateup_silu(hidden, w1, topk_id, intermediate, M, K, N, topk):
    cfg = _gateup_cfg()
    BLOCK_MK = cfg["BLOCK_MK"]
    BLOCK_N  = cfg["BLOCK_N"]
    BLOCK_K  = cfg["BLOCK_K"]

    MK = M * topk

    if _is_ascend():
        grid = (triton.cdiv(N, BLOCK_N),)
    else:
        grid = (MK, triton.cdiv(N, BLOCK_N))

    _gateup_silu_kernel[grid](
        hidden, w1, topk_id, intermediate,
        M, MK,
        K=K, N=N, TWO_N=2 * N,
        TOPK=topk,
        BLOCK_MK=BLOCK_MK,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        IS_ASCEND=_is_ascend(),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    """Stub — kernels added in subsequent commits."""
    raise NotImplementedError("v1 kernels not yet wired")
