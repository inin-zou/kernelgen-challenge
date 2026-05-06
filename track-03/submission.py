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


def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    """Stub — kernels added in subsequent commits."""
    raise NotImplementedError("v1 kernels not yet wired")
