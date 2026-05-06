"""
Submission file for GOSIM KernelGen Track-02: DeepSeek mHC (hc_split_sinkhorn).

v6 changes vs v5:
- **Also bypass autotune on Ascend.** v5 caught Ascend's autotune failure
  via try/except — correct, but the failed sweep itself wastes time
  compiling configs before raising, hurting Ascend's score (7.41 in v5
  vs 9.41 in v1 with no autotune attempt at all). Same pattern as MetaX:
  detect Ascend (`backend == "npu"`) and route directly to static.
- **Re-add BLOCK_T=128 autotune configs.** Originally dropped to avoid
  UB OVERFLOW on Ascend/MetaX during autotune; now those backends never
  enter the autotune path so the constraint is gone. Hygon, Iluvatar,
  and MTT (the only autotune consumers now) all use NV-style upstream
  Triton which gracefully skips configs that fail to compile. v2 had
  MTT=149 with BLOCK_T=128 in the sweep; v5 dropped to 146 without it.

v5 mechanisms preserved:
- MetaX detection + bypass (canonical mcTriton check).
- TRITON_DISABLE_SWIZZLE=1 defensive env var.
- Two kernels (runtime-loop + static-loop), try/except still wrapping the
  autotune call as a backstop for backends we haven't explicitly bypassed.

This is the file uploaded to the judge.
- All numerical computation lives inside @triton.jit kernels.
- Torch ops are used only for memory operations (alloc, view, contiguous).
- Evaluation environment: Python 3 + Triton 3.5.
"""
import os

# Defensive env var — mcTriton-specific, no-op on other backends.
# Disables shared-memory swizzle, sometimes avoids cpasync crashes on MetaX.
os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune sweep.
# v6: BLOCK_T=128 configs restored. Previously dropped to keep Ascend/MetaX
# autotune from blowing up on UB OVERFLOW — but those two backends now
# bypass autotune entirely (see _is_metax_backend / _is_ascend_backend).
# The remaining autotune consumers (Hygon, Iluvatar, MTT) are NV-style
# backends whose upstream Triton autotune gracefully skips any config that
# fails to compile, so adding speculative large-tile configs is upside-only.
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_T': 1},   num_warps=1),
    triton.Config({'BLOCK_T': 4},   num_warps=1),
    triton.Config({'BLOCK_T': 8},   num_warps=1),
    triton.Config({'BLOCK_T': 16},  num_warps=1),
    triton.Config({'BLOCK_T': 16},  num_warps=2),
    triton.Config({'BLOCK_T': 32},  num_warps=2),
    triton.Config({'BLOCK_T': 32},  num_warps=4),
    triton.Config({'BLOCK_T': 64},  num_warps=2),
    triton.Config({'BLOCK_T': 64},  num_warps=4),
    triton.Config({'BLOCK_T': 128}, num_warps=4),    # restored — likely MTT optimum (v2 had 149)
    triton.Config({'BLOCK_T': 128}, num_warps=8),    # speculative — more parallelism on MTT
]


# ---------------------------------------------------------------------------
# Kernel #1 — runtime-loop variant
# Used through @triton.autotune. Wins big on Moore Threads muTriton (149x in
# v2/v3) by avoiding 19-iter compile-time unroll bloat. Loop count is passed
# as a *non-constexpr* int (n_extra_iters) so the compiler is forced to emit
# a runtime loop without depending on the `tl.range` API directly.
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_runtime(
    mixes_ptr, scale_ptr, base_ptr,
    pre_ptr, post_ptr, comb_ptr,
    n_tokens,
    eps,
    n_extra_iters,                      # non-constexpr → runtime loop
    HC: tl.constexpr,
    HC2: tl.constexpr,
    HC3: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = tok < n_tokens

    h = tl.arange(0, HC)
    r = tl.arange(0, HC)
    c = tl.arange(0, HC)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    base_pre  = tl.load(base_ptr + h)
    base_post = tl.load(base_ptr + HC + h)
    base_comb = tl.load(base_ptr + 2 * HC + r[:, None] * HC + c[None, :])

    pre_off = tok[:, None] * HC3 + h[None, :]
    pre_in  = tl.load(mixes_ptr + pre_off, mask=tmask[:, None], other=0.0)
    pre_out = tl.sigmoid(pre_in * s0 + base_pre[None, :]) + eps

    post_off = tok[:, None] * HC3 + HC + h[None, :]
    post_in  = tl.load(mixes_ptr + post_off, mask=tmask[:, None], other=0.0)
    post_out = 2.0 * tl.sigmoid(post_in * s1 + base_post[None, :])

    comb_off = (
        tok[:, None, None] * HC3
        + 2 * HC
        + r[None, :, None] * HC
        + c[None, None, :]
    )
    comb_in = tl.load(mixes_ptr + comb_off, mask=tmask[:, None, None], other=0.0)
    comb = comb_in * s2 + base_comb[None, :, :]

    # softmax(comb, dim=-1) + eps
    row_max = tl.max(comb, axis=2)
    comb = tl.exp(comb - row_max[:, :, None])
    row_sum = tl.sum(comb, axis=2)
    comb = comb / row_sum[:, :, None] + eps

    # First column-normalize
    col_sum = tl.sum(comb, axis=1)
    comb = comb / (col_sum[:, None, :] + eps)

    # (sinkhorn_iters - 1) × (row, col) — runtime loop
    for _ in range(n_extra_iters):
        row_sum = tl.sum(comb, axis=2)
        comb = comb / (row_sum[:, :, None] + eps)

        col_sum = tl.sum(comb, axis=1)
        comb = comb / (col_sum[:, None, :] + eps)

    pre_st_off  = tok[:, None] * HC + h[None, :]
    post_st_off = tok[:, None] * HC + h[None, :]
    comb_st_off = tok[:, None, None] * HC2 + r[None, :, None] * HC + c[None, None, :]
    tl.store(pre_ptr  + pre_st_off,  pre_out,  mask=tmask[:, None])
    tl.store(post_ptr + post_st_off, post_out, mask=tmask[:, None])
    tl.store(comb_ptr + comb_st_off, comb,     mask=tmask[:, None, None])


# Autotune-wrapped runtime-loop kernel.
_kernel_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS, key=['n_tokens'],
)(_kernel_runtime)


# ---------------------------------------------------------------------------
# Kernel #2 — static-loop fallback (mirrors v1 design)
# No autotune. SINKHORN_ITERS is constexpr; the loop is compile-time unrolled
# via tl.static_range. Used only when the autotuned path raises an exception
# (e.g. on Triton-Ascend / mcTriton where autotune or runtime loops misbehave).
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_static(
    mixes_ptr, scale_ptr, base_ptr,
    pre_ptr, post_ptr, comb_ptr,
    n_tokens,
    eps,
    HC: tl.constexpr,
    HC2: tl.constexpr,
    HC3: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = tok < n_tokens

    h = tl.arange(0, HC)
    r = tl.arange(0, HC)
    c = tl.arange(0, HC)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    base_pre  = tl.load(base_ptr + h)
    base_post = tl.load(base_ptr + HC + h)
    base_comb = tl.load(base_ptr + 2 * HC + r[:, None] * HC + c[None, :])

    pre_off = tok[:, None] * HC3 + h[None, :]
    pre_in  = tl.load(mixes_ptr + pre_off, mask=tmask[:, None], other=0.0)
    pre_out = tl.sigmoid(pre_in * s0 + base_pre[None, :]) + eps

    post_off = tok[:, None] * HC3 + HC + h[None, :]
    post_in  = tl.load(mixes_ptr + post_off, mask=tmask[:, None], other=0.0)
    post_out = 2.0 * tl.sigmoid(post_in * s1 + base_post[None, :])

    comb_off = (
        tok[:, None, None] * HC3
        + 2 * HC
        + r[None, :, None] * HC
        + c[None, None, :]
    )
    comb_in = tl.load(mixes_ptr + comb_off, mask=tmask[:, None, None], other=0.0)
    comb = comb_in * s2 + base_comb[None, :, :]

    row_max = tl.max(comb, axis=2)
    comb = tl.exp(comb - row_max[:, :, None])
    row_sum = tl.sum(comb, axis=2)
    comb = comb / row_sum[:, :, None] + eps

    col_sum = tl.sum(comb, axis=1)
    comb = comb / (col_sum[:, None, :] + eps)

    # Compile-time unrolled — known-good across all backends per v1 results.
    for _ in tl.static_range(SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb, axis=2)
        comb = comb / (row_sum[:, :, None] + eps)

        col_sum = tl.sum(comb, axis=1)
        comb = comb / (col_sum[:, None, :] + eps)

    pre_st_off  = tok[:, None] * HC + h[None, :]
    post_st_off = tok[:, None] * HC + h[None, :]
    comb_st_off = tok[:, None, None] * HC2 + r[None, :, None] * HC + c[None, None, :]
    tl.store(pre_ptr  + pre_st_off,  pre_out,  mask=tmask[:, None])
    tl.store(post_ptr + post_st_off, post_out, mask=tmask[:, None])
    tl.store(comb_ptr + comb_st_off, comb,     mask=tmask[:, None, None])


# ---------------------------------------------------------------------------
# Process-level flags — set once and reused.
# ---------------------------------------------------------------------------
_AUTOTUNE_OK = True
_IS_METAX_CACHED = None
_IS_ASCEND_CACHED = None


def _detect_backend(target_names) -> bool:
    """Backend detection using Triton's runtime API only.
    No torch ops here — strictly compliant with the contest rule that torch
    is reserved for memory operations. mcTriton's own matmul tutorial uses
    exactly this check (`triton.runtime.driver.active.get_current_target().backend`).
    """
    try:
        target = triton.runtime.driver.active.get_current_target()
        backend = getattr(target, "backend", "")
        return backend in target_names
    except Exception:
        return False


def _is_metax_backend() -> bool:
    """Detect MetaX (沐曦) backend. Cached after first call."""
    global _IS_METAX_CACHED
    if _IS_METAX_CACHED is None:
        _IS_METAX_CACHED = _detect_backend(("maca",))
    return _IS_METAX_CACHED


def _is_ascend_backend() -> bool:
    """Detect Huawei Ascend (昇腾) backend. Cached after first call."""
    global _IS_ASCEND_CACHED
    if _IS_ASCEND_CACHED is None:
        _IS_ASCEND_CACHED = _detect_backend(("npu", "ascend", "ascendc"))
    return _IS_ASCEND_CACHED


def _static_config(n: int):
    """v1-replicated (BLOCK_T, num_warps) for the static path.
    v1 used BLOCK_T=16, num_warps=1 for ALL shapes and got 9.41x on Ascend
    and 21.66x on MetaX. v4 tried being clever with per-n dispatch and saw
    Ascend regress to 7.75x. Lesson: when in doubt, copy v1 exactly."""
    return 16, 1


# ---------------------------------------------------------------------------
# Wrapper — entry point called by the judge
# ---------------------------------------------------------------------------
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
        hc_mult:        int, number of streams (default 4)
        sinkhorn_iters: int, iterations (default 20)
        eps:            float (default 1e-6)

    Returns:
        pre:    [b, s, hc_mult]
        post:   [b, s, hc_mult]
        comb:   [b, s, hc_mult, hc_mult]
    """
    global _AUTOTUNE_OK

    assert mixes.dim() == 3, f"mixes must be 3D, got shape {tuple(mixes.shape)}"
    assert hc_mult == 4, f"this kernel is specialized for hc_mult=4, got {hc_mult}"
    b, s, m = mixes.shape
    HC  = hc_mult
    HC2 = HC * HC
    HC3 = 2 * HC + HC2
    assert m == HC3, f"mixes last dim must be {HC3}, got {m}"

    mixes_c    = mixes.contiguous()
    hc_scale_c = hc_scale.contiguous()
    hc_base_c  = hc_base.contiguous()

    n = b * s
    pre  = torch.empty(b, s, HC,     dtype=mixes.dtype, device=mixes.device)
    post = torch.empty(b, s, HC,     dtype=mixes.dtype, device=mixes.device)
    comb = torch.empty(b, s, HC, HC, dtype=mixes.dtype, device=mixes.device)

    n_extra_iters = sinkhorn_iters - 1

    # ---- Bypass autotune on backends where it doesn't pay off. ----
    # MetaX:  mcTriton's autotune fails by SIGSEGV/SIGABRT (uncatchable).
    # Ascend: try/except works but the failed sweep wastes time first
    #         (v5 Ascend 7.41 vs v1's 9.41 — gap from wasted attempt).
    # Both go straight to the static v1-style path.
    skip_autotune = _is_metax_backend() or _is_ascend_backend()

    # ---- Primary path: autotuned runtime-loop kernel ----
    if _AUTOTUNE_OK and not skip_autotune:
        try:
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK_T']),)
            _kernel_autotuned[grid](
                mixes_c.view(n, HC3),
                hc_scale_c,
                hc_base_c,
                pre.view(n, HC),
                post.view(n, HC),
                comb.view(n, HC, HC),
                n,
                eps,
                n_extra_iters,
                HC=HC,
                HC2=HC2,
                HC3=HC3,
            )
            return pre, post, comb
        except Exception:
            # Autotune machinery and/or runtime loop failed on this backend.
            # Disable it process-wide and fall through to the static path.
            _AUTOTUNE_OK = False

    # ---- Fallback: static-config kernel with compile-time unrolled loop ----
    BLOCK_T, num_warps = _static_config(n)
    grid = (triton.cdiv(n, BLOCK_T),)
    _kernel_static[grid](
        mixes_c.view(n, HC3),
        hc_scale_c,
        hc_base_c,
        pre.view(n, HC),
        post.view(n, HC),
        comb.view(n, HC, HC),
        n,
        eps,
        HC=HC,
        HC2=HC2,
        HC3=HC3,
        SINKHORN_ITERS=sinkhorn_iters,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
    )
    return pre, post, comb
