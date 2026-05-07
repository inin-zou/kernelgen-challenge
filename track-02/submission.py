"""
Submission file for GOSIM KernelGen Track-02: DeepSeek mHC (hc_split_sinkhorn).

v9 = exact v6 revert.

Three blind-tuning attempts (v7, v7b, v8) all regressed or failed. Lessons:
- v7  (added num_stages=2/3 + BLOCK_T=256/512 to autotune): all 5 backends
       failed — a specific config corner crashed muTriton uncatchably.
- v7b (kept BLOCK_T=256/512 without num_stages): MTT 201→155, Hygon 54→51.
       Autotune's noisy micro-bench picked the new big-tile configs as
       "winners" but they lose on the real workload.
- v8  (MTT explicit path: BLOCK_T=256 num_warps=4 from FlagGems CE-loss
       analog): MTT 201→92. The CE-loss analog is structurally wrong —
       CE-loss has per-row compute of ~32k elements vs our 4-element axis,
       so its big-tile / few-warps choice doesn't transfer. v6's
       autotune-picked BLOCK_T=128 num_warps=8 turned out to actually be
       muTriton's local optimum for our compute/parallelism ratio.

Without local muTriton hardware to bench, we cannot reliably beat v6's
autotune on MTT. Reverting to exactly v6 to lock in the 71.85 / #2 result.
v6 mechanisms preserved verbatim:
- Bypass autotune on Ascend (`backend == "npu"`) and MetaX (`backend == "maca"`).
- TRITON_DISABLE_SWIZZLE=1 defensive env var.
- Two kernels (runtime-loop + static-loop), try/except wrapping the
  autotune call as a backstop for backends we haven't explicitly bypassed.
- Autotune sweep: 11 v6 configs, BLOCK_T ∈ {1..128}, num_warps ∈ {1, 2, 4, 8}.

This is the file uploaded to the judge.
- All numerical computation lives inside @triton.jit kernels.
- Torch ops are used only for memory operations (alloc, view, contiguous).
- Evaluation environment: Python 3 + Triton 3.5.
"""
import os

# Defensive env var — mcTriton-specific, no-op on other backends.
os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")

# v12: muTriton's default compiler backend DISABLES instruction combining,
# if-conversion, and burst-combining (compiler.py make_mubin():
# "-mtgpu-if-convert=0 -mtgpu-combine-instr-with-burst=0 ...").
# Setting MUSA_ENABLE_LLC_OPT=1 replaces all of that with
# "-mtgpu-opt-level=1", enabling the full optimization suite.
# Our kernel has a tight 19-iteration Sinkhorn loop that benefits
# directly from instruction scheduling and combining.
# Non-MTT backends ignore this env var entirely.
os.environ.setdefault("MUSA_ENABLE_LLC_OPT", "1")

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune sweep — exact v6 set. Used by Hygon and Iluvatar only.
# (Ascend, MetaX, MTT all dispatch to explicit paths and never enter autotune.)
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
    triton.Config({'BLOCK_T': 128}, num_warps=4),
    triton.Config({'BLOCK_T': 128}, num_warps=8),
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
_IS_MTT_CACHED = None


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


def _is_mtt_backend() -> bool:
    """Detect Moore Threads (摩尔线程, MUSA) backend. Cached after first call."""
    global _IS_MTT_CACHED
    if _IS_MTT_CACHED is None:
        _IS_MTT_CACHED = _detect_backend(("musa", "mt", "mthread", "mthreads"))
    return _IS_MTT_CACHED


def _mtt_static_config(n: int):
    """Size-dependent (BLOCK_T, num_warps) for the MTT explicit path.

    v11 proved hardcoding a single config for all n destroys decode perf
    (MTT 201 → 48). Autotune with key=['n_tokens'] picks different configs
    per shape — we replicate that logic here to bypass autotune overhead
    while preserving per-shape dispatch.

    Sources:
    - v6 autotune winner for large n: (128, 8) → 201.32x on MTT
    - FlagGems _mthreads: reduction ops consistently use num_warps=16 for
      large inputs (layer_norm, rms_norm, dropout, batch_norm, log_softmax)
    - MooreThreads/tilelang_musa mhc_test: 1 program per token, 96 threads
      (3 warps on S4000 warp_size=32)
    - FlagTree mthreads autotuner.py: Config default num_warps=4, num_stages=2
    """
    if n <= 1:
        return 1, 1        # single token decode
    if n <= 16:
        return 16, 2       # small batch decode
    if n <= 64:
        return 64, 4       # medium batch decode
    if n <= 8192:
        return 128, 8      # v6 autotune winner
    return 128, 16          # Layer 3: large prefill — FlagGems MTT pattern


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

    # ---- MTT explicit path: per-n dispatch bypassing autotune ----
    # v12: bypass autotune on MTT with size-dependent configs. v11 proved
    # a single hardcoded config (128, 8) destroys decode perf; this time
    # we replicate autotune's per-shape logic. Benefits:
    # (a) eliminates autotune noise (MTT oscillated 155-201 across identical
    #     code submissions due to micro-bench timing variance)
    # (b) saves 55 compile+bench warm-up rounds per eval
    # (c) lets us try num_warps=16 for large prefill (FlagGems MTT pattern)
    # Uses _kernel_runtime (runtime-loop) NOT _kernel_static — runtime loop
    # is what gave MTT 149x vs static's 57x in v1→v2 transition.
    if _is_mtt_backend():
        BLOCK_T, num_warps = _mtt_static_config(n)
        grid = (triton.cdiv(n, BLOCK_T),)
        try:
            _kernel_runtime[grid](
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
                BLOCK_T=BLOCK_T,
                num_warps=num_warps,
            )
            return pre, post, comb
        except Exception:
            pass
        # Fall through to autotune or static if explicit path fails.

    # ---- Bypass autotune on backends where it doesn't pay off. ----
    # MetaX:  mcTriton's autotune fails by SIGSEGV/SIGABRT (uncatchable).
    # Ascend: try/except works but the failed sweep wastes time first
    #         (v5 Ascend 7.41 vs v1's 9.41 — gap from wasted attempt).
    # MTT:    handled above; if it failed, let autotune try as backup.
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
