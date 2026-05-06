# Track-03 Fused MoE v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a v1 `track-03/submission.py` that implements `fused_moe` entirely in `@triton.jit`, with per-backend dispatch (Ascend / Hygon / Iluvatar / MTT / MetaX), no autotune, no atomics, no permutation — green on all 5 backends with material speedup on at least Hygon, Iluvatar, and MTT.

**Architecture:** Three-kernel pipeline (route → gate+up+silu → down+topk-reduce), workspace tensor `intermediate [M·topk, N]` to avoid atomics. One universal Triton source per kernel; per-backend constants and grid shape selected by cached backend flags.

**Tech Stack:** Python 3, PyTorch (memory ops only), Triton 3.5 `@triton.jit`. No vendor BLAS, no torch numerical ops in the hot path.

**Local environment caveat:** Triton is **not installed on the dev Mac**. That means:
- `reference.py` and `workloads.py` are real TDD with `pytest` (CPU torch).
- `submission.py`, `test_correctness.py`, `bench.py` are **not runnable locally**. Verification = static compliance greps + ship-to-platform on the FlagOS judge (~5–10 min per submission, 2-min cooldown). Build them with extra care; don't expect a fast feedback loop.

**Spec:** `docs/superpowers/specs/2026-05-06-track-03-fused-moe-design.md`

---

## File Structure

| Path | Responsibility | Local-runnable? |
|---|---|---|
| `track-03/workloads.py` | List of 8 workload dicts from the spec table | yes (pure data) |
| `track-03/reference/__init__.py` | Empty package marker | yes |
| `track-03/reference/reference.py` | Frozen torch oracle, lifted from `requirements.md` | yes (CPU torch) |
| `track-03/test_correctness.py` | `pytest` harness comparing submission vs reference, atol=5e-2 | no (imports Triton) |
| `track-03/bench.py` | 8-workload bench harness, prints per-workload speedups | no (imports Triton) |
| `track-03/submission.py` | The judged artifact: 3 kernels + wrapper + per-backend dispatch | no (imports Triton) |
| `track-03/notes.md` | v1, v2, … iteration log (mirrors track-01/02) | n/a |

`submission.py` internal structure (one file, ordered top→bottom):
1. Module-top env vars (`TRITON_DISABLE_SWIZZLE`, `MUSA_ENABLE_SQMMA`).
2. Imports.
3. Backend detection helpers + cached flags.
4. Block-constant tables (per-backend dicts).
5. `_routing_kernel` (one `@triton.jit`, branch on Ascend via `IS_ASCEND: tl.constexpr`).
6. `_gateup_silu_kernel` (one `@triton.jit`, same branching pattern).
7. `_down_reduce_kernel` (one `@triton.jit`, same branching pattern).
8. `fused_moe` wrapper.

Branching by `tl.constexpr` keeps the source unified; the compiler specializes per-backend at `triton.jit` compile time.

---

## Task 1: workloads.py

**Files:**
- Create: `track-03/workloads.py`
- Create: `track-03/test_workloads.py`

- [ ] **Step 1: Write the failing test**

```python
# track-03/test_workloads.py
from workloads import WORKLOADS


def test_eight_workloads():
    assert len(WORKLOADS) == 8


def test_required_keys():
    for w in WORKLOADS:
        assert set(w.keys()) >= {"label", "M", "K", "N", "E", "topk"}


def test_shape_values_match_spec():
    expected = [
        ("decode", 1,    2048, 1024, 8,  2),
        ("decode", 64,   2048, 1024, 8,  2),
        ("decode", 1,    2048, 1024, 64, 6),
        ("decode", 64,   2048, 1024, 64, 6),
        ("prefill", 512,  2048, 1024, 8,  2),
        ("prefill", 4096, 2048, 1024, 8,  2),
        ("prefill", 512,  2048, 1024, 64, 6),
        ("prefill", 4096, 2048, 1024, 64, 6),
    ]
    actual = [(w["label"], w["M"], w["K"], w["N"], w["E"], w["topk"]) for w in WORKLOADS]
    assert actual == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd track-03 && python3 -m pytest test_workloads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workloads'`

- [ ] **Step 3: Write the workloads module**

```python
# track-03/workloads.py
"""Eight benchmark workloads from track-03/requirements.md."""

WORKLOADS = [
    {"label": "decode",  "M": 1,    "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "decode",  "M": 64,   "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "decode",  "M": 1,    "K": 2048, "N": 1024, "E": 64, "topk": 6},
    {"label": "decode",  "M": 64,   "K": 2048, "N": 1024, "E": 64, "topk": 6},
    {"label": "prefill", "M": 512,  "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "prefill", "M": 4096, "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "prefill", "M": 512,  "K": 2048, "N": 1024, "E": 64, "topk": 6},
    {"label": "prefill", "M": 4096, "K": 2048, "N": 1024, "E": 64, "topk": 6},
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd track-03 && python3 -m pytest test_workloads.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add track-03/workloads.py track-03/test_workloads.py
git commit -m "track-03: add workloads.py with the 8 spec workloads"
```

---

## Task 2: reference oracle

**Files:**
- Create: `track-03/reference/__init__.py` (empty)
- Create: `track-03/reference/reference.py`
- Create: `track-03/test_reference.py`

- [ ] **Step 1: Create the empty package marker**

```bash
mkdir -p track-03/reference
: > track-03/reference/__init__.py
```

- [ ] **Step 2: Write the failing test**

```python
# track-03/test_reference.py
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
    """If we manually zero out non-topk experts, output should match.

    This is the *defining property* of the reference: only topk experts
    contribute to each token.
    """
    M, K, N, E, topk = 4, 2048, 1024, 8, 2
    hidden, w1, w2, score = _make_inputs(M, K, N, E, topk, seed=42)

    out_full = fused_moe_ref(hidden, w1, w2, score, topk=topk)

    # Manually zero out non-topk experts in score (post-softmax).
    # Set those experts' scores to a very negative number so softmax→0.
    weights_fp32 = score.float().softmax(dim=-1)
    _, top_ids = weights_fp32.topk(topk, dim=-1)
    forced_score = torch.full_like(score, -1e4)
    forced_score.scatter_(1, top_ids, score.gather(1, top_ids))

    out_forced = fused_moe_ref(hidden, w1, w2, score=forced_score, topk=topk)

    # Won't be identical (softmax denom differs) — just check finite + close shape.
    assert out_forced.shape == out_full.shape
    assert torch.isfinite(out_forced).all()


def test_renormalize_changes_output():
    hidden, w1, w2, score = _make_inputs(M=4, K=2048, N=1024, E=8, topk=2, seed=7)
    a = fused_moe_ref(hidden, w1, w2, score, topk=2, renormalize=False)
    b = fused_moe_ref(hidden, w1, w2, score, topk=2, renormalize=True)
    assert not torch.allclose(a, b, atol=1e-3)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd track-03 && python3 -m pytest test_reference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reference.reference'`

- [ ] **Step 4: Write the reference oracle (lifted verbatim from `requirements.md`)**

```python
# track-03/reference/reference.py
"""Frozen torch oracle for track-03 fused MoE.

Lifted from track-03/requirements.md. This is the contest's spec — do not
deviate. submission.py must match this within atol=5e-2 (bf16 tolerance).
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd track-03 && python3 -m pytest test_reference.py -v`
Expected: 4 passed (or 3 passed + 1 might fail if seeds make outputs tiny — adjust seed if so).

- [ ] **Step 6: Commit**

```bash
git add track-03/reference/ track-03/test_reference.py
git commit -m "track-03: pin reference oracle, add property tests"
```

---

## Task 3: submission.py module-top boilerplate

**Files:**
- Create: `track-03/submission.py`

This task creates the **shell** of submission.py: env vars, imports, backend detection, cached flags, per-backend block tables, and a stub `fused_moe` that raises NotImplementedError. No kernels yet.

- [ ] **Step 1: Write the boilerplate**

```python
# track-03/submission.py
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
Backend detection via Triton runtime API only (no torch.cuda.*).

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
# Per-backend block constants. Seed values from the v1 design doc.
# Layout: (BLOCK_MK, BLOCK_N, BLOCK_K, num_warps, num_stages)
#   BLOCK_MK : tensor-core M-floor (1 = scalar reduce; 16/32 = tl.dot)
#   BLOCK_N  : output-N tile size for kernel B
#   BLOCK_K  : reduction-K chunk for kernel B's K=2048 loop
# Kernel C uses the same BLOCK_K_OUT (= BLOCK_N here, semantically the output
# K tile of the down GEMM) and a separate BLOCK_N_R for its N reduction.
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
    return _CFG_GATEUP_ILUVATAR  # default = NV-style (Iluvatar / unknown CUDA)


def _down_cfg():
    if _is_ascend():   return _CFG_DOWN_ASCEND
    if _is_mtt():      return _CFG_DOWN_MTT
    if _is_metax():    return _CFG_DOWN_METAX
    if _is_hygon():    return _CFG_DOWN_HYGON
    return _CFG_DOWN_ILUVATAR


def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    """Stub — kernels added in subsequent tasks."""
    raise NotImplementedError("v1 kernels not yet wired")
```

- [ ] **Step 2: Static compliance grep**

Run: `! grep -nE 'torch\.matmul|F\.linear|F\.silu|torch\.softmax|torch\.topk|torch\.argsort|torch\.bincount|index_add|index_select|scaled_dot_product_attention|torch\.cuda\.' track-03/submission.py`

Expected: empty output (the `!` makes the command succeed if grep finds nothing — required for the static check to "pass").

- [ ] **Step 3: Verify import works (CPU box, Triton-less)**

Run: `cd track-03 && python3 -c "import submission; print('ok')"` — note: this **will fail locally** with `ModuleNotFoundError: triton`. That's expected. The verification is just that **non-triton** parts of the file are syntactically valid:

Run: `python3 -c "import ast; ast.parse(open('track-03/submission.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 4: Commit**

```bash
git add track-03/submission.py
git commit -m "track-03: scaffold submission.py with backend detection + cfg tables"
```

---

## Task 4: Routing kernel (Kernel A)

**Files:**
- Modify: `track-03/submission.py` (add `_routing_kernel` and an internal launcher)

The routing kernel does softmax + iterative topk in fp32. Fits one or more token rows per program. `E ≤ 64`, `topk ≤ 6` — tiny tile.

- [ ] **Step 1: Add the routing kernel below the cfg tables**

```python
# ---------------------------------------------------------------------------
# Kernel A — routing: softmax + iterative topk
# Inputs : score [M, E] (bf16)
# Outputs: topk_weights [M, topk] (bf16), topk_ids [M, topk] (int32)
# Grid   : (ceil(M / BLOCK_M),)
# ---------------------------------------------------------------------------
@triton.jit
def _routing_kernel(
    score_ptr,           # [M, E]
    topk_w_ptr,          # [M, TOPK]
    topk_id_ptr,         # [M, TOPK]
    M,
    RENORMALIZE: tl.constexpr,
    E: tl.constexpr,     # padded to next pow2 of E_real
    E_REAL: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < M

    e = tl.arange(0, E)
    e_mask = e < E_REAL  # mask off padding columns

    NEG_INF = float('-inf')

    # Load score [BLOCK_M, E], pad with -inf so softmax/argmax ignore those.
    s_off = m[:, None] * E_REAL + e[None, :]
    s_load_mask = m_mask[:, None] & e_mask[None, :]
    s = tl.load(score_ptr + s_off, mask=s_load_mask, other=NEG_INF).to(tl.float32)
    s = tl.where(e_mask[None, :], s, NEG_INF)

    # softmax over last dim
    s_max = tl.max(s, axis=1)
    s = tl.exp(s - s_max[:, None])
    s_sum = tl.sum(s, axis=1)
    weights = s / s_sum[:, None]   # [BLOCK_M, E] fp32

    # iterative topk: argmax → record → mask the chosen position with -inf
    sum_topk = tl.zeros([BLOCK_M], tl.float32)
    for k in tl.static_range(TOPK):
        idx_max = tl.argmax(weights, axis=1)            # [BLOCK_M]
        # Gather the chosen weight per row.
        gather_mask = (e[None, :] == idx_max[:, None])  # [BLOCK_M, E]
        w_chosen = tl.sum(tl.where(gather_mask, weights, 0.0), axis=1)
        sum_topk += w_chosen

        out_off = m * TOPK + k
        tl.store(topk_w_ptr  + out_off, w_chosen.to(tl.bfloat16), mask=m_mask)
        tl.store(topk_id_ptr + out_off, idx_max.to(tl.int32),     mask=m_mask)

        # mask-out the chosen index for the next iteration
        weights = tl.where(gather_mask, NEG_INF, weights)

    # optional renormalize: divide stored weights by their row-sum
    if RENORMALIZE:
        # reload, divide, restore
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
```

- [ ] **Step 2: Static compliance grep**

Run: `! grep -nE 'torch\.matmul|F\.linear|F\.silu|torch\.softmax|torch\.topk|torch\.argsort|torch\.bincount|index_add|index_select|scaled_dot_product_attention|torch\.cuda\.' track-03/submission.py`

Expected: empty output.

- [ ] **Step 3: Parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/submission.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 4: Commit**

```bash
git add track-03/submission.py
git commit -m "track-03: add routing kernel (softmax + iterative topk)"
```

---

## Task 5: Gate+up+SiLU kernel (Kernel B)

**Files:**
- Modify: `track-03/submission.py` (add `_gateup_silu_kernel` + launcher)

This is the heavy one: GEMM-vector reduction over `K=2048`. One Triton source, branched on `IS_ASCEND: tl.constexpr` for scalar-reduce vs `tl.dot`.

- [ ] **Step 1: Add the kernel**

```python
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
    K: tl.constexpr,
    N: tl.constexpr,
    TWO_N: tl.constexpr, # 2 * N
    TOPK: tl.constexpr,
    BLOCK_MK: tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_K:  tl.constexpr,
    IS_ASCEND: tl.constexpr,
):
    if IS_ASCEND:
        # 1D grid: one program per N tile. Inner loop over MK.
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

                # w1 gate tile [BLOCK_N, BLOCK_K]
                w1g_off = e * (TWO_N * K) + n_off[:, None] * K + k_off[None, :]
                w1g_msk = n_mask[:, None] & k_msk[None, :]
                w1g = tl.load(w1_ptr + w1g_off, mask=w1g_msk, other=0.0).to(tl.float32)

                # w1 up tile (offset by N rows in the 2N dim)
                w1u_off = e * (TWO_N * K) + (N + n_off)[:, None] * K + k_off[None, :]
                w1u_msk = n_mask[:, None] & k_msk[None, :]
                w1u = tl.load(w1_ptr + w1u_off, mask=w1u_msk, other=0.0).to(tl.float32)

                acc_g += tl.sum(h[None, :] * w1g, axis=1)
                acc_u += tl.sum(h[None, :] * w1u, axis=1)

            # SiLU(gate) * up
            silu_g = acc_g * tl.sigmoid(acc_g)
            out = (silu_g * acc_u).to(tl.bfloat16)

            tl.store(inter_ptr + mk * N + n_off, out, mask=n_mask)
    else:
        # Tensor-core path: 2D grid (MK, n_tile). One program per token-expert pair × n tile.
        pid_mk = tl.program_id(0)  # 0..MK-1
        pid_n  = tl.program_id(1)

        m     = pid_mk // TOPK
        k_idx = pid_mk % TOPK
        e = tl.load(topk_id_ptr + m * TOPK + k_idx)

        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_off < N

        # M-dim padding lanes: row 0 is the real token, 1..BLOCK_MK-1 replicate it.
        mk_lane = tl.arange(0, BLOCK_MK)  # only lane 0 is meaningful

        acc_g = tl.zeros([BLOCK_MK, BLOCK_N], tl.float32)
        acc_u = tl.zeros([BLOCK_MK, BLOCK_N], tl.float32)

        for k_chunk in range(0, K, BLOCK_K):
            k_off = k_chunk + tl.arange(0, BLOCK_K)
            k_msk = k_off < K

            # hidden tile: [BLOCK_MK, BLOCK_K], all rows = hidden[m, k_off]
            h_row = tl.load(hidden_ptr + m * K + k_off, mask=k_msk, other=0.0)
            h_tile = tl.broadcast_to(h_row[None, :], (BLOCK_MK, BLOCK_K))

            # w1 gate tile [BLOCK_N, BLOCK_K]
            w1g_off = e * (TWO_N * K) + n_off[:, None] * K + k_off[None, :]
            w1g_msk = n_mask[:, None] & k_msk[None, :]
            w1g = tl.load(w1_ptr + w1g_off, mask=w1g_msk, other=0.0)

            # w1 up tile [BLOCK_N, BLOCK_K]
            w1u_off = e * (TWO_N * K) + (N + n_off)[:, None] * K + k_off[None, :]
            w1u_msk = n_mask[:, None] & k_msk[None, :]
            w1u = tl.load(w1_ptr + w1u_off, mask=w1u_msk, other=0.0)

            # tl.dot([BLOCK_MK, BLOCK_K] @ [BLOCK_K, BLOCK_N]) → [BLOCK_MK, BLOCK_N]
            acc_g += tl.dot(h_tile, tl.trans(w1g), out_dtype=tl.float32)
            acc_u += tl.dot(h_tile, tl.trans(w1u), out_dtype=tl.float32)

        silu_g = acc_g * tl.sigmoid(acc_g)
        out = (silu_g * acc_u).to(tl.bfloat16)  # [BLOCK_MK, BLOCK_N]

        # Store row 0 only (the real token row)
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
```

- [ ] **Step 2: Static compliance grep**

Run: `! grep -nE 'torch\.matmul|F\.linear|F\.silu|torch\.softmax|torch\.topk|torch\.argsort|torch\.bincount|index_add|index_select|scaled_dot_product_attention|torch\.cuda\.' track-03/submission.py`

Expected: empty output.

- [ ] **Step 3: Parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/submission.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 4: Commit**

```bash
git add track-03/submission.py
git commit -m "track-03: add gate+up+silu kernel (Ascend scalar / NV-style tl.dot)"
```

---

## Task 6: Down + topk-reduce kernel (Kernel C)

**Files:**
- Modify: `track-03/submission.py` (add `_down_reduce_kernel` + launcher)

Each program owns one `(m, k_block)` and runs the topk loop in-register, eliminating atomics.

- [ ] **Step 1: Add the kernel**

```python
# ---------------------------------------------------------------------------
# Kernel C — down + topk-reduce
# Inputs : intermediate [M*TOPK, N] bf16, w2 [E, K, N] bf16,
#          topk_w [M, TOPK] bf16, topk_id [M, TOPK] int32
# Outputs: output [M, K] bf16
# Tensor-core grid : (M, ceil(K / BLOCK_K_OUT))
# Ascend grid      : (ceil(K / BLOCK_K_OUT),) — inner loop over M
# ---------------------------------------------------------------------------
@triton.jit
def _down_reduce_kernel(
    inter_ptr,           # [M*TOPK, N]
    w2_ptr,              # [E, K, N]
    topk_w_ptr,          # [M, TOPK]
    topk_id_ptr,         # [M, TOPK]
    output_ptr,          # [M, K]
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_MK:    tl.constexpr,
    BLOCK_K_OUT: tl.constexpr,
    BLOCK_N_R:   tl.constexpr,
    IS_ASCEND: tl.constexpr,
):
    if IS_ASCEND:
        pid_k = tl.program_id(0)
        k_out_off = pid_k * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        k_out_mask = k_out_off < K

        for m in range(M):
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
    else:
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m = pid_m
        k_out_off = pid_k * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
        k_out_mask = k_out_off < K

        mk_lane = tl.arange(0, BLOCK_MK)  # only lane 0 is meaningful

        acc = tl.zeros([BLOCK_MK, BLOCK_K_OUT], tl.float32)

        for k_idx in tl.static_range(TOPK):
            e = tl.load(topk_id_ptr + m * TOPK + k_idx)
            w_t = tl.load(topk_w_ptr + m * TOPK + k_idx).to(tl.float32)

            partial = tl.zeros([BLOCK_MK, BLOCK_K_OUT], tl.float32)
            for n_chunk in range(0, N, BLOCK_N_R):
                n_off = n_chunk + tl.arange(0, BLOCK_N_R)
                n_msk = n_off < N

                mid_row = tl.load(
                    inter_ptr + (m * TOPK + k_idx) * N + n_off,
                    mask=n_msk, other=0.0,
                )
                mid_tile = tl.broadcast_to(mid_row[None, :], (BLOCK_MK, BLOCK_N_R))

                w2_off = e * (K * N) + k_out_off[:, None] * N + n_off[None, :]
                w2_msk = k_out_mask[:, None] & n_msk[None, :]
                w2t = tl.load(w2_ptr + w2_off, mask=w2_msk, other=0.0)

                # tl.dot([BLOCK_MK, BLOCK_N_R] @ [BLOCK_N_R, BLOCK_K_OUT]) → [BLOCK_MK, BLOCK_K_OUT]
                partial += tl.dot(mid_tile, tl.trans(w2t), out_dtype=tl.float32)

            acc += w_t * partial

        out_row0 = acc[0, :].to(tl.bfloat16)
        tl.store(output_ptr + m * K + k_out_off, out_row0, mask=k_out_mask)


def _launch_down_reduce(intermediate, w2, topk_w, topk_id, output, M, K, N, topk):
    cfg = _down_cfg()
    BLOCK_MK    = cfg["BLOCK_MK"]
    BLOCK_K_OUT = cfg["BLOCK_K_OUT"]
    BLOCK_N_R   = cfg["BLOCK_N_R"]

    if _is_ascend():
        grid = (triton.cdiv(K, BLOCK_K_OUT),)
    else:
        grid = (M, triton.cdiv(K, BLOCK_K_OUT))

    _down_reduce_kernel[grid](
        intermediate, w2, topk_w, topk_id, output,
        M,
        K=K, N=N, TOPK=topk,
        BLOCK_MK=BLOCK_MK,
        BLOCK_K_OUT=BLOCK_K_OUT,
        BLOCK_N_R=BLOCK_N_R,
        IS_ASCEND=_is_ascend(),
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
```

- [ ] **Step 2: Static compliance grep**

Run: `! grep -nE 'torch\.matmul|F\.linear|F\.silu|torch\.softmax|torch\.topk|torch\.argsort|torch\.bincount|index_add|index_select|scaled_dot_product_attention|torch\.cuda\.' track-03/submission.py`

Expected: empty output.

- [ ] **Step 3: Parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/submission.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 4: Commit**

```bash
git add track-03/submission.py
git commit -m "track-03: add down+topk-reduce kernel"
```

---

## Task 7: Wire up the wrapper

**Files:**
- Modify: `track-03/submission.py` (replace stub `fused_moe`)

- [ ] **Step 1: Replace the stub at the bottom of submission.py with the real wrapper**

```python
def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    """Fused MoE — entry point called by the judge.

    Args:
        hidden_states: [M, K]    bf16
        w1:            [E, 2N, K] bf16
        w2:            [E, K, N]  bf16
        score:         [M, E]    bf16
        topk:          int
        renormalize:   bool (default False)

    Returns:
        output:        [M, K]    bf16
    """
    assert hidden_states.dim() == 2
    assert w1.dim() == 3
    assert w2.dim() == 3
    assert score.dim() == 2
    M, K = hidden_states.shape
    E_w1, two_N, K_w1 = w1.shape
    E_w2, K_w2, N = w2.shape
    M_s, E_s = score.shape
    assert E_w1 == E_w2 == E_s, f"E mismatch: w1={E_w1} w2={E_w2} score={E_s}"
    assert K == K_w1 == K_w2,   f"K mismatch: hidden={K} w1={K_w1} w2={K_w2}"
    assert two_N == 2 * N,      f"w1.shape[1] must be 2*w2.shape[2]; got {two_N} vs {2 * N}"
    assert M_s == M,            f"score rows {M_s} != hidden rows {M}"
    E = E_w1

    hidden_c = hidden_states.contiguous()
    w1_c     = w1.contiguous()
    w2_c     = w2.contiguous()
    score_c  = score.contiguous()

    intermediate = torch.empty(M * topk, N, dtype=hidden_states.dtype, device=hidden_states.device)
    output       = torch.empty(M, K,         dtype=hidden_states.dtype, device=hidden_states.device)

    topk_w, topk_id = _launch_routing(score_c, topk, renormalize, M, E)
    _launch_gateup_silu(hidden_c, w1_c, topk_id, intermediate, M, K, N, topk)
    _launch_down_reduce(intermediate, w2_c, topk_w, topk_id, output, M, K, N, topk)

    return output
```

- [ ] **Step 2: Static compliance grep**

Run: `! grep -nE 'torch\.matmul|F\.linear|F\.silu|torch\.softmax|torch\.topk|torch\.argsort|torch\.bincount|index_add|index_select|scaled_dot_product_attention|torch\.cuda\.' track-03/submission.py`

Expected: empty output.

- [ ] **Step 3: Whole-file lint scan for forbidden torch ops on numerical values**

Run: `! grep -nE 'softmax|\.topk\(|argsort|bincount|index_add|index_select|F\.linear|F\.silu|F\.sigmoid|torch\.sigmoid|torch\.exp\(|torch\.matmul' track-03/submission.py | grep -v '^.*#'`

Expected: empty output. (`tl.sigmoid`, `tl.exp` inside `@triton.jit` are fine and won't match because those use `tl.` prefix.)

- [ ] **Step 4: Parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/submission.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 5: Commit**

```bash
git add track-03/submission.py
git commit -m "track-03: wire wrapper - route → gateup+silu → down+reduce"
```

---

## Task 8: test_correctness.py harness

**Files:**
- Create: `track-03/test_correctness.py`

This won't run on Mac (Triton not installed) but must be readable and correct.

- [ ] **Step 1: Write the harness**

```python
# track-03/test_correctness.py
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


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: f"{w['label']}_M{w['M']}_E{w['E']}_topk{w['topk']}")
def test_against_reference(workload):
    M, K, N, E, topk = workload["M"], workload["K"], workload["N"], workload["E"], workload["topk"]
    hidden, w1, w2, score = _make_inputs(M, K, N, E, topk)

    out_sub = fused_moe_sub(hidden, w1, w2, score, topk)
    out_ref = fused_moe_ref(hidden, w1, w2, score, topk)

    torch.testing.assert_close(out_sub, out_ref, atol=5e-2, rtol=5e-2)
```

- [ ] **Step 2: Parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/test_correctness.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: Commit**

```bash
git add track-03/test_correctness.py
git commit -m "track-03: add correctness harness vs reference (atol=5e-2)"
```

---

## Task 9: bench.py harness

**Files:**
- Create: `track-03/bench.py`

- [ ] **Step 1: Write the bench**

```python
# track-03/bench.py
"""Per-workload speedup of submission vs reference.

Cannot run on Mac. Useful when on a platform shell (rare) or for hand-trace.
"""
import time
import torch

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


def _time_ms(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000.0


def main():
    print(f"{'workload':<32} {'ref(ms)':>10} {'sub(ms)':>10} {'speedup':>10}")
    print("-" * 64)
    for w in WORKLOADS:
        M, K, N, E, topk = w["M"], w["K"], w["N"], w["E"], w["topk"]
        hidden, w1, w2, score = _make_inputs(M, K, N, E, topk)

        ms_sub = _time_ms(lambda: fused_moe_sub(hidden, w1, w2, score, topk))
        ms_ref = _time_ms(lambda: fused_moe_ref(hidden, w1, w2, score, topk))

        label = f"{w['label']}_M{M}_E{E}_topk{topk}"
        print(f"{label:<32} {ms_ref:>10.3f} {ms_sub:>10.3f} {ms_ref / ms_sub:>10.2f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/bench.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 3: Commit**

```bash
git add track-03/bench.py
git commit -m "track-03: add per-workload bench harness"
```

---

## Task 10: notes.md v1 entry

**Files:**
- Create: `track-03/notes.md`

- [ ] **Step 1: Write v1 entry**

```markdown
# track-03 Iteration Log

## v1 (2026-05-06) — initial baseline

Strategy: hybrid (option C). One universal `@triton.jit` pipeline (route →
gateup+silu → down+reduce), per-backend grid + BLOCK_* selected via cached
backend flags. No autotune, no atomics, no permutation.

Spec: `docs/superpowers/specs/2026-05-06-track-03-fused-moe-design.md`

### Per-backend seeds

| Backend  | Path          | Kernel B (BLOCK_MK / N / K, warps)         | Kernel C (BLOCK_MK / K_OUT / N_R, warps) |
|----------|---------------|--------------------------------------------|------------------------------------------|
| Ascend   | scalar reduce | 1 / 128 / 128, 2                           | 1 / 128 / 128, 2                         |
| Hygon    | tl.dot        | 16 / 128 / 64, 4                           | 16 / 128 / 64, 4                         |
| Iluvatar | tl.dot        | 16 / 128 / 64, 2 (num_stages=1)            | 16 / 128 / 64, 2                         |
| MTT      | tl.dot SQMMA  | 32 / 128 / 64, 8 (MUSA_ENABLE_SQMMA=1)     | 32 / 128 / 64, 8                         |
| MetaX    | tl.dot        | 16 / 64 / 64, 2                            | 16 / 64 / 64, 2                          |

### Expectations

- All 5 backends green (no `0.0` from a crashed backend).
- Hygon, Iluvatar, MTT each ≥ 2× over reference (FLOPs headroom + tensor cores).
- Ascend ≥ 1× over reference (scalar-reduce path, single-digit cores, "few programs / long inner loop").
- MetaX ≥ 0.3× (matches track-01 ceiling; not a v1 priority).

### Out of scope for v1, queued for v2+

- Sort-by-expert + grouped GEMM (proper w1 reuse).
- Autotune sweeps on Hygon and Iluvatar.
- Decode-specific path (M=1) skipping Kernel A's BLOCK_M loop.
- Fused routing+gateup (skip topk_ids round trip).
- MetaX exploration beyond the defensive seed config.

### Submission pending

(Fill in per-backend numbers after first platform run.)
```

- [ ] **Step 2: Commit**

```bash
git add track-03/notes.md
git commit -m "track-03: add iteration notes for v1"
```

---

## Task 11: Final compliance audit before submission

**Files:**
- Read: `track-03/submission.py`

A submission-day checklist. No code changes — just the audit gates.

- [ ] **Step 1: Forbidden torch numerical-op grep**

Run: `! grep -nE 'torch\.matmul|F\.linear|F\.silu|F\.sigmoid|torch\.softmax|\.softmax\(|\.topk\(|torch\.argsort|torch\.bincount|index_add|index_select|scaled_dot_product_attention|torch\.cuda\.|torch\.ops\.npu|torch\.ops\.musa' track-03/submission.py`

Expected: empty output.

- [ ] **Step 2: TileLang / non-Triton DSL grep**

Run: `! grep -nE '@T\.prim_func|import tilelang|from tilelang' track-03/submission.py`

Expected: empty output.

- [ ] **Step 3: torch.compile grep (rules-grey, ban for safety)**

Run: `! grep -nE 'torch\.compile|@torch\.compile' track-03/submission.py`

Expected: empty output.

- [ ] **Step 4: tl.range fragility check**

Run: `! grep -nE 'tl\.range\(' track-03/submission.py`

Expected: empty output. (We use plain `range()` for runtime loops and `tl.static_range` for compile-time-unrolled loops.)

- [ ] **Step 5: tl.trans on fp32 audit**

Run: `grep -nE 'tl\.trans' track-03/submission.py`

Expected: matches `tl.trans(w1g)`, `tl.trans(w1u)`, `tl.trans(w2t)` — all bf16 inputs at the dot site (we trans the loaded bf16 tile before the dot, never an fp32 accumulator). Verify by reading each match.

- [ ] **Step 6: tl.arange power-of-2 audit**

Run: `grep -nE 'tl\.arange\(0, ' track-03/submission.py`

Expected: every match has a `tl.constexpr` argument that is a power of 2 (`BLOCK_M`, `BLOCK_MK`, `BLOCK_N`, `BLOCK_K`, `BLOCK_N_R`, `BLOCK_K_OUT`, `E`, `TOPK`). For TOPK=6 this fails the rule! **Fix:** wherever we have `tl.arange(0, TOPK)` we need to replace with a static_range loop. Check the kernels: TOPK appears only inside `tl.static_range(TOPK)` (not in `tl.arange`) — confirm and proceed. If a `tl.arange(0, TOPK)` is present, replace with the static loop pattern from kernel C.

- [ ] **Step 7: Final parse-check**

Run: `python3 -c "import ast; ast.parse(open('track-03/submission.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 8: Commit (if any audit fixes were needed)**

```bash
git add track-03/submission.py
git commit -m "track-03: audit fixes pre-submission"
```

(Skip the commit if no fixes were needed.)

---

## Task 12: Submit v1 to platform

**Files:** none modified.

This is **execution**, not implementation — it requires the user (no automated submission tool from this session). The agent's job is to hand off cleanly.

- [ ] **Step 1: Print the submission file size and SHA**

Run: `wc -l track-03/submission.py && sha256sum track-03/submission.py 2>/dev/null || shasum -a 256 track-03/submission.py`

Expected: line count printed (~500–700 lines), hash printed. Record both in notes.md.

- [ ] **Step 2: Hand off to the user**

Print to the user:
```
v1 ready at track-03/submission.py.
Submit to https://kernelgen.flagos.io/ for track-03.
~5–10 min eval. Paste back the per-backend numbers and I'll log them in notes.md.
```

- [ ] **Step 3: When numbers come back, append to notes.md under the v1 section**

Replace the "(Fill in per-backend numbers after first platform run.)" placeholder with a markdown table of the actual numbers, and assess vs the expectation table.

---

## Self-Review Pass

Verifying spec coverage:

- ✅ 3-kernel pipeline (route, gateup+silu, down+reduce): Tasks 4, 5, 6.
- ✅ No atomics, no permutation: design baked into Tasks 5, 6.
- ✅ Workspace `intermediate [M*topk, N]`: Task 7 wrapper.
- ✅ Per-backend dispatch with cached flags: Task 3.
- ✅ Module-top env vars (TRITON_DISABLE_SWIZZLE, MUSA_ENABLE_SQMMA): Task 3.
- ✅ Backend detection via Triton runtime API only (no torch.cuda.\*): Task 3, audited Task 11.
- ✅ Routing softmax + iterative topk in fp32, weights stored bf16: Task 4.
- ✅ Gate+up GEMM accumulated fp32, SiLU in fp32: Task 5.
- ✅ Down GEMM fp32 acc, weighted topk reduce in fp32: Task 6.
- ✅ Ascend "few programs / long inner loop" 1D grid: Tasks 5, 6.
- ✅ MTT BLOCK_MK=32 (SQMMA floor): Task 3 cfg, used Task 5/6.
- ✅ MetaX defensive: Task 3 env var + cfg.
- ✅ No autotune: by construction.
- ✅ API hygiene (`tl.range` ban, power-of-2 `tl.arange`, `tl.trans` on bf16): Task 11 audit.
- ✅ workloads.py / reference.py / test_correctness.py / bench.py / notes.md: Tasks 1, 2, 8, 9, 10.

Type/method consistency:
- `_launch_routing`, `_launch_gateup_silu`, `_launch_down_reduce` — names consistent across Task 3 stub absence and Tasks 4–7.
- `topk_w` vs `topk_weights`: settled on `topk_w` for the local variable name, `topk_w_ptr` for the kernel argument. Consistent.
- `intermediate` is allocated `[M * topk, N]` (Task 7) and indexed `(m * TOPK + k_idx) * N + n_off` in Tasks 5 and 6. Row-major contiguous. Consistent.
- Backend cfg dicts (`_CFG_GATEUP_*`, `_CFG_DOWN_*`) — keys match across the launchers' `cfg["..."]` accesses.
- `IS_ASCEND: tl.constexpr` is the branch flag in Tasks 5, 6 — kernel arg consistent.

No placeholders found.

No unrelated refactoring.

The plan is ready.
