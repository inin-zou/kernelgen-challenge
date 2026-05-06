# Track-03 Fused MoE — v1 Design

**Date:** 2026-05-06
**Status:** Draft (pre-implementation)
**Strategy:** Hybrid (option C) — one universal `@triton.jit` pipeline with per-backend grid shape and block constants chosen at dispatch time.

## Context

GOSIM KernelGen 2026, track-03. Implement `fused_moe(hidden_states, w1, w2, score, topk, renormalize=False)` entirely in Triton (no torch numerical ops, no vendor BLAS). Target the same 5 backends as tracks 01/02: Ascend (`npu`), Hygon (`hip`), MetaX (`maca`), Moore Threads (`musa`), Iluvatar (`cuda` ∧ ¬musa).

**Operating constraints:**
- No local benchmarking. Every iteration is a platform submission (~5–10 min, 2-min cooldown). Robustness > cleverness.
- No autotune in v1 — uncatchable SIGSEGV on MetaX/MTT, wasted wall time on Ascend.
- Submitter takes the best of all submissions, so v1 just needs to be a correct, all-5-backends-green floor.

## Workloads (from `requirements.md`)

`hidden_size K = 2048`, `intermediate_size N = 1024`, dtype = bf16.

| Class   | M    | E  | topk |
|---------|------|----|------|
| Decode  | 1    | 8  | 2    |
| Decode  | 64   | 8  | 2    |
| Decode  | 1    | 64 | 6    |
| Decode  | 64   | 64 | 6    |
| Prefill | 512  | 8  | 2    |
| Prefill | 4096 | 8  | 2    |
| Prefill | 512  | 64 | 6    |
| Prefill | 4096 | 64 | 6    |

## FLOPs headroom

Reference loops over all `E` experts and runs full `[M,K] @ [2N,K].T` GEMMs, masking afterwards. We compute only the `M·topk` token-expert pairs that are actually selected:

- E=64, topk=6 → **10.7× fewer FLOPs** than reference.
- E=8, topk=2 → **4× fewer FLOPs** than reference.

Even a "naive" Triton implementation with imperfect memory reuse should beat the reference by a comfortable margin on these shapes. Aggressive optimizations (sort-by-expert, grouped GEMM with shared w1 tiles) are deferred to v2+.

## Architecture — three kernels, no atomics, no permutation

```
score [M,E] ───────────► [Kernel A: route]
                              ├─► topk_weights [M, topk]   bf16
                              └─► topk_ids     [M, topk]   int32

hidden [M,K], w1 [E,2N,K] ──► [Kernel B: gateup + silu]
                              └─► intermediate [M·topk, N]  bf16

intermediate, w2 [E,K,N], topk_weights, topk_ids
                          ──► [Kernel C: down + topk-reduce]
                              └─► output [M, K]            bf16
```

**Why this split:**

- No `tl.atomic_add` anywhere. Atomics on bf16/fp32 are flaky on Ascend and slow on MTT. Topk results combine inside Kernel C: each program owns one output tile `[m, k_chunk]` and runs an inner loop over the `topk` experts, accumulating in fp32 registers.
- No sort/permute kernel. Token-major layout — each token-expert pair is independent. Trades w1 reuse for simplicity. Acceptable v1 because the FLOPs headroom is large.
- `intermediate` workspace is `[M·topk, N]` bf16. Worst case (M=4096, topk=6, N=1024) = 48 MB. Allocated by wrapper via `torch.empty`.

### Kernel A — routing

- **Inputs:** `score [M, E]` bf16
- **Outputs:** `topk_weights [M, topk]` bf16, `topk_ids [M, topk]` int32
- **Grid:** `(ceil(M / BLOCK_M_R),)` 1D
- **Per program:** load `BLOCK_M_R` token rows, cast to fp32, softmax over `E` dim, then `topk` rounds of (argmax → store id+weight → mask the chosen index with -inf). E ≤ 64 so the row tile fits in registers easily.
- **Note:** spec defaults `renormalize=False`. We accept the flag and divide by `sum` at the end if true; not optimized for that case.

### Kernel B — gate+up + SiLU

- **Inputs:** `hidden [M, K]` bf16, `w1 [E, 2N, K]` bf16, `topk_ids [M, topk]` int32
- **Outputs:** `intermediate [M·topk, N]` bf16
- **Grid (tensor-core backends Hygon / Iluvatar / MetaX / MTT):** `(M·topk, ceil(N / BLOCK_N))` 2D
- **Grid (Ascend):** `(ceil(N / BLOCK_N),)` 1D — single program, inner loop over all `M·topk` pairs
- **Per program (tensor-core path):** decode `(m, k_idx) = divmod(pid_0, topk)`. Load `e = topk_ids[m, k_idx]`. Load `hidden[m, :]` into a `[BLOCK_MK, K]` tile padded by row replication (rows 1..BLOCK_MK-1 are duplicates of row 0; we only use row 0 of the output). For each K-chunk, load `w1[e, n_block_gate, :]` and `w1[e, n_block_up, :]` of shape `[BLOCK_N, K_CHUNK]`, do two `tl.dot`s to produce gate and up vectors, then store `silu(gate) * up` into `intermediate[m·topk + k_idx, n_block]`.
- **Per program (Ascend path, scalar reduce):** same logic but use `acc = tl.sum(hidden_tile[None,:] * w1_tile, axis=1)` instead of `tl.dot`. No padding waste because there are no tensor cores to feed.
- **Padding waste:** `BLOCK_MK = 16` (Hygon/Iluvatar/MetaX) → 15/16 wasted tl.dot lanes; `BLOCK_MK = 32` (MTT SQMMA floor) → 31/32 wasted. Tensor cores are 5–10× faster than scalar, so net is still a win on those backends.

### Kernel C — down + topk-reduce

- **Inputs:** `intermediate [M·topk, N]` bf16, `w2 [E, K, N]` bf16, `topk_weights [M, topk]` bf16, `topk_ids [M, topk]` int32
- **Outputs:** `output [M, K]` bf16
- **Grid (tensor-core backends):** `(M, ceil(K / BLOCK_K_OUT))` 2D
- **Grid (Ascend):** `(ceil(K / BLOCK_K_OUT),)` 1D — single program, inner loop over all M
- **Per program:** for each `(m, k_block)`:
  ```
  acc = zeros([BLOCK_K_OUT], fp32)
  for k_idx in range(topk):
      e   = topk_ids[m, k_idx]
      w_t = topk_weights[m, k_idx].to(fp32)
      mid = intermediate[m·topk + k_idx, :N]                # bf16, broadcast row
      w2_tile = w2[e, k_block, :N]                          # [BLOCK_K_OUT, N]
      acc += w_t * tl.dot(mid_padded, w2_tile.T, fp32)      # tensor-core path
      # or acc += w_t * tl.sum(mid[None,:] * w2_tile, axis=1)  # Ascend path
  output[m, k_block] = acc.to(bf16)
  ```
- This is the only place topk experts combine. No atomics needed because each `(m, k_block)` pair is owned by exactly one program.

## Per-backend dispatch table

| Backend     | Detect              | Kernel B path | Kernel B blocks                          | Kernel C path | Notes |
|-------------|---------------------|---------------|------------------------------------------|---------------|-------|
| Ascend      | `npu`               | scalar reduce | 1D grid over N tiles, long inner loop    | scalar reduce | 32 vector cores → few programs. No autotune. Static fallback always. |
| Hygon       | `hip`               | tl.dot        | `BLOCK_MK=16, BLOCK_N=128, num_warps=4`  | tl.dot        | Standard NV-style, hundreds of CUs. |
| Iluvatar    | `cuda` ∧ ¬musa      | tl.dot        | `BLOCK_MK=16, BLOCK_N=128, num_warps=2, num_stages=1` | tl.dot | FlagGems _iluvatar pattern. |
| MTT         | `musa`              | tl.dot        | `BLOCK_MK=32, BLOCK_N=128, num_warps=8`  | tl.dot        | SQMMA floor 32. `MUSA_ENABLE_SQMMA=1` env. Bypass autotune. |
| MetaX       | `maca`              | tl.dot        | `BLOCK_MK=16, BLOCK_N=64, num_warps=2, num_stages=1` | tl.dot | `tl.dot(..., acc=acc)` form. `TRITON_DISABLE_SWIZZLE=1`. Bypass autotune. Expected 0.2–0.5× — same fundamental mismatch as track-01. |

Block sizes are seed values from track-01/02 lessons + FlagGems patterns. Fine-tuning happens in v2 once we have v1 platform numbers.

## Wrapper contract

```python
def fused_moe(
    hidden_states: torch.Tensor,   # [M, K]    bf16
    w1:            torch.Tensor,   # [E, 2N, K] bf16
    w2:            torch.Tensor,   # [E, K, N]  bf16
    score:         torch.Tensor,   # [M, E]    bf16
    topk:          int,
    renormalize:   bool = False,
) -> torch.Tensor:                 # [M, K]    bf16
```

Wrapper responsibilities (memory-only ops permitted):
1. `assert` shapes, derive `M, K, E, N`.
2. `topk_weights = torch.empty(M, topk, dtype=bf16, device=...)`
3. `topk_ids = torch.empty(M, topk, dtype=int32, device=...)`
4. `intermediate = torch.empty(M * topk, N, dtype=bf16, device=...)`
5. `output = torch.empty(M, K, dtype=bf16, device=...)`
6. Detect backend (cached) → pick block constants and grid functions.
7. Launch Kernel A → Kernel B → Kernel C.
8. Return `output`.

No `softmax`, `topk`, `argsort`, `index_select`, `bincount`, `linear`, `silu`, etc. on the torch side.

## Robustness mechanisms (track-02 pattern)

1. **Module-top env vars:**
   ```python
   os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")  # MetaX defensive
   os.environ.setdefault("MUSA_ENABLE_SQMMA", "1")       # MTT bf16 tensor core
   ```
2. **Cached backend flags** (`_IS_ASCEND`, `_IS_METAX`, `_IS_MTT`, `_IS_HYGON`, `_IS_ILUVATAR`) computed once via `triton.runtime.driver.active.get_current_target().backend`.
3. **No autotune.** All configs hardcoded per backend.
4. **Static fallback path per kernel.** Selected when any of `_IS_ASCEND or _IS_METAX or _IS_MTT` is set, or when the fast path raises and a process-wide `_FAST_OK` flag flips to False.
5. **Try/except around the fast path** with `_FAST_OK` flip — catches the catchable subset; uncatchable backends never enter the fast path.
6. **API hygiene:**
   - Plain `range(non_constexpr)` for runtime loops (never `tl.range`).
   - `tl.trans` only on bf16 tiles, never fp32.
   - All `tl.arange(0, X)` use power-of-2 X.
   - `BLOCK_MK` is `tl.constexpr` and a power of 2.

## Numerical strategy

- Routing softmax + topk computed in fp32, weights stored as bf16 (matches reference's `score.softmax(dtype=torch.float32).to(dtype)`).
- Gate+up GEMMs accumulated in fp32, SiLU computed in fp32, result stored bf16.
- Down GEMM accumulated in fp32, weighted-sum across topk in fp32, result stored bf16.
- Tolerance budget: bf16 atol≈5e-2 (track-01/02 contest convention).

## File layout

```
track-03/
├── submission.py            # Uploaded to judge — sole runtime artifact
├── reference/
│   └── reference.py         # Frozen torch oracle, lifted from requirements.md
├── workloads.py             # 8 workloads from spec table as a list of dicts
├── test_correctness.py      # vs reference, atol=5e-2, bf16 — won't run on Mac, ships to platform
├── bench.py                 # 8-workload bench harness — same caveat
├── notes.md                 # v1, v2, ... iteration log (track-01/02 style)
└── requirements.md          # Already exists, unchanged
```

Even though `test_correctness.py` and `bench.py` cannot execute on Mac, they pin down the contract precisely, become useful on-platform debug artifacts, and `notes.md` is the iteration log we'll lean on heavily without local benchmarks.

## Out of scope for v1

Logged here so v2+ has a backlog:

- Sort-by-expert + grouped GEMM (proper w1 reuse).
- Autotune sweeps on Hygon and Iluvatar.
- Decode-specific specialization (M=1 path that skips Kernel A's BLOCK_M_R loop).
- Fused routing+gateup (skip `topk_ids` write/read round trip).
- MetaX exploration beyond the defensive seed config.

## Success criteria for v1

- All 5 backends report a finite speedup (no `0.0` from a crashed backend).
- Hygon, Iluvatar, MTT each ≥ 2× over reference (FLOPs headroom + tensor cores).
- Ascend ≥ 1× over reference (scalar-reduce path, single-digit cores).
- MetaX ≥ 0.3× (matches track-01 ceiling; not a v1 priority).
