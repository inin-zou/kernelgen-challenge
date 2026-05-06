# track-03 Iteration Log

## v1 (2026-05-06) — initial baseline

Strategy: hybrid (option C). One universal `@triton.jit` pipeline (route →
gateup+silu → down+reduce), per-backend grid + `BLOCK_*` selected via cached
backend flags. No autotune, no atomics, no permutation.

Spec: `docs/superpowers/specs/2026-05-06-track-03-fused-moe-design.md`
Plan: `docs/superpowers/plans/2026-05-06-track-03-fused-moe.md`

### Why we expect speedup at all

Reference loops over all `E` experts and runs full `[M,K] @ [2N,K].T` GEMMs,
masking afterwards. We compute only the `M·topk` token-expert pairs that are
actually selected:

- E=64, topk=6 → **10.7× fewer FLOPs** than reference.
- E=8,  topk=2 → **4× fewer FLOPs** than reference.

So even a naive Triton pipeline with imperfect memory reuse should beat the
reference by a comfortable margin on these shapes.

### Per-backend seeds

| Backend  | Path          | Kernel B (BLOCK_MK / N / K, warps)         | Kernel C (BLOCK_MK / K_OUT / N_R, warps) |
|----------|---------------|--------------------------------------------|------------------------------------------|
| Ascend   | scalar reduce | 1 / 128 / 128, 2                           | 1 / 128 / 128, 2                         |
| Hygon    | tl.dot        | 16 / 128 / 64, 4                           | 16 / 128 / 64, 4                         |
| Iluvatar | tl.dot        | 16 / 128 / 64, 2 (num_stages=1)            | 16 / 128 / 64, 2                         |
| MTT      | tl.dot SQMMA  | 32 / 128 / 64, 8 (MUSA_ENABLE_SQMMA=1)     | 32 / 128 / 64, 8                         |
| MetaX    | tl.dot        | 16 /  64 / 64, 2                           | 16 /  64 / 64, 2                         |

### Expectations

- All 5 backends green (no `0.0` from a crashed backend).
- Hygon, Iluvatar, MTT each ≥ 2× over reference (FLOPs headroom + tensor cores).
- Ascend ≥ 1× over reference (scalar-reduce path, single-digit cores, "few programs / long inner loop").
- MetaX ≥ 0.3× (matches track-01 ceiling; not a v1 priority).

### Out of scope for v1, queued for v2+

- Sort-by-expert + grouped GEMM (proper w1 reuse).
- Autotune sweeps on Hygon and Iluvatar.
- Decode-specific path (M=1) skipping Kernel A's `BLOCK_M` loop.
- Fused routing+gateup (skip `topk_ids` round trip).
- MetaX exploration beyond the defensive seed config.

### v1 result

| Backend  | Speedup |
|----------|---------|
| Ascend   | Failed  |
| Iluvatar | Failed  |
| Hygon    | Failed  |
| MTT      | Failed  |
| MetaX    | Failed  |

Universal failure across all 5 backends → almost certainly a Triton API used
by code that runs on every backend. No per-backend log fetched, so v2 is a
blind defensive pass at the most likely culprits.

## v2 (2026-05-06) — defensive API portability fixes

Three changes:

1. **Routing kernel: drop `tl.argmax`.** Replaced with `tl.max` + sentinel-min
   pattern (encode each position by its index, set non-max positions to a
   sentinel = E, take row min). `tl.argmax` is the most-recent API in v1 and
   the most likely to be missing or buggy on Triton-Ascend / mcTriton / muTriton.

2. **Kernels B and C: drop `tl.broadcast_to`.** The hidden/intermediate row
   was being broadcast to `[BLOCK_MK, BLOCK_K]` via `tl.broadcast_to`.
   Replaced with pointer-arithmetic 2D loads:

       mk_lane = tl.arange(0, BLOCK_MK)
       h_off_2d = m * K + mk_lane[:, None] * 0 + k_off[None, :]
       h_tile   = tl.load(hidden_ptr + h_off_2d, mask=..., other=0.0)

   The `mk_lane * 0` introduces the `BLOCK_MK` row dim into pointer arithmetic
   without changing values. Wider compiler support across Triton forks.

3. **`tl.dot(..., acc=acc)` form** instead of `acc += tl.dot(...)`. Track-01
   lesson: this fuses the accumulator on MetaX for ~30% perf on the dot path.
   Functionally equivalent on other backends.

### v2 result

| Backend  | Speedup |
|----------|---------|
| Ascend   | Failed  |
| Iluvatar | Failed  |
| Hygon    | Failed  |
| MTT      | Failed  |
| MetaX    | Failed  |

Identical pattern to v1 — universal failure on all 5. That rules out the v2
hypotheses (`tl.argmax`, `tl.broadcast_to`) since v2 specifically removed both
and the failure didn't change. The bug is shared by v1 and v2.

## v3 (2026-05-06) — minimum-viable diagnostic

Strategy: strip back to the most basic Triton API surface. If v3 still fails
universally, the bug is in routing or the wrapper, not in the compute kernels.
If v3 passes anywhere, we know v1/v2's tensor-core paths were the issue.

Changes vs v2:

1. **Drop `tl.dot` entirely.** All "matmul" is now `tl.sum(a[None, :] * b, axis=1)`
   scalar reduction. ~3–5× slower than tl.dot on tensor-core hardware, but
   uses only `tl.load`/`tl.store`/`tl.sum`/`tl.max`/`tl.min`/`tl.where`/`tl.exp`/
   `tl.sigmoid` — every fork's bedrock.
2. **Drop `IS_ASCEND: tl.constexpr` branching.** One code path everywhere.
3. **Drop `acc=` and `out_dtype=` keyword args** (gone with tl.dot).
4. **Drop pointer-arithmetic broadcast tricks** (no 2D tl.dot operand needed).
5. **Add `TRITON_ALL_BLOCKS_PARALLEL=1`** env var — Ascend's grid > 65535 escape hatch.
6. **2D grid for both compute kernels on every backend.**
7. `num_warps=1` on Ascend, 2 elsewhere. Conservative.
8. `renormalize` arg accepted but ignored in v3 (not on hot path; default False).

### v3 expectations

If v3 passes on at least Hygon/Iluvatar (the most standard Triton): the
v1/v2 bug was in tensor-core code paths and we re-introduce `tl.dot` carefully
in v4 with one targeted change at a time.

If v3 still fails everywhere: the bug is in the routing kernel, the wrapper,
or our backend-detection. Next move would be a routing-only smoke submission
to isolate further.

(Fill in per-backend numbers after v3 platform run.)
