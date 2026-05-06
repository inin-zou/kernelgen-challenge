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

### v2 expectations

Best case: all 5 backends now compile and run, producing v1's intended speedups
(Hygon/Iluvatar/MTT ≥ 2x, Ascend ≥ 1x, MetaX ≥ 0.3x).

Worst case: still failing — meaning the issue is something else (most likely
candidates if v2 fails: `if IS_ASCEND:` constexpr branch, `tl.dot` `acc=`
keyword unsupported on some fork, or runtime loop bound issue on Ascend).

(Fill in per-backend numbers after v2 platform run.)
