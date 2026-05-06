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

### Submission pending

(Fill in per-backend numbers after first platform run.)
