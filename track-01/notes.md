# Track-01 Sparse Attention — Tuning Log

## Reference algorithm

```
For each (b, m, h):
    1. Gather KV[b, topk_idxs[b,m,:], :]  → [topk, d]
    2. scores = q[b,m,h,:] @ KV_gathered.T * scale  → [topk]
    3. Append attn_sink[h]  → [topk+1]
    4. softmax → [topk+1]
    5. o[b,m,h,:] = softmax[:-1] @ KV_gathered (sink excluded from V mixing)
```

Key facts:
- d=512, h=16, bf16 in/out, fp32 compute (per reference's `.float()`)
- KV is *shared* (same tensor for K and V — single load)
- 10 workloads: decode (m=1, batch ≤ 64) + prefill (m up to 16384)
- topk varies 128 → 640

## v1 design (2026-05-06)

**One program per (b, m).** All H=16 heads share gathered KV per topk chunk.

FlashAttention-style online softmax over topk dimension:
```
for each BLOCK_TOPK chunk of topk:
    gather KV_chunk [BLOCK_TOPK, D]   # random-access along k, contiguous along d
    scores = Q [H, D] @ KV_chunk^T [D, BLOCK_TOPK] * scale
    update m_i, l_i, o_i (online softmax)
```

Sink absorbed into denominator AFTER the topk loop (correct math: sink is
the (topk+1)-th score in a single softmax; output uses only the first topk
attn weights). Implementation:
```
m_total   = max(m_i, sink)
alpha     = exp(m_i - m_total)
sink_term = exp(sink - m_total)
l_total   = alpha * l_i + sink_term
o_final   = (alpha * o_i) / l_total
```

**Reused infrastructure from track-02 v6:**
- Backend detection (`_is_metax_backend`, `_is_ascend_backend`) — Triton
  runtime API only, no torch.cuda.* (strict rule compliance).
- MetaX/Ascend bypass autotune entirely.
- `@triton.autotune` on the runtime-loop variant for MTT/Hygon/Iluvatar.
- Try/except backstop for any remaining backend that surprises us.
- TRITON_DISABLE_SWIZZLE=1 defensive env var.

**Single kernel.** Track-02 needed two kernels (runtime/static) because
Sinkhorn's iter count was constexpr-friendly. Here the topk loop is
inherently runtime (TOPK varies 128–640), so one kernel serves both
autotune and static paths — call with explicit BLOCK_TOPK in fallback.

**Autotune sweep:** `BLOCK_TOPK ∈ {16,32,64,128}`, `num_warps ∈ {2,4,8}`.
The (128, 8) config was the surprise winner on MTT in track-02; including
similar large-tile + many-warps configs here.

**Static fallback:** BLOCK_TOPK=64, num_warps=4. Mid-range for safety.

## Possible v2+ levers

| Lever | Why |
|-------|-----|
| Split D dimension (BLOCK_D) into chunks | If [H, D=512] tile too big for some backends (Ascend UB?), split |
| Persistent kernel (b*m programs → fewer waves) | Ascend has only 25 cores; 65k programs is 2600 waves of overhead |
| KV cast bf16→fp32 lazy (keep bf16 until tl.dot) | Use tensor cores for matmul; faster, possibly less precise |
| BLOCK_M (process multiple queries per program) | More register reuse if adjacent queries share KV indices (unlikely) |
| 2D grid (b, m_block) | If m is huge (16384), a 1D m grid means 16k programs — fine usually |
| Scale as constexpr | Triton constant-folds the multiply |

## Submission tracking

| Version | Avg | Ascend | Iluvatar | Hygon | MTT | MetaX | Notes |
|---------|-----|--------|----------|-------|-----|-------|-------|
| v1      | 0.22 | failed | failed | 0.37 | 0.74 | failed | fp32 tl.dot triggered TF32 fallback; tl.trans on fp32 broke 3 backends |
| v2      | 1.00 | 1.69   | 1.99   | 1.11 | failed | 0.21 | bf16 path recovered 3 backends; MTT crashed during autotune; MetaX too small static |
| v3      | 0.62 | failed | 2.00 | 1.11 | failed | failed | bumping Ascend/MetaX static to 32 crashed both (UB OVERFLOW); MTT autotune still uncatchable, never reached fallback |
| v4      | 1.02 | 1.70   | 2.00 | 1.12 | 0.06   | 0.21 | MTT detect via "musa" worked; but MTT bf16 path 12x slower than v1's fp32. Avg new best 1.02. |
| v5      | 1.17 | 1.71   | 2.01 | 1.11 | 0.79   | 0.21 | MTT fp32 hypothesis confirmed (0.06 → 0.79 = 13x). Avg new best 1.17 (rank #6) |
| v6      | 0.95 | 1.67   | 1.98 | 1.11 | failed | failed | Both hypotheses crashed: MetaX fp32 (8,2) and MTT (128,4). v5 still Best (1.17). |
| v7      | 1.17 | 1.70   | 2.10 | 1.09 | 0.74   | 0.21 | Flash-Decoding had ~zero impact (Iluvatar +0.09 only). Wrong assumption — workload not SM-bound. |
| v8      | 0.88 | failed | **3.98** | 0.22 | failed | 0.18 | FlagGems port: Iluvatar exploded to 3.98 BUT Ascend/MTT failed, Hygon regressed 5x. Root cause: FlagGems sparse_attn is broken upstream (CI test disabled). |
| v9      | 1.55 | 1.71   | 3.97 | 1.12 | 0.74   | 0.21 | Predicted exactly. Iluvatar 3.97 from v8 universal; others reverted to v5/v7. Rank #5. |
| v10     | 1.96 | 1.72   | 3.98 | 1.12 | 2.71   | 0.27 | MTT exploded 0.74→2.71 (H_padded=32+SQMMA verified). MetaX small bump 0.21→0.27. Rank #4! |
| v11     | 1.61 | 0.04   | 4.00 | 1.11 | 2.65   | 0.27 | Ascend 2-level kernel ran but 25x slower than ref (0.04). FlagGems _ascend pattern not optimal. v10 still Best. |
| v12     | 1.96 | 1.68   | 3.97 | 1.11 | 2.77   | 0.27 | Persistent kernel didn't help Ascend (no real reuse across tasks). avg same as v10. MTT slight bump 2.71→2.77 from acc form. |
| v13     | 1.62 | 0.03   | 4.00 | 1.11 | 2.67   | 0.27 | Ascend BLOCK=32+stages=2+acc-form crashed to 0.03. Pattern: ANY change beyond v9 baseline hurts Ascend. v10 still Best 1.96. |
| v14     | 1.78 | 0.77   | 4.00 | 1.11 | 2.75   | 0.27 | extract_slice compiled but Ascend per-chunk overhead net negative. v10 still Best 1.96. |
| v15     | 1.92 | 1.67   | 3.96 | 1.11 | 2.60   | 0.27 | Bigger BLOCK didn't move MetaX (still 0.27). Other backends within noise. v10 still Best. |
| v16     | TBD  |        |      |      |        |      | MetaX byte-exact FlagGems kernel (+= form, num_warps=8, BLOCK=16). Hypothesis: acc= form silently slow on mcTriton. |
