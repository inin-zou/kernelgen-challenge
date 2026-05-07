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

## Deep research on MetaX 0.27 ceiling — 2026-05-06

Cloned MetaX-MACA/mcTriton (commit `1122c717`), MetaX-MACA/vLLM-metax,
MetaX-MACA/flashattn, MetaX-MACA/mcoplib, FlagGems _metax tree.

Key findings:

1. **Root cause of 0.27 ceiling**: mcTriton's chain-dot OPT MMA fast path
   (the path FlashAttention-fwd kernels rely on) requires `num_stages == 2`
   per `AccelerateMETAXMatmul.cpp:177-185` & `:371-381`. With num_stages=3
   (default) it falls through to a slower codegen branch. v16 byte-exact
   port did not set num_stages → got default 3 → silently slow.

2. **FlagGems flash_mla MetaX config (compute-cap 8 / C500)**:
   `_metax/fused/flash_mla.py:177-189` ships exactly
   `BLOCK_H=16, BLOCK_N=32, num_warps=8, num_stages=2`. Our kernel is
   structurally identical (chain dot in loop, bf16 IO, per-iter
   correction). FlagGems sparse_attention.py is a less-optimized cousin.

3. **MACAOptions hand-tuned LLVM preset for FlashAttention-fwd**
   (`mcTriton compiler.py:371-372`): passing `scenario="flashattn-fwd"`
   activates `metaxgpu-mma-sched=true`,
   `metaxgpu-sched-select=metaxgpu-minreg`, `map-use-pk-fma=1`. The minreg
   scheduler directly addresses our D=512 register pressure
   (acc[H=16, D=512] fp32 = 32 KB live registers per program).

4. **MetaX has officially abandoned Triton for sparse-attention**:
   `vLLM-metax PR #119` switches to C++/CUDA `flash_mla_sparse_fwd`. Our
   ceiling reflects a real Triton-vs-FlashMLA gap, not a config bug. But
   chain-dot OPT + LLVM preset are fully untapped — no contestant has
   tried this combo (verified: `inin-zou/kernelgen-challenge` also
   stuck at 0.27 with byte-port FlagGems config).

5. **Anti-leads (do not retry)**:
   - Bigger BLOCK alone without num_stages=2 (v15 confirmed dead)
   - +=  vs acc= dot form variations (v16 confirmed dead)
   - D-chunk extract_slice unrolling (Ascend-specific Cube trick, useless on MetaX)

6. **Wildcards for v18+ if v17 underdelivers**:
   - `correction = exp(scores_max_prev - scores_max)` may need TMP scratchpad
     round-trip per `MetaX-MACA/flashattn:flash_attn_triton.py:223-225`
   - `pipeline="cpasync"` activates async loads — addresses gather latency
   - `compute_cap.major == 9` (C550) wants completely different config
     (BLOCK_H=64, num_stages=3) — print arch once to verify C500 vs C550

---

## v17 — 2026-05-06

**Strategy: minimum diff to trigger MetaX chain-dot OPT MMA.**

Only the MetaX dispatch branch changes. Other 4 backends byte-identical to v16.

Tier sequence (try-except chain, fall through on any failure):
- Tier 1: `BLOCK=32, num_warps=8, num_stages=2, pipeline="basic", scenario="flashattn-fwd"`
- Tier 2: `BLOCK=32, num_warps=8, num_stages=2` (no MACAOptions kwargs, in case Triton's launcher rejects them)
- Tier 3: `BLOCK=16, num_warps=8` (v16 byte-exact — our 0.27 floor)
- Tier 4: universal kernel autotune (existing)
- Tier 5: v9 simple bf16 (existing)

**Risk model:**
- Tier 1 most aggressive — could be rejected by mcTriton's launcher if
  `pipeline=` / `scenario=` kwargs aren't directly accepted at launch site.
  In that case Triton raises → caught → Tier 2 runs.
- Tier 2 isolates the `num_stages=2 + BLOCK=32` change. If chain-dot OPT
  fires but LLVM preset doesn't (Tier 1 fails, Tier 2 runs), we still
  expect substantial uplift from chain-dot opt alone.
- Tier 3 is exactly v16 = 0.27 floor.
- All other backends (Ascend, Iluvatar, Hygon, MTT) untouched.

**Expected outcomes (per Tier that actually runs):**
- Tier 1 (best case): MetaX 0.27 → 0.5–0.8. Avg 1.96 → ~2.0–2.1.
- Tier 2 only: MetaX 0.27 → 0.4–0.6. Avg 1.96 → ~1.99–2.0.
- Tier 3 only: MetaX = 0.27. Avg = 1.96 (unchanged).

**Submit, observe.**

---

## v17 platform results — 2026-05-06

| Backend       | v10 (Best) | v17     | diff   |
|---------------|-----------|---------|--------|
| 华为昇腾       | 1.72      | 1.72    | 0      |
| 天数 Iluvatar  | 3.98      | 3.96    | -0.02 (noise) |
| 海光 Hygon     | 1.12      | 1.12    | 0      |
| 摩尔线程 MTT   | 2.71      | 2.57    | -0.14 (noise band) |
| 沐曦 MetaX     | **0.27**  | **0.30**| **+0.03** ✓ |
| **avg**       | 1.96      | 1.93    | -0.03 (MTT noise) |

**MetaX 16-iteration ceiling broken.** First time above 0.27 since v10. v10
still listed as Best on platform (1.96 > 1.93 due to MTT noise dragging avg).

---

## Follow-up research (2026-05-06): mcTriton scenario/pipeline env vars

Question: did Tier 1's `pipeline=` / `scenario=` kwargs actually fire, or
did they silently fall to Tier 2?

Source dive (mcTriton master @ commit 1122c717):

1. **No env var exists** for `scenario` / `pipeline`. They are dataclass
   attributes on `MACAOptions` (compiler.py:115-116) with hardcoded defaults
   (`pipeline="basic", scenario=""`). No `os.environ.get(...)` populates them.
2. **Launch-kwarg path verified working** by reading `jit.py:605-636`. The
   `parse_options(kwargs)` filters by `MACAOptions.__dataclass_fields__.keys()`
   so `scenario` and `pipeline` ARE accepted; they just get silently dropped
   if MISSPELLED. Our kwarg names matched, so Tier 1 fired.
3. **Conclusion**: 0.30 is the actual ceiling of "chain-dot OPT MMA +
   flashattn-fwd LLVM preset" on our specific kernel. The +11% (vs predicted
   2x+) is because `flashattn-fwd` LLVM preset is tuned for **dense**
   FlashAttention; our gather-then-attend pattern doesn't hit the same
   bottlenecks (`-map-use-pk-fma=1` and `metaxgpu-mma-sched=true` give us
   marginal lift; `metaxgpu-sched-select=metaxgpu-minreg` doesn't apply
   since our register pressure isn't the binding constraint).

4. **Untried env vars surfaced** (each maps to a separate compiler pass):
   - `TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP` (compiler.py:220)
   - `TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT`           (compiler.py:222)
   - `TRITON_ENABLE_SMEM_OFFSET_CACHE`                   (compiler.py:304)
   - `TRITON_ENABLE_BSM_INDEX_OPT`                       (compiler.py:305)

5. **Anti-leads (env-side)**:
   - `MACAOptions.__dataclass_fields__["scenario"].default = ...` monkey-patch
     would be equivalent to the kwarg path we already use — same effect, no
     incremental upside.
   - `TRITON_DISABLE_*` flags all DEFAULT-ENABLED paths; flipping them only
     hurts.

---

## v18 — 2026-05-06

**Strategy: probe the 4 ENABLE-style env vars in one shot.**

Single-line additions only:
```python
os.environ.setdefault("TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP", "1")
os.environ.setdefault("TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT", "1")
os.environ.setdefault("TRITON_ENABLE_SMEM_OFFSET_CACHE", "1")
os.environ.setdefault("TRITON_ENABLE_BSM_INDEX_OPT", "1")
```

No other code changes. v17's MetaX dispatch tiers preserved verbatim.

**Most plausible winners for our kernel:**
- `MERGE_CONVERT_LAYOUT`: our chain-dot kernel emits a `tl.bfloat16` cast on
  `p` between the two dots. If the compiler can fuse the convert_layout
  with adjacent layout transforms, the second dot's input gets there
  faster. Strongest signal among the four.
- `MOVE_DOT_OPERANDS_OUT_LOOP`: `q_block` is loaded outside the topk loop
  but the dot consumes it in every iteration. If this pass lifts the
  load-to-MMA-operand transform out, register pressure drops.

**Risk:**
- Each env var is gating a different compiler pass. Probing all 4 at once
  gives us less debugging signal — if MetaX regresses, we won't know which
  flag caused it. But probing one at a time is 4 submissions × 10 minutes
  vs 1 × 10 minutes. Given platform takes BEST per submission, the
  combined probe has bounded downside.
- Non-MetaX backends never read these names → other backends unchanged.
- If MetaX score regresses, v17 (or v10) remains Best on platform.

**Expected outcomes:**
- Best case: one of the 4 flags wins, MetaX 0.30 → 0.4-0.5.
- Realistic: marginal +5-10%, MetaX 0.30 → 0.32-0.34.
- Null: no movement, MetaX stays at 0.30 (or noise band 0.27-0.30).
- Negative: one flag regresses MetaX. v19 = identify offender via bisect.

**Side note (codex review finding)**: the v17 MetaX try/except chain
re-runs all failing tiers on every call. In practice Tier 1 succeeded so
no fall-through happens, but this is a real performance hazard for any
future MetaX tier change. Address in v19 with module-level tier-success
cache. v18 is env-only — no code path changes.

**Submit, observe.**

---

## v18 platform results — 2026-05-06

| Backend       | v17    | v18    | diff   |
|---------------|--------|--------|--------|
| 华为昇腾       | 1.72   | 1.73   | +0.01 (noise) |
| 天数 Iluvatar  | 3.96   | 3.97   | +0.01 (noise) |
| 海光 Hygon     | 1.12   | 1.11   | -0.01 (noise) |
| 摩尔线程 MTT   | 2.57   | 2.61   | +0.04 (noise) |
| 沐曦 MetaX     | 0.30   | 0.31   | +0.01 (noise) |
| **avg**       | 1.93   | 1.95   | +0.02  |

The 4 enable env vars added marginal lift (~3% on MetaX). v17 + v18 together
took us from 0.27 to 0.31 on MetaX. v10 still listed as Best (1.96 > 1.95
due to MTT noise dragging avg). **0.31 was framed as "ceiling" but that was
wrong** — see leaderboard analysis below.

---

## Leaderboard reality check — 2026-05-06

User pulled the actual leaderboard. Critical context I had been missing:

| Rank | Player | Ascend | Iluvatar | Hygon | MTT | MetaX | avg |
|------|--------|--------|----------|-------|-----|-------|-----|
| 1 | hankli | 35246 | 27685 | 1833 | 42332 | 1061 | 21632 (degenerate output exploit, ignore) |
| 2 | wqing  | 7.11 | 1.19 | 1.64 | 5.34 | **3.13** | 3.68 |
| 3 | cherufeta | 6.49 | 1.95 | 2.37 | 4.23 | **2.56** | 3.52 |
| 4 | the4pe18 | 1.27 | 2.64 | 2.88 | 3.53 | **2.85** | 2.63 |
| 5 | **us** | 1.68 | 3.97 | 1.11 | 2.77 | **0.27** | 1.96 |
| 6 | yijun.yu | 1.06 | 4.16 | 1.24 | 1.15 | **1.17** | 1.76 |
| 7 | 2861258657 | – | 1.11 | 2.57 | 1.48 | **1.69** | 1.37 |
| 8 | qlou005 | 1.04 | 1.17 | 1.23 | 1.15 | **1.33** | 1.18 |
| 11 | learning | 1.00 | 1.00 | 1.00 | 1.01 | **1.00** | 1.00 (baseline) |

**Six contestants land MetaX 1.17–3.13x with rules-compliant Triton.** We are
the only legitimate submission below 1.0 — meaning our kernel is ~5–10x
slower than what's achievable. The "0.31 ceiling = hardware mismatch"
hypothesis was wrong. Our kernel has structural flaws.

---

## Deep research: NSA forward kernel (fla-org) — 2026-05-06

Studied `fla-org/native-sparse-attention/native_sparse_attention/ops/parallel.py`
parallel_nsa_fwd_kernel (L716-797). Key contrasts with our kernel:

1. **Grid is 3D**: `(T, NV, B*H)` (L892). pid0=token, **pid1=V-axis chunk**,
   pid2=batch×kv_head. We use `(m, b)` — no V-axis parallelism.
2. **acc_o is V-chunked**: `b_o = tl.zeros([G, BV], fp32)` (L758) where
   `BV=min(256, npow2(V))`. We use `acc_o[H, D=512]` full-width — 32 KB live
   register tile per program, 4× larger than NSA's.
3. **Plain `for i in range(NS)`** (L729) — no `tl.range`, no `static_range`,
   no `num_stages` autotune. Autotune only sweeps `num_warps ∈ {1, 2, 4}`
   (L661-667).
4. **Q baked with scale before loop**: `b_q = (b_q * scale).to(b_q.dtype)`
   (L753). Saves a multiply per iteration.
5. **`b_p.to(b_q.dtype)` cast before second dot** (L742). We do the same
   already.
6. **Block-coarse KV indexing**: NSA's `block_indices[i] * BS` enables
   contiguous `[BK, BS]` block-ptr loads. Our spec gives per-token random
   indices — fundamental constraint, not fixable.

The single highest-payoff change is the **3D grid + V-chunked acc_o**. NSA
explicitly cites this pattern as the canonical Triton structure for
`per-query attend over selected KV` — same problem class as ours.

---

## v19 — 2026-05-06

**Strategy: rewrite MetaX kernel per NSA pattern.**

New kernel `_sparse_attn_kernel_v19_v_chunked`:
- Grid: `(m, b, NV)` with `NV = cdiv(d, BV)`, `BV=128` → NV=4 for d=512.
- Per program: Q full-D loaded once, K full-D loaded per topk chunk, V
  loaded as `[BLOCK, BV]` slice only (8 KB instead of 32 KB).
- `acc_o = [H, BV]` fp32 = 8 KB per program — quartered from 32 KB.
- All V-chunk programs converge to identical softmax stats (deterministic
  QK pass, replicated). Output correctness preserved.

Cost: redundant QK dot per V-chunk. ~2.5× total compute, but parallelism
4× and register pressure 4× lower. NSA accepts this trade.

Added as Tier 0 in MetaX dispatch chain. Tier 1–5 (v17/v18 chain-dot OPT
path, v16 byte-exact, autotune, bf16 fallback) preserved verbatim. If v19
fails for any reason (e.g. compile error on an edge config), graceful
fall-through.

**Risk model:**
- Worst case (v19 raises every call): MetaX falls to v17/v18 ≈ 0.30 — no
  regression vs current state.
- Other backends never enter v19 path → unchanged.
- Platform takes BEST → v10 (1.96) still locked in if v19 underdelivers.

**Expected outcomes:**
- Best case: MetaX 0.31 → 1.5–2.5x (matches median competitor). Avg
  1.96 → ~2.2–2.4.
- Realistic: MetaX 0.31 → 0.8–1.5x (meaningful improvement but not
  full competitor parity).
- Null: v19 also stuck at 0.3. Tier fallback preserves 0.30 floor.

**Submit, observe — watch for correctness failures**: this is the first
time we're shipping a kernel that's NOT a byte-port. Bugs are possible.
Tier 1–5 fallback is the safety net.
