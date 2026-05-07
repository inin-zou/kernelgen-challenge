# Track-02 Tuning Log

Each entry: date, change, decode/prefill latency on 4 workloads, notes.

## Reference: TileLang upstream design (deepseek-ai/TileKernels)

Two-kernel pipeline: `pre_split_mixes_kernel` then `sinkhorn_kernel`.
Our submission must fuse them into one Triton kernel.

Key TileLang choices to consider porting:
- `TL_DISABLE_WARP_SPECIALIZED=True` → Triton `num_warps=1` or `num_warps=2`
- 4×4 `comb` fragment kept in registers across all 39 normalize passes
- Sinkhorn = 1 softmax + 1 col-norm + 19 × (row-norm, col-norm) = 39 passes total
- eps placement: softmax → `+ eps`, divide → `/ (sum + eps)`
- fp32 throughout (no bf16 path needed for hc_split_sinkhorn)
- post = `2 * sigmoid(...)` → bake `2 *` into kernel (don't multiply outside)

## v0 (placeholder) — superseded

PyTorch passthrough. Used to validate the harness only.

---

## v1 (first Triton kernel) — 2026-05-05

**Design:**
- One program processes `BLOCK_T = 16` tokens.
- Single 24-element load per token, split into pre / post / comb sections.
- `comb` lives as a `[BLOCK_T, 4, 4]` register tile across all 39 normalize passes.
- `tl.static_range(SINKHORN_ITERS - 1)` unrolls the 19 (row, col) iteration pairs at compile time.
- `num_warps=1` (small tiles, no benefit from extra warps for 4-element reductions).
- All scales / base values loaded once per program, reused across BLOCK_T tokens.

**Order-of-ops matches the reference exactly:**
- `pre  = sigmoid(...) + eps`            (eps OUTSIDE sigmoid)
- `post = 2 * sigmoid(...)`              (factor 2 fused into kernel)
- `comb`: softmax(-1) → `+ eps` → 1 col-norm → 19 × (row-norm, col-norm)
- Every divide is `/ (sum + eps)`

**Correctness:** Run `python test_correctness.py`.
**Benchmarks:** Run `python bench.py`.

| workload | stage    | shape              | ref | ours | speedup |
|----------|----------|--------------------|-----|------|---------|
| 16k-1k-1 | decode   | (1, 1, 24)         | TBD | TBD  | TBD     |
| 16k-1k-1 | prefill  | (1, 16384, 24)     | TBD | TBD  | TBD     |
| 4k-1k-16 | decode   | (16, 1, 24)        | TBD | TBD  | TBD     |
| 4k-1k-16 | prefill  | (16, 4096, 24)     | TBD | TBD  | TBD     |
| 1k-1k-64 | decode   | (64, 1, 24)        | TBD | TBD  | TBD     |
| 1k-1k-64 | prefill  | (64, 1024, 24)     | TBD | TBD  | TBD     |
| 1k-1-64  | decode   | (64, 1, 24)        | TBD | TBD  | TBD     |
| 1k-1-64  | prefill  | (64, 1024, 24)     | TBD | TBD  | TBD     |

**Post-review fixes applied (still v1):**
- `EPS` moved out of `tl.constexpr` to a regular float kernel arg.
  Reason: float constexpr is not a documented stable contract on non-NV
  Triton backends (FlagTree's Ascend / Hygon / MUSA forks). Keeping
  HC/HC2/HC3/SINKHORN_ITERS/BLOCK_T as constexpr.
- Added `assert hc_mult == 4` in the wrapper. Matches the contest spec
  and prevents accidental misuse with non-power-of-2 hc_mult (`tl.arange`
  requires power of 2).
- Comment fix: total normalize passes = 40, not 39 (1 softmax + 1 col + 19×2).

**Knobs to tune in v2+:**
- `BLOCK_T` (try 1, 4, 16, 32, 64) — small decode shapes (`n=1`) waste threads with BLOCK_T>1
- `num_warps` (try 1, 2, 4)
- Scaffold a separate kernel for tiny `n` (decode-heavy) vs large `n` (prefill)
- Consider 2D grid over `(token_block, ?)` if reduce dim grows
- Try pulling base_comb load outside the BLOCK_T loop into a true broadcast

---

## v1 platform results — 2026-05-06

Submitted to GOSIM platform. Avg speedup **40.11x**, **rank #3 globally**.

| Backend       | v1 score | #1 yijun.yu | gap     |
|---------------|----------|-------------|---------|
| 华为昇腾       | **9.41** | 8.40        | +1.01 ✅ |
| 天数 (Iluvatar)| 72.86    | 71.80       | +1.06 ✅ |
| 海光 (Hygon)   | **38.70**| 37.45       | +1.25 ✅ |
| 摩尔线程 (MTT) | 57.94    | **96.42**   | **−38.48** 🚨 |
| 沐曦 (MetaX)   | 21.66    | 21.66       | 0.00    |
| **avg**       | **40.11**| **47.15**   | -7.04   |

We lead 3/5 backends, tie on 沐曦, but lose hard on 摩尔线程. Hypothesis:
v1's `tl.static_range(19)` full unroll causes register spilling / slow
compile on muTriton. Lift 摩尔线程 to ~96 → avg becomes 47.81 → #1 globally.

---

## v2 — 2026-05-06

**Changes:**
1. `tl.static_range(19)` → `tl.range(SINKHORN_ITERS - 1)` (runtime loop).
   Removes 19× code unroll. Targets muTriton compile health.
2. `@triton.autotune` over 10 configs:
   - `BLOCK_T ∈ {1, 4, 8, 16, 32, 64, 128}` × `num_warps ∈ {1, 2, 4}`
   - `key=['n_tokens']` so each shape (decode n=1/16/64, prefill n=16k/65k)
     gets its own tuned config.
3. `grid` becomes a `lambda meta:` callable so it picks up autotune's BLOCK_T.

**Risks:**
- First-call autotune cost (10 configs × 5 distinct n_tokens = 50 compile+bench).
  Should amortize during real benchmarking, but if judge measures cold-start
  this could regress.
- `tl.range` may not pipeline correctly on every backend (e.g. Ascend); if so
  consider `range(...)` (Python) which Triton may auto-handle.
- Runtime loop adds per-iter overhead vs unroll. For 19 iters of small body,
  unroll _might_ have been faster on strong compilers (Ascend, MetaX). Watch
  for regression on those backends.

**Expected wins:**
- 摩尔线程: 57 → 80-100 (the main target)
- 海光 / 天数: stay similar or slight improvement from autotune picking better BLOCK_T
- 昇腾: possibly improves with BLOCK_T=64 or 128 (architectural preference for fewer programs)
- 沐曦: probably still 21.66 (sw-locked)

**Submit, then read leaderboard, decide v3.**

---

## v2 platform results — 2026-05-06

**🥇 #1 globally with avg 55.89x** (vs prior #1 yijun.yu @ 47.15).

| Backend       | v1 score | v2 score   | diff        |
|---------------|----------|------------|-------------|
| 华为昇腾       | 9.41     | **failed** | −9.41 ❌    |
| 天数 Iluvatar  | 72.86    | 77.18      | +4.32 ✅    |
| 海光 Hygon     | 38.70    | 53.22      | +14.52 ✅   |
| 摩尔线程 MTT   | 57.94    | **149.05** | **+91.11** 🚀 |
| 沐曦 MetaX     | 21.66    | **failed** | −21.66 ❌   |
| **avg**       | 40.11    | **55.89**  | +15.78      |

The runtime-loop hypothesis was correct — MTT exploded ~2.6×. But Ascend
and MetaX both crashed (showing `–`), most likely because **`tl.range`
isn't fully implemented in their Triton forks** (Triton-Ascend / mcTriton).
Numerical correctness can't be the issue: MTT passed, so the algorithm is
right. It's a compile/runtime crash on the `tl.range` codegen path.

---

## v3 — 2026-05-06

**Hypothesis:** `tl.range` is the only thing that breaks Ascend + MetaX.
Replace it with a non-constexpr int loop bound — Triton cannot unroll
because the count isn't known at compile time, so it emits the same
runtime loop on every backend. No `tl.range` call site at all.

**Single change:** new kernel arg `n_extra_iters: int` (regular, not
constexpr), used as `for _ in range(n_extra_iters)`. Wrapper passes
`sinkhorn_iters - 1`.

**Expected outcomes:**
- Best case: Ascend + MetaX recover to v1 levels, MTT stays at 149.
  Avg = (9.41 + 77.18 + 53.22 + 149.05 + 21.66) / 5 = **62.10x**.
- Neutral: Ascend recovers but MetaX still fails (different cause).
  Avg = (9.41 + 77.18 + 53.22 + 149.05 + 0) / 5 = 57.77.
- Worst: change degrades MTT somehow → drop in the rankings.

**Risk minimization:** v3 is a near-minimal diff from v2. If MTT regresses
significantly we can re-submit v2 immediately (2-min cooldown).

---

## v3 platform results — 2026-05-06

| Backend       | v2     | v3     | diff   |
|---------------|--------|--------|--------|
| 华为昇腾       | failed | failed | —      |
| 天数 Iluvatar  | 77.18  | 74.78  | -2.40  |
| 海光 Hygon     | 53.22  | 49.90  | -3.32  |
| 摩尔线程 MTT   | 149.05 | 149.12 | +0.07  |
| 沐曦 MetaX     | failed | failed | —      |
| **avg**       | 55.89  | **54.76** | -1.13 |

`tl.range` was NOT the only issue — Ascend + MetaX still fail. Slight
regression on Hygon/Iluvatar (probably loss of `tl.range` pipelining hints).

---

## Chinese-internet research findings (2026-05-06)

Searched Zhihu, CSDN, Aliyun blog, Gitee issues, Triton-Ascend tutorials.
Key insights:

1. **Runtime loops with non-constexpr bounds almost certainly break Ascend.**
   Every official Ascend Triton tutorial uses `tl.static_range` with
   constexpr bounds. Open `ConvertTritonIRToLinalgIR` bugs on Gitee.
   Upstream `tl.range` segfaults (#4368, #8259) inherited by forks.
2. **`@triton.autotune` is "supported but fragile"** on both Ascend and
   mcTriton. Mechanism exists; the failure mode is **a single config
   exceeding UB / shared memory crashes the whole autotune** because the
   launcher has no graceful per-config skip. mcTriton even extends
   `triton.Config` with `pipeline`/`scenario` keys.
3. **>5 pointer args known to segfault on Triton-Ascend** (Gitee #ICNAR5).
   Our kernel has 6 pointer args, but v1 worked, so this isn't fatal —
   probably depends on shape / layout.
4. **Canonical Ascend pattern is constexpr-only loops + manual config.**
   No published autotune workflow uses non-constexpr loops.
5. **No GOSIM KernelGen 2026 contestant writeups exist yet** — too new.

---

## v4 — 2026-05-06

**Design: dual-kernel + try/except dispatch**

- **Primary path (fast)**: `@triton.autotune` over `_kernel_runtime` (uses
  `range(n_extra_iters)` runtime loop). Wins on MTT.
- **Fallback path (portable)**: explicit `(BLOCK_T, num_warps)` call into
  `_kernel_static` (uses `tl.static_range(SINKHORN_ITERS - 1)` constexpr
  unroll, mirroring v1 design which worked on every backend).
- Module-level `_AUTOTUNE_OK` flag — once autotune raises, stop retrying.

**Autotune trimming based on research:**
- Dropped `(BLOCK_T=128, num_warps=4)` to avoid UB OVERFLOW crashing the
  whole autotune sweep on Ascend.
- 8 configs total (was 10).

**Why this should work:**
- Hygon / Iluvatar / MTT: autotune path succeeds → keeps v3 scores
  (MTT ~149, Hygon ~50, Iluvatar ~75).
- Ascend / MetaX: autotune raises (whether due to autotune itself or
  the runtime loop) → caught → fall back to static path → matches v1
  known-good configuration → recovers ~9-12x and ~21x respectively.

**Expected outcome:**
- Best: (12 + 75 + 50 + 149 + 22) / 5 = **61.6x** (lock #1)
- Realistic: (10 + 73 + 48 + 145 + 21) / 5 = **59.4x**
- Worst (fallback also crashes): same as v3 (54.76x) — no regression vs current

**Submit, observe.**

---

## v4 platform results — 2026-05-06 (marked "Best")

| Backend       | v3     | v4         | diff   |
|---------------|--------|------------|--------|
| 华为昇腾       | failed | **7.75**   | RECOVERED ✅ |
| 天数 Iluvatar  | 74.78  | 75.23      | +0.45  |
| 海光 Hygon     | 49.90  | 51.83      | +1.93  |
| 摩尔线程 MTT   | 149.12 | 146.35     | -2.77  |
| 沐曦 MetaX     | failed | **failed** | still ❌ |
| **avg**       | 54.76  | **56.23**  | +1.47  |

try/except worked for Ascend (autotune raises catchable Exception) but
NOT for MetaX. Hypothesis: MetaX's failure is uncatchable (process kill /
segfault). Confirmed by deep research below.

---

## Deep research on MetaX (2026-05-06, second pass)

Searched MetaX-MACA/mcTriton repo, mcTriton User Guide, FlagTree
third_party/metax, FlagGems, Zhihu, CSDN, vLLM-metax.

Key findings:

1. **mcTriton autotune failure is SIGSEGV/SIGABRT**, not a Python exception.
   The crash happens inside `mxgpu_llvm` / `metaxTritonPlugin` native code
   when a tested config hits an MLIR backend assertion. Python try/except
   has no chance.
2. **Canonical fix from mcTriton's own matmul tutorial:**
   ```python
   triton.runtime.driver.active.get_current_target().backend == "maca"
   ```
   Returns `"maca"` only on MetaX (`"cuda"` on Iluvatar/MTT, `"hip"` on
   Hygon, `"npu"` on Ascend). The matmul example branches on this and
   uses different config lists per backend.
3. **mcTriton extends triton.Config** with `pipeline` (`basic`/`cpasync`)
   and `scenario` keys, but they're optional. Not the failure cause.
4. **MetaX's own non-GEMM tutorials avoid @triton.autotune entirely** —
   strong tell that they consider it fragile.
5. **No `MACA_DISABLE_AUTOTUNE` env var** exists. Useful env vars:
   `TRITON_DISABLE_SWIZZLE=1` (defensive, may avoid cpasync crash class),
   `TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS=1` (caches autotune to disk
   — only helps if pre-seeded, which we can't do).
6. **No GOSIM contestant writeups exist** for any backend yet.

---

## v5 — 2026-05-06

**Strategy:** detect MetaX at runtime and bypass autotune entirely on it,
straight to the v1-style static path. Keep autotune+runtime-loop for the
other 4 backends.

Changes from v4:
- Added `_is_metax_backend()` using mcTriton's canonical detection
  (`triton.runtime.driver.active.get_current_target().backend == "maca"`)
  with `torch.cuda.get_device_name()` fallback. Cached after first call.
- In wrapper: `if _is_metax_backend(): skip autotune`.
- Set `TRITON_DISABLE_SWIZZLE=1` at module top (no-op except on MetaX).
- Reverted `_static_config()` to always return `(16, 1)` — exact v1 config
  that gave 9.41 on Ascend and 21.66 on MetaX. v4's per-n dispatch
  hurt Ascend (9.41 → 7.75); v1's flat config is safer.

**Expected outcome:**
- MetaX: skip autotune → static path → ~21x (recovers v1 score)
- Ascend: same v4 try/except + static path, but better config → ~9x (back to v1)
- Hygon / Iluvatar / MTT: autotune unchanged → keep v4 scores
- Best: (9 + 75 + 52 + 146 + 22) / 5 = **60.8x**
- Worst: detection misses MetaX → same as v4 (56.23x), no regression

**Why it can't make things worse:**
- If MetaX detection works → MetaX recovers, no other backend touched
- If detection fails (returns False on MetaX) → behavior identical to v4

This is a strict-improvement-or-no-op patch.

---

## v5 platform results — 2026-05-06

🥇 **#1 globally with avg 60.32x** (vs prior #2 yijun.yu @ 47.15 — +13.17 lead).

| Backend       | v4         | v5         | diff   |
|---------------|------------|------------|--------|
| 华为昇腾       | 7.75       | 7.41       | -0.34  |
| 天数 Iluvatar  | 75.23      | 74.23      | -1.00  |
| 海光 Hygon     | 51.83      | 52.22      | +0.39  |
| 摩尔线程 MTT   | 146.35     | 146.43     | +0.08  |
| 沐曦 MetaX     | failed     | **21.32**  | RECOVERED ✅ |
| **avg**       | 56.23      | **60.32**  | +4.09  |

MetaX detection worked perfectly. Ascend slightly regressed (-0.34) — the
failed autotune attempt itself costs time even though try/except recovers.

---

## v6 — 2026-05-06

**Two changes targeting the remaining headroom:**

### Change 1: also bypass autotune on Ascend
v5 Ascend 7.41 < v1's 9.41. The 2-point gap is the wasted autotune sweep
that try/except catches but doesn't reverse. Add `_is_ascend_backend()`
using `target.backend == "npu"` (Triton-Ascend's identifier per FlagTree)
plus device-name fallback. Same dispatch as MetaX → straight to static.

### Change 2: re-add `BLOCK_T=128` autotune configs
v3 dropped `(BLOCK_T=128, num_warps=4)` to keep Ascend/MetaX autotune from
crashing on UB OVERFLOW. Now both bypass autotune entirely, so the
constraint is moot. The remaining autotune consumers (Hygon, Iluvatar,
MTT) are all NV-style backends — upstream Triton autotune gracefully
skips configs that fail to compile, so adding speculative large-tile
configs is upside-only. Also added `(BLOCK_T=128, num_warps=8)` as
speculation for MTT.

### Refactor
Extracted `_detect_backend(target_names, device_name_hints)` helper so
MetaX and Ascend detection share the same two-layer (driver target +
device name string) logic. Cached results in module-level flags.

### Expected outcome
- Ascend: 7.41 → ~9.4 (recovers v1 baseline by skipping wasted attempt)
- MTT: 146 → ~149-155 (recovers v2's BLOCK_T=128 win)
- Other backends: same as v5
- avg: 60.32 → **~62-63x**

### Risks
- `_is_ascend_backend()` mis-identifying another backend as Ascend → that
  backend regresses to v1 static (16x worst case) → avg could drop ~10
  points. **Mitigation:** two-layer detection with conservative substring
  list ("ascend", "910", "davinci", "huawei").
- BLOCK_T=128 num_warps=8 might fail to compile on Iluvatar/Hygon →
  upstream autotune skips, picks next config → no harm.
- Platform takes best, so worst case = stay at v5's 60.32.

---

## v6 platform results — 2026-05-06

🥈 **#2 globally with avg 71.85x.** Lost #1 to mdzaidlm1019 @ 117.69.

| Backend       | v5     | v6         | diff   |
|---------------|--------|------------|--------|
| 华为昇腾       | 7.41   | **7.51**   | +0.10  |
| 天数 Iluvatar  | 74.23  | 74.68      | +0.45  |
| 海光 Hygon     | 52.22  | 54.23      | +2.01  |
| 摩尔线程 MTT   | 146.43 | **201.32** | **+54.89** 🚀 |
| 沐曦 MetaX     | 21.32  | 21.51      | +0.19  |
| **avg**       | 60.32  | **71.85**  | +11.53 |

MTT jumped much more than predicted (+54 vs predicted +5), almost certainly
because BLOCK_T=128 num_warps=8 was actually a great config and v2 had only
tested num_warps=4. Ascend bypass also recovered the +0.10.

**Competitor analysis:** mdzaidlm1019 has Ascend=1.00 (essentially baseline,
not failed=0 — the platform reports 0.00 for failed runs), Iluvatar=45.07,
Hygon=28.36, **MTT=499.82**, MetaX=14.20. They're worse than us on 4 of 5
backends but their MTT is 2.49× ours, and the arithmetic mean lets that one
score dominate. Their strategy: optimize MTT to the moon, leave others at
baseline. We need to either (a) match their MTT while keeping our floor, or
(b) accept the same trade.

---

## v7 — 2026-05-06

**Strategy: expand autotune sweep on the MTT-relevant axis.**

v6's autotune sweep had two unexplored dimensions:
1. **`num_stages`** never set anywhere → defaults to 1. NV-style backends
   typically benefit from `num_stages=2/3` software pipelining.
2. **`BLOCK_T` capped at 128.** Per-token register footprint is just 16 fp32
   (`comb` 4×4) ≈ 64 B, so 256-token tiles should fit easily.

**Changes:**
- Added 4 `num_stages=2/3` configs at BLOCK_T 64/128.
- Added 4 BLOCK_T=256 configs (with and without pipelining).
- Total configs: 11 → 19.

**Risk model:**
- Hygon / Iluvatar / MTT (all autotune consumers): NV-style upstream
  Triton skips uncompilable configs gracefully → upside-only.
- Ascend / MetaX: bypass autotune entirely → unaffected.
- First-call autotune cost: 19 × 5 distinct n = 95 compile/bench. Higher
  warm-up cost than v6, but `triton.testing.do_bench` does its own warmup
  before timing. Should be fine.

**Expected outcome (per backend, conservative → optimistic):**
- MTT: 201 → 250–400 if pipelining or BLOCK_T=256 helps.
- Hygon: 54 → 55–65 (already well-tuned, small upside).
- Iluvatar: 75 → 75–85 (small upside).
- Ascend / MetaX: unchanged.

**Worst case:** autotune still picks v6's BLOCK_T=128, num_warps=8 → score
identical to v6 → 71.85. No regression risk because new configs are pure
additions.

**Submit, observe, decide v8.**

---

## v7 platform results — 2026-05-06 (FAILED, all 5 backends)

| Backend       | v6     | v7     | diff |
|---------------|--------|--------|------|
| 华为昇腾       | 7.51   | failed | ❌   |
| 天数 Iluvatar  | 74.68  | failed | ❌   |
| 海光 Hygon     | 54.23  | failed | ❌   |
| 摩尔线程 MTT   | 201.32 | failed | ❌   |
| 沐曦 MetaX     | 21.51  | failed | ❌   |

All 5 failed simultaneously → module-level breakage. Hypothesis at the time:
`triton.Config(..., num_stages=2)` rejected by some vendor fork's Config
class at module import. **Subsequent research disproved this** — every fork
(upstream 3.5.x, FlagTree main + metax/mthreads/hcu overrides, Triton-Ascend
3.5.0, Iluvatar) accepts `num_stages` in `Config.__init__`. The actual cause
was almost certainly a **specific (BLOCK_T, num_warps, num_stages) corner
in the sweep crashing one or more backends' compilers uncatchably**.

---

## v7b platform results — 2026-05-06 (regressed)

Defensive `_make_config` wrapper + dropped `num_stages`, kept BLOCK_T=256/512.

| Backend       | v6     | v7b    | diff   |
|---------------|--------|--------|--------|
| 华为昇腾       | 7.51   | 7.70   | +0.19  |
| 天数 Iluvatar  | 74.68  | 74.94  | +0.26  |
| 海光 Hygon     | 54.23  | 50.60  | -3.63 ❌|
| 摩尔线程 MTT   | 201.32 | 154.99 | -46.33 ❌|
| 沐曦 MetaX     | 21.51  | 21.28  | -0.23  |
| **avg**       | 71.85  | 61.90  | -9.95  |

MTT regression: autotune picked one of the new BLOCK_T=256/512 configs as
"winner" in its noisy micro-bench, but the picked config is slower than v6's
BLOCK_T=128 winner on the actual workload. Hygon similar mechanism.

**Lesson**: adding configs to autotune is NOT free upside. Without local
benchmark capability, every speculative addition risks autotune picking a
noise-driven false positive.

---

## Research pass: FlagGems MUSA non-MMA configs (2026-05-06)

Cloned `FlagOpen/FlagGems` master and surveyed `src/flag_gems/runtime/backend/_mthreads/`.
Goal: find evidence-backed configs for non-MMA reduction-style kernels on
muTriton, since our `hc_split_sinkhorn` is structurally that.

Key findings:

1. **`num_stages>1` IS safe inside `@triton.autotune` on MUSA for non-MMA
   ops** — FlagGems ships amax/max/argmin/prod/log/celu with mixed
   `num_stages=1, num_stages=2` configs in autotune sweeps that pass MUSA CI.
   The cap is **num_stages ≤ 2** for reductions; ≥3 only appears in GEMM
   paths (mm/mv/nonzero).
2. **Closest analog kernel**: `cross_entropy_loss` — per-token softmax over
   tiny axis + pointwise. Same structural pattern as ours. Its MUSA config:
   `BLOCK_C ∈ {256, 512, 1024}, num_warps=4, no num_stages set`.
3. **`_num_warps()` heuristic** (`heuristics_config_utils.py:35-42`): for
   block sizes 128–256, FlagGems picks `num_warps=4`.
4. No public muTriton non-MMA tutorial. FlagGems is the only public
   reference for what works on MUSA.
5. `MUSA_ENABLE_SQMMA=1` is only relevant for MMA paths. Our fp32 reduction
   kernel doesn't need it; no other MUSA env vars matter for this case.

---

## v8 — 2026-05-06

**Strategy: MTT-explicit path based on FlagGems CE-loss MUSA pattern.**

Changes from v6:
- Added `_is_mtt_backend()` (`backend == "musa"`) detection.
- New explicit MTT path in wrapper, bypassing autotune entirely:
  - prefill (n > 64): `BLOCK_T=256, num_warps=4` (no `num_stages`)
  - decode (n ≤ 64): `BLOCK_T=next_pow2(n), num_warps=1`
- MTT added to `skip_autotune` so failed-explicit-path falls into static, not autotune.
- Reverted `_AUTOTUNE_CONFIGS` to v6's exact 11 entries (no v7b additions).
- Removed `_make_config` defensive wrapper — only used in v7b, no longer needed.

**Why this should beat v6 MTT=201:**
- v6 autotune picked BLOCK_T=128 num_warps=8 as winner. FlagGems MUSA sweet
  spot for reductions is BLOCK ∈ [256, 1024] with num_warps=4. v6's choice
  is below MUSA's optimum band.
- BLOCK_T=256 ships as a default in FlagGems' production MUSA backend ops,
  so we know it compiles cleanly and runs correctly — no crash risk.

**Why this can't make v6 worse on other backends:**
- Hygon, Iluvatar: identical to v6 (autotune sweep restored, MTT branch
  inert because `_is_mtt_backend()` returns False).
- Ascend, MetaX: identical to v6 (still bypass to static).

**Risk:**
- If our hypothesis is wrong and MUSA actually prefers BLOCK_T=128 num_warps=8,
  MTT regresses from 201 to whatever 256/4 gives. Bounded downside — we
  can't fail since the config is FlagGems-validated. Worst case is a small
  regression on MTT only.
- Platform takes best, so failure mode = wasted submission, not lost rank.

**Expected outcome:**
- MTT: 201 → 250–350 (if FlagGems analog holds)
- Other backends: identical to v6
- Avg: 71.85 → ~80–100

---

## v8 platform results — 2026-05-06 (regressed badly)

| Backend       | v6     | v8     | diff    |
|---------------|--------|--------|---------|
| 华为昇腾       | 7.51   | 7.41   | -0.10   |
| 天数 Iluvatar  | 74.68  | 78.21  | +3.53 ✓ |
| 海光 Hygon     | 54.23  | 52.18  | -2.05   |
| 摩尔线程 MTT   | 201.32 | **92.32** | **-108.83** ❌❌❌ |
| 沐曦 MetaX     | 21.51  | 21.35  | -0.16   |
| **avg**       | 71.85  | 50.30  | -21.55  |

**The FlagGems CE-loss analogy was structurally wrong.** CE-loss processes
`[BATCH, C]` where C is vocab size (~32k) — per-row compute is enormous
relative to the row count. Its `BLOCK_C=256, num_warps=4` choice optimizes
for that ratio.

Our kernel is the inverse: per-token compute is tiny (4-element softmax + 39
4-element reductions = ~160 ops/token) and we have many tokens. v6's
autotune-picked `BLOCK_T=128, num_warps=8` is actually the correct shape:
- 8 warps × 32 = 256 threads, 128 tokens → 2 threads / token → parallelism
  for the 4-element axis reductions
- vs v8's 4 warps × 32 = 128 threads, 256 tokens → 0.5 threads / token →
  serial processing per thread, lower parallelism on the inner axis

**Lesson**: per-row compute size is a first-order parameter that determines
the BLOCK / num_warps trade-off. FlagGems' MUSA non-MMA kernels span
dropout (huge per-row), CE-loss (huge), layer_norm (huge), batch_norm
(small). Only batch_norm has a similar shape to ours, and it doesn't use
autotune (heuristic-derived). No public MUSA kernel resembles "tiny
per-token + many tokens" closely enough to transfer configs.

---

## v9 = exact v6 revert — 2026-05-06

3 blind tunings (v7, v7b, v8) all hurt. Without local muTriton hardware to
bench, we cannot reliably beat v6's autotune on MTT. Reverting to exactly
v6 to restore the 71.85 / #2 baseline.

Changes: removed `_is_mtt_backend`, `_IS_MTT_CACHED`, the explicit MTT
dispatch path, and any v7b/v8 residue. Result is byte-equivalent (modulo
docstring) to the v6 file that scored 71.85.

**Conclusion on track-02 MTT optimization**: parked. To go further we
need either (a) muTriton hardware to bench candidate configs locally, or
(b) a fundamentally different algorithm (not just config tuning) — e.g.,
fewer sinkhorn iterations via convergence detection, or a closed-form
doubly-stochastic projection. Both are higher risk and orthogonal to
config-level work.

---

## Notes on workload mix

- Three of four scenarios are **decode-dominant** (99.8% calls).
  Decode shape is always small (`b≤64, s=1`), so launch overhead matters a lot.
- The `1k-1-64` scenario has **67% prefill** — different optimization target.
  If our decode kernel suffers on prefill, this workload will tank.
- The bench script's `weighted` row uses the per-workload call frequencies
  — that's the number we actually want to maximize.

## Leaderboard reset — 2026-05-06

User pulled the latest leaderboard. Several new entries massively pushed MTT:

| Rank | Player | Ascend | Iluvatar | Hygon | MTT | MetaX | avg |
|------|--------|--------|----------|-------|-----|-------|-----|
| 1 | flagos | 13.32 | – | 63.80 | **520.67** | **26.28** | 124.81 |
| 2 | 1550266278 | 12.68 | – | 34.25 | 538.60 | 15.85 | 120.28 |
| 3 | mdzaidlm1019 | 1.00 | 45.07 | 28.36 | 499.82 | 14.20 | 117.69 |
| 4 | farid (GOSIM) | 8.44 | 55.46 | 30.12 | 292.93 | 14.73 | 80.33 |
| **5** | **us (GOSIM)** | 7.51 | **74.68** | **54.23** | 201.32 | **21.51** | 71.85 |

Strategic situation:
- **We're #1 on Iluvatar (74.68)**, #1 on Hygon (54.23), #2 on MetaX (21.51).
- We're losing primarily on MTT (201 vs 290–540).
- **farid is our direct GOSIM Winner competitor** (also has GOSIM tag).
  Beating farid requires avg ≥ 80.33.
- MTT 201 → ~290 with all else equal would give us avg = (7.51+74.68+54.23+290+21.51)/5 = 89.6 → beat farid for GOSIM Winner.
- We can't bench MTT locally; track-02 v7-v8 confirmed blind MTT changes
  are dangerous. Pivoting to MetaX where the gap is achievable: 21.51 → 26+
  matches #1 and adds another #1 backend to our wins.

---

## Deep research: MetaX track-02 specific (~200 tool calls) — 2026-05-06

Surveyed MetaX-MACA org (vLLM-metax, mcTriton, mcoplib, TileKernels-Metax,
McFlashInfer, flashattn), FlagGems _metax/, vLLM upstream, DeepSeek
TileKernels, Chinese tech sources, GOSIM forum. 185 tool calls.

**Decisive finding**: **MetaX-MACA/TileKernels-Metax** (created 2026-04-29)
is DeepSeek's mHC TileKernels port to MetaX. It contains
`tile_kernels/mhc/pre_big_fuse_kernel.py` with `tests/mhc/test_pre_big_fuse.py`
testing **the exact operation our `hc_split_sinkhorn` implements**.

DeepSeek's MetaX reference uses:
- 1 program per token (`T.Kernel(num_tokens, threads=96)`)
- 96 threads = 1.5 warps × 64 ≈ Triton `num_warps=2`
- `TL_DISABLE_VECTORIZE_256: True` — 256-bit vectorization HURTS on MetaX

**MetaX-MACA/vLLM-metax/.../fused_moe.py:1554-1594** is vendor-shipping
production code with size-dependent heuristic:
```
M ≤ 32 → block_m = 16
M ≤ 128 → num_warps = 4
else → num_warps = 8
```
Plus line 1518 explicit comment: `num_stages > 2 OOMs on Maca`. They
hardcode `num_stages_maca = 2`.

**MetaX-MACA/mcoplib c500-optimization-guide.md** (official MetaX repo,
HEAD 2026-05-06):
- 104 SMs (grid sizes should be multiples of 104)
- **Warp size = 64** (NOT 32 like NVIDIA — `num_warps=N` in Triton =
  N × 64 threads on MetaX!)
- 64 K registers / SM (256 regs/thread max)
- 64 KB shared memory / SM
- "Reduction: Threads/Block 512–1024, Warps 16–32" (= num_warps=8 in Triton)

**FlagGems `_metax` non-MMA kernels survey** (the closest analogs to ours):
| Op | Configs |
|----|---------|
| log_softmax | N≤1024→num_warps=1, ≤2048→4, else 8 |
| dropout | N≤512→warps=4, ≤1024→8, else **16** |
| layer_norm_loop | TILE_N ∈ {1024, 2048, 4096, 8192}, num_warps ∈ {4, 8, 16} |
| mean/sum reductions | BLOCK_M ∈ {1,2,4,8}, num_warps=4 fixed |

**Anti-leads (confirmed from research)**:
- `pipeline="cpasync"` — only fires when MMA optimizer is on; useless for our
  non-MMA kernel
- `scenario="flashattn-fwd"` / `"mla"` — MMA-specific, won't help us
- `num_stages > 2` — explicit OOM crash per vLLM-MetaX comment
- `num_warps=16` for our 4-element reduction axis — wastes occupancy
  (60+ idle threads / reduction)
- Adding `num_stages` to autotune sweep — v7 confirmed crashes

---

## v10 — 2026-05-06

**Strategy: codify vllm-metax + DeepSeek TileKernels patterns into MetaX-only static config.**

Single change to wrapper. New `_metax_static_config(n)`:
```python
def _metax_static_config(n: int):
    if n <= 64:    return 16, 1   # decode — v9 verified 21.51
    if n <= 512:   return 32, 4
    if n <= 8192:  return 64, 4
    return 128, 8                 # large prefill — vllm-metax M>128
```

Static path now branches: MetaX uses `_metax_static_config` + `num_stages=1`,
Ascend keeps `_static_config = (16, 1)` per v4 lesson. Try/except wraps
the num_stages=1 launch — if rejected, retries without num_stages (= v9
behavior at the new size-dependent BLOCK_T).

**Workload impact**:
| Workload | Frequency | n | v9 config | v10 config |
|----------|-----------|---|-----------|------------|
| 16k-1k-1 decode | 99.8% | 1 | (16, 1) | (16, 1) [unchanged] |
| 16k-1k-1 prefill | 0.2% | 16384 | (16, 1) | **(128, 8)** |
| 4k-1k-16 decode | 99.8% | 16 | (16, 1) | (16, 1) [unchanged] |
| 4k-1k-16 prefill | 0.2% | 65536 | (16, 1) | **(128, 8)** |
| 1k-1k-64 decode | 99.8% | 64 | (16, 1) | (16, 1) [unchanged] |
| 1k-1k-64 prefill | 0.2% | 65536 | (16, 1) | **(128, 8)** |
| **1k-1-64 decode** | **33%** | 64 | (16, 1) | (16, 1) [unchanged] |
| **1k-1-64 prefill** | **67%** | 65536 | (16, 1) | **(128, 8)** ⭐ |

The 1k-1-64 prefill path (67% of calls in that workload) is the primary
beneficiary. Other 3 workloads' decode (99%) is unchanged.

**Risk model:**
- Decode (16, 1) unchanged → v9's 21.51 floor preserved on the 99% path.
- Prefill bumped to (128, 8) — vendor-validated config that ships in
  vllm-metax production. Compile failure unlikely; performance regression
  at worst.
- num_stages=1 wrapped in try/except. If MACA OOMs (per vLLM comment, only
  >2 OOMs but be safe), falls back to no num_stages.
- Other backends UNTOUCHED.

**Expected outcomes:**
- Best case: MetaX 21.51 → 28-32 (matches/beats #1 flagos at 26.28). Avg
  71.85 → ~73-75. Still need MTT for GOSIM Winner threshold.
- Realistic: MetaX 21.51 → 24-26. Avg 71.85 → ~72-73.
- Null: prefill bump didn't help (compute-axis bound, not parallelism).
  Decode unchanged → MetaX = 21.51, avg = 71.85.

**Submit, observe.**

---

## Decision log

(Empty — populate as we make non-obvious choices.)
