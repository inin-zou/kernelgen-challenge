# GOSIM KernelGen 2026 — Project Context

Triton kernel optimization hackathon. Three tracks targeting 5 domestic Chinese AI accelerators via FlagTree (Huawei Ascend, Iluvatar CoreX, Hygon DCU, Moore Threads MUSA, MetaX C500).

## Submission Rules (NON-NEGOTIABLE)

> Must all computation implements via `@triton.jit`. The torch op can be used **only for memory operations such as `torch.ones`** in the wrapper. Vendor BLAS is **not allowed**.

Specifically:
- ❌ `torch.matmul`, `F.linear`, `torch.softmax`, `torch.gather` in hot path
- ❌ `torch.nn.functional.scaled_dot_product_attention` (vendor flash attn)
- ❌ `torch.ops.npu.*`, `torch.ops.musa.*` etc. (vendor ops)
- ❌ `torch.cuda.*` even for backend detection (use `triton.runtime.driver.active.get_current_target().backend`)
- ❌ TileLang (`@T.prim_func`) — different DSL, not `@triton.jit`
- ⚠️ `torch.compile` — generates Triton internally but rules-grey
- ✅ `tensor.contiguous()`, `tensor.view()`, `torch.empty()` — pure memory ops
- ✅ `triton.cdiv`, `triton.next_power_of_2` — Triton helpers
- ✅ Borrowing FlagGems / TileLang **algorithmic patterns**, re-implementing in `@triton.jit`

## Tracks

### track-01: Sparse Attention
- `sparse_attn(q[b,m,h=16,d=512] bf16, kv[b,kv_len,d] bf16 (K==V shared), attn_sink[h] fp32, topk_idxs[b,m,topk] int32, scale)`
- 10 workloads: 5 prefill (m up to 16384) + 5 decode (m=1, b≤64), topk 128–640
- **Current rank: #4, avg 1.96x** (v10 / v16 area).
- See `track-01/notes.md` for the v1–v16 iteration history.

### track-02: DeepSeek mHC
- `hc_split_sinkhorn(mixes [b,s,24], hc_scale [3], hc_base [24], hc_mult=4, sinkhorn_iters=20, eps=1e-6)`
- `pre`, `post`, `comb` outputs; sink-style softmax + Sinkhorn iterations
- **Final rank: #1 globally, avg 71.85x** (v6).
- Big win: `(BLOCK_T=128, num_warps=8)` on Moore Threads — surprise gem from autotune.

### track-03: Fused MoE
- Spec only, not implemented.

## 5-Backend Dispatch Pattern

Common skeleton in both submission.py files:

```python
def _detect_backend(target_names):
    target = triton.runtime.driver.active.get_current_target()
    return getattr(target, "backend", "") in target_names

def _is_metax_backend():    return _detect_backend(("maca",))
def _is_ascend_backend():   return _detect_backend(("npu", "ascend", "ascendc"))
def _is_mtt_backend():      return _detect_backend(("musa", "mt", "mthread", "mthreads"))
def _is_hygon_backend():    return _detect_backend(("hip",))
def _is_iluvatar_backend(): return _detect_backend(("cuda",)) and not _is_mtt_backend()
```

## Per-Backend Lessons Learned

| Backend | Identifier | Critical knobs / pitfalls |
|---------|-----------|---------------------------|
| Ascend (NPU) | `"npu"` | Triton-Ascend compiler is FRAGILE. Simple `(BLOCK=16, num_warps=2)` is the ceiling — every "optimization" attempt (2-level tiling, persistent kernel, num_stages=2, D-chunked extract_slice) made it slower or crashed. UB ≈ 192 KB. Set `TRITON_ALL_BLOCKS_PARALLEL=1` only when grid > 65535. |
| Iluvatar | `"cuda"` (also MTT) | Standard NV-style. `(BLOCK=16, num_warps=2, num_stages=1)` for d=512 (FlagGems _iluvatar pattern). |
| Hygon | `"hip"` | ROCm-style. Autotune over BLOCK ∈ {32, 64, 128} works well. |
| Moore Threads | `"musa"` | bf16 mma path requires `MUSA_ENABLE_SQMMA=1` env var around launch + `H_padded = max(32, npow2(h)) = 32`. WITHOUT both, bf16 falls back to scalar (12x slower). FlagGems config: BLOCK=32, num_warps=16, num_stages=6. |
| MetaX | `"maca"` | mcTriton's `tl.dot` `acc=` form helped slightly (0.21→0.27); bigger BLOCK didn't move the needle; `num_warps=8` actually hurt vs `num_warps=2`. **Stuck at 0.27 — fundamental algorithm/hardware mismatch we couldn't crack.** Set `TRITON_DISABLE_SWIZZLE=1` defensively. |

## Backend Failure Modes

- **MetaX**: autotune crashes are **uncatchable SIGSEGV/SIGABRT** — Python `try/except` does NOT save you. MUST detect and bypass autotune upfront.
- **Ascend**: autotune raises catchable exception; can be wrapped. But ANY config beyond simple breaks performance.
- **MTT**: autotune crashes uncatchable too — bypass autotune, hardcode config.
- **Hygon / Iluvatar**: autotune works fine.

## Project Structure

```
kernelgen-challenge/
├── .claude/
│   ├── CLAUDE.md            # This file
│   └── settings.local.json  # Context7 MCP setup
├── track-01/
│   ├── submission.py        # Submitted to judge
│   ├── reference/reference.py  # Frozen PyTorch oracle
│   ├── workloads.py         # 10 test shape configs
│   ├── test_correctness.py  # vs reference (atol=5e-2, bf16)
│   ├── bench.py             # 10-workload benchmark
│   ├── profile.py           # Profiling helper
│   ├── notes.md             # v1-v16 tuning log
│   └── requirements.md      # Task spec
├── track-02/  (same layout, won #1 globally)
├── track-03/  (requirements.md only)
└── .gitignore               # Excludes tilelang_reference, .flaggems_research, __pycache__
```

## External Reference Material (gitignored)

When researching, often cloned to local but not committed:
- `track-02/tilelang_reference/`: DeepSeek's TileKernels repo (TileLang impl)
- `track-02/.flaggems_research/`: FlagGems per-vendor sparse_attention kernels (Apache-2.0)

These are read-only references; we **port algorithmic patterns** into our `@triton.jit` kernels but do NOT import them at runtime (compliance risk).

## Submission Workflow Notes

- Platform: https://kernelgen.flagos.io/ (FlagOS-hosted)
- 2-minute cooldown between submissions
- Platform takes **best** of all submissions (not latest) — safe to experiment
- Each evaluation runs all 5 backends sequentially (~5–10 min total)
- Score = arithmetic mean of per-backend speedups vs reference; failed backend = 0

## Gotchas

1. **Tolerances are loose** (atol=5e-2 for bf16) — fp32-vs-bf16 numerical drift across backends is expected.
2. **Triton 3.5** in eval — keep API to broadly-supported subset.
3. **`tl.range`** is fragile across forks; prefer plain `range(non_constexpr_int)` for runtime loops.
4. **`tl.trans`** on fp32 tiles fails on some forks; use it on bf16 tiles where possible.
5. **`H` in `tl.arange(0, H)`** must be power of 2 → for h=16 we're fine; pad to 32 on MTT.
