# GOSIM KernelGen 2026

Triton kernel optimization for 5 domestic Chinese AI accelerators via [FlagTree](https://github.com/FlagTree/flagtree). Competing on [kernelgen.flagos.io](https://kernelgen.flagos.io/).

**Team**: yongkang.zou.ai@gmail.com (GOSIM on-site)

## Tracks

### Track 01 — Sparse Attention

Dynamic sparse attention kernel (`sparse_attn`) for DeepSeek-style top-k gather-then-attend.

- **Signature**: `sparse_attn(q[b,m,h=16,d=512] bf16, kv[b,kv_len,d] bf16, attn_sink[h] fp32, topk_idxs[b,m,topk] int32, scale) -> o[b,m,h,d] bf16`
- **Best score**: avg **1.97x** (#5 global)
- **Per-backend bests**: Ascend 1.73 | Iluvatar 3.99 (#1) | Hygon 1.12 | MTT 2.77 | MetaX 0.31
- **Key techniques**: FlashAttention-style online softmax, per-backend dispatch (5 backends), bf16 tensor-core path for Iluvatar/MTT, fp32 fallback for Ascend, NSA V-chunked kernel for MetaX

### Track 02 — DeepSeek mHC (hc_split_sinkhorn)

Fused multi-stream mix + doubly-stochastic Sinkhorn projection for DeepSeek Manifold-Constrained Hyper-Connections.

- **Signature**: `hc_split_sinkhorn(mixes[b,s,24] fp32, hc_scale[3], hc_base[24], hc_mult=4, sinkhorn_iters=20, eps=1e-6) -> (pre[b,s,4], post[b,s,4], comb[b,s,4,4])`
- **Best score**: avg **71.85x** (#5 global)
- **Per-backend bests**: Ascend 7.51 | Iluvatar 74.68 (#1) | Hygon 54.23 (#1) | MTT 201.32 | MetaX 21.51 (#2)
- **Key techniques**: Runtime-loop Sinkhorn (avoids compile-time unroll bloat on MTT), `@triton.autotune` for Hygon/Iluvatar/MTT, static fallback for Ascend/MetaX, per-n size-dependent MTT dispatch, `MUSA_ENABLE_LLC_OPT=1` compiler flag

### Track 03 — Fused MoE

Spec only. Minimal diagnostic submissions exploring backend compatibility.

## 5-Backend Dispatch Pattern

All tracks share a common backend detection skeleton:

```python
target = triton.runtime.driver.active.get_current_target()
backend = getattr(target, "backend", "")
# "maca" = MetaX, "npu" = Ascend, "musa" = MTT, "hip" = Hygon, "cuda" = Iluvatar
```

Each backend has different Triton fork behavior (autotune crashes on MetaX/MTT, `tl.range` fragile on Ascend, etc.), so per-backend dispatch with try/except fallback chains is essential.

## Per-Backend Lessons

| Backend | Key insight |
|---------|------------|
| **Ascend** | Fragile compiler — only `(BLOCK=16, num_warps=2)` works reliably. ANY deviation regresses. |
| **Iluvatar** | Standard CUDA-style. FlagGems universal kernel + autotune works well. Our strongest backend. |
| **Hygon** | ROCm-style. Autotune over BLOCK works. Stable. |
| **MTT (MUSA)** | Runtime loop critical (static unroll 3x slower). `MUSA_ENABLE_LLC_OPT=1` enables disabled compiler optimizations. Autotune noise requires explicit per-n dispatch. `num_warps=16` for large prefill per FlagGems pattern. |
| **MetaX** | Autotune SIGSEGV (uncatchable). `num_stages=2` triggers chain-dot OPT MMA. `TRITON_DISABLE_SWIZZLE=1` defensive. NSA V-chunked kernel addresses register spill at D=512. |

## Project Structure

```
kernelgen-challenge/
├── .claude/CLAUDE.md          # AI assistant project context
├── track-01/
│   ├── submission.py          # Submitted to judge
│   ├── reference/reference.py # Frozen PyTorch oracle
│   ├── workloads.py           # 10 test shape configs
│   ├── test_correctness.py    # Correctness vs reference
│   ├── bench.py               # Benchmark harness
│   └── notes.md               # v1-v19 iteration log
├── track-02/                  # Same layout
│   └── notes.md               # v1-v12 iteration log
└── track-03/                  # Minimal
```

## Rules

- All computation via `@triton.jit` — no `torch.matmul`, `F.softmax`, vendor BLAS
- `torch` ops only for memory allocation (`torch.empty`, `.contiguous()`, `.view()`)
- Evaluation on Triton 3.5 across all 5 FlagTree backends
- Score = arithmetic mean of per-backend speedups vs PyTorch reference
