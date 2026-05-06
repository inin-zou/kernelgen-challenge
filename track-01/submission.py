"""
Submission file for GOSIM KernelGen Track-01: Sparse Attention.

v10 changes vs v9 (v9 = 1.55 rank #5):

Targeted MetaX + MTT algorithm-level breakthroughs (research-validated):

- **MTT (0.74 → expected 4+):** Use universal kernel + H_padded=32 (NOT 16).
  v8 failed because SQMMA needs M >= 32; we used H=16. With H_padded=32 + SQMMA
  env + (BLOCK=32, num_warps=16, num_stages=6) — exact FlagGems _mthreads
  recipe — should unlock bf16 tensor core.
- **MetaX (0.21 → expected 2.5+):** mcTriton autotune over BLOCK 16-64,
  num_warps 2-8, num_stages 1-3. Plus universal kernel uses accumulator-form
  `tl.dot(p, kv, acc=acc_o)` (research recommendation from FlagGems flash_mla.py).
- **Universal kernel improved**: now uses `acc=` form on tl.dot. Should also
  help Iluvatar (no expected regression, possibly small uplift).
- Both MTT and MetaX paths wrapped in try/except — fall back to v9 known-good
  (fp32 for MTT, simple bf16 for MetaX) if the new path crashes.

v9 mechanisms preserved for Ascend/Hygon/Iluvatar (all verified working).

Compliance:
- All numerical compute in @triton.jit.
- Torch ops only for memory operations.
- Backend detection via Triton runtime API only.

Source attribution: kernel structure adapted from FlagOpen/FlagGems
runtime/backend/_*/fused/sparse_attention.py and flash_mla.py (Apache-2.0).

v9 changes vs v8 (kept):

Diagnosis (from research):
- FlagGems sparse_attention is BROKEN UPSTREAM (issues #2669, #2809; PR #2819
  disabled accuracy test in CI). We ported code FlagGems team itself can't
  pass tests with. Iluvatar happened to land in a working sweet spot.
- v8's MTT config (num_warps=16, num_stages=6) over-specs muTriton (Triton
  3.1, max num_warps=8 on non-NV; num_stages>=4 needs unimplemented async).
- v8's Ascend module-top env var is racy + `BATCH_STRIDE: tl.constexpr=m`
  causes recompile per shape.
- v8's Hygon hardcoded (BLOCK=16, num_warps=8) — FlagGems Hygon is a
  placeholder kernel with no tune configs; v7 autotune was strictly better.

v9 strategy: keep what worked, revert what didn't.

| Backend  | v9 path                                              |
|----------|------------------------------------------------------|
| Iluvatar | v8 universal kernel + (16, 2, 1)  ← Verified 3.98    |
| Ascend   | v7 simple bf16 kernel + (16, 2)   ← Verified 1.70    |
| MetaX    | v7 simple bf16 kernel + (16, 2)   ← Verified 0.21    |
| Hygon    | v7 autotune over BLOCK_TOPK       ← Verified 1.09    |
| MTT      | v5 fp32 kernel + (64, 4)          ← Verified 0.79    |

Expected avg: (1.70 + 3.98 + 1.09 + 0.79 + 0.21) / 5 = 1.55 → rank #4

Compliance:
- All numerical compute in @triton.jit.
- Torch ops only for memory operations.
- Backend detection via Triton runtime API only.

Source attribution: universal kernel (used for Iluvatar) adapted from
FlagOpen/FlagGems runtime/backend/_*/fused/sparse_attention.py (Apache-2.0).
"""
import os
os.environ.setdefault("TRITON_DISABLE_SWIZZLE", "1")

import torch
import triton
import triton.language as tl


# ===========================================================================
# Kernel 1: SIMPLE bf16 kernel (v7-style — verified on Ascend, MetaX, Hygon)
# Loads Q without masking (assumes H = power of 2 = 16 in our case).
# Uses tl.where on scores directly. Sink uses max(m_i, sink) rescale.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_bf16(
    Q, KV, SINK, IDX, O,
    M, KV_LEN, TOPK,
    SCALE,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base   = pid_b * (M * H * D) + pid_m * (H * D)
    kv_base  = pid_b * (KV_LEN * D)
    idx_base = pid_b * (M * TOPK) + pid_m * TOPK
    o_base   = q_base

    h_idx = tl.arange(0, H)
    d_idx = tl.arange(0, D)

    q_ptrs = Q + q_base + h_idx[:, None] * D + d_idx[None, :]
    q_bf   = tl.load(q_ptrs)

    sink = tl.load(SINK + h_idx)

    NEG_INF = float('-inf')
    m_i = tl.full([H], NEG_INF, tl.float32)
    l_i = tl.zeros([H],         tl.float32)
    o_i = tl.zeros([H, D],      tl.float32)

    for k_start in range(0, TOPK, BLOCK_TOPK):
        k_off  = k_start + tl.arange(0, BLOCK_TOPK)
        k_mask = k_off < TOPK

        idx = tl.load(IDX + idx_base + k_off, mask=k_mask, other=0)

        kv_ptrs = KV + kv_base + idx[:, None] * D + d_idx[None, :]
        kv_bf   = tl.load(kv_ptrs, mask=k_mask[:, None], other=0.0)

        scores = tl.dot(q_bf, tl.trans(kv_bf), out_dtype=tl.float32) * SCALE
        scores = tl.where(k_mask[None, :], scores, NEG_INF)

        m_new      = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha      = tl.exp(m_i - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        scores_exp = tl.where(k_mask[None, :], scores_exp, 0.0)

        l_i = alpha * l_i + tl.sum(scores_exp, axis=1)
        scores_exp_bf = scores_exp.to(tl.bfloat16)
        o_i = alpha[:, None] * o_i + tl.dot(scores_exp_bf, kv_bf, out_dtype=tl.float32)
        m_i = m_new

    m_total   = tl.maximum(m_i, sink)
    alpha     = tl.exp(m_i - m_total)
    sink_term = tl.exp(sink - m_total)
    l_total   = alpha * l_i + sink_term
    o_final   = (alpha[:, None] * o_i) / l_total[:, None]

    o_ptrs = O + o_base + h_idx[:, None] * D + d_idx[None, :]
    tl.store(o_ptrs, o_final.to(tl.bfloat16))


# ===========================================================================
# Kernel 2: fp32 kernel (v5-style — verified on MTT)
# muTriton's bf16 mma path is unreliable; fp32 input → TF32 path works.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_fp32(
    Q, KV, SINK, IDX, O,
    M, KV_LEN, TOPK,
    SCALE,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base   = pid_b * (M * H * D) + pid_m * (H * D)
    kv_base  = pid_b * (KV_LEN * D)
    idx_base = pid_b * (M * TOPK) + pid_m * TOPK
    o_base   = q_base

    h_idx = tl.arange(0, H)
    d_idx = tl.arange(0, D)

    q_ptrs = Q + q_base + h_idx[:, None] * D + d_idx[None, :]
    q      = tl.load(q_ptrs).to(tl.float32)

    sink = tl.load(SINK + h_idx)

    NEG_INF = float('-inf')
    m_i = tl.full([H], NEG_INF, tl.float32)
    l_i = tl.zeros([H],         tl.float32)
    o_i = tl.zeros([H, D],      tl.float32)

    for k_start in range(0, TOPK, BLOCK_TOPK):
        k_off  = k_start + tl.arange(0, BLOCK_TOPK)
        k_mask = k_off < TOPK

        idx = tl.load(IDX + idx_base + k_off, mask=k_mask, other=0)

        kv_ptrs = KV + kv_base + idx[:, None] * D + d_idx[None, :]
        kv_bf   = tl.load(kv_ptrs, mask=k_mask[:, None], other=0.0)
        kv_f    = kv_bf.to(tl.float32)

        scores = tl.dot(q, tl.trans(kv_f), out_dtype=tl.float32) * SCALE
        scores = tl.where(k_mask[None, :], scores, NEG_INF)

        m_new      = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha      = tl.exp(m_i - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        scores_exp = tl.where(k_mask[None, :], scores_exp, 0.0)

        l_i = alpha * l_i + tl.sum(scores_exp, axis=1)
        o_i = alpha[:, None] * o_i + tl.dot(scores_exp, kv_f, out_dtype=tl.float32)
        m_i = m_new

    m_total   = tl.maximum(m_i, sink)
    alpha     = tl.exp(m_i - m_total)
    sink_term = tl.exp(sink - m_total)
    l_total   = alpha * l_i + sink_term
    o_final   = (alpha[:, None] * o_i) / l_total[:, None]

    o_ptrs = O + o_base + h_idx[:, None] * D + d_idx[None, :]
    tl.store(o_ptrs, o_final.to(tl.bfloat16))


# ===========================================================================
# Kernel 3: UNIVERSAL bf16 kernel (FlagGems-style — verified 3.98 on Iluvatar)
# Uses h_mask, idx==-1 sentinel, simpler sink. Different structural choices
# than the simple kernel above, which happen to win big on Iluvatar/d=512.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_universal(
    Q, KV, O, ATTN_SINK, IDX,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kvb, stride_kvn, stride_kvd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_idxb, stride_idxm, stride_idxk,
    SCALE,
    TOPK,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (TOPK + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < TOPK
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(kv_block))
        acc_s = acc_s * SCALE
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        # Accumulator-form tl.dot — keeps acc in MMA registers, avoids extra
        # register move. FlagGems _metax/fused/flash_mla.py uses this; per
        # research it's ~1.3-2x speedup on MetaX vs `acc_o += tl.dot(...)`.
        acc_o = acc_o * correction[:, None]
        acc_o = tl.dot(p.to(tl.bfloat16), kv_block, acc=acc_o)

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# ===========================================================================
# Kernel 7 (v14): Ascend D-chunked output update via tl.extract_slice
# Per Triton-Ascend tutorial 04-fused-attention pattern for HEAD_DIM>=256:
# split [H, D=512] acc/kv matmul into chunks of [H, BLOCK_D=128].
# Each smaller matmul is more Cube-friendly (16x16x128 fits L0A/L0B).
# Keep first matmul (scores) full-width since it's K-reduce-heavy.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_ascend_dchunk(
    Q, KV, SINK, IDX, O,
    M, KV_LEN, TOPK,
    SCALE,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base   = pid_b * (M * H * D) + pid_m * (H * D)
    kv_base  = pid_b * (KV_LEN * D)
    idx_base = pid_b * (M * TOPK) + pid_m * TOPK
    o_base   = q_base

    h_idx = tl.arange(0, H)
    d_idx = tl.arange(0, D)

    q_ptrs = Q + q_base + h_idx[:, None] * D + d_idx[None, :]
    q_bf   = tl.load(q_ptrs)

    sink = tl.load(SINK + h_idx)

    NEG_INF = float('-inf')
    m_i = tl.full([H], NEG_INF, tl.float32)
    l_i = tl.zeros([H],         tl.float32)
    o_i = tl.zeros([H, D],      tl.float32)

    for k_start in range(0, TOPK, BLOCK_TOPK):
        k_off  = k_start + tl.arange(0, BLOCK_TOPK)
        k_mask = k_off < TOPK

        idx = tl.load(IDX + idx_base + k_off, mask=k_mask, other=0)

        kv_ptrs = KV + kv_base + idx[:, None] * D + d_idx[None, :]
        kv_bf   = tl.load(kv_ptrs, mask=k_mask[:, None], other=0.0)

        # First matmul: scores [H, BLOCK_TOPK] = Q @ KV^T (full D, K-reduce-heavy)
        scores = tl.dot(q_bf, tl.trans(kv_bf), out_dtype=tl.float32) * SCALE
        scores = tl.where(k_mask[None, :], scores, NEG_INF)

        m_new      = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha      = tl.exp(m_i - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        scores_exp = tl.where(k_mask[None, :], scores_exp, 0.0)

        l_i = alpha * l_i + tl.sum(scores_exp, axis=1)
        scores_exp_bf = scores_exp.to(tl.bfloat16)

        # Second matmul, D-CHUNKED via extract_slice/insert_slice
        # Each chunk: [H, BLOCK_D] += scores_exp [H, BLOCK_TOPK] @ kv_chunk [BLOCK_TOPK, BLOCK_D]
        o_i = o_i * alpha[:, None]
        # Static range — Triton-Ascend may unroll for better scheduling.
        # D=512, BLOCK_D=128 → 4 chunks.
        for d_off in tl.static_range(0, D, BLOCK_D):
            kv_chunk = tl.extract_slice(kv_bf, [0, d_off], [BLOCK_TOPK, BLOCK_D])
            acc_chunk = tl.extract_slice(o_i, [0, d_off], [H, BLOCK_D])
            acc_chunk = tl.dot(scores_exp_bf, kv_chunk, acc=acc_chunk, out_dtype=tl.float32)
            o_i = tl.insert_slice(o_i, acc_chunk, [0, d_off])

        m_i = m_new

    m_total   = tl.maximum(m_i, sink)
    alpha     = tl.exp(m_i - m_total)
    sink_term = tl.exp(sink - m_total)
    l_total   = alpha * l_i + sink_term
    o_final   = (alpha[:, None] * o_i) / l_total[:, None]

    o_ptrs = O + o_base + h_idx[:, None] * D + d_idx[None, :]
    tl.store(o_ptrs, o_final.to(tl.bfloat16))


# ===========================================================================
# Kernel 6 (v13): Ascend TileLang-style optimized kernel
# Adopts TileLang-Ascend bench_sfa SparseFlashAttention patterns within
# Triton-Ascend's available primitives:
#   - Larger BLOCK_TOPK (32 vs v9's 16) — fewer iterations, more parallelism
#     per program, while staying within Ascend's 192KB UB budget
#       q[16,512]bf16 (16KB) + kv[32,512]bf16 (32KB) + acc[16,512]fp32 (32KB)
#       + scratch ≈ 86 KB total, safe
#   - num_stages=2 — Triton-Ascend's auto-pipelining hint, MTE2-Cube overlap
#   - tl.dot(p, kv_block, acc=acc_o) accumulator form — keeps acc in MMA regs
#   - Same simple structure as v9 simple bf16 (no h_mask, no 2-level)
# Expected: 1.72 → 2.5-3.5 (1.5-2x), pure config + pipelining hints.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_ascend_tiled(
    Q, KV, SINK, IDX, O,
    M, KV_LEN, TOPK,
    SCALE,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    q_base   = pid_b * (M * H * D) + pid_m * (H * D)
    kv_base  = pid_b * (KV_LEN * D)
    idx_base = pid_b * (M * TOPK) + pid_m * TOPK
    o_base   = q_base

    h_idx = tl.arange(0, H)
    d_idx = tl.arange(0, D)

    q_ptrs = Q + q_base + h_idx[:, None] * D + d_idx[None, :]
    q_bf   = tl.load(q_ptrs)

    sink = tl.load(SINK + h_idx)

    NEG_INF = float('-inf')
    m_i = tl.full([H], NEG_INF, tl.float32)
    l_i = tl.zeros([H],         tl.float32)
    o_i = tl.zeros([H, D],      tl.float32)

    for k_start in range(0, TOPK, BLOCK_TOPK):
        k_off  = k_start + tl.arange(0, BLOCK_TOPK)
        k_mask = k_off < TOPK

        idx = tl.load(IDX + idx_base + k_off, mask=k_mask, other=0)

        kv_ptrs = KV + kv_base + idx[:, None] * D + d_idx[None, :]
        kv_bf   = tl.load(kv_ptrs, mask=k_mask[:, None], other=0.0)

        scores = tl.dot(q_bf, tl.trans(kv_bf), out_dtype=tl.float32) * SCALE
        scores = tl.where(k_mask[None, :], scores, NEG_INF)

        m_new      = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha      = tl.exp(m_i - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        scores_exp = tl.where(k_mask[None, :], scores_exp, 0.0)

        l_i = alpha * l_i + tl.sum(scores_exp, axis=1)
        scores_exp_bf = scores_exp.to(tl.bfloat16)
        # TileLang accumulator-form: keep acc in MMA registers
        acc_correction = alpha[:, None] * o_i
        o_i = tl.dot(scores_exp_bf, kv_bf, acc=acc_correction, out_dtype=tl.float32)
        m_i = m_new

    m_total   = tl.maximum(m_i, sink)
    alpha     = tl.exp(m_i - m_total)
    sink_term = tl.exp(sink - m_total)
    l_total   = alpha * l_i + sink_term
    o_final   = (alpha[:, None] * o_i) / l_total[:, None]

    o_ptrs = O + o_base + h_idx[:, None] * D + d_idx[None, :]
    tl.store(o_ptrs, o_final.to(tl.bfloat16))


# ===========================================================================
# Kernel 5 (v12): Ascend Fixed-Core Persistent Kernel
# Per research (TileLang-Ascend SparseFlashAttention bench_sfa, achieves
# 0.91x of hand-written AscendC):
#   - Launch grid = num_cores (24 on A2), NOT b*m
#   - Each core internally loops over its assigned (b, m) tasks
#   - Workspace stays L2-resident (192 MB) instead of spilling to HBM per
#     program; saves initialization + launch overhead per task
# Body of the inner loop = same as simple bf16 kernel (v9 verified).
# Expected speedup: 1.5-2.5x on Ascend (1.72 → 2.5-3.5).
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_ascend_persistent(
    Q, KV, SINK, IDX, O,
    M, KV_LEN, TOPK,
    SCALE,
    B,                                  # runtime int
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)              # 0..NUM_CORES-1
    total_tasks = B * M
    tasks_per_core = (total_tasks + NUM_CORES - 1) // NUM_CORES
    start = pid * tasks_per_core
    end = tl.minimum(start + tasks_per_core, total_tasks)

    h_idx = tl.arange(0, H)
    d_idx = tl.arange(0, D)
    NEG_INF = float('-inf')

    # Loop over tasks assigned to this core.
    for task_id in range(start, end):
        pid_b = task_id // M
        pid_m = task_id % M

        q_base   = pid_b * (M * H * D) + pid_m * (H * D)
        kv_base  = pid_b * (KV_LEN * D)
        idx_base = pid_b * (M * TOPK) + pid_m * TOPK
        o_base   = q_base

        q_ptrs = Q + q_base + h_idx[:, None] * D + d_idx[None, :]
        q_bf   = tl.load(q_ptrs)

        sink = tl.load(SINK + h_idx)

        m_i = tl.full([H], NEG_INF, tl.float32)
        l_i = tl.zeros([H],         tl.float32)
        o_i = tl.zeros([H, D],      tl.float32)

        for k_start in range(0, TOPK, BLOCK_TOPK):
            k_off  = k_start + tl.arange(0, BLOCK_TOPK)
            k_mask = k_off < TOPK

            idx = tl.load(IDX + idx_base + k_off, mask=k_mask, other=0)

            kv_ptrs = KV + kv_base + idx[:, None] * D + d_idx[None, :]
            kv_bf   = tl.load(kv_ptrs, mask=k_mask[:, None], other=0.0)

            scores = tl.dot(q_bf, tl.trans(kv_bf), out_dtype=tl.float32) * SCALE
            scores = tl.where(k_mask[None, :], scores, NEG_INF)

            m_new      = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha      = tl.exp(m_i - m_new)
            scores_exp = tl.exp(scores - m_new[:, None])
            scores_exp = tl.where(k_mask[None, :], scores_exp, 0.0)

            l_i = alpha * l_i + tl.sum(scores_exp, axis=1)
            scores_exp_bf = scores_exp.to(tl.bfloat16)
            o_i = alpha[:, None] * o_i + tl.dot(scores_exp_bf, kv_bf, out_dtype=tl.float32)
            m_i = m_new

        m_total   = tl.maximum(m_i, sink)
        alpha     = tl.exp(m_i - m_total)
        sink_term = tl.exp(sink - m_total)
        l_total   = alpha * l_i + sink_term
        o_final   = (alpha[:, None] * o_i) / l_total[:, None]

        o_ptrs = O + o_base + h_idx[:, None] * D + d_idx[None, :]
        tl.store(o_ptrs, o_final.to(tl.bfloat16))


# ===========================================================================
# Kernel 4 (v11): Ascend 2-level tiling kernel — fixed per research
# Differences from v8 attempt that failed:
# - BATCH_STRIDE is now a *runtime int* (not constexpr) → no recompile per
#   shape. Per vllm-ascend PR #7645 this is the canonical Ascend pattern.
# - TRITON_ALL_BLOCKS_PARALLEL env var set inside wrapper at launch time
#   (not at module import) — per vllm-ascend PR #6301 (thread-safety).
# - Outer BLOCK=64, inner BLOCK_SUB=16 fits Ascend's 192KB UB.
# - 1D grid avoids NPU 65535 coreDim limit.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_ascend(
    Q, KV, O, ATTN_SINK, IDX,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kvb, stride_kvn, stride_kvd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_idxb, stride_idxm, stride_idxk,
    SCALE,
    TOPK,
    KV_LEN,
    H_ACTUAL,
    BATCH_STRIDE,           # runtime int (was constexpr in v8 — that was the bug)
    BLOCK: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // BATCH_STRIDE
    pid_m = pid % BATCH_STRIDE

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_block_iter = (TOPK + BLOCK - 1) // BLOCK
    num_sub_iter = (BLOCK + BLOCK_SUB - 1) // BLOCK_SUB
    offs_blk = tl.arange(0, BLOCK_SUB)

    for t in range(num_block_iter):
        block_start = t * BLOCK
        for s in range(num_sub_iter):
            sub_start = block_start + s * BLOCK_SUB
            raw_offs = sub_start + offs_blk
            idx_mask = raw_offs < TOPK
            idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=0)

            # NPU torch_npu gather quirk: clamp negative indices.
            idxs = tl.where(idxs < 0, 0, idxs)
            index_valid = (idxs >= 0) & (idxs < KV_LEN)
            valid_mask = idx_mask & index_valid

            kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
            kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

            acc_s = tl.dot(q_block, tl.trans(kv_block))
            acc_s = acc_s * SCALE
            mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
            acc_s = acc_s + mask_bias[None, :]

            scores_max_prev = scores_max
            block_max = tl.max(acc_s, axis=1)
            scores_max = tl.maximum(scores_max, block_max)

            correction = tl.exp(scores_max_prev - scores_max)
            p = tl.exp(acc_s - scores_max[:, None])

            acc_o = acc_o * correction[:, None]
            acc_o = tl.dot(p.to(tl.bfloat16), kv_block, acc=acc_o)

            scores_sum = tl.sum(p, axis=1)
            sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# MetaX autotune sweep (v16) — try mcTriton-specific config keywords.
# `pipeline` and `scenario` are mcTriton-only Config extensions documented in
# the User Guide. The kernel below declares them as tl.constexpr params so
# the autotune machinery passes them through. mcTriton interprets them as
# compile-time codegen hints; standard Triton would also accept them as
# unused constexpr constants.
_METAX_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK': 16}, num_warps=8, num_stages=1),
    triton.Config({'BLOCK': 32}, num_warps=8, num_stages=1),
    triton.Config({'BLOCK': 64}, num_warps=8, num_stages=1),
    triton.Config({'BLOCK': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 128}, num_warps=8, num_stages=1),
]

_sparse_attn_universal_autotuned = triton.autotune(
    configs=_METAX_AUTOTUNE_CONFIGS, key=['TOPK', 'H_ACTUAL', 'D'],
)(_sparse_attn_kernel_universal)


# ===========================================================================
# v16: MetaX-exact kernel — byte-identical to FlagGems _metax/fused/sparse_attention.py
# (uses += form NOT acc= form; num_warps=8). Trying this as MetaX dropped to
# 0.27 with our universal kernel — worse than what FlagGems baseline reportedly
# delivers. Maybe the acc= form is silently slow on mcTriton.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_metax_exact(
    Q, KV, O, ATTN_SINK, IDX,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kvb, stride_kvn, stride_kvd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_idxb, stride_idxm, stride_idxk,
    SCALE,
    TOPK,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (TOPK + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < TOPK
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(kv_block))
        acc_s = acc_s * SCALE
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        # FlagGems-metax style: += form (NOT acc= form like our universal kernel)
        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# Autotune for Hygon (verified strategy from v7).
_AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_TOPK': 16}, num_warps=2),
    triton.Config({'BLOCK_TOPK': 32}, num_warps=2),
    triton.Config({'BLOCK_TOPK': 32}, num_warps=4),
    triton.Config({'BLOCK_TOPK': 64}, num_warps=2),
    triton.Config({'BLOCK_TOPK': 64}, num_warps=4),
    triton.Config({'BLOCK_TOPK': 128}, num_warps=4),
]

_sparse_attn_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS, key=['M', 'TOPK'],
)(_sparse_attn_kernel_bf16)


# ===========================================================================
# Backend detection (Triton runtime API only).
# ===========================================================================
_IS_METAX_CACHED = None
_IS_ASCEND_CACHED = None
_IS_MTT_CACHED = None
_IS_HYGON_CACHED = None
_IS_ILUVATAR_CACHED = None
_AUTOTUNE_OK = True


def _detect_backend(target_names) -> bool:
    try:
        target = triton.runtime.driver.active.get_current_target()
        backend = getattr(target, "backend", "")
        return backend in target_names
    except Exception:
        return False


def _is_metax_backend() -> bool:
    global _IS_METAX_CACHED
    if _IS_METAX_CACHED is None:
        _IS_METAX_CACHED = _detect_backend(("maca",))
    return _IS_METAX_CACHED


def _is_ascend_backend() -> bool:
    global _IS_ASCEND_CACHED
    if _IS_ASCEND_CACHED is None:
        _IS_ASCEND_CACHED = _detect_backend(("npu", "ascend", "ascendc"))
    return _IS_ASCEND_CACHED


def _is_mtt_backend() -> bool:
    global _IS_MTT_CACHED
    if _IS_MTT_CACHED is None:
        _IS_MTT_CACHED = _detect_backend(("musa", "mt", "mthread", "mthreads"))
    return _IS_MTT_CACHED


def _is_hygon_backend() -> bool:
    global _IS_HYGON_CACHED
    if _IS_HYGON_CACHED is None:
        _IS_HYGON_CACHED = _detect_backend(("hip",))
    return _IS_HYGON_CACHED


def _is_iluvatar_backend() -> bool:
    """Iluvatar reports backend == "cuda". Disambiguate from MTT."""
    global _IS_ILUVATAR_CACHED
    if _IS_ILUVATAR_CACHED is None:
        _IS_ILUVATAR_CACHED = _detect_backend(("cuda",)) and not _is_mtt_backend()
    return _IS_ILUVATAR_CACHED


# ===========================================================================
# Wrapper — entry point called by the judge
# ===========================================================================
def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """
    Args:
        q:          [b, m, h, d]   bf16
        kv:         [b, kv_len, d] bf16  (K and V SHARED)
        attn_sink:  [h]            fp32
        topk_idxs:  [b, m, topk]   int32
        scale:      float

    Returns:
        o:          [b, m, h, d]   bf16
    """
    global _AUTOTUNE_OK

    assert q.dim() == 4
    b, m, h, d = q.shape
    kv_len = kv.shape[1]
    topk = topk_idxs.shape[-1]

    # Memory ops only.
    q_c    = q.contiguous()
    kv_c   = kv.contiguous()
    sink_c = attn_sink.contiguous()
    idx_c  = topk_idxs.contiguous()

    o = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)

    # ---- Iluvatar: universal kernel + (16, 2, 1) — v8 verified 3.98 ----
    if _is_iluvatar_backend():
        h_padded = max(16, triton.next_power_of_2(h))
        grid = (m, b)
        _sparse_attn_kernel_universal[grid](
            q_c, kv_c, o, sink_c, idx_c,
            q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
            kv_c.stride(0), kv_c.stride(1), kv_c.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
            scale,
            topk,
            h,
            BLOCK=16,
            D=d,
            H=h_padded,
            num_warps=2,
            num_stages=1,
        )
        return o

    # ---- MTT (v10): try universal kernel + H_padded=32 + SQMMA env ----
    # FlagGems _mthreads exact recipe. SQMMA needs M >= 32, hence H_padded=32.
    # Try universal first; fall back to v5 fp32 if it crashes.
    if _is_mtt_backend():
        try:
            h_padded = max(32, triton.next_power_of_2(h))
            grid = (m, b)
            prev_sqmma = os.environ.get("MUSA_ENABLE_SQMMA")
            os.environ["MUSA_ENABLE_SQMMA"] = "1"
            try:
                _sparse_attn_kernel_universal[grid](
                    q_c, kv_c, o, sink_c, idx_c,
                    q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
                    kv_c.stride(0), kv_c.stride(1), kv_c.stride(2),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
                    scale,
                    topk,
                    h,
                    BLOCK=32,
                    D=d,
                    H=h_padded,
                    num_warps=16,
                    num_stages=6,
                )
            finally:
                if prev_sqmma is None:
                    os.environ.pop("MUSA_ENABLE_SQMMA", None)
                else:
                    os.environ["MUSA_ENABLE_SQMMA"] = prev_sqmma
            return o
        except Exception:
            # Fallback: v5 fp32 kernel
            grid = (b, m)
            _sparse_attn_kernel_fp32[grid](
                q_c, kv_c, sink_c, idx_c, o,
                m, kv_len, topk,
                scale,
                H=h,
                D=d,
                BLOCK_TOPK=64,
                num_warps=4,
            )
            return o

    # ---- MetaX (v16): try byte-exact FlagGems metax kernel first ----
    # We've been at 0.27 with our universal kernel (acc= form). FlagGems
    # _metax/fused/sparse_attention.py uses += form + num_warps=8 + BLOCK=16
    # — try it verbatim. Then fall back through several MetaX-specific configs
    # before giving up to v9 simple bf16.
    if _is_metax_backend():
        h_padded = max(16, triton.next_power_of_2(h))
        # Tier 1: byte-exact FlagGems _metax kernel + their config
        try:
            grid = (m, b)
            _sparse_attn_kernel_metax_exact[grid](
                q_c, kv_c, o, sink_c, idx_c,
                q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
                kv_c.stride(0), kv_c.stride(1), kv_c.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
                scale,
                topk,
                h,
                BLOCK=16,                      # FlagGems metax exact
                D=d,
                H=h_padded,
                num_warps=8,                   # FlagGems metax exact
            )
            return o
        except Exception:
            pass
        # Tier 2: universal kernel with autotune over expanded MetaX configs
        try:
            grid = (m, b)
            _sparse_attn_universal_autotuned[grid](
                q_c, kv_c, o, sink_c, idx_c,
                q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
                kv_c.stride(0), kv_c.stride(1), kv_c.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
                scale,
                topk,
                h,
                D=d,
                H=h_padded,
            )
            return o
        except Exception:
            pass
        # Tier 3: v9 simple bf16 fallback
        grid = (b, m)
        _sparse_attn_kernel_bf16[grid](
            q_c, kv_c, sink_c, idx_c, o,
            m, kv_len, topk,
            scale,
            H=h,
            D=d,
            BLOCK_TOPK=16,
            num_warps=2,
        )
        return o

    # ---- Hygon: autotune over simple bf16 kernel — v7 verified ~1.09 ----
    if _is_hygon_backend():
        if _AUTOTUNE_OK:
            try:
                grid = (b, m)
                _sparse_attn_autotuned[grid](
                    q_c, kv_c, sink_c, idx_c, o,
                    m, kv_len, topk,
                    scale,
                    H=h,
                    D=d,
                )
                return o
            except Exception:
                _AUTOTUNE_OK = False
        # Fallback: hardcoded mid config
        grid = (b, m)
        _sparse_attn_kernel_bf16[grid](
            q_c, kv_c, sink_c, idx_c, o,
            m, kv_len, topk,
            scale,
            H=h,
            D=d,
            BLOCK_TOPK=64,
            num_warps=4,
        )
        return o

    # ---- Ascend (v15): revert to v10 baseline, simple bf16 (16, 2) → 1.72 ----
    # All Ascend optimization experiments (v8 2-level, v11 fixed, v12 persistent,
    # v13 BLOCK=32+stages=2, v14 D-chunk) made it slower. Triton-Ascend's
    # compiler is fragile; the simplest kernel wins.
    # NOOP — fall through to the generic simple bf16 path at the bottom.

    # ---- Ascend / unknown / generic: simple bf16 kernel + (16, 2) ----
    # v15: Ascend explicitly falls through here. Verified in v9-v12 that simple
    # kernel at (16, 2) is the Triton-Ascend ceiling (~1.72).
    grid = (b, m)
    _sparse_attn_kernel_bf16[grid](
        q_c, kv_c, sink_c, idx_c, o,
        m, kv_len, topk,
        scale,
        H=h,
        D=d,
        BLOCK_TOPK=16,
        num_warps=2,
    )
    return o
