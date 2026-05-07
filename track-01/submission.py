"""
Submission file for GOSIM KernelGen Track-01: Sparse Attention.

v19 changes vs v18 (MetaX still 0.31 — far below leaderboard median 1.5–2.5):

Realization: the 0.27 → 0.31 ceiling we hit through v17/v18 is NOT a hardware
mismatch. Six leaderboard contestants land MetaX 1.17–3.13x with rules-
compliant Triton kernels. We've been byte-porting FlagGems' broken upstream
sparse_attention kernel (its CI test is disabled — confirmed via repo dive)
and inheriting its structural flaws.

Per fla-org/native-sparse-attention parallel.py L703/L758 (the production
NSA Triton implementation), the canonical fix is **3D grid with V-axis
chunking**:
- grid = (m, b, NV) where NV = cdiv(d=512, BV=128) = 4
- acc_o = [H, BV] not [H, D] — 8 KB per program instead of 32 KB
- Each program redundantly recomputes QK (cheap relative to PV); the V
  slice [BLOCK, BV] = 8 KB replaces the full [BLOCK, D] = 32 KB tile

Adds Tier 0 to the MetaX dispatch chain calling the new
`_sparse_attn_kernel_v19_v_chunked`. Existing v17/v18 tiers preserved as
Tier 1–5 fallback. Other 4 backends untouched.

v18 (kept) — MetaX env-var probes for 4 compiler passes at module top.

Background on v17 (kept):
- **Probe 4 MetaX compiler-pass enable env vars** at module top (see import
  block below). Source: mcTriton compiler.py lines 220, 222, 304, 305.
- All four default OFF in mcTriton; setting them is a zero-touch experiment:
  non-MetaX backends never read these names. Most relevant to our kernel:
  MERGE_CONVERT_LAYOUT (we emit a P bf16-convert between two dots) and
  MOVE_DOT_OPERANDS_OUT_LOOP (lifts loop-invariant operand loads).
- No code changes outside the env block. v17's tier 1/2/3 MetaX dispatch
  preserved. Other 4 backends untouched.

Background on v17 (kept):

Target: MetaX 0.27 → 0.5–0.8 by triggering chain-dot OPT MMA + flashattn-fwd
LLVM scheduler. Other backends untouched.

Research-validated changes (mcTriton 1122c717 source dive + FlagGems
_metax/fused/flash_mla.py):
- **`num_stages=2`** (was unset → defaulted to 3): mcTriton's chain-dot OPT
  MMA fast path is gated on `numStages == 2` per
  AccelerateMETAXMatmul.cpp:177-185. With 3 it falls back to the slow path —
  this is the most likely root cause of the 0.27 ceiling.
- **`BLOCK=32`** (was 16): FlagGems flash_mla on C500 (compute-cap 8) uses
  BLOCK_N=32. Larger tile gives the chain-dot opt enough work to use 8 warps.
- **`pipeline="basic", scenario="flashattn-fwd"`** (NEW MACAOptions kwargs):
  scenario="flashattn-fwd" activates a hand-tuned LLVM flag preset
  (mcTriton compiler.py:371-372): `metaxgpu-mma-sched=true`,
  `metaxgpu-sched-select=metaxgpu-minreg`, `map-use-pk-fma=1` —
  the minreg scheduler directly addresses our D=512 register pressure.

Tiered fallback if any v17 kwarg is rejected:
  Tier 1: full v17 (num_stages=2 + BLOCK=32 + MACAOptions kwargs)
  Tier 2: num_stages=2 + BLOCK=32 only (drop MACAOptions kwargs)
  Tier 3: v16 byte-exact (BLOCK=16, no num_stages) — our 0.27 floor
  Tier 4: universal autotune
  Tier 5: v9 simple bf16

Worst case = Tier 3 = current 0.27. Other backends (Ascend, Iluvatar, Hygon,
MTT) are not touched — their dispatch paths are byte-identical to v16.

v10 mechanisms preserved for Ascend/Hygon/Iluvatar/MTT (all verified working).

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

# v18: probe additional MetaX (mcTriton) compiler-pass enable flags. All four
# default to OFF in mcTriton compiler.py; setting them at module load is a
# zero-touch experiment — non-MetaX backends ignore unknown TRITON_* env vars.
# Sources (mcTriton commit 1122c717, third_party/metax/backend/compiler.py):
#   :220 — TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP gates
#          add_tritonmetaxgpu_move_dot_operands_out_loop_pass (lifts dot
#          operand loads above the loop, reduces register churn).
#   :222 — TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT gates
#          add_tritonmetaxgpu_merge_convert_layout_pass (fuses redundant
#          layout conversions — directly relevant since our chain-dot kernel
#          emits a P bf16-convert before the second dot).
#   :304 — TRITON_ENABLE_SMEM_OFFSET_CACHE = scenario "smemOffsetCache".
#   :305 — TRITON_ENABLE_BSM_INDEX_OPT    = scenario "smemIndexOpt".
# Worst case any one of these regresses MetaX — each is reversible by a
# single-flag flip in v19. Other backends never read these names.
os.environ.setdefault("TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP", "1")
os.environ.setdefault("TRITON_ENABLE_MACA_MERGE_CONVERT_LAYOUT", "1")
os.environ.setdefault("TRITON_ENABLE_SMEM_OFFSET_CACHE", "1")
os.environ.setdefault("TRITON_ENABLE_BSM_INDEX_OPT", "1")

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


# ===========================================================================
# v19: NSA-style V-chunked kernel for MetaX
# ---------------------------------------------------------------------------
# The 0.27 ceiling we hit through v18 is NOT a hardware ceiling — 6 other
# leaderboard contestants hit 1.17–3.13x with rules-compliant Triton kernels
# while we (and inin-zou, who byte-ported the same FlagGems _metax kernel)
# are stuck at 0.27. FlagGems' _metax/fused/sparse_attention.py is broken
# upstream (its CI test is disabled), so byte-porting it just inherits its
# brokenness. Restructured per fla-org/native-sparse-attention parallel.py
# (L703, L758) — the production NSA Triton implementation.
#
# Key structural change: 3D grid (m, b, NV) with NV = cdiv(D, BV). Each
# program owns acc_o[H, BV] only — at BV=128, that's 8 KB instead of 32 KB,
# matching NSA's `b_o = tl.zeros([G, BV], fp32)` pattern. acc_o no longer
# spills out of registers on C500.
#
# Cost: each (b, m, v_chunk) program redundantly recomputes the full QK
# dot. That's O(BLOCK*D) per iter, vs O(BLOCK*BV) for the non-redundant
# PV dot. Net work increase ~2.5x but parallelism increases 4x and register
# pressure drops 4x. NSA accepts this trade-off; it's MetaX's binding
# constraint.
#
# All V-chunk programs converge to identical softmax stats (m_i, sum_exp)
# because the QK pass is deterministic and replicated. Output is correct.
# ===========================================================================
@triton.jit
def _sparse_attn_kernel_v19_v_chunked(
    Q, KV, O, ATTN_SINK, IDX,
    stride_qb, stride_qm, stride_qh, stride_qd,
    stride_kvb, stride_kvn, stride_kvd,
    stride_ob, stride_om, stride_oh, stride_od,
    stride_idxb, stride_idxm, stride_idxk,
    SCALE,
    TOPK,
    H_ACTUAL,
    BLOCK: tl.constexpr,        # topk chunk size
    D: tl.constexpr,            # full d (used for QK dot)
    BV: tl.constexpr,           # v-chunk size; acc_o = [H, BV]
    H: tl.constexpr,            # padded head count
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_v = tl.program_id(2)        # v-axis chunk index

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_bv = pid_v * BV + tl.arange(0, BV)
    bv_mask = offs_bv < D
    h_mask = offs_h < H_ACTUAL

    # Load Q full-D once — used for QK dot every iteration.
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = IDX + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, BV], dtype=tl.float32)         # NSA pattern: per-chunk
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (TOPK + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < TOPK
        idxs = tl.load(idx_base + raw_offs * stride_idxk, mask=idx_mask, other=-1)
        valid_mask = idxs != -1

        # K load: full D for QK. Live tile [BLOCK, D] bf16 = 32 KB at BLOCK=32.
        k_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        k_block = tl.load(k_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(k_block))
        acc_s = acc_s * SCALE
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        # V load: only this chunk's BV slice. Tile [BLOCK, BV] bf16 = 8 KB at
        # BLOCK=32, BV=128. NSA L734 uses make_block_ptr offset by i_v*BV; we
        # use direct ptr arithmetic since our gather is per-token random.
        v_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_bv[None, :] * stride_kvd
        v_block = tl.load(
            v_ptrs,
            mask=valid_mask[:, None] & bv_mask[None, :],
            other=0.0,
        )

        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), v_block)         # NSA L742 cast pattern

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    # Sink only contributes to the denominator (sink V is excluded by spec).
    sink_vals = tl.load(ATTN_SINK + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    # Write only this chunk's BV slice of the output.
    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_bv[None, :] * stride_od
    tl.store(
        o_ptrs,
        acc_o.to(tl.bfloat16),
        mask=h_mask[:, None] & bv_mask[None, :],
    )


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

    # ---- MetaX (v19): NSA-style V-chunked kernel — primary path ----
    # The 0.27 → 0.31 ceiling we hit through v18 was NOT a hardware ceiling.
    # Per leaderboard, 6 other contestants land MetaX 1.17–3.13x with
    # rules-compliant Triton; we (and inin-zou with the same FlagGems _metax
    # byte-port) are stuck at 0.27 because we replicate FlagGems' broken
    # upstream kernel. Restructured per fla-org/native-sparse-attention:
    # 3D grid (m, b, NV) chunks the d=512 V-axis into NV=cdiv(D, BV) shards.
    # Each program owns acc_o[H, BV=128] = 8 KB instead of 32 KB — addresses
    # the register-spill hypothesis. Redundant QK dot per chunk (~2.5x extra
    # compute) is dwarfed by 4x parallelism + 4x register-pressure drop.
    # If v19 fails for any reason, fall through to v17/v16 chain-dot path.
    if _is_metax_backend():
        h_padded = max(16, triton.next_power_of_2(h))
        BV = 128
        # Tier 0: v19 V-chunked NSA-pattern kernel
        try:
            NV = (d + BV - 1) // BV
            grid = (m, b, NV)
            _sparse_attn_kernel_v19_v_chunked[grid](
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
                BV=BV,
                H=h_padded,
                num_warps=4,            # NSA autotune sweeps {1, 2, 4}; 4 conservative
            )
            return o
        except Exception:
            pass

    # ---- MetaX (v17): activate chain-dot OPT MMA + flashattn-fwd LLVM scheduler ----
    # Research finding (mcTriton 1122c717 AccelerateMETAXMatmul.cpp:177-185 + 371-381):
    # MetaX's chain-dot OPT MMA fast path requires num_stages == 2 to fire. Without
    # it (default 3), our existing byte-port v16 silently took the slow path,
    # explaining the 0.27 ceiling. FlagGems flash_mla (the closest production
    # MetaX kernel structurally matching ours — chain dot in loop, bf16 IO) ships
    # exactly BLOCK_N=32, num_warps=8, num_stages=2 on compute-cap 8 (C500).
    #
    # Lead 2 (mcTriton compiler.py:371-372): scenario="flashattn-fwd" activates
    # MetaX's hand-tuned LLVM flag preset:
    #   -metaxgpu-mma-sched=true -metaxgpu-sched-select=metaxgpu-minreg
    #   -map-use-pk-fma=1
    # The minreg scheduler directly addresses our D=512 register pressure.
    # pipeline="basic" + scenario="flashattn-fwd" are MACAOptions kwargs;
    # Triton forwards unknown launch kwargs to backend options. On non-MetaX
    # backends these would either be ignored or raise — but this branch only
    # runs on MetaX so other backends are unaffected.
    #
    # Tier sequence: Tier 1 (most aggressive: num_stages=2 + BLOCK=32 + MACAOpts)
    # → Tier 2 (num_stages=2 + BLOCK=32 only, drops MACAOpts if those kwargs fail)
    # → Tier 3 (v16 byte-exact, our 0.27 floor) → Tier 4 (autotune) → Tier 5 (bf16).
    if _is_metax_backend():
        h_padded = max(16, triton.next_power_of_2(h))
        # Tier 1: chain-dot OPT MMA + flashattn-fwd LLVM preset (full v17)
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
                BLOCK=32,                          # was 16; FlagGems flash_mla C500 uses 32
                D=d,
                H=h_padded,
                num_warps=8,
                num_stages=2,                      # KEY: enables chain-dot OPT MMA
                pipeline="basic",                  # MACAOptions: standard pipeline
                scenario="flashattn-fwd",          # MACAOptions: hand-tuned LLVM preset
            )
            return o
        except Exception:
            pass
        # Tier 2: drop MACAOptions kwargs in case Triton's launcher rejects them,
        # but keep num_stages=2 + BLOCK=32 (the chain-dot OPT trigger).
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
                BLOCK=32,
                D=d,
                H=h_padded,
                num_warps=8,
                num_stages=2,
            )
            return o
        except Exception:
            pass
        # Tier 3: v16 byte-exact FlagGems _metax config — our known 0.27 floor.
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
                BLOCK=16,
                D=d,
                H=h_padded,
                num_warps=8,
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
