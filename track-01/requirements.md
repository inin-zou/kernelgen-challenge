# Task 01: Sparse Attention（动态稀疏注意力）

实现 Top-K 稀疏注意力 Kernel，通过 gather 选取 KV 子集计算注意力，附带 attention sink 机制。

**核心难点**：高效 gather KV、稀疏 score 计算、attention sink 融合

## Implementation Constraints — Only Triton

This challenge is **Triton-only**. The scope was clarified as follows:

**Q (a):** Must all GEMMs / heavy compute be implemented as `@triton.jit` kernels?
Or is `torch.nn.functional.linear` (vendor BLAS) still allowed?

**A:** All computation must be implemented via `@triton.jit`. Torch ops may be
used **only for data preparation in the wrapper**. **Vendor BLAS is not allowed.**

**Q (b):** Are PyTorch ops for routing setup still allowed?
e.g., `softmax`, `topk`, `argsort`, `bincount`, `index_add`, `index_select`.

**A:** **No.** Torch ops may only be used for memory operations such as
`torch.ones`, `torch.zeros`, allocation, reshape/view, etc. Any numerical
transformation (softmax, topk, argsort, bincount, index_add, index_select,
linear, sigmoid, gather-with-compute, etc.) must live inside a `@triton.jit` kernel.

**Practical implications for this track:**
- The QK matmul, scaled scores, sink concat, softmax, and AV matmul all run
  inside Triton (one fused kernel ideally).
- KV gather via `topk_idxs` is performed inside the Triton kernel using
  pointer arithmetic — no `torch.gather` in the hot path.
- The wrapper may allocate the output tensor and validate shapes, nothing more.

## 模型实际参数（来自真实推理 profiling）

- Head Dim：512，数据类型：bf16
- scale：0.04419417382415922
- 函数签名：`sparse_attn(q [b,m,h,d], kv [b,kv_len,d], attn_sink [h], topk_idxs [b,m,topk], scale)`

## 测试 Workload（代表性 config）

| 类别    | batch | seq_len | kv_len | topk | heads |
|---------|-------|---------|--------|------|-------|
| Prefill | 64    | 1024    | 1024   | 128  | 16    |
| Prefill | 64    | 1024    | 1280   | 384  | 16    |
| Prefill | 16    | 4096    | 4096   | 128  | 16    |
| Prefill | 1     | 16384   | 16384  | 128  | 16    |
| Prefill | 1     | 16384   | 20480  | 640  | 16    |
| Decode  | 64    | 1       | 128    | 128  | 16    |
| Decode  | 64    | 1       | 400    | 392  | 16    |
| Decode  | 64    | 1       | 640    | 640  | 16    |
| Decode  | 16    | 1       | 1408   | 640  | 16    |
| Decode  | 1     | 1       | 4480   | 640  | 16    |

## 参考实现

```python
import torch
import torch.nn.functional as F


@register("sparse_attn", False)
def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """
    Reference sparse attention in pure PyTorch.

    Args:
        q:          [b, m, h, d], bf16
        kv:         [b, kv_len, d], bf16
        attn_sink:  [h], fp32
        topk_idxs:  [b, m, topk], int32
        scale:      float

    Returns:
        o: [b, m, h, d], same dtype as q
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    flat_idx = topk_idxs.long().reshape(b, m * topk)
    gathered_kv = torch.gather(kv, 1, flat_idx.unsqueeze(-1).expand(-1, -1, d)).reshape(b, m, topk, d)

    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * scale

    sink = attn_sink[None, None, :, None].expand(b, m, h, 1)
    scores_with_sink = torch.cat([scores, sink], dim=-1)
    attn = torch.softmax(scores_with_sink, dim=-1)

    attn_kv = attn[:, :, :, :-1]
    o = torch.einsum("bmht,bmtd->bmhd", attn_kv, gathered_kv.float())
    return o.to(q.dtype)
```

## 算子入参说明

```python
def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """
    Args:
        q:          Tensor [b, m, h, d]     — Query，bf16
                    b=batch, m=seq_len, h=heads, d=512
        kv:         Tensor [b, kv_len, d]   — Key-Value（共享），bf16
                    kv_len 为完整 KV cache 长度
        attn_sink:  Tensor [h]              — 每个 head 的 attention sink 偏置，fp32
        topk_idxs:  Tensor [b, m, topk]     — 每个 query token 选中的 KV 索引，int32
                    topk 为每个 token 关注的 KV 数量
        scale:      float                   — attention scale 因子

    Returns:
        o:          Tensor [b, m, h, d]     — 输出，bf16
    """
```

## 计算逻辑

对每个 query token，根据 `topk_idxs` gather 出 KV 子集，计算 `Q @ K^T * scale`，拼接 `attn_sink` 后 softmax，再乘 V 得到输出。
