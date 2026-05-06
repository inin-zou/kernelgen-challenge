# Task 03: Fused MoE（融合混合专家）

实现 Mixture-of-Experts 融合 Kernel，将 softmax routing、top-k 选择、gate-up GEMM、SiLU 门控激活、down GEMM 和加权归约融合为单次调用。

**核心难点**：token-to-expert 分发效率、多专家 GEMM 并行、SiLU 门控融合、加权归约

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
linear, silu, sigmoid, etc.) must live inside a `@triton.jit` kernel.

**Practical implications for this track:**
- Routing softmax, top-k selection, and (optional) renormalize all run inside
  Triton kernels — `score.softmax(...)` and `.topk(...)` from PyTorch are not
  allowed.
- The gate+up GEMM (`w1`), SiLU gating, the down GEMM (`w2`), and the
  weighted reduction across experts must all be Triton kernels (vendor BLAS
  via `F.linear` / `torch.matmul` is forbidden).
- The wrapper may allocate output / scratch tensors and reshape inputs,
  nothing more.

## 模型实际参数（来自 Mixtral / DeepSeek-V2 推理）

- `hidden_size (K)`：2048，`intermediate_size (N)`：1024
- `num_experts (E)`：8 / 64，`topk`：2 / 6
- 数据类型：bf16
- 激活函数：SiLU gating（`silu(gate) * up`），`w1` 包含 gate+up 两个投影
- 函数签名：`fused_moe(hidden_states [M,K], w1 [E,2*N,K], w2 [E,K,N], score [M,E], topk)`

## 测试 Workload（代表性 config）

| 类别    | M    | K    | N    | E  | topk |
|---------|------|------|------|----|------|
| Decode  | 1    | 2048 | 1024 | 8  | 2    |
| Decode  | 64   | 2048 | 1024 | 8  | 2    |
| Decode  | 1    | 2048 | 1024 | 64 | 6    |
| Decode  | 64   | 2048 | 1024 | 64 | 6    |
| Prefill | 512  | 2048 | 1024 | 8  | 2    |
| Prefill | 4096 | 2048 | 1024 | 8  | 2    |
| Prefill | 512  | 2048 | 1024 | 64 | 6    |
| Prefill | 4096 | 2048 | 1024 | 64 | 6    |

## 参考实现与测试

```python
import torch
import torch.nn.functional as F


def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[-1]
    dtype = hidden_states.dtype

    topk_weights = score.softmax(dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    final_out = torch.zeros(M, K, device=hidden_states.device, dtype=dtype)

    for expert_idx in range(E):
        mask = (topk_ids == expert_idx)
        token_weights = (topk_weights * mask).sum(dim=-1, keepdim=True)

        x = F.linear(hidden_states, w1[expert_idx])
        gate = F.silu(x[:, :N])
        x = x[:, N:] * gate
        x = F.linear(x, w2[expert_idx])

        final_out += x * token_weights

    return final_out
```

## 算子入参说明

```python
def fused_moe(hidden_states, w1, w2, score, topk, renormalize=False):
    """
    Args:
        hidden_states:  Tensor [M, K]       — 输入 token 特征，bf16
        w1:             Tensor [E, 2*N, K]  — gate+up 融合权重，bf16
        w2:             Tensor [E, K, N]    — down 投影权重，bf16
        score:          Tensor [M, E]       — router logits（路由分数），bf16
        topk:           int                 — 每个 token 选择的专家数
        renormalize:    bool                — 是否对 topk 权重归一化（默认 False）

    Returns:
        output:         Tensor [M, K]       — 输出 token 特征，bf16
    """
```

## 计算逻辑

对 `score` 做 softmax 后选 top-k 个专家，对每个专家：将 token 乘以 `w1[i]` 得到 gate+up 两路输出，gate 路经 SiLU 激活后与 up 路逐元素相乘，再乘以 `w2[i]` 得到 down 投影结果，最后按 routing 权重加权求和得到最终输出。
