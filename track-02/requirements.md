# Task 02: DeepSeek mHC（Manifold-Constrained Hyper-Connections）

实现多流混合 + 双随机矩阵投影的融合算子，一次访存完成多路残差读取与加权组合。

**核心难点**：寄存器压力控制、多流访存的 Pipeline Hiding

## 提交要求 (Submission)

- **截止时间**：May 6, 2026 22:00 (Beijing) / 16:00 (Paris)
- **文件格式**：单个 `.py` 文件，最大 10MB，UTF-8 编码
- **评测环境**：Python 3 + Triton **3.5**（注意版本兼容性）
- **函数名 / 入参 / 返回值**必须严格匹配 README 中的签名（见下方"算子入参说明"）
- **提交频率**：两次提交间隔至少 2 分钟

提交的 `.py` 文件需自包含：所有 `@triton.jit` kernel + 入口 wrapper 函数 `hc_split_sinkhorn(...)`，可以直接被评测脚本 `import` 调用。

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
linear, sigmoid, etc.) must live inside a `@triton.jit` kernel.

**Practical implications for this track:**
- The sigmoid / softmax / Sinkhorn iterations all run inside Triton.
- The `* hc_scale + hc_base` bias-and-scale is fused into the kernel.
- The wrapper may allocate output tensors and reshape inputs, nothing more.
- `F.linear` (the upstream `mixes = x @ hc_fn`) is out of scope here — but if
  you also implement `hc_pre` / `hc_post`, those linears must be Triton GEMMs.

## 模型实际参数（来自 FlagOS 推理代码）

- `hc_mult = 4`（4 条流），`dim = 4096`
- `sinkhorn_iters = 20`，`eps = 1e-6`
- 数据类型：计算在 fp32，输入输出 bf16
- 每层调用两次（attn 前后 + ffn 前后）

## 算子调用链

- `hc_pre(x [b,s,4,4096])`:
  - flatten → RMSNorm-style rsqrt → `F.linear(x, hc_fn [24, 16384])` → `hc_split_sinkhorn`
  - 输出 `pre [b,s,4]` 加权求和 → `y [b,s,4096]`
- 经过 attn / ffn
- `hc_post(x [b,s,4096], residual [b,s,4,4096], post [b,s,4], comb [b,s,4,4])`:
  - `y = post * x + comb @ residual` → `[b,s,4,4096]`

## 核心 kernel `hc_split_sinkhorn`

输入：`mixes [n, 24]`（n = b*s），`hc_scale [3]`，`hc_base [24]`

拆分为：

- `pre [n, 4]`：sigmoid 激活
- `post [n, 4]`：2 * sigmoid 激活
- `comb [n, 4, 4]`：softmax + 20 轮 Sinkhorn 迭代 → 双随机矩阵

输出：`pre`、`post`、`comb`

## 测试 Workload（来自真实 serving log）

4 个场景，文件名格式 `{context_len}-{prefill_tokens}-{batch}`：

| 场景        | mixes shape (Decode) | 调用占比 | mixes shape (Prefill) | 调用占比 |
|-------------|----------------------|----------|-----------------------|----------|
| 16k-1k-1    | [1, 1, 24]           | 99.8%    | [1, 16384, 24]        | 0.2%     |
| 4k-1k-16    | [16, 1, 24]          | 99.8%    | [16, 4096, 24]        | 0.2%     |
| 1k-1k-64    | [64, 1, 24]          | 99.8%    | [64, 1024, 24]        | 0.2%     |
| 1k-1-64     | [64, 1, 24]          | 33.3%    | [64, 1024, 24]        | 67.2%    |

## 参考实现

```python
import torch
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.shape
    hc = hc_mult

    pre_raw = mixes[..., :hc]
    post_raw = mixes[..., hc:2 * hc]
    comb_raw = mixes[..., 2 * hc:].reshape(b, s, hc, hc)

    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc:2 * hc])
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)

    comb = F.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb
```

## 算子入参说明

```python
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """
    Args:
        mixes:          Tensor [b, s, (2+hc_mult)*hc_mult]  — 线性投影后的混合系数，fp32
        hc_scale:       Tensor [3]                           — pre/post/comb 三组的缩放因子，fp32
        hc_base:        Tensor [(2+hc_mult)*hc_mult]         — pre/post/comb 三组的偏置，fp32
        hc_mult:        int                                  — 流数（默认 4）
        sinkhorn_iters: int                                  — Sinkhorn 迭代次数（默认 20）
        eps:            float                                — 数值稳定常数（默认 1e-6）

    Returns:
        pre:    Tensor [b, s, hc_mult]              — 输入流加权系数，sigmoid 激活
        post:   Tensor [b, s, hc_mult]              — 输出流加权系数，2*sigmoid 激活
        comb:   Tensor [b, s, hc_mult, hc_mult]     — 流间组合矩阵，双随机矩阵（Sinkhorn 归一化）
    """
```

## 计算逻辑

将 `mixes` 拆分为 `pre`/`post`/`comb` 三部分，各自乘以 `hc_scale` 加 `hc_base` 后：`pre` 经 sigmoid + eps，`post` 经 2*sigmoid，`comb` 经 softmax 再做 `sinkhorn_iters` 轮行列交替归一化，得到近似双随机矩阵。
