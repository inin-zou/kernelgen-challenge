"""
Workload configurations for track-01 (Sparse Attention).
Source: requirements.md "测试 Workload (代表性 config)"
"""
import torch

DTYPE = torch.bfloat16
SINK_DTYPE = torch.float32
HEAD_DIM = 512
NUM_HEADS = 16
SCALE = 0.04419417382415922


# (label, batch, seq_len, kv_len, topk)
WORKLOADS = [
    ("prefill-64x1024-1024-128",  "Prefill", 64, 1024,  1024,  128),
    ("prefill-64x1024-1280-384",  "Prefill", 64, 1024,  1280,  384),
    ("prefill-16x4096-4096-128",  "Prefill", 16, 4096,  4096,  128),
    ("prefill-1x16384-16384-128", "Prefill",  1, 16384, 16384, 128),
    ("prefill-1x16384-20480-640", "Prefill",  1, 16384, 20480, 640),
    ("decode-64x1-128-128",       "Decode",  64,     1,   128, 128),
    ("decode-64x1-400-392",       "Decode",  64,     1,   400, 392),
    ("decode-64x1-640-640",       "Decode",  64,     1,   640, 640),
    ("decode-16x1-1408-640",      "Decode",  16,     1,  1408, 640),
    ("decode-1x1-4480-640",       "Decode",   1,     1,  4480, 640),
]


def make_inputs(b, m, kv_len, topk, h=NUM_HEADS, d=HEAD_DIM, device="cuda", seed=0):
    """Generate random inputs matching the kernel signature."""
    g = torch.Generator(device=device).manual_seed(seed)
    q   = torch.randn(b, m, h, d, dtype=DTYPE,      device=device, generator=g)
    kv  = torch.randn(b, kv_len, d, dtype=DTYPE,    device=device, generator=g)
    sink = torch.randn(h,           dtype=SINK_DTYPE, device=device, generator=g)
    # topk_idxs: random valid positions in [0, kv_len)
    idx = torch.randint(0, kv_len, (b, m, topk), dtype=torch.int32, device=device)
    return q, kv, sink, idx, SCALE


def all_workloads():
    """Yield (label, b, m, kv_len, topk) for every workload."""
    for label, _, b, m, kv_len, topk in WORKLOADS:
        yield label, b, m, kv_len, topk
