"""
Workload configurations for track-02 (DeepSeek mHC).

Source: requirements.md "测试 Workload (来自真实 serving log)"
Each scenario has a decode shape (high frequency) and a prefill shape (lower frequency).
The 1k-1-64 scenario is the prefill-heavy one (67% prefill).
"""
import torch

DTYPE = torch.float32       # mixes, hc_scale, hc_base are all fp32
HC_MULT = 4
SINKHORN_ITERS = 20
EPS = 1e-6
MIXES_DIM = (2 + HC_MULT) * HC_MULT   # 24


# (name, decode_shape, decode_freq, prefill_shape, prefill_freq)
WORKLOADS = [
    ("16k-1k-1",  (1,  1, MIXES_DIM), 0.998, (1,  16384, MIXES_DIM), 0.002),
    ("4k-1k-16",  (16, 1, MIXES_DIM), 0.998, (16, 4096,  MIXES_DIM), 0.002),
    ("1k-1k-64",  (64, 1, MIXES_DIM), 0.998, (64, 1024,  MIXES_DIM), 0.002),
    ("1k-1-64",   (64, 1, MIXES_DIM), 0.333, (64, 1024,  MIXES_DIM), 0.672),
]


def make_inputs(shape, device="cuda", seed=0):
    """Generate random inputs matching the kernel signature."""
    g = torch.Generator(device=device).manual_seed(seed)
    mixes    = torch.randn(*shape, dtype=DTYPE, device=device, generator=g)
    hc_scale = torch.randn(3,           dtype=DTYPE, device=device, generator=g)
    hc_base  = torch.randn(MIXES_DIM,   dtype=DTYPE, device=device, generator=g)
    return mixes, hc_scale, hc_base


def all_shapes():
    """Yield (label, shape) for every (decode, prefill) combo."""
    for name, decode_shape, _, prefill_shape, _ in WORKLOADS:
        yield (f"{name}-decode",  decode_shape)
        yield (f"{name}-prefill", prefill_shape)
