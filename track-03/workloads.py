"""Eight benchmark workloads from track-03/requirements.md."""

WORKLOADS = [
    {"label": "decode",  "M": 1,    "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "decode",  "M": 64,   "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "decode",  "M": 1,    "K": 2048, "N": 1024, "E": 64, "topk": 6},
    {"label": "decode",  "M": 64,   "K": 2048, "N": 1024, "E": 64, "topk": 6},
    {"label": "prefill", "M": 512,  "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "prefill", "M": 4096, "K": 2048, "N": 1024, "E": 8,  "topk": 2},
    {"label": "prefill", "M": 512,  "K": 2048, "N": 1024, "E": 64, "topk": 6},
    {"label": "prefill", "M": 4096, "K": 2048, "N": 1024, "E": 64, "topk": 6},
]
