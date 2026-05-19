"""Standalone baseline KV caches.

Each baseline implements its own attention path and HF dispatcher integration —
mirroring KVCascade's pattern but with the cache's specific eviction/quantization
policy. Use as side-by-side baselines in the iso-byte head-to-head experiment.

Eviction-only baselines (single fp buffer, single eviction policy):
  - H2OCache          — cumulative-attention scoring, online decode eviction
  - SnapKVCache       — prefill obs-window scoring, frozen post-prefill
  - StreamingLLMCache — first-N attention sinks + last-K recency FIFO
  - AdaSnapKVCache    — SnapKV scoring with per-head adaptive heavy-hitter budget

Quantization-only baseline (no eviction, every token kept quantized):
  - KIVICache         — per-channel K + per-token V asymmetric INT-k, residual fp ring

Composition baseline (eviction + quant, no demote-on-loss):
  - SnapKVTurboCache  — SnapKV selection + PolarQuant quantization of heavy hitters
"""

from .ada_snapkv import AdaSnapKVCache, install_ada_snapkv
from .h2o import H2OCache, install_h2o
from .kivi import KIVICache, install_kivi
from .snapkv import SnapKVCache, install_snapkv
from .snapkv_turbo import SnapKVTurboCache, install_snapkv_turbo
from .streamingllm import StreamingLLMCache, install_streamingllm

__all__ = [
    "H2OCache",          "install_h2o",
    "SnapKVCache",       "install_snapkv",
    "StreamingLLMCache", "install_streamingllm",
    "AdaSnapKVCache",    "install_ada_snapkv",
    "KIVICache",         "install_kivi",
    "SnapKVTurboCache",  "install_snapkv_turbo",
]
