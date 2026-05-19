"""Smoke tests for the six baseline KV caches.

Verifies:
  1. Each cache can be constructed and produces finite attention outputs on a
     synthetic Q/K/V batch.
  2. H2O compresses to `cache_size` after prefill via top-K cumulative attention.
  3. StreamingLLM keeps the first `sink_count` tokens forever after prefill;
     non-sink older tokens fall out as the recency window slides past them.
  4. SnapKV compresses after prefill: total valid slots == fp_capacity + window.
  5. Ada-SnapKV redistributes per-head budgets while preserving the layer total.
  6. KIVI flushes the residual buffer when full and keeps all tokens.
  7. SnapKV+TurboQuant composes prefill selection with PolarQuant simulation.
  8. All six compose with GQA (num_kv_heads < num_heads).
"""

import math
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "src"))

import torch

from baselines import (H2OCache, StreamingLLMCache, SnapKVCache,
                       AdaSnapKVCache, KIVICache, SnapKVTurboCache)


def _ref_attn(q, k, v):
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(q.shape[0], q.shape[1], k.shape[-2], k.shape[-1])
        v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(q.shape[0], q.shape[1], v.shape[-2], v.shape[-1])
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    a = torch.softmax(s, dim=-1)
    return a @ v


def test_streamingllm_runs():
    torch.manual_seed(0)
    L, B, H_q, H_kv, D = 2, 1, 8, 4, 32
    cache = StreamingLLMCache(num_layers=L, batch_size=B, num_heads=H_q,
                               num_kv_heads=H_kv, head_dim=D,
                               sink_count=4, recent_count=16,
                               device=torch.device("cpu"), dtype=torch.float32)
    # Prefill 20 tokens — first 4 become sinks, last 16 fill recent ring.
    T_pre = 20
    Q = torch.randn(B, H_q, T_pre, D)
    K = torch.randn(B, H_kv, T_pre, D)
    V = torch.randn(B, H_kv, T_pre, D)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (B, H_q, T_pre, D)
    assert torch.isfinite(out).all()
    # Sinks filled fully on layer 0.
    assert cache._sinks_filled[0] == 4
    # Pos for sink slots = first 4 sequence positions.
    sink_pos = cache.pos[0, 0, 0, :4].tolist()
    assert sink_pos == [0, 1, 2, 3], sink_pos
    print(f"  PASS: streamingllm prefill, sinks at pos {sink_pos}, "
          f"valid count = {(cache.pos[0] >= 0).sum().item() // (B * H_kv)}")

    # Decode 10 more tokens.
    Q1 = torch.randn(B, H_q, 1, D)
    K1 = torch.randn(B, H_kv, 1, D)
    V1 = torch.randn(B, H_kv, 1, D)
    for k in range(10):
        out = cache.attention(0, Q1, k_new=K1, v_new=V1)
        assert torch.isfinite(out).all()
    # After 10 decode steps, sinks unchanged. Recent ring should contain positions
    # in [20 + 10 - 16, 20 + 10) = [14, 30) effectively, modulo the FIFO mechanism.
    sink_pos_after = cache.pos[0, 0, 0, :4].tolist()
    assert sink_pos_after == [0, 1, 2, 3], f"sinks moved! {sink_pos_after}"
    # Recent ring's largest pos should be 29 (the last decode token).
    recent_max = cache.pos[0, 0, 0, 4:].max().item()
    assert recent_max == 29, f"recent max pos {recent_max} (expected 29)"
    print(f"  PASS: streamingllm post-decode sinks still {sink_pos_after}, "
          f"recent max pos = {recent_max}")


def test_snapkv_runs():
    torch.manual_seed(1)
    L, B, H_q, H_kv, D = 2, 1, 8, 4, 32
    cache = SnapKVCache(num_layers=L, batch_size=B, num_heads=H_q,
                        num_kv_heads=H_kv, head_dim=D,
                        fp_capacity=16, window=8, pool_kernel=5,
                        device=torch.device("cpu"), dtype=torch.float32)
    # Prefill 64 tokens — should compress to 16 heavy hitters + 8 recent window.
    T_pre = 64
    Q = torch.randn(B, H_q, T_pre, D)
    K = torch.randn(B, H_kv, T_pre, D)
    V = torch.randn(B, H_kv, T_pre, D)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (B, H_q, T_pre, D)
    assert torch.isfinite(out).all()
    assert cache._compressed[0], "SnapKV did not compress after prefill"

    # Verify post-compress state.
    valid_count = (cache.pos[0, 0, 0] >= 0).sum().item()
    expected = 16 + 8                                  # fp_capacity + window
    assert valid_count == expected, f"expected {expected} valid slots, got {valid_count}"

    # Window slots should hold the last 8 prefill positions [56..63].
    window_pos = cache.pos[0, 0, 0, 16:24].tolist()
    assert sorted(window_pos) == list(range(56, 64)), f"window has {sorted(window_pos)}"
    print(f"  PASS: snapkv compressed: {valid_count} valid slots, "
          f"window pos {window_pos[:3]}..{window_pos[-1:]}")

    # Decode 5 more tokens; window FIFO should slide.
    Q1 = torch.randn(B, H_q, 1, D)
    K1 = torch.randn(B, H_kv, 1, D)
    V1 = torch.randn(B, H_kv, 1, D)
    for k in range(5):
        out = cache.attention(0, Q1, k_new=K1, v_new=V1)
        assert torch.isfinite(out).all()
    # Heavy-hitter slots untouched; window now contains 5 new tokens + 3 old.
    win_pos_after = sorted(cache.pos[0, 0, 0, 16:24].tolist())
    assert win_pos_after[-1] == 68, f"window max = {win_pos_after[-1]} (expected 68)"
    print(f"  PASS: snapkv decode FIFO, window pos now {win_pos_after}")


def test_streamingllm_gqa():
    """GQA: num_heads=16, num_kv_heads=4 → n_rep=4. Must run without shape errors."""
    torch.manual_seed(2)
    cache = StreamingLLMCache(num_layers=1, batch_size=1, num_heads=16,
                               num_kv_heads=4, head_dim=64,
                               sink_count=4, recent_count=32,
                               device=torch.device("cpu"), dtype=torch.float32)
    Q = torch.randn(1, 16, 50, 64)
    K = torch.randn(1, 4, 50, 64)
    V = torch.randn(1, 4, 50, 64)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (1, 16, 50, 64)
    assert torch.isfinite(out).all()
    print("  PASS: streamingllm composes with GQA (n_rep=4)")


def test_snapkv_gqa():
    torch.manual_seed(3)
    cache = SnapKVCache(num_layers=1, batch_size=1, num_heads=16,
                        num_kv_heads=4, head_dim=64,
                        fp_capacity=32, window=16, pool_kernel=7,
                        device=torch.device("cpu"), dtype=torch.float32)
    Q = torch.randn(1, 16, 128, 64)
    K = torch.randn(1, 4, 128, 64)
    V = torch.randn(1, 4, 128, 64)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (1, 16, 128, 64)
    assert torch.isfinite(out).all()
    assert cache._compressed[0]
    print("  PASS: snapkv composes with GQA (n_rep=4)")


def test_h2o_runs():
    torch.manual_seed(4)
    L, B, H_q, H_kv, D = 2, 1, 8, 4, 32
    cache = H2OCache(num_layers=L, batch_size=B, num_heads=H_q,
                     num_kv_heads=H_kv, head_dim=D,
                     cache_size=24,
                     device=torch.device("cpu"), dtype=torch.float32)
    # Prefill 64 tokens — should compress to 24 by cumulative attention.
    T_pre = 64
    Q = torch.randn(B, H_q, T_pre, D)
    K = torch.randn(B, H_kv, T_pre, D)
    V = torch.randn(B, H_kv, T_pre, D)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (B, H_q, T_pre, D)
    assert torch.isfinite(out).all()
    assert cache._compressed[0], "H2O did not compress after prefill"
    valid_count = (cache.pos[0, 0, 0] >= 0).sum().item()
    assert valid_count == 24, f"expected 24 valid slots, got {valid_count}"
    print(f"  PASS: h2o prefill compressed to {valid_count} slots")

    # Decode 10 more tokens with online eviction.
    Q1 = torch.randn(B, H_q, 1, D)
    K1 = torch.randn(B, H_kv, 1, D)
    V1 = torch.randn(B, H_kv, 1, D)
    for k in range(10):
        out = cache.attention(0, Q1, k_new=K1, v_new=V1)
        assert torch.isfinite(out).all()
    # Cache size invariant.
    valid_after = (cache.pos[0, 0, 0] >= 0).sum().item()
    assert valid_after == 24, f"expected 24 valid slots after decode, got {valid_after}"
    print(f"  PASS: h2o decode online eviction, {valid_after} slots maintained")


def test_h2o_gqa():
    torch.manual_seed(5)
    cache = H2OCache(num_layers=1, batch_size=1, num_heads=16,
                     num_kv_heads=4, head_dim=64,
                     cache_size=48,
                     device=torch.device("cpu"), dtype=torch.float32)
    Q = torch.randn(1, 16, 128, 64)
    K = torch.randn(1, 4, 128, 64)
    V = torch.randn(1, 4, 128, 64)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (1, 16, 128, 64)
    assert torch.isfinite(out).all()
    assert cache._compressed[0]
    print("  PASS: h2o composes with GQA (n_rep=4)")


def test_ada_snapkv_runs():
    torch.manual_seed(6)
    L, B, H_q, H_kv, D = 2, 1, 8, 4, 32
    fp_cap = 16
    window = 8
    cache = AdaSnapKVCache(num_layers=L, batch_size=B, num_heads=H_q,
                            num_kv_heads=H_kv, head_dim=D,
                            fp_capacity=fp_cap, window=window,
                            pool_kernel=5, safety_factor=2,
                            device=torch.device("cpu"), dtype=torch.float32)
    # Prefill 64 tokens.
    T_pre = 64
    Q = torch.randn(B, H_q, T_pre, D)
    K = torch.randn(B, H_kv, T_pre, D)
    V = torch.randn(B, H_kv, T_pre, D)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (B, H_q, T_pre, D)
    assert torch.isfinite(out).all()
    assert cache._compressed[0]
    # Layer total heavy-cap should equal H_kv * fp_cap (== iso-byte to SnapKV).
    eff = cache.eff_heavy[0, 0]
    layer_total = int(eff.sum().item())
    expected = min(H_kv * fp_cap, H_kv * (T_pre - window))
    assert layer_total == expected, f"layer heavy total {layer_total} != expected {expected}"
    # Per-head must be within [0, safety_factor*fp_cap].
    max_h = 2 * fp_cap
    assert (eff <= max_h).all(), f"some head exceeded safety cap: {eff.tolist()}"
    print(f"  PASS: ada-snapkv prefill — per-head heavy = {eff.tolist()} "
          f"(layer sum = {layer_total}, max = {int(eff.max().item())})")

    # Decode 5 tokens.
    Q1 = torch.randn(B, H_q, 1, D)
    K1 = torch.randn(B, H_kv, 1, D)
    V1 = torch.randn(B, H_kv, 1, D)
    for k in range(5):
        out = cache.attention(0, Q1, k_new=K1, v_new=V1)
        assert torch.isfinite(out).all()
    print("  PASS: ada-snapkv decode window FIFO produces finite output")


def test_ada_snapkv_gqa():
    torch.manual_seed(7)
    cache = AdaSnapKVCache(num_layers=1, batch_size=1, num_heads=16,
                            num_kv_heads=4, head_dim=64,
                            fp_capacity=32, window=16, pool_kernel=7,
                            safety_factor=2,
                            device=torch.device("cpu"), dtype=torch.float32)
    Q = torch.randn(1, 16, 128, 64)
    K = torch.randn(1, 4, 128, 64)
    V = torch.randn(1, 4, 128, 64)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (1, 16, 128, 64)
    assert torch.isfinite(out).all()
    assert cache._compressed[0]
    # Layer heavy sum invariant.
    assert int(cache.eff_heavy[0, 0].sum().item()) == 4 * 32
    print("  PASS: ada-snapkv composes with GQA (n_rep=4)")


def test_kivi_runs():
    torch.manual_seed(8)
    L, B, H_q, H_kv, D = 2, 1, 8, 4, 32
    T_max, R = 256, 32
    cache = KIVICache(num_layers=L, batch_size=B, num_heads=H_q,
                      num_kv_heads=H_kv, head_dim=D,
                      max_seq_len=T_max, bits=2, residual_length=R,
                      device=torch.device("cpu"), dtype=torch.float16)
    # Prefill 100 tokens — should flush 96/32 = 3 chunks, leave 4 in residual.
    T_pre = 100
    Q = torch.randn(B, H_q, T_pre, D, dtype=torch.float32)
    K = torch.randn(B, H_kv, T_pre, D, dtype=torch.float16)
    V = torch.randn(B, H_kv, T_pre, D, dtype=torch.float16)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (B, H_q, T_pre, D)
    assert torch.isfinite(out).all()
    # After ingest: main = 96 tokens (3 full chunks), residual = 4.
    assert cache.main_count[0] == 96, f"main_count={cache.main_count[0]}, expected 96"
    assert cache.resid_count[0] == 4, f"resid_count={cache.resid_count[0]}, expected 4"
    print(f"  PASS: kivi prefill — main={cache.main_count[0]}, residual={cache.resid_count[0]}")

    # Decode 30 more tokens; should trigger another flush after residual hits 32.
    Q1 = torch.randn(B, H_q, 1, D, dtype=torch.float32)
    K1 = torch.randn(B, H_kv, 1, D, dtype=torch.float16)
    V1 = torch.randn(B, H_kv, 1, D, dtype=torch.float16)
    for k in range(30):
        out = cache.attention(0, Q1, k_new=K1, v_new=V1)
        assert torch.isfinite(out).all()
    # After 30 more: prev_main=96 + 1 flush of 32 = 128. residual = 4+30-32 = 2.
    assert cache.main_count[0] == 128, f"main_count={cache.main_count[0]}"
    assert cache.resid_count[0] == 2, f"resid_count={cache.resid_count[0]}"
    print(f"  PASS: kivi decode flush — main={cache.main_count[0]}, residual={cache.resid_count[0]}")


def test_kivi_gqa():
    torch.manual_seed(9)
    cache = KIVICache(num_layers=1, batch_size=1, num_heads=16,
                      num_kv_heads=4, head_dim=64,
                      max_seq_len=256, bits=2, residual_length=32,
                      device=torch.device("cpu"), dtype=torch.float16)
    Q = torch.randn(1, 16, 128, 64, dtype=torch.float32)
    K = torch.randn(1, 4, 128, 64, dtype=torch.float16)
    V = torch.randn(1, 4, 128, 64, dtype=torch.float16)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (1, 16, 128, 64)
    assert torch.isfinite(out).all()
    print("  PASS: kivi composes with GQA (n_rep=4)")


def test_snapkv_turbo_runs():
    torch.manual_seed(10)
    L, B, H_q, H_kv, D = 2, 1, 8, 4, 32
    cache = SnapKVTurboCache(num_layers=L, batch_size=B, num_heads=H_q,
                              num_kv_heads=H_kv, head_dim=D,
                              heavy_capacity=24, window=8,
                              k_bits=6, v_bits=2, pool_kernel=5,
                              seed=11,
                              device=torch.device("cpu"), dtype=torch.float32)
    # Prefill 64.
    T_pre = 64
    Q = torch.randn(B, H_q, T_pre, D)
    K = torch.randn(B, H_kv, T_pre, D)
    V = torch.randn(B, H_kv, T_pre, D)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (B, H_q, T_pre, D)
    assert torch.isfinite(out).all()
    assert cache._compressed[0]
    valid_count = (cache.pos[0, 0, 0] >= 0).sum().item()
    expected = 24 + 8
    assert valid_count == expected, f"expected {expected} valid slots, got {valid_count}"
    print(f"  PASS: snapkv_turbo prefill — {valid_count} valid slots, heavy quantized")

    # Decode 5 tokens.
    Q1 = torch.randn(B, H_q, 1, D)
    K1 = torch.randn(B, H_kv, 1, D)
    V1 = torch.randn(B, H_kv, 1, D)
    for k in range(5):
        out = cache.attention(0, Q1, k_new=K1, v_new=V1)
        assert torch.isfinite(out).all()
    print("  PASS: snapkv_turbo decode FIFO produces finite output")


def test_snapkv_turbo_gqa():
    torch.manual_seed(12)
    cache = SnapKVTurboCache(num_layers=1, batch_size=1, num_heads=16,
                              num_kv_heads=4, head_dim=64,
                              heavy_capacity=48, window=16,
                              k_bits=4, v_bits=2, pool_kernel=7,
                              seed=13,
                              device=torch.device("cpu"), dtype=torch.float32)
    Q = torch.randn(1, 16, 128, 64)
    K = torch.randn(1, 4, 128, 64)
    V = torch.randn(1, 4, 128, 64)
    out = cache.attention(0, Q, k_new=K, v_new=V)
    assert out.shape == (1, 16, 128, 64)
    assert torch.isfinite(out).all()
    assert cache._compressed[0]
    print("  PASS: snapkv_turbo composes with GQA (n_rep=4)")


if __name__ == "__main__":
    print("test_h2o_runs:")
    test_h2o_runs()
    print("test_streamingllm_runs:")
    test_streamingllm_runs()
    print("test_snapkv_runs:")
    test_snapkv_runs()
    print("test_ada_snapkv_runs:")
    test_ada_snapkv_runs()
    print("test_kivi_runs:")
    test_kivi_runs()
    print("test_snapkv_turbo_runs:")
    test_snapkv_turbo_runs()
    print("test_h2o_gqa:")
    test_h2o_gqa()
    print("test_streamingllm_gqa:")
    test_streamingllm_gqa()
    print("test_snapkv_gqa:")
    test_snapkv_gqa()
    print("test_ada_snapkv_gqa:")
    test_ada_snapkv_gqa()
    print("test_kivi_gqa:")
    test_kivi_gqa()
    print("test_snapkv_turbo_gqa:")
    test_snapkv_turbo_gqa()
    print("\nAll baseline cache checks passed.")
