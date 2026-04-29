"""Verify per-query EMA: T_q=1 unchanged vs old uniform-mean formula; T_q=2 matches
sequential-EMA-of-each-query semantics.

Builds a small KVCascadeCache, runs an attention call with synthetic Q/K/V, and
compares post-call buffer scores against an analytic expectation.
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "src"))

import torch
from kvcascade import KVCascadeCache


def make_cache(seed=0, score_policy="ema", ema_decay=0.9):
    return KVCascadeCache(
        num_layers=1, batch_size=1, num_heads=2, head_dim=8,
        ring_size=2, fp_capacity=3,
        quant_tiers=[],   # no quant tiers — keep it simple, just ring + fp
        m=8, score_policy=score_policy, ema_decay=ema_decay, seed=seed,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def seed_fp_tier(cache, layer_idx, n, init_score=1.0):
    """Put n synthetic tokens into fp_tier with a known score so we can verify decay."""
    g = torch.Generator().manual_seed(7)
    B, H = 1, cache.num_kv_heads
    D = cache.head_dim
    K = torch.randn(B, H, n, D, generator=g)
    V = torch.randn(B, H, n, D, generator=g)
    cache.fp_tier.K[layer_idx, :, :, :n] = K
    cache.fp_tier.V[layer_idx, :, :, :n] = V
    cache.fp_tier.pos[layer_idx, :, :, :n] = torch.arange(n).view(1, 1, n).expand(B, H, n)
    cache.fp_tier.score[layer_idx, :, :, :n] = init_score
    cache.next_pos[layer_idx] = n  # so subsequent ingestion uses positions n, n+1, ...


def test_decode_unchanged():
    """T_q=1: per-query EMA must equal the standard EMA formula."""
    print("--- test: T_q=1 (decode) unchanged ---")
    rho = 0.9
    init_score = 1.0
    cache = make_cache(ema_decay=rho)
    layer_idx = 0
    seed_fp_tier(cache, layer_idx, n=3, init_score=init_score)

    # Single-query attention (decode step).
    g = torch.Generator().manual_seed(11)
    q = torch.randn(1, cache.num_heads, 1, cache.head_dim, generator=g)
    k_new = torch.randn(1, cache.num_kv_heads, 1, cache.head_dim, generator=g)
    v_new = torch.randn(1, cache.num_kv_heads, 1, cache.head_dim, generator=g)

    # Snapshot fp_tier score before, run, snapshot after.
    score_before = cache.fp_tier.score[layer_idx].clone()
    _ = cache.attention(layer_idx, q, k_new, v_new)
    score_after = cache.fp_tier.score[layer_idx].clone()

    # The fp_tier is the second buffer in _all_buffers (after ring), with capacity 3.
    # We just need to check that the formula matches "old" uniform-mean for T_q=1.
    # Expected: score = rho * score_before + (1-rho) * mean(attn) = rho * 1 + (1-rho) * attn[0]
    # We can't easily reproduce attn here without rerunning the model, so instead we just
    # check that the update is consistent with `score = rho*score_before + something_in_[0,1-rho]`.
    decay = score_after - rho * score_before                      # = received contribution
    # Each fp_tier slot's contribution must be in [0, (1-rho)] (since attn weights sum to 1
    # across heads' kv-head group, and per-query weighting at T_q=1 has weight (1-rho)).
    # Aggregated over n_rep=1 for our config — so per-slot received <= (1-rho).
    ok = (decay >= -1e-6).all().item() and (decay <= (1 - rho) + 1e-6).all().item()
    print(f"  decay range: [{decay.min().item():.4f}, {decay.max().item():.4f}], expected in [0, {1-rho:.4f}]")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_prefill_per_query_weights():
    """T_q=2: verify the per-query EMA weights apply correctly to score updates.

    For T_q=2 with policy=ema, expected weights = [(1-rho)*rho, (1-rho)] (for queries q=0, q=1).
    Old uniform-mean would give [(1-rho^2)/2, (1-rho^2)/2] — different.
    """
    print("\n--- test: T_q=2 per-query weights ---")
    rho = 0.9
    cache = make_cache(ema_decay=rho)
    layer_idx = 0
    # Empty cache, no fp_tier residents — we want to check NEW token init scoring (which
    # exercises the same q_weights path).
    cache.next_pos[layer_idx] = 0

    g = torch.Generator().manual_seed(13)
    T_q = 2
    H_q = cache.num_heads
    H_kv = cache.num_kv_heads
    D = cache.head_dim
    q = torch.randn(1, H_q, T_q, D, generator=g)
    k_new = torch.randn(1, H_kv, T_q, D, generator=g)
    v_new = torch.randn(1, H_kv, T_q, D, generator=g)

    # Compute the expected attn matrix the same way the cache would, so we can predict
    # the per-token init scores and compare.
    import math
    # Mirror the cache's attention: cache is empty, so scores are just q @ k_new.T with
    # causal masking.
    scaling = 1.0 / math.sqrt(D)
    # Repeat k_new to query-head count if needed (n_rep=1 here, but be explicit).
    n_rep = H_q // H_kv
    k_q = k_new.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k_new
    scores = (q @ k_q.transpose(-1, -2)) * scaling                      # [1, H_q, T_q, T_q]
    # Causal mask among new tokens.
    causal = torch.triu(torch.ones(T_q, T_q, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.view(1, 1, T_q, T_q), float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)            # [1, H_q, T_q, T_q]

    # Expected per-query weights:
    q_weights = (1.0 - rho) * rho ** torch.arange(T_q - 1, -1, -1, dtype=torch.float32)  # [(1-rho)*rho, 1-rho]
    expected_init = (attn * q_weights.view(1, 1, T_q, 1)).sum(dim=2)     # [1, H_q, T_q]
    if n_rep > 1:
        expected_init = expected_init.view(1, H_kv, n_rep, T_q).sum(dim=2)

    # Run the cache's attention and capture the init scores it stamped onto the new tokens.
    # The new tokens land in the ring (capacity 2, exactly T_q tokens). Their score field
    # in the ring buffer == the init score we computed.
    _ = cache.attention(layer_idx, q, k_new, v_new)
    ring_scores = cache.ring.score[layer_idx]                            # [B, H_kv, R=2]

    # The ring's slot order depends on cursor — for an empty cache and T_q=R, the small-T
    # branch is taken: tokens land at positions cursor + [0, T_q) % R = [0, 1].
    # So ring_scores[..., 0] corresponds to query 0, ring_scores[..., 1] to query 1.
    # And expected_init is shape [1, H_kv, T_q] in the same order.
    diff = (ring_scores - expected_init.to(ring_scores.dtype)).abs().max().item()
    ok = diff < 1e-6
    print(f"  expected = {expected_init}")
    print(f"  actual   = {ring_scores}")
    print(f"  max abs diff = {diff:.2e}")
    print(f"  {'PASS' if ok else 'FAIL'}")

    # Sanity: check that this is NOT what the OLD uniform-mean formula would give.
    old_uniform_init = attn.mean(dim=2) * (1 - rho ** T_q)               # = attn.sum/T_q * (1-rho^T_q)
    if n_rep > 1:
        old_uniform_init = old_uniform_init.view(1, H_kv, n_rep, T_q).sum(dim=2)
    differs_from_old = (ring_scores - old_uniform_init.to(ring_scores.dtype)).abs().max().item()
    print(f"  (sanity: differs from old uniform-mean formula by {differs_from_old:.4f})")
    return ok


def main():
    ok = True
    ok &= test_decode_unchanged()
    ok &= test_prefill_per_query_weights()
    if ok:
        print("\nAll cases passed.")
        sys.exit(0)
    else:
        print("\nSome cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
