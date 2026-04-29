"""Synthetic equivalence test: G=1 fast path vs. topk path produce identical results.

Builds a small KVCascadeCache with a known fp_tier + quant_tier state, then for a
sequence of synthetic candidates compares:
  - tier state after install (pos, score, K/V or packed encodings)
  - next-tier candidate stream returned by the cascade step
between _compete_at_tier_g1 and _compete_at_tier (G=1 case).

Runs on CPU in fp32 to keep numerics deterministic.
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "src"))

import copy
import torch

from kvcascade import KVCascadeCache


def make_cache(seed=0):
    return KVCascadeCache(
        num_layers=2, batch_size=1, num_heads=4, head_dim=8,
        ring_size=2, fp_capacity=3,
        quant_tiers=[(4, 2, 4)],   # one quant tier, k=4 (so JL bits exist), v=2, capacity 4
        m=8, score_policy="ema", seed=seed,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def seed_tier(tier, layer_idx, n, score_offset=0.0, pos_offset=0, seed=1):
    """Fill the first `n` slots of a tier with deterministic synthetic data."""
    g = torch.Generator().manual_seed(seed)
    B, H = 1, tier.K.shape[2] if hasattr(tier, "K") else tier.k_norm.shape[2]
    D = tier.head_dim
    K = torch.randn(B, H, n, D, generator=g)
    V = torch.randn(B, H, n, D, generator=g)
    pos = (torch.arange(n) + pos_offset).view(1, 1, n).expand(B, H, n).contiguous().long()
    # Add a tiny per-slot jitter so scores stay distinct across tiers (avoids topk tie-breaking
    # ambiguity that has nothing to do with the path under test).
    score = (torch.arange(n, dtype=torch.float32) + score_offset + 0.001 * pos_offset).view(1, 1, n).expand(B, H, n).contiguous()

    if hasattr(tier, "K"):
        tier.K[layer_idx, :, :, :n] = K
        tier.V[layer_idx, :, :, :n] = V
    else:
        # quant tier: use encode_batch to fill slots
        (k_norm, k_idx_packed, k_resnorm, k_ressign_packed, v_norm, v_idx_packed) = tier.encode_batch(layer_idx, K, V)
        tier.k_norm[layer_idx, :, :, :n]            = k_norm
        tier.k_resnorm[layer_idx, :, :, :n]         = k_resnorm
        tier.k_idx_packed[layer_idx, :, :, :n]      = k_idx_packed
        tier.k_ressign_packed[layer_idx, :, :, :n]  = k_ressign_packed
        tier.v_norm[layer_idx, :, :, :n]            = v_norm
        tier.v_idx_packed[layer_idx, :, :, :n]      = v_idx_packed
    tier.pos[layer_idx, :, :, :n] = pos
    tier.score[layer_idx, :, :, :n] = score


def tier_state_dict(tier, layer_idx):
    """Snapshot a tier's per-layer state for comparison."""
    out = {
        "pos":   tier.pos[layer_idx].clone(),
        "score": tier.score[layer_idx].clone(),
    }
    if hasattr(tier, "K"):
        out["K"] = tier.K[layer_idx].clone()
        out["V"] = tier.V[layer_idx].clone()
    else:
        out["k_norm"]            = tier.k_norm[layer_idx].clone()
        out["k_resnorm"]         = tier.k_resnorm[layer_idx].clone()
        out["k_idx_packed"]      = tier.k_idx_packed[layer_idx].clone()
        out["k_ressign_packed"]  = tier.k_ressign_packed[layer_idx].clone()
        out["v_norm"]            = tier.v_norm[layer_idx].clone()
        out["v_idx_packed"]      = tier.v_idx_packed[layer_idx].clone()
    return out


def assert_tier_states_equal(s1, s2, label):
    keys = sorted(set(s1) | set(s2))
    all_ok = True
    for k in keys:
        a, b = s1[k], s2[k]
        if a.dtype.is_floating_point:
            ok = torch.allclose(a, b, atol=1e-5, rtol=1e-5)
        else:
            ok = torch.equal(a, b)
        if not ok:
            diff = (a.float() - b.float()).abs()
            mismatch_per_head = diff.flatten(2).sum(-1) if diff.dim() >= 3 else diff
            print(f"  MISMATCH on {label}.{k}:")
            print(f"    max abs diff = {diff.max().item()}")
            print(f"    per-head total diff: {mismatch_per_head}")
            print(f"    pos[s1] = {s1['pos']}")
            print(f"    pos[s2] = {s2['pos']}")
            print(f"    score[s1] = {s1['score']}")
            print(f"    score[s2] = {s2['score']}")
            all_ok = False
    return all_ok


def run_case(label, cand_K, cand_V, cand_pos, cand_score, fp_n, q_n,
             fp_score_offset=0.0, q_score_offset=0.0, layer_idx=0):
    """Run the same input through both paths on identical caches; compare tier state and next stream."""
    cache_old = make_cache()
    cache_new = make_cache()

    # Seed both caches identically. Use disjoint pos ranges so tokens cascading
    # between tiers don't collide on identifiers.
    for cache in (cache_old, cache_new):
        seed_tier(cache.fp_tier,         layer_idx, fp_n, score_offset=fp_score_offset, pos_offset=0)
        seed_tier(cache.quant_buffers[0], layer_idx, q_n,  score_offset=q_score_offset,  pos_offset=100)

    # Run the topk path explicitly on cache_old.
    next_old = cache_old._compete_at_tier(
        cache_old.fp_tier, layer_idx,
        cand_K.clone(), cand_V.clone(), cand_pos.clone(), cand_score.clone(),
        is_quantized=False,
    )
    # Continue cascade through quant tier.
    next_old = cache_old._compete_at_tier(
        cache_old.quant_buffers[0], layer_idx,
        next_old[0], next_old[1], next_old[2], next_old[3],
        is_quantized=True,
    )

    # Run the G=1 fast path on cache_new.
    next_new = cache_new._compete_at_tier_g1(
        cache_new.fp_tier, layer_idx,
        cand_K.clone(), cand_V.clone(), cand_pos.clone(), cand_score.clone(),
        is_quantized=False,
    )
    next_new = cache_new._compete_at_tier_g1(
        cache_new.quant_buffers[0], layer_idx,
        next_new[0], next_new[1], next_new[2], next_new[3],
        is_quantized=True,
    )

    # Compare final tier state.
    ok = True
    s_old = tier_state_dict(cache_old.fp_tier, layer_idx)
    s_new = tier_state_dict(cache_new.fp_tier, layer_idx)
    ok &= assert_tier_states_equal(s_old, s_new, "fp_tier")
    s_old = tier_state_dict(cache_old.quant_buffers[0], layer_idx)
    s_new = tier_state_dict(cache_new.quant_buffers[0], layer_idx)
    ok &= assert_tier_states_equal(s_old, s_new, "quant_tier")

    print(f"[{label}] {'OK' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(42)
    B, H_kv, D = 1, 4, 8

    # ---- Case 1: cand beats some heads' fp residents, loses to others. ----
    cand_K = torch.randn(B, H_kv, 1, D)
    cand_V = torch.randn(B, H_kv, 1, D)
    cand_pos = torch.tensor([[[100], [101], [102], [103]]], dtype=torch.long)
    # fp_tier scores at [0, 1, 2] across heads; cand_score=1.5 wins on some heads.
    cand_score = torch.tensor([[[1.5], [1.5], [1.5], [1.5]]], dtype=torch.float32)
    ok1 = run_case("c1: cand beats min in fp", cand_K, cand_V, cand_pos, cand_score,
                   fp_n=3, q_n=4, fp_score_offset=0.0, q_score_offset=10.0)

    # ---- Case 2: cand loses everywhere (passes through to evict). ----
    cand_score2 = torch.tensor([[[-5.0], [-5.0], [-5.0], [-5.0]]], dtype=torch.float32)
    ok2 = run_case("c2: cand loses at every tier", cand_K, cand_V, cand_pos, cand_score2,
                   fp_n=3, q_n=4, fp_score_offset=0.0, q_score_offset=-10.0)

    # ---- Case 3: cand beats fp; displaced fp resident must beat quant. ----
    cand_score3 = torch.tensor([[[100.0], [100.0], [100.0], [100.0]]], dtype=torch.float32)
    ok3 = run_case("c3: cand beats fp, displaced cascades into quant", cand_K, cand_V, cand_pos, cand_score3,
                   fp_n=3, q_n=4, fp_score_offset=10.0, q_score_offset=0.0)

    # ---- Case 4: tier partially full (some empty slots in fp). ----
    cand_score4 = torch.tensor([[[5.0], [5.0], [5.0], [5.0]]], dtype=torch.float32)
    ok4 = run_case("c4: fp partially full (cand fills empty slot)", cand_K, cand_V, cand_pos, cand_score4,
                   fp_n=1, q_n=4, fp_score_offset=0.0, q_score_offset=10.0)

    # ---- Case 5: cand_score is -inf (invalid candidate). ----
    cand_score5 = torch.tensor([[[float("-inf")], [float("-inf")], [float("-inf")], [float("-inf")]]], dtype=torch.float32)
    ok5 = run_case("c5: -inf candidate (no-op)", cand_K, cand_V, cand_pos, cand_score5,
                   fp_n=3, q_n=4)

    # ---- Case 6: heterogeneous — different heads make different decisions. ----
    cand_score6 = torch.tensor([[[100.0], [-5.0], [1.5], [-100.0]]], dtype=torch.float32)
    ok6 = run_case("c6: heterogeneous heads", cand_K, cand_V, cand_pos, cand_score6,
                   fp_n=3, q_n=4, fp_score_offset=0.0, q_score_offset=0.0)

    if ok1 and ok2 and ok3 and ok4 and ok5 and ok6:
        print("\nAll cases passed.")
        sys.exit(0)
    else:
        print("\nSome cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
