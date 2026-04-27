"""Tiered KV cache with recency ring + N quantization tiers, per (layer, kv_head).

Tokens enter via a fp recency ring buffer. When the ring evicts (FIFO), the evicted
token undergoes a competitive cascade through the persistent tiers:

    ring (fp) -> tier 1 (fp) -> tier 2 (TurboQuant regular) -> tier 3 (TurboQuant aggressive) -> evicted

At each tier, the graduating token competes with the lowest-importance resident; if
its EMA importance score beats theirs, it takes the slot and the displaced resident
cascades to the next tier (re-quantized as it goes). Once placed, a token can only
stay or demote — there is no promotion (matches the design discussion).

Importance is tracked per (layer, kv_head, slot) as an EMA of *per-query* attention
weight (not summed-over-queries — the EMA is workload-independent). Effective decay
over T_q queries in a single attention call is rho ** T_q, which makes prefill and
decode behave as if the same sequence of single-query updates were applied.

The cascade is vectorized across (B, H_kv) and across all graduates: each tier runs a
single topk over [residents | candidates] and uses scatter_/gather to install winners.

Memory is bounded: ring + tier capacities are fixed at construction time.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from polar_quant import PolarQuant
from turbo_quant import QuantizedKV, TurboQuant
from turbo_attn import pack_bits, unpack_bits, _repeat_head, _force_set_attn_impl


_NEG_INF = float("-inf")


# ======================================================================================
# Per-tier storage
# ======================================================================================

class FpBuffer:
    """Fixed-capacity buffer storing fp K, V per (layer, B, H_kv, slot)."""

    def __init__(self, num_layers: int, batch_size: int, num_kv_heads: int,
                 capacity: int, head_dim: int,
                 device=None, dtype=torch.float32):
        L, B, H, C, D = num_layers, batch_size, num_kv_heads, capacity, head_dim
        self.capacity = capacity
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.K = torch.zeros(L, B, H, C, D, device=device, dtype=dtype)
        self.V = torch.zeros(L, B, H, C, D, device=device, dtype=dtype)
        self.pos = torch.full((L, B, H, C), -1, device=device, dtype=torch.long)
        self.score = torch.zeros(L, B, H, C, device=device, dtype=torch.float32)

    def valid_mask(self, layer_idx: int) -> torch.Tensor:
        """[B, H_kv, C] bool — true where a slot holds a real token."""
        return self.pos[layer_idx] >= 0

    def bytes_per_token(self) -> int:
        return 2 * self.head_dim * self.K.element_size()


class TurboBuffer:
    """Fixed-capacity buffer storing TurboQuant K + PolarQuant V at given bit budgets."""

    def __init__(self, num_layers: int, batch_size: int, num_kv_heads: int,
                 capacity: int, head_dim: int,
                 k_bits: int, v_bits: int, m: int,
                 seed: int = 0, device=None, dtype=torch.float32):
        L, B, H, C, D = num_layers, batch_size, num_kv_heads, capacity, head_dim
        self.capacity = capacity
        self.head_dim = head_dim
        self.k_total_bits = k_bits
        self.v_total_bits = v_bits
        self._k_idx_bits = k_bits - 1
        self._v_idx_bits = v_bits
        self.m = m
        self.device = device
        self.dtype = dtype

        self.k_quantizers = [
            TurboQuant(k_bits, head_dim, m, seed=seed + 2 * l, device=device, dtype=dtype)
            for l in range(num_layers)
        ]
        self.v_quantizers = [
            PolarQuant(v_bits, head_dim, seed=seed + 2 * l + 1, device=device, dtype=dtype)
            for l in range(num_layers)
        ]

        self._bytes_k_idx  = (head_dim * self._k_idx_bits + 7) // 8
        self._bytes_v_idx  = (head_dim * self._v_idx_bits + 7) // 8
        self._bytes_k_sign = (m + 7) // 8
        self._fp_bytes = torch.empty((), dtype=dtype).element_size()

        self.k_norm           = torch.zeros(L, B, H, C,                     device=device, dtype=dtype)
        self.k_idx_packed     = torch.zeros(L, B, H, C, self._bytes_k_idx,  device=device, dtype=torch.uint8)
        self.k_resnorm        = torch.zeros(L, B, H, C,                     device=device, dtype=dtype)
        self.k_ressign_packed = torch.zeros(L, B, H, C, self._bytes_k_sign, device=device, dtype=torch.uint8)
        self.v_norm           = torch.zeros(L, B, H, C,                     device=device, dtype=dtype)
        self.v_idx_packed     = torch.zeros(L, B, H, C, self._bytes_v_idx,  device=device, dtype=torch.uint8)
        self.pos              = torch.full((L, B, H, C), -1, device=device, dtype=torch.long)
        self.score            = torch.zeros(L, B, H, C, device=device, dtype=torch.float32)

    def valid_mask(self, layer_idx: int) -> torch.Tensor:
        return self.pos[layer_idx] >= 0

    def bytes_per_token(self) -> int:
        return (3 * self._fp_bytes
                + self._bytes_k_idx + self._bytes_k_sign + self._bytes_v_idx)

    def encode_batch(self, layer_idx: int, K: torch.Tensor, V: torch.Tensor):
        """Encode a [B, H_kv, T, D] batch of K, V into the tier's encoded fields.
        Returns the packed/normed components ready for scatter into storage:
            (k_norm, k_idx_packed, k_resnorm, k_ressign_packed, v_norm, v_idx_packed)
        """
        kq = self.k_quantizers[layer_idx].quantize(K)
        v_norm, v_idx = self.v_quantizers[layer_idx].encode(V)

        sign01 = (kq.res_signs.long() + 1) >> 1
        k_idx_packed     = pack_bits(kq.x_indices, self._k_idx_bits)
        k_ressign_packed = pack_bits(sign01, 1)
        v_idx_packed     = pack_bits(v_idx, self._v_idx_bits)
        return (kq.x_norm, k_idx_packed, kq.res_norm, k_ressign_packed,
                v_norm, v_idx_packed)

    def dequantize_all(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize every slot back to fp [B, H_kv, C, D] for K and V.
        Used by the cascade when residents are demoted to a lower tier.
        Note: the TurboQuant K reconstruction recovers only the MSE part; the QJL
        sign bits exist solely for IP estimation and are dropped here. This is the
        same lossy step that `reconstruct` does for the single-slot case.
        """
        D = self.head_dim
        L = layer_idx

        k_idx = unpack_bits(self.k_idx_packed[L], self._k_idx_bits, D)               # [B, H_kv, C, D]
        u_hat = self.k_quantizers[L].mse_quantizer.decode_rotated(k_idx)             # [B, H_kv, C, D]
        K_recon = self.k_norm[L].unsqueeze(-1) * (u_hat @ self.k_quantizers[L].R)    # [B, H_kv, C, D]

        v_idx = unpack_bits(self.v_idx_packed[L], self._v_idx_bits, D)
        V_recon = self.v_quantizers[L].decode(self.v_norm[L], v_idx)                 # [B, H_kv, C, D]
        return K_recon, V_recon

    # --- Vectorized score / V access for the hot path ---

    def score_pairwise(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        """Vectorized IP estimation against this tier's stored keys.
        q: [B, H_q, T_q, D]; returns [B, H_q, T_q, C].
        Invalid slots get 0 score (caller masks separately)."""
        D = self.head_dim
        k_idx = unpack_bits(self.k_idx_packed[layer_idx], self._k_idx_bits, D)        # [B, H_kv, C, D]
        sign01 = unpack_bits(self.k_ressign_packed[layer_idx], 1, self.m)             # [B, H_kv, C, m]
        signs = sign01.to(torch.int8) * 2 - 1
        x_norm   = self.k_norm[layer_idx]                                              # [B, H_kv, C]
        res_norm = self.k_resnorm[layer_idx]                                           # [B, H_kv, C]

        H_kv = x_norm.shape[1]
        H_q = q.shape[1]
        n_rep = H_q // H_kv
        if n_rep > 1:
            x_norm   = _repeat_head(x_norm,   n_rep)
            k_idx    = _repeat_head(k_idx,    n_rep)
            res_norm = _repeat_head(res_norm, n_rep)
            signs    = _repeat_head(signs,    n_rep)

        kq = QuantizedKV(x_norm=x_norm, x_indices=k_idx,
                         res_norm=res_norm, res_signs=signs)
        return self.k_quantizers[layer_idx].estimate_ip_pairwise(kq, q)

    def values(self, layer_idx: int, dtype: torch.dtype) -> torch.Tensor:
        """Vectorized V dequantization. Returns [B, H_kv, C, D] in `dtype`."""
        D = self.head_dim
        v_idx = unpack_bits(self.v_idx_packed[layer_idx], self._v_idx_bits, D)
        return self.v_quantizers[layer_idx].decode(self.v_norm[layer_idx], v_idx).to(dtype)


# ======================================================================================
# TieredKVCache
# ======================================================================================

class TieredKVCache:
    """Recency ring + always-on fp tier + N configurable TurboQuant tiers, per (layer, kv-head).

    Cascade order: ring (fp, FIFO) -> fp_tier (importance) -> quant_tiers[0] -> ... -> evict.
    The TQ tiers are listed in cascade order; conventionally, earlier = more bits / less aggressive.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        ring_size: int,
        fp_capacity: int,
        quant_tiers: list[tuple[int, int, int]],
        m: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        score_policy: str = "ema",
        ema_decay: float = 0.98,
        seed: int = 0,
        device=None,
        dtype=torch.float32,
    ):
        """
        Args:
            ring_size: capacity of the FIFO recency ring (always fp).
            fp_capacity: capacity of the persistent fp tier (importance-managed). Set
                to 0 to disable.
            quant_tiers: list of (k_bits, v_bits, capacity) tuples — one per TurboQuant
                tier, listed in cascade order (most-precise first). Pass [] for ring +
                fp only (no TQ tiers, with eviction past the fp tier).
            score_policy: how attention received drives importance score.
                - "ema": exponential moving average of per-query mean attention.
                  Decaying — old high scores fade if not reaffirmed. Workload-independent
                  (per-query mean keeps magnitudes bounded). Effective decay over T_q
                  queries is `ema_decay ** T_q`. Use a higher `ema_decay` (e.g. 0.996)
                  to approximate a longer "window" of past attention.
                - "cumulative": H2O-style cumulative sum across all attention received.
                  Monotonic — once a slot has accumulated attention, it stays heavy.
                  Workload-dependent (a 200-query prefill pumps 200× more into the
                  score than one decode query). Best fit for one-shot retrieval where
                  the relevant tokens are attended once and shouldn't be forgotten.
            ema_decay: per-query EMA decay rho (used when score_policy="ema"). Effective
                decay over a T_q-query attention call is rho ** T_q, so prefill and
                decode behave consistently.
        """
        assert batch_size == 1, "v1 supports batch_size=1 only"
        assert score_policy in ("ema", "cumulative"), \
            f"score_policy must be 'ema' or 'cumulative', got {score_policy!r}"
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads                  # query heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.m = m if m is not None else head_dim
        self.score_policy = score_policy
        self.ema_decay = ema_decay
        self.device = device
        self.dtype = dtype

        # Buffers.
        self.ring = FpBuffer(num_layers, batch_size, self.num_kv_heads,
                             ring_size, head_dim, device=device, dtype=dtype)
        self.fp_tier = FpBuffer(num_layers, batch_size, self.num_kv_heads,
                                fp_capacity, head_dim, device=device, dtype=dtype)
        self.quant_buffers: list[TurboBuffer] = []
        for i, (k_bits, v_bits, cap) in enumerate(quant_tiers):
            self.quant_buffers.append(TurboBuffer(
                num_layers, batch_size, self.num_kv_heads,
                cap, head_dim, k_bits=k_bits, v_bits=v_bits, m=self.m,
                seed=seed + 1000 * (i + 1), device=device, dtype=dtype,
            ))

        # Ring write cursor per (layer, b, h). Stays uniform across (b, h) as long as
        # all heads receive the same number of new tokens per attention call.
        self.ring_cursor = torch.zeros(num_layers, batch_size, self.num_kv_heads,
                                       device=device, dtype=torch.long)
        # Token write counter (next position to be written).
        self.next_pos = [0] * num_layers

        # Cascade chain: graduates from the ring enter fp_tier first, then cascade through
        # the quant tiers in declared order, then evict.
        self._tier_chain = [self.fp_tier] + self.quant_buffers
        self._tier_is_quantized = [False] + [True] * len(self.quant_buffers)
        # All cache buffers (used for score gathering & EMA in declaration order).
        self._all_buffers = [self.ring, self.fp_tier] + self.quant_buffers

    # ---------- memory accounting ----------

    def bytes_total(self) -> int:
        L, B, H = self.num_layers, self.batch_size, self.num_kv_heads
        per_lh = sum(buf.capacity * buf.bytes_per_token() for buf in self._all_buffers)
        return L * B * H * per_lh

    def reset(self) -> None:
        for buf in self._all_buffers:
            buf.pos.fill_(-1); buf.score.zero_()
        self.ring_cursor.zero_()
        self.next_pos = [0] * self.num_layers

    # ======================================================================================
    # Vectorized cascade
    # ======================================================================================

    def _cascade_vectorized(self, layer_idx: int,
                            cand_K: torch.Tensor, cand_V: torch.Tensor,
                            cand_pos: torch.Tensor, cand_score: torch.Tensor) -> None:
        """Vectorized cascade through tier1 -> tier2 -> tier3.

        cand_K, cand_V: [B, H_kv, G, D] fp (graduates from the ring).
        cand_pos:       [B, H_kv, G] long.
        cand_score:     [B, H_kv, G] float32 (use -inf for invalid/padding entries).

        Tokens that lose at every tier are evicted from the cache entirely.
        """
        for tier_idx, tier in enumerate(self._tier_chain):
            is_quant = self._tier_is_quantized[tier_idx]
            cand_K, cand_V, cand_pos, cand_score = self._compete_at_tier(
                tier, layer_idx, cand_K, cand_V, cand_pos, cand_score, is_quant
            )
            # Early exit: if all remaining candidate scores are -inf, nothing to cascade.
            if torch.isinf(cand_score).all() and (cand_score <= 0).all():
                break

    def _compete_at_tier(self, tier, layer_idx: int,
                         cand_K: torch.Tensor, cand_V: torch.Tensor,
                         cand_pos: torch.Tensor, cand_score: torch.Tensor,
                         is_quantized: bool):
        """Single-tier competition. All candidates compete against current residents
        for the C tier slots; winners (top-C by score) are installed, losers cascade.

        Returns (next_K, next_V, next_pos, next_score) sized [B, H_kv, C+G, ...] —
        the inputs to the next tier's competition. Entries with score == -inf are
        non-cascading (either kept residents or rejected -inf-padded candidates).
        """
        B = self.batch_size
        H_kv = self.num_kv_heads
        C = tier.capacity
        G = cand_score.shape[-1]
        D = self.head_dim
        device = cand_score.device
        score_dtype = torch.float32

        # Zero-capacity tier: pass-through. All candidates cascade to the next tier
        # (or get evicted past the chain end).
        if C == 0:
            return cand_K, cand_V, cand_pos, cand_score

        # ---- 1. Read current residents and mask invalid as -inf ----
        res_pos    = tier.pos[layer_idx]                                          # [B, H_kv, C]
        res_valid  = res_pos >= 0
        res_score  = tier.score[layer_idx].to(score_dtype)
        NEG = torch.full_like(res_score, _NEG_INF)
        res_score_masked = torch.where(res_valid, res_score, NEG)

        cand_score_f = cand_score.to(score_dtype)

        # ---- 2. Pool [residents | candidates] and pick top-C by score ----
        pool_score = torch.cat([res_score_masked, cand_score_f], dim=-1)          # [B, H_kv, C+G]
        # Note: top_idx values < C refer to residents; values >= C refer to candidates.
        top_idx = pool_score.topk(C, dim=-1, largest=True).indices                # [B, H_kv, C]

        # ---- 3. Compute kept_residents: which resident slots survived top-C ----
        # scatter into a (C+1)-sized buffer; the trailing slot absorbs candidate
        # entries (top_idx >= C) so they don't mark anything as kept.
        is_res_in_top = top_idx < C                                               # [B, H_kv, C]
        sentinel_C = torch.full_like(top_idx, C)
        kept_target = torch.where(is_res_in_top, top_idx, sentinel_C)             # [B, H_kv, C], values in [0, C]
        kept_ext = torch.zeros(B, H_kv, C + 1, dtype=torch.bool, device=device)
        kept_ext.scatter_(2, kept_target, torch.ones_like(kept_target, dtype=torch.bool))
        kept_residents = kept_ext[..., :C]                                        # [B, H_kv, C]

        # ---- 4. Compute cand_accepted: which candidates made it into top-C ----
        is_cand_in_top = top_idx >= C
        sentinel_G = torch.full_like(top_idx, G)
        cand_target = torch.where(is_cand_in_top, top_idx - C, sentinel_G)        # values in [0, G]
        cand_ext = torch.zeros(B, H_kv, G + 1, dtype=torch.bool, device=device)
        cand_ext.scatter_(2, cand_target, torch.ones_like(cand_target, dtype=torch.bool))
        cand_accepted = cand_ext[..., :G]                                         # [B, H_kv, G]

        # ---- 5. Pair open slots <-> accepted candidates (both in ascending order) ----
        open_mask = ~kept_residents                                               # [B, H_kv, C]
        arange_C = torch.arange(C, device=device).expand(B, H_kv, C)
        # For non-open slots, sort key is C (sentinel); they sort to the tail.
        open_sort_key = torch.where(open_mask, arange_C, torch.full_like(arange_C, C))
        open_slot_indices, _ = open_sort_key.sort(dim=-1)                         # [B, H_kv, C]

        arange_G = torch.arange(G, device=device).expand(B, H_kv, G)
        cand_sort_key = torch.where(cand_accepted, arange_G, torch.full_like(arange_G, G))
        cand_indices_chosen, _ = cand_sort_key.sort(dim=-1)                       # [B, H_kv, G]

        # Pad/truncate cand_indices_chosen to length C so it pairs slot-for-slot with
        # open_slot_indices. By construction num_open == num_accepted, so the valid
        # prefix of both has the same length.
        if G < C:
            pad = torch.full((B, H_kv, C - G), G, device=device, dtype=cand_indices_chosen.dtype)
            cand_indices_chosen = torch.cat([cand_indices_chosen, pad], dim=-1)
        elif G > C:
            cand_indices_chosen = cand_indices_chosen[..., :C]
        # Now both are [B, H_kv, C].

        # ---- 6. Gather candidate data for accepted candidates, in pairing order ----
        safe_cand = cand_indices_chosen.clamp(max=max(G - 1, 0))                  # for sentinel pairs, this is bogus
        new_K_open   = cand_K.gather(2, safe_cand.unsqueeze(-1).expand(-1, -1, -1, D))  # [B, H_kv, C, D]
        new_V_open   = cand_V.gather(2, safe_cand.unsqueeze(-1).expand(-1, -1, -1, D))
        new_pos_open = cand_pos.gather(2, safe_cand)                              # [B, H_kv, C]
        new_score_open = cand_score_f.gather(2, safe_cand)

        # A pair is "real" when both ends point inside their valid range AND the gathered
        # candidate has a non-(-inf) score. We allow +inf (used by `pin_sinks` to mark
        # un-evictable sinks); we reject -inf (which is how we mark padding / invalid
        # entries that may have leaked into top-C when C exceeds the count of real candidates).
        pair_valid = (cand_indices_chosen < G) & (open_slot_indices < C) & (new_score_open > _NEG_INF)
        new_pos_open   = torch.where(pair_valid, new_pos_open,   torch.full_like(new_pos_open, -1))
        new_score_open = torch.where(pair_valid, new_score_open, torch.zeros_like(new_score_open))
        # For invalid pairs we can leave K/V garbage; the slot pos == -1 masks them in attention.

        # ---- 7. Build next-tier candidate stream ----
        # Residents that left (~kept & valid): keep their original score for cascading.
        # Residents that stayed: -inf score (won't cascade).
        # Candidates accepted: -inf (won't cascade).
        # Candidates rejected: keep their score.
        next_res_score  = torch.where(kept_residents, NEG, res_score_masked)
        next_cand_score = torch.where(cand_accepted,
                                       torch.full_like(cand_score_f, _NEG_INF),
                                       cand_score_f)

        if is_quantized:
            res_K_fp, res_V_fp = tier.dequantize_all(layer_idx)
        else:
            res_K_fp = tier.K[layer_idx]
            res_V_fp = tier.V[layer_idx]

        next_K     = torch.cat([res_K_fp, cand_K],          dim=2)                # [B, H_kv, C+G, D]
        next_V     = torch.cat([res_V_fp, cand_V],          dim=2)
        next_pos   = torch.cat([res_pos,  cand_pos],        dim=2)                # [B, H_kv, C+G]
        next_score = torch.cat([next_res_score, next_cand_score], dim=2)

        # ---- 8. Install winners into the tier's storage ----
        self._scatter_install(tier, layer_idx, open_slot_indices,
                              new_K_open, new_V_open, new_pos_open, new_score_open,
                              is_quantized)

        return next_K, next_V, next_pos, next_score

    def _scatter_install(self, tier, layer_idx: int, open_slots: torch.Tensor,
                         K_new: torch.Tensor, V_new: torch.Tensor,
                         pos_new: torch.Tensor, score_new: torch.Tensor,
                         is_quantized: bool) -> None:
        """Write [B, H_kv, C, ...] data into tier slots indexed by `open_slots`.

        `open_slots` has shape [B, H_kv, C], with values in [0, C]. Values equal to C
        are sentinels (no-write); they go to a throwaway slot in a (C+1)-extended
        scratch buffer, which is then sliced off.
        """
        B = self.batch_size
        H_kv = self.num_kv_heads
        C = tier.capacity
        D = self.head_dim
        device = open_slots.device

        def _scatter_3d(field_layer: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            ext = torch.zeros((B, H_kv, C + 1), dtype=field_layer.dtype, device=device)
            ext[..., :C] = field_layer
            ext.scatter_(2, open_slots, src.to(field_layer.dtype))
            return ext[..., :C]

        def _scatter_4d(field_layer: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            extra = src.shape[-1]
            ext = torch.zeros((B, H_kv, C + 1, extra), dtype=field_layer.dtype, device=device)
            ext[..., :C, :] = field_layer
            idx = open_slots.unsqueeze(-1).expand(-1, -1, -1, extra)
            ext.scatter_(2, idx, src.to(field_layer.dtype))
            return ext[..., :C, :]

        # pos / score are always 3D.
        tier.pos[layer_idx]   = _scatter_3d(tier.pos[layer_idx],   pos_new)
        tier.score[layer_idx] = _scatter_3d(tier.score[layer_idx], score_new)

        if is_quantized:
            # Encode the winners and scatter the encoded fields. Note: we encode all C
            # winners (including bogus data at sentinel slots); their pos == -1 makes
            # them invisible to attention regardless.
            (k_norm_new, k_idx_packed_new, k_resnorm_new,
             k_ressign_packed_new, v_norm_new, v_idx_packed_new) = tier.encode_batch(
                layer_idx, K_new, V_new
            )
            tier.k_norm[layer_idx]    = _scatter_3d(tier.k_norm[layer_idx],    k_norm_new)
            tier.k_resnorm[layer_idx] = _scatter_3d(tier.k_resnorm[layer_idx], k_resnorm_new)
            tier.v_norm[layer_idx]    = _scatter_3d(tier.v_norm[layer_idx],    v_norm_new)
            tier.k_idx_packed[layer_idx]     = _scatter_4d(tier.k_idx_packed[layer_idx],     k_idx_packed_new)
            tier.k_ressign_packed[layer_idx] = _scatter_4d(tier.k_ressign_packed[layer_idx], k_ressign_packed_new)
            tier.v_idx_packed[layer_idx]     = _scatter_4d(tier.v_idx_packed[layer_idx],     v_idx_packed_new)
        else:
            tier.K[layer_idx] = _scatter_4d(tier.K[layer_idx], K_new)
            tier.V[layer_idx] = _scatter_4d(tier.V[layer_idx], V_new)

    # ======================================================================================
    # Vectorized ring ingestion
    # ======================================================================================

    def _ingest_into_ring(self, layer_idx: int,
                          K_new: torch.Tensor, V_new: torch.Tensor,
                          positions: list[int],
                          initial_scores: Optional[torch.Tensor] = None) -> None:
        """Write fresh tokens into the recency ring, evicting overwritten occupants
        into the tier cascade. Vectorized across (B, H_kv) and across T_new.

        K_new, V_new : [B, H_kv, T_new, D]
        initial_scores: [B, H_kv, T_new] or None
        """
        B, H_kv, T_new, D = K_new.shape
        if T_new == 0:
            return
        R = self.ring.capacity
        device = K_new.device
        cursor = int(self.ring_cursor[layer_idx, 0, 0].item())

        positions_t = torch.tensor(positions, device=device, dtype=torch.long)    # [T_new]

        if initial_scores is None:
            init_scores = torch.zeros(B, H_kv, T_new, device=device,
                                       dtype=self.ring.score.dtype)
        else:
            init_scores = initial_scores.to(self.ring.score.dtype)

        if T_new <= R:
            # Each ring slot is written at most once. Graduates are the existing
            # occupants of the slots about to be overwritten.
            write_idx = (cursor + torch.arange(T_new, device=device)) % R         # [T_new]

            # Read current occupants at write_idx.
            old_K     = self.ring.K[layer_idx, :, :, write_idx, :].clone()        # [B, H_kv, T_new, D]
            old_V     = self.ring.V[layer_idx, :, :, write_idx, :].clone()
            old_pos   = self.ring.pos[layer_idx, :, :, write_idx].clone()         # [B, H_kv, T_new]
            old_score = self.ring.score[layer_idx, :, :, write_idx].clone()
            old_valid = old_pos >= 0
            grad_score = torch.where(old_valid, old_score.to(torch.float32),
                                      torch.full_like(old_score, _NEG_INF, dtype=torch.float32))

            # Install new tokens.
            self.ring.K[layer_idx, :, :, write_idx, :]   = K_new
            self.ring.V[layer_idx, :, :, write_idx, :]   = V_new
            self.ring.pos[layer_idx, :, :, write_idx]    = positions_t.view(1, 1, T_new).expand(B, H_kv, T_new)
            self.ring.score[layer_idx, :, :, write_idx]  = init_scores

            if old_valid.any():
                self._cascade_vectorized(layer_idx, old_K, old_V, old_pos, grad_score)
        else:
            # T_new > R: ring overflows. Net effect:
            #   - All R existing ring tokens are evicted.
            #   - First (T_new - R) new tokens are written then immediately overwritten.
            #   - Last R new tokens stay in the ring.
            excess = T_new - R

            existing_K     = self.ring.K[layer_idx].clone()                       # [B, H_kv, R, D]
            existing_V     = self.ring.V[layer_idx].clone()
            existing_pos   = self.ring.pos[layer_idx].clone()                     # [B, H_kv, R]
            existing_score = self.ring.score[layer_idx].clone()
            existing_valid = existing_pos >= 0
            existing_score_for_cascade = torch.where(
                existing_valid, existing_score.to(torch.float32),
                torch.full_like(existing_score, _NEG_INF, dtype=torch.float32),
            )

            cascade_K     = K_new[:, :, :excess, :]
            cascade_V     = V_new[:, :, :excess, :]
            cascade_pos   = positions_t[:excess].view(1, 1, excess).expand(B, H_kv, excess).contiguous()
            cascade_score = init_scores[:, :, :excess].to(torch.float32)

            all_K     = torch.cat([existing_K,     cascade_K],     dim=2)
            all_V     = torch.cat([existing_V,     cascade_V],     dim=2)
            all_pos   = torch.cat([existing_pos,   cascade_pos],   dim=2)
            all_score = torch.cat([existing_score_for_cascade, cascade_score], dim=2)

            # Last R new tokens become the new ring contents.
            last_K   = K_new[:, :, excess:, :]
            last_V   = V_new[:, :, excess:, :]
            last_pos = positions_t[excess:].view(1, 1, R).expand(B, H_kv, R).contiguous()
            last_score = init_scores[:, :, excess:]

            # Their physical ring slots are (cursor + excess + k) % R for k in [0, R),
            # which is a permutation of [0, R) — exactly fills the ring.
            write_offsets = (cursor + excess + torch.arange(R, device=device)) % R  # [R]
            self.ring.K[layer_idx, :, :, write_offsets, :]  = last_K
            self.ring.V[layer_idx, :, :, write_offsets, :]  = last_V
            self.ring.pos[layer_idx, :, :, write_offsets]   = last_pos
            self.ring.score[layer_idx, :, :, write_offsets] = last_score

            self._cascade_vectorized(layer_idx, all_K, all_V, all_pos, all_score)

        self.ring_cursor[layer_idx].fill_((cursor + T_new) % R)

    # ======================================================================================
    # Score gathering for attention
    # ======================================================================================

    def _scores_and_meta(self, layer_idx: int, q: torch.Tensor):
        """Concatenated scores, positions, and validity across all 4 buffers.
        Returns:
            scores [B, H_q, T_q, Ctot]
            pos    [B, H_q, Ctot]
            valid  [B, H_q, Ctot] bool
        """
        n_rep = self.n_rep
        q_dtype = q.dtype

        score_chunks: list[torch.Tensor] = []
        pos_chunks:   list[torch.Tensor] = []
        valid_chunks: list[torch.Tensor] = []

        # Helper: fp tier scoring with proper dtype handling.
        def _fp_score(K: torch.Tensor) -> torch.Tensor:
            K_q = K.to(q_dtype)
            if n_rep > 1:
                K_q = _repeat_head(K_q, n_rep)
            return q @ K_q.transpose(-1, -2)                                       # [B, H_q, T_q, C]

        def _expand_meta(pos: torch.Tensor):
            valid = pos >= 0
            if n_rep > 1:
                pos = _repeat_head(pos, n_rep)
                valid = _repeat_head(valid, n_rep)
            return pos, valid

        # Walk all buffers in declaration order: ring, fp_tier, *quant_tiers.
        for buf in self._all_buffers:
            if isinstance(buf, FpBuffer):
                score_chunks.append(_fp_score(buf.K[layer_idx]))
            else:  # TurboBuffer
                score_chunks.append(buf.score_pairwise(layer_idx, q).to(q_dtype))
            p, v = _expand_meta(buf.pos[layer_idx])
            pos_chunks.append(p); valid_chunks.append(v)

        scores    = torch.cat(score_chunks, dim=-1)                                # [B, H_q, T_q, Ctot]
        positions = torch.cat(pos_chunks,   dim=-1)                                # [B, H_q, Ctot]
        valid     = torch.cat(valid_chunks, dim=-1)
        return scores, positions, valid

    def _values(self, layer_idx: int, dtype: torch.dtype) -> torch.Tensor:
        """Concatenated V across all buffers, expanded to query heads. [B, H_q, Ctot, D]."""
        v_chunks = []
        for buf in self._all_buffers:
            if isinstance(buf, FpBuffer):
                v_chunks.append(buf.V[layer_idx].to(dtype))
            else:
                v_chunks.append(buf.values(layer_idx, dtype))
        V = torch.cat(v_chunks, dim=-2)                                            # [B, H_kv, Ctot, D]
        if self.n_rep > 1:
            V = _repeat_head(V, self.n_rep)
        return V

    # ======================================================================================
    # Attention forward
    # ======================================================================================

    def attention(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        scaling: Optional[float] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention against (cached prefix in 4 buffers) ⊕ (fresh K/V).
        After attention, fresh K/V are pushed into the ring (which may evict tokens
        into the tier cascade), and importance scores are EMA-updated.

        q     : [B, H_q,  T_q,   D]
        k_new : [B, H_kv, T_new, D]
        v_new : [B, H_kv, T_new, D]
        Returns: [B, H_q, T_q, D].
        """
        B, H_q, T_q, D = q.shape
        T_new = k_new.shape[-2]
        H_kv = self.num_kv_heads
        n_rep = self.n_rep
        q_dtype = q.dtype
        device = q.device
        if scaling is None:
            scaling = 1.0 / math.sqrt(D)

        new_positions = list(range(self.next_pos[layer_idx],
                                   self.next_pos[layer_idx] + T_new))
        q_positions = torch.tensor(new_positions, device=device, dtype=torch.long)

        # ----- score the cache -----
        cache_scores, cache_pos, cache_valid = self._scores_and_meta(layer_idx, q)
        cache_scores = cache_scores * scaling

        # ----- score the new (fresh) tokens -----
        k_new_q = k_new.to(q_dtype)
        if n_rep > 1:
            k_new_q = _repeat_head(k_new_q, n_rep)
        score_new = (q @ k_new_q.transpose(-1, -2)) * scaling                      # [B, H_q, T_q, T_new]

        # ----- masking -----
        invalid = ~cache_valid.unsqueeze(2)                                        # [B, H_q, 1, Ctot]
        future = cache_pos.unsqueeze(2) > q_positions.view(1, 1, -1, 1)            # [B, H_q, T_q, Ctot]
        cache_scores = cache_scores.masked_fill(invalid | future, _NEG_INF)

        # Causal among new tokens (assumes T_q == T_new with shared positions).
        new_pos_for_q = q_positions.view(-1, 1)                                    # [T_q, 1]
        new_pos_for_k = q_positions.view(1, -1)                                    # [1, T_new]
        future_new = new_pos_for_k > new_pos_for_q                                 # [T_q, T_new]
        score_new = score_new.masked_fill(future_new.view(1, 1, T_q, T_new), _NEG_INF)

        scores = torch.cat([cache_scores, score_new], dim=-1)                      # [B, H_q, T_q, Ctot+T_new]

        # HF padding/causal mask: pad zeros on the left for cache positions.
        if attn_mask is not None:
            T_total = scores.shape[-1]
            if attn_mask.shape[-1] < T_total:
                pad_shape = list(attn_mask.shape)
                pad_shape[-1] = T_total - attn_mask.shape[-1]
                left_pad = torch.zeros(pad_shape, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([left_pad, attn_mask], dim=-1)
            scores = scores + attn_mask[..., :T_total]

        # ----- softmax + V multiply -----
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_dtype)          # [B, H_q, T_q, Ctot+T_new]

        V_cache = self._values(layer_idx, q_dtype)                                 # [B, H_q, Ctot, D]
        v_new_q = v_new.to(q_dtype)
        if n_rep > 1:
            v_new_q = _repeat_head(v_new_q, n_rep)
        V_full = torch.cat([V_cache, v_new_q], dim=-2)                             # [B, H_q, Ctot+T_new, D]
        out = attn @ V_full                                                        # [B, H_q, T_q, D]

        # ----- importance score update on cache portion -----
        # Two policies; both aggregate over the n_rep query heads that share each kv-head
        # (so a kv slot's score reflects total attention from its group).
        Ctot = cache_pos.shape[-1]
        attn_cache_fp32 = attn[..., :Ctot].to(torch.float32)                       # [B, H_q, T_q, Ctot]

        if self.score_policy == "ema":
            # Per-query mean attention (workload-independent: bounded in [0, 1] regardless of T_q).
            # Effective decay over T_q queries is rho ** T_q, matching the result of
            # T_q sequential single-query EMA updates with received ≈ constant.
            attn_cache_received = attn_cache_fp32.mean(dim=2)                      # [B, H_q, Ctot]
        else:  # "cumulative" — H2O-style sum
            # Per-step sum: the score grows by total attention received this step.
            # Workload-dependent (long prefill = big bump), monotonic non-decreasing.
            attn_cache_received = attn_cache_fp32.sum(dim=2)                       # [B, H_q, Ctot]

        if n_rep > 1:
            attn_cache_received = attn_cache_received.view(B, H_kv, n_rep, Ctot).sum(dim=2)

        rho = self.ema_decay
        effective_rho = rho ** T_q

        # Build per-buffer slices in declaration order (ring, fp_tier, *quant_tiers).
        offset = 0
        for buf in self._all_buffers:
            lo, hi = offset, offset + buf.capacity
            received = attn_cache_received[:, :, lo:hi].to(buf.score.dtype)
            if self.score_policy == "ema":
                buf.score[layer_idx] = effective_rho * buf.score[layer_idx] + (1 - effective_rho) * received
            else:  # "cumulative"
                buf.score[layer_idx] = buf.score[layer_idx] + received
            offset = hi

        # ----- seed initial scores for the new tokens -----
        # Compute the per-query received attention for new tokens, normalized by T_q so
        # that blocked queries contribute 0 (preserves sink signal). Then convert to
        # whichever policy's score scale.
        attn_new = attn[..., Ctot:].to(torch.float32)                              # [B, H_q, T_q, T_new]
        attn_new_sum = attn_new.sum(dim=2)                                          # [B, H_q, T_new]
        if n_rep > 1:
            attn_new_sum = attn_new_sum.view(B, H_kv, n_rep, T_new).sum(dim=2)      # [B, H_kv, T_new]

        if self.score_policy == "ema":
            attn_new_mean = attn_new_sum / float(T_q)
            new_received_init = (1 - effective_rho) * attn_new_mean                # one effective EMA step from prior 0
        else:  # "cumulative"
            new_received_init = attn_new_sum                                        # cumulative starts from 0 + sum_attn

        # ----- ingest into ring (may trigger cascade) -----
        self._ingest_into_ring(layer_idx, k_new, v_new, new_positions,
                               initial_scores=new_received_init)
        self.next_pos[layer_idx] += T_new

        return out


# ======================================================================================
# HF dispatcher integration
# ======================================================================================

def tiered_attn_function(module, query, key, value, attention_mask=None,
                         scaling=None, dropout=0.0, **kwargs):
    """HF ALL_ATTENTION_FUNCTIONS callable for the tiered cache."""
    cache: TieredKVCache = module._tiered_cache
    layer_idx: int = module.layer_idx

    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out, None


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["tiered"] = tiered_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def install_tiered_attention(model: nn.Module, cache: TieredKVCache) -> int:
    """Switch the model's attention dispatcher to 'tiered' and attach the cache."""
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required")
    if hasattr(model, "config"):
        _force_set_attn_impl(model.config, "tiered")
    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config, "tiered")
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._tiered_cache = cache
            n += 1
    return n