"""H2O: Heavy-Hitter Oracle. Cumulative-attention scored eviction.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models" (NeurIPS 2023).
https://arxiv.org/abs/2306.14048

Mechanics:
  - Maintain a fixed `cache_size` per (layer, kv-head) of fp K, V slots.
  - Each slot has a `score` = cumulative attention received across all prior
    queries. Monotonic — once heavy, stays heavy.
  - At prefill end, compress to the top-`cache_size` prefill tokens by score.
  - At decode, each new token is appended into the cache via online eviction:
    if any slot has lower score, the new token replaces the lowest; otherwise
    the new token is discarded (its score is too small to evict anyone).

This is the "pure H2O" mode — NO recency window. For recency-aware retention,
use `StreamingLLMCache` (position-based recency) or `SnapKVCache` (observation-
window scoring with a local fp window).

batch_size=1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo_attn import _force_set_attn_impl, _repeat_head


_NEG_INF = float("-inf")


class H2OCache:
    """Pure cumulative-attention eviction. Fixed `cache_size` slots per kv-head."""

    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int,
                 cache_size: int,
                 num_kv_heads: int = None,
                 device=None, dtype=torch.float32):
        assert batch_size == 1, "v1 supports batch_size=1 only"
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.cache_size = cache_size
        self.device = device
        self.dtype = dtype

        L, B, H_kv, C, D = num_layers, batch_size, self.num_kv_heads, cache_size, head_dim
        self.K = torch.zeros(L, B, H_kv, C, D, device=device, dtype=dtype)
        self.V = torch.zeros(L, B, H_kv, C, D, device=device, dtype=dtype)
        self.pos = torch.full((L, B, H_kv, C), -1, device=device, dtype=torch.long)
        self.score = torch.zeros(L, B, H_kv, C, device=device, dtype=torch.float32)

        self._compressed = [False] * num_layers
        self.next_pos = [0] * num_layers

    def reset(self) -> None:
        self.pos.fill_(-1)
        self.score.zero_()
        self._compressed = [False] * self.num_layers
        self.next_pos = [0] * self.num_layers

    def bytes_per_token(self) -> int:
        # K + V (dtype-sized) + score (fp32) + pos (long).
        return 2 * self.head_dim * self.K.element_size() + 4 + 8

    def bytes_total(self) -> int:
        return (self.num_layers * self.batch_size * self.num_kv_heads
                * self.cache_size * self.bytes_per_token())

    # ------------------------------------------------------------------
    # Prefill compression: top-C by accumulated attention.
    # ------------------------------------------------------------------
    def _compress_prefill(self, layer_idx: int,
                           K_all: torch.Tensor, V_all: torch.Tensor,
                           positions: torch.Tensor,
                           attn_all: torch.Tensor) -> None:
        """Score every prefill token by the sum of attention received from every
        prefill query, then keep the top `cache_size` per (B, H_kv).

        attn_all: [B, H_q, T_pre, T_pre] post-softmax attention from the prefill.
        """
        B, H_kv, T_pre, D = K_all.shape
        n_rep = self.n_rep
        C = self.cache_size

        # Sum across queries -> per-key cumulative score.
        score = attn_all.sum(dim=2).float()                # [B, H_q, T_pre]
        if n_rep > 1:
            score = score.view(B, H_kv, n_rep, T_pre).sum(dim=2)
        # score: [B, H_kv, T_pre]

        if T_pre <= C:
            # Everything fits; just store.
            self.K[layer_idx, :, :, :T_pre, :] = K_all
            self.V[layer_idx, :, :, :T_pre, :] = V_all
            self.pos[layer_idx, :, :, :T_pre] = positions.view(1, 1, T_pre)
            self.score[layer_idx, :, :, :T_pre] = score.to(self.score.dtype)
            self._compressed[layer_idx] = True
            return

        # Top-C per head.
        top_idx = score.topk(C, dim=-1, largest=True).indices  # [B, H_kv, C]
        idx_4d = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        sel_K = K_all.gather(2, idx_4d)
        sel_V = V_all.gather(2, idx_4d)
        sel_pos = positions[top_idx]                            # [B, H_kv, C]
        sel_score = score.gather(2, top_idx)                    # [B, H_kv, C]

        self.K[layer_idx]     = sel_K
        self.V[layer_idx]     = sel_V
        self.pos[layer_idx]   = sel_pos
        self.score[layer_idx] = sel_score.to(self.score.dtype)
        self._compressed[layer_idx] = True

    # ------------------------------------------------------------------
    # Decode: online eviction (replace lowest-scored slot if new token beats it).
    # ------------------------------------------------------------------
    def _decode_ingest(self, layer_idx: int,
                        K_new: torch.Tensor, V_new: torch.Tensor,
                        positions: torch.Tensor,
                        score_new: torch.Tensor) -> None:
        """K_new, V_new: [B, H_kv, T_new, D]. score_new: [B, H_kv, T_new] cumulative
        attention received this step (already aggregated over the n_rep group)."""
        B, H_kv = self.batch_size, self.num_kv_heads
        T_new = K_new.shape[-2]
        D = self.head_dim
        if T_new == 0:
            return

        # Process new tokens one at a time (matches H2O's online behavior; T_new is
        # 1 during decode anyway).
        for t in range(T_new):
            k = K_new[:, :, t:t+1, :]                  # [B, H_kv, 1, D]
            v = V_new[:, :, t:t+1, :]
            p = positions[t:t+1].view(1, 1, 1)         # [1, 1, 1]
            s = score_new[:, :, t:t+1]                 # [B, H_kv, 1]

            # Per-head min slot.
            cur_score = self.score[layer_idx]          # [B, H_kv, C]
            cur_valid = self.pos[layer_idx] >= 0
            masked_score = torch.where(cur_valid, cur_score,
                                       torch.full_like(cur_score, _NEG_INF))
            min_score, min_idx = masked_score.min(dim=-1)   # [B, H_kv]

            # New token's score vs lowest-scored existing slot. Empty slots beat -inf
            # so the first cache_size new tokens fill in automatically.
            should_swap = s.squeeze(-1) > min_score    # [B, H_kv]

            idx_3d = min_idx.unsqueeze(-1)             # [B, H_kv, 1]
            idx_4d_D = min_idx.view(B, H_kv, 1, 1).expand(B, H_kv, 1, D)

            new_pos_at_min   = torch.where(should_swap, p.expand(B, H_kv, 1).squeeze(-1),
                                            self.pos[layer_idx].gather(-1, idx_3d).squeeze(-1)
                                            ).unsqueeze(-1)
            new_score_at_min = torch.where(should_swap, s.squeeze(-1),
                                            cur_score.gather(-1, idx_3d).squeeze(-1)
                                            ).unsqueeze(-1)
            self.pos[layer_idx].scatter_(-1, idx_3d, new_pos_at_min)
            self.score[layer_idx].scatter_(-1, idx_3d, new_score_at_min.to(self.score.dtype))

            swap_4d_D = should_swap.view(B, H_kv, 1, 1).expand(B, H_kv, 1, D)
            existing_K_at_min = self.K[layer_idx].gather(2, idx_4d_D)
            existing_V_at_min = self.V[layer_idx].gather(2, idx_4d_D)
            new_K_at_min = torch.where(swap_4d_D, k, existing_K_at_min)
            new_V_at_min = torch.where(swap_4d_D, v, existing_V_at_min)
            self.K[layer_idx].scatter_(2, idx_4d_D, new_K_at_min)
            self.V[layer_idx].scatter_(2, idx_4d_D, new_V_at_min)

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    def attention(self, layer_idx: int, q: torch.Tensor,
                  k_new: torch.Tensor, v_new: torch.Tensor,
                  scaling: float = None,
                  attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, H_q, T_q, D = q.shape
        T_new = k_new.shape[-2]
        n_rep = self.n_rep
        q_dtype = q.dtype
        device = q.device
        if scaling is None:
            scaling = 1.0 / math.sqrt(D)

        new_positions = torch.arange(self.next_pos[layer_idx],
                                     self.next_pos[layer_idx] + T_new,
                                     device=device, dtype=torch.long)
        compressed = self._compressed[layer_idx]

        if not compressed:
            # PREFILL path. Compute full attention against fresh K/V (no cache yet),
            # then compress to top-cache_size by accumulated attention.
            k_new_q = _repeat_head(k_new, n_rep) if n_rep > 1 else k_new
            v_new_q = _repeat_head(v_new, n_rep) if n_rep > 1 else v_new
            scores = (q @ k_new_q.transpose(-1, -2).to(q_dtype)) * scaling   # [B, H_q, T_q, T_new]
            # Causal among fresh.
            new_pos_for_q = new_positions.view(-1, 1)
            new_pos_for_k = new_positions.view(1, -1)
            future_new = new_pos_for_k > new_pos_for_q
            scores = scores.masked_fill(future_new.view(1, 1, T_q, T_new), _NEG_INF)
            if attn_mask is not None:
                scores = scores + attn_mask[..., :scores.shape[-1]]
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_dtype)
            out = attn @ v_new_q.to(q_dtype)

            self._compress_prefill(layer_idx, k_new, v_new, new_positions,
                                    attn.to(torch.float32))
            self.next_pos[layer_idx] += T_new
            return out

        # DECODE path. Attention against (cache ⊕ fresh K/V).
        K_cache = self.K[layer_idx].to(q_dtype)
        V_cache = self.V[layer_idx].to(q_dtype)
        pos_cache = self.pos[layer_idx]
        if n_rep > 1:
            K_cache = _repeat_head(K_cache, n_rep)
            V_cache = _repeat_head(V_cache, n_rep)
            pos_cache = _repeat_head(pos_cache, n_rep)
        valid_cache = pos_cache >= 0

        cache_scores = (q @ K_cache.transpose(-1, -2)) * scaling             # [B, H_q, T_q, C]
        invalid = ~valid_cache.unsqueeze(2)
        future = pos_cache.unsqueeze(2) > new_positions.view(1, 1, -1, 1)
        cache_scores = cache_scores.masked_fill(invalid | future, _NEG_INF)

        k_new_q = _repeat_head(k_new, n_rep) if n_rep > 1 else k_new
        v_new_q = _repeat_head(v_new, n_rep) if n_rep > 1 else v_new
        score_new = (q @ k_new_q.transpose(-1, -2).to(q_dtype)) * scaling
        new_pos_for_q = new_positions.view(-1, 1)
        new_pos_for_k = new_positions.view(1, -1)
        future_new = new_pos_for_k > new_pos_for_q
        score_new = score_new.masked_fill(future_new.view(1, 1, T_q, T_new), _NEG_INF)

        scores = torch.cat([cache_scores, score_new], dim=-1)
        if attn_mask is not None:
            T_total = scores.shape[-1]
            if attn_mask.shape[-1] < T_total:
                pad = list(attn_mask.shape); pad[-1] = T_total - attn_mask.shape[-1]
                left_pad = torch.zeros(pad, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([left_pad, attn_mask], dim=-1)
            scores = scores + attn_mask[..., :T_total]

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_dtype)
        V_full = torch.cat([V_cache, v_new_q.to(q_dtype)], dim=-2)
        out = attn @ V_full

        # ----- update cumulative scores on the cache portion -----
        C = self.cache_size
        attn_cache = attn[..., :C].to(torch.float32)                 # [B, H_q, T_q, C]
        # Sum over queries, then sum over the n_rep group -> [B, H_kv, C].
        received = attn_cache.sum(dim=2)                             # [B, H_q, C]
        if n_rep > 1:
            received = received.view(B, self.num_kv_heads, n_rep, C).sum(dim=2)
        self.score[layer_idx] = self.score[layer_idx] + received.to(self.score.dtype)

        # ----- ingest fresh tokens with their initial score = attention received -----
        attn_new = attn[..., C:].to(torch.float32)                   # [B, H_q, T_q, T_new]
        score_init = attn_new.sum(dim=2)                             # [B, H_q, T_new]
        if n_rep > 1:
            score_init = score_init.view(B, self.num_kv_heads, n_rep, T_new).sum(dim=2)
        self._decode_ingest(layer_idx, k_new, v_new, new_positions, score_init)
        self.next_pos[layer_idx] += T_new
        return out


# ======================================================================================
# HF dispatcher integration
# ======================================================================================

def h2o_attn_function(module, query, key, value, attention_mask=None,
                     scaling=None, dropout=0.0, **kwargs):
    cache: H2OCache = module._h2o_cache
    layer_idx: int = module.layer_idx
    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out, None


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["h2o"] = h2o_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def install_h2o(model: nn.Module, cache: H2OCache) -> int:
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required for install_h2o")
    if hasattr(model, "config"):
        _force_set_attn_impl(model.config, "h2o")
    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config, "h2o")
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._h2o_cache = cache
            n += 1
    return n
