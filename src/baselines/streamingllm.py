"""StreamingLLM: first-N attention sinks + last-K sliding recency window.

Reference: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
(ICLR 2024). https://arxiv.org/abs/2309.17453

Mechanics:
  - Keep the first `sink_count` tokens of the sequence forever (the "attention
    sinks"), regardless of attention received.
  - Keep the last `recent_count` tokens in a FIFO window.
  - Everything in between is evicted as it falls past the recent window.

No scoring; retention is purely position-based. batch_size=1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo_attn import _force_set_attn_impl, _repeat_head


_NEG_INF = float("-inf")


class StreamingLLMCache:
    """First `sink_count` tokens (forever) + last `recent_count` tokens (FIFO)."""

    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int,
                 sink_count: int = 4,
                 recent_count: int = 1024,
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
        self.sink_count = sink_count
        self.recent_count = recent_count
        self.total = sink_count + recent_count
        self.device = device
        self.dtype = dtype

        L, B, H_kv, T, D = (num_layers, batch_size, self.num_kv_heads,
                             self.total, head_dim)
        self.K = torch.zeros(L, B, H_kv, T, D, device=device, dtype=dtype)
        self.V = torch.zeros(L, B, H_kv, T, D, device=device, dtype=dtype)
        self.pos = torch.full((L, B, H_kv, T), -1, device=device, dtype=torch.long)

        self._recent_cursor = [0] * num_layers
        self._sinks_filled  = [0] * num_layers
        self.next_pos       = [0] * num_layers

    def reset(self) -> None:
        self.pos.fill_(-1)
        self._recent_cursor = [0] * self.num_layers
        self._sinks_filled  = [0] * self.num_layers
        self.next_pos       = [0] * self.num_layers

    def bytes_per_token(self) -> int:
        return 2 * self.head_dim * self.K.element_size() + 8  # +long for pos

    def bytes_total(self) -> int:
        return (self.num_layers * self.batch_size * self.num_kv_heads
                * self.total * self.bytes_per_token())

    # ------------------------------------------------------------------
    # Ingest: fill sinks (if empty) + append to recent ring (FIFO).
    # ------------------------------------------------------------------
    def _ingest(self, layer_idx: int,
                K_new: torch.Tensor, V_new: torch.Tensor,
                positions: torch.Tensor) -> None:
        T_new = K_new.shape[-2]
        if T_new == 0:
            return
        N = self.sink_count
        R = self.recent_count

        idx_in = 0
        # 1. Fill empty sink slots.
        filled = self._sinks_filled[layer_idx]
        if filled < N and T_new > 0:
            n_take = min(N - filled, T_new)
            self.K  [layer_idx, :, :, filled:filled + n_take, :] = K_new[:, :, idx_in:idx_in + n_take, :]
            self.V  [layer_idx, :, :, filled:filled + n_take, :] = V_new[:, :, idx_in:idx_in + n_take, :]
            self.pos[layer_idx, :, :, filled:filled + n_take]    = positions[idx_in:idx_in + n_take].view(1, 1, -1)
            self._sinks_filled[layer_idx] = filled + n_take
            idx_in += n_take

        # 2. Remaining tokens into recent FIFO.
        remaining = T_new - idx_in
        if remaining > 0:
            cursor = self._recent_cursor[layer_idx]
            device = K_new.device
            local = (cursor + torch.arange(remaining, device=device, dtype=torch.long)) % R
            slot = N + local
            self.K  [layer_idx, :, :, slot, :] = K_new[:, :, idx_in:idx_in + remaining, :]
            self.V  [layer_idx, :, :, slot, :] = V_new[:, :, idx_in:idx_in + remaining, :]
            self.pos[layer_idx, :, :, slot]    = positions[idx_in:idx_in + remaining].view(1, 1, -1)
            self._recent_cursor[layer_idx] = (cursor + remaining) % R

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

        K_cache = self.K[layer_idx].to(q_dtype)
        V_cache = self.V[layer_idx].to(q_dtype)
        pos_cache = self.pos[layer_idx]
        if n_rep > 1:
            K_cache = _repeat_head(K_cache, n_rep)
            V_cache = _repeat_head(V_cache, n_rep)
            pos_cache = _repeat_head(pos_cache, n_rep)
        valid_cache = pos_cache >= 0

        new_positions = torch.arange(self.next_pos[layer_idx],
                                     self.next_pos[layer_idx] + T_new,
                                     device=device, dtype=torch.long)
        cache_scores = (q @ K_cache.transpose(-1, -2)) * scaling
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

        # Ingest after computing attention against current state.
        self._ingest(layer_idx, k_new, v_new, new_positions)
        self.next_pos[layer_idx] += T_new
        return out


# ======================================================================================
# HF dispatcher integration
# ======================================================================================

def streamingllm_attn_function(module, query, key, value, attention_mask=None,
                                 scaling=None, dropout=0.0, **kwargs):
    cache: StreamingLLMCache = module._streamingllm_cache
    layer_idx: int = module.layer_idx
    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out, None


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["streamingllm"] = streamingllm_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def install_streamingllm(model: nn.Module, cache: StreamingLLMCache) -> int:
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required for install_streamingllm")
    if hasattr(model, "config"):
        _force_set_attn_impl(model.config, "streamingllm")
    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config, "streamingllm")
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._streamingllm_cache = cache
            n += 1
    return n
