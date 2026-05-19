"""Ada-SnapKV: per-head adaptive heavy-hitter budget on top of SnapKV scoring.

Reference: Feng et al., "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget
Allocation for Efficient LLM Inference" (2024). https://arxiv.org/abs/2407.11550

Mechanics:
  - Use SnapKV's prefill-time observation-window scoring (mean of last `window`
    queries' attention, max-pooled along time).
  - For each (batch, layer), the total heavy-hitter budget is
    `H_kv * fp_capacity` tokens — same as SnapKV at iso-byte.
  - Pool all per-head scores globally and select the top B_layer; each head's
    effective heavy-cap = how many of its tokens make the global cut.
  - Storage is padded at `safety_factor * fp_capacity` per head so heavy
    individual budgets up to `safety_factor * fp_capacity` are representable.
    Per-head caps that exceed this ceiling are clamped and the overflow
    redistributed to heads with slack — keeps the layer sum exactly = budget.
  - Local recency window is uniform per head (size `window`), same as SnapKV.
  - Frozen post-prefill: decode tokens enter the local window only.

Iso-byte vs SnapKV: identical total slots (`H_kv * (fp_capacity + window)`),
just redistributed across heads.

batch_size=1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo_attn import _force_set_attn_impl, _repeat_head


_NEG_INF = float("-inf")


class AdaSnapKVCache:
    """Per-head adaptive heavy-hitter budget over SnapKV's obs-window scoring."""

    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int,
                 fp_capacity: int = 256,
                 window: int = 32,
                 pool_kernel: int = 7,
                 pool: str = "maxpool",
                 safety_factor: int = 2,
                 num_kv_heads: int = None,
                 device=None, dtype=torch.float32):
        assert batch_size == 1, "v1 supports batch_size=1 only"
        assert pool in ("maxpool", "avgpool")
        assert safety_factor >= 1
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.fp_capacity = fp_capacity
        self.window = window
        self.pool_kernel = pool_kernel
        self.pool = pool
        self.safety_factor = safety_factor
        self.max_heavy = safety_factor * fp_capacity
        self.total = self.max_heavy + window
        self.device = device
        self.dtype = dtype

        L, B, H_kv, T, D = (num_layers, batch_size, self.num_kv_heads,
                             self.total, head_dim)
        self.K = torch.zeros(L, B, H_kv, T, D, device=device, dtype=dtype)
        self.V = torch.zeros(L, B, H_kv, T, D, device=device, dtype=dtype)
        self.pos = torch.full((L, B, H_kv, T), -1, device=device, dtype=torch.long)
        # Per-(layer, batch, kv-head) actual heavy-hitter count after compress.
        self.eff_heavy = torch.zeros(L, B, H_kv, device=device, dtype=torch.long)

        self._compressed    = [False] * num_layers
        self._window_cursor = [0] * num_layers
        self.next_pos       = [0] * num_layers

    def reset(self) -> None:
        self.pos.fill_(-1)
        self.eff_heavy.zero_()
        self._compressed    = [False] * self.num_layers
        self._window_cursor = [0] * self.num_layers
        self.next_pos       = [0] * self.num_layers

    def bytes_per_token(self) -> int:
        return 2 * self.head_dim * self.K.element_size() + 8

    def bytes_total(self) -> int:
        """Iso-byte report: H_kv * (avg heavy_cap + window) * slot_bytes.
        Padding to safety_factor * fp_capacity is implementation overhead and
        not counted, matching the AdaKV paper's accounting convention."""
        per_head_slots = self.fp_capacity + self.window
        return (self.num_layers * self.batch_size * self.num_kv_heads
                * per_head_slots * self.bytes_per_token())

    # ------------------------------------------------------------------
    # Per-head budget allocation via global top-K + ceiling redistribution.
    # ------------------------------------------------------------------
    def _allocate(self, score_layer: torch.Tensor) -> torch.Tensor:
        """score_layer: [B, H_kv, T_pool]. Returns eff_heavy: [B, H_kv] (long).
        Sum across heads = min(B_layer, H_kv * T_pool); each head <= max_heavy."""
        B, H_kv, T_pool = score_layer.shape
        budget = H_kv * self.fp_capacity
        budget = min(budget, H_kv * T_pool)
        # Global top-K per batch sample.
        flat = score_layer.reshape(B, H_kv * T_pool)
        K_sel = min(budget, flat.shape[1])
        top_idx = flat.topk(K_sel, dim=-1).indices                   # [B, K_sel]
        head_of_idx = top_idx // T_pool                              # [B, K_sel]
        eff = F.one_hot(head_of_idx, num_classes=H_kv).sum(dim=1).long()  # [B, H_kv]

        # Cap at max_heavy per head, redistribute overflow to heads with slack.
        max_h = self.max_heavy
        if (eff > max_h).any():
            for b in range(B):
                over = (eff[b] - max_h).clamp_min(0).sum().item()
                eff[b].clamp_(max=max_h)
                while over > 0:
                    slack = max_h - eff[b]
                    n_with_slack = (slack > 0).sum().item()
                    if n_with_slack == 0:
                        break
                    give = min(over, n_with_slack)
                    cand = torch.where(slack > 0)[0][:give]
                    eff[b, cand] += 1
                    over -= give
        return eff

    # ------------------------------------------------------------------
    # Prefill compression: SnapKV scoring + adaptive per-head selection.
    # ------------------------------------------------------------------
    def _compress_prefill(self, layer_idx: int,
                           K_all: torch.Tensor, V_all: torch.Tensor,
                           positions: torch.Tensor,
                           attn_window: torch.Tensor) -> None:
        """attn_window: [B, H_q, window, T_pre]. Score selectable region
        [0, T_pre - window) per SnapKV, then adaptively allocate per-head budget."""
        B, H_kv, T_pre, D = K_all.shape
        n_rep = self.n_rep
        W = self.window

        T_pool = T_pre - W
        if T_pool <= 0:
            # Prefill too short: just stash recent tokens, no heavy hitters.
            n_recent = min(T_pre, W)
            self.K  [layer_idx, :, :, self.max_heavy:self.max_heavy + n_recent, :] = K_all[:, :, -n_recent:, :]
            self.V  [layer_idx, :, :, self.max_heavy:self.max_heavy + n_recent, :] = V_all[:, :, -n_recent:, :]
            self.pos[layer_idx, :, :, self.max_heavy:self.max_heavy + n_recent]    = positions[-n_recent:].view(1, 1, -1)
            self._window_cursor[layer_idx] = n_recent % W
            self._compressed[layer_idx] = True
            return

        sig = attn_window[:, :, :, :T_pool].float().mean(dim=2)        # [B, H_q, T_pool]
        if self.pool_kernel > 1:
            pad = self.pool_kernel // 2
            if self.pool == "maxpool":
                sig = F.max_pool1d(sig, kernel_size=self.pool_kernel, padding=pad, stride=1)
            else:
                sig = F.avg_pool1d(sig, kernel_size=self.pool_kernel, padding=pad, stride=1)
        if n_rep > 1:
            sig = sig.view(B, H_kv, n_rep, -1).max(dim=2).values
        # sig: [B, H_kv, T_pool]

        eff = self._allocate(sig)                                       # [B, H_kv]
        self.eff_heavy[layer_idx] = eff

        # Per-head top-eff[h] selection. Loop is fine — H_kv is small (<=32 typical).
        for b in range(B):
            for h in range(H_kv):
                cap_h = int(eff[b, h].item())
                if cap_h <= 0:
                    continue
                score_h = sig[b, h]                                     # [T_pool]
                idx_h = score_h.topk(cap_h, largest=True).indices       # [cap_h]
                self.K  [layer_idx, b, h, :cap_h, :] = K_all[b, h, idx_h, :]
                self.V  [layer_idx, b, h, :cap_h, :] = V_all[b, h, idx_h, :]
                self.pos[layer_idx, b, h, :cap_h]    = positions[idx_h]

        # Window slots: last `window` prefill tokens, in order, same for all heads.
        mh = self.max_heavy
        self.K  [layer_idx, :, :, mh:mh + W, :] = K_all[:, :, -W:, :]
        self.V  [layer_idx, :, :, mh:mh + W, :] = V_all[:, :, -W:, :]
        self.pos[layer_idx, :, :, mh:mh + W]    = positions[-W:].view(1, 1, W)
        self._window_cursor[layer_idx] = 0
        self._compressed[layer_idx] = True

    def _decode_append(self, layer_idx: int,
                        K_new: torch.Tensor, V_new: torch.Tensor,
                        positions: torch.Tensor) -> None:
        """FIFO write into the window region (heavy region is frozen)."""
        T_new = K_new.shape[-2]
        if T_new == 0:
            return
        W = self.window
        mh = self.max_heavy
        cursor = self._window_cursor[layer_idx]
        device = K_new.device
        local = (cursor + torch.arange(T_new, device=device, dtype=torch.long)) % W
        slot = mh + local
        self.K  [layer_idx, :, :, slot, :] = K_new
        self.V  [layer_idx, :, :, slot, :] = V_new
        self.pos[layer_idx, :, :, slot]    = positions.view(1, 1, -1)
        self._window_cursor[layer_idx] = (cursor + T_new) % W

    # ------------------------------------------------------------------
    # Attention (same shape as SnapKV; the padding is masked out via pos==-1).
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

        compressed = self._compressed[layer_idx]
        new_positions = torch.arange(self.next_pos[layer_idx],
                                     self.next_pos[layer_idx] + T_new,
                                     device=device, dtype=torch.long)

        if not compressed:
            # PREFILL path.
            k_new_q = _repeat_head(k_new, n_rep) if n_rep > 1 else k_new
            v_new_q = _repeat_head(v_new, n_rep) if n_rep > 1 else v_new
            scores = (q @ k_new_q.transpose(-1, -2).to(q_dtype)) * scaling
            new_pos_for_q = new_positions.view(-1, 1)
            new_pos_for_k = new_positions.view(1, -1)
            future_new = new_pos_for_k > new_pos_for_q
            scores = scores.masked_fill(future_new.view(1, 1, T_q, T_new), _NEG_INF)
            if attn_mask is not None:
                scores = scores + attn_mask[..., :scores.shape[-1]]
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_dtype)
            out = attn @ v_new_q.to(q_dtype)

            W = self.window
            if T_q >= W:
                attn_window = attn[:, :, -W:, :].to(torch.float32)
                self._compress_prefill(layer_idx, k_new, v_new, new_positions, attn_window)
            else:
                self._compress_prefill(layer_idx, k_new, v_new, new_positions,
                                        attn[:, :, :, :].to(torch.float32))
            self.next_pos[layer_idx] += T_new
            return out

        # DECODE path.
        K_cache = self.K[layer_idx].to(q_dtype)
        V_cache = self.V[layer_idx].to(q_dtype)
        pos_cache = self.pos[layer_idx]
        if n_rep > 1:
            K_cache = _repeat_head(K_cache, n_rep)
            V_cache = _repeat_head(V_cache, n_rep)
            pos_cache = _repeat_head(pos_cache, n_rep)
        valid_cache = pos_cache >= 0

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

        self._decode_append(layer_idx, k_new, v_new, new_positions)
        self.next_pos[layer_idx] += T_new
        return out


# ======================================================================================
# HF dispatcher integration
# ======================================================================================

def ada_snapkv_attn_function(module, query, key, value, attention_mask=None,
                              scaling=None, dropout=0.0, **kwargs):
    cache: AdaSnapKVCache = module._ada_snapkv_cache
    layer_idx: int = module.layer_idx
    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out, None


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["ada_snapkv"] = ada_snapkv_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def install_ada_snapkv(model: nn.Module, cache: AdaSnapKVCache) -> int:
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required for install_ada_snapkv")
    if hasattr(model, "config"):
        _force_set_attn_impl(model.config, "ada_snapkv")
    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config, "ada_snapkv")
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._ada_snapkv_cache = cache
            n += 1
    return n
