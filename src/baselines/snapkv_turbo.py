"""SnapKV + TurboQuant: the naive composition pipeline.

The two-stage strawman that KVCascade aims to beat:
  1. SnapKV's prefill-time observation-window scoring selects top heavy hitters.
  2. The selected K, V tensors are quantized once using PolarQuant (TurboQuant's
     MSE variant) and stored as dequantized fp values for attention.
  3. A small trailing window of `window` tokens stays at full fp precision.
  4. Decode tokens enter the local window FIFO; the quantized heavy set is
     FROZEN — no further selection, no demotion, no further quantization.

This composition is "naive" because there is no demote-on-loss feedback between
the two stages: SnapKV picks once at prefill end, TurboQuant quantizes once,
and from then on the cache is static.

Iso-byte accounting:
  bytes_total = H_kv * (heavy_cap * turbo_slot_bytes + window * fp_slot_bytes)

Quant mode is MSE-only (PolarQuant) for now: the Prod variant's JL sketch
contributes to inner-product estimation but doesn't yield a clean fp K
reconstruction, which we'd need to store for the no-demote path. KVCascade
itself handles Prod natively via TurboBuffer.

batch_size=1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from polar_quant import PolarQuant
from turbo_attn import _force_set_attn_impl, _repeat_head


_NEG_INF = float("-inf")


def _polar_qdq(pq: PolarQuant, x: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize using PolarQuant. x: [..., D]. Returns same shape fp."""
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1]).to(pq.dtype)
    norms, idx = pq.encode(flat)
    return pq.decode(norms, idx).reshape(orig_shape).to(x.dtype)


class SnapKVTurboCache:
    """SnapKV selection + PolarQuant quantization on the heavy-hitter region."""

    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int,
                 heavy_capacity: int,
                 window: int = 32,
                 k_bits: int = 6,
                 v_bits: int = 2,
                 pool_kernel: int = 7,
                 pool: str = "maxpool",
                 num_kv_heads: int = None,
                 seed: int = 0,
                 device=None, dtype=torch.float32):
        assert batch_size == 1, "v1 supports batch_size=1 only"
        assert pool in ("maxpool", "avgpool")
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.heavy_capacity = heavy_capacity
        self.window = window
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.pool_kernel = pool_kernel
        self.pool = pool
        self.total = heavy_capacity + window
        self.device = device
        self.dtype = dtype

        L, B, H_kv, T, D = (num_layers, batch_size, self.num_kv_heads,
                             self.total, head_dim)
        # Dequantized-fp storage for the heavy region; fp storage for the window region.
        # We use a single tensor for simplicity — the byte budget is computed
        # analytically from heavy_capacity * turbo_slot_bytes + window * fp_slot_bytes.
        self.K = torch.zeros(L, B, H_kv, T, D, device=device, dtype=dtype)
        self.V = torch.zeros(L, B, H_kv, T, D, device=device, dtype=dtype)
        self.pos = torch.full((L, B, H_kv, T), -1, device=device, dtype=torch.long)

        # PolarQuant instances — shared across all (layer, head) since the rotation
        # depends only on head_dim.
        pq_dtype = torch.float32  # quant codebook lives in fp32 for stability.
        self.pq_k = PolarQuant(bits=k_bits, dim=head_dim, seed=seed,
                                device=device, dtype=pq_dtype)
        self.pq_v = PolarQuant(bits=v_bits, dim=head_dim, seed=seed + 1,
                                device=device, dtype=pq_dtype)

        self._compressed    = [False] * num_layers
        self._window_cursor = [0] * num_layers
        self.next_pos       = [0] * num_layers

    def reset(self) -> None:
        self.pos.fill_(-1)
        self._compressed    = [False] * self.num_layers
        self._window_cursor = [0] * self.num_layers
        self.next_pos       = [0] * self.num_layers

    @staticmethod
    def turbo_slot_bytes_mse(head_dim: int, k_bits: int, v_bits: int, fp_size: int) -> int:
        """PolarQuant (MSE) slot bytes: 2 fp norms + packed K bits + packed V bits."""
        return (2 * fp_size                            # k_norm, v_norm
                + (head_dim * k_bits + 7) // 8         # k_idx packed
                + (head_dim * v_bits + 7) // 8)        # v_idx packed

    def bytes_total(self) -> int:
        fp_size = self.K.element_size()
        slot_q  = self.turbo_slot_bytes_mse(self.head_dim, self.k_bits, self.v_bits, fp_size)
        slot_fp = 2 * self.head_dim * fp_size + 8     # K + V + long pos
        # Position-long overhead also applies to quantized slots.
        slot_q_w_pos = slot_q + 8
        per_lh = self.heavy_capacity * slot_q_w_pos + self.window * slot_fp
        return self.num_layers * self.batch_size * self.num_kv_heads * per_lh

    # ------------------------------------------------------------------
    # Prefill compression: SnapKV-style top-K selection + PolarQuant storage.
    # ------------------------------------------------------------------
    def _compress_prefill(self, layer_idx: int,
                           K_all: torch.Tensor, V_all: torch.Tensor,
                           positions: torch.Tensor,
                           attn_window: torch.Tensor) -> None:
        B, H_kv, T_pre, D = K_all.shape
        n_rep = self.n_rep
        W = self.window
        C = self.heavy_capacity

        T_pool = T_pre - W
        if T_pool <= 0:
            # Short prefill: no heavy hitters, just stash recent tokens fp.
            n_recent = min(T_pre, W)
            self.K  [layer_idx, :, :, C:C + n_recent, :] = K_all[:, :, -n_recent:, :]
            self.V  [layer_idx, :, :, C:C + n_recent, :] = V_all[:, :, -n_recent:, :]
            self.pos[layer_idx, :, :, C:C + n_recent]    = positions[-n_recent:].view(1, 1, -1)
            self._window_cursor[layer_idx] = n_recent % W if W > 0 else 0
            self._compressed[layer_idx] = True
            return

        sig = attn_window[:, :, :, :T_pool].float().mean(dim=2)
        if self.pool_kernel > 1:
            pad = self.pool_kernel // 2
            if self.pool == "maxpool":
                sig = F.max_pool1d(sig, kernel_size=self.pool_kernel, padding=pad, stride=1)
            else:
                sig = F.avg_pool1d(sig, kernel_size=self.pool_kernel, padding=pad, stride=1)
        if n_rep > 1:
            sig = sig.view(B, H_kv, n_rep, -1).max(dim=2).values

        C_take = min(C, T_pool)
        top_idx = sig.topk(C_take, dim=-1, largest=True).indices
        idx_4d = top_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        sel_K = K_all.gather(2, idx_4d)
        sel_V = V_all.gather(2, idx_4d)
        sel_pos = positions[top_idx]

        # Quantize the selected K, V (PolarQuant simulate quant-dequant).
        sel_K_q = _polar_qdq(self.pq_k, sel_K)
        sel_V_q = _polar_qdq(self.pq_v, sel_V)

        self.K  [layer_idx, :, :, :C_take, :] = sel_K_q
        self.V  [layer_idx, :, :, :C_take, :] = sel_V_q
        self.pos[layer_idx, :, :, :C_take]    = sel_pos

        # Window slots: last `window` prefill tokens, fp.
        self.K  [layer_idx, :, :, C:C + W, :] = K_all[:, :, -W:, :]
        self.V  [layer_idx, :, :, C:C + W, :] = V_all[:, :, -W:, :]
        self.pos[layer_idx, :, :, C:C + W]    = positions[-W:].view(1, 1, W)
        self._window_cursor[layer_idx] = 0
        self._compressed[layer_idx] = True

    def _decode_append(self, layer_idx: int,
                        K_new: torch.Tensor, V_new: torch.Tensor,
                        positions: torch.Tensor) -> None:
        T_new = K_new.shape[-2]
        if T_new == 0:
            return
        C = self.heavy_capacity
        W = self.window
        cursor = self._window_cursor[layer_idx]
        device = K_new.device
        local = (cursor + torch.arange(T_new, device=device, dtype=torch.long)) % W
        slot = C + local
        self.K  [layer_idx, :, :, slot, :] = K_new
        self.V  [layer_idx, :, :, slot, :] = V_new
        self.pos[layer_idx, :, :, slot]    = positions.view(1, 1, -1)
        self._window_cursor[layer_idx] = (cursor + T_new) % W

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

        compressed = self._compressed[layer_idx]
        new_positions = torch.arange(self.next_pos[layer_idx],
                                     self.next_pos[layer_idx] + T_new,
                                     device=device, dtype=torch.long)

        if not compressed:
            # PREFILL path. Dense attention against fresh K/V, then compress.
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

def snapkv_turbo_attn_function(module, query, key, value, attention_mask=None,
                                scaling=None, dropout=0.0, **kwargs):
    cache: SnapKVTurboCache = module._snapkv_turbo_cache
    layer_idx: int = module.layer_idx
    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out, None


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["snapkv_turbo"] = snapkv_turbo_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def install_snapkv_turbo(model: nn.Module, cache: SnapKVTurboCache) -> int:
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required for install_snapkv_turbo")
    if hasattr(model, "config"):
        _force_set_attn_impl(model.config, "snapkv_turbo")
    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config, "snapkv_turbo")
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._snapkv_turbo_cache = cache
            n += 1
    return n
