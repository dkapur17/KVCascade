"""TurboAttention via HF's ALL_ATTENTION_FUNCTIONS dispatcher.

Replaces only the SDPA step. Q/K/V come in already RoPE'd, QK-normed, GQA-grouped
(we expand K/V heads here), so RoPE / QK-norm / sliding window / etc. flow through
unchanged. Output goes back into the model's o_proj path unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from polar_quant import PolarQuant
from turbo_quant import QuantizedKV, TurboQuant


# --------------------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------------------

def pack_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack N integer values (each in [0, 2**bits)) along the last dim.
    x: [..., N] -> [..., ceil(N*bits/8)] uint8."""
    if bits == 8:
        return x.to(torch.uint8)
    *prefix, N = x.shape
    n_bytes = (N * bits + 7) // 8
    if x.numel() == 0:
        # 0-element input (e.g. some prefix dim is 0 — empty tier buffers).
        return torch.zeros(*prefix, n_bytes, dtype=torch.uint8, device=x.device)
    x = x.long()
    bit_idx = torch.arange(bits, device=x.device, dtype=torch.long)
    bits_t = (x.unsqueeze(-1) >> bit_idx) & 1                           # [..., N, bits]
    flat = bits_t.reshape(*prefix, N * bits)
    pad = (-flat.shape[-1]) % 8
    if pad:
        flat = F.pad(flat, (0, pad))
    # Explicit n_bytes avoids the `-1` ambiguity when total elements is 0.
    flat = flat.reshape(*prefix, n_bytes, 8)
    weights = 1 << torch.arange(8, device=x.device, dtype=torch.long)
    return (flat * weights).sum(dim=-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, bits: int, N: int) -> torch.Tensor:
    """Reverse pack_bits. packed:[..., n_bytes] uint8 -> [..., N] uint8."""
    if bits == 8:
        return packed[..., :N]
    *prefix, n_bytes = packed.shape
    bit_idx = torch.arange(8, device=packed.device, dtype=torch.long)
    flat = (packed.long().unsqueeze(-1) >> bit_idx) & 1                 # [..., n_bytes, 8]
    flat = flat.reshape(*prefix, n_bytes * 8)[..., : N * bits]
    flat = flat.reshape(*prefix, N, bits)
    weights = 1 << torch.arange(bits, device=packed.device, dtype=torch.long)
    return (flat * weights).sum(dim=-1).to(torch.uint8)


# --------------------------------------------------------------------------------------
# KV cache
# --------------------------------------------------------------------------------------

def _repeat_head(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand a tensor along dim=1 (head dim) by n_rep. Works for any trailing shape."""
    if n_rep == 1:
        return x
    B, H, *rest = x.shape
    return (x.unsqueeze(2)
              .expand(B, H, n_rep, *rest)
              .reshape(B, H * n_rep, *rest))


class TurboQuantKVCache:
    """KV cache with bit-packed TurboQuant K and PolarQuant V storage.

    Storage is at **kv-head** granularity (so GQA savings are preserved). Attention
    expansion to query-head count happens at compute time.

    Per-layer layout (`H_kv` = num_kv_heads):
        k_norm           : [L, B, H_kv, T]                     fp
        k_idx_packed     : [L, B, H_kv, T, ceil(D*(bits-1)/8)] uint8
        k_resnorm        : [L, B, H_kv, T]                     fp
        k_ressign_packed : [L, B, H_kv, T, ceil(m/8)]          uint8
        v_norm           : [L, B, H_kv, T]                     fp
        v_idx_packed     : [L, B, H_kv, T, ceil(D*bits/8)]     uint8
    """

    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int,
                 bits: int = 4,
                 k_bits: int | None = None,
                 v_bits: int | None = None,
                 m: int | None = None,
                 num_kv_heads: int | None = None,
                 max_seq_len: int | None = None,
                 seed: int = 0,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32):
        """K uses TurboQuant with `k_bits` total budget per coordinate (k_bits-1 for the
        Lloyd-Max codebook + 1 for the JL residual sketch, with m JL projections).
        V uses PolarQuant with `v_bits` per coordinate.

        K budget should be conservative (errors feed into softmax). V can be much smaller
        (errors average out through attn @ V, and Lloyd-Max centroids are unbiased).
        Pass either `bits` (sets both) or `k_bits` / `v_bits` to override individually.

        `max_seq_len` is optional: if provided, buffers are pre-allocated to that size
        (and overflow is an error). If None, buffers grow on demand (capacity doubles
        on overflow).
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads                  # query-head count
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len              # optional hard cap

        # User-facing total budgets per coordinate.
        self.k_total_bits = k_bits if k_bits is not None else bits
        self.v_total_bits = v_bits if v_bits is not None else bits
        # Internal storage bits: K uses (total - 1) for MSE indices (1 bit goes to JL sketch),
        # V uses all bits for PolarQuant.
        self.k_bits = self.k_total_bits - 1
        self.v_bits = self.v_total_bits
        self.m = m if m is not None else head_dim
        self.device = device
        self.dtype = dtype

        # Pre-compute trailing storage sizes (so bytes_per_token is buffer-independent).
        self._bytes_k_idx  = (head_dim * self.k_bits + 7) // 8
        self._bytes_v_idx  = (head_dim * self.v_bits + 7) // 8
        self._bytes_k_sign = (self.m + 7) // 8
        self._fp_bytes = torch.empty((), dtype=dtype).element_size()

        self.k_quantizers = [
            TurboQuant(self.k_total_bits, head_dim, self.m, seed=seed + 2 * l,
                       device=device, dtype=dtype)
            for l in range(num_layers)
        ]
        self.v_quantizers = [
            PolarQuant(self.v_total_bits, head_dim, seed=seed + 2 * l + 1,
                       device=device, dtype=dtype)
            for l in range(num_layers)
        ]

        # Buffers allocated lazily by _grow.
        self.k_norm           = None
        self.k_idx_packed     = None
        self.k_resnorm        = None
        self.k_ressign_packed = None
        self.v_norm           = None
        self.v_idx_packed     = None
        self._capacity = 0
        self.cur_len = [0] * num_layers

        if max_seq_len is not None:
            self._grow(max_seq_len)

    def _grow(self, target: int) -> None:
        """Ensure buffers can hold at least `target` tokens. Doubles capacity on overflow."""
        if target <= self._capacity:
            return
        if self.max_seq_len is not None and target > self.max_seq_len:
            raise AssertionError(f"exceeded max_seq_len ({target} > {self.max_seq_len})")
        new_cap = max(target, max(self._capacity * 2, 16))
        if self.max_seq_len is not None:
            new_cap = min(new_cap, self.max_seq_len)

        L, B, H_kv = self.num_layers, self.batch_size, self.num_kv_heads
        fields = [
            ("k_norm",           self.dtype,      ()),
            ("k_idx_packed",     torch.uint8,     (self._bytes_k_idx,)),
            ("k_resnorm",        self.dtype,      ()),
            ("k_ressign_packed", torch.uint8,     (self._bytes_k_sign,)),
            ("v_norm",           self.dtype,      ()),
            ("v_idx_packed",     torch.uint8,     (self._bytes_v_idx,)),
        ]
        for name, dtype, trailing in fields:
            new = torch.zeros(L, B, H_kv, new_cap, *trailing, device=self.device, dtype=dtype)
            old = getattr(self, name)
            if old is not None and self._capacity > 0:
                new[:, :, :, : self._capacity] = old
            setattr(self, name, new)
        self._capacity = new_cap

    def reset(self) -> None:
        """Reset write cursors. Keeps allocated capacity so subsequent fills are allocation-free."""
        self.cur_len = [0] * self.num_layers

    def bytes_per_token(self) -> int:
        """Per-(layer, kv-head, token) storage cost in bytes."""
        return (
            2 * self._fp_bytes                      # k_norm + k_resnorm
            + self._bytes_k_idx
            + self._bytes_k_sign
            + self._fp_bytes                        # v_norm
            + self._bytes_v_idx
        )

    def bytes_total(self) -> int:
        """Total currently-allocated bytes (based on current capacity, not cur_len)."""
        return (self.num_layers * self.batch_size * self.num_kv_heads
                * self._capacity * self.bytes_per_token())

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> int:
        """k_new, v_new: [B, H, T_new, D]. Returns total seq len for the layer."""
        T_new = k_new.shape[-2]
        s, e = self.cur_len[layer_idx], self.cur_len[layer_idx] + T_new
        self._grow(e)

        kq = self.k_quantizers[layer_idx].quantize(k_new)
        self.k_norm[layer_idx, :, :, s:e]    = kq.x_norm
        self.k_resnorm[layer_idx, :, :, s:e] = kq.res_norm
        self.k_idx_packed[layer_idx, :, :, s:e]     = pack_bits(kq.x_indices, self.k_bits)
        sign01 = (kq.res_signs.long() + 1) >> 1                         # {-1,+1} -> {0,1}
        self.k_ressign_packed[layer_idx, :, :, s:e] = pack_bits(sign01, 1)

        v_norm, v_idx = self.v_quantizers[layer_idx].encode(v_new)
        self.v_norm[layer_idx, :, :, s:e]       = v_norm
        self.v_idx_packed[layer_idx, :, :, s:e] = pack_bits(v_idx, self.v_bits)

        self.cur_len[layer_idx] = e
        return e

    def _key_view(self, layer_idx: int, T: int) -> QuantizedKV:
        """Unpack K view at kv-head granularity and expand to query-head count."""
        k_idx = unpack_bits(self.k_idx_packed[layer_idx, :, :, :T],
                            self.k_bits, self.head_dim)              # [B, H_kv, T, D]
        sign01 = unpack_bits(self.k_ressign_packed[layer_idx, :, :, :T], 1, self.m)
        signs = sign01.to(torch.int8) * 2 - 1                        # [B, H_kv, T, m]
        x_norm   = self.k_norm[layer_idx, :, :, :T]                  # [B, H_kv, T]
        res_norm = self.k_resnorm[layer_idx, :, :, :T]               # [B, H_kv, T]

        if self.n_rep > 1:
            x_norm   = _repeat_head(x_norm,   self.n_rep)
            k_idx    = _repeat_head(k_idx,    self.n_rep)
            res_norm = _repeat_head(res_norm, self.n_rep)
            signs    = _repeat_head(signs,    self.n_rep)

        return QuantizedKV(
            x_norm=x_norm,
            x_indices=k_idx,
            res_norm=res_norm,
            res_signs=signs,
        )

    def attention(self, layer_idx: int, q: torch.Tensor,
                  k_new: torch.Tensor | None = None,
                  v_new: torch.Tensor | None = None,
                  scaling: float | None = None,
                  attn_mask: torch.Tensor | None = None,
                  causal: bool = True) -> torch.Tensor:
        """q: [B, H_q, T_q, D] -> [B, H_q, T_q, D].

        If `k_new` / `v_new` are provided (shapes `[B, H_kv, T_new, D]`), they are used
        EXACTLY (no quantization round-trip) for this step's attention, concatenated with
        the quantized cache prefix. Caller must call `update(layer_idx, k_new, v_new)`
        afterwards to persist them. This makes prefill bit-exact to fp attention and
        keeps the just-arrived decode token from taking an unnecessary quantization hit.

        If `k_new` / `v_new` are None, attention runs against the current cache state only.

        scaling   : if None, defaults to 1/sqrt(head_dim).
        attn_mask : additive mask broadcastable to [B, H, T_q, T_total]. If provided,
                    `causal` is ignored (mask is assumed to already encode causality).
        """
        T_prefix = self.cur_len[layer_idx]
        T_new = 0 if k_new is None else k_new.shape[-2]
        T_total = T_prefix + T_new
        D = self.head_dim
        if scaling is None:
            scaling = 1.0 / math.sqrt(D)

        # ----- scores -----
        score_chunks: list[torch.Tensor] = []
        if T_prefix > 0:
            kq_view = self._key_view(layer_idx, T_prefix)
            score_chunks.append(
                self.k_quantizers[layer_idx].estimate_ip_pairwise(kq_view, q) * scaling
            )
        if T_new > 0:
            k_new_q = _repeat_head(k_new, self.n_rep) if self.n_rep > 1 else k_new
            score_chunks.append((q @ k_new_q.transpose(-1, -2)) * scaling)
        scores = torch.cat(score_chunks, dim=-1) if len(score_chunks) > 1 else score_chunks[0]

        if attn_mask is not None:
            scores = scores + attn_mask[..., :scores.shape[-1]]
        elif causal:
            Qn = q.shape[-2]
            base = T_total - Qn
            i = torch.arange(Qn,      device=q.device).unsqueeze(1)
            j = torch.arange(T_total, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(j > (base + i), float("-inf"))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # ----- values -----
        v_chunks: list[torch.Tensor] = []
        if T_prefix > 0:
            v_idx = unpack_bits(self.v_idx_packed[layer_idx, :, :, :T_prefix], self.v_bits, D)
            V_prefix = self.v_quantizers[layer_idx].decode(
                self.v_norm[layer_idx, :, :, :T_prefix], v_idx
            )                                                       # [B, H_kv, T_prefix, D]
            if self.n_rep > 1:
                V_prefix = _repeat_head(V_prefix, self.n_rep)
            v_chunks.append(V_prefix)
        if T_new > 0:
            v_new_q = _repeat_head(v_new, self.n_rep) if self.n_rep > 1 else v_new
            v_chunks.append(v_new_q)
        V = torch.cat(v_chunks, dim=-2) if len(v_chunks) > 1 else v_chunks[0]

        return attn @ V


# --------------------------------------------------------------------------------------
# HF ALL_ATTENTION_FUNCTIONS integration
# --------------------------------------------------------------------------------------

def turbo_attn_function(module, query, key, value, attention_mask=None,
                        scaling=None, dropout=0.0, **kwargs):
    """HF ALL_ATTENTION_FUNCTIONS callable.

    Inputs (already RoPE'd / QK-normed by the calling attention module):
        query : [B, H_q,  T_q, D]
        key   : [B, H_kv, T_k, D]
        value : [B, H_kv, T_k, D]
    Returns: (attn_output [B, T_q, H_q, D], None)  per HF contract.

    Attention this step is computed against (quantized prefix) ⊕ (fresh full-precision
    K/V), so the just-arrived tokens contribute exactly. The fresh K/V are then quantized
    and stored for future decode steps. Prefill (T_prefix == 0) is bit-exact to fp attention.
    """
    cache: TurboQuantKVCache = module._turbo_cache
    layer_idx: int = module.layer_idx

    # When we have a quantized prefix but HF only built the mask for the new tokens
    # (no past_key_values were passed), pad with zeros (prefix is all visible).
    if attention_mask is not None:
        T_prefix = cache.cur_len[layer_idx]
        T_new = key.shape[-2]
        T_total = T_prefix + T_new
        if attention_mask.shape[-1] < T_total:
            pad_len = T_total - attention_mask.shape[-1]
            pad_shape = list(attention_mask.shape)
            pad_shape[-1] = pad_len
            prefix_pad = torch.zeros(pad_shape, device=attention_mask.device,
                                     dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_pad, attention_mask], dim=-1)

    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    cache.update(layer_idx, key, value)

    # HF expects [B, T, H, D] from an attn_fn (the caller does the final reshape + o_proj).
    out = out.transpose(1, 2).contiguous()
    return out, None


# Register at import time so users can pick "turbo" via `attn_implementation`.
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["turbo"] = turbo_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def _force_set_attn_impl(cfg, value: str = "turbo") -> None:
    """Set _attn_implementation, bypassing the validator if it rejects unknown names."""
    try:
        cfg._attn_implementation = value
    except Exception:
        cfg.__dict__["_attn_implementation"] = value


def install_turbo_attention(model: nn.Module, cache: TurboQuantKVCache) -> int:
    """Switch the model's attention dispatcher to 'turbo' and attach the cache to every
    attention module.

    Works on any HF model whose attention forward dispatches through
    ALL_ATTENTION_FUNCTIONS — Llama, Qwen, Mistral, Gemma, modern GPT-2, etc.
    Returns the number of attention modules that received the cache reference.
    """
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required for install_turbo_attention")

    if hasattr(model, "config"):
        _force_set_attn_impl(model.config)

    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config)
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._turbo_cache = cache
            n += 1
    return n
