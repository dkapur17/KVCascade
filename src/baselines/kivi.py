"""KIVI: per-channel asymmetric K + per-token asymmetric V quantization.

Reference: Liu et al., "KIVI: A Tuning-Free Asymmetric 2-bit Quantization for
KV Cache" (ICML 2024). https://arxiv.org/abs/2402.02750

Mechanics:
  - **K**: per-channel asymmetric INT-`bits` quantization. Within each
    `residual_length`-token chunk, every channel gets its own (scale, zero)
    computed across the R tokens.
  - **V**: per-token asymmetric INT-`bits` quantization. Within each chunk,
    every token gets its own (scale, zero) across the D channels.
  - **Residual buffer**: the last `residual_length` tokens stay fp16; once the
    residual is full, the oldest R tokens are flushed (quantize-dequantize) into
    the main store. This protects recent tokens from quantization error and
    avoids quantizing very small token groups.

No eviction — every token is kept; the cache grows with sequence length. This
is the "quantization-only" baseline.

We simulate quantization by storing the dequantized fp values; `bytes_total()`
reports the actual int-storage byte budget that the paper would report.

batch_size=1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from turbo_attn import _force_set_attn_impl, _repeat_head


_NEG_INF = float("-inf")


def _qdq_per_channel(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel asymmetric quant-dequant. x: [..., R, D]. Scale/zero per
    channel (last dim) computed across the R-token slice."""
    levels = (1 << bits) - 1
    x_min = x.amin(dim=-2, keepdim=True)
    x_max = x.amax(dim=-2, keepdim=True)
    scale = ((x_max - x_min) / levels).clamp_min(1e-9)
    q = ((x - x_min) / scale).round().clamp(0, levels)
    return x_min + scale * q


def _qdq_per_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token asymmetric quant-dequant. x: [..., R, D]. Scale/zero per
    token (second-to-last dim) computed across the D channels."""
    levels = (1 << bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / levels).clamp_min(1e-9)
    q = ((x - x_min) / scale).round().clamp(0, levels)
    return x_min + scale * q


class KIVICache:
    """KIVI quant-only cache. Keeps every token; residual buffer holds the last
    `residual_length` fp16; older tokens are simulated-quantized into main."""

    def __init__(self, num_layers: int, batch_size: int, num_heads: int,
                 head_dim: int,
                 max_seq_len: int,
                 bits: int = 2,
                 residual_length: int = 128,
                 num_kv_heads: int = None,
                 device=None, dtype=torch.float16):
        assert batch_size == 1, "v1 supports batch_size=1 only"
        assert 1 <= bits <= 8
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0
        self.n_rep = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.bits = bits
        self.R = residual_length
        self.device = device
        self.dtype = dtype

        L, B, H_kv, D = num_layers, batch_size, self.num_kv_heads, head_dim
        # Main store, holds the dequantized fp values of all quantized tokens.
        self.K_main = torch.zeros(L, B, H_kv, max_seq_len, D, device=device, dtype=dtype)
        self.V_main = torch.zeros(L, B, H_kv, max_seq_len, D, device=device, dtype=dtype)
        # Residual ring (fp).
        self.K_resid = torch.zeros(L, B, H_kv, self.R, D, device=device, dtype=dtype)
        self.V_resid = torch.zeros(L, B, H_kv, self.R, D, device=device, dtype=dtype)
        # Per-layer counts.
        self.main_count  = [0] * num_layers
        self.resid_count = [0] * num_layers
        self.next_pos    = [0] * num_layers

    def reset(self) -> None:
        self.main_count  = [0] * self.num_layers
        self.resid_count = [0] * self.num_layers
        self.next_pos    = [0] * self.num_layers

    def bytes_per_token_main(self) -> float:
        """Bytes per quantized token: 2 * int_bytes + K scales/zeros (per-channel,
        amortized over R) + V scales/zeros (per-token)."""
        fp_sz = self.K_main.element_size()
        D = self.head_dim
        return (2 * D * self.bits) / 8 + (2 * D * fp_sz) / self.R + 2 * fp_sz

    def bytes_per_token_resid(self) -> float:
        fp_sz = self.K_main.element_size()
        return 2 * self.head_dim * fp_sz

    def bytes_total(self) -> int:
        """Steady-state at max_seq_len: (T_max - R) tokens in main + R in residual.
        Matches what the KIVI paper would report for a sequence of this length."""
        T = self.max_seq_len
        R = self.R
        main_tok = max(0, T - R)
        resid_tok = min(T, R)
        per_lh = main_tok * self.bytes_per_token_main() + resid_tok * self.bytes_per_token_resid()
        return int(self.num_layers * self.batch_size * self.num_kv_heads * per_lh)

    # ------------------------------------------------------------------
    # Flush: drain `R` oldest residual tokens through quantize-dequantize
    # into the main fp store.
    # ------------------------------------------------------------------
    def _flush(self, layer_idx: int) -> None:
        R = self.R
        assert self.resid_count[layer_idx] >= R
        M = self.main_count[layer_idx]
        # Quantize-dequantize the full residual block.
        K_blk = self.K_resid[layer_idx, :, :, :R, :]                # [B, H_kv, R, D]
        V_blk = self.V_resid[layer_idx, :, :, :R, :]
        K_q = _qdq_per_channel(K_blk.float(), self.bits).to(self.dtype)
        V_q = _qdq_per_token  (V_blk.float(), self.bits).to(self.dtype)
        self.K_main[layer_idx, :, :, M:M + R, :] = K_q
        self.V_main[layer_idx, :, :, M:M + R, :] = V_q
        self.main_count[layer_idx] = M + R
        # Shift the rest of residual down.
        leftover = self.resid_count[layer_idx] - R
        if leftover > 0:
            self.K_resid[layer_idx, :, :, :leftover, :] = self.K_resid[layer_idx, :, :, R:R + leftover, :].clone()
            self.V_resid[layer_idx, :, :, :leftover, :] = self.V_resid[layer_idx, :, :, R:R + leftover, :].clone()
        self.resid_count[layer_idx] = leftover

    def _ingest(self, layer_idx: int,
                K_new: torch.Tensor, V_new: torch.Tensor) -> None:
        T_new = K_new.shape[-2]
        R = self.R
        src = 0
        while src < T_new:
            space = R - self.resid_count[layer_idx]
            n_take = min(space, T_new - src)
            ri = self.resid_count[layer_idx]
            self.K_resid[layer_idx, :, :, ri:ri + n_take, :] = K_new[:, :, src:src + n_take, :]
            self.V_resid[layer_idx, :, :, ri:ri + n_take, :] = V_new[:, :, src:src + n_take, :]
            self.resid_count[layer_idx] += n_take
            src += n_take
            if self.resid_count[layer_idx] >= R:
                self._flush(layer_idx)

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

        M = self.main_count[layer_idx]
        R_cur = self.resid_count[layer_idx]
        next_pos = M + R_cur

        # Cache portion (main ⊕ resid), already simulated-quantized via flush.
        K_main = self.K_main[layer_idx, :, :, :M, :].to(q_dtype)
        V_main = self.V_main[layer_idx, :, :, :M, :].to(q_dtype)
        K_resid = self.K_resid[layer_idx, :, :, :R_cur, :].to(q_dtype)
        V_resid = self.V_resid[layer_idx, :, :, :R_cur, :].to(q_dtype)
        K_cache_kv = torch.cat([K_main, K_resid], dim=-2)            # [B, H_kv, M+R_cur, D]
        V_cache_kv = torch.cat([V_main, V_resid], dim=-2)

        K_cache = _repeat_head(K_cache_kv, n_rep) if n_rep > 1 else K_cache_kv
        V_cache = _repeat_head(V_cache_kv, n_rep) if n_rep > 1 else V_cache_kv
        k_new_q = _repeat_head(k_new, n_rep) if n_rep > 1 else k_new
        v_new_q = _repeat_head(v_new, n_rep) if n_rep > 1 else v_new

        # Causal mask: K positions are 0..M+R_cur-1 (cache) and next_pos..next_pos+T_new-1 (fresh).
        new_positions = torch.arange(next_pos, next_pos + T_new, device=device, dtype=torch.long)
        cache_positions = torch.arange(0, M + R_cur, device=device, dtype=torch.long)
        all_k_positions = torch.cat([cache_positions, new_positions], dim=0)  # [M+R_cur+T_new]

        K_full = torch.cat([K_cache, k_new_q.to(q_dtype)], dim=-2)
        V_full = torch.cat([V_cache, v_new_q.to(q_dtype)], dim=-2)
        scores = (q @ K_full.transpose(-1, -2)) * scaling                     # [B, H_q, T_q, T_k]
        future = all_k_positions.view(1, -1) > new_positions.view(-1, 1)
        scores = scores.masked_fill(future.view(1, 1, T_q, -1), _NEG_INF)
        if attn_mask is not None:
            T_total = scores.shape[-1]
            if attn_mask.shape[-1] < T_total:
                pad = list(attn_mask.shape); pad[-1] = T_total - attn_mask.shape[-1]
                left_pad = torch.zeros(pad, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([left_pad, attn_mask], dim=-1)
            scores = scores + attn_mask[..., :T_total]
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_dtype)
        out = attn @ V_full

        # Ingest fresh tokens AFTER attention so the new tokens themselves stay
        # at full fp precision for this step (they go into the residual buffer).
        self._ingest(layer_idx, k_new, v_new)
        self.next_pos[layer_idx] += T_new
        return out


# ======================================================================================
# HF dispatcher integration
# ======================================================================================

def kivi_attn_function(module, query, key, value, attention_mask=None,
                       scaling=None, dropout=0.0, **kwargs):
    cache: KIVICache = module._kivi_cache
    layer_idx: int = module.layer_idx
    out = cache.attention(layer_idx, query, k_new=key, v_new=value,
                          scaling=scaling, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out, None


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["kivi"] = kivi_attn_function
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None


def install_kivi(model: nn.Module, cache: KIVICache) -> int:
    if ALL_ATTENTION_FUNCTIONS is None:
        raise RuntimeError("transformers is required for install_kivi")
    if hasattr(model, "config"):
        _force_set_attn_impl(model.config, "kivi")
    n = 0
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            _force_set_attn_impl(module.config, "kivi")
        if hasattr(module, "layer_idx") and isinstance(module.layer_idx, int):
            module._kivi_cache = cache
            n += 1
    return n
