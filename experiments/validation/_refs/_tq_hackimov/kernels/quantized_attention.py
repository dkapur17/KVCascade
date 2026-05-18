import math
from typing import Optional

import torch
import triton
import triton.language as tl

from .attention_mask import mask_add_arg


@triton.jit
def turboquant_attention_scores_kernel(
    # Precomputed projections
    QPI_ptr,  # [B, H, M, K]
    QS_ptr,  # [B, H, M, K]
    # Compressed KV (for keys)
    K_idx_ptr,  # [B, H, N, K] int32 indices into centroids
    K_norm_ptr,  # [B, H, N, 1] float
    K_sign_ptr,  # [B, H, N, K] +/-1 float
    K_gamma_ptr,  # [B, H, N, 1] float
    # Codebook and scalars
    centroids_ptr,  # [levels] float
    qjl_factor: tl.constexpr,  # scalar
    inv_sqrt_k: tl.constexpr,  # scalar = 1/sqrt(K)
    # Output
    Out_ptr,  # [B, H, M, N]
    # Strides for QPI
    stride_qpi_b, stride_qpi_h, stride_qpi_m, stride_qpi_k,
    # Strides for QS
    stride_qs_b, stride_qs_h, stride_qs_m, stride_qs_k,
    # Strides for K_idx
    stride_kidx_b, stride_kidx_h, stride_kidx_n, stride_kidx_k,
    # Strides for K_norm
    stride_knorm_b, stride_knorm_h, stride_knorm_n, stride_knorm_1,
    # Strides for K_sign
    stride_ksign_b, stride_ksign_h, stride_ksign_n, stride_ksign_k,
    # Strides for K_gamma
    stride_kgamma_b, stride_kgamma_h, stride_kgamma_n, stride_kgamma_1,
    # Strides for Out
    stride_ob, stride_oh, stride_om, stride_on,
    # Optional additive mask [B, H, M, N] (stride 0 = broadcast)
    Mask_ptr,
    stride_mask_b,
    stride_mask_h,
    stride_mask_m,
    stride_mask_n,
    # Shapes
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KV_HEAD_GROUPS: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    b = pid_bh // H
    h_q = pid_bh - b * H
    h_kv = h_q // KV_HEAD_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # Accumulators: [BLOCK_M, BLOCK_N]
    mse_term = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qjl_sign_dot = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Load per-key scalars once: [BLOCK_N]
    kn_ptrs = (
        K_norm_ptr
        + b * stride_knorm_b
        + h_kv * stride_knorm_h
        + offs_n * stride_knorm_n
        + 0 * stride_knorm_1
    )
    kg_ptrs = (
        K_gamma_ptr
        + b * stride_kgamma_b
        + h_kv * stride_kgamma_h
        + offs_n * stride_kgamma_n
        + 0 * stride_kgamma_1
    )
    k_norm = tl.load(kn_ptrs, mask=offs_n < N, other=0.0).to(tl.float32)  # [BLOCK_N]
    k_gamma = tl.load(kg_ptrs, mask=offs_n < N, other=0.0).to(tl.float32)  # [BLOCK_N]

    # Projected query tensors: [BLOCK_M, K] accessed in chunks
    # Key chunks: K_idx/K_sign and centroids selection.
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k  # [BLOCK_K]
        mask_mk = (offs_m[:, None] < M) & (k_ids[None, :] < K)  # [BLOCK_M, BLOCK_K]
        mask_nk = (offs_n[:, None] < N) & (k_ids[None, :] < K)  # [BLOCK_N, BLOCK_K]

        qpi_ptrs = (
            QPI_ptr
            + b * stride_qpi_b
            + h_q * stride_qpi_h
            + offs_m[:, None] * stride_qpi_m
            + k_ids[None, :] * stride_qpi_k
        )
        qs_ptrs = (
            QS_ptr
            + b * stride_qs_b
            + h_q * stride_qs_h
            + offs_m[:, None] * stride_qs_m
            + k_ids[None, :] * stride_qs_k
        )
        q_pi = tl.load(qpi_ptrs, mask=mask_mk, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        q_s = tl.load(qs_ptrs, mask=mask_mk, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        kidx_ptrs = (
            K_idx_ptr
            + b * stride_kidx_b
            + h_kv * stride_kidx_h
            + offs_n[:, None] * stride_kidx_n
            + k_ids[None, :] * stride_kidx_k
        )
        k_sign_ptrs = (
            K_sign_ptr
            + b * stride_ksign_b
            + h_kv * stride_ksign_h
            + offs_n[:, None] * stride_ksign_n
            + k_ids[None, :] * stride_ksign_k
        )

        # centroids[idx] lookup: centroid_val[n,k] in [BLOCK_N, BLOCK_K]
        k_idx = tl.load(kidx_ptrs, mask=mask_nk, other=0).to(tl.int32)  # [BLOCK_N, BLOCK_K]
        centroid_vals = tl.load(centroids_ptr + k_idx, mask=mask_nk, other=0.0).to(tl.float32)
        k_sign = tl.load(k_sign_ptrs, mask=mask_nk, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # mse_term[m,n] += sum_k q_pi[m,k] * centroid_vals[n,k]
        mse_term += tl.sum(q_pi[:, None, :] * centroid_vals[None, :, :], axis=2)  # [BLOCK_M, BLOCK_N]

        # qjl_sign_dot[m,n] += sum_k q_s[m,k] * k_sign[n,k]
        qjl_sign_dot += tl.sum(q_s[:, None, :] * k_sign[None, :, :], axis=2)  # [BLOCK_M, BLOCK_N]

    # Final score:
    # score = k_norm * (mse_term + qjl_factor * k_gamma * qjl_sign_dot) * inv_sqrt_k
    scores = k_norm[None, :] * (mse_term + (qjl_factor * k_gamma[None, :]) * qjl_sign_dot)
    scores *= inv_sqrt_k

    if HAS_MASK:
        mask_ptrs = (
            Mask_ptr
            + b * stride_mask_b
            + h_q * stride_mask_h
            + offs_m[:, None] * stride_mask_m
            + offs_n[None, :] * stride_mask_n
        )
        mask_add = tl.load(
            mask_ptrs,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        scores = scores + mask_add

    mask_valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if CAUSAL:
        mask_valid = mask_valid & (offs_n[None, :] <= offs_m[:, None])

    out_ptrs = (
        Out_ptr
        + b * stride_ob
        + h_q * stride_oh
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    tl.store(out_ptrs, scores, mask=mask_valid)


def turboquant_attention(
    q_pi: torch.Tensor,
    q_s: torch.Tensor,
    kv_dict: dict,
    *,
    centroids: torch.Tensor,
    qjl_factor: float,
    num_kv_heads: Optional[int] = None,
    causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Compute attention scores [B, H, M, N] for TurboQuant-compressed KV.

    Parameters
    ----------
    q_pi: torch.Tensor
        q @ Pi.T, shape [B, H, M, head_dim]
    q_s: torch.Tensor
        q @ S.T, shape [B, H, M, head_dim]
    kv_dict: dict
        Output of `TurboQuantProd.quantize_kv(..., return_compressed=True)` for K.
        Required keys: k_idx, k_norm, k_sign, k_gamma.
    centroids: torch.Tensor
        1D tensor with Quantmse centroids for `mse_bits = bits - 1`.
    qjl_factor: float
        sqrt(pi/2)/head_dim from TurboQuant algorithm.
    attention_mask: torch.Tensor, optional
        Additive mask (bool or float), broadcastable to ``[B, H, M, N]``; applied before ``causal``.
    """
    if "k_idx" not in kv_dict or "k_norm" not in kv_dict or "k_sign" not in kv_dict or "k_gamma" not in kv_dict:
        raise ValueError("kv_dict must include k_idx, k_norm, k_sign, k_gamma")

    if q_pi.ndim != 4 or q_s.ndim != 4:
        raise ValueError("q_pi and q_s must have shape [B, H, M, head_dim]")

    if q_pi.shape != q_s.shape:
        raise ValueError("q_pi and q_s must have the same shape")

    B, H, M, K = q_pi.shape
    N = kv_dict["k_idx"].shape[2]
    h_kv = num_kv_heads if num_kv_heads is not None else H
    if H % h_kv != 0:
        raise ValueError(f"num_query_heads ({H}) must be divisible by num_kv_heads ({h_kv})")
    if kv_dict["k_idx"].shape[1] != h_kv:
        raise ValueError(
            f"KV head dimension mismatch: k_idx has {kv_dict['k_idx'].shape[1]} heads, expected num_kv_heads={h_kv}"
        )
    kv_head_groups = H // h_kv

    k_idx = kv_dict["k_idx"].to(torch.int32)
    k_norm = kv_dict["k_norm"][..., 0]
    k_sign = kv_dict["k_sign"]
    k_gamma = kv_dict["k_gamma"][..., 0]

    # Kernel expects fp32 for numerically stable accumulation.
    q_pi = q_pi.contiguous().to(torch.float32)
    q_s = q_s.contiguous().to(torch.float32)
    k_sign = k_sign.contiguous().to(torch.float32)
    centroids = centroids.contiguous().to(torch.float32)
    k_norm = k_norm.contiguous().to(torch.float32).unsqueeze(-1)
    k_gamma = k_gamma.contiguous().to(torch.float32).unsqueeze(-1)

    out = torch.full((B, H, M, N), float("-inf"), device=q_pi.device, dtype=torch.float32)

    mask_pack = mask_add_arg(attention_mask, B, H, M, N, device=q_pi.device)
    has_mask = mask_pack is not None
    if has_mask:
        mask_t, sb, sh, sm, sn = mask_pack
    else:
        mask_t, sb, sh, sm, sn = q_pi, 0, 0, 0, 0  # dummies; unused when HAS_MASK False

    # Tile sizes: conservative defaults for correctness.
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), B * H)

    inv_sqrt_k = 1.0 / math.sqrt(float(K))

    turboquant_attention_scores_kernel[grid](
        q_pi,
        q_s,
        k_idx,
        k_norm,
        k_sign,
        k_gamma,
        centroids,
        qjl_factor=qjl_factor,
        inv_sqrt_k=inv_sqrt_k,
        Out_ptr=out,
        stride_qpi_b=q_pi.stride(0),
        stride_qpi_h=q_pi.stride(1),
        stride_qpi_m=q_pi.stride(2),
        stride_qpi_k=q_pi.stride(3),
        stride_qs_b=q_s.stride(0),
        stride_qs_h=q_s.stride(1),
        stride_qs_m=q_s.stride(2),
        stride_qs_k=q_s.stride(3),
        stride_kidx_b=k_idx.stride(0),
        stride_kidx_h=k_idx.stride(1),
        stride_kidx_n=k_idx.stride(2),
        stride_kidx_k=k_idx.stride(3),
        stride_knorm_b=k_norm.stride(0),
        stride_knorm_h=k_norm.stride(1),
        stride_knorm_n=k_norm.stride(2),
        stride_knorm_1=k_norm.stride(3),
        stride_ksign_b=k_sign.stride(0),
        stride_ksign_h=k_sign.stride(1),
        stride_ksign_n=k_sign.stride(2),
        stride_ksign_k=k_sign.stride(3),
        stride_kgamma_b=k_gamma.stride(0),
        stride_kgamma_h=k_gamma.stride(1),
        stride_kgamma_n=k_gamma.stride(2),
        stride_kgamma_1=k_gamma.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_om=out.stride(2),
        stride_on=out.stride(3),
        Mask_ptr=mask_t,
        stride_mask_b=sb,
        stride_mask_h=sh,
        stride_mask_m=sm,
        stride_mask_n=sn,
        B=B,
        H=H,
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KV_HEAD_GROUPS=kv_head_groups,
        CAUSAL=causal,
        HAS_MASK=has_mask,
        num_warps=4,
        num_stages=2,
    )

    return out