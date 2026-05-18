"""
Fused attention: softmax(QK^T / sqrt(d)) @ V over TurboQuant-compressed K and V.

- Dense KV: ``k_*``, ``v_*`` with shape ``[B, H, N, D]``.
- Paged KV: physical ``[P, block_size, H, D]`` + ``block_tables[B, num_lb]`` + ``context_lens[B]``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

import triton
import triton.language as tl

from .attention_mask import mask_add_arg


@triton.jit
def turboquant_fused_attention_dense_kernel(
    QPI_ptr,
    QS_ptr,
    K_idx_ptr,
    K_norm_ptr,
    K_sign_ptr,
    K_gamma_ptr,
    V_idx_ptr,
    V_norm_ptr,
    V_sign_ptr,
    V_gamma_ptr,
    Pi_ptr,
    S_ptr,
    centroids_ptr,
    Out_ptr,
    stride_qpi_b,
    stride_qpi_h,
    stride_qpi_m,
    stride_qpi_d,
    stride_qs_b,
    stride_qs_h,
    stride_qs_m,
    stride_qs_d,
    stride_kidx_b,
    stride_kidx_h,
    stride_kidx_n,
    stride_kidx_d,
    stride_knorm_b,
    stride_knorm_h,
    stride_knorm_n,
    stride_knorm_1,
    stride_ksign_b,
    stride_ksign_h,
    stride_ksign_n,
    stride_ksign_d,
    stride_kgamma_b,
    stride_kgamma_h,
    stride_kgamma_n,
    stride_kgamma_1,
    stride_vidx_b,
    stride_vidx_h,
    stride_vidx_n,
    stride_vidx_d,
    stride_vnorm_b,
    stride_vnorm_h,
    stride_vnorm_n,
    stride_vnorm_1,
    stride_vsign_b,
    stride_vsign_h,
    stride_vsign_n,
    stride_vsign_d,
    stride_vgamma_b,
    stride_vgamma_h,
    stride_vgamma_n,
    stride_vgamma_1,
    stride_pi_0,
    stride_pi_1,
    stride_s_0,
    stride_s_1,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    Mask_ptr,
    stride_mask_b,
    stride_mask_h,
    stride_mask_m,
    stride_mask_n,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    qjl_factor: tl.constexpr,
    inv_sqrt_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KV_HEAD_GROUPS: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid % M
    t = pid // M
    h_q = t % H
    b = t // H
    h_kv = h_q // KV_HEAD_GROUPS

    offs_d = tl.arange(0, D)

    m_max = -float("inf")
    l_sum = 0.0
    o_acc = tl.zeros([D], dtype=tl.float32)

    for n_base in range(0, N, BLOCK_N):
        offs_n = n_base + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        kn_ptrs = (
            K_norm_ptr
            + b * stride_knorm_b
            + h_kv * stride_knorm_h
            + offs_n * stride_knorm_n
        )
        kg_ptrs = (
            K_gamma_ptr
            + b * stride_kgamma_b
            + h_kv * stride_kgamma_h
            + offs_n * stride_kgamma_n
        )
        k_norm = tl.load(kn_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        k_gamma = tl.load(kg_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        mse_term = tl.zeros([BLOCK_N], dtype=tl.float32)
        qjl_sign_dot = tl.zeros([BLOCK_N], dtype=tl.float32)

        for k0 in range(0, D, BLOCK_K):
            kid = k0 + tl.arange(0, BLOCK_K)
            mask_k = kid < D
            mask_nk = mask_n[:, None] & mask_k[None, :]

            qp = tl.load(
                QPI_ptr + b * stride_qpi_b + h_q * stride_qpi_h + m * stride_qpi_m + kid * stride_qpi_d,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            qs = tl.load(
                QS_ptr + b * stride_qs_b + h_q * stride_qs_h + m * stride_qs_m + kid * stride_qs_d,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)

            kidx_ptrs = (
                K_idx_ptr
                + b * stride_kidx_b
                + h_kv * stride_kidx_h
                + offs_n[:, None] * stride_kidx_n
                + kid[None, :] * stride_kidx_d
            )
            ksign_ptrs = (
                K_sign_ptr
                + b * stride_ksign_b
                + h_kv * stride_ksign_h
                + offs_n[:, None] * stride_ksign_n
                + kid[None, :] * stride_ksign_d
            )
            k_idx = tl.load(kidx_ptrs, mask=mask_nk, other=0).to(tl.int32)
            centroid_vals = tl.load(centroids_ptr + k_idx, mask=mask_nk, other=0.0).to(tl.float32)
            k_sign = tl.load(ksign_ptrs, mask=mask_nk, other=0.0).to(tl.float32)

            mse_term += tl.sum(qp[None, :] * centroid_vals, axis=1)
            qjl_sign_dot += tl.sum(qs[None, :] * k_sign, axis=1)

        scores = k_norm * (mse_term + (qjl_factor * k_gamma) * qjl_sign_dot) * inv_sqrt_d
        scores = tl.where(mask_n, scores, -float("inf"))
        if HAS_MASK:
            mask_ptrs = (
                Mask_ptr
                + b * stride_mask_b
                + h_q * stride_mask_h
                + m * stride_mask_m
                + offs_n * stride_mask_n
            )
            scores = scores + tl.load(mask_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        if CAUSAL:
            scores = tl.where(offs_n <= m, scores, -float("inf"))

        m_blk = tl.max(scores)
        m_new = tl.maximum(m_max, m_blk)
        scale_old = tl.exp(m_max - m_new)
        l_sum = l_sum * scale_old
        o_acc = o_acc * scale_old
        exp_s = tl.exp(scores - m_new)
        exp_s = tl.where(mask_n, exp_s, 0.0)
        l_sum += tl.sum(exp_s)
        m_max = m_new

        for j in range(BLOCK_N):
            nn = n_base + j
            mask_j = nn < N
            es_j = tl.sum(tl.where(tl.arange(0, BLOCK_N) == j, exp_s, 0.0))

            x_mse = tl.zeros([D], dtype=tl.float32)
            x_qjl = tl.zeros([D], dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                kid = k0 + tl.arange(0, BLOCK_K)
                mask_k = kid < D
                mask_kj = mask_k & mask_j

                vidx_ptrs = (
                    V_idx_ptr
                    + b * stride_vidx_b
                    + h_kv * stride_vidx_h
                    + nn * stride_vidx_n
                    + kid * stride_vidx_d
                )
                vsign_ptrs = (
                    V_sign_ptr
                    + b * stride_vsign_b
                    + h_kv * stride_vsign_h
                    + nn * stride_vsign_n
                    + kid * stride_vsign_d
                )
                vi = tl.load(vidx_ptrs, mask=mask_kj, other=0).to(tl.int32)
                cvals = tl.load(centroids_ptr + vi, mask=mask_kj, other=0.0).to(tl.float32)
                vs = tl.load(vsign_ptrs, mask=mask_kj, other=0.0).to(tl.float32)

                pi_blk = tl.load(
                    Pi_ptr + kid[:, None] * stride_pi_0 + offs_d[None, :] * stride_pi_1,
                    mask=mask_kj[:, None],
                    other=0.0,
                ).to(tl.float32)
                s_blk = tl.load(
                    S_ptr + kid[:, None] * stride_s_0 + offs_d[None, :] * stride_s_1,
                    mask=mask_kj[:, None],
                    other=0.0,
                ).to(tl.float32)

                x_mse += tl.sum(cvals[:, None] * pi_blk, axis=0)
                x_qjl += tl.sum(vs[:, None] * s_blk, axis=0)

            vn_ptr = V_norm_ptr + b * stride_vnorm_b + h_kv * stride_vnorm_h + nn * stride_vnorm_n
            vg_ptr = V_gamma_ptr + b * stride_vgamma_b + h_kv * stride_vgamma_h + nn * stride_vgamma_n
            v_norm = tl.load(vn_ptr, mask=mask_j, other=0.0).to(tl.float32)
            v_gamma = tl.load(vg_ptr, mask=mask_j, other=0.0).to(tl.float32)

            v_unit = x_mse + qjl_factor * v_gamma * x_qjl
            v_rec = v_unit * v_norm
            o_acc += tl.where(mask_j, es_j * v_rec, tl.zeros([D], dtype=tl.float32))

    out = o_acc / l_sum
    tl.store(
        Out_ptr + b * stride_ob + h_q * stride_oh + m * stride_om + offs_d * stride_od,
        out,
    )


@triton.jit
def turboquant_fused_attention_paged_kernel(
    QPI_ptr,
    QS_ptr,
    K_idx_phys_ptr,
    K_norm_phys_ptr,
    K_sign_phys_ptr,
    K_gamma_phys_ptr,
    V_idx_phys_ptr,
    V_norm_phys_ptr,
    V_sign_phys_ptr,
    V_gamma_phys_ptr,
    block_tables_ptr,
    context_lens_ptr,
    Pi_ptr,
    S_ptr,
    centroids_ptr,
    Out_ptr,
    stride_qpi_b,
    stride_qpi_h,
    stride_qpi_m,
    stride_qpi_d,
    stride_qs_b,
    stride_qs_h,
    stride_qs_m,
    stride_qs_d,
    stride_kp_pb,
    stride_kp_t,
    stride_kp_h,
    stride_kp_d,
    stride_knorm_pb,
    stride_knorm_t,
    stride_knorm_h,
    stride_knorm_1,
    stride_ksign_pb,
    stride_ksign_t,
    stride_ksign_h,
    stride_ksign_d,
    stride_kgamma_pb,
    stride_kgamma_t,
    stride_kgamma_h,
    stride_kgamma_1,
    stride_vp_pb,
    stride_vp_t,
    stride_vp_h,
    stride_vp_d,
    stride_vnorm_pb,
    stride_vnorm_t,
    stride_vnorm_h,
    stride_vnorm_1,
    stride_vsign_pb,
    stride_vsign_t,
    stride_vsign_h,
    stride_vsign_d,
    stride_vgamma_pb,
    stride_vgamma_t,
    stride_vgamma_h,
    stride_vgamma_1,
    stride_bt_b,
    stride_bt_lb,
    stride_pi_0,
    stride_pi_1,
    stride_s_0,
    stride_s_1,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    Mask_ptr,
    stride_mask_b,
    stride_mask_h,
    stride_mask_m,
    stride_mask_n,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    max_ctx: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PHYS_P: tl.constexpr,
    NUM_LOGICAL_BLOCKS: tl.constexpr,
    qjl_factor: tl.constexpr,
    inv_sqrt_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KV_HEAD_GROUPS: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    One program = (tile of ``BLOCK_M`` query rows) × (one batch row ``b``, one query head ``h_q``).
    ``NUM_PHYS_P``: number of physical pages ``P``; table entries ``pb < 0`` or ``pb >= P`` are masked
    (vLLM-style padding / allocator slots).
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h_q = pid_bh - b * H
    h_kv = h_q // KV_HEAD_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    N = tl.load(context_lens_ptr + b).to(tl.int32)
    N = tl.maximum(N, 1)

    offs_d = tl.arange(0, D)

    m_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for n_base in range(0, max_ctx, BLOCK_N):
        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for j in range(BLOCK_N):
            nn = n_base + j
            active = nn < N
            lb = nn // BLOCK_SIZE
            tt = nn % BLOCK_SIZE
            addr_ok = active & (lb < NUM_LOGICAL_BLOCKS)
            pb = tl.load(
                block_tables_ptr + b * stride_bt_b + lb * stride_bt_lb,
                mask=addr_ok,
                other=-1,
            ).to(tl.int32)
            pb_ok = (pb >= 0) & (pb < NUM_PHYS_P)
            active_key = active & pb_ok

            kn_p = (
                K_norm_phys_ptr
                + pb * stride_knorm_pb
                + tt * stride_knorm_t
                + h_kv * stride_knorm_h
            )
            kg_p = (
                K_gamma_phys_ptr
                + pb * stride_kgamma_pb
                + tt * stride_kgamma_t
                + h_kv * stride_kgamma_h
            )
            k_norm = tl.load(kn_p, mask=active_key, other=0.0).to(tl.float32)
            k_gamma = tl.load(kg_p, mask=active_key, other=0.0).to(tl.float32)

            mse_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
            qjl_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                kid = k0 + tl.arange(0, BLOCK_K)
                mask_k = kid < D
                mask_mk = mask_m[:, None] & mask_k[None, :]

                qp = tl.load(
                    QPI_ptr
                    + b * stride_qpi_b
                    + h_q * stride_qpi_h
                    + offs_m[:, None] * stride_qpi_m
                    + kid[None, :] * stride_qpi_d,
                    mask=mask_mk,
                    other=0.0,
                ).to(tl.float32)
                qs = tl.load(
                    QS_ptr
                    + b * stride_qs_b
                    + h_q * stride_qs_h
                    + offs_m[:, None] * stride_qs_m
                    + kid[None, :] * stride_qs_d,
                    mask=mask_mk,
                    other=0.0,
                ).to(tl.float32)

                kidx_p = (
                    K_idx_phys_ptr
                    + pb * stride_kp_pb
                    + tt * stride_kp_t
                    + h_kv * stride_kp_h
                    + kid * stride_kp_d
                )
                ksign_p = (
                    K_sign_phys_ptr
                    + pb * stride_ksign_pb
                    + tt * stride_ksign_t
                    + h_kv * stride_ksign_h
                    + kid * stride_ksign_d
                )
                mask_kv = active_key & mask_k
                ki = tl.load(kidx_p, mask=mask_kv, other=0).to(tl.int32)
                cv = tl.load(centroids_ptr + ki, mask=mask_kv, other=0.0).to(tl.float32)
                sg = tl.load(ksign_p, mask=mask_kv, other=0.0).to(tl.float32)

                mse_acc += tl.sum(qp * cv[None, :], axis=1)
                qjl_acc += tl.sum(qs * sg[None, :], axis=1)

            col_sc = k_norm * (mse_acc + qjl_factor * k_gamma * qjl_acc) * inv_sqrt_d
            col_sc = tl.where(active_key, col_sc, tl.full([BLOCK_M], -float("inf"), dtype=tl.float32))
            j_ar = tl.arange(0, BLOCK_N)
            scores = tl.where(j_ar[None, :] == j, col_sc[:, None], scores)

        scores = tl.where(
            mask_m[:, None],
            scores,
            tl.full([BLOCK_M, BLOCK_N], -float("inf"), dtype=tl.float32),
        )

        offs_n = n_base + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        if HAS_MASK:
            mask_ptrs = (
                Mask_ptr
                + b * stride_mask_b
                + h_q * stride_mask_h
                + offs_m[:, None] * stride_mask_m
                + offs_n[None, :] * stride_mask_n
            )
            scores = scores + tl.load(
                mask_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
        if CAUSAL:
            scores = tl.where(
                offs_n[None, :] <= offs_m[:, None],
                scores,
                tl.full([BLOCK_M, BLOCK_N], -float("inf"), dtype=tl.float32),
            )

        mask_mn = mask_m[:, None] & mask_n[None, :]
        m_blk = tl.max(scores, axis=1)
        m_new = tl.maximum(m_max, m_blk)
        scale_old = tl.exp(m_max - m_new)
        l_sum = l_sum * scale_old
        o_acc = o_acc * scale_old[:, None]
        exp_s = tl.exp(scores - m_new[:, None])
        exp_s = tl.where(mask_mn, exp_s, 0.0)
        l_sum = l_sum + tl.sum(exp_s, axis=1)
        m_max = m_new

        for j in range(BLOCK_N):
            nn = n_base + j
            active_nn = nn < N
            lb = nn // BLOCK_SIZE
            tt = nn % BLOCK_SIZE
            addr_ok = active_nn & (lb < NUM_LOGICAL_BLOCKS)
            pb = tl.load(
                block_tables_ptr + b * stride_bt_b + lb * stride_bt_lb,
                mask=addr_ok,
                other=-1,
            ).to(tl.int32)
            pb_ok = (pb >= 0) & (pb < NUM_PHYS_P)
            active_key = active_nn & pb_ok
            j_ar_n = tl.arange(0, BLOCK_N)
            es_j = tl.sum(tl.where(j_ar_n[None, :] == j, exp_s, 0.0), axis=1)

            x_mse = tl.zeros([D], dtype=tl.float32)
            x_qjl = tl.zeros([D], dtype=tl.float32)
            for k0 in range(0, D, BLOCK_K):
                kid = k0 + tl.arange(0, BLOCK_K)
                mask_k = kid < D
                mask_kj = active_key & mask_k

                vidx_p = (
                    V_idx_phys_ptr
                    + pb * stride_vp_pb
                    + tt * stride_vp_t
                    + h_kv * stride_vp_h
                    + kid * stride_vp_d
                )
                vsign_p = (
                    V_sign_phys_ptr
                    + pb * stride_vsign_pb
                    + tt * stride_vsign_t
                    + h_kv * stride_vsign_h
                    + kid * stride_vsign_d
                )
                vi = tl.load(vidx_p, mask=mask_kj, other=0).to(tl.int32)
                cvals = tl.load(centroids_ptr + vi, mask=mask_kj, other=0.0).to(tl.float32)
                vs = tl.load(vsign_p, mask=mask_kj, other=0.0).to(tl.float32)

                pi_blk = tl.load(
                    Pi_ptr + kid[:, None] * stride_pi_0 + offs_d[None, :] * stride_pi_1,
                    mask=mask_kj[:, None],
                    other=0.0,
                ).to(tl.float32)
                s_blk = tl.load(
                    S_ptr + kid[:, None] * stride_s_0 + offs_d[None, :] * stride_s_1,
                    mask=mask_kj[:, None],
                    other=0.0,
                ).to(tl.float32)

                x_mse += tl.sum(cvals[:, None] * pi_blk, axis=0)
                x_qjl += tl.sum(vs[:, None] * s_blk, axis=0)

            vn_p = (
                V_norm_phys_ptr
                + pb * stride_vnorm_pb
                + tt * stride_vnorm_t
                + h_kv * stride_vnorm_h
            )
            vg_p = (
                V_gamma_phys_ptr
                + pb * stride_vgamma_pb
                + tt * stride_vgamma_t
                + h_kv * stride_vgamma_h
            )
            v_norm = tl.load(vn_p, mask=active_key, other=0.0).to(tl.float32)
            v_gamma = tl.load(vg_p, mask=active_key, other=0.0).to(tl.float32)

            v_unit = x_mse + qjl_factor * v_gamma * x_qjl
            v_rec = v_unit * v_norm
            o_acc += tl.where(active_key, es_j[:, None] * v_rec[None, :], tl.zeros([BLOCK_M, D], dtype=tl.float32))

    denom = tl.maximum(l_sum, 1e-10)
    out_rows = tl.where(mask_m[:, None], o_acc / denom[:, None], tl.zeros([BLOCK_M, D], dtype=tl.float32))
    out_ptrs = (
        Out_ptr
        + b * stride_ob
        + h_q * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out_rows, mask=mask_m[:, None])


def _supported_head_dim(d: int) -> None:
    if d not in (16, 32, 64, 128, 256):
        raise ValueError(f"fused Triton attention supports head_dim in (16,32,64,128,256), got {d}")


def _block_n_for_n(n: int) -> int:
    if n <= 0:
        return 1
    p2 = triton.next_power_of_2(n)
    return int(min(p2, 32))


def turboquant_fused_attention_dense(
    q_pi: torch.Tensor,
    q_s: torch.Tensor,
    kv_dict: Dict[str, Any],
    *,
    centroids: torch.Tensor,
    qjl_factor: float,
    pi: torch.Tensor,
    s: torch.Tensor,
    num_kv_heads: Optional[int] = None,
    causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    softmax(scores) @ V_recon with dense KV.

    ``q_*`` projections have shape ``[B, H_q, M, D]``. KV tensors use ``H_kv`` heads
    (GQA/MQA: ``H_q % H_kv == 0``), same layout as ``quantize_kv`` on ``k``/``v`` with
    ``[B, H_kv, N, D]``.

    ``attention_mask`` is added to logits (bool or float); shapes ``[M,N]``, ``[B,M,N]``, or ``[B,H,M,N]``.
    """
    required = ("k_idx", "k_norm", "k_sign", "k_gamma", "v_idx", "v_norm", "v_sign", "v_gamma")
    if any(k not in kv_dict for k in required):
        raise ValueError(f"kv_dict must contain {required}")

    B, H, M, D = q_pi.shape
    N = kv_dict["k_idx"].shape[2]
    _supported_head_dim(D)

    h_kv = num_kv_heads if num_kv_heads is not None else H
    if H % h_kv != 0:
        raise ValueError(f"num query heads ({H}) must be divisible by num_kv_heads ({h_kv})")
    if kv_dict["k_idx"].shape[1] != h_kv:
        raise ValueError(
            f"KV head mismatch: k_idx.shape[1]={kv_dict['k_idx'].shape[1]}, expected num_kv_heads={h_kv}"
        )
    kv_head_groups = H // h_kv

    k_idx = kv_dict["k_idx"].to(torch.int32).contiguous()
    k_norm = kv_dict["k_norm"][..., 0].contiguous().to(torch.float32).unsqueeze(-1)
    k_sign = kv_dict["k_sign"].contiguous().to(torch.float32)
    k_gamma = kv_dict["k_gamma"][..., 0].contiguous().to(torch.float32).unsqueeze(-1)
    v_idx = kv_dict["v_idx"].to(torch.int32).contiguous()
    v_norm = kv_dict["v_norm"][..., 0].contiguous().to(torch.float32).unsqueeze(-1)
    v_sign = kv_dict["v_sign"].contiguous().to(torch.float32)
    v_gamma = kv_dict["v_gamma"][..., 0].contiguous().to(torch.float32).unsqueeze(-1)

    q_pi = q_pi.contiguous().to(torch.float32)
    q_s = q_s.contiguous().to(torch.float32)
    centroids = centroids.contiguous().to(torch.float32)
    pi = pi.contiguous().to(torch.float32)
    s_mat = s.contiguous().to(torch.float32)

    out = torch.empty((B, H, M, D), device=q_pi.device, dtype=torch.float32)

    mask_pack = mask_add_arg(attention_mask, B, H, M, N, device=q_pi.device)
    has_mask = mask_pack is not None
    if has_mask:
        mask_t, smb, smh, smm, smn = mask_pack
    else:
        mask_t, smb, smh, smm, smn = q_pi, 0, 0, 0, 0

    BLOCK_N = _block_n_for_n(N)
    BLOCK_K = _block_n_for_n(D)
    inv_sqrt_d = 1.0 / math.sqrt(float(D))

    grid = (B * H * M,)
    turboquant_fused_attention_dense_kernel[grid](
        q_pi,
        q_s,
        k_idx,
        k_norm,
        k_sign,
        k_gamma,
        v_idx,
        v_norm,
        v_sign,
        v_gamma,
        pi,
        s_mat,
        centroids,
        out,
        q_pi.stride(0),
        q_pi.stride(1),
        q_pi.stride(2),
        q_pi.stride(3),
        q_s.stride(0),
        q_s.stride(1),
        q_s.stride(2),
        q_s.stride(3),
        k_idx.stride(0),
        k_idx.stride(1),
        k_idx.stride(2),
        k_idx.stride(3),
        k_norm.stride(0),
        k_norm.stride(1),
        k_norm.stride(2),
        k_norm.stride(3),
        k_sign.stride(0),
        k_sign.stride(1),
        k_sign.stride(2),
        k_sign.stride(3),
        k_gamma.stride(0),
        k_gamma.stride(1),
        k_gamma.stride(2),
        k_gamma.stride(3),
        v_idx.stride(0),
        v_idx.stride(1),
        v_idx.stride(2),
        v_idx.stride(3),
        v_norm.stride(0),
        v_norm.stride(1),
        v_norm.stride(2),
        v_norm.stride(3),
        v_sign.stride(0),
        v_sign.stride(1),
        v_sign.stride(2),
        v_sign.stride(3),
        v_gamma.stride(0),
        v_gamma.stride(1),
        v_gamma.stride(2),
        v_gamma.stride(3),
        pi.stride(0),
        pi.stride(1),
        s_mat.stride(0),
        s_mat.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Mask_ptr=mask_t,
        stride_mask_b=smb,
        stride_mask_h=smh,
        stride_mask_m=smm,
        stride_mask_n=smn,
        B=B,
        H=H,
        M=M,
        N=N,
        D=D,
        qjl_factor=float(qjl_factor),
        inv_sqrt_d=inv_sqrt_d,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KV_HEAD_GROUPS=kv_head_groups,
        CAUSAL=causal,
        HAS_MASK=has_mask,
        num_warps=4,
    )
    return out


def pack_dense_kv_to_paged(
    kv_dict: Dict[str, torch.Tensor],
    block_size: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Pack dense ``[B,H,N,D]`` quantized KV into physical blocks ``[P, block_size, H, D]``.

    Pads ``N`` to a multiple of ``block_size``. Returns ``(paged_dict, block_tables, context_lens)``.
    ``block_tables`` has shape ``[B, num_lb]`` with contiguous physical block ids per batch row.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    k_idx = kv_dict["k_idx"]
    B, H, N, D = k_idx.shape
    pad_n = (block_size - (N % block_size)) % block_size
    N_pad = N + pad_n

    def pad_n_dim(t: torch.Tensor) -> torch.Tensor:
        if pad_n == 0:
            return t
        pad_shape = list(t.shape)
        pad_shape[2] = pad_n
        z = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
        return torch.cat([t, z], dim=2)

    k_idx_p = pad_n_dim(kv_dict["k_idx"])
    k_norm_p = pad_n_dim(kv_dict["k_norm"])
    k_sign_p = pad_n_dim(kv_dict["k_sign"])
    k_gamma_p = pad_n_dim(kv_dict["k_gamma"])
    v_idx_p = pad_n_dim(kv_dict["v_idx"])
    v_norm_p = pad_n_dim(kv_dict["v_norm"])
    v_sign_p = pad_n_dim(kv_dict["v_sign"])
    v_gamma_p = pad_n_dim(kv_dict["v_gamma"])

    num_lb = N_pad // block_size
    P = B * num_lb

    def to_phys(x: torch.Tensor) -> torch.Tensor:
        ld = x.shape[-1]
        y = x.view(B, H, num_lb, block_size, ld).permute(0, 2, 3, 1, 4).contiguous()
        return y.view(P, block_size, H, ld)

    paged = {
        "k_idx_phys": to_phys(k_idx_p),
        "k_norm_phys": to_phys(k_norm_p),
        "k_sign_phys": to_phys(k_sign_p),
        "k_gamma_phys": to_phys(k_gamma_p),
        "v_idx_phys": to_phys(v_idx_p),
        "v_norm_phys": to_phys(v_norm_p),
        "v_sign_phys": to_phys(v_sign_p),
        "v_gamma_phys": to_phys(v_gamma_p),
    }
    block_tables = torch.arange(P, device=k_idx.device, dtype=torch.int32).view(B, num_lb)
    context_lens = torch.full((B,), N, device=k_idx.device, dtype=torch.int32)
    return paged, block_tables, context_lens


def turboquant_fused_attention_paged(
    q_pi: torch.Tensor,
    q_s: torch.Tensor,
    paged_kv: Dict[str, Any],
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_ctx: int,
    *,
    centroids: torch.Tensor,
    qjl_factor: float,
    pi: torch.Tensor,
    s: torch.Tensor,
    num_kv_heads: Optional[int] = None,
    causal: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Paged fused attention; physical tensors ``[P, block_size, H_kv, D]``.

    ``block_tables[b, lb]`` are physical page ids (vLLM allocator). Entries may be ``-1`` for unused
    logical blocks; keys with ``n >= context_lens[b]`` do not read the table. Physical ids must satisfy
    ``0 <= id < P`` where ``P`` is ``k_idx_phys.shape[0]``.
    """
    B, H, M, D = q_pi.shape
    _supported_head_dim(D)

    h_kv = num_kv_heads if num_kv_heads is not None else H
    if H % h_kv != 0:
        raise ValueError(f"num query heads ({H}) must be divisible by num_kv_heads ({h_kv})")
    kv_head_groups = H // h_kv

    q_pi = q_pi.contiguous().to(torch.float32)
    q_s = q_s.contiguous().to(torch.float32)
    centroids = centroids.contiguous().to(torch.float32)
    pi = pi.contiguous().to(torch.float32)
    s_mat = s.contiguous().to(torch.float32)

    k_idx_p = paged_kv["k_idx_phys"].to(torch.int32).contiguous()
    k_norm_p = paged_kv["k_norm_phys"].contiguous().to(torch.float32)
    k_sign_p = paged_kv["k_sign_phys"].contiguous().to(torch.float32)
    k_gamma_p = paged_kv["k_gamma_phys"].contiguous().to(torch.float32)
    v_idx_p = paged_kv["v_idx_phys"].to(torch.int32).contiguous()
    v_norm_p = paged_kv["v_norm_phys"].contiguous().to(torch.float32)
    v_sign_p = paged_kv["v_sign_phys"].contiguous().to(torch.float32)
    v_gamma_p = paged_kv["v_gamma_phys"].contiguous().to(torch.float32)

    block_tables = block_tables.to(torch.int32).contiguous()
    context_lens = context_lens.to(torch.int32).contiguous()
    num_logical_blocks = int(block_tables.shape[1])

    out = torch.empty((B, H, M, D), device=q_pi.device, dtype=torch.float32)

    mask_pack = mask_add_arg(attention_mask, B, H, M, max_ctx, device=q_pi.device)
    has_mask = mask_pack is not None
    if has_mask:
        mask_t, smb, smh, smm, smn = mask_pack
    else:
        mask_t, smb, smh, smm, smn = q_pi, 0, 0, 0, 0

    P, bs, Hk, Dk = k_idx_p.shape
    if Hk != h_kv or Dk != D or bs != block_size:
        raise ValueError(
            f"paged tensor shape mismatch: got (P,bs,H,D)=({P},{bs},{Hk},{Dk}), "
            f"expected H_kv={h_kv}, D={D}, block_size={block_size}"
        )

    BLOCK_N = _block_n_for_n(max_ctx)
    BLOCK_K = _block_n_for_n(D)
    BLOCK_M = 16
    inv_sqrt_d = 1.0 / math.sqrt(float(D))

    grid = (triton.cdiv(M, BLOCK_M), B * H)
    turboquant_fused_attention_paged_kernel[grid](
        q_pi,
        q_s,
        k_idx_p,
        k_norm_p,
        k_sign_p,
        k_gamma_p,
        v_idx_p,
        v_norm_p,
        v_sign_p,
        v_gamma_p,
        block_tables,
        context_lens,
        pi,
        s_mat,
        centroids,
        out,
        q_pi.stride(0),
        q_pi.stride(1),
        q_pi.stride(2),
        q_pi.stride(3),
        q_s.stride(0),
        q_s.stride(1),
        q_s.stride(2),
        q_s.stride(3),
        k_idx_p.stride(0),
        k_idx_p.stride(1),
        k_idx_p.stride(2),
        k_idx_p.stride(3),
        k_norm_p.stride(0),
        k_norm_p.stride(1),
        k_norm_p.stride(2),
        k_norm_p.stride(3),
        k_sign_p.stride(0),
        k_sign_p.stride(1),
        k_sign_p.stride(2),
        k_sign_p.stride(3),
        k_gamma_p.stride(0),
        k_gamma_p.stride(1),
        k_gamma_p.stride(2),
        k_gamma_p.stride(3),
        v_idx_p.stride(0),
        v_idx_p.stride(1),
        v_idx_p.stride(2),
        v_idx_p.stride(3),
        v_norm_p.stride(0),
        v_norm_p.stride(1),
        v_norm_p.stride(2),
        v_norm_p.stride(3),
        v_sign_p.stride(0),
        v_sign_p.stride(1),
        v_sign_p.stride(2),
        v_sign_p.stride(3),
        v_gamma_p.stride(0),
        v_gamma_p.stride(1),
        v_gamma_p.stride(2),
        v_gamma_p.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        pi.stride(0),
        pi.stride(1),
        s_mat.stride(0),
        s_mat.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Mask_ptr=mask_t,
        stride_mask_b=smb,
        stride_mask_h=smh,
        stride_mask_m=smm,
        stride_mask_n=smn,
        B=B,
        H=H,
        M=M,
        max_ctx=max_ctx,
        D=D,
        BLOCK_SIZE=block_size,
        NUM_PHYS_P=P,
        NUM_LOGICAL_BLOCKS=num_logical_blocks,
        qjl_factor=float(qjl_factor),
        inv_sqrt_d=inv_sqrt_d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KV_HEAD_GROUPS=kv_head_groups,
        CAUSAL=causal,
        HAS_MASK=has_mask,
        num_warps=4,
    )
    return out
