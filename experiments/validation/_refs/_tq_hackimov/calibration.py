"""
Dataset calibration for :class:`~turboquant.TurboQuantProd`.

Fits scalar Quantmse codebook centroids in rotated coordinates ``y = x_unit @ Π^T`` using 1D k-means
on a sample of KV (or any head_dim vectors). Π and S stay fixed (from ``seed``); only centroids change.

Modes
-----
- **paper** — ``K = 2^(bits-1)`` levels (extended **4-bit+** when ``bits >= 5``).
- **ternary** — ``K = 3`` levels (~**1.58 bit** entropy per MSE index + 1 bit QJL per dim).
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Union

import torch

from .core import CodebookKind, TurboQuantProd, _centroid_levels_for


class CalibrationMode(str, Enum):
    """Which codebook shape to fit."""

    PAPER_POW2 = "paper_pow2"
    TERNARY_158 = "ternary_158"


def _flatten_unit_rotated(
    samples: torch.Tensor,
    Pi: torch.Tensor,
    *,
    max_values: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Unit-normalize rows, project with Pi, return 1D float tensor of y coordinates."""
    x = samples.reshape(-1, Pi.shape[0]).float()
    n = x.shape[0]
    if n == 0:
        raise ValueError("samples must be non-empty")
    xn = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
    dev = Pi.device
    y = (xn.to(dev) @ Pi.T).float().reshape(-1)
    nv = y.numel()
    if nv > max_values:
        idx = torch.randperm(nv, generator=generator, device=y.device)[:max_values]
        y = y[idx]
    return y


def kmeans_1d(
    values: torch.Tensor,
    k: int,
    *,
    n_iter: int = 40,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Lloyd on scalars. Returns ascending sorted centroids of shape ``[k]`` (float32 on ``values.device``).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    x = values.float().reshape(-1)
    if x.numel() < k:
        raise ValueError(f"need at least k={k} samples, got {x.numel()}")
    dev = x.device
    if generator is None:
        generator = torch.Generator(device=dev)
        generator.manual_seed(0)
    pick = torch.randperm(x.numel(), generator=generator, device=dev)[:k]
    c = x[pick].clone()
    for _ in range(n_iter):
        dist = torch.abs(x.unsqueeze(1) - c.unsqueeze(0))
        lab = dist.argmin(dim=1)
        c_new = torch.empty_like(c)
        for i in range(k):
            m = lab == i
            if m.any():
                c_new[i] = x[m].mean()
            else:
                c_new[i] = c[i]
        if torch.max(torch.abs(c_new - c)) < 1e-7:
            c = c_new
            break
        c = c_new
    c, _ = torch.sort(c)
    return c


def calibrate_turboquant_from_tensor(
    samples: torch.Tensor,
    *,
    head_dim: int,
    mode: Union[CalibrationMode, str] = CalibrationMode.PAPER_POW2,
    bits: int = 3,
    seed: int = 0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    max_samples: int = 1_048_576,
    kmeans_iters: int = 40,
) -> TurboQuantProd:
    """
    Build a quantizer with Π, S from ``seed``, then replace centroids by k-means on projected data.

    Parameters
    ----------
    samples
        Tensor with last dim ``head_dim`` (e.g. ``[B, H, T, D]`` or ``[N, D]``).
    mode
        ``paper_pow2`` → ``K = 2^(bits-1)``. ``ternary_158`` → ``K = 3`` (``bits`` only affects metadata;
        use ``bits=3`` when serializing next to paper-style runs).
    max_samples
        Max number of scalar ``y`` entries used (subsampled randomly if exceeded).
    """
    if isinstance(mode, str):
        mode = CalibrationMode(mode)
    if samples.shape[-1] != head_dim:
        raise ValueError(f"samples last dim {samples.shape[-1]} != head_dim {head_dim}")

    codebook: CodebookKind = "ternary" if mode == CalibrationMode.TERNARY_158 else "paper"
    k = _centroid_levels_for(bits, codebook)

    base = TurboQuantProd(
        bits=bits,
        head_dim=head_dim,
        device=device,
        seed=seed,
        dtype=dtype,
        codebook=codebook,
    )
    gen = torch.Generator(device=base.Pi.device)
    gen.manual_seed(int(seed))
    y = _flatten_unit_rotated(samples, base.Pi, max_values=int(max_samples), generator=gen)
    centroids = kmeans_1d(y, k, n_iter=kmeans_iters, generator=gen).to(device=base.device, dtype=dtype)

    return TurboQuantProd(
        bits=bits,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        codebook=codebook,
        Pi=base.Pi.detach().clone(),
        S=base.S.detach().clone(),
        centroids=centroids,
    )


def calibrate_turboquant_from_batches(
    batches: Iterable[torch.Tensor],
    *,
    head_dim: int,
    mode: Union[CalibrationMode, str] = CalibrationMode.PAPER_POW2,
    bits: int = 3,
    seed: int = 0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    max_samples: int = 1_048_576,
    kmeans_iters: int = 40,
) -> TurboQuantProd:
    """
    Same as :func:`calibrate_turboquant_from_tensor`, but concatenates tensors from an iterable
    until ``max_samples`` scalars are collected (or iterator ends).
    """
    if isinstance(mode, str):
        mode = CalibrationMode(mode)

    codebook: CodebookKind = "ternary" if mode == CalibrationMode.TERNARY_158 else "paper"
    k = _centroid_levels_for(bits, codebook)

    base = TurboQuantProd(
        bits=bits,
        head_dim=head_dim,
        device=device,
        seed=seed,
        dtype=dtype,
        codebook=codebook,
    )
    gen = torch.Generator(device=base.Pi.device)
    gen.manual_seed(int(seed))

    chunks: list[torch.Tensor] = []
    total = 0
    for b in batches:
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b)
        if b.shape[-1] != head_dim:
            raise ValueError(f"batch last dim {b.shape[-1]} != head_dim {head_dim}")
        x = b.reshape(-1, head_dim).float()
        xn = x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        y = (xn.to(base.Pi.device) @ base.Pi.T).float().reshape(-1)
        chunks.append(y)
        total += y.numel()
        if total >= max_samples:
            break

    if not chunks:
        raise ValueError("batches iterable produced no tensors")

    y_all = torch.cat(chunks, dim=0)
    if y_all.numel() > max_samples:
        idx = torch.randperm(y_all.numel(), generator=gen, device=y_all.device)[:max_samples]
        y_all = y_all[idx]

    centroids = kmeans_1d(y_all, k, n_iter=kmeans_iters, generator=gen).to(device=base.device, dtype=dtype)

    return TurboQuantProd(
        bits=bits,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        codebook=codebook,
        Pi=base.Pi.detach().clone(),
        S=base.S.detach().clone(),
        centroids=centroids,
    )
