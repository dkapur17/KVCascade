"""Polar (norm + Haar-rotated Lloyd-Max) quantization. Batched on last dim."""

import math

import torch

from lloyd_max import LloydMaxCodebook


def haar_orthogonal(d: int, seed: int | None,
                    device: torch.device | None,
                    dtype: torch.dtype) -> torch.Tensor:
    """Sample a d x d Haar-uniform orthogonal matrix via QR of a Gaussian."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    sign = torch.sign(torch.diagonal(R))
    return (Q * sign).to(device=device, dtype=dtype)


class PolarQuant:
    """Norm + Lloyd-Max indices on a Haar-rotated unit vector. Batched on last dim."""

    def __init__(self, bits: int, dim: int, tol: float = 1e-8,
                 seed: int | None = None,
                 R: torch.Tensor | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32):
        self.bits = bits
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.sigma = 1.0 / math.sqrt(dim)
        self.codebook = LloydMaxCodebook(bits, tol, device=device, dtype=dtype)
        self.R = (haar_orthogonal(dim, seed, device, dtype)
                  if R is None else R.to(device=device, dtype=dtype))

    def encode(self, x: torch.Tensor):
        """x: [..., D] -> (norms[...], indices[..., D] uint8)."""
        norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = (x / norms) @ self.R.T
        idx = self.codebook.quantize(u / self.sigma)
        return norms.squeeze(-1), idx

    def decode_rotated(self, idx: torch.Tensor) -> torch.Tensor:
        """Reconstruction in rotated unit-sphere space (no norm applied)."""
        return self.codebook.dequantize(idx) * self.sigma

    def decode(self, norms: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        u_hat = self.decode_rotated(idx)
        return norms.unsqueeze(-1) * (u_hat @ self.R)
