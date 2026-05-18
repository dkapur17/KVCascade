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
    """Norm + Lloyd-Max indices on a Haar-rotated unit vector. Batched on last dim.

    Exposes a `TurboQuant`-shaped duck-typed interface (`.mse_quantizer`,
    `.estimate_ip_pairwise`) so call sites that operate on either quantizer
    don't need to branch. In TurboQuant terminology, this is the MSE variant
    (no JL residual sketch).
    """

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

    # --- TurboQuant-shaped surface so kvcascade.TurboBuffer can use us as the K quantizer ---
    @property
    def mse_quantizer(self):
        # In MSE mode, the "mse_quantizer" *is* this object.
        return self

    def estimate_ip_pairwise(self, qkv_like, y: torch.Tensor) -> torch.Tensor:
        """Direct IP via dequantized K. Drop-in replacement for TurboQuant.estimate_ip_pairwise.

        qkv_like: object with `.x_norm` [..., K] and `.x_indices` [..., K, D] —
                  a QuantizedKV (we use only the MSE-coarse fields; JL fields are ignored).
        y       : [..., Q, D].
        Returns : [..., Q, K].
        """
        K_hat = self.decode(qkv_like.x_norm, qkv_like.x_indices)        # [..., K, D]
        return y @ K_hat.transpose(-1, -2)

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

    # --- TurboQuant.quantize-shaped: returns a QuantizedKV with zero JL fields ---
    def quantize(self, x: torch.Tensor):
        """Returns a QuantizedKV with `res_norm=0`, `res_signs=zeros`. Drop-in for
        TurboQuant.quantize so TurboBuffer.encode_batch can stay uniform."""
        from turbo_quant import QuantizedKV
        x_norm, x_indices = self.encode(x)
        # 0-size last-dim residual signs — these are not used in MSE mode but the
        # signature matches what TurboBuffer.encode_batch expects.
        res_norm = torch.zeros_like(x_norm)
        res_signs = torch.zeros(*x_norm.shape, 0, device=x.device, dtype=torch.int8)
        return QuantizedKV(x_norm=x_norm, x_indices=x_indices,
                           res_norm=res_norm, res_signs=res_signs)
