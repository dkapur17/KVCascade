"""TurboQuant: combined polar/Lloyd-Max code plus 1-bit JL sketch of the residual."""

from dataclasses import dataclass

import torch

from polar_quant import PolarQuant
from jl_quant import JLQuantizer


@dataclass
class QuantizedKV:
    """Compressed representation of a (batched) tensor with shape [..., D]."""
    x_norm: torch.Tensor       # [...]
    x_indices: torch.Tensor    # [..., D]
    res_norm: torch.Tensor     # [...]
    res_signs: torch.Tensor    # [..., m]


class TurboQuant:
    """Polar/Lloyd-Max coarse code + 1-bit JL residual sketch in rotated space."""

    def __init__(self, bits: int, dim: int, m: int, tol: float = 1e-8,
                 seed: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32):
        assert bits >= 2, "need bits >= 2 (mse_quantizer uses bits-1)"
        self.bits = bits
        self.dim = dim
        self.m = m
        self.device = device
        self.dtype = dtype

        self.mse_quantizer = PolarQuant(bits - 1, dim, tol, seed=seed,
                                         device=device, dtype=dtype)
        # Independent seed for JL so the projection matrix G is independent of R.
        seed_jl = None if seed is None else seed + 1
        self.ip_quantizer = JLQuantizer(dim, m, seed=seed_jl,
                                         device=device, dtype=dtype)
        self.R = self.mse_quantizer.R

    def quantize(self, x: torch.Tensor) -> QuantizedKV:
        """x: [..., D] -> QuantizedKV with matching leading dims."""
        x_norm, x_indices = self.mse_quantizer.encode(x)
        u = (x / x_norm.unsqueeze(-1).clamp_min(1e-12)) @ self.R.T
        u_hat = self.mse_quantizer.decode_rotated(x_indices)
        res_norm, res_signs = self.ip_quantizer.encode(u - u_hat)
        return QuantizedKV(x_norm, x_indices, res_norm, res_signs)

    def estimate_ip(self, q: QuantizedKV, y: torch.Tensor) -> torch.Tensor:
        """Element-wise IP estimate. y:[...,D] -> [...]."""
        y_rot = y @ self.R.T
        u_hat = self.mse_quantizer.decode_rotated(q.x_indices)
        mse = (u_hat * y_rot).sum(dim=-1)
        jl = self.ip_quantizer.estimate_ip(q.res_norm, q.res_signs, y_rot)
        return q.x_norm * (mse + jl)

    def estimate_ip_pairwise(self, q: QuantizedKV, y: torch.Tensor) -> torch.Tensor:
        """Pairwise IP. y:[...,Q,D]; q stores K keys -> [...,Q,K]."""
        y_rot = y @ self.R.T                                    # [..., Q, D]
        u_hat = self.mse_quantizer.decode_rotated(q.x_indices)  # [..., K, D]
        mse = y_rot @ u_hat.transpose(-1, -2)                   # [..., Q, K]
        jl = self.ip_quantizer.estimate_ip_pairwise(q.res_norm, q.res_signs, y_rot)
        return q.x_norm.unsqueeze(-2) * (mse + jl)
