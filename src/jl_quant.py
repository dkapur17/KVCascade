"""1-bit Johnson-Lindenstrauss sketch for inner-product estimation."""

import math

import torch


class JLQuantizer:
    """1-bit JL sketch: stores ||x|| and sign(G x). Batched on last dim."""

    def __init__(self, dim: int, m: int, seed: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.m = m
        self.device = device
        self.dtype = dtype
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        self.G = torch.randn(m, dim, generator=g, dtype=dtype).to(device=device)

    def encode(self, x: torch.Tensor):
        """x: [..., D] -> (rho[...], signs[..., m] int8 in {-1, +1})."""
        rho = x.norm(dim=-1)
        signs = torch.sign(x @ self.G.T)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return rho, signs.to(torch.int8)

    def estimate_ip(self, rho: torch.Tensor, signs: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
        """Element-wise IP estimate. signs:[...,m], y:[...,D] -> [...]."""
        gy = y @ self.G.T
        return math.sqrt(math.pi / 2) * rho * (signs.to(y.dtype) * gy).mean(dim=-1)

    def estimate_ip_pairwise(self, rho: torch.Tensor, signs: torch.Tensor,
                              y: torch.Tensor) -> torch.Tensor:
        """Pairwise IP. rho:[...,K], signs:[...,K,m], y:[...,Q,D] -> [...,Q,K]."""
        gy = y @ self.G.T                                   # [..., Q, m]
        s = signs.to(y.dtype)                               # [..., K, m]
        scores = gy @ s.transpose(-1, -2)                   # [..., Q, K]
        return math.sqrt(math.pi / 2) * rho.unsqueeze(-2) * scores / self.m
