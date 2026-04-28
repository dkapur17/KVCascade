"""Lloyd-Max quantization codebook for unit Gaussian inputs."""

import functools

import numpy as np
import torch
from scipy.stats import norm as gaussian


def _centroids_np(b):
    pdf = gaussian.pdf(b)
    cdf = gaussian.cdf(b)
    return (pdf[:-1] - pdf[1:]) / (cdf[1:] - cdf[:-1])


def _boundaries_np(c):
    inner = (c[:-1] + c[1:]) / 2
    return np.concatenate([[-np.inf], inner, [np.inf]])


def _distortion_np(b, c):
    cdf = gaussian.cdf(b)
    p = cdf[1:] - cdf[:-1]
    return 1.0 - float((p * c * c).sum())


@functools.lru_cache(maxsize=None)
def _solve_lloyd_max(bits: int, tol: float):
    """Compute (boundaries, centroids, distortion) for the unit-Gaussian Lloyd-Max
    codebook with 2**bits levels. Pure function of (bits, tol); memoized so repeated
    instantiations across layers/quantizers reuse the same solution."""
    k = 2 ** bits
    boundaries = np.concatenate([
        [-np.inf],
        gaussian.ppf(np.arange(1, k) / k),
        [np.inf],
    ])
    centroids = _centroids_np(boundaries)
    while True:
        old = boundaries.copy()
        boundaries = _boundaries_np(centroids)
        centroids = _centroids_np(boundaries)
        if np.max(np.abs(boundaries[1:-1] - old[1:-1])) < tol:
            break
    distortion = _distortion_np(boundaries, centroids)
    return boundaries, centroids, distortion


class LloydMaxCodebook:
    """Lloyd-Max codebook for a unit Gaussian. Boundaries and centroids are stored
    as torch tensors. quantize/dequantize work on tensors of any shape (last dim is data).
    The numpy solve is memoized by (bits, tol) at module level."""

    def __init__(self, bits: int, tol: float = 1e-8,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32):
        assert 1 <= bits <= 6, "bits must be in [1, 6]"
        self.bits = bits
        self.tol = tol
        self.k = 2 ** bits
        self.device = device
        self.dtype = dtype

        boundaries, centroids, distortion = _solve_lloyd_max(bits, tol)
        self.distortion = float(distortion)
        self.boundaries = torch.from_numpy(boundaries).to(device=device, dtype=dtype)
        self.centroids = torch.from_numpy(centroids).to(device=device, dtype=dtype)
        self._interior = self.boundaries[1:-1].contiguous()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self._interior, x.contiguous())
        return idx.to(torch.uint8) if self.bits <= 8 else idx.to(torch.int16)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centroids[idx.long()]

    def __repr__(self):
        return f"LloydMaxCodebook(bits={self.bits}, k={self.k}, distortion={self.distortion:.4f})"
