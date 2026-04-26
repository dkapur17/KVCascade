"""Lloyd-Max quantization codebook for unit Gaussian inputs."""

import numpy as np
import torch
from scipy.stats import norm as gaussian


class LloydMaxCodebook:
    """Lloyd-Max codebook for a unit Gaussian. Boundaries and centroids are stored
    as torch tensors. quantize/dequantize work on tensors of any shape (last dim is data)."""

    def __init__(self, bits: int, tol: float = 1e-8,
                 device: torch.device | None = None,
                 dtype: torch.dtype = torch.float32):
        assert 1 <= bits <= 6, "bits must be in [1, 6]"
        self.bits = bits
        self.tol = tol
        self.k = 2 ** bits
        self.device = device
        self.dtype = dtype

        boundaries = np.concatenate([[-np.inf],
                                     [gaussian.ppf(i / self.k) for i in range(1, self.k)],
                                     [np.inf]])
        centroids = self._centroids_np(boundaries)
        while True:
            old = boundaries.copy()
            boundaries = self._boundaries_np(centroids)
            centroids = self._centroids_np(boundaries)
            if np.max(np.abs(boundaries[1:-1] - old[1:-1])) < tol:
                break

        self.distortion = float(self._distortion_np(boundaries, centroids))
        self.boundaries = torch.from_numpy(boundaries).to(device=device, dtype=dtype)
        self.centroids = torch.from_numpy(centroids).to(device=device, dtype=dtype)
        self._interior = self.boundaries[1:-1].contiguous()

    @staticmethod
    def _centroids_np(b):
        out = []
        for b1, b2 in zip(b[:-1], b[1:]):
            out.append((gaussian.pdf(b1) - gaussian.pdf(b2)) / (gaussian.cdf(b2) - gaussian.cdf(b1)))
        return np.array(out)

    @staticmethod
    def _boundaries_np(c):
        out = [-np.inf]
        for c1, c2 in zip(c[:-1], c[1:]):
            out.append((c1 + c2) / 2)
        out.append(np.inf)
        return np.array(out)

    @staticmethod
    def _distortion_np(b, c):
        acc = 0.0
        for b1, b2, ci in zip(b[:-1], b[1:], c):
            p = gaussian.cdf(b2) - gaussian.cdf(b1)
            acc += p * ci * ci
        return 1.0 - acc

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self._interior, x.contiguous())
        return idx.to(torch.uint8) if self.bits <= 8 else idx.to(torch.int16)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centroids[idx.long()]

    def __repr__(self):
        return f"LloydMaxCodebook(bits={self.bits}, k={self.k}, distortion={self.distortion:.4f})"
