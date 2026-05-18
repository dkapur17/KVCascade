import torch
import numpy as np
import os
from scipy.stats import norm
from scipy.cluster.vq import kmeans
from . import core # Relative import of the compiled C++ extension

class TurboQuantEngine:
    """
    TurboQuant Engine: High-performance Quantization with 
    Lloyd-Max Centroids and QJL Residual Compensation.
    """
    def __init__(self, d: int, b: int, device: str = "cpu", cache: bool = True, cache_dir: str = ".tq_cache"):
        self.d = d
        self.b = b
        self.device = device
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Load or Generate Lloyd-Max Centroids
        self.centroids = self._get_centroids(cache).to(device)
        
        # 2. Load or Generate Orthogonal Matrices (Pi and S)
        self.pi, self.s = self._get_matrices(cache)
        self.pi = self.pi.to(device)
        self.s = self.s.to(device)
        
        print(f"[TurboQuant] Engine initialized (d={d}, b={b}, device={device})")

    def _get_centroids(self, use_cache: bool):
        """Generates or loads optimal centroids for a Gaussian distribution."""
        path = os.path.join(self.cache_dir, f"centroids_b{self.b}.pt")
        
        if use_cache and os.path.exists(path):
            print(f"[TurboQuant] Loading cached centroids: {path}")
            return torch.load(path, weights_only=True)
        
        print(f"[TurboQuant] Generating Lloyd-Max centroids for {self.b}-bit...")
        # Sample from standard normal distribution (y is Gaussian after Pi rotation)
        samples = norm.rvs(size=100000).astype(np.float32)
        # Use K-Means to find optimal Lloyd-Max centroids
        centroids, _ = kmeans(samples, 2**self.b)
        centroids.sort()
        tensor = torch.from_numpy(centroids)
        
        if use_cache:
            torch.save(tensor, path)
            print(f"[TurboQuant] Centroids cached to {path}")
        return tensor

    def _get_matrices(self, use_cache: bool):
        """Generates or loads orthogonal matrices Pi and S via QR decomposition."""
        pi_path = os.path.join(self.cache_dir, f"pi_d{self.d}.pt")
        s_path = os.path.join(self.cache_dir, f"s_d{self.d}.pt")
        
        if use_cache and os.path.exists(pi_path) and os.path.exists(s_path):
            print(f"[TurboQuant] Loading cached matrices (Pi, S) for d={self.d}")
            return torch.load(pi_path, weights_only=True), torch.load(s_path, weights_only=True)
        
        print(f"[TurboQuant] Generating orthogonal matrices (QR decomposition) for d={self.d}...")
        # Generate random Gaussian matrices and orthogonalize them
        pi, _ = torch.linalg.qr(torch.randn(self.d, self.d))
        s, _ = torch.linalg.qr(torch.randn(self.d, self.d))
        
        if use_cache:
            torch.save(pi, pi_path)
            torch.save(s, s_path)
            print(f"[TurboQuant] Matrices cached to {self.cache_dir}")
        return pi, s

    def encode(self, x: torch.Tensor, gamma_scale: float = 1.0):
        """
        Quantizes the input vector x.
        Returns: (indices, qjl_symbols, dynamic_gamma)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        
        # Ensure input is on the correct device and flattened
        x = x.to(self.device).view(-1)
        
        # Call the C++ core quantization logic
        idx, qjl = core.quantize(x, self.pi, self.s, self.centroids)
        
        # Calculate dynamic Gamma (Standard deviation of residuals in Pi space)
        # y = Pi * x
        y = torch.mv(self.pi, x)
        # y_q = quantized version in Pi space
        y_q = self.centroids[idx.long()]
        
        # Gamma normalization factor from the paper
        gamma = (torch.norm(y - y_q).item() / np.sqrt(self.d)) * gamma_scale
        
        return idx, qjl, gamma

    def decode(self, idx: torch.Tensor, qjl: torch.Tensor, gamma: float):
        """
        Dequantizes and reconstructs the vector.
        Returns: reconstructed_x
        """
        return core.dequantize(idx, qjl, self.pi, self.s, self.centroids, gamma)

    def clear_cache(self):
        """Utility to clear the local cache directory."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            print(f"[TurboQuant] Cache cleared: {self.cache_dir}")