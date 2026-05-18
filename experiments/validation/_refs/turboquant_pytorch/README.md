# TurboQuant-Pytorch
<h3 >High-Performance Vector Quantization Engine 

A Implementation of the turboquant paper by Pytorch C++</h3>



[![Language](https://img.shields.io/badge/language-python%20%2F%20C%2B%2B-blue.svg)](https://www.python.org/)
[![Library](https://img.shields.io/badge/library-PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English Version](README.md) | [繁體中文版本](README_zh.md)

---

<a name="english-version"></a>
## English Version

**TurboQuant** is a specialized, high-performance vector quantization library designed for Large Language Models (LLMs) and vector search applications. residual compensation.

### Key Features
- **Turbo-Charged C++ Core**: Core operations like rotation, projection, and quantization are implemented in optimized C++ for millisecond-level inference.
- **Lloyd-Max Optimization**: Automatically computes the most efficient centroids for Gaussian distributions using Scipy's K-Means.
- **Unbiased Residual Compensation**: Uses QJL signs to preserve vector magnitude and direction, minimizing cumulative error in deep networks.
- **Smart Matrix Caching**: Automatically caches trained centroids ($\mathcal{C}$) and orthogonal matrices ($\Pi, S$) for instant engine startup.
- **Adaptive Dimension Support**: Fully compatible with any dimension $d$ and any bit-rate $b$ (from 1-bit to 8-bit).
TurboQuant is a high-performance quantization library designed for Large Language Models (LLMs) and vector search applications. By offloading core computations to C++ and integrating mathematical optimization, TurboQuant significantly reduces memory overhead while maintaining near-lossless precision.

### Key Highlights

* **Extreme C++ Acceleration**: Core operations like Matrix Rotation, Projection, and quantization logic are deeply optimized using C++/LibTorch to achieve millisecond-level inference.
* **Lloyd-Max Mathematical Optimization**: Automatically calculates optimal centroids for Gaussian distributions using Scipy-based K-Means, ensuring high-precision quantization.
* **Unbiased Residual Compensation**: Utilizes QJL sign bits to preserve vector direction and magnitude, solving cumulative error issues in deep neural networks.
* **Intelligent Matrix Caching** : Automatically caches trained centroids and orthogonal matrices (Pi, S) for "instant-on" engine startup.
* **High Elasticity & Customization**: Supports arbitrary dimensions d and dynamic bitrate switching from 1-bit to 8-bit.
## **Comparison**


  <img src="turboquant_benchmark_large.png" alt="TurboQuant Performance Benchmark">

### *How to read the benchmark chart:*
- **Y-Axis (Fidelity)**: Higher Cosine Similarity means more accurate vector reconstruction.
- **X-Axis (Latency)**: Lower values indicate faster C++/LibTorch execution.
- **Bubble Size (Memory)**: **Larger bubbles represent higher memory compression ratios.**
  - **1-bit**: 32x Compression (Largest Bubble)
  - **2-bit**: 16x Compression
  - **4-bit**: 8x Compression
  - **Int8 (Baselines)**: 4x Compression (Smallest Bubble)

## Installation
```bash
# Clone the repository
git clone https://github.com/ericoder960803/TurboQuant.git

cd TurboQuant

# Install in editable mode (Builds C++ extension automatically)
pip install -e .
```
##  Usage

TurboQuant provides a seamless PyTorch-like API. You can easily integrate it into your inference pipeline.

### Quick Start

```python
import torch
from turboquant import TurboQuantEngine

# 1. Initialize the engine
# d: vector dimension, b: target bit-rate (1, 2, 4, or 8)
d = 1024
b = 2
engine = TurboQuantEngine(d=d, b=b, cache=True)

# 2. Prepare your high-precision vector (FP32)
x = torch.randn(d)

# 3. Encode (Compression)
# idx: Lloyd-Max centroids indices
# qjl: 1-bit residual signs
# gamma: Dynamic scaling factor for reconstruction
idx, qjl, gamma = engine.encode(x)

# 4. Decode (Decompression)
x_hat = engine.decode(idx, qjl, gamma)

# 5. Check Fidelity
similarity = torch.nn.functional.cosine_similarity(x.unsqueeze(0), x_hat.unsqueeze(0))
print(f"Reconstruction Cosine Similarity: {similarity.item():.4f}")

```
## Usage Examples

### 1. LLM KV-Cache Management (16x Memory Saving)
Ideal for long-context LLM inference (e.g., Llama-3) where KV-cache memory is the primary bottleneck.

```python
import torch
from turboquant import TurboQuantEngine

class TurboQuantKVCache:
    def __init__(self, dim=4096, bits=2):
        # 2-bit quantization compresses 4KB into 0.25KB
        self.engine = TurboQuantEngine(d=dim, b=bits, cache=True)
        self.cache = [] 

    def push(self, key_tensor):
        """Compress and store in cache"""
        packet = self.engine.encode(key_tensor)
        self.cache.append(packet)

    def fetch_all(self):
        """Decompress all vectors for Attention calculation"""
        if not self.cache: return None
        return torch.stack([self.engine.decode(*p) for p in self.cache])
# Usage
kv_manager = TurboQuantKVCache(dim=4096, bits=2)
kv_manager.push(torch.randn(4096)) # Encode new key
keys = kv_manager.fetch_all()      # Restore for Attention [Seq_Len, Dim]

```

### 2. High-Speed Vector Search
Enables high-fidelity vector databases or RAG systems with minimal storage footprint.
```python
import torch
from turboquant import TurboQuantEngine

# 1. Setup Database (10,000 vectors)
D, B = 1024, 2
engine = TurboQuantEngine(d=D, b=B)
database = torch.randn(10000, D)

# 2. Offline Compression
compressed_db = [engine.encode(v) for v in database]

# 3. Online Search
query = torch.randn(D)
reconstructed_db = torch.stack([engine.decode(*p) for p in compressed_db])
scores = torch.nn.functional.cosine_similarity(query.unsqueeze(0), reconstructed_db)

# 4. Get Top-K
top_values, top_indices = torch.topk(scores, k=5)
print(f"Top Indices: {top_indices.tolist()}")
```
###  Mathematical Foundation
The reconstruction $\hat{x}$ is computed as:
$$\hat{x} = \Pi^T ( \mathcal{C}_{idx} + \gamma \cdot \sqrt{\frac{\pi}{2d}} \cdot S^T q_{jnl} )$$
Where:
- $\Pi$: Orthogonal Rotation Matrix
- $\mathcal{C}$: Lloyd-Max Optimal Centroids
- $S$: QJL Projection Matrix
##  Citation

If you find **TurboQuant-PyTorch** useful in your research or project, please cite the original paper:

### Original Paper (arXiv:2504.19874)
```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
@misc{ericliam2026turboquant,
  author = {Eric Liam},
  title = {TurboQuant-PyTorch: High-Performance C++/LibTorch Implementation},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{[https://github.com/ericoder960803/TurboQuant-PyTorch](https://github.com/ericoder960803/TurboQuant-PyTorch)}}
}