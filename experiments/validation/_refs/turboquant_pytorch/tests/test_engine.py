import torch
import unittest
from turboquant import TurboQuantEngine

class TestTurboQuant(unittest.TestCase):
    def test_reconstruction_fidelity(self):
        d, b = 512, 2
        engine = TurboQuantEngine(d=d, b=b)
        x = torch.randn(d)
        
        idx, qjl, gamma = engine.encode(x)
        x_hat = engine.decode(idx, qjl, gamma)
        
        similarity = torch.nn.functional.cosine_similarity(x.unsqueeze(0), x_hat.unsqueeze(0))
        # 確保 2-bit 量化的相似度至少在 0.8 以上 (根據你的 Benchmark)
        self.assertGreater(similarity.item(), 0.8)

if __name__ == '__main__':
    unittest.main()