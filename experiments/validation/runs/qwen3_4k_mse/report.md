# KVCascade evaluation: `Qwen/Qwen3-0.6B`

- **Generated**: 2026-05-18 18:55:42
- **Total runtime**: 16.0 minutes
- **Samples**: 20 non-overlapping wikitext-103 chunks
- **Context length**: 4096 (prefill 4032, decode 64)
- **Dtype**: `bfloat16`, **device**: `cuda`, **seed**: 42
- **Quant tier**: `k_bits=6`, `v_bits=2`, single tier

## Model

| Property | Value |
|---|---|
| Name | `Qwen/Qwen3-0.6B` |
| Layers | 28 |
| Query heads | 16 |
| KV heads | 8 |
| Head dim | 128 |
| fp16 baseline cache | 458,752 KiB |

## Experiment 2: Iso-byte head-to-head

At the same byte budget (= uniform's), compare four cache strategies.

| Config | Bytes (KiB) | Compression vs fp16 | Top-1 | Cos sim | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|---|---|---|
| full-fp (ref) | 458,752 | 1.00× | 100.0% ± 0.0% | 1.0000 ± 0.0000 | — | — |
| uniform k=6/v=2 | 118,272 | 3.88× | 85.9% ± 5.0% | 0.9818 ± 0.0051 | 2641.7 | 9.0 |
| H2O (ring=0) | 118,272 | 3.88× | 21.8% ± 5.2% | 0.5587 ± 0.0625 | 1825.0 | 10.3 |
| H2O (ring=8) | 118,272 | 3.88× | 74.9% ± 4.7% | 0.9677 ± 0.0133 | 1817.7 | 8.6 |
| KVCascade (ring + fp + quant) | 118,272 | 3.88× | 95.4% ± 2.6% | 0.9971 ± 0.0042 | 1090.2 | 4.7 |

![Iso-byte bar chart](figures/exp2_isobyte.png)

**Δ at iso-byte**: KVCascade vs uniform = +9.5 pp.
  H2O (ring=0) vs uniform = -64.1 pp.
  H2O (ring=8) vs uniform = -10.9 pp.
  Recency-ring lift on H2O = +53.1 pp (adding ring=8 on top of plain H2O).
  Quantization lift on H2O+ring = +20.5 pp (KVCascade adds the quant tier on top of H2O+ring).

---

*Raw per-sample results in `raw.json`. Reproduce with: `eval.py --model Qwen/Qwen3-0.6B --samples 20 --ctx-len 4096 --decode-len 64 --skip-exp1 --skip-viz --out /workspace/kvcascade/experiments/validation/runs/qwen3_4k_mse --quant-mode mse`*