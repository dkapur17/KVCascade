# TurboAttention

Bit-packed KV-cache quantization (TurboQuant) plus a drop-in attention replacement
that hooks into HuggingFace's `ALL_ATTENTION_FUNCTIONS` dispatcher — so it works on
any modern decoder (vanilla / RoPE / QK-norm / GQA) without touching the model's
pre-SDPA pipeline.

## Layout

```
src/
  lloyd_max.py    LloydMaxCodebook — Lloyd-Max codebook for unit Gaussians
  polar_quant.py  PolarQuant       — norm + Haar-rotated Lloyd-Max, batched on last dim
  jl_quant.py     JLQuantizer      — 1-bit JL sketch for inner-product estimation
  turbo_quant.py  TurboQuant       — Polar coarse code + JL residual sketch
  turbo_attn.py   bit packing, TurboQuantKVCache, HF dispatcher integration

notebooks/
  turboquant.ipynb        original numpy walkthrough of every quantizer
  turboattn_gpt2.ipynb    PyTorch port + first GPT-2 integration
  turbo_attn_demo.ipynb   cross-architecture demo (GPT-2 / SmolLM2 / Qwen3) +
                          V-bit tradeoff sweep
```

## Quick start

```python
import sys; sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbo_attn import TurboQuantKVCache, install_turbo_attention

model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
tok   = AutoTokenizer.from_pretrained("gpt2")
ids   = tok("hello world " * 16, return_tensors="pt").input_ids

cfg = model.config
cache = TurboQuantKVCache(
    num_layers=cfg.n_layer, batch_size=1,
    num_heads=cfg.n_head, num_kv_heads=cfg.n_head,
    head_dim=cfg.n_embd // cfg.n_head,
    k_bits=4, v_bits=2,        # asymmetric budgets — see "K vs V" below
)
install_turbo_attention(model, cache)
out = model(ids, use_cache=False)
print(f"{cache.bytes_total()} bytes for {ids.shape[-1]} tokens")
```

For RoPE / QK-norm models (Llama, Qwen, etc.) just swap the model name. The
dispatcher hook intercepts only the SDPA step — RoPE, QK-norm, GQA, sliding window,
and anything else the model does to Q/K/V before SDPA flows through unchanged.

## Key ideas

**TurboQuant** decomposes a vector into

  1. norm (fp scalar),
  2. Haar-rotated direction quantized with Lloyd-Max at `bits-1` bits per coord,
  3. residual in rotated space sketched with a 1-bit JL projection (m sign bits).

The Haar rotation Gaussianizes the unit-sphere directions so a single Lloyd-Max
codebook (computed once for `N(0, 1)`) is near-optimal across all heads / layers.
The JL residual gives an unbiased inner-product estimator with variance `O(1/m)`.

**Asymmetric K/V budgets.** K errors propagate through softmax, so K wants the
full TurboQuant treatment. V errors get averaged by `attn @ V` (and Lloyd-Max
centroids are unbiased per-coordinate), so V tolerates aggressive PolarQuant — `bits=2`
is usually free. Pass `k_bits` and `v_bits` to `TurboQuantKVCache` independently.

**GQA-aware storage.** K/V live at kv-head granularity; the cache expands to
query-head count only at attention-compute time. So compression vs an fp16 KV cache
stays apples-to-apples on grouped-query models.

**HF dispatcher integration.** `turbo_attn.py` registers a function under
`transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["turbo"]`. `install_turbo_attention(model, cache)`
flips `model.config._attn_implementation` to `"turbo"` and stamps the cache onto every
attention module via `module.layer_idx`. No model-class-specific wrappers.

## K vs V budget — sweep across architectures

The hybrid attention path makes prefill bit-exact to fp (just-arrived K/V contribute
*exactly* this step; only the cached prefix goes through the IP estimator). So
quantization quality only shows up on tokens that attend against the cached prefix.

The tables below run an 80/20 prefill→decode split on a 223-token prompt and compare
the decode-row logits against an fp single-shot reference. K-budget grid: `{3, 4, 5, 6}`,
V-budget grid: `{1, 2, 4}`. **Bold** rows are the recommended config per model.

### gpt2 — 12 layers, MHA, learned positional

| k_bits | v_bits | compression | cos sim | top-1 |
|---|---|---|---|---|
| 3 | 1 | 5.82× | 0.771 | 97.8% |
| 3 | 2 | 4.92× | 0.911 | 100.0% |
| 3 | 4 | 3.76× | 0.911 | 100.0% |
| 4 | 1 | 4.92× | 0.731 |  95.6% |
| **4** | **2** | **4.27×** | **0.911** | **100.0%** |
| 4 | 4 | 3.37× | 1.000 | 100.0% |
| 5 | 1 | 4.27× | 0.832 | 100.0% |
| 5 | 2 | 3.76× | 1.000 | 100.0% |
| 5 | 4 | 3.05× | 1.000 | 100.0% |
| 6 | 1 | 3.76× | 0.910 | 100.0% |
| 6 | 2 | 3.37× | 1.000 | 100.0% |
| 6 | 4 | 2.78× | 1.000 | 100.0% |

### SmolLM2-135M — 30 layers, RoPE, GQA (9 q-heads → 3 kv-heads)

| k_bits | v_bits | compression | cos sim | top-1 |
|---|---|---|---|---|
| 3 | 1 | 5.82× | 0.517 |  91.1% |
| 3 | 2 | 4.92× | 0.851 |  95.6% |
| 3 | 4 | 3.76× | 0.852 |  95.6% |
| 4 | 1 | 4.92× | 0.552 |  95.6% |
| **4** | **2** | **4.27×** | **0.860** | **100.0%** |
| 4 | 4 | 3.37× | 0.942 | 100.0% |
| 5 | 1 | 4.27× | 0.579 |  95.6% |
| 5 | 2 | 3.76× | 0.946 | 100.0% |
| 5 | 4 | 3.05× | 0.980 | 100.0% |
| 6 | 1 | 3.76× | 0.554 |  95.6% |
| 6 | 2 | 3.37× | 0.950 | 100.0% |
| 6 | 4 | 2.78× | 0.991 | 100.0% |

### Qwen3-0.6B — 28 layers, RoPE + QK-norm, GQA (16 q-heads → 8 kv-heads)

| k_bits | v_bits | compression | cos sim | top-1 |
|---|---|---|---|---|
| 3 | 1 | 6.74× | 0.561 |   8.9% |
| 3 | 2 | 5.57× | 0.636 |   8.9% |
| 3 | 4 | 4.13× | 0.693 |   6.7% |
| 4 | 1 | 5.57× | 0.520 |  26.7% |
| 4 | 2 | 4.74× | 0.620 |  13.3% |
| 4 | 4 | 3.66× | 0.704 |  22.2% |
| 5 | 1 | 4.74× | 0.679 |  88.9% |
| 5 | 2 | 4.13× | 0.783 |  88.9% |
| 5 | 4 | 3.28× | 0.838 |  91.1% |
| 6 | 1 | 4.13× | 0.801 |  95.6% |
| **6** | **2** | **3.66×** | **0.888** | **100.0%** |
| 6 | 4 | 2.98× | 0.923 | 100.0% |

### Recommended configurations

| model | `(k_bits, v_bits)` | compression | top-1 |
|---|---|---|---|
| gpt2 | (4, 2) | 4.27× | 100% |
| SmolLM2-135M | (4, 2) | 4.27× | 100% |
| Qwen3-0.6B | (6, 2) | 3.66× | 100% |

Two takeaways:

1. **`v_bits=2` is the sweet spot across all three architectures.** `v_bits=1` tanks
   cos sim consistently (1-bit V is just sign quantization — too lossy even after
   averaging). `v_bits=4` is wasted budget. The asymmetric-K/V intuition holds
   across vanilla, RoPE, and QK-norm.

2. **K budget scales with attention sharpness.** Vanilla and RoPE-only models work
   at `k_bits=4`. QK-norm sharpens the softmax (K coordinates have unit RMS, so
   scaled scores are larger), which amplifies K-side IP noise through the softmax
   non-linearity. The cliff between `k=4` and `k=5` on Qwen3 is striking: top-1
   jumps from 13% to 88% with one extra bit on K.
