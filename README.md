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

## K vs V budget — rule of thumb

The `turbo_attn_function` hooks attention so the just-arrived K/V contribute *exactly*
to this step's output and only get quantized when stored for future steps. So a
single-shot prefill is bit-exact to fp; quantization only shows up on tokens that
attend against the cached prefix. Sweep below is on gpt2 with the demo notebook's
prefill + decode split (48 prefill / 13 decode tokens, decode-row metrics):

| `v_bits` | compression | top-1 agreement | cos sim |
|---|---|---|---|
| 4 | 3.37x | 100% | 1.000 |
| 3 | 3.76x | 100% | 1.000 |
| 2 | 4.27x | 100% | 0.956 |
| 1 | 4.92x | 100% | 0.956 |

V quantization adds zero-mean noise to the attention output — scales logit
*magnitudes* but doesn't shift their *ranking*. Top-1 is the right metric, and
it stays at 100% across the whole V sweep.

QK-norm models (Qwen3, OLMo-2, …) need a higher K budget because QK-norm sharpens
the softmax, which amplifies any K-side IP estimation noise. `k_bits=6, v_bits=2`
is the recommended starting point for Qwen3-class models.
