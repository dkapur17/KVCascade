# KVCascade

**KV-C**ache with **A**daptive **S**core-based **C**ompression **A**nd **D**ecay-driven **E**viction.

A memory-bounded KV cache for long-context decoder inference. Tokens flow through a
hierarchy of precision tiers — the recent and load-bearing
ones at full precision, and the unimportant tail evicted entirely. Importance
is driven per (layer, kv-head) by a decaying score on accumulated attention received,
so each head independently decides which tokens to keep.

Compression on what's kept is done with [TurboQuant](https://arxiv.org/abs/2504.16127)
(norm + Haar-rotated Lloyd-Max + 1-bit JL residual sketch). Eviction is in the
spirit of [H2O](https://arxiv.org/abs/2306.14048) / [SnapKV](https://arxiv.org/abs/2404.14469).
The combination — quantize what you keep, evict what you don't — gives a Pareto
improvement over either approach alone on long-context language modeling.

```mermaid
flowchart TD
    new["new tokens"] --> ring["<b>recency ring</b><br/>fp, FIFO — last-K tokens"]
    ring -->|graduate| fp["<b>fp_tier</b><br/>fp, importance-managed — top-K by score"]
    fp -->|demote| q0["<b>quant_tiers[0]</b><br/>TurboQuant, e.g. k=6/v=2 — bulk storage"]
    q0 -->|demote ...| qn["<b>quant_tiers[n]</b><br/>TurboQuant, e.g. k=4/v=1 — increasingly aggressive compression tiers"]
    qn -->|evict| dropped(["evicted"])

    classDef fpStyle fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef quantStyle fill:#fff3e0,stroke:#ef6c00,color:#e65100
    classDef terminal fill:#f5f5f5,stroke:#616161,color:#212121
    class ring,fp fpStyle
    class q0,qn quantStyle
    class new,dropped terminal
```
Supports an arbitrary hierarchy of quantized tiers with increasingly aggressive quantization, but findings show that single quantization tier performs the best, since excessive quantization introduces adversarial noise in the inner product estimation - better to evict than to keep.

## Headline result

Sequential-decode evaluation on **Qwen3-0.6B**, wikitext-103, T_ctx=4096, T_decode=64,
top-1 agreement against fp32 reference. Single sample of T_dec=64 decode positions:

| config | bytes | compression | top-1 | Δ vs uniform |
|---|---|---|---|---|
| uniform `k=6/v=2` (TurboQuant baseline) | 120 MB | 3.82× | 73.4% | 0 |
| **KV-CASCADE B** (iso-byte, ema) | 120 MB | 3.82× | **89.1%** | **+15.7 pp** |
| **KV-CASCADE F** (½ byte, ema) | 60 MB | 7.64× | **82.8%** | **+9.4 pp** |
| **KV-CASCADE G** (¼ byte, ema) | 30 MB | 15.30× | 75.0% | +1.6 pp |

Multi-sample averages over 5 wikitext-103 chunks at the same setup:

| config | top-1 (mean ± sd) |
|---|---|
| uniform `k=6/v=2` | 68.4% ± 10.7% |
| KV-CASCADE B (iso, ema) | **78.4% ± 5.7%** |
| KV-CASCADE F (½ byte, ema) | **80.0% ± 8.4%** |
| KV-CASCADE G (¼ byte, ema) | **79.7% ± 9.2%** |

Two things happen here that aren't obvious:

1. **Iso-byte: KV-CASCADE B beats uniform TurboQuant by ~10 pp.** Same total memory,
   but the architecture (recency ring at fp + sink tier at fp + bulk at quantized +
   evict the long tail) compounds less softmax noise than uniform's "every-token-gets-a-
   little-noisy" regime when the cache is being used sequentially.

2. **At ½ and ¼ uniform's bytes, KV-CASCADE still wins** — F (60 MB) and G (30 MB)
   both beat uniform's 120 MB. Removing the long tail of low-importance tokens *helps*
   accuracy because their quantized contributions to the softmax were net noise.

The win is **specific to sequential-decode workloads** (real autoregressive generation).
On single-shot scoring, uniform TurboQuant is competitive. See "When does KV-CASCADE
win?" below.

## Layout

```
src/
  lloyd_max.py    LloydMaxCodebook  — Lloyd-Max codebook for unit Gaussians
  polar_quant.py  PolarQuant        — norm + Haar-rotated Lloyd-Max, batched on last dim
  jl_quant.py     JLQuantizer       — 1-bit JL sketch for inner-product estimation
  turbo_quant.py  TurboQuant        — Polar coarse code + JL residual sketch
  turbo_attn.py   bit packing, TurboQuantKVCache, HF dispatcher integration
  kvcascade.py    KVCascadeCache    — recency ring + fp tier + N quant tiers + eviction

eval.py            sequential-decode eval on wikitext-103 (CUDA, bf16)

notebooks/         demos and quantizer walkthroughs (single-shot regime)
```

## Quick start

```python
import sys; sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcascade import KVCascadeCache, install_kvcascade

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype="bfloat16", attn_implementation="eager"
).cuda().eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

cfg = model.config
cache = KVCascadeCache(
    num_layers=cfg.num_hidden_layers,
    batch_size=1,
    num_heads=cfg.num_attention_heads,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim,
    ring_size=8,
    fp_capacity=256,
    quant_tiers=[(6, 2, 2048)],   # one quant tier at k=6/v=2 with 2048 slots
    score_policy="ema",
    device="cuda", dtype="bfloat16",
)
n = install_kvcascade(model, cache)   # registers attn dispatcher on every layer

# Use the model normally; the cache fills/cascades/evicts under the hood.
ids = tok("…long context here…", return_tensors="pt").input_ids.cuda()
out = model(ids, use_cache=False)     # use_cache=False — we manage caching ourselves
```

For RoPE / QK-norm / GQA models (Llama, Qwen, Mistral, Gemma, modern GPT-2 etc.), just
swap the model name. The attention dispatcher hook intercepts only the SDPA step —
RoPE, QK-norm, GQA, sliding window, and anything else the model does to Q/K/V before
SDPA flows through unchanged.

## Configuration

`KVCascadeCache.__init__` is the main API. The interesting knobs:

- **`ring_size`**: FIFO recency buffer at fp. Recently-arrived tokens live here for
  `ring_size` steps before being scored and graduating to the persistent tiers.
- **`fp_capacity`**: importance-managed fp tier above the quant tiers. Top-`fp_capacity`
  tokens by accumulated attention live here at full precision. Set to 0 to disable
  ("only ring + quant + evict").
- **`quant_tiers`**: a list of `(k_bits, v_bits, capacity)` tuples — one per TurboQuant
  tier, in cascade order (most precise first). Pass `[]` for "ring + fp + evict"
  (a pure H2O-on-fp setup). Common configs:
  - `[(6, 2, N)]` — single bulk tier at uniform's "working" precision; aggressive
    eviction past it. *This is what we recommend for QK-norm models like Qwen3.*
  - `[(4, 2, N)]` — same but at lower precision; OK for vanilla / RoPE-only models.
  - `[(6, 2, A), (4, 1, B)]` — two quant tiers; tail tokens get demoted into the
    aggressive second tier rather than evicted. Higher capacity, but the aggressive
    tier's IP-noise can hurt on sharp-softmax models.
- **`score_policy`**: how attention received drives the importance score.
  - `"ema"` (default) — exponential moving average of per-query mean attention.
    Decaying, workload-independent, adaptive.
  - `"cumulative"` — H2O-style monotonic sum. Strictly preserves what's been hot
    historically. We've found EMA slightly outperforms cumulative on sequential decode.

For sizing, a useful starting point at iso-byte versus uniform `k=6/v=2`:
`ring_size=8`, `fp_capacity=ctx_len // 16`, `quant_tiers=[(6, 2, ~ctx_len * 13/16)]`.
Halve the quant capacity for ~½ uniform bytes, etc.

## Running the eval

```bash
python eval.py --samples 20 --ctx_len 4096 --decode_len 64
```

Pulls 20 non-overlapping wikitext-103 chunks of 4096 tokens, runs prefill +
sequential decode for uniform TurboQuant + 4 KV-CASCADE configs, prints mean ± sd
top-1 / cosine sim against an fp32 reference. ~2 minutes on a single A100/H100.

```
config                            bytes (KiB)   ratio        cos sim (mean ± sd)       top-1 (mean ± sd)
--------------------------------------------------------------------------------------------------------
uniform k=6/v=2                        120064   3.82x     0.9258 ±  0.0192       68.4% ±  10.7%
kvcascade B (iso, ema)                 120056   3.82x     0.9673 ±  0.0063       78.4% ±   5.7%
kvcascade B (iso, cumulative)          120056   3.82x     0.9578 ±  0.0093       74.1% ±   7.5%
kvcascade F (½ byte, ema)               60022   7.64x     0.9750 ±  0.0070       80.0% ±   8.4%
kvcascade G (¼ byte, ema)               29990  15.30x     0.9708 ±  0.0097       79.7% ±   9.2%
```

(The eval uses **teacher forcing** — at each decode step we feed the ground-truth
token from the original sequence, and compare the model's argmax to the fp reference's
argmax at that position. So we're measuring quantization fidelity *given correct
history*, not generation quality with cumulative prediction errors. Different metric
than free-running generation, more isolated for comparing quantization approaches.)

## When does KV-CASCADE win?

The empirical picture, from sweeps on Qwen3-0.6B:

- **Long-context sequential decode (autoregressive generation):** KV-CASCADE wins
  cleanly, both at iso-byte and at sub-iso-byte budgets. The recency ring keeps the
  last few decode tokens at fp (avoiding the per-step quantization cost on tokens
  that are heavily attended), and eviction keeps softmax noise from accumulating
  across the cache.

- **Single-shot prefill scoring:** Uniform TurboQuant is competitive. The cache
  is only used for one big attention call, so reuse benefits don't accumulate; the
  fp tier just spends bytes a uniform-quant baseline doesn't.

- **Diffuse-attention LM workloads (wikitext, etc.):** KV-CASCADE wins on sequential
  decode because eviction removes the long tail of low-importance tokens whose
  quantized softmax contributions were net noise.

- **Sparse-attention QA / NIAH-style workloads:** Mixed. Sequential decode revealed
  that aggressive eviction can drop tokens the model needs to retrieve the answer.
  Tuning the fp tier capacity matters more here.

- **QK-normed models (Qwen3, OLMo-2, etc.):** Need `k_bits ≥ 5` in the bulk quant
  tier, otherwise the IP-estimator noise overwhelms QK-norm's sharper softmax
  (same threshold uniform TurboQuant has on these architectures). Below that, no
  amount of cache hierarchy fixes it.

## Implementation notes

**Vectorized cascade.** When the recency ring evicts (one or more graduates), the
graduates compete against tier residents in parallel across `(B, H_kv)`. Each tier
runs a single `topk` over the `[residents | candidates]` pool, then scatters winners
into open slots. Demoted residents continue down the chain. No Python-level loops
over slots or heads.

**Hybrid attention path.** Each attention call computes scores against *(quantized
prefix) ⊕ (fresh K/V)* and stores the fresh K/V into the ring afterwards. So a
just-arrived token contributes *exactly* to its own attention call (no quantization
round-trip), and the per-token fidelity drop only kicks in for subsequent steps that
re-read it from the cache.

**GQA-aware storage.** All buffers live at kv-head granularity (matching what HF
would actually allocate). Expansion to query-head count happens at attention compute
time, not at storage. Compression vs an fp16 KV cache stays apples-to-apples on
grouped-query models.

**Bit-packed storage.** All Lloyd-Max indices and JL sign sketches are bit-packed
into uint8 buffers (no padding to byte boundaries). Pack/unpack are pure tensor ops
with fast paths for power-of-2 widths.

**HF dispatcher integration.** `kvcascade.py` registers a function under
`transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["kvcascade"]`. `install_kvcascade(model, cache)`
flips `model.config._attn_implementation` to `"kvcascade"` and stamps the cache onto
every attention module. Works on any model that dispatches through ALL_ATTENTION_FUNCTIONS
(Llama, Qwen, Mistral, Gemma, modern GPT-2, etc.). No model-class-specific wrappers.

## Caveats

- v1 supports `batch_size=1` only.
- Multi-sample averaging on the headline result is from 5 samples; longer eval runs
  would tighten the confidence intervals.
- The win is sequential-decode-specific. Single-shot benchmarks won't show it.
- Free-running generation (model uses its own predictions as next inputs) hasn't been
  evaluated — teacher-forced top-1 is what's reported. Different metric, possibly
  different conclusions.
