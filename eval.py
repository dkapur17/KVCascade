"""Robust sequential-decode evaluation of TieredKVCache vs uniform TurboQuantKVCache
on wikitext-103, against an fp reference. CUDA-targeted, bf16 throughout.

Methodology:
  1. Pull N non-overlapping chunks of `--ctx_len` tokens from wikitext-103-test.
  2. For each chunk:
     a. Compute fp reference logits via single forward.
     b. For each cache config: prefill the first (ctx_len - decode_len) tokens, then run
        `decode_len` sequential one-token forwards. Each step's output reflects cache state
        evolved by all prior steps' policy-specific updates — i.e., the cache is being
        used as a cache, not just one-shot scored.
     c. Compare each step's logits against fp reference at the corresponding position.
  3. Aggregate top-1 / cosine similarity across all samples. Report mean ± stdev.

Run:
    python scripts/eval_wikitext_seq.py --samples 20 --ctx_len 4096 --decode_len 64

Outputs the table at the end. Optionally writes per-sample raw CSV via --csv.
"""

import argparse
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

# Make src/ importable when run from project root.
sys.path.insert(0, "./src")

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiered_cache import TieredKVCache, install_tiered_attention
from turbo_attn import TurboQuantKVCache, install_turbo_attention, _force_set_attn_impl


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--samples", type=int, default=20,
                   help="number of wikitext chunks to evaluate")
    p.add_argument("--ctx_len", type=int, default=4096,
                   help="total prompt length per chunk (prefill + decode)")
    p.add_argument("--decode_len", type=int, default=64,
                   help="number of sequential decode steps per chunk")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--csv", default=None,
                   help="optional CSV path for per-sample raw results")
    return p.parse_args()


# ----------------------------------------------------------------------------- data

def get_samples(tokenizer, n_samples: int, ctx_len: int) -> list[torch.Tensor]:
    """Pull n_samples non-overlapping ctx_len chunks from wikitext-103 test."""
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test", streaming=True)
    chunks, total_chars = [], 0
    target_chars = n_samples * ctx_len * 6  # generous overestimate (~6 chars/token)
    for item in ds:
        if item["text"].strip():
            chunks.append(item["text"])
            total_chars += len(item["text"])
            if total_chars >= target_chars:
                break
    big_text = "".join(chunks)
    all_ids = tokenizer(big_text, return_tensors="pt").input_ids[0]
    samples = []
    for i in range(n_samples):
        start = i * ctx_len
        end = start + ctx_len
        if end > len(all_ids):
            print(f"warning: only {i} samples available "
                  f"(needed {n_samples}; tokenized text only {len(all_ids)} tokens)",
                  flush=True)
            break
        samples.append(all_ids[start:end].unsqueeze(0))  # [1, ctx_len]
    return samples


# ---------------------------------------------------------------- sequential decode

def sequential_decode(model, ids: torch.Tensor, T_pre: int, T_dec: int,
                      device: torch.device) -> torch.Tensor:
    """Prefill then run T_dec sequential 1-token forwards.
    Returns concatenated decode logits [1, T_dec, V]."""
    with torch.no_grad():
        # Prefill — fills the cache.
        _ = model(input_ids=ids[:, :T_pre], use_cache=False)
        # Sequential decode — each call uses the cache state evolved by prior calls.
        out_list = []
        for k in range(T_dec):
            input_id = ids[:, T_pre + k : T_pre + k + 1]
            pos_id = torch.tensor([[T_pre + k]], device=device, dtype=torch.long)
            out = model(input_ids=input_id, position_ids=pos_id, use_cache=False)
            out_list.append(out.logits)  # [1, 1, V]
    return torch.cat(out_list, dim=1)  # [1, T_dec, V]


# ------------------------------------------------------------------------- configs

def make_uniform(n_l, n_q, n_kv, D, ctx_len, k_bits, v_bits, seed, device, dtype):
    return TurboQuantKVCache(
        num_layers=n_l, batch_size=1, num_heads=n_q, num_kv_heads=n_kv,
        head_dim=D, max_seq_len=ctx_len, k_bits=k_bits, v_bits=v_bits, m=D,
        seed=seed, device=device, dtype=dtype,
    )


def make_tiered(n_l, n_q, n_kv, D, R, fp_cap, qt, policy, seed, device, dtype):
    return TieredKVCache(
        num_layers=n_l, batch_size=1, num_heads=n_q, num_kv_heads=n_kv,
        head_dim=D, ring_size=R, fp_capacity=fp_cap, quant_tiers=qt,
        score_policy=policy, m=D, seed=seed, device=device, dtype=dtype,
    )


def build_configs(n_l, n_q, n_kv, D, ctx_len, seed, device, dtype):
    """Return list of (label, cache_factory, install_fn).

    Tiered configs:
      - "B" lands at iso-byte (matches uniform's footprint).
      - "F" lands at ~½ uniform bytes.
      - "G" lands at ~¼ uniform bytes.

    fp_capacity scales with ctx_len (fp_cap ≈ ctx_len/16 for B, /32 for F, /64 for G)
    so the design's "fraction of context held at fp" stays constant — otherwise iso-byte
    at large ctx_len has fp_cap negligibly small and the design degenerates to "uniform
    with a small ring." Each tiered config keeps R=8 fp ring + that fp tier + a
    `k=6/v=2` quant tier sized to fit the byte target. Eviction past tier 2.
    """
    fp_bytes_scalar = torch.empty((), dtype=dtype).element_size()
    fp_slot_bytes = 2 * D * fp_bytes_scalar              # K+V per slot in fp ring/tier
    turbo_slot_bytes = (3 * fp_bytes_scalar              # k_norm, k_resnorm, v_norm
                        + (D * 5 + 7) // 8                 # k_idx_packed for k=6
                        + (D * 2 + 7) // 8                 # v_idx_packed for v=2
                        + (D + 7) // 8)                    # k_ressign_packed (1 bit per m=D)

    target_bytes_per_lh = ctx_len * turbo_slot_bytes     # uniform's footprint
    print(f"target B/(L,kv) = {target_bytes_per_lh}, fp slot = {fp_slot_bytes}, "
          f"turbo slot = {turbo_slot_bytes}", flush=True)

    def tiered_for(R, fp_cap, target_bytes, policy):
        budget = target_bytes - R * fp_slot_bytes - fp_cap * fp_slot_bytes
        qt_cap = max(0, budget // turbo_slot_bytes)
        return (R, fp_cap, [(6, 2, int(qt_cap))], policy)

    fp_cap_B = max(8, ctx_len // 16)
    fp_cap_F = max(4, ctx_len // 32)
    fp_cap_G = max(2, ctx_len // 64)

    B_ema = tiered_for(8, fp_cap_B, target_bytes_per_lh,       "ema")
    B_cum = tiered_for(8, fp_cap_B, target_bytes_per_lh,       "cumulative")
    F_ema = tiered_for(8, fp_cap_F, target_bytes_per_lh // 2, "ema")
    G_ema = tiered_for(8, fp_cap_G, target_bytes_per_lh // 4, "ema")
    print(f"  B: R=8 fp={fp_cap_B} qt={B_ema[2][0][2]}", flush=True)
    print(f"  F: R=8 fp={fp_cap_F} qt={F_ema[2][0][2]}", flush=True)
    print(f"  G: R=8 fp={fp_cap_G} qt={G_ema[2][0][2]}", flush=True)

    def factory(make_fn, *args):
        return lambda: make_fn(*args)

    cfgs = [
        ("uniform k=6/v=2",
         factory(make_uniform, n_l, n_q, n_kv, D, ctx_len, 6, 2, seed, device, dtype),
         install_turbo_attention),
        ("tiered B (iso, ema)",
         factory(make_tiered, n_l, n_q, n_kv, D, *B_ema, seed, device, dtype),
         install_tiered_attention),
        ("tiered B (iso, cumulative)",
         factory(make_tiered, n_l, n_q, n_kv, D, *B_cum, seed, device, dtype),
         install_tiered_attention),
        ("tiered F (½ byte, ema)",
         factory(make_tiered, n_l, n_q, n_kv, D, *F_ema, seed, device, dtype),
         install_tiered_attention),
        ("tiered G (¼ byte, ema)",
         factory(make_tiered, n_l, n_q, n_kv, D, *G_ema, seed, device, dtype),
         install_tiered_attention),
    ]
    return cfgs


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]

    print(f"model={args.model}  device={device}  dtype={dtype}", flush=True)
    print(f"samples={args.samples}  ctx_len={args.ctx_len}  decode_len={args.decode_len}\n",
          flush=True)
    assert args.decode_len < args.ctx_len, "decode_len must be < ctx_len"

    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading wikitext-103 samples...", flush=True)
    samples = get_samples(tok, args.samples, args.ctx_len)
    print(f"got {len(samples)} samples\n", flush=True)
    if not samples:
        print("no samples, aborting"); return

    def fresh_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="eager",
        ).to(device).eval()

    # Probe model dims.
    probe = fresh_model()
    cfg = probe.config
    n_l = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    D = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    fp16_baseline_bytes = 2 * n_l * n_kv * args.ctx_len * D * 2
    print(f"L={n_l}  q={n_q}  kv={n_kv}  D={D}  fp16 baseline = {fp16_baseline_bytes/1024:.1f} KiB\n",
          flush=True)
    del probe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    configs = build_configs(n_l, n_q, n_kv, D, args.ctx_len, args.seed, device, dtype)
    T_pre = args.ctx_len - args.decode_len

    # One-time setup: a single shared model plus one cache per config. Each sample swaps
    # the model's attention dispatcher between "eager" (for the fp reference) and the
    # config's impl. install_*_attention is idempotent — it just toggles the dispatcher
    # string and attaches the cache.
    def set_attn_impl(model: torch.nn.Module, value: str) -> None:
        _force_set_attn_impl(model.config, value)
        for sub in model.modules():
            if hasattr(sub, "config") and sub is not model:
                _force_set_attn_impl(sub.config, value)

    print("building model and caches (one-time setup)...", flush=True)
    setup_t0 = time.time()
    model = fresh_model()
    cache_specs = [(label, cache_factory(), install_fn)
                   for label, cache_factory, install_fn in configs]
    print(f"setup done in {time.time() - setup_t0:.1f}s\n", flush=True)

    # Per-config aggregates.
    results = {label: {"cos": [], "top1": [], "bytes": None}
               for label, _, _ in configs}

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["sample_idx", "config", "bytes", "cos_sim", "top1"])

    # Pre-compute fp reference logits for every sample (single eager pass per sample).
    # Stored on CPU to keep GPU memory free for the cache configs.
    print("computing fp reference logits for all samples...", flush=True)
    ref_t0 = time.time()
    set_attn_impl(model, "eager")
    ref_decs: list[torch.Tensor] = []
    with torch.no_grad():
        for ids in samples:
            ids_d = ids.to(device)
            ref = model(input_ids=ids_d, use_cache=False).logits
            ref_decs.append(ref[:, T_pre : T_pre + args.decode_len, :].float().cpu())
            del ref
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"fp reference done in {time.time() - ref_t0:.1f}s "
          f"({len(ref_decs)} samples)\n", flush=True)

    # For each config: install once, then iterate samples (cache.reset() between).
    overall_t0 = time.time()
    for label, cache, install_fn in cache_specs:
        print(f"=== config: {label}  "
              f"(T_pre={T_pre}, T_dec={args.decode_len}) ===", flush=True)
        install_fn(model, cache)
        bytes_total = cache.bytes_total()
        ratio = fp16_baseline_bytes / bytes_total
        results[label]["bytes"] = bytes_total

        for sample_idx, ids in enumerate(samples):
            ids_d = ids.to(device)
            ref_dec = ref_decs[sample_idx].to(device)
            cache.reset()
            t_start = time.time()
            logits = sequential_decode(model, ids_d, T_pre, args.decode_len, device).float()
            cos = F.cosine_similarity(logits, ref_dec, dim=-1).mean().item()
            top1 = (logits.argmax(-1) == ref_dec.argmax(-1)).float().mean().item()

            results[label]["cos"].append(cos)
            results[label]["top1"].append(top1)

            print(f"  sample {sample_idx + 1:>3}/{len(samples)}  "
                  f"bytes={bytes_total/1024:>6.0f}KiB  ratio={ratio:>5.2f}x  "
                  f"cos={cos:.4f}  top1={top1*100:>5.1f}%  "
                  f"({time.time() - t_start:.1f}s)", flush=True)

            if csv_writer is not None:
                csv_writer.writerow([sample_idx, label, bytes_total, cos, top1])
                csv_file.flush()

            del logits, ref_dec
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(flush=True)

    if csv_file is not None:
        csv_file.close()

    # ----------- summary -----------
    print(f"\n========== SUMMARY  (total {time.time() - overall_t0:.0f}s) ==========")
    print(f"samples={len(samples)}  ctx_len={args.ctx_len}  decode_len={args.decode_len}\n")
    header = (f"{'config':<32} {'bytes (KiB)':>12} {'ratio':>7} "
              f"{'cos sim (mean ± sd)':>26}  {'top-1 (mean ± sd)':>22}")
    print(header)
    print("-" * len(header))
    for label, _, _ in configs:
        cos_list = results[label]["cos"]
        top1_list = results[label]["top1"]
        bytes_total = results[label]["bytes"]
        cos_mean = statistics.mean(cos_list)
        cos_std = statistics.stdev(cos_list) if len(cos_list) > 1 else 0.0
        top1_mean = statistics.mean(top1_list)
        top1_std = statistics.stdev(top1_list) if len(top1_list) > 1 else 0.0
        ratio = fp16_baseline_bytes / bytes_total
        print(f"{label:<32} {bytes_total/1024:>12.0f} {ratio:>6.2f}x  "
              f"{cos_mean:>9.4f} ± {cos_std:>7.4f}  "
              f"{top1_mean*100:>9.1f}% ± {top1_std*100:>5.1f}%")


if __name__ == "__main__":
    main()
