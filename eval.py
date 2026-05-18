"""KVCascade comprehensive evaluation on wikitext-103.

Runs three experiments and an attention-pattern analysis, comparing KVCascade
against uniform TurboQuant, H2O, and full-fp baselines on a configurable model:

  Exp 1: Compression sweep — find the byte budget where KVCascade matches uniform's
         quality (top-1 within 1 pp). Sweep at 1×, ½×, ¼×, … of uniform's bytes.
  Exp 2: Iso-byte head-to-head — at uniform's bytes, compare full-fp / uniform /
         H2O / KVCascade.
  Exp 3: Split sweep — within KVCascade at fixed budget, sweep fp_capacity (and
         derive quant_capacity from budget) to find the optimal split.

Outputs a markdown report (`report.md`) with tables + embedded figures and
saves per-figure PNGs to `<out>/figures/`. Raw per-sample results are written
to `raw.json` for later reanalysis.

Usage:
    python eval.py --model Qwen/Qwen3-0.6B --samples 20
    python eval.py --model meta-llama/Llama-3.2-1B --samples 50 --out outputs/llama
    python eval.py --skip-viz --skip-exp1                   # only Exp 2 + Exp 3
"""

import argparse
import gc
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "src"))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from kvcascade import KVCascadeCache, install_kvcascade, make_h2o_cache
from turbo_attn import TurboQuantKVCache, install_turbo_attention, _force_set_attn_impl


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="HF model name (default: Qwen/Qwen3-0.6B)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--ctx-len", "--ctx_len", type=int, default=4096, dest="ctx_len")
    p.add_argument("--decode-len", "--decode_len", type=int, default=64, dest="decode_len")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Output directory (default: outputs/<model_slug>_<timestamp>/)")
    p.add_argument("--viz-ctx-len", type=int, default=1024,
                   help="Context length for the attention-pattern viz pass (default: 1024). "
                        "Smaller than --ctx-len to keep [L, H, T, T] under VRAM.")
    p.add_argument("--k-bits", type=int, default=6)
    p.add_argument("--v-bits", type=int, default=2)
    p.add_argument("--ring-size", type=int, default=8,
                   help="KVCascade recency ring size (default: 8)")
    p.add_argument("--quant-mode", default="prod", choices=("prod", "mse"),
                   help="K quantization variant for uniform + KVCascade caches: "
                        "'prod' (TurboQuant Prod = MSE coarse + 1-bit JL sketch, default) "
                        "or 'mse' (PolarQuant only, no JL). MSE saves a fp byte per token "
                        "and typically lowers attention reconstruction error on real K, at "
                        "the cost of losing the unbiased-IP guarantee. Default 'prod' "
                        "matches all pre-2026-05 eval numbers in the README.")
    p.add_argument("--exp1-ratios", default="1.0,0.5,0.25,0.125,0.0625",
                   help="Comma-separated byte ratios (× uniform iso-byte) for Exp 1")
    p.add_argument("--exp3-fp-caps", default="0,32,64,128,256,512,1024",
                   help="Comma-separated fp_capacity values for Exp 3 (qt derived from budget)")
    p.add_argument("--threshold-pp", type=float, default=1.0,
                   help="Top-1 percentage-point threshold for the Exp 1 'matches uniform' headline (default: 1.0)")
    p.add_argument("--skip-viz",  action="store_true")
    p.add_argument("--skip-exp1", action="store_true")
    p.add_argument("--skip-exp2", action="store_true")
    # exp3 defaults to skipped; pass --no-skip-exp3 to bring it back.
    p.add_argument("--skip-exp3", default=True, action=argparse.BooleanOptionalAction)
    return p.parse_args()


# ============================================================================
# Setup
# ============================================================================

def setup_output_dir(args) -> Path:
    if args.out:
        out = Path(args.out)
    else:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.model).strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs") / f"{slug}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    return out


def setup_model(args):
    dtype = DTYPE_MAP[args.dtype]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"loading {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    return model, tok, device, dtype


def get_dims(model) -> dict:
    cfg = model.config
    n_q = cfg.num_attention_heads
    return {
        "n_layers": cfg.num_hidden_layers,
        "n_heads": n_q,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", n_q),
        "head_dim": getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q),
    }


def get_samples(tok, n_samples: int, ctx_len: int):
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    chunks, total_chars = [], 0
    target_chars = n_samples * ctx_len * 6
    for item in ds:
        if item["text"].strip():
            chunks.append(item["text"])
            total_chars += len(item["text"])
            if total_chars >= target_chars:
                break
    big_text = "".join(chunks)
    # Bump model_max_length so the corpus tokenization doesn't print a length warning;
    # we chunk to ctx_len below before any model forward.
    saved_max_len = tok.model_max_length
    tok.model_max_length = int(1e12)
    all_ids = tok(big_text, return_tensors="pt").input_ids[0]
    tok.model_max_length = saved_max_len
    out = []
    for i in range(n_samples):
        s, e = i * ctx_len, (i + 1) * ctx_len
        if e > len(all_ids):
            print(f"  warning: only {i}/{n_samples} samples fit in tokenized text", flush=True)
            break
        out.append(all_ids[s:e].unsqueeze(0))
    return out


def set_attn_impl(model, value: str) -> None:
    _force_set_attn_impl(model.config, value)
    for sub in model.modules():
        if hasattr(sub, "config") and sub is not model:
            _force_set_attn_impl(sub.config, value)


def compute_fp_reference(model, samples, T_pre: int, T_dec: int, device):
    set_attn_impl(model, "eager")
    refs = []
    with torch.no_grad():
        for ids in samples:
            ids_d = ids.to(device)
            logits = model(input_ids=ids_d, use_cache=False).logits
            refs.append(logits[:, T_pre:T_pre + T_dec, :].float().cpu())
            del logits
    torch.cuda.empty_cache()
    return refs


# ============================================================================
# Byte accounting
# ============================================================================

def fp_slot_bytes(dims: dict, dtype) -> int:
    return 2 * dims["head_dim"] * torch.empty((), dtype=dtype).element_size()


def turbo_slot_bytes(dims: dict, dtype, k_bits: int, v_bits: int,
                     m: int = None, quant_mode: str = "prod") -> int:
    """Per-slot bytes for a TurboQuant K + PolarQuant V quantized cache slot.

    Prod: 3 fp (k_norm, k_resnorm, v_norm) + (k_bits-1) coarse + 1-bit JL sketch + v_idx
    MSE : 2 fp (k_norm, v_norm) + k_bits coarse + 0 JL + v_idx
    """
    if m is None:
        m = dims["head_dim"]
    fp = torch.empty((), dtype=dtype).element_size()
    D = dims["head_dim"]
    if quant_mode == "prod":
        return (3 * fp                            # k_norm, k_resnorm, v_norm
                + (D * (k_bits - 1) + 7) // 8     # k_idx_packed
                + (D * v_bits + 7) // 8           # v_idx_packed
                + (m + 7) // 8)                   # k_ressign_packed
    else:  # mse
        return (2 * fp                            # k_norm, v_norm
                + (D * k_bits + 7) // 8           # k_idx_packed (full bit budget)
                + (D * v_bits + 7) // 8)          # v_idx_packed


def kvc_qt_cap_at_budget(target_bytes_per_lh: int, ring_size: int, fp_capacity: int,
                          dims: dict, dtype, k_bits: int, v_bits: int,
                          quant_mode: str = "prod") -> int:
    fp_b = (ring_size + fp_capacity) * fp_slot_bytes(dims, dtype)
    rem = target_bytes_per_lh - fp_b
    return max(0, rem // turbo_slot_bytes(dims, dtype, k_bits, v_bits, quant_mode=quant_mode))


def fp16_baseline_bytes(args, dims: dict, dtype) -> int:
    return dims["n_layers"] * dims["n_kv_heads"] * args.ctx_len * fp_slot_bytes(dims, dtype)


# ============================================================================
# Cache constructors
# ============================================================================

def make_uniform(args, dims: dict, dtype, device):
    return TurboQuantKVCache(
        num_layers=dims["n_layers"], batch_size=1,
        num_heads=dims["n_heads"], num_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"], max_seq_len=args.ctx_len,
        k_bits=args.k_bits, v_bits=args.v_bits, m=dims["head_dim"],
        quant_mode=getattr(args, "quant_mode", "prod"),
        seed=args.seed, device=device, dtype=dtype,
    )


def make_h2o(args, dims: dict, dtype, device, cache_size: int, recency_window: int = 0):
    return make_h2o_cache(
        num_layers=dims["n_layers"], batch_size=1,
        num_heads=dims["n_heads"], num_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        cache_size=cache_size, recency_window=recency_window,
        score_policy="cumulative",
        seed=args.seed, device=device, dtype=dtype,
    )


def make_kvc(args, dims: dict, dtype, device, ring_size: int, fp_capacity: int, quant_capacity: int):
    quant_tiers = [(args.k_bits, args.v_bits, quant_capacity)] if quant_capacity > 0 else []
    return KVCascadeCache(
        num_layers=dims["n_layers"], batch_size=1,
        num_heads=dims["n_heads"], num_kv_heads=dims["n_kv_heads"],
        head_dim=dims["head_dim"],
        ring_size=ring_size, fp_capacity=fp_capacity,
        quant_tiers=quant_tiers,
        m=dims["head_dim"], score_policy="ema",
        quant_mode=getattr(args, "quant_mode", "prod"),
        seed=args.seed, device=device, dtype=dtype,
    )


# ============================================================================
# Sequential decode + per-config eval
# ============================================================================

def sequential_decode(model, ids, T_pre: int, T_dec: int, device):
    """Returns (logits[B, T_dec, V], prefill_seconds, decode_seconds).
    Both timings are CUDA-synchronized so they reflect actual GPU work."""
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids[:, :T_pre], use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pre = time.perf_counter() - t0

        out_list = []
        t1 = time.perf_counter()
        for k in range(T_dec):
            iid = ids[:, T_pre + k:T_pre + k + 1]
            pid = torch.tensor([[T_pre + k]], device=device, dtype=torch.long)
            o = model(input_ids=iid, position_ids=pid, use_cache=False)
            out_list.append(o.logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dec = time.perf_counter() - t1
    return torch.cat(out_list, dim=1), t_pre, t_dec


def evaluate_config(model, cache, install_fn, samples, ref_decs,
                    T_pre: int, T_dec: int, device, label: str = "") -> dict:
    install_fn(model, cache)
    bytes_total = cache.bytes_total()
    cos_list, top1_list = [], []
    pre_times, dec_times = [], []
    t0 = time.time()
    for ids, ref in zip(samples, ref_decs):
        ids_d = ids.to(device)
        ref_d = ref.to(device)
        cache.reset()
        logits, t_pre, t_dec = sequential_decode(model, ids_d, T_pre, T_dec, device)
        logits = logits.float()
        pre_times.append(t_pre)
        dec_times.append(t_dec)
        cos_list.append(F.cosine_similarity(logits, ref_d, dim=-1).mean().item())
        top1_list.append((logits.argmax(-1) == ref_d.argmax(-1)).float().mean().item())
        del logits, ref_d
        torch.cuda.empty_cache()
    elapsed = time.time() - t0
    n = len(samples)
    total_pre = sum(pre_times)
    total_dec = sum(dec_times)
    return {
        "label": label,
        "bytes": int(bytes_total),
        "top1_mean": statistics.mean(top1_list),
        "top1_std":  statistics.stdev(top1_list) if len(top1_list) > 1 else 0.0,
        "cos_mean":  statistics.mean(cos_list),
        "cos_std":   statistics.stdev(cos_list) if len(cos_list) > 1 else 0.0,
        "top1": top1_list, "cos": cos_list, "elapsed_s": elapsed,
        "prefill_s_per_sample": total_pre / n if n else 0.0,
        "decode_s_per_sample":  total_dec / n if n else 0.0,
        "prefill_tok_per_s": (n * T_pre) / total_pre if total_pre > 0 else float("nan"),
        "decode_tok_per_s":  (n * T_dec) / total_dec if total_dec > 0 else float("nan"),
    }


def fmt_result(r: dict) -> str:
    dec_tps = r.get("decode_tok_per_s", float("nan"))
    pre_tps = r.get("prefill_tok_per_s", float("nan"))
    return (f"top1 = {r['top1_mean']*100:5.1f}% ± {r['top1_std']*100:4.1f}%  "
            f"cos = {r['cos_mean']:.4f} ± {r['cos_std']:.4f}  "
            f"bytes = {r['bytes']/1024:>8.0f} KiB  "
            f"prefill = {pre_tps:>7.1f} tok/s  decode = {dec_tps:>5.1f} tok/s  "
            f"({r['elapsed_s']:.0f}s)")


# ============================================================================
# Attention pattern visualization
# ============================================================================

def viz_attention_patterns(model, samples, dims: dict, viz_ctx_len: int,
                            out_dir: Path, device) -> dict:
    n_layers, n_heads = dims["n_layers"], dims["n_heads"]
    # Clamp to the actual sample length — when ctx_len is smaller than the requested
    # viz_ctx_len, we just use whatever's available.
    viz_ctx_len = min(viz_ctx_len, samples[0].shape[1])
    ids_viz = samples[0][:, :viz_ctx_len].to(device)
    print(f"viz forward on first {viz_ctx_len} tokens of sample 0", flush=True)

    entropies_cpu = torch.zeros(n_layers, n_heads)
    received_cpu  = torch.zeros(n_layers, n_heads, viz_ctx_len)

    def _stats_attn(module, query, key, value, attention_mask=None,
                     scaling=None, dropout=0.0, **kwargs):
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])
        n_rep = query.shape[1] // key.shape[1]
        k_e = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
        v_e = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
        s = torch.matmul(query, k_e.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            s = s + attention_mask[..., :s.shape[-1]]
        a = torch.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(a, v_e).transpose(1, 2).contiguous()
        li = module.layer_idx
        a32 = a.float()
        a32_safe = a32.clamp_min(1e-12)
        entropies_cpu[li] = (-(a32_safe * a32_safe.log()).sum(dim=-1)).mean(dim=(0, 2)).cpu()
        received_cpu[li]  = a32.sum(dim=2).squeeze(0).cpu()
        return out, None

    ALL_ATTENTION_FUNCTIONS["__kvc_viz_stats__"] = _stats_attn
    set_attn_impl(model, "__kvc_viz_stats__")
    with torch.no_grad():
        model(input_ids=ids_viz, use_cache=False)
    set_attn_impl(model, "eager")
    torch.cuda.empty_cache()

    entropies = entropies_cpu.numpy()

    # Figure 1: per-layer attention received heatmaps for representative layers.
    select_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
    fig, axes = plt.subplots(len(select_layers), 1, figsize=(12, 1.8 * len(select_layers)),
                              squeeze=False)
    for ax, l in zip(axes[:, 0], select_layers):
        rec = received_cpu[l].numpy()
        BLOCK = max(1, rec.shape[1] // 256)
        if BLOCK > 1:
            T_k = rec.shape[1]
            rec = rec[:, : T_k - T_k % BLOCK].reshape(rec.shape[0], -1, BLOCK).mean(-1)
        ax.imshow(np.log1p(rec), aspect="auto", cmap="viridis")
        ax.set_title(f"layer {l}: log(1 + attention received) per (head, token)")
        ax.set_xlabel("token position (block-meaned)")
        ax.set_ylabel("head")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "attn_received.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Figure 2: entropy histogram.
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(entropies.flatten(), bins=40, edgecolor="k")
    mean_e, med_e = float(entropies.mean()), float(np.median(entropies))
    ax.axvline(mean_e, color="red",    ls="--", label=f"mean = {mean_e:.2f}")
    ax.axvline(med_e,  color="orange", ls="--", label=f"median = {med_e:.2f}")
    ax.set_xlabel("attention entropy (nats; higher = more diffuse)")
    ax.set_ylabel("count of (layer, head) pairs")
    ax.set_title("Per-head attention entropy distribution")
    ax.legend()
    plt.savefig(out_dir / "figures" / "entropy_histogram.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Detail pass: capture full attn for the peakiest and most diffuse layers.
    flat = entropies.flatten()
    peaky_l, peaky_h     = divmod(int(flat.argmin()), n_heads)
    diffuse_l, diffuse_h = divmod(int(flat.argmax()), n_heads)
    target_layers = {peaky_l, diffuse_l}
    detail_attn: dict[int, torch.Tensor] = {}

    def _detail_attn(module, query, key, value, attention_mask=None,
                      scaling=None, dropout=0.0, **kwargs):
        if scaling is None:
            scaling = 1.0 / math.sqrt(query.shape[-1])
        n_rep = query.shape[1] // key.shape[1]
        k_e = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key
        v_e = value.repeat_interleave(n_rep, dim=1) if n_rep > 1 else value
        s = torch.matmul(query, k_e.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            s = s + attention_mask[..., :s.shape[-1]]
        a = torch.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(a, v_e).transpose(1, 2).contiguous()
        if module.layer_idx in target_layers:
            detail_attn[module.layer_idx] = a.squeeze(0).float().cpu()
        return out, None

    ALL_ATTENTION_FUNCTIONS["__kvc_viz_detail__"] = _detail_attn
    set_attn_impl(model, "__kvc_viz_detail__")
    with torch.no_grad():
        model(input_ids=ids_viz, use_cache=False)
    set_attn_impl(model, "eager")
    torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (l, h, lbl) in zip(axes,
            [(peaky_l, peaky_h, "peakiest"), (diffuse_l, diffuse_h, "most diffuse")]):
        a = detail_attn[l][h].numpy()
        BLOCK = max(1, a.shape[0] // 256)
        if BLOCK > 1:
            T = a.shape[0] // BLOCK * BLOCK
            a = a[:T, :T].reshape(T // BLOCK, BLOCK, T // BLOCK, BLOCK).mean(axis=(1, 3))
        ax.imshow(np.log1p(a * 1000), aspect="auto", cmap="viridis")
        ax.set_title(f"{lbl}: layer {l}, head {h} (entropy={entropies[l, h]:.2f})")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "peaky_diffuse_detail.png", dpi=120, bbox_inches="tight")
    plt.close()

    del detail_attn
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "entropy_mean":   mean_e,
        "entropy_median": med_e,
        "entropy_min":    float(entropies.min()),
        "entropy_max":    float(entropies.max()),
        "uniform_max":    float(math.log(viz_ctx_len)),
        "peaky":   [int(peaky_l),   int(peaky_h)],
        "diffuse": [int(diffuse_l), int(diffuse_h)],
        "viz_ctx_len": viz_ctx_len,
    }


# ============================================================================
# Experiments
# ============================================================================

def exp1_compression_sweep(model, samples, ref_decs, args, dims, dtype, device,
                            T_pre, out_dir) -> dict:
    print("\n" + "=" * 70)
    print("Experiment 1: compression sweep")
    print("=" * 70)

    UB = args.ctx_len * turbo_slot_bytes(dims, dtype, args.k_bits, args.v_bits, quant_mode=args.quant_mode)
    ratios = [float(r) for r in args.exp1_ratios.split(",") if r.strip()]

    print("running uniform k=6/v=2...", flush=True)
    res_uni = evaluate_config(model, make_uniform(args, dims, dtype, device),
                              install_turbo_attention, samples, ref_decs,
                              T_pre, args.decode_len, device, "uniform")
    print(f"  {fmt_result(res_uni)}", flush=True)

    rows = []
    for ratio in ratios:
        target = int(UB * ratio)
        # Scale fp_capacity proportionally to budget so the design's "fraction at fp"
        # stays roughly constant across ratios.
        fp_cap = max(8, int(round(args.ctx_len * ratio / 16)))
        qt_cap = int(kvc_qt_cap_at_budget(target, args.ring_size, fp_cap, dims, dtype,
                                           args.k_bits, args.v_bits, quant_mode=args.quant_mode))
        label = f"KVC @ {ratio:>6.4f}x  (ring={args.ring_size}, fp={fp_cap}, qt={qt_cap})"
        print(f"running {label}", flush=True)
        cache = make_kvc(args, dims, dtype, device, args.ring_size, fp_cap, qt_cap)
        res = evaluate_config(model, cache, install_kvcascade, samples, ref_decs,
                              T_pre, args.decode_len, device, label)
        rows.append({"ratio": ratio, "fp_cap": fp_cap, "qt_cap": qt_cap, **res})
        print(f"  {fmt_result(res)}", flush=True)

    # Headline: smallest ratio that matches uniform within --threshold-pp.
    threshold = res_uni["top1_mean"] - args.threshold_pp / 100.0
    passing = [r for r in rows if r["top1_mean"] >= threshold]
    headline = None
    if passing:
        best = min(r["ratio"] for r in passing)
        headline = (f"matches uniform within {args.threshold_pp:.1f} pp at "
                    f"{best:.4f}× bytes (= {1/best:.1f}× compression vs uniform)")

    # Plot.
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    xs   = [r["ratio"] for r in rows]
    ys   = [r["top1_mean"] * 100 for r in rows]
    errs = [r["top1_std"]  * 100 for r in rows]
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, label="KVCascade")
    ax.axhline(res_uni["top1_mean"] * 100, color="red", ls="--",
               label=f"uniform @ 1x ({res_uni['top1_mean']*100:.1f}%)")
    ax.fill_between([min(xs), max(xs)],
                    (res_uni["top1_mean"] - args.threshold_pp / 100) * 100,
                    (res_uni["top1_mean"] + args.threshold_pp / 100) * 100,
                    color="red", alpha=0.1, label=f"±{args.threshold_pp:.0f} pp from uniform")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("byte budget (× uniform iso-byte)")
    ax.set_ylabel("top-1 (%)")
    ax.set_title(f"Exp 1: KVCascade compression curve ({args.model}, N={len(samples)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(out_dir / "figures" / "exp1_compression.png", dpi=120, bbox_inches="tight")
    plt.close()

    return {"uniform": res_uni, "rows": rows, "headline": headline,
            "uniform_bytes_per_lh": UB}


def exp2_iso_byte(model, samples, ref_decs, args, dims, dtype, device,
                   T_pre, out_dir, exp1_results=None) -> dict:
    print("\n" + "=" * 70)
    print("Experiment 2: iso-byte head-to-head")
    print("=" * 70)

    UB = args.ctx_len * turbo_slot_bytes(dims, dtype, args.k_bits, args.v_bits, quant_mode=args.quant_mode)
    fp16_total = fp16_baseline_bytes(args, dims, dtype)

    # Full-fp upper bound by construction.
    full_fp = {
        "label": "full-fp (ref)",
        "bytes": int(fp16_total),
        "top1_mean": 1.0, "top1_std": 0.0,
        "cos_mean":  1.0, "cos_std":  0.0,
        "top1": [1.0] * len(samples), "cos": [1.0] * len(samples),
        "elapsed_s": 0.0,
    }

    # Uniform: reuse Exp 1's if available; else run.
    if exp1_results is not None:
        res_uni = exp1_results["uniform"]
    else:
        print("running uniform k=6/v=2...", flush=True)
        res_uni = evaluate_config(model, make_uniform(args, dims, dtype, device),
                                  install_turbo_attention, samples, ref_decs,
                                  T_pre, args.decode_len, device, "uniform")
        print(f"  {fmt_result(res_uni)}", flush=True)

    # H2O at iso-byte (eviction only, no recency ring).
    h2o_cap = int(UB // fp_slot_bytes(dims, dtype))
    print(f"running H2O @ iso-byte (cache_size={h2o_cap}, ring=0)...", flush=True)
    res_h2o = evaluate_config(model, make_h2o(args, dims, dtype, device, h2o_cap),
                              install_kvcascade, samples, ref_decs,
                              T_pre, args.decode_len, device, "h2o")
    print(f"  {fmt_result(res_h2o)}", flush=True)

    # H2O + recency ring at iso-byte (ablation: adds the recency buffer to plain H2O,
    # isolating the ring's contribution before quantization is layered on top).
    print(f"running H2O+ring @ iso-byte (cache_size={h2o_cap}, ring={args.ring_size})...", flush=True)
    res_h2o_ring = evaluate_config(
        model,
        make_h2o(args, dims, dtype, device, h2o_cap, recency_window=args.ring_size),
        install_kvcascade, samples, ref_decs,
        T_pre, args.decode_len, device, "h2o+ring",
    )
    print(f"  {fmt_result(res_h2o_ring)}", flush=True)

    # KVCascade at iso-byte: reuse Exp 1's 1.0× row if present.
    kvc_iso = None
    if exp1_results is not None:
        for r in exp1_results["rows"]:
            if r["ratio"] == 1.0:
                kvc_iso = r
                break
    if kvc_iso is None:
        fp_cap = args.ctx_len // 16
        qt_cap = int(kvc_qt_cap_at_budget(UB, args.ring_size, fp_cap, dims, dtype,
                                           args.k_bits, args.v_bits, quant_mode=args.quant_mode))
        label = f"KVC @ 1x (ring={args.ring_size}, fp={fp_cap}, qt={qt_cap})"
        print(f"running {label}", flush=True)
        cache = make_kvc(args, dims, dtype, device, args.ring_size, fp_cap, qt_cap)
        kvc_iso = evaluate_config(model, cache, install_kvcascade, samples, ref_decs,
                                  T_pre, args.decode_len, device, label)
        kvc_iso = {"ratio": 1.0, "fp_cap": fp_cap, "qt_cap": qt_cap, **kvc_iso}
        print(f"  {fmt_result(kvc_iso)}", flush=True)

    rows = [
        ("full-fp (ref)", full_fp),
        ("uniform k=6/v=2", res_uni),
        ("H2O (ring=0)", res_h2o),
        (f"H2O (ring={args.ring_size})", res_h2o_ring),
        ("KVCascade (ring + fp + quant)", kvc_iso),
    ]

    # Bar plot (skip full-fp since it's by-construction 100%). Color order traces the
    # ablation: uniform (gray) → H2O (red, eviction-only) → H2O+ring (orange, +recency)
    # → KVCascade (blue, +quantization).
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    plot_rows = [(name, r) for name, r in rows if "ref" not in name]
    names = [n for n, _ in plot_rows]
    top1s = [r["top1_mean"] * 100 for _, r in plot_rows]
    errs  = [r["top1_std"]  * 100 for _, r in plot_rows]
    bars = ax.bar(names, top1s, yerr=errs, capsize=5,
                  color=["#888", "#c66", "#e9a23b", "#79b"])
    ax.axhline(100, color="green", ls=":", alpha=0.7, label="full-fp upper bound")
    ax.set_ylabel("top-1 (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Exp 2: iso-byte head-to-head ({args.model}, N={len(samples)})")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    for bar, val in zip(bars, top1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "exp2_isobyte.png", dpi=120, bbox_inches="tight")
    plt.close()

    return {"rows": rows, "uniform_bytes": int(UB), "fp16_baseline_bytes": int(fp16_total)}


def exp3_split_sweep(model, samples, ref_decs, args, dims, dtype, device,
                      T_pre, out_dir) -> dict:
    print("\n" + "=" * 70)
    print("Experiment 3: split sweep at fixed budget")
    print("=" * 70)

    UB = args.ctx_len * turbo_slot_bytes(dims, dtype, args.k_bits, args.v_bits, quant_mode=args.quant_mode)
    fp_caps = [int(c) for c in args.exp3_fp_caps.split(",") if c.strip()]

    rows = []
    for fp_cap in fp_caps:
        qt_cap = int(kvc_qt_cap_at_budget(UB, args.ring_size, fp_cap, dims, dtype,
                                           args.k_bits, args.v_bits, quant_mode=args.quant_mode))
        if qt_cap <= 0 and fp_cap > 0:
            print(f"skipping fp_cap={fp_cap}: budget exhausted (qt_cap=0)", flush=True)
            continue
        label = f"ring={args.ring_size}, fp={fp_cap:>4}, qt={qt_cap:>4}"
        print(f"running {label}", flush=True)
        cache = make_kvc(args, dims, dtype, device, args.ring_size, fp_cap, qt_cap)
        res = evaluate_config(model, cache, install_kvcascade, samples, ref_decs,
                              T_pre, args.decode_len, device, label)
        rows.append({"fp_cap": fp_cap, "qt_cap": qt_cap, **res})
        print(f"  {fmt_result(res)}", flush=True)

    # Plot.
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    xs   = [r["fp_cap"]      for r in rows]
    ys   = [r["top1_mean"] * 100 for r in rows]
    errs = [r["top1_std"]  * 100 for r in rows]
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, label="KVCascade")
    for r in rows:
        ax.annotate(f"qt={r['qt_cap']}", (r["fp_cap"], r["top1_mean"] * 100),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
    ax.set_xlabel("fp_capacity (heavy-hitter slots at fp)")
    ax.set_ylabel("top-1 (%)")
    ax.set_title(f"Exp 3: top-1 vs fp/quant split (uniform iso-byte, ring={args.ring_size})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(out_dir / "figures" / "exp3_split.png", dpi=120, bbox_inches="tight")
    plt.close()

    best = max(rows, key=lambda r: r["top1_mean"]) if rows else None
    return {"rows": rows, "best": best, "uniform_bytes_per_lh": int(UB)}


# ============================================================================
# Markdown report
# ============================================================================

def _trim(r: dict) -> dict:
    """Strip per-sample arrays from a result row before serializing to JSON."""
    return {k: v for k, v in r.items() if k not in ("top1", "cos")}


def _fmt_tps(v) -> str:
    if v is None:
        return "—"
    try:
        if math.isnan(float(v)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(v):.1f}"


def _row_md(name: str, bytes_b: int, fp16_total: int, top1_m: float, top1_s: float,
            cos_m: float, cos_s: float,
            pre_tps=None, dec_tps=None) -> str:
    ratio = fp16_total / bytes_b if bytes_b > 0 else float("inf")
    return (f"| {name} | {bytes_b/1024:,.0f} | {ratio:.2f}× | "
            f"{top1_m*100:.1f}% ± {top1_s*100:.1f}% | "
            f"{cos_m:.4f} ± {cos_s:.4f} | "
            f"{_fmt_tps(pre_tps)} | {_fmt_tps(dec_tps)} |")


def write_report(out_dir: Path, args, dims: dict, runtime_s: float,
                  viz: dict | None, exp1: dict | None,
                  exp2: dict | None, exp3: dict | None) -> None:
    fp16_total = fp16_baseline_bytes(args, dims, DTYPE_MAP[args.dtype])
    L = []
    L.append(f"# KVCascade evaluation: `{args.model}`")
    L.append("")
    L.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- **Total runtime**: {runtime_s/60:.1f} minutes")
    L.append(f"- **Samples**: {args.samples} non-overlapping wikitext-103 chunks")
    L.append(f"- **Context length**: {args.ctx_len} (prefill {args.ctx_len - args.decode_len}, decode {args.decode_len})")
    L.append(f"- **Dtype**: `{args.dtype}`, **device**: `{args.device}`, **seed**: {args.seed}")
    L.append(f"- **Quant tier**: `k_bits={args.k_bits}`, `v_bits={args.v_bits}`, single tier")
    L.append("")
    L.append("## Model")
    L.append("")
    L.append("| Property | Value |")
    L.append("|---|---|")
    L.append(f"| Name | `{args.model}` |")
    L.append(f"| Layers | {dims['n_layers']} |")
    L.append(f"| Query heads | {dims['n_heads']} |")
    L.append(f"| KV heads | {dims['n_kv_heads']} |")
    L.append(f"| Head dim | {dims['head_dim']} |")
    L.append(f"| fp16 baseline cache | {fp16_total/1024:,.0f} KiB |")
    L.append("")

    if viz is not None:
        frac_uni = viz["entropy_mean"] / viz["uniform_max"]
        L.append("## Attention pattern analysis")
        L.append("")
        L.append(f"Computed on the first sample's first {viz['viz_ctx_len']} tokens.")
        L.append("")
        L.append("| Statistic | Value |")
        L.append("|---|---|")
        L.append(f"| Mean entropy | {viz['entropy_mean']:.2f} nats ({frac_uni:.1%} of uniform-max {viz['uniform_max']:.2f}) |")
        L.append(f"| Median entropy | {viz['entropy_median']:.2f} nats |")
        L.append(f"| Range | [{viz['entropy_min']:.2f}, {viz['entropy_max']:.2f}] |")
        L.append(f"| Peakiest head | layer {viz['peaky'][0]}, head {viz['peaky'][1]} |")
        L.append(f"| Most diffuse head | layer {viz['diffuse'][0]}, head {viz['diffuse'][1]} |")
        L.append("")
        if frac_uni > 0.7:
            L.append("> Mean entropy > 70% of uniform — attention is **diffuse** on this workload. "
                     "Eviction-only caches (H2O) should struggle; mixed-precision (KVCascade) should win.")
        elif frac_uni < 0.3:
            L.append("> Mean entropy < 30% of uniform — attention is **peaky**. "
                     "Eviction-based caches have a structural advantage.")
        L.append("")
        L.append("![Per-layer attention received](figures/attn_received.png)")
        L.append("")
        L.append("![Entropy histogram](figures/entropy_histogram.png)")
        L.append("")
        L.append("![Peakiest vs most-diffuse head](figures/peaky_diffuse_detail.png)")
        L.append("")

    if exp1 is not None:
        L.append("## Experiment 1: Compression sweep")
        L.append("")
        L.append("How few bytes does KVCascade need to match uniform TurboQuant's quality?")
        L.append("")
        L.append("| Config | Bytes (KiB) | Compression vs fp16 | Top-1 | Cos sim | Prefill (tok/s) | Decode (tok/s) |")
        L.append("|---|---|---|---|---|---|---|")
        L.append(_row_md("uniform `k=6/v=2`", exp1["uniform"]["bytes"], fp16_total,
                          exp1["uniform"]["top1_mean"], exp1["uniform"]["top1_std"],
                          exp1["uniform"]["cos_mean"],  exp1["uniform"]["cos_std"],
                          exp1["uniform"].get("prefill_tok_per_s"),
                          exp1["uniform"].get("decode_tok_per_s")))
        for r in exp1["rows"]:
            label = f"KVCascade @ {r['ratio']:.4g}× (fp={r['fp_cap']}, qt={r['qt_cap']})"
            L.append(_row_md(label, r["bytes"], fp16_total,
                              r["top1_mean"], r["top1_std"], r["cos_mean"], r["cos_std"],
                              r.get("prefill_tok_per_s"), r.get("decode_tok_per_s")))
        L.append("")
        if exp1["headline"]:
            L.append(f"**Headline**: KVCascade {exp1['headline']}.")
            L.append("")
        L.append("![Compression curve](figures/exp1_compression.png)")
        L.append("")

    if exp2 is not None:
        L.append("## Experiment 2: Iso-byte head-to-head")
        L.append("")
        L.append("At the same byte budget (= uniform's), compare four cache strategies.")
        L.append("")
        L.append("| Config | Bytes (KiB) | Compression vs fp16 | Top-1 | Cos sim | Prefill (tok/s) | Decode (tok/s) |")
        L.append("|---|---|---|---|---|---|---|")
        for name, r in exp2["rows"]:
            L.append(_row_md(name, r["bytes"], fp16_total,
                              r["top1_mean"], r["top1_std"], r["cos_mean"], r["cos_std"],
                              r.get("prefill_tok_per_s"), r.get("decode_tok_per_s")))
        L.append("")
        L.append("![Iso-byte bar chart](figures/exp2_isobyte.png)")
        L.append("")
        # Auto-observation: ablation deltas (uniform → H2O → H2O+ring → KVCascade).
        rows_d = {n: r for n, r in exp2["rows"]}
        h2o_ring_key = f"H2O (ring={args.ring_size})"
        if "uniform k=6/v=2" in rows_d and "KVCascade (ring + fp + quant)" in rows_d:
            uni      = rows_d["uniform k=6/v=2"]["top1_mean"]
            kvc      = rows_d["KVCascade (ring + fp + quant)"]["top1_mean"]
            h2o      = rows_d.get("H2O (ring=0)",  {}).get("top1_mean")
            h2o_ring = rows_d.get(h2o_ring_key,    {}).get("top1_mean")
            L.append(f"**Δ at iso-byte**: KVCascade vs uniform = {(kvc-uni)*100:+.1f} pp.")
            if h2o is not None:
                L.append(f"  H2O (ring=0) vs uniform = {(h2o-uni)*100:+.1f} pp.")
            if h2o_ring is not None:
                L.append(f"  H2O (ring={args.ring_size}) vs uniform = {(h2o_ring-uni)*100:+.1f} pp.")
                if h2o is not None:
                    L.append(f"  Recency-ring lift on H2O = {(h2o_ring-h2o)*100:+.1f} pp "
                             f"(adding ring={args.ring_size} on top of plain H2O).")
            if h2o_ring is not None:
                L.append(f"  Quantization lift on H2O+ring = {(kvc-h2o_ring)*100:+.1f} pp "
                         f"(KVCascade adds the quant tier on top of H2O+ring).")
            L.append("")

    if exp3 is not None:
        L.append("## Experiment 3: Split sweep at fixed budget")
        L.append("")
        L.append(f"Total bytes fixed at uniform's iso-byte budget. `ring_size={args.ring_size}` "
                 f"throughout; `fp_capacity` is swept and `quant_capacity` is derived from the budget.")
        L.append("")
        L.append("| ring | fp_cap | qt_cap | Bytes (KiB) | Top-1 | Cos sim | Prefill (tok/s) | Decode (tok/s) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in exp3["rows"]:
            L.append(f"| {args.ring_size} | {r['fp_cap']} | {r['qt_cap']} | "
                     f"{r['bytes']/1024:,.0f} | "
                     f"{r['top1_mean']*100:.1f}% ± {r['top1_std']*100:.1f}% | "
                     f"{r['cos_mean']:.4f} ± {r['cos_std']:.4f} | "
                     f"{_fmt_tps(r.get('prefill_tok_per_s'))} | {_fmt_tps(r.get('decode_tok_per_s'))} |")
        L.append("")
        if exp3["best"]:
            b = exp3["best"]
            L.append(f"**Best split**: `fp={b['fp_cap']}, qt={b['qt_cap']}` → "
                     f"top-1 {b['top1_mean']*100:.1f}% ± {b['top1_std']*100:.1f}%.")
            L.append("")
        L.append("![Split sweep](figures/exp3_split.png)")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"*Raw per-sample results in `raw.json`. Reproduce with: `{Path(sys.argv[0]).name} "
             + " ".join(sys.argv[1:]) + "`*")

    (out_dir / "report.md").write_text("\n".join(L))


def write_raw_json(out_dir: Path, args, dims: dict, runtime_s: float,
                    viz: dict | None, exp1: dict | None,
                    exp2: dict | None, exp3: dict | None) -> None:
    payload = {
        "args": vars(args),
        "dims": dims,
        "runtime_s": runtime_s,
        "viz": viz,
        "exp1": None,
        "exp2": None,
        "exp3": None,
    }
    if exp1 is not None:
        payload["exp1"] = {
            "uniform": _trim(exp1["uniform"]),
            "rows": [_trim(r) for r in exp1["rows"]],
            "headline": exp1["headline"],
            "uniform_bytes_per_lh": exp1["uniform_bytes_per_lh"],
        }
    if exp2 is not None:
        payload["exp2"] = {
            "rows": [(name, _trim(r)) for name, r in exp2["rows"]],
            "uniform_bytes": exp2["uniform_bytes"],
            "fp16_baseline_bytes": exp2["fp16_baseline_bytes"],
        }
    if exp3 is not None:
        payload["exp3"] = {
            "rows": [_trim(r) for r in exp3["rows"]],
            "best": _trim(exp3["best"]) if exp3["best"] else None,
            "uniform_bytes_per_lh": exp3["uniform_bytes_per_lh"],
        }
    (out_dir / "raw.json").write_text(json.dumps(payload, indent=2, default=str))


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    out_dir = setup_output_dir(args)
    print(f"output: {out_dir}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    t0 = time.time()
    model, tok, device, dtype = setup_model(args)
    dims = get_dims(model)
    print(f"L={dims['n_layers']}  H_q={dims['n_heads']}  H_kv={dims['n_kv_heads']}  D={dims['head_dim']}",
          flush=True)

    samples = get_samples(tok, args.samples, args.ctx_len)
    if not samples:
        print("no samples loaded; aborting", flush=True)
        return
    T_pre = args.ctx_len - args.decode_len
    print(f"loaded {len(samples)} samples; T_pre={T_pre}, T_dec={args.decode_len}", flush=True)

    print("computing fp reference logits...", flush=True)
    ref_decs = compute_fp_reference(model, samples, T_pre, args.decode_len, device)

    viz = None
    if not args.skip_viz:
        viz = viz_attention_patterns(model, samples, dims, args.viz_ctx_len, out_dir, device)

    exp1 = exp2 = exp3 = None
    if not args.skip_exp1:
        exp1 = exp1_compression_sweep(model, samples, ref_decs, args, dims, dtype,
                                       device, T_pre, out_dir)
    if not args.skip_exp2:
        exp2 = exp2_iso_byte(model, samples, ref_decs, args, dims, dtype,
                              device, T_pre, out_dir, exp1_results=exp1)
    if not args.skip_exp3:
        exp3 = exp3_split_sweep(model, samples, ref_decs, args, dims, dtype,
                                 device, T_pre, out_dir)

    runtime = time.time() - t0
    write_raw_json(out_dir, args, dims, runtime, viz, exp1, exp2, exp3)
    write_report(out_dir, args, dims, runtime, viz, exp1, exp2, exp3)
    print(f"\nReport written to {out_dir / 'report.md'}", flush=True)
    print(f"Total runtime: {runtime/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
