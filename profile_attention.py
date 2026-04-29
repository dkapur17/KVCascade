"""Profile per-phase timings inside KVCascade vs uniform TurboQuant attention.

Drives both cache classes directly with synthetic Q/K/V (no model forward), so the
measurements isolate cache work from model compute. Wraps key methods with cuda-sync'd
timers via runtime monkey-patching — no changes to the implementation files.

Splits the report into two phases:
  - PREFILL  : 1 call per layer with T_q = PREFILL_LEN
  - DECODE   : DECODE_STEPS calls per layer with T_q = 1

Run:
    /venv/kvcascade/bin/python profile_attention.py
"""

import contextlib
import statistics
import sys
import time

sys.path.insert(0, "src")

import torch

from kvcascade import KVCascadeCache
from turbo_attn import TurboQuantKVCache


# ---- Config (matches eval.py for Qwen3-0.6B B-iso) -----------------------------------
DEVICE        = torch.device("cuda")
DTYPE         = torch.bfloat16

NUM_LAYERS    = 28
NUM_HEADS     = 16
NUM_KV_HEADS  = 8
HEAD_DIM      = 128
CTX_LEN       = 4096
DECODE_STEPS  = 64
PREFILL_LEN   = CTX_LEN - DECODE_STEPS
SEED          = 42

RING_SIZE     = 8
FP_CAPACITY   = CTX_LEN // 16     # 256
QT_CAPACITY   = 3087              # iso-byte vs uniform k=6/v=2

K_BITS        = 6
V_BITS        = 2


# ---- Timer -----------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.acc = {}        # phase -> name -> [ms times]
        self._phase = "warmup"

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    @contextlib.contextmanager
    def measure(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000.0
            self.acc.setdefault(self._phase, {}).setdefault(name, []).append(ms)

    def report(self, label: str) -> None:
        print(f"\n========== {label} ==========")
        for phase in ("prefill", "decode"):
            phase_data = self.acc.get(phase, {})
            if not phase_data:
                continue
            print(f"\n--- {phase.upper()} (per layer call) ---")
            rows = [(name, times) for name, times in phase_data.items()]
            rows.sort(key=lambda r: -sum(r[1]))
            for name, times in rows:
                n = len(times)
                mean = statistics.mean(times)
                med  = statistics.median(times)
                total = sum(times)
                print(f"  {name:46}  n={n:>5}  "
                      f"mean={mean:>8.3f}ms  med={med:>8.3f}ms  total={total:>9.1f}ms")


# ---- Monkey-patch helpers --------------------------------------------------------------
def wrap_method(obj, attr: str, label: str, timer: Timer) -> None:
    """Replace bound method `obj.attr` with a timed wrapper. Behavior unchanged."""
    orig = getattr(obj, attr)
    def wrapped(*args, **kwargs):
        with timer.measure(label):
            return orig(*args, **kwargs)
    setattr(obj, attr, wrapped)


def patch_kvcascade(cache: KVCascadeCache, timer: Timer) -> None:
    wrap_method(cache, "attention",         "kvc.attention_TOTAL",     timer)
    wrap_method(cache, "_scores_and_meta",  "kvc.scores_and_meta",     timer)
    wrap_method(cache, "_values",           "kvc.values_dequant",      timer)
    wrap_method(cache, "_ingest_into_ring", "kvc.ingest_and_cascade",  timer)
    wrap_method(cache, "_cascade_vectorized", "kvc.cascade_only",      timer)
    # Per-quant-tier breakdown — these are the operations the user is most interested in.
    for i, buf in enumerate(cache.quant_buffers):
        wrap_method(buf, "score_pairwise", f"kvc.qbuf{i}.score_pairwise (K)", timer)
        wrap_method(buf, "values",         f"kvc.qbuf{i}.values (V)",          timer)


def patch_uniform(cache: TurboQuantKVCache, timer: Timer) -> None:
    wrap_method(cache, "attention", "uniform.attention_TOTAL", timer)
    wrap_method(cache, "_key_view", "uniform.key_view (K unpack)", timer)
    wrap_method(cache, "update",    "uniform.update (encode new)", timer)
    # Hot-path quantizer methods invoked inline in attention.
    for kq in cache.k_quantizers:
        wrap_method(kq, "estimate_ip_pairwise", "uniform.k.estimate_ip_pairwise", timer)
    for vq in cache.v_quantizers:
        wrap_method(vq, "decode", "uniform.v.decode", timer)


# ---- Cache constructors ----------------------------------------------------------------
def make_kvcascade() -> KVCascadeCache:
    return KVCascadeCache(
        num_layers=NUM_LAYERS, batch_size=1,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        ring_size=RING_SIZE, fp_capacity=FP_CAPACITY,
        quant_tiers=[(K_BITS, V_BITS, QT_CAPACITY)],
        m=HEAD_DIM, score_policy="ema", seed=SEED,
        device=DEVICE, dtype=DTYPE,
    )


def make_uniform() -> TurboQuantKVCache:
    return TurboQuantKVCache(
        num_layers=NUM_LAYERS, batch_size=1,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, max_seq_len=CTX_LEN,
        k_bits=K_BITS, v_bits=V_BITS, m=HEAD_DIM,
        seed=SEED, device=DEVICE, dtype=DTYPE,
    )


# ---- Synthetic Q/K/V generator ---------------------------------------------------------
def synth_qkv(T_q: int, T_k: int, gen: torch.Generator):
    Q = torch.randn(1, NUM_HEADS,    T_q, HEAD_DIM, generator=gen).to(device=DEVICE, dtype=DTYPE)
    K = torch.randn(1, NUM_KV_HEADS, T_k, HEAD_DIM, generator=gen).to(device=DEVICE, dtype=DTYPE)
    V = torch.randn(1, NUM_KV_HEADS, T_k, HEAD_DIM, generator=gen).to(device=DEVICE, dtype=DTYPE)
    return Q, K, V


# ---- Drivers ---------------------------------------------------------------------------
def run_kvcascade(cache: KVCascadeCache, timer: Timer) -> None:
    gen = torch.Generator().manual_seed(SEED)
    timer.set_phase("prefill")
    for l in range(NUM_LAYERS):
        Q, K, V = synth_qkv(PREFILL_LEN, PREFILL_LEN, gen)
        cache.attention(l, Q, K, V)
        del Q, K, V
    timer.set_phase("decode")
    for _ in range(DECODE_STEPS):
        for l in range(NUM_LAYERS):
            Q, K, V = synth_qkv(1, 1, gen)
            cache.attention(l, Q, K, V)
            del Q, K, V


def run_uniform(cache: TurboQuantKVCache, timer: Timer) -> None:
    gen = torch.Generator().manual_seed(SEED)
    timer.set_phase("prefill")
    for l in range(NUM_LAYERS):
        Q, K, V = synth_qkv(PREFILL_LEN, PREFILL_LEN, gen)
        cache.attention(l, Q, k_new=K, v_new=V)
        cache.update(l, K, V)
        del Q, K, V
    timer.set_phase("decode")
    for _ in range(DECODE_STEPS):
        for l in range(NUM_LAYERS):
            Q, K, V = synth_qkv(1, 1, gen)
            cache.attention(l, Q, k_new=K, v_new=V)
            cache.update(l, K, V)
            del Q, K, V


# ---- Main ------------------------------------------------------------------------------
def main() -> None:
    print(f"device={DEVICE}, dtype={DTYPE}")
    print(f"L={NUM_LAYERS}  H_q={NUM_HEADS}  H_kv={NUM_KV_HEADS}  D={HEAD_DIM}")
    print(f"ctx={CTX_LEN}  prefill={PREFILL_LEN}  decode_steps={DECODE_STEPS}")
    print(f"KVCascade: ring={RING_SIZE}  fp_cap={FP_CAPACITY}  qt_cap={QT_CAPACITY} (k={K_BITS}/v={V_BITS})")
    print(f"Uniform:   k={K_BITS}/v={V_BITS}, max_seq_len={CTX_LEN}\n")

    # Warmup: two short attention calls to amortize allocator + first-launch costs.
    print("warmup...")
    w_kvc = make_kvcascade()
    w_uni = make_uniform()
    gen   = torch.Generator().manual_seed(0)
    Q, K, V = synth_qkv(64, 64, gen)
    for _ in range(2):
        w_kvc.attention(0, Q, K, V)
    for _ in range(2):
        w_uni.attention(0, Q, k_new=K, v_new=V); w_uni.update(0, K, V)
    del w_kvc, w_uni, Q, K, V
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    print("running KVCascade...")
    timer_kvc = Timer()
    cache_kvc = make_kvcascade()
    patch_kvcascade(cache_kvc, timer_kvc)
    run_kvcascade(cache_kvc, timer_kvc)
    timer_kvc.report("KVCascade B (iso, ema)")
    del cache_kvc
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    print("\nrunning uniform TurboQuant...")
    timer_uni = Timer()
    cache_uni = make_uniform()
    patch_uniform(cache_uni, timer_uni)
    run_uniform(cache_uni, timer_uni)
    timer_uni.report("Uniform TurboQuant")

    # Side-by-side rollup of the totals so the gap is obvious at a glance.
    def total(timer, phase, name):
        return sum(timer.acc.get(phase, {}).get(name, []))

    print("\n========== ROLLUP ==========")
    for phase in ("prefill", "decode"):
        kvc_total = total(timer_kvc, phase, "kvc.attention_TOTAL")
        uni_total = total(timer_uni, phase, "uniform.attention_TOTAL")
        uni_update = total(timer_uni, phase, "uniform.update (encode new)")
        # uniform's "comparable total" includes update() since KVCascade's attention
        # already does its own ingest internally.
        uni_comparable = uni_total + uni_update
        ratio = kvc_total / uni_comparable if uni_comparable > 0 else float("nan")
        print(f"\n--- {phase.upper()} ---")
        print(f"  KVCascade attention_TOTAL          : {kvc_total:>9.1f} ms")
        print(f"  Uniform   attention_TOTAL + update : {uni_comparable:>9.1f} ms"
              f"  ({uni_total:.1f} attn + {uni_update:.1f} update)")
        print(f"  Ratio (KVC / Uniform)              : {ratio:>9.2f}x")


if __name__ == "__main__":
    main()
