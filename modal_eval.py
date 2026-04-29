"""Run KVCascade evals on Modal — one container per (model, ctx) config, in parallel.

Mirrors run_evals.sh but parallelized: 6 containers fire concurrently, each on its
own GPU. Total wall time ~= longest single run instead of the sum.

One-time setup:
    pip install modal
    modal token new                                    # auth
    modal secret create huggingface HF_TOKEN=hf_xxx    # for gated Llama-3.2

Usage:
    modal run modal_eval.py                              # all 6 runs in parallel on A100
    modal run modal_eval.py --runs 4k                    # only the ctx=4096 set
    modal run modal_eval.py --runs qwen3_4k              # one specific key
    modal run modal_eval.py --runs qwen3_4k,olmo_8k
    KVCASCADE_GPU=H100 modal run modal_eval.py           # bump GPU type for this run
    KVCASCADE_GPU=H200 modal run modal_eval.py --runs 8k

Pull results to local once everything finishes:
    modal volume get kvcascade-eval-out / ./outputs/
"""

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

# ----------------------------------------------------------------------
# Image: project source + ML stack
# ----------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.45.0",
        "datasets>=2.20.0",
        "matplotlib",
        "numpy",
        "scipy",
        "accelerate",
        "huggingface_hub",
    )
    .add_local_dir(
        str(REPO_ROOT),
        "/workspace",
        ignore=["outputs/*", "__pycache__/*", ".git/*", "*.ipynb"],
    )
)

# Persistent volumes: outputs survive container exit; HF cache amortizes
# model downloads across reruns.
out_volume = modal.Volume.from_name("kvcascade-eval-out", create_if_missing=True)
hf_cache   = modal.Volume.from_name("kvcascade-hf-cache", create_if_missing=True)

app = modal.App("kvcascade-eval")

# ----------------------------------------------------------------------
# GPU. Set the KVCASCADE_GPU env var to override per invocation
# (decorator captures this at module load, so a CLI flag wouldn't work).
# Approximate Modal pricing (subject to change):
#   "A100"      ~ $1.32/hr  (40GB)        cheapest that fits the 1-2B models
#   "A100-80GB" ~ $1.85/hr
#   "H100"      ~ $3.95/hr
#   "H200"      ~ $4.54/hr               fastest, more HBM bandwidth
# ----------------------------------------------------------------------

GPU = os.environ.get("KVCASCADE_GPU", "A100")

# ----------------------------------------------------------------------
# Run definitions (mirror of run_evals.sh)
# ----------------------------------------------------------------------

# (key, model, ctx_len, decode_len, samples, out_subdir)
RUNS = [
    ("qwen3_4k", "Qwen/Qwen3-0.6B",         4096,  64, 50, "qwen3_0.6B_4k"),
    ("llama_4k", "meta-llama/Llama-3.2-1B", 4096,  64, 50, "llama_1B_4k"),
    ("olmo_4k",  "allenai/OLMo-2-0425-1B",  4096,  64, 50, "olmo2_1B_4k"),
    ("qwen3_8k", "Qwen/Qwen3-0.6B",         8192, 128, 50, "qwen3_0.6B_8k"),
    ("llama_8k", "meta-llama/Llama-3.2-1B", 8192, 128, 50, "llama_1B_8k"),
    ("olmo_8k",  "allenai/OLMo-2-0425-1B",  8192, 128, 50, "olmo2_1B_8k"),
]


# ----------------------------------------------------------------------
# Per-run container
# ----------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 6,                          # 6h hard cap per run
    volumes={
        "/outputs": out_volume,
        "/root/.cache/huggingface": hf_cache,     # persistent HF cache across runs
    },
    secrets=[modal.Secret.from_name("huggingface")],   # provides HF_TOKEN for gated Llama-3.2
)
def run_eval(key: str, model: str, ctx_len: int, decode_len: int,
             samples: int, out_subdir: str) -> dict:
    import os
    import subprocess
    import time

    # Surface the token under HF_TOKEN (the env var transformers checks). Modal
    # secrets set whatever keys are in them — if the user set the secret with a
    # non-canonical name, normalize here so transformers picks it up. We DON'T
    # call huggingface_hub.login() because it makes a whoami() HTTP call that
    # can transiently fail at container start; the env var alone is sufficient
    # for transformers.from_pretrained.
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
    if token:
        os.environ["HF_TOKEN"] = token
        print(f"[{key}] HF token loaded ({len(token)} chars)", flush=True)
    else:
        print(f"[{key}] WARNING: no HF token found in env "
              f"(checked HF_TOKEN, HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_HUB_TOKEN, HUGGINGFACE_TOKEN). "
              f"Gated models like Llama will fail.",
              flush=True)

    out_path = f"/outputs/{out_subdir}"
    print(f"[{key}] starting {model}  ctx={ctx_len}  dec={decode_len}  N={samples}",
          flush=True)
    t0 = time.time()
    result = subprocess.run(
        [
            "python", "/workspace/eval.py",
            "--model", model,
            "--ctx-len",    str(ctx_len),
            "--decode-len", str(decode_len),
            "--samples",    str(samples),
            "--out", out_path,
        ],
        cwd="/workspace",
    )
    elapsed = time.time() - t0
    out_volume.commit()                            # flush so the local `volume get` sees the writes

    ok = result.returncode == 0
    status = "OK" if ok else f"FAIL (exit {result.returncode})"
    print(f"[{key}] {status} in {elapsed/60:.1f} min", flush=True)
    return {"key": key, "ok": ok, "elapsed_min": elapsed / 60,
            "out_subdir": out_subdir}


# ----------------------------------------------------------------------
# Local entrypoint
# ----------------------------------------------------------------------

@app.local_entrypoint()
def main(runs: str = "all"):
    """Launch eval runs in parallel on Modal.

    runs: "all" (default), or comma-separated keys (e.g. "qwen3_4k,llama_8k"),
          or a substring filter (e.g. "4k" runs every key containing "4k").

    To change GPU type, set KVCASCADE_GPU before invoking, e.g.:
        KVCASCADE_GPU=H100 modal run modal_eval.py
    """
    if runs == "all":
        selected = RUNS
    else:
        keys = set(runs.split(","))
        # Allow exact match OR substring filter (e.g. "4k" matches all 4k runs).
        selected = [r for r in RUNS if r[0] in keys or any(k in r[0] for k in keys)]

    if not selected:
        print(f"no runs match: {runs!r}")
        print(f"available keys: {', '.join(r[0] for r in RUNS)}")
        return

    print(f"Launching {len(selected)} parallel runs on Modal ({GPU}):")
    for r in selected:
        print(f"  {r[0]:12} {r[1]:30} ctx={r[2]:>5} dec={r[3]:>3} N={r[4]}")
    print()

    # .starmap fans tuple args out to one container per tuple, all concurrent.
    # return_exceptions=True so a single container failure doesn't cancel the
    # other in-flight runs (default behavior is to abort the whole batch).
    raw_results = list(run_eval.starmap(selected, return_exceptions=True))

    # Pair each result back with its input key so failed slots are still labeled.
    results = []
    for run_args, r in zip(selected, raw_results):
        key = run_args[0]
        if isinstance(r, BaseException):
            results.append({"key": key, "ok": False, "elapsed_min": 0.0,
                            "out_subdir": run_args[5], "error": repr(r)})
        else:
            results.append(r)

    print("\n========== summary ==========")
    for r in sorted(results, key=lambda x: x["key"]):
        flag = "OK  " if r["ok"] else "FAIL"
        line = f"  {flag}  {r['key']:12}  {r['elapsed_min']:>5.1f} min  -> /outputs/{r['out_subdir']}"
        if not r["ok"] and r.get("error"):
            line += f"   error: {r['error']}"
        print(line)
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{n_ok}/{len(results)} runs succeeded")
    print("\nPull results locally with:")
    print("  modal volume get kvcascade-eval-out / ./outputs/")
