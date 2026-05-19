"""Aggregate Modal-eval outputs into a single per-(model, ctx) comparison table.

Reads runs/<slug>_<ctxk>k_{prod,mse}/raw.json (or outputs/<slug>_<ctxk>k_{prod,mse}/raw.json),
where each raw.json comes from an eval.py invocation with
`--quant-mode {prod,mse} --adakv-alpha 1.0 --also-adakv-alpha 0.5`. That invocation
produces Exp 2 rows for: uniform, H2O, H2O+ring, KVCascade-uniform, KVCascade-Ada-KV.

Per (model, ctx), we merge the prod and mse outputs to get 8 unique configs:
  - uniform-Prod, uniform-MSE
  - H2O (ring=0) — identical across modes; we keep the prod copy
  - H2O+ring=R   — identical across modes
  - KVCascade-uniform-Prod, KVCascade-uniform-MSE
  - KVCascade-AdaKV-Prod,   KVCascade-AdaKV-MSE

Writes `experiments/validation/full_comparison.md`.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
# Search both validation/runs (sprint outputs) and project-level outputs.
SEARCH_DIRS = [HERE / "runs", HERE.parent.parent / "outputs"]


def _fmt_pct(payload):
    if not payload or "top1_mean" not in payload:
        return "—"
    return f"{payload['top1_mean']*100:.1f}% ± {payload.get('top1_std',0.0)*100:.1f}%"


def _fmt_cos(payload):
    if not payload or "cos_mean" not in payload:
        return "—"
    return f"{payload['cos_mean']:.4f}"


def _fmt_tps(payload):
    if not payload or "decode_tok_per_s" not in payload:
        return "—"
    return f"{payload['decode_tok_per_s']:.1f}"


def _find_run(slug_prefix: str, ctxk: int, mode: str):
    """Return raw.json dict for runs/<slug_prefix>_<ctxk>k_<mode>/ if it exists."""
    candidates = [
        f"{slug_prefix}_{ctxk}k_{mode}",
        f"{slug_prefix}_4k_{mode}" if ctxk == 4 else None,
        f"{slug_prefix}_8k_{mode}" if ctxk == 8 else None,
        f"{slug_prefix}_{ctxk}K_{mode}",
    ]
    candidates = [c for c in candidates if c]
    for base in SEARCH_DIRS:
        for c in candidates:
            p = base / c / "raw.json"
            if p.exists():
                with open(p) as f:
                    return json.load(f), p
    return None, None


def _extract_row(data, label_substring: str):
    """Find an Exp 2 row by label substring."""
    rows = (data.get("exp2") or {}).get("rows", [])
    for label, payload in rows:
        if label_substring.lower() in label.lower():
            return payload
    return None


def _extract_kvc_rows(data):
    """Return (kvc_uniform, kvc_adakv) payloads from an Exp 2 with both KVC variants."""
    rows = (data.get("exp2") or {}).get("rows", [])
    uni, ada = None, None
    for label, payload in rows:
        l = label.lower()
        if "kvcascade" not in l and "kvc" not in l:
            continue
        if "adakv" in l or "α=" in label or "alpha" in l:
            ada = payload
        elif "uniform" in l:
            uni = payload
    # If no uniform-tagged row, fall back to a row without an Ada-KV marker.
    if uni is None:
        for label, payload in rows:
            l = label.lower()
            if ("kvcascade" in l or "kvc" in l) and "adakv" not in l and "α=" not in label:
                uni = payload
                break
    return uni, ada


def main():
    # (slug_prefix, model_pretty, ctxk, decode_len)
    cells = [
        ("qwen3_0.6B", "Qwen3-0.6B", 4, 64),
        ("qwen3_0.6B", "Qwen3-0.6B", 8, 128),
        ("llama_1B",   "Llama-3.2-1B", 4, 64),
        ("llama_1B",   "Llama-3.2-1B", 8, 128),
        ("olmo2_1B",   "OLMo-2-1B",   4, 64),
        ("olmo2_1B",   "OLMo-2-1B",   8, 128),
    ]
    lines = [
        "# Full comparison: 8 configs × 6 (model, ctx) cells\n",
        "Each cell shows top-1 ± sd. Configs:",
        "- **uniform-Prod / uniform-MSE**: TurboQuant baselines at the chosen quant variant.",
        "- **H2O**: eviction-only (ring=0), no quant — identical across {prod, mse}.",
        "- **H2O+ring**: H2O with the SnapKV-style recency window (ring=8 by default).",
        "- **KVC uniform-Prod / -MSE**: KVCascade with per-head capacity uniform across heads.",
        "- **KVC AdaKV-Prod / -MSE**: KVCascade with Ada-KV adaptive per-head capacity (`floor_alpha=0.5`).",
        "",
        "Source raw.json files: `experiments/validation/runs/<slug>_<ctxk>k_<mode>/raw.json` or `outputs/<slug>_<ctxk>k_<mode>/raw.json`.",
        "",
    ]
    for slug, pretty, ctxk, _ in cells:
        prod_data, prod_path = _find_run(slug, ctxk, "prod")
        mse_data,  mse_path  = _find_run(slug, ctxk, "mse")
        if not prod_data and not mse_data:
            lines.append(f"\n## {pretty} @ ctx={ctxk*1024}\n\n_no data yet_\n")
            continue
        # Extract per-config payloads.
        uni_p   = _extract_row(prod_data, "uniform") if prod_data else None
        uni_m   = _extract_row(mse_data,  "uniform") if mse_data  else None
        h2o     = _extract_row(prod_data or mse_data, "H2O (ring=0)")
        h2or    = _extract_row(prod_data or mse_data, "H2O (ring=")
        if h2or and h2o and h2or is h2o:
            # Same row matched — try the second one explicitly.
            rows = ((prod_data or mse_data).get("exp2") or {}).get("rows", [])
            h2o_count = 0
            for label, payload in rows:
                if "H2O" in label:
                    h2o_count += 1
                    if h2o_count == 2:
                        h2or = payload
                        break
        kvc_uni_p, kvc_ada_p = _extract_kvc_rows(prod_data) if prod_data else (None, None)
        kvc_uni_m, kvc_ada_m = _extract_kvc_rows(mse_data)  if mse_data  else (None, None)

        lines.append(f"\n## {pretty} @ ctx={ctxk*1024}")
        if prod_path:
            lines.append(f"\nsources: `{prod_path.parent.name}`" + (f", `{mse_path.parent.name}`" if mse_path else ""))
        lines.append("")
        lines.append("| metric | uniform-Prod | uniform-MSE | H2O | H2O+ring | KVC uniform-Prod | KVC uniform-MSE | KVC AdaKV-Prod | KVC AdaKV-MSE |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        rows_iter = [
            ("top-1",        _fmt_pct, [uni_p, uni_m, h2o, h2or, kvc_uni_p, kvc_uni_m, kvc_ada_p, kvc_ada_m]),
            ("cos sim",      _fmt_cos, [uni_p, uni_m, h2o, h2or, kvc_uni_p, kvc_uni_m, kvc_ada_p, kvc_ada_m]),
            ("decode tok/s", _fmt_tps, [uni_p, uni_m, h2o, h2or, kvc_uni_p, kvc_uni_m, kvc_ada_p, kvc_ada_m]),
        ]
        for name, fmt, payloads in rows_iter:
            row = f"| {name} |" + "".join(f" {fmt(p)} |" for p in payloads)
            lines.append(row)

        # Δ analyses
        def _m(p):
            return None if (not p or "top1_mean" not in p) else p["top1_mean"]*100
        m = [_m(p) for p in [uni_p, uni_m, h2o, h2or, kvc_uni_p, kvc_uni_m, kvc_ada_p, kvc_ada_m]]
        uni_strong = max(x for x in [m[0], m[1]] if x is not None) if (m[0] is not None or m[1] is not None) else None
        notes = []
        if m[0] is not None and m[1] is not None:
            notes.append(f"uniform-MSE − uniform-Prod = {m[1]-m[0]:+.1f} pp")
        if m[4] is not None and m[6] is not None:
            notes.append(f"KVC AdaKV-Prod − KVC uniform-Prod = {m[6]-m[4]:+.1f} pp")
        if m[5] is not None and m[7] is not None:
            notes.append(f"KVC AdaKV-MSE − KVC uniform-MSE = {m[7]-m[5]:+.1f} pp")
        if uni_strong is not None:
            for label, idx in [("KVC uniform-Prod", 4), ("KVC uniform-MSE", 5),
                                ("KVC AdaKV-Prod", 6), ("KVC AdaKV-MSE", 7)]:
                if m[idx] is not None:
                    notes.append(f"{label} − strongest uniform = {m[idx]-uni_strong:+.1f} pp")
        if notes:
            lines.append("")
            for n in notes:
                lines.append(f"- {n}")
    out = HERE / "full_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
