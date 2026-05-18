"""Task 4 — Build quant_mode_comparison.md from per-mode `raw.json` files in runs/.

Reads runs/qwen3_4k_prod/raw.json and runs/qwen3_4k_mse/raw.json (and any other
(model, ctx, mode) tuples that exist), pulls Exp 2 (iso-byte head-to-head) results,
and emits a table per (model, ctx) showing all four configs:
  uniform-Prod, uniform-MSE, KVCascade-Prod, KVCascade-MSE
with top-1 mean ± sd, cos sim, KVCascade Δ over the strongest uniform, and decode tok/s.
"""

import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"


def fmt_pct(vals):
    if not vals:
        return "—"
    m = statistics.mean(vals) * 100
    sd = statistics.pstdev(vals) * 100 if len(vals) > 1 else 0.0
    return f"{m:.1f}% ± {sd:.1f}%"


def fmt_cos(vals):
    if not vals:
        return "—"
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"


def find_runs():
    """Return {(model_slug, ctx, mode): raw_dict} for each runs/<key>/raw.json."""
    out = {}
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.iterdir()):
        rj = d / "raw.json"
        if not rj.exists():
            continue
        # Expected dir name format: <slug>_<ctx>k_<mode>
        m = re.match(r"(.+?)_(\d+)k_(prod|mse)$", d.name)
        if not m:
            continue
        slug, ctxk, mode = m.group(1), int(m.group(2)), m.group(3)
        ctx = ctxk * 1024
        with open(rj) as f:
            data = json.load(f)
        out[(slug, ctx, mode)] = (data, d)
    return out


def extract_exp2(data, label_substr: str):
    """Pull aggregated Exp 2 result by label substring match.

    eval.py raw.json format: exp2.rows = [[label_str, payload_dict], ...]. Payload has
    top1_mean/std, cos_mean/std, decode_tok_per_s, etc. We don't get per-sample arrays,
    so we report eval's pre-aggregated mean ± std directly.
    """
    rows = (data.get("exp2") or {}).get("rows", [])
    for label, payload in rows:
        if label_substr.lower() in label.lower():
            return payload
    return None


def fmt_pct_agg(payload):
    if not payload or "top1_mean" not in payload:
        return "—"
    m = payload["top1_mean"] * 100
    sd = payload.get("top1_std", 0.0) * 100
    return f"{m:.1f}% ± {sd:.1f}%"


def fmt_cos_agg(payload):
    if not payload or "cos_mean" not in payload:
        return "—"
    return f"{payload['cos_mean']:.4f} ± {payload.get('cos_std', 0.0):.4f}"


def main():
    runs = find_runs()
    if not runs:
        print(f"No runs found under {RUNS}")
        return

    # Group by (slug, ctx)
    grouped = {}
    for (slug, ctx, mode), (data, d) in runs.items():
        grouped.setdefault((slug, ctx), {})[mode] = (data, d)

    lines = [
        "# Iso-byte head-to-head: Prod vs MSE quant mode (Task 4)\n",
        "For each (model, ctx) pair we ran eval.py's Exp 2 (iso-byte head-to-head) "
        "twice — once with `--quant-mode prod` (TurboQuant Prod, the existing baseline) "
        "and once with `--quant-mode mse` (PolarQuant only, the new variant). The "
        "quant_mode applies to BOTH the uniform baseline and KVCascade's quant tier.",
        "",
        "Both modes are at iso-byte: the eval automatically derives KVCascade's qt_cap "
        "from uniform's total byte budget for that mode, so per-cell within a column we "
        "compare KVCascade against the strongest uniform variant at the same total bytes.",
        "",
    ]

    for (slug, ctx), modes in sorted(grouped.items()):
        lines.append(f"## {slug} @ ctx={ctx}")
        lines.append("")
        lines.append("| metric | uniform-Prod | uniform-MSE | KVCascade-Prod | KVCascade-MSE |")
        lines.append("|---|---|---|---|---|")

        cells = {"top1": {}, "cos": {}, "decode": {}, "bytes": {}}
        for mode_key, label_uni, label_kvc in [
            ("prod", "uniform-Prod", "KVCascade-Prod"),
            ("mse",  "uniform-MSE",  "KVCascade-MSE"),
        ]:
            data_modes = modes.get(mode_key)
            if not data_modes:
                for label in (label_uni, label_kvc):
                    cells["top1"][label] = "—"
                    cells["cos"][label] = "—"
                    cells["decode"][label] = "—"
                    cells["bytes"][label] = "—"
                continue
            data, _ = data_modes
            for substr, label in [("uniform", label_uni), ("KVCascade", label_kvc)]:
                p = extract_exp2(data, substr)
                cells["top1"][label] = fmt_pct_agg(p)
                cells["cos"][label]  = fmt_cos_agg(p)
                cells["decode"][label] = (f"{p['decode_tok_per_s']:.1f}"
                                          if p and "decode_tok_per_s" in p else "—")
                cells["bytes"][label] = (f"{p['bytes']/1024:.0f} KiB"
                                         if p and "bytes" in p else "—")

        for metric, prettyname in [("top1", "top-1"), ("cos", "cos sim"),
                                    ("decode", "decode tok/s"), ("bytes", "bytes")]:
            row = f"| {prettyname} |"
            for label in ["uniform-Prod", "uniform-MSE", "KVCascade-Prod", "KVCascade-MSE"]:
                row += f" {cells[metric].get(label, '—')} |"
            lines.append(row)

        # Δ KVCascade vs strongest uniform
        # Use top1 mean only.
        def _mean(s):
            if s == "—" or s is None:
                return None
            try:
                return float(s.split("%")[0])
            except Exception:
                return None
        unip = _mean(cells["top1"].get("uniform-Prod"))
        unim = _mean(cells["top1"].get("uniform-MSE"))
        kvcp = _mean(cells["top1"].get("KVCascade-Prod"))
        kvcm = _mean(cells["top1"].get("KVCascade-MSE"))
        strongest_uni = max(x for x in [unip, unim] if x is not None) if (unip or unim) else None
        if strongest_uni is not None:
            lines.append("")
            if kvcp is not None:
                lines.append(f"- KVCascade-Prod vs strongest uniform (max of Prod/MSE): "
                             f"Δ = {kvcp - strongest_uni:+.1f} pp")
            if kvcm is not None:
                lines.append(f"- KVCascade-MSE vs strongest uniform: "
                             f"Δ = {kvcm - strongest_uni:+.1f} pp")
            if unip is not None and unim is not None:
                lines.append(f"- uniform-MSE − uniform-Prod = {unim - unip:+.1f} pp "
                             f"(does MSE help the uniform baseline on this config?)")
        lines.append("")

    (HERE / "quant_mode_comparison.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'quant_mode_comparison.md'}")


if __name__ == "__main__":
    main()
