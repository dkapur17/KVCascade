# Iso-byte head-to-head: Prod vs MSE quant mode (Task 4)

For each (model, ctx) pair we ran eval.py's Exp 2 (iso-byte head-to-head) twice — once with `--quant-mode prod` (TurboQuant Prod, the existing baseline) and once with `--quant-mode mse` (PolarQuant only, the new variant). The quant_mode applies to BOTH the uniform baseline and KVCascade's quant tier.

Both modes are at iso-byte: the eval automatically derives KVCascade's qt_cap from uniform's total byte budget for that mode, so per-cell within a column we compare KVCascade against the strongest uniform variant at the same total bytes.

## qwen3 @ ctx=4096

| metric | uniform-Prod | uniform-MSE | KVCascade-Prod | KVCascade-MSE |
|---|---|---|---|---|
| top-1 | 68.6% ± 7.6% | 85.9% ± 5.0% | 87.6% ± 5.0% | 95.4% ± 2.6% |
| cos sim | 0.9245 ± 0.0221 | 0.9818 ± 0.0051 | 0.9864 ± 0.0077 | 0.9971 ± 0.0042 |
| decode tok/s | 6.9 | 9.0 | 4.3 | 4.7 |
| bytes | 120064 KiB | 118272 KiB | 120056 KiB | 118272 KiB |

- KVCascade-Prod vs strongest uniform (max of Prod/MSE): Δ = +1.7 pp
- KVCascade-MSE vs strongest uniform: Δ = +9.5 pp
- uniform-MSE − uniform-Prod = +17.3 pp (does MSE help the uniform baseline on this config?)

