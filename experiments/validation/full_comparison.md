# Full comparison: 8 configs × 6 (model, ctx) cells

Each cell shows top-1 ± sd. Configs:
- **uniform-Prod / uniform-MSE**: TurboQuant baselines at the chosen quant variant.
- **H2O**: eviction-only (ring=0), no quant — identical across {prod, mse}.
- **H2O+ring**: H2O with the SnapKV-style recency window (ring=8 by default).
- **KVC uniform-Prod / -MSE**: KVCascade with per-head capacity uniform across heads.
- **KVC AdaKV-Prod / -MSE**: KVCascade with Ada-KV adaptive per-head capacity (`floor_alpha=0.5`).

Source raw.json files: `experiments/validation/runs/<slug>_<ctxk>k_<mode>/raw.json` or `outputs/<slug>_<ctxk>k_<mode>/raw.json`.


## Qwen3-0.6B @ ctx=4096

| metric | uniform-Prod | uniform-MSE | H2O | H2O+ring | KVC uniform-Prod | KVC uniform-MSE | KVC AdaKV-Prod | KVC AdaKV-MSE |
|---|---|---|---|---|---|---|---|---|
| top-1 | — | 84.4% ± 5.8% | 20.5% ± 5.5% | 73.4% ± 5.3% | — | 95.0% ± 3.1% | — | 94.5% ± 2.9% |
| cos sim | — | 0.9812 | 0.5580 | 0.9679 | — | 0.9974 | — | 0.9972 |
| decode tok/s | — | 13.8 | 15.4 | 13.1 | — | 7.2 | — | 6.6 |

- KVC AdaKV-MSE − KVC uniform-MSE = -0.5 pp
- KVC uniform-MSE − strongest uniform = +10.6 pp
- KVC AdaKV-MSE − strongest uniform = +10.2 pp

## Qwen3-0.6B @ ctx=8192

_no data yet_


## Llama-3.2-1B @ ctx=4096

sources: `llama_1B_4k_prod`, `llama_1B_4k_mse`

| metric | uniform-Prod | uniform-MSE | H2O | H2O+ring | KVC uniform-Prod | KVC uniform-MSE | KVC AdaKV-Prod | KVC AdaKV-MSE |
|---|---|---|---|---|---|---|---|---|
| top-1 | 87.2% ± 5.2% | 88.2% ± 5.3% | 24.1% ± 6.6% | 76.1% ± 6.2% | 96.7% ± 2.3% | 96.8% ± 2.4% | 96.6% ± 2.1% | 96.8% ± 2.3% |
| cos sim | 0.9839 | 0.9868 | 0.7455 | 0.9675 | 0.9987 | 0.9988 | 0.9986 | 0.9985 |
| decode tok/s | 15.1 | 27.2 | 23.5 | 19.4 | 9.2 | 13.9 | 8.4 | 13.0 |

- uniform-MSE − uniform-Prod = +1.0 pp
- KVC AdaKV-Prod − KVC uniform-Prod = -0.1 pp
- KVC AdaKV-MSE − KVC uniform-MSE = -0.1 pp
- KVC uniform-Prod − strongest uniform = +8.4 pp
- KVC uniform-MSE − strongest uniform = +8.6 pp
- KVC AdaKV-Prod − strongest uniform = +8.3 pp
- KVC AdaKV-MSE − strongest uniform = +8.6 pp

## Llama-3.2-1B @ ctx=8192

_no data yet_


## OLMo-2-1B @ ctx=4096

sources: `olmo2_1B_4k_prod`, `olmo2_1B_4k_mse`

| metric | uniform-Prod | uniform-MSE | H2O | H2O+ring | KVC uniform-Prod | KVC uniform-MSE | KVC AdaKV-Prod | KVC AdaKV-MSE |
|---|---|---|---|---|---|---|---|---|
| top-1 | 91.5% ± 3.4% | 91.2% ± 3.6% | 25.6% ± 7.1% | 79.1% ± 5.3% | 96.8% ± 2.3% | 97.2% ± 1.9% | 96.5% ± 2.4% | 96.8% ± 2.3% |
| cos sim | 0.9997 | 0.9997 | 0.9723 | 0.9984 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| decode tok/s | 20.3 | 26.5 | 30.4 | 20.3 | 12.6 | 13.9 | 11.1 | 12.4 |

- uniform-MSE − uniform-Prod = -0.2 pp
- KVC AdaKV-Prod − KVC uniform-Prod = -0.2 pp
- KVC AdaKV-MSE − KVC uniform-MSE = -0.4 pp
- KVC uniform-Prod − strongest uniform = +5.3 pp
- KVC uniform-MSE − strongest uniform = +5.7 pp
- KVC AdaKV-Prod − strongest uniform = +5.0 pp
- KVC AdaKV-MSE − strongest uniform = +5.3 pp

## OLMo-2-1B @ ctx=8192

_no data yet_

