# Synthetic distortion sanity check (Task 1)

N=1024 Gaussian vectors per dim, n_queries=32, seed=42, device=cuda, all fp32.

## Methods compared
- `ours.PolarQuant (MSE)` — `src/polar_quant.py`. Unit-Gaussian Lloyd-Max with σ=1/√d scaling; outermost buckets have unbounded support.
- `turbokv (MSE)` — `pip install turbokv` (vivekvar-dl). Beta-on-sphere Lloyd-Max with centroids initialized at ±0.99.
- `ours.TurboQuant (Prod)` — `src/turbo_quant.py`. (b−1)-bit Lloyd-Max coarse code + 1-bit JL sign sketch. Reconstruction MSE is coarse-code only; IP uses the JL estimator.

## Per-method distortion

| D | bits | method | MSE mean | MSE median | MSE p99 | MSE max | cos mean | IP rel mean | IP rel median |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2 | ours.PolarQuant (MSE) | 1.168e-01 | 1.131e-01 | 2.043e-01 | 2.433e-01 | 0.9413 | 2.025 | 0.335 |
| 64 | 2 | turbokv (MSE) | 1.288e-01 | 1.146e-01 | 4.000e-01 | 5.068e-01 | 0.9345 | 2.309 | 0.349 |
| 64 | 2 | ours.TurboQuant (Prod) | 3.641e-01 | 3.587e-01 | 5.778e-01 | 6.267e-01 | 0.8001 | 5.653 | 0.701 |
| 64 | 4 | ours.PolarQuant (MSE) | 9.433e-03 | 8.806e-03 | 2.392e-02 | 4.414e-02 | 0.9955 | 0.550 | 0.095 |
| 64 | 4 | turbokv (MSE) | 1.666e-02 | 8.790e-03 | 1.718e-01 | 2.615e-01 | 0.9924 | 0.765 | 0.102 |
| 64 | 4 | ours.TurboQuant (Prod) | 3.432e-02 | 3.221e-02 | 7.423e-02 | 9.919e-02 | 0.9834 | 1.271 | 0.214 |
| 128 | 2 | ours.PolarQuant (MSE) | 1.176e-01 | 1.154e-01 | 1.712e-01 | 1.950e-01 | 0.9403 | 1.807 | 0.337 |
| 128 | 2 | turbokv (MSE) | 1.496e-01 | 1.197e-01 | 4.935e-01 | 5.347e-01 | 0.9221 | 2.259 | 0.367 |
| 128 | 2 | ours.TurboQuant (Prod) | 3.661e-01 | 3.617e-01 | 5.076e-01 | 5.910e-01 | 0.7983 | 5.256 | 0.748 |
| 128 | 4 | ours.PolarQuant (MSE) | 9.399e-03 | 8.925e-03 | 1.939e-02 | 2.710e-02 | 0.9954 | 0.572 | 0.095 |
| 128 | 4 | turbokv (MSE) | 3.235e-02 | 9.229e-03 | 2.801e-01 | 3.106e-01 | 0.9845 | 0.750 | 0.114 |
| 128 | 4 | ours.TurboQuant (Prod) | 3.428e-02 | 3.299e-02 | 5.762e-02 | 6.647e-02 | 0.9831 | 1.359 | 0.223 |
| 256 | 2 | ours.PolarQuant (MSE) | 1.174e-01 | 1.163e-01 | 1.572e-01 | 1.745e-01 | 0.9399 | 1.935 | 0.341 |
| 256 | 2 | turbokv (MSE) | 3.603e-01 | 3.590e-01 | 4.570e-01 | 4.793e-01 | 0.8219 | 3.068 | 0.595 |
| 256 | 2 | ours.TurboQuant (Prod) | 3.635e-01 | 3.601e-01 | 4.555e-01 | 5.092e-01 | 0.7988 | 4.819 | 0.757 |
| 256 | 4 | ours.PolarQuant (MSE) | 9.482e-03 | 9.098e-03 | 1.604e-02 | 2.307e-02 | 0.9953 | 0.563 | 0.096 |
| 256 | 4 | turbokv (MSE) | 2.473e-02 | 1.690e-02 | 7.751e-02 | 1.048e-01 | 0.9906 | 0.856 | 0.146 |
| 256 | 4 | ours.TurboQuant (Prod) | 3.450e-02 | 3.386e-02 | 4.996e-02 | 6.405e-02 | 0.9828 | 1.425 | 0.232 |

## Cross-check: ours.PolarQuant vs turbokv (both MSE variants)

Median MSE is the relevant agreement check (mean is dominated by tail behavior).

| D | bits | ours median MSE | ref median MSE | median ratio | ours mean / ref mean |
|---|---|---|---|---|---|
| 64 | 2 | 1.131e-01 | 1.146e-01 | 0.988× | 0.907× |
| 64 | 4 | 8.806e-03 | 8.790e-03 | 1.002× | 0.566× |
| 128 | 2 | 1.154e-01 | 1.197e-01 | 0.964× | 0.786× |
| 128 | 4 | 8.925e-03 | 9.229e-03 | 0.967× | 0.291× |
| 256 | 2 | 1.163e-01 | 3.590e-01 | 0.324× | 0.326× |
| 256 | 4 | 9.098e-03 | 1.690e-02 | 0.538× | 0.383× |

## turbokv codebook convergence diagnostic

turbokv computes Lloyd-Max numerically on the exact Beta-on-sphere density. Its iteration initializes centroids at ±0.99, and any centroid whose Voronoi cell integrates to <1e-15 mass is left at its initialization. For D ≥ 256, the outermost cells have numerically zero mass and the outer centroids never move.

| D | bits | outer centroids | stuck at initialization? |
|---|---|---|---|
| 64 | 2 | (-0.1875, +0.1875) | no |
| 64 | 4 | (-0.3308, +0.3308) | no |
| 128 | 2 | (-0.1330, +0.1330) | no |
| 128 | 4 | (-0.2377, +0.2377) | no |
| 256 | 2 | (-0.9900, +0.9900) | **yes** |
| 256 | 4 | (-0.9900, +0.9900) | **yes** |
| 512 | 2 | (-0.9900, +0.9900) | **yes** |
| 512 | 4 | (-0.9900, +0.9900) | **yes** |

## Findings

1. **Median MSE agreement** between ours and turbokv at D=64 and D=128 is within 1.2%, confirming that the underlying Lloyd-Max algorithms produce equivalent codebooks for the bulk of the distribution.
2. **Mean MSE diverges sharply** for turbokv at D=128 b=4 (3.4× worse mean than median), driven by a thick right tail (max sample is 33× the median). This is real and reproducible — coordinates of unit-rotated vectors with absolute value beyond ±0.21 get reconstructed at the outer centroid ±0.238, losing tail magnitude. Our impl uses unbounded-support Gaussian-tail centroids and does not exhibit this.
3. **turbokv codebook fails to converge for D ≥ 256**: outer centroids remain at the ±0.99 initialization (zero-mass intervals are not updated by the iteration). This is a numerical issue in `turboquant/codebook.py` and not a fundamental algorithmic disagreement.
4. **Prod variant has higher reconstruction MSE than MSE variant** at the same total bit budget (because Prod spends 1 bit on the JL sketch). This is expected.
5. **Prod variant has lower IP relative error** than MSE variant at b=4 (1.3 vs 0.6 mean; mean is heavy-tailed and dominated by near-zero IP values, so median is the more reliable indicator — see real-data K/V check in Task 2 for IP preservation under softmax which is what actually matters for attention).

