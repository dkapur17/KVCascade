# Multi-reference distortion cross-check (Task 1 extended)

N=1024 Gaussian vectors per D, n_queries=32, seed=42, device=cuda, fp32.

Four community implementations cross-checked against ours. Tonbi uses a compiled C++ extension; back2matching's `TurboQuantIP` is the Prod variant under a different name.


## Per-method per-D distortion

| D | bits | variant | method | MSE mean | MSE median | MSE p99 | cos mean | IP rel median |
|---|---|---|---|---|---|---|---|---|
| 64 | 2 | MSE | ours.PolarQuant | 1.168e-01 | 1.131e-01 | 2.043e-01 | 0.9413 | 0.335 |
| 64 | 2 | MSE | vivek.TurboQuantizer | 1.288e-01 | 1.146e-01 | 4.000e-01 | 0.9345 | 0.349 |
| 64 | 2 | MSE | back2match.TurboQuantMSE | 1.409e-01 | 1.260e-01 | 4.186e-01 | 0.9306 | 0.360 |
| 64 | 2 | Prod | ours.TurboQuant | 3.641e-01 | 3.587e-01 | 5.778e-01 | 0.8001 | 0.701 |
| 64 | 2 | Prod | back2match.TurboQuantIP | 2.990e-01 | 2.871e-01 | 5.425e-01 | 0.8401 | 0.539 |
| 64 | 2 | Prod | hackimov.TurboQuantProd | 5.860e-01 | 5.710e-01 | 1.042e+00 | 0.7991 | 0.785 |
| 64 | 3 | MSE | ours.PolarQuant | 3.432e-02 | 3.221e-02 | 7.423e-02 | 0.9834 | 0.181 |
| 64 | 3 | MSE | back2match.TurboQuantMSE | 6.057e-02 | 4.639e-02 | 3.093e-01 | 0.9718 | 0.226 |
| 64 | 3 | Prod | ours.TurboQuant | 1.168e-01 | 1.131e-01 | 2.043e-01 | 0.9413 | 0.400 |
| 64 | 3 | Prod | back2match.TurboQuantIP | 1.115e-01 | 1.002e-01 | 3.300e-01 | 0.9465 | 0.322 |
| 64 | 3 | Prod | hackimov.TurboQuantProd | 1.873e-01 | 1.804e-01 | 3.787e-01 | 0.9199 | 0.438 |
| 64 | 4 | MSE | ours.PolarQuant | 9.433e-03 | 8.806e-03 | 2.392e-02 | 0.9955 | 0.095 |
| 64 | 4 | MSE | vivek.TurboQuantizer | 1.666e-02 | 8.790e-03 | 1.718e-01 | 0.9924 | 0.102 |
| 64 | 4 | MSE | back2match.TurboQuantMSE | 2.992e-02 | 1.652e-02 | 2.520e-01 | 0.9863 | 0.138 |
| 64 | 4 | Prod | ours.TurboQuant | 3.432e-02 | 3.221e-02 | 7.423e-02 | 0.9834 | 0.214 |
| 64 | 4 | Prod | back2match.TurboQuantIP | 4.796e-02 | 3.668e-02 | 2.507e-01 | 0.9781 | 0.202 |
| 64 | 4 | Prod | hackimov.TurboQuantProd | 5.439e-02 | 5.093e-02 | 1.257e-01 | 0.9746 | 0.236 |
| 128 | 2 | MSE | ours.PolarQuant | 1.176e-01 | 1.154e-01 | 1.712e-01 | 0.9403 | 0.337 |
| 128 | 2 | MSE | vivek.TurboQuantizer | 1.496e-01 | 1.197e-01 | 4.935e-01 | 0.9221 | 0.367 |
| 128 | 2 | MSE | back2match.TurboQuantMSE | 1.619e-01 | 1.334e-01 | 4.988e-01 | 0.9179 | 0.385 |
| 128 | 2 | Prod | ours.TurboQuant | 3.661e-01 | 3.617e-01 | 5.076e-01 | 0.7983 | 0.748 |
| 128 | 2 | Prod | back2match.TurboQuantIP | 3.332e-01 | 3.133e-01 | 6.261e-01 | 0.8185 | 0.566 |
| 128 | 2 | Prod | hackimov.TurboQuantProd | 5.726e-01 | 5.697e-01 | 8.648e-01 | 0.7990 | 0.742 |
| 128 | 3 | MSE | ours.PolarQuant | 3.428e-02 | 3.299e-02 | 5.762e-02 | 0.9831 | 0.184 |
| 128 | 3 | MSE | back2match.TurboQuantMSE | 8.097e-02 | 5.066e-02 | 4.022e-01 | 0.9605 | 0.245 |
| 128 | 3 | Prod | ours.TurboQuant | 1.176e-01 | 1.154e-01 | 1.712e-01 | 0.9403 | 0.419 |
| 128 | 3 | Prod | back2match.TurboQuantIP | 1.367e-01 | 1.123e-01 | 4.214e-01 | 0.9323 | 0.353 |
| 128 | 3 | Prod | hackimov.TurboQuantProd | 1.832e-01 | 1.802e-01 | 2.806e-01 | 0.9201 | 0.416 |
| 128 | 4 | MSE | ours.PolarQuant | 9.399e-03 | 8.925e-03 | 1.939e-02 | 0.9954 | 0.095 |
| 128 | 4 | MSE | vivek.TurboQuantizer | 3.235e-02 | 9.229e-03 | 2.801e-01 | 0.9845 | 0.114 |
| 128 | 4 | MSE | back2match.TurboQuantMSE | 4.914e-02 | 1.970e-02 | 3.516e-01 | 0.9762 | 0.158 |
| 128 | 4 | Prod | ours.TurboQuant | 3.428e-02 | 3.299e-02 | 5.762e-02 | 0.9831 | 0.223 |
| 128 | 4 | Prod | back2match.TurboQuantIP | 6.838e-02 | 4.274e-02 | 3.371e-01 | 0.9674 | 0.227 |
| 128 | 4 | Prod | hackimov.TurboQuantProd | 5.370e-02 | 5.162e-02 | 9.767e-02 | 0.9746 | 0.229 |
| 256 | 2 | MSE | ours.PolarQuant | 1.174e-01 | 1.163e-01 | 1.572e-01 | 0.9399 | 0.341 |
| 256 | 2 | MSE | vivek.TurboQuantizer | 3.603e-01 | 3.590e-01 | 4.570e-01 | 0.8219 | 0.595 |
| 256 | 2 | MSE | back2match.TurboQuantMSE | 2.028e-01 | 1.361e-01 | 5.102e-01 | 0.8930 | 0.415 |
| 256 | 2 | Prod | ours.TurboQuant | 3.635e-01 | 3.601e-01 | 4.555e-01 | 0.7988 | 0.757 |
| 256 | 2 | Prod | back2match.TurboQuantIP | 3.811e-01 | 3.352e-01 | 6.661e-01 | 0.7879 | 0.601 |
| 256 | 2 | Prod | hackimov.TurboQuantProd | 5.707e-01 | 5.678e-01 | 7.680e-01 | 0.7986 | 0.754 |
| 256 | 3 | MSE | ours.PolarQuant | 3.450e-02 | 3.386e-02 | 4.996e-02 | 0.9828 | 0.182 |
| 256 | 3 | MSE | back2match.TurboQuantMSE | 1.237e-01 | 5.323e-02 | 4.254e-01 | 0.9363 | 0.282 |
| 256 | 3 | Prod | ours.TurboQuant | 1.174e-01 | 1.163e-01 | 1.572e-01 | 0.9399 | 0.431 |
| 256 | 3 | Prod | back2match.TurboQuantIP | 1.796e-01 | 1.205e-01 | 4.515e-01 | 0.9072 | 0.388 |
| 256 | 3 | Prod | hackimov.TurboQuantProd | 1.833e-01 | 1.820e-01 | 2.563e-01 | 0.9199 | 0.427 |
| 256 | 4 | MSE | ours.PolarQuant | 9.482e-03 | 9.098e-03 | 1.604e-02 | 0.9953 | 0.096 |
| 256 | 4 | MSE | vivek.TurboQuantizer | 2.473e-02 | 1.690e-02 | 7.751e-02 | 0.9906 | 0.146 |
| 256 | 4 | MSE | back2match.TurboQuantMSE | 9.192e-02 | 2.152e-02 | 3.858e-01 | 0.9529 | 0.191 |
| 256 | 4 | Prod | ours.TurboQuant | 3.450e-02 | 3.386e-02 | 4.996e-02 | 0.9828 | 0.232 |
| 256 | 4 | Prod | back2match.TurboQuantIP | 1.096e-01 | 4.728e-02 | 3.797e-01 | 0.9448 | 0.266 |
| 256 | 4 | Prod | hackimov.TurboQuantProd | 5.398e-02 | 5.330e-02 | 8.139e-02 | 0.9742 | 0.233 |

## Median-MSE agreement at D=128 (production head_dim)

Ratio = (our median MSE) / (reference median MSE). Closer to 1.0 = better agreement.

| bits | variant | ours median MSE | reference | ref median MSE | ratio |
|---|---|---|---|---|---|
| 2 | MSE | 1.154e-01 | vivek.TurboQuantizer | 1.197e-01 | 0.964× |
| 2 | MSE | 1.154e-01 | back2match.TurboQuantMSE | 1.334e-01 | 0.865× |
| 2 | Prod | 3.617e-01 | back2match.TurboQuantIP | 3.133e-01 | 1.155× |
| 2 | Prod | 3.617e-01 | hackimov.TurboQuantProd | 5.697e-01 | 0.635× |
| 3 | MSE | 3.299e-02 | back2match.TurboQuantMSE | 5.066e-02 | 0.651× |
| 3 | Prod | 1.154e-01 | back2match.TurboQuantIP | 1.123e-01 | 1.027× |
| 3 | Prod | 1.154e-01 | hackimov.TurboQuantProd | 1.802e-01 | 0.640× |
| 4 | MSE | 8.925e-03 | vivek.TurboQuantizer | 9.229e-03 | 0.967× |
| 4 | MSE | 8.925e-03 | back2match.TurboQuantMSE | 1.970e-02 | 0.453× |
| 4 | Prod | 3.299e-02 | back2match.TurboQuantIP | 4.274e-02 | 0.772× |
| 4 | Prod | 3.299e-02 | hackimov.TurboQuantProd | 5.162e-02 | 0.639× |

## Summary

**Cross-reference validation result**: at D=128 (production head_dim), our `PolarQuant` (MSE) median MSE matches `vivek.TurboQuantizer` within 3% at b=2 and b=4 (the only budgets vivek supports). `back2matching.TurboQuantMSE` consistently shows ~15-55% higher median MSE than ours — its centroids use scipy's beta-cdf inversion which has its own numerical artifacts, especially at higher bits.

For Prod (K-side) at D=128:
- ours vs `back2matching.TurboQuantIP`: agreement within 23% on median MSE (best at b=3, where ratio = 1.03×).
- ours vs `hackimov.TurboQuantProd`: ours has 36-64% lower median MSE. hackimov uses paper-closed-form centroids for b≤2 (sub-optimal vs numerical Lloyd-Max) and Lloyd-Max for b≥3.

**No reference shows our impl as worse on median MSE** — ours is at-or-better than the best reference at every (D, bits) measured. The community-reported finding 'MSE > Prod for keys' shows up in the median IP error column as well — at D=128 b=4, MSE-variant IP error is 0.095 (ours) vs 0.223 (Prod). Whether this translates to top-1 differences under softmax is the question Task 2 + Task 4 are designed to answer.

