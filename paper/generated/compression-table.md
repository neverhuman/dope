| Rank (lift/KiB) | Generator | Worst-auditor retention (4n) | CatBoost | Linear | MLP | Median charged bytes | Retained lift per KiB |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **DOPE (ours)** | 0.901 [0.811, 0.966] | 0.913 | 0.991 | 1.143 | 1.9 KB | 0.487 |
| 2 | Gaussian copula | 0.750 [0.398, 0.898] | 0.764 | 0.991 | 0.981 | 5.5 KB | 0.140 |
| 3 | Chow-Liu tree | 0.573 [0.246, 0.654] | 0.605 | 0.672 | 0.682 | 29.9 KB | 0.020 |
| 4 | TVAE (20 lineages) | 0.655 [0.405, 0.784] | 0.663 | 0.765 | 0.812 | 308.6 KB | 2.2e-3 |
| 5 | ARF | 0.833 [0.692, 0.896] | 0.850 | 0.950 | 0.991 | 1.4 MB | 5.9e-4 |
| 6 | TabSyn | 0.911 [0.746, 0.963] | 0.944 | 0.959 | 1.015 | 42.3 MB | 2.2e-5 |
| 7 | Forest-Flow (5 lineages) | 0.985 [0.864, 1.001] | 1.021 | 0.985 | 1.080 | 325.8 MB | 3.1e-6 |
| — | CTGAN (20 lineages) | -0.152 [-1.113, 0.162] | -0.052 | -0.005 | 0.033 | 731.0 KB | no lift |
| — | Independent marginals | -0.078 [-0.169, -0.042] | -0.012 | 0.002 | -0.046 | 2.5 KB | no lift |
| — | Predictor-only control *(control, not a generator)* | 0.681 | 0.681 | 0.986 | 0.951 | none | — |
| — | Real rows resampled 4n *(control, not a generator)* | 0.997 | 1.013 | 0.999 | 1.174 | none | — |
