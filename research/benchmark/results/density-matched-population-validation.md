# Matched S3 density and DOPE validation

100 matched S3 lineages; fixed four DOPE GPU research profiles versus default/native-selected density methods; descriptive validation only

Baselines use their frozen native likelihood objectives. Common retention never selects baseline configurations. Native KPI values are reported per method without cross-method ranking.

All100 dataset views match the same five training-derived numeric files. One fit seed and three sample seeds at n/4n; every failure and artifact charge remains visible. Comparisons below describe unconstrained quality. Release-safe L3/MFS-v2/PTF-v1 and superiority stay null.

| Method | Configuration | Size | CatBoost informative /100 | Median retention | Linear informative /100 | Median retention | MLP informative /100 | Median retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DOPE | features12_steps512 | 1 | 97 | 0.757065 | 92 | 0.982217 | 75 | 0.954734 |
| DOPE | features12_steps512 | 4 | 97 | 0.799674 | 92 | 0.984425 | 75 | 0.99886 |
| DOPE | features12_steps2048 | 1 | 97 | 0.885562 | 92 | 0.983611 | 75 | 0.971284 |
| DOPE | features12_steps2048 | 4 | 97 | 0.939876 | 92 | 0.994046 | 75 | 1.06112 |
| DOPE | features24_steps512 | 1 | 93 | 0.757065 | 88 | 0.982217 | 72 | 0.961807 |
| DOPE | features24_steps512 | 4 | 93 | 0.799674 | 88 | 0.986092 | 72 | 0.998482 |
| DOPE | features24_steps2048 | 1 | 93 | 0.882459 | 88 | 0.976904 | 72 | 0.969295 |
| DOPE | features24_steps2048 | 4 | 93 | 0.927634 | 88 | 0.989589 | 72 | 1.04829 |
| GaussianCopula | default | 1 | 99 | 0.548633 | 94 | 0.921043 | 76 | 0.772446 |
| GaussianCopula | default | 4 | 99 | 0.608878 | 94 | 0.95642 | 76 | 0.902056 |
| GaussianCopula | native_selected | 1 | 99 | 0.615605 | 94 | 0.961251 | 76 | 0.822257 |
| GaussianCopula | native_selected | 4 | 99 | 0.655681 | 94 | 0.987224 | 76 | 0.928192 |
| independent_marginals | default | 1 | 99 | -0.0456602 | 94 | -0.0492166 | 76 | -0.111285 |
| independent_marginals | default | 4 | 99 | -0.0192243 | 94 | 0.000511216 | 76 | -0.0345777 |
| independent_marginals | native_selected | 1 | 99 | -0.0444258 | 94 | -0.0500669 | 76 | -0.127883 |
| independent_marginals | native_selected | 4 | 99 | -0.0117284 | 94 | 0.00205646 | 76 | -0.0273173 |
| Chow-Liu | default | 1 | 99 | 0.574258 | 94 | 0.661219 | 76 | 0.572492 |
| Chow-Liu | default | 4 | 99 | 0.63278 | 94 | 0.726516 | 76 | 0.773033 |
| Chow-Liu | native_selected | 1 | 99 | 0.477656 | 94 | 0.636132 | 76 | 0.470152 |
| Chow-Liu | native_selected | 4 | 99 | 0.582318 | 94 | 0.65922 | 76 | 0.689101 |

Paired descriptive JSON uses common complete informative lineages and the median of dataset differences. It makes no significance, win or selected-family claim. Earlier DOPE architecture research spend and native density tuning costs remain separately reported.
