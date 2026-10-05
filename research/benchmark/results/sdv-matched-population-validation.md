# Matched S3 neural and DOPE validation

100 S3 lineages fully accounted; four fixed DOPE profiles versus original author-default/native-selected CTGAN and TVAE. Only existing samples are evaluated. Descriptive validation; one fit seed.

CTGAN/TVAE maximize their original native SDMetrics regression efficacy KPI. Common retention never selects their configurations; native KPI values are not ranked across methods.

All 100 lineages remain in the coverage denominator. SDV has 486 measured cells and 1,914 sample-unavailable cells: 81/400 configuration bindings have complete existing samples. Native research ended with 154 successful new trials, 524 deadline-unstarted trials, one deadline-truncated trial and 25 infrastructure interruptions. Scheduling cutoffs do not establish method failures.

Each paired number uses lineages where all three sample seeds are informative for both methods at the stated size. Availability can bias this subset. DOPE profiles are fixed research configurations; no global family or production winner is selected.

| DOPE profile | Native baseline | Size | CatBoost paired lineages | DOPE median | Baseline median | Median paired difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| features12_steps512 | CTGAN | 1n | 21 | 0.780082 | -0.0909244 | 0.721212 |
| features12_steps512 | CTGAN | 4n | 21 | 0.799674 | -0.0812588 | 0.713864 |
| features12_steps512 | TVAE | 1n | 21 | 0.780082 | 0.597683 | 0.130762 |
| features12_steps512 | TVAE | 4n | 21 | 0.799674 | 0.669033 | 0.0820766 |
| features12_steps2048 | CTGAN | 1n | 21 | 0.897392 | -0.0909244 | 0.881629 |
| features12_steps2048 | CTGAN | 4n | 21 | 0.953366 | -0.0812588 | 0.880662 |
| features12_steps2048 | TVAE | 1n | 21 | 0.897392 | 0.597683 | 0.238131 |
| features12_steps2048 | TVAE | 4n | 21 | 0.953366 | 0.669033 | 0.23611 |
| features24_steps512 | CTGAN | 1n | 21 | 0.780082 | -0.0909244 | 0.721212 |
| features24_steps512 | CTGAN | 4n | 21 | 0.799674 | -0.0812588 | 0.713864 |
| features24_steps512 | TVAE | 1n | 21 | 0.780082 | 0.597683 | 0.130762 |
| features24_steps512 | TVAE | 4n | 21 | 0.799674 | 0.669033 | 0.0820766 |
| features24_steps2048 | CTGAN | 1n | 21 | 0.897392 | -0.0909244 | 0.789025 |
| features24_steps2048 | CTGAN | 4n | 21 | 0.926131 | -0.0812588 | 0.874566 |
| features24_steps2048 | TVAE | 1n | 21 | 0.897392 | 0.597683 | 0.21293 |
| features24_steps2048 | TVAE | 4n | 21 | 0.926131 | 0.669033 | 0.204814 |

JSON includes default and native-selected results, all three auditors, per-cell utility/privacy controls, native KPIs, charged model+projection bytes, costs and unavailable reasons. These are unconstrained-quality validation comparisons. Official tests remain sealed; MFS-v2/PTF-v1/release-safe L3/superiority stay null.
