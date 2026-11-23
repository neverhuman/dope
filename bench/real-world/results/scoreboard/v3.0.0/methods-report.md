# Third-party baselines on the DEV report schedule (dope_kpi receipts)

Run dir `/home/ubuntu/dope-data/kpi-runs/v3-dev-report` (xbabe0), schedule `schedule-report-dev.jsonl` sha256 `14238f1d5b9e`, realref `7d377b8ab47f`, 108 cells (64 headline). Library defaults, CPU, 1 thread, **not byte-capped** (score `--budget-bytes 1000000000`). DEV estimate, not a sealed test.

| # | arm | class | TRG 4x [95% CI] | TRG 1x | regr / bin / multi (4x) | bytes median / max (64 headline files) | realism / privacy cov | exact-copy | failed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | copy_a | control | -0.039 [-0.069, -0.014] | -0.039 | -0.017 / -0.056 / -0.045 | - | n/e / n/e | n/e | 0/44 |
| 2 | twin_c | control | -0.060 [-0.060, -0.060] | -0.060 | -0.074 / - / -0.046 | - | n/e / n/e | n/e | 0/8 |
| 3 | bootstrap_a | control | -0.088 [-0.147, -0.036] | -0.214 | -0.085 / -0.039 / -0.139 | - | n/e / n/e | n/e | 0/44 |
| 4 | dope-v3b-auto-v8 | source_free | -0.121 [-0.189, -0.049] | -0.309 | -0.137 / -0.136 / -0.089 | 1,787 / 2,048 | 0.194 / 0.688 | 0.000 | 0/44 |
| 5 | dope-2.0.0-auto | source_free | -0.181 [-0.280, -0.085] | -0.333 | -0.118 / -0.278 / -0.146 | 1,818 / 2,048 | 0.306 / 0.719 | 0.000 | 0/44 |
| 6 | **arfpy-arf** | source_free | -0.228 [-0.416, -0.100] | -0.370 | -0.071 / -0.412 / -0.203 | 2,937,782 / 24,068,428 | 0.278 / 0.031 | 0.000 | 0/44 |
| 7 | dope-1.3.0-small | source_free | -0.266 [-0.399, -0.133] | -0.468 | -0.249 / -0.438 / -0.109 | 1,508 / 2,043 | 0.306 / 0.507 | 0.000 | 7/44 |
| 8 | dope-2.0.0-small | source_free | -0.267 [-0.402, -0.133] | -0.470 | -0.250 / -0.438 / -0.112 | 1,508 / 2,043 | 0.306 / 0.507 | 0.000 | 7/44 |
| 9 | dse8-p0-v3b | source_free | -0.342 [-0.652, -0.103] | -0.452 | -0.094 / -0.276 / -0.655 | 868 / 2,048 | 0.194 / 0.753 | 0.000 | 0/44 |
| 10 | pr390-v7-step0 | source_free | -0.432 [-0.585, -0.273] | -0.395 | -0.102 / -0.545 / -0.648 | 782 / 2,026 | 0.194 / 0.774 | 0.000 | 0/44 |
| 11 | **sdv-gaussian-copula** | source_free | -0.598 [-0.721, -0.482] | -0.828 | -0.168 / -0.612 / -1.015 | 140,676 / 1,145,929 | 0.111 / 0.601 | 0.005 | 0/44 |
| 12 | **sdv-tvae** | source_free | -1.039 [-1.262, -0.822] | -1.161 | -0.700 / -1.432 / -0.986 | 475,126 / 3,340,780 | 0.000 / 0.326 | 0.000 | 5/44 |
| 13 | label_shuffle | control | -1.164 [-1.269, -1.063] | -1.164 | -1.235 / -1.004 / -1.251 | - | n/e / n/e | n/e | 0/44 |
| 14 | indep_marginals | control | -1.222 [-1.387, -1.095] | -1.222 | -1.135 / -1.381 / -1.149 | - | n/e / n/e | n/e | 0/44 |
| 15 | **sdv-ctgan** | source_free | -1.692 [-1.827, -1.552] | -1.694 | -1.502 / -1.888 / -1.685 | 1,053,958 / 5,404,151 | 0.000 / 0.743 | 0.006 | 0/44 |

Coverage = share of headline cells passing the dope-core realism / empirical-privacy gate (`dope_kpi guards`; n/e = control, not evaluated); exact-copy = mean exact-copy excess at 4x (guard bound 0.005). Failed = informative headline cells at the -1 floor (failed, single-class or over budget).

## Compare vs dope-v3b-auto-v8 (`compare-methods.json`, Holm m=4)

| candidate | delta [95% CI] | p_holm | outcome | failed guards |
|---|---|---|---|---|
| sdv-gaussian-copula | -0.478 [-0.626, -0.337] | 1.0000 | REJECT | task_ci_low, mlp_delta, realism_coverage_delta, privacy_coverage_delta, bytes |
| sdv-ctgan | -1.571 [-1.736, -1.418] | 1.0000 | REJECT | task_ci_low, mlp_delta, realism_coverage_delta, exact_copy_excess, bytes |
| sdv-tvae | -0.919 [-1.123, -0.719] | 1.0000 | REJECT | task_ci_low, mlp_delta, realism_coverage_delta, privacy_coverage_delta, bytes |
| arfpy-arf | -0.108 [-0.295, +0.035] | 1.0000 | HOLD | task_ci_low, privacy_coverage_delta, bytes |

## Runs

| arm | version | cells ok / failed (reasons) | score failures | CPU s (fit + sample) | wall (emit, 108 cells) | packet bytes median / max (108 cells) |
|---|---|---|---|---|---|---|
| sdv-gaussian-copula | sdv 2.2.0 | 108 / 0 (none) | none | 1,648 (743 + 905) | 0h18m emit (0h03m gate-held) | 131,539 / 2,103,746 |
| sdv-ctgan | sdv 2.2.0 | 108 / 0 (none) | none | 7,281 (6,373 + 908) | 0h39m emit (0h16m gate-held) | 1,018,144 / 7,195,115 |
| sdv-tvae | sdv 2.2.0 | 108 / 0 (none) | single_class_target 120 | 3,292 (2,362 + 931) | 0h36m emit (0h16m gate-held) | 447,998 / 4,892,096 |
| arfpy-arf | arfpy 0.1.1 | 108 / 0 (none) | none | 1,519 (1,117 + 402) | 0h23m emit (0h13m gate-held) | 1,358,942 / 24,068,428 |

Versions: sdv 2.2.0, ctgan 0.13.0, copulas 0.15.0, rdt 2.1.0, torch 2.10.0, numpy 1.26.4, pandas 2.1.4+dfsg, scikit-learn 1.7.2, scipy 1.16.3; Python 3.12.3. Wheel sha256s, recipes and arm JSON: `MANIFEST.json`.
Wall = job elapsed on shared xbabe0 (CPUs 104-119, nice 19, SCHED_IDLE, budget guard 0.60), including time frozen by the JSR timing gate; CPU s is the compute actually spent.

## Notes

- Sanity: `dope_kpi primary` re-run on xbabe0 over the 11 committed arms is byte-identical to the committed `primary.json` (sha256 55974867...); the 11 arms' entries in `primary-methods.json` equal it field for field.
- artifact_class `source_free` for all four: audited on the smoke cells (no reachable container with the fit row count, no stored copy of a fit column); arfpy's packet drops `x_real` (the fit rows) and `clf`, which forge() never reads (stripped object forges identical rows).
- sdv-tvae: 20 cells (12 binary + 7 multiclass headline, 1 small_n) sampled a single class (score `single_class_target`, -1 by protocol); defaults give one optimizer step per epoch on tables under 500 rows.
- sdv-ctgan sits below `indep_marginals`: auditors fitted on its rows do worse than the null model on real test rows. A toy check (y = 3*x0 + noise, 350 rows) keeps corr(x0, y) 0.99 with GaussianCopula but 0.11 with CTGAN, so the wrapper is not the cause.
- arfpy-arf HOLD = delta CI includes 0 but no promotion condition passes; it is not shown better than DOPE.
- SDV is Business Source License 1.1 (research/benchmark use); arfpy is MIT. Hyperparameters were never tuned.
