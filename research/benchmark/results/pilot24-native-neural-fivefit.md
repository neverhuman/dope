# DOPE, CTGAN and TVAE: five-fit matched pilot validation

Adult and California use the same grouped training-derived fit/validation inputs for all methods. Official tests remain sealed. CTGAN and TVAE retain their author defaults and configurations previously selected only by SDMetrics ML efficacy (binary F1 or regression R²). The fixed replication adds no tuning. Native KPI values are reported separately and are never ranked against each other.

CTGAN/TVAE citation key: `xu2019modeling` from the corrected local bibliography. Exact author-library sources, licenses, native metric implementation and configuration grids are recorded in [SDV_METHOD_AUDIT.md](../SDV_METHOD_AUDIT.md).

Five fit seeds (11, 23, 37, 53, 71), three sample seeds (101, 211, 307), and n/2n/4n/8n generation sizes. Shared utility is measured at n and 4n by CatBoost, linear/logistic and MLP auditors. Fit23 reuses the verified prior fixed fit and all 60 successful size/sample receipts; its 12 unavailable default samples remain explicit. The complete comparator schedule accounts for all 360 sample identities. Default and tuned configurations that are identical share one physical fit.

## Common validation utility

Each fit value is the median of three informative sample-seed retentions; each group value is the median of all five complete fit values. Incomplete groups have a null complete estimate. Failures never become a DOPE win.

| Dataset | Method | Configuration | Size | Complete fits | CatBoost | Linear/logistic | MLP |
|---|---|---|---:|---:|---:|---:|---:|
| Adult | CTGAN | default | 1n | 0/5 | unavailable | unavailable | unavailable |
| Adult | CTGAN | default | 4n | 0/5 | unavailable | unavailable | unavailable |
| Adult | CTGAN | tuned | 1n | 5/5 | -0.1704 | -0.2331 | -0.2303 |
| Adult | CTGAN | tuned | 4n | 5/5 | -0.1711 | -0.2324 | -0.2585 |
| Adult | DOPE | q8_selected | 1n | 5/5 | 0.8232 | 0.9008 | 0.8926 |
| Adult | DOPE | q8_selected | 4n | 5/5 | 0.8289 | 0.9186 | 0.9155 |
| Adult | TVAE | default_and_tuned | 1n | 5/5 | 0.8180 | 0.8123 | 0.5443 |
| Adult | TVAE | default_and_tuned | 4n | 5/5 | 0.8128 | 0.7914 | 0.4934 |
| California | CTGAN | default_and_tuned | 1n | 5/5 | 0.5790 | 0.6354 | 0.6232 |
| California | CTGAN | default_and_tuned | 4n | 5/5 | 0.5811 | 0.6304 | 0.6786 |
| California | DOPE | q8_selected | 1n | 5/5 | 0.7032 | 0.9674 | 0.8619 |
| California | DOPE | q8_selected | 4n | 5/5 | 0.6991 | 0.9665 | 0.8631 |
| California | TVAE | default | 1n | 5/5 | 0.8750 | 0.8330 | 0.8671 |
| California | TVAE | default | 4n | 5/5 | 0.8808 | 0.8298 | 0.8994 |
| California | TVAE | tuned | 1n | 5/5 | 0.8929 | 0.8797 | 0.8859 |
| California | TVAE | tuned | 4n | 5/5 | 0.8964 | 0.8779 | 0.9432 |

## Per-fit CatBoost outcomes

| Dataset | Method | Configuration | Fit seed | n | 4n | Charged bytes |
|---|---|---|---:|---:|---:|---:|
| Adult | CTGAN | default | 11 | unavailable | unavailable | unavailable |
| Adult | CTGAN | default | 23 | unavailable | unavailable | unavailable |
| Adult | CTGAN | default | 37 | unavailable | unavailable | unavailable |
| Adult | CTGAN | default | 53 | unavailable | unavailable | unavailable |
| Adult | CTGAN | default | 71 | unavailable | unavailable | unavailable |
| Adult | CTGAN | tuned | 11 | -0.1285 | -0.1309 | 3032318 |
| Adult | CTGAN | tuned | 23 | 0.0139 | 0.0171 | 3032190 |
| Adult | CTGAN | tuned | 37 | -0.3702 | -0.3627 | 3032190 |
| Adult | CTGAN | tuned | 53 | -0.1704 | -0.1711 | 3032254 |
| Adult | CTGAN | tuned | 71 | -0.3209 | -0.3229 | 3032254 |
| Adult | DOPE | q8_selected | 11 | 0.8232 | 0.8303 | 6341 |
| Adult | DOPE | q8_selected | 23 | 0.8275 | 0.8293 | 6337 |
| Adult | DOPE | q8_selected | 37 | 0.8184 | 0.8280 | 6338 |
| Adult | DOPE | q8_selected | 53 | 0.8176 | 0.8287 | 6335 |
| Adult | DOPE | q8_selected | 71 | 0.8235 | 0.8289 | 6342 |
| Adult | TVAE | default_and_tuned | 11 | 0.8110 | 0.8155 | 1665620 |
| Adult | TVAE | default_and_tuned | 23 | 0.7951 | 0.7788 | 1665556 |
| Adult | TVAE | default_and_tuned | 37 | 0.8180 | 0.8079 | 1665620 |
| Adult | TVAE | default_and_tuned | 53 | 0.8794 | 0.8574 | 1665620 |
| Adult | TVAE | default_and_tuned | 71 | 0.8239 | 0.8128 | 1665620 |
| California | CTGAN | default_and_tuned | 11 | 0.5755 | 0.5811 | 941787 |
| California | CTGAN | default_and_tuned | 23 | 0.6652 | 0.6748 | 941787 |
| California | CTGAN | default_and_tuned | 37 | 0.5790 | 0.5806 | 941787 |
| California | CTGAN | default_and_tuned | 53 | 0.5949 | 0.5941 | 941787 |
| California | CTGAN | default_and_tuned | 71 | 0.5197 | 0.5240 | 941787 |
| California | DOPE | q8_selected | 11 | 0.7088 | 0.7044 | 2524 |
| California | DOPE | q8_selected | 23 | 0.7008 | 0.6948 | 2525 |
| California | DOPE | q8_selected | 37 | 0.6982 | 0.6903 | 2525 |
| California | DOPE | q8_selected | 53 | 0.7032 | 0.6991 | 2524 |
| California | DOPE | q8_selected | 71 | 0.7095 | 0.7030 | 2524 |
| California | TVAE | default | 11 | 0.8720 | 0.8808 | 373361 |
| California | TVAE | default | 23 | 0.8750 | 0.8756 | 373361 |
| California | TVAE | default | 37 | 0.8793 | 0.8856 | 373425 |
| California | TVAE | default | 53 | 0.8737 | 0.8737 | 373361 |
| California | TVAE | default | 71 | 0.8833 | 0.8858 | 373361 |
| California | TVAE | tuned | 11 | 0.9006 | 0.9019 | 340593 |
| California | TVAE | tuned | 23 | 0.9154 | 0.9163 | 340593 |
| California | TVAE | tuned | 37 | 0.8929 | 0.8964 | 340657 |
| California | TVAE | tuned | 53 | 0.8915 | 0.8904 | 340593 |
| California | TVAE | tuned | 71 | 0.8899 | 0.8953 | 340593 |

## Native validation objectives (separate values)

| Dataset | Method | Native objective | Default KPI | Selected KPI | Trial | Tuning attempts |
|---|---|---|---:|---:|---:|---:|
| Adult | CTGAN | sdmetrics_mean_binary_f1 | unavailable | 0.3376 | 4 | 8 |
| Adult | TVAE | sdmetrics_mean_binary_f1 | 0.6532 | 0.6532 | 0 | 4 |
| California | CTGAN | sdmetrics_mean_regression_r2 | 0.4138 | 0.4138 | 0 | 4 |
| California | TVAE | sdmetrics_mean_regression_r2 | 0.5642 | 0.5854 | 3 | 4 |

## Failures and cost

The DOPE replication also retains three preflight launch failures before a GPU fit or metric started: a missing research package path, a receipt glob that matched host inventory JSON, and a relative invocation path under scratch. Their phases, source identities and immutable ledger hashes remain in `dope_preflight`; no extra fit or successful metric is inferred.

Comparator fixed-fit attempt statuses: {"ok": 25, "timeout": 5}. Complete 360-cell comparator sample statuses: {"fit_unavailable": 60, "ok": 300}. New streaming sample statuses: {"fit_unavailable": 48, "ok": 240}. Every failed attempt and its elapsed time remain in the JSON. No source-unavailable status or utility value is inferred for executable failures.

Summed comparator fixed-fit time: 9060.89s. Summed complete attempt time: 9726.29s. Prior native tuning is separately reported at 8250.84s. DOPE fixed q8 replication costs 116.88s; 55 prior pilot research fits add 708.76s separately. That pilot subtotal includes News follow-ups and excludes earlier S3 discovery/confirmation and CPU development: total campaign R&D is not yet completely accounted and is not claimed equal between methods. Fixed-fit costs include the prior seed23 once, with shared configurations counted once. Sample elapsed time includes auditor evaluation/replays and is reported separately.

| Dataset | Method | Configuration | Fit seed | Host | Fit status | Fit seconds | Peak GPU MiB |
|---|---|---|---:|---|---|---:|---:|
| Adult | CTGAN | default | 11 | xbabe1 | timeout | 600.67 | 664 |
| Adult | CTGAN | tuned | 11 | xbabe1 | ok | 156.12 | 664 |
| Adult | TVAE | default_and_tuned | 11 | xbabe1 | ok | 377.00 | 630 |
| Adult | CTGAN | default | 37 | xbabe1 | timeout | 600.69 | 664 |
| Adult | CTGAN | tuned | 37 | xbabe1 | ok | 155.17 | 664 |
| Adult | TVAE | default_and_tuned | 37 | xbabe1 | ok | 391.11 | 630 |
| Adult | CTGAN | default | 53 | xbabe1 | timeout | 600.76 | 664 |
| Adult | CTGAN | tuned | 53 | xbabe1 | ok | 157.95 | 664 |
| Adult | TVAE | default_and_tuned | 53 | xbabe1 | ok | 391.64 | 630 |
| Adult | CTGAN | default | 71 | xbabe1 | timeout | 600.33 | 664 |
| Adult | CTGAN | tuned | 71 | xbabe1 | ok | 157.18 | 664 |
| Adult | TVAE | default_and_tuned | 71 | xbabe1 | ok | 390.53 | 630 |
| Adult | CTGAN | default | 23 | xbabe1 | timeout | 600.50 | 664 |
| Adult | CTGAN | tuned | 23 | xbabe1 | ok | 153.04 | 664 |
| Adult | TVAE | default_and_tuned | 23 | xbabe1 | ok | 375.15 | 630 |
| California | CTGAN | default_and_tuned | 11 | xbabe3 | ok | 213.30 | 476 |
| California | TVAE | default | 11 | xbabe3 | ok | 144.04 | 466 |
| California | TVAE | tuned | 11 | xbabe3 | ok | 312.62 | 462 |
| California | CTGAN | default_and_tuned | 37 | xbabe3 | ok | 211.30 | 476 |
| California | TVAE | default | 37 | xbabe3 | ok | 142.46 | 466 |
| California | TVAE | tuned | 37 | xbabe3 | ok | 311.89 | 462 |
| California | CTGAN | default_and_tuned | 53 | xbabe3 | ok | 211.39 | 476 |
| California | TVAE | default | 53 | xbabe3 | ok | 122.09 | 466 |
| California | TVAE | tuned | 53 | xbabe3 | ok | 321.10 | 462 |
| California | CTGAN | default_and_tuned | 71 | xbabe3 | ok | 202.07 | 476 |
| California | TVAE | default | 71 | xbabe3 | ok | 130.25 | 466 |
| California | TVAE | tuned | 71 | xbabe3 | ok | 307.68 | 462 |
| California | CTGAN | default_and_tuned | 23 | xbabe3 | ok | 208.68 | 476 |
| California | TVAE | default | 23 | xbabe3 | ok | 144.04 | 466 |
| California | TVAE | tuned | 23 | xbabe3 | ok | 370.12 | 462 |

## Copy controls and artifact limits

| Dataset | Method | Configuration | Size | Exact-copy cells | Synthetic near-match rate | Real control near-match rate |
|---|---|---|---:|---:|---:|---:|
| Adult | CTGAN | tuned | 1n | 0/15 | 0.000000 | 0.026740 |
| Adult | CTGAN | tuned | 4n | 0/15 | 0.000000 | 0.026740 |
| Adult | DOPE | q8_selected | 1n | 0/15 | 0.000038 | 0.026740 |
| Adult | DOPE | q8_selected | 4n | 0/15 | 0.000067 | 0.026740 |
| Adult | TVAE | default_and_tuned | 1n | 0/15 | 0.019286 | 0.026740 |
| Adult | TVAE | default_and_tuned | 4n | 0/15 | 0.019670 | 0.026740 |
| California | CTGAN | default_and_tuned | 1n | 0/15 | 0.000000 | 0.000242 |
| California | CTGAN | default_and_tuned | 4n | 0/15 | 0.000000 | 0.000242 |
| California | DOPE | q8_selected | 1n | 0/15 | 0.000000 | 0.000242 |
| California | DOPE | q8_selected | 4n | 0/15 | 0.000000 | 0.000242 |
| California | TVAE | default | 1n | 0/15 | 0.000000 | 0.000242 |
| California | TVAE | default | 4n | 0/15 | 0.000000 | 0.000242 |
| California | TVAE | tuned | 1n | 0/15 | 0.000000 | 0.000242 |
| California | TVAE | tuned | 4n | 0/15 | 0.000000 | 0.000242 |

Near counts use the fixed normalized RMS1e-3 screen and real-vs-real controls. These screens are not a complete privacy attack profile. All learned model, preprocessor, configuration and projection bytes are charged. L3 byte eligibility is reported separately from release safety; baseline artifacts above10,240 bytes remain visible as unconstrained quality.

MFS-v2, PTF-v1, release-safe status and production certification remain null/unestablished. These two validation datasets do not establish the preregistered public-core paired claim or product coverage. All per-cell auditor losses, null losses, copy controls, C2ST/fidelity screens and immutable evidence hashes are retained in JSON.
