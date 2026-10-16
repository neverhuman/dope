# DOPE q8: five fit seeds on Adult and California

Fixed q8 configuration, five fits per dataset, three sample seeds, and n/2n/4n/8n output sizes. The shared CatBoost auditor was measured and replayed at n and 4n on training-derived validation. Official tests stayed sealed.

| Dataset | Fit seed | n median retention | 4n median retention | Charged bytes |
|---|---:|---:|---:|---:|
| Adult | 11 | 0.8232 | 0.8303 | 6,341 |
| Adult | 23 | 0.8275 | 0.8293 | 6,337 |
| Adult | 37 | 0.8184 | 0.8280 | 6,338 |
| Adult | 53 | 0.8176 | 0.8287 | 6,335 |
| Adult | 71 | 0.8235 | 0.8289 | 6,342 |
| California | 11 | 0.7088 | 0.7044 | 2,524 |
| California | 23 | 0.7008 | 0.6948 | 2,525 |
| California | 37 | 0.6982 | 0.6903 | 2,525 |
| California | 53 | 0.7032 | 0.6991 | 2,524 |
| California | 71 | 0.7095 | 0.7030 | 2,524 |

Median across fit-seed medians:

| Dataset | n | 4n |
|---|---:|---:|
| Adult | 0.8232 | 0.8289 |
| California | 0.7032 | 0.6991 |

## Fit costs and launch accounting

Ten successful fits consumed 116.88 seconds of summed fit elapsed time under a frozen 6,000-second ceiling. Every fit stayed below 600 seconds and 16 GiB. These costs exclude earlier architecture research and metric evaluation.

| Dataset | Fit seed | Host | Fit seconds | Peak GPU MiB | Estimated joules |
|---|---:|---|---:|---:|---:|
| Adult | 11 | xbabe1 | 18.43 | 616 | 927.96 |
| Adult | 23 | xbabe3 | 18.48 | 460 | 1179.44 |
| Adult | 37 | xbabe1 | 18.43 | 616 | 924.52 |
| Adult | 53 | xbabe3 | 18.48 | 460 | 1179.95 |
| Adult | 71 | xbabe1 | 18.44 | 616 | 920.06 |
| California | 11 | xbabe3 | 4.10 | 1 | 282.75 |
| California | 23 | xbabe1 | 6.15 | 598 | 326.41 |
| California | 37 | xbabe3 | 4.11 | 1 | 282.60 |
| California | 53 | xbabe1 | 4.10 | 9 | 217.07 |
| California | 71 | xbabe3 | 6.16 | 442 | 411.90 |

Three preflight launch failures started no GPU fit or metric evaluation. Their source and failure receipts remain linked in the JSON.

| Phase | Error | Cause |
|---|---|---|
| v1_freeze_import | ModuleNotFoundError | research package path not set |
| v1_freeze_job_reconciliation | KeyError | attempt glob also matched host inventory JSON |
| v2_run_invocation | FileNotFoundError | relative script path resolved under scratch working directory |

All 120 sample cells matched repeat hashes; all 60 n/4n metric payloads replayed exactly. A separate frozen stratified audit replayed 16 cells exactly, excluding only metric elapsed time.

Near-match rates use the fixed 1e-3 normalized RMS pilot threshold. The real-vs-real control uses held-out validation rows against training rows; rates are median per cell.

| Dataset | Size | Synthetic near-match rate | Real control near-match rate |
|---|---:|---:|---:|
| Adult | 1n | 0.000038 | 0.026740 |
| Adult | 4n | 0.000067 | 0.026740 |
| California | 1n | 0.000000 | 0.000242 |
| California | 4n | 0.000000 | 0.000242 |

These are validation-only stability estimates. The existing native-tuned CTGAN and TVAE panel uses one fit seed and is reported separately. Every q8 artifact fits the 10,240-byte L3 byte cap, but other release gates are incomplete or failed. MFS-v2, PTF-v1, and certification are null.
