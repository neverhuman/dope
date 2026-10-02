# S3 matched discovery validation

Single-fit-seed discovery research, not public-core paired analysis, five-fit stability, confirmation or certification. Training cohorts are bounded subsets of official S3 training partitions; native selection and these descriptive outcomes use training-derived validation. All four DOPE profiles retained; production family unselected.

Official tests sealed. MFS-v2 / PTF-v1 / release-safe L3: null. Bars and table values are sample medians from one fit seed, without fit uncertainty.

| Dataset | Method/config | Size | Charged bytes | Sample statuses | CatBoost | Linear | MLP |
|---|---|---:|---:|---|---:|---:|---:|
| 210_cloud (0d482c1dd7cc6592) | CTGAN/default | 1n | 708208 | {"ok": 3} | -0.026517 | -0.264647 | -0.346108 |
| 210_cloud (0d482c1dd7cc6592) | CTGAN/default | 4n | 708208 | {"ok": 3} | -0.494016 | -0.343032 | -0.358608 |
| 210_cloud (0d482c1dd7cc6592) | CTGAN/native_selected | 1n | 572016 | {"ok": 3} | 0.282856 | 0.336671 | -0.004395 |
| 210_cloud (0d482c1dd7cc6592) | CTGAN/native_selected | 4n | 572016 | {"ok": 3} | -0.033833 | 0.178173 | 0.185502 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features12_steps2048 | 1n | 1844 | {"ok": 3} | 1.037394 | 1.115910 | 0.856622 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features12_steps2048 | 4n | 1844 | {"ok": 3} | 0.977180 | 1.094373 | 1.023879 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features12_steps512 | 1n | 1839 | {"ok": 3} | 1.016275 | 1.061952 | 0.851543 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features12_steps512 | 4n | 1839 | {"ok": 3} | 0.938967 | 1.055657 | 0.982770 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features24_steps2048 | 1n | 1844 | {"ok": 3} | 1.037394 | 1.115910 | 0.856622 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features24_steps2048 | 4n | 1844 | {"ok": 3} | 0.977180 | 1.094373 | 1.023879 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features24_steps512 | 1n | 1839 | {"ok": 3} | 1.016275 | 1.061952 | 0.851543 |
| 210_cloud (0d482c1dd7cc6592) | DOPE/features24_steps512 | 4n | 1839 | {"ok": 3} | 0.938967 | 1.055657 | 0.982770 |
| 210_cloud (0d482c1dd7cc6592) | TVAE/default | 1n | 256070 | {"ok": 3} | 0.321609 | 0.833802 | 0.047333 |
| 210_cloud (0d482c1dd7cc6592) | TVAE/default | 4n | 256070 | {"ok": 3} | 0.514047 | 0.920242 | 0.794867 |
| 210_cloud (0d482c1dd7cc6592) | TVAE/native_selected | 1n | 223302 | {"ok": 3} | 0.489562 | 0.972796 | 0.100055 |
| 210_cloud (0d482c1dd7cc6592) | TVAE/native_selected | 4n | 223302 | {"ok": 3} | 0.599914 | 0.912488 | 0.766932 |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | CTGAN/default | 1n | 1122279 | {"ok": 3} | -0.851171 | -1.052843 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | CTGAN/default | 4n | 1122279 | {"ok": 3} | -0.671751 | -1.032755 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | CTGAN/native_selected | 1n | 970407 | {"ok": 3} | -0.092561 | 0.047492 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | CTGAN/native_selected | 4n | 970407 | {"ok": 3} | -0.166211 | -0.209327 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features12_steps2048 | 1n | 4183 | {"ok": 3} | 0.809722 | 0.930489 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features12_steps2048 | 4n | 4183 | {"ok": 3} | 0.876097 | 0.917310 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features12_steps512 | 1n | 4174 | {"ok": 3} | 0.648318 | 0.752366 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features12_steps512 | 4n | 4174 | {"ok": 3} | 0.655713 | 0.741435 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features24_steps2048 | 1n | 4496 | {"ok": 3} | 0.828288 | 0.922718 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features24_steps2048 | 4n | 4496 | {"ok": 3} | 0.859769 | 0.959054 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features24_steps512 | 1n | 4486 | {"ok": 3} | 0.742214 | 0.823012 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | DOPE/features24_steps512 | 4n | 4486 | {"ok": 3} | 0.763210 | 0.852339 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | TVAE/default | 1n | 517245 | {"ok": 3} | 0.123150 | -2.266047 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | TVAE/default | 4n | 517245 | {"ok": 3} | 0.283758 | -1.549611 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | TVAE/native_selected | 1n | 517245 | {"ok": 3} | 0.538268 | 0.326038 | null |
| 1191_BNG_pbc (0f86a714ec5f3ac6) | TVAE/native_selected | 4n | 517245 | {"ok": 3} | 0.328746 | 0.279098 | null |
| feynman_test_1 (1a4456d1cc71a26a) | CTGAN/default | 1n | 851576 | {"ok": 3} | -0.159936 | -0.229868 | null |
| feynman_test_1 (1a4456d1cc71a26a) | CTGAN/default | 4n | 851576 | {"ok": 3} | -0.127171 | -0.142609 | null |
| feynman_test_1 (1a4456d1cc71a26a) | CTGAN/native_selected | 1n | 851576 | {"ok": 3} | -0.159936 | -0.229868 | null |
| feynman_test_1 (1a4456d1cc71a26a) | CTGAN/native_selected | 4n | 851576 | {"ok": 3} | -0.127171 | -0.142609 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features12_steps2048 | 1n | 2191 | {"ok": 3} | 1.060790 | 0.951854 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features12_steps2048 | 4n | 2191 | {"ok": 3} | 1.167076 | 0.981591 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features12_steps512 | 1n | 2189 | {"ok": 3} | 0.424489 | 0.209318 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features12_steps512 | 4n | 2189 | {"ok": 3} | 0.473910 | 0.246194 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features24_steps2048 | 1n | 2191 | {"ok": 3} | 1.060790 | 0.951854 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features24_steps2048 | 4n | 2191 | {"ok": 3} | 1.167076 | 0.981591 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features24_steps512 | 1n | 2189 | {"ok": 3} | 0.424489 | 0.209318 | null |
| feynman_test_1 (1a4456d1cc71a26a) | DOPE/features24_steps512 | 4n | 2189 | {"ok": 3} | 0.473910 | 0.246194 | null |
| feynman_test_1 (1a4456d1cc71a26a) | TVAE/default | 1n | 338190 | {"ok": 3} | -0.071813 | -0.021455 | null |
| feynman_test_1 (1a4456d1cc71a26a) | TVAE/default | 4n | 338190 | {"ok": 3} | -0.076572 | -0.013937 | null |
| feynman_test_1 (1a4456d1cc71a26a) | TVAE/native_selected | 1n | 338190 | {"ok": 3} | 0.353544 | 0.489876 | null |
| feynman_test_1 (1a4456d1cc71a26a) | TVAE/native_selected | 4n | 338190 | {"ok": 3} | 0.348579 | 0.473311 | null |
| 603_fri_c0_250_50 (297b7688a045c583) | CTGAN/default | 1n | 2097537 | {"ok": 3} | -0.979645 | -1.016752 | -9.249821 |
| 603_fri_c0_250_50 (297b7688a045c583) | CTGAN/default | 4n | 2097537 | {"ok": 3} | -0.851341 | -0.962056 | -7.804332 |
| 603_fri_c0_250_50 (297b7688a045c583) | CTGAN/native_selected | 1n | 2097537 | {"ok": 3} | -0.979645 | -1.016752 | -9.249821 |
| 603_fri_c0_250_50 (297b7688a045c583) | CTGAN/native_selected | 4n | 2097537 | {"ok": 3} | -0.851341 | -0.962056 | -7.804332 |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features12_steps2048 | 1n | 10171 | {"ok": 3} | 0.884811 | 0.975332 | 0.880288 |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features12_steps2048 | 4n | 10171 | {"ok": 3} | 1.156250 | 1.070026 | 2.688317 |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features12_steps512 | 1n | 10167 | {"ok": 3} | 0.825850 | 1.002934 | 1.094819 |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features12_steps512 | 4n | 10167 | {"ok": 3} | 1.085062 | 1.051004 | 3.273683 |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features24_steps2048 | 1n | 10787 | {"fit_unavailable": 3} | null | null | null |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features24_steps2048 | 4n | 10787 | {"fit_unavailable": 3} | null | null | null |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features24_steps512 | 1n | 10781 | {"fit_unavailable": 3} | null | null | null |
| 603_fri_c0_250_50 (297b7688a045c583) | DOPE/features24_steps512 | 4n | 10781 | {"fit_unavailable": 3} | null | null | null |
| 603_fri_c0_250_50 (297b7688a045c583) | TVAE/default | 1n | 1242839 | {"ok": 3} | 0.047560 | -0.301750 | -2.933926 |
| 603_fri_c0_250_50 (297b7688a045c583) | TVAE/default | 4n | 1242839 | {"ok": 3} | 0.338497 | 0.256065 | -0.822498 |
| 603_fri_c0_250_50 (297b7688a045c583) | TVAE/native_selected | 1n | 1210071 | {"ok": 3} | 0.187582 | -0.162563 | -3.360202 |
| 603_fri_c0_250_50 (297b7688a045c583) | TVAE/native_selected | 4n | 1210071 | {"ok": 3} | 0.438423 | 0.452113 | -0.537029 |
| feynman_test_11 (471f11cc2d38853a) | CTGAN/default | 1n | 750091 | {"ok": 3} | -0.689578 | -0.760562 | -0.537397 |
| feynman_test_11 (471f11cc2d38853a) | CTGAN/default | 4n | 750091 | {"ok": 3} | -0.537696 | -0.694894 | -0.495958 |
| feynman_test_11 (471f11cc2d38853a) | CTGAN/native_selected | 1n | 750091 | {"ok": 3} | 0.076731 | 0.187522 | 0.076045 |
| feynman_test_11 (471f11cc2d38853a) | CTGAN/native_selected | 4n | 750091 | {"ok": 3} | 0.091855 | 0.181993 | 0.118518 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features12_steps2048 | 1n | 1481 | {"ok": 3} | 0.964485 | 0.998317 | 0.969435 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features12_steps2048 | 4n | 1481 | {"ok": 3} | 0.976037 | 0.997459 | 0.982618 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features12_steps512 | 1n | 1479 | {"ok": 3} | 0.954247 | 1.013400 | 0.954734 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features12_steps512 | 4n | 1479 | {"ok": 3} | 0.966658 | 1.013124 | 0.966089 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features24_steps2048 | 1n | 1481 | {"ok": 3} | 0.964485 | 0.998317 | 0.969435 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features24_steps2048 | 4n | 1481 | {"ok": 3} | 0.976037 | 0.997459 | 0.982618 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features24_steps512 | 1n | 1479 | {"ok": 3} | 0.954247 | 1.013400 | 0.954734 |
| feynman_test_11 (471f11cc2d38853a) | DOPE/features24_steps512 | 4n | 1479 | {"ok": 3} | 0.966658 | 1.013124 | 0.966089 |
| feynman_test_11 (471f11cc2d38853a) | TVAE/default | 1n | 269601 | {"ok": 3} | 0.368931 | 0.596270 | 0.373183 |
| feynman_test_11 (471f11cc2d38853a) | TVAE/default | 4n | 269601 | {"ok": 3} | 0.424758 | 0.593426 | 0.393244 |
| feynman_test_11 (471f11cc2d38853a) | TVAE/native_selected | 1n | 269601 | {"ok": 3} | 0.620482 | 0.771758 | 0.632428 |
| feynman_test_11 (471f11cc2d38853a) | TVAE/native_selected | 4n | 269601 | {"ok": 3} | 0.632602 | 0.773612 | 0.659597 |
| 663_rabe_266 (a863cd42be4d05d4) | CTGAN/default | 1n | 636219 | {"ok": 3} | 0.181200 | 0.528883 | 0.261810 |
| 663_rabe_266 (a863cd42be4d05d4) | CTGAN/default | 4n | 636219 | {"ok": 3} | 0.191475 | 0.302265 | 0.210462 |
| 663_rabe_266 (a863cd42be4d05d4) | CTGAN/native_selected | 1n | 636219 | {"ok": 3} | 0.181200 | 0.528883 | 0.261810 |
| 663_rabe_266 (a863cd42be4d05d4) | CTGAN/native_selected | 4n | 636219 | {"ok": 3} | 0.191475 | 0.302265 | 0.210462 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features12_steps2048 | 1n | 1014 | {"ok": 3} | 0.978144 | 0.992725 | 0.969156 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features12_steps2048 | 4n | 1014 | {"ok": 3} | 0.993285 | 0.996669 | 1.052096 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features12_steps512 | 1n | 1016 | {"ok": 3} | 0.983858 | 0.996237 | 0.994151 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features12_steps512 | 4n | 1016 | {"ok": 3} | 0.996699 | 0.998974 | 1.050867 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features24_steps2048 | 1n | 1014 | {"ok": 3} | 0.978144 | 0.992725 | 0.969156 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features24_steps2048 | 4n | 1014 | {"ok": 3} | 0.993285 | 0.996669 | 1.052096 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features24_steps512 | 1n | 1016 | {"ok": 3} | 0.983858 | 0.996237 | 0.994151 |
| 663_rabe_266 (a863cd42be4d05d4) | DOPE/features24_steps512 | 4n | 1016 | {"ok": 3} | 0.996699 | 0.998974 | 1.050867 |
| 663_rabe_266 (a863cd42be4d05d4) | TVAE/default | 1n | 206865 | {"ok": 3} | 0.473080 | 0.676316 | 0.526531 |
| 663_rabe_266 (a863cd42be4d05d4) | TVAE/default | 4n | 206865 | {"ok": 3} | 0.631509 | 0.674766 | 0.617548 |
| 663_rabe_266 (a863cd42be4d05d4) | TVAE/native_selected | 1n | 174097 | {"ok": 3} | 0.634947 | 0.743016 | 0.666983 |
| 663_rabe_266 (a863cd42be4d05d4) | TVAE/native_selected | 4n | 174097 | {"ok": 3} | 0.629241 | 0.667754 | 0.508587 |

Native objectives: CTGAN/TVAE use frozen author-library efficacy: mean LinearRegression and MLPRegressor R2, maximize on validation, then bytes/config digest. Common utility and DOPE KPI never select baselines; native KPIs are not ranked across methods.

All four declared DOPE research profiles remain visible; no production family is selected. Baseline artifacts exceed L3 where charged bytes exceed 10,240; their common utility is unconstrained quality.

Failures and low-signal groups yield null medians. No failed or unavailable cell supplies a DOPE win. Copy/control counts are evidence, without a complete privacy gate or formal DP claim.

Cost: {"cpu_energy_joules": null, "dope_declared_gpu_fit_ceiling_seconds": 14400, "dope_dispatch_seconds_including_hash_and_admission_overhead": 1509.7510590329766, "dope_gpu_fit_seconds": 114.98480828106403, "gpu_device_energy_estimate_joules_including_idle": 10694.65019357225, "gpu_fit_hosts": ["xbabe1", "xbabe3"], "limitations": "Operation wall times include sample and metric replays and exclude admission scans. Prior native trial wall includes fit/sample/native efficacy. Energy and earlier architecture R&D are not fully accounted here. GPU energy is device-wide, includes idle, and is not baseline-attributed; per-cell parity is not equal total R&D spend.", "native_new_fits": 0, "native_new_tuning_trials": 0, "native_prior_48_trial_wall_seconds": 2144.4270977308042, "peak_observed_gpu_used_mib": 596, "sample_repeat_metric_operation_seconds": 3296.0290340210777}

Citation keys: xu2019modeling
