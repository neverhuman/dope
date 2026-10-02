# S3 matched confirmation with original Forest-Flow

Single-fit-seed matched S3 confirmation validation. Training cohorts are bounded subsets of official S3 training partitions. These descriptive validation outcomes are not public-core paired analysis, five-fit stability or certification. All four DOPE profiles remain visible; no production family is selected.

Official tests sealed. MFS-v2 / PTF-v1 / release-safe L3: null. One fit seed; sample medians have no fit uncertainty.

| Dataset | Method/config | Size | Charged bytes | Statuses | CatBoost | Linear | MLP |
|---|---|---:|---:|---|---:|---:|---:|
| 1193_BNG_lowbwt | CTGAN/default | 1n | 833110 | {"ok": 3} | -0.159522 | -0.019407 | -0.047608 |
| 1193_BNG_lowbwt | CTGAN/default | 4n | 833110 | {"ok": 3} | -0.085486 | -0.097863 | -0.148511 |
| 1193_BNG_lowbwt | CTGAN/native_selected | 1n | 691862 | {"ok": 3} | 0.376424 | 0.439616 | 0.442264 |
| 1193_BNG_lowbwt | CTGAN/native_selected | 4n | 691862 | {"ok": 3} | 0.352534 | 0.455827 | 0.460463 |
| 1193_BNG_lowbwt | DOPE/features12_steps2048 | 1n | 2436 | {"ok": 3} | 0.963524 | 1.014821 | 0.988963 |
| 1193_BNG_lowbwt | DOPE/features12_steps2048 | 4n | 2436 | {"ok": 3} | 0.970335 | 1.003039 | 0.987064 |
| 1193_BNG_lowbwt | DOPE/features12_steps512 | 1n | 2426 | {"ok": 3} | 0.968886 | 1.008658 | 1.012778 |
| 1193_BNG_lowbwt | DOPE/features12_steps512 | 4n | 2426 | {"ok": 3} | 0.987364 | 1.010096 | 1.046575 |
| 1193_BNG_lowbwt | DOPE/features24_steps2048 | 1n | 2436 | {"ok": 3} | 0.963524 | 1.014821 | 0.988963 |
| 1193_BNG_lowbwt | DOPE/features24_steps2048 | 4n | 2436 | {"ok": 3} | 0.970335 | 1.003039 | 0.987064 |
| 1193_BNG_lowbwt | DOPE/features24_steps512 | 1n | 2426 | {"ok": 3} | 0.968886 | 1.008658 | 1.012778 |
| 1193_BNG_lowbwt | DOPE/features24_steps512 | 4n | 2426 | {"ok": 3} | 0.987364 | 1.010096 | 1.046575 |
| 1193_BNG_lowbwt | ForestDiffusion/Forest-Flow/default | 1n | 416921273 | {"ok": 3} | 0.991586 | 0.980575 | 1.029450 |
| 1193_BNG_lowbwt | ForestDiffusion/Forest-Flow/default | 4n | 416921273 | {"ok": 3} | 1.004435 | 0.991521 | 1.019504 |
| 1193_BNG_lowbwt | ForestDiffusion/Forest-Flow/native_selected | 1n | 416921273 | {"ok": 3} | 0.991586 | 0.980575 | 1.029450 |
| 1193_BNG_lowbwt | ForestDiffusion/Forest-Flow/native_selected | 4n | 416921273 | {"ok": 3} | 1.004435 | 0.991521 | 1.019504 |
| 1193_BNG_lowbwt | TVAE/default | 1n | 326636 | {"ok": 3} | 0.912612 | -9.492015 | -1.055974 |
| 1193_BNG_lowbwt | TVAE/default | 4n | 326636 | {"ok": 3} | 0.852489 | 0.105466 | -0.398052 |
| 1193_BNG_lowbwt | TVAE/native_selected | 1n | 293868 | {"ok": 3} | 1.019705 | 0.878072 | 0.490938 |
| 1193_BNG_lowbwt | TVAE/native_selected | 4n | 293868 | {"ok": 3} | 1.028785 | 0.876098 | 0.623959 |
| 582_fri_c1_500_25 | CTGAN/default | 1n | 1399953 | {"ok": 3} | -0.081438 | 0.019885 | null |
| 582_fri_c1_500_25 | CTGAN/default | 4n | 1399953 | {"ok": 3} | -0.083984 | -0.516484 | null |
| 582_fri_c1_500_25 | CTGAN/native_selected | 1n | 1399953 | {"ok": 3} | -0.081438 | 0.019885 | null |
| 582_fri_c1_500_25 | CTGAN/native_selected | 4n | 1399953 | {"ok": 3} | -0.083984 | -0.516484 | null |
| 582_fri_c1_500_25 | DOPE/features12_steps2048 | 1n | 5670 | {"ok": 3} | 0.750988 | 0.614203 | null |
| 582_fri_c1_500_25 | DOPE/features12_steps2048 | 4n | 5670 | {"ok": 3} | 0.908751 | 0.717291 | null |
| 582_fri_c1_500_25 | DOPE/features12_steps512 | 1n | 5654 | {"ok": 3} | 0.370490 | 0.864137 | null |
| 582_fri_c1_500_25 | DOPE/features12_steps512 | 4n | 5654 | {"ok": 3} | 0.403672 | 0.885445 | null |
| 582_fri_c1_500_25 | DOPE/features24_steps2048 | 1n | 6307 | {"ok": 3} | 0.273301 | 0.268228 | null |
| 582_fri_c1_500_25 | DOPE/features24_steps2048 | 4n | 6307 | {"ok": 3} | 0.344220 | 0.641018 | null |
| 582_fri_c1_500_25 | DOPE/features24_steps512 | 1n | 6286 | {"ok": 3} | 0.322243 | 0.921536 | null |
| 582_fri_c1_500_25 | DOPE/features24_steps512 | 4n | 6286 | {"ok": 3} | 0.342249 | 1.008212 | null |
| 582_fri_c1_500_25 | ForestDiffusion/Forest-Flow/default | 1n | 1085155062 | {"ok": 3} | 0.729714 | 0.753766 | null |
| 582_fri_c1_500_25 | ForestDiffusion/Forest-Flow/default | 4n | 1085155062 | {"ok": 3} | 0.883115 | 0.977239 | null |
| 582_fri_c1_500_25 | ForestDiffusion/Forest-Flow/native_selected | 1n | 1085155062 | {"ok": 3} | 0.729714 | 0.753766 | null |
| 582_fri_c1_500_25 | ForestDiffusion/Forest-Flow/native_selected | 4n | 1085155062 | {"ok": 3} | 0.883115 | 0.977239 | null |
| 582_fri_c1_500_25 | TVAE/default | 1n | 741479 | {"ok": 3} | 0.153790 | -0.384858 | null |
| 582_fri_c1_500_25 | TVAE/default | 4n | 741479 | {"ok": 3} | 0.119764 | -0.510344 | null |
| 582_fri_c1_500_25 | TVAE/native_selected | 1n | 708711 | {"ok": 3} | 0.351584 | 0.337962 | null |
| 582_fri_c1_500_25 | TVAE/native_selected | 4n | 708711 | {"ok": 3} | 0.394542 | 0.515618 | null |
| feynman_I_12_1 | CTGAN/default | 1n | 671458 | {"ok": 3} | -0.282761 | -0.107248 | 0.102643 |
| feynman_I_12_1 | CTGAN/default | 4n | 671458 | {"ok": 3} | -0.070396 | -0.046316 | -0.034623 |
| feynman_I_12_1 | CTGAN/native_selected | 1n | 535010 | {"ok": 3} | 0.706253 | 0.768538 | 0.752520 |
| feynman_I_12_1 | CTGAN/native_selected | 4n | 535010 | {"ok": 3} | 0.719484 | 0.787654 | 0.751917 |
| feynman_I_12_1 | DOPE/features12_steps2048 | 1n | 1032 | {"ok": 3} | 0.989216 | 0.986066 | 0.986120 |
| feynman_I_12_1 | DOPE/features12_steps2048 | 4n | 1032 | {"ok": 3} | 0.991049 | 0.989189 | 0.993280 |
| feynman_I_12_1 | DOPE/features12_steps512 | 1n | 1032 | {"ok": 3} | 0.995779 | 0.999395 | 0.994849 |
| feynman_I_12_1 | DOPE/features12_steps512 | 4n | 1032 | {"ok": 3} | 0.997964 | 0.999575 | 1.000994 |
| feynman_I_12_1 | DOPE/features24_steps2048 | 1n | 1032 | {"ok": 3} | 0.989216 | 0.986066 | 0.986120 |
| feynman_I_12_1 | DOPE/features24_steps2048 | 4n | 1032 | {"ok": 3} | 0.991049 | 0.989189 | 0.993280 |
| feynman_I_12_1 | DOPE/features24_steps512 | 1n | 1032 | {"ok": 3} | 0.995779 | 0.999395 | 0.994849 |
| feynman_I_12_1 | DOPE/features24_steps512 | 4n | 1032 | {"ok": 3} | 0.997964 | 0.999575 | 1.000994 |
| feynman_I_12_1 | ForestDiffusion/Forest-Flow/default | 1n | 123784681 | {"ok": 3} | 0.998528 | 1.000539 | 0.999866 |
| feynman_I_12_1 | ForestDiffusion/Forest-Flow/default | 4n | 123784681 | {"ok": 3} | 1.000596 | 1.000716 | 1.003483 |
| feynman_I_12_1 | ForestDiffusion/Forest-Flow/native_selected | 1n | 123784681 | {"ok": 3} | 0.998528 | 1.000539 | 0.999866 |
| feynman_I_12_1 | ForestDiffusion/Forest-Flow/native_selected | 4n | 123784681 | {"ok": 3} | 1.000596 | 1.000716 | 1.003483 |
| feynman_I_12_1 | TVAE/default | 1n | 221624 | {"ok": 3} | 0.465753 | 0.531712 | 0.494526 |
| feynman_I_12_1 | TVAE/default | 4n | 221624 | {"ok": 3} | 0.484608 | 0.565966 | 0.524939 |
| feynman_I_12_1 | TVAE/native_selected | 1n | 188856 | {"ok": 3} | 0.934359 | 0.937210 | 0.953470 |
| feynman_I_12_1 | TVAE/native_selected | 4n | 188856 | {"ok": 3} | 0.947771 | 0.943502 | 0.958151 |
| 596_fri_c2_250_5 | CTGAN/default | 1n | 730818 | {"ok": 3} | -0.308843 | null | null |
| 596_fri_c2_250_5 | CTGAN/default | 4n | 730818 | {"ok": 3} | -0.608029 | null | null |
| 596_fri_c2_250_5 | CTGAN/native_selected | 1n | 593602 | {"ok": 3} | 0.150477 | null | null |
| 596_fri_c2_250_5 | CTGAN/native_selected | 4n | 593602 | {"ok": 3} | 0.045582 | null | null |
| 596_fri_c2_250_5 | DOPE/features12_steps2048 | 1n | 1698 | {"ok": 3} | 0.754642 | null | null |
| 596_fri_c2_250_5 | DOPE/features12_steps2048 | 4n | 1698 | {"ok": 3} | 0.807228 | null | null |
| 596_fri_c2_250_5 | DOPE/features12_steps512 | 1n | 1693 | {"ok": 3} | 0.248916 | null | null |
| 596_fri_c2_250_5 | DOPE/features12_steps512 | 4n | 1693 | {"ok": 3} | 0.212330 | null | null |
| 596_fri_c2_250_5 | DOPE/features24_steps2048 | 1n | None | {"fit_unavailable": 3} | null | null | null |
| 596_fri_c2_250_5 | DOPE/features24_steps2048 | 4n | None | {"fit_unavailable": 3} | null | null | null |
| 596_fri_c2_250_5 | DOPE/features24_steps512 | 1n | 1693 | {"ok": 3} | 0.248916 | null | null |
| 596_fri_c2_250_5 | DOPE/features24_steps512 | 4n | 1693 | {"ok": 3} | 0.212330 | null | null |
| 596_fri_c2_250_5 | ForestDiffusion/Forest-Flow/default | 1n | 234776554 | {"ok": 3} | 0.841336 | null | null |
| 596_fri_c2_250_5 | ForestDiffusion/Forest-Flow/default | 4n | 234776554 | {"ok": 3} | 0.968133 | null | null |
| 596_fri_c2_250_5 | ForestDiffusion/Forest-Flow/native_selected | 1n | 234776554 | {"ok": 3} | 0.841336 | null | null |
| 596_fri_c2_250_5 | ForestDiffusion/Forest-Flow/native_selected | 4n | 234776554 | {"ok": 3} | 0.968133 | null | null |
| 596_fri_c2_250_5 | TVAE/default | 1n | 274904 | {"ok": 3} | 0.190247 | null | null |
| 596_fri_c2_250_5 | TVAE/default | 4n | 274904 | {"ok": 3} | 0.246773 | null | null |
| 596_fri_c2_250_5 | TVAE/native_selected | 1n | 274904 | {"ok": 3} | 0.190247 | null | null |
| 596_fri_c2_250_5 | TVAE/native_selected | 4n | 274904 | {"ok": 3} | 0.246773 | null | null |
| strogatz_vdp1 | CTGAN/default | 1n | 650291 | {"ok": 3} | -0.088939 | -0.229981 | -0.163270 |
| strogatz_vdp1 | CTGAN/default | 4n | 650291 | {"ok": 3} | -0.214633 | -0.640034 | 0.267955 |
| strogatz_vdp1 | CTGAN/native_selected | 1n | 650291 | {"ok": 3} | -0.088939 | -0.229981 | -0.163270 |
| strogatz_vdp1 | CTGAN/native_selected | 4n | 650291 | {"ok": 3} | -0.214633 | -0.640034 | 0.267955 |
| strogatz_vdp1 | DOPE/features12_steps2048 | 1n | 984 | {"ok": 3} | 0.522816 | 1.155061 | 0.608214 |
| strogatz_vdp1 | DOPE/features12_steps2048 | 4n | 984 | {"ok": 3} | 0.581229 | 1.286740 | 1.240096 |
| strogatz_vdp1 | DOPE/features12_steps512 | 1n | 991 | {"ok": 3} | 0.234889 | 0.014153 | 0.968880 |
| strogatz_vdp1 | DOPE/features12_steps512 | 4n | 991 | {"ok": 3} | 0.312646 | 0.768517 | 1.108657 |
| strogatz_vdp1 | DOPE/features24_steps2048 | 1n | None | {"fit_unavailable": 3} | null | null | null |
| strogatz_vdp1 | DOPE/features24_steps2048 | 4n | None | {"fit_unavailable": 3} | null | null | null |
| strogatz_vdp1 | DOPE/features24_steps512 | 1n | 991 | {"ok": 3} | 0.234889 | 0.014153 | 0.968880 |
| strogatz_vdp1 | DOPE/features24_steps512 | 4n | 991 | {"ok": 3} | 0.312646 | 0.768517 | 1.108657 |
| strogatz_vdp1 | ForestDiffusion/Forest-Flow/default | 1n | 115202253 | {"ok": 3} | 0.958339 | 1.082719 | 0.317612 |
| strogatz_vdp1 | ForestDiffusion/Forest-Flow/default | 4n | 115202253 | {"ok": 3} | 1.036219 | 0.985127 | 1.467559 |
| strogatz_vdp1 | ForestDiffusion/Forest-Flow/native_selected | 1n | 115202253 | {"ok": 3} | 0.958339 | 1.082719 | 0.317612 |
| strogatz_vdp1 | ForestDiffusion/Forest-Flow/native_selected | 4n | 115202253 | {"ok": 3} | 1.036219 | 0.985127 | 1.467559 |
| strogatz_vdp1 | TVAE/default | 1n | 212745 | {"ok": 3} | 0.014259 | 0.180714 | 0.655985 |
| strogatz_vdp1 | TVAE/default | 4n | 212745 | {"ok": 3} | 0.076805 | 0.196118 | 0.375301 |
| strogatz_vdp1 | TVAE/native_selected | 1n | 212745 | {"ok": 3} | 0.257146 | 0.940436 | 0.605635 |
| strogatz_vdp1 | TVAE/native_selected | 4n | 212745 | {"ok": 3} | 0.226255 | 0.633748 | 0.461072 |
| 503_wind | CTGAN/default | 1n | 1109070 | {"ok": 3} | -0.023761 | 0.167131 | -0.035474 |
| 503_wind | CTGAN/default | 4n | 1109070 | {"ok": 3} | -0.140690 | 0.199288 | -0.035508 |
| 503_wind | CTGAN/native_selected | 1n | 953166 | {"ok": 3} | 0.587945 | 0.507995 | 0.451293 |
| 503_wind | CTGAN/native_selected | 4n | 953166 | {"ok": 3} | 0.489104 | 0.517808 | 0.451820 |
| 503_wind | DOPE/features12_steps2048 | 1n | 3846 | {"ok": 3} | 0.968952 | 0.998344 | 1.014047 |
| 503_wind | DOPE/features12_steps2048 | 4n | 3846 | {"ok": 3} | 0.996152 | 1.001693 | 1.036675 |
| 503_wind | DOPE/features12_steps512 | 1n | 3842 | {"ok": 3} | 0.966336 | 0.989656 | 0.992720 |
| 503_wind | DOPE/features12_steps512 | 4n | 3842 | {"ok": 3} | 0.985299 | 0.987685 | 1.021307 |
| 503_wind | DOPE/features24_steps2048 | 1n | 3947 | {"ok": 3} | 1.004274 | 0.999079 | 1.024080 |
| 503_wind | DOPE/features24_steps2048 | 4n | 3947 | {"ok": 3} | 1.050481 | 1.008858 | 1.082482 |
| 503_wind | DOPE/features24_steps512 | 1n | 3948 | {"ok": 3} | 0.985347 | 0.975288 | 0.979919 |
| 503_wind | DOPE/features24_steps512 | 4n | 3948 | {"ok": 3} | 0.990486 | 0.972959 | 0.977652 |
| 503_wind | ForestDiffusion/Forest-Flow/default | 1n | 637056689 | {"ok": 3} | 0.980539 | 0.970833 | 0.961042 |
| 503_wind | ForestDiffusion/Forest-Flow/default | 4n | 637056689 | {"ok": 3} | 1.015820 | 0.984333 | 1.067698 |
| 503_wind | ForestDiffusion/Forest-Flow/native_selected | 1n | 637056689 | {"ok": 3} | 0.980539 | 0.970833 | 0.961042 |
| 503_wind | ForestDiffusion/Forest-Flow/native_selected | 4n | 637056689 | {"ok": 3} | 1.015820 | 0.984333 | 1.067698 |
| 503_wind | TVAE/default | 1n | 503908 | {"ok": 3} | 0.563548 | 0.613220 | 0.584351 |
| 503_wind | TVAE/default | 4n | 503908 | {"ok": 3} | 0.592297 | 0.639665 | 0.649112 |
| 503_wind | TVAE/native_selected | 1n | 471140 | {"ok": 3} | 0.752420 | 0.809012 | 0.819500 |
| 503_wind | TVAE/native_selected | 4n | 471140 | {"ok": 3} | 0.802182 | 0.836371 | 0.887151 |

Original Forest-Flow selects from its frozen two-configuration grid by the author mean regression R2 over linear, AdaBoost, random forest and XGBoost auditors, five auditor fit seeds. CTGAN/TVAE retain their prior frozen native efficacy selections. Native values are never ranked across methods.

All sampler and projection bytes are charged. Baseline values with artifacts over 10,240 bytes describe unconstrained quality. Copy counts and controls are incomplete privacy evidence; failed or unavailable cells never supply a DOPE win.

Cost: {"forest_cpu_energy_joules": null, "forest_fit_core_seconds": 953.4443792356178, "forest_fit_native_sample_operation_seconds": 8801.492797114886, "forest_gpu_host": "xbabe1", "forest_new_gpu_fits": 12, "forest_peak_gpu_used_mib": 3181, "forest_shared_evaluator_seconds": 228.73225121363066, "forest_whole_device_energy_estimate_joules": 357200.3873267654, "limitations": "Prior immutable metrics add no new compute. Operation time includes hashing, generation and native auditors; device power includes idle and is not job-attributed. Earlier DOPE architecture research is a separate cost; per-cell budgets do not imply equal total research spend.", "prior_comparison_new_fits": 0, "prior_comparison_new_tuning_trials": 0, "shared_evaluator_host": "xbabe2"}

Citation keys: xu2019modeling, jolicoeurmartineau2024generating
