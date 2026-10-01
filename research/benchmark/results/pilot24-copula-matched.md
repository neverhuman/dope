# DOPE versus native-tuned GaussianCopula: validation pilot

Same Adult, California, and News training-derived validation partitions. GaussianCopula was selected only by its own held-out mean log density. The common outcome below is null-normalized CatBoost TSTR–TRTR retention over fit seed 23, sample seeds 101/211, and sample sizes n/4n.

| Dataset | DOPE median | Copula median | Median paired difference (DOPE − Copula) | DOPE bytes / L3 | Copula bytes / L3 |
|---|---:|---:|---:|---:|---:|
| Adult | 0.829 | 0.712 | +0.119 | 6,337 / yes | 270,086 / no |
| California | 0.698 | 0.723 | -0.024 | 2,525 / yes | 4,316 / yes |
| News | -3.861 | -0.518 | -3.309 | 13,603 / no | 91,117 / no |

| Dataset | Copula selected native mean log density | Selected native trial |
|---|---:|---:|
| Adult | -303.411932 | 1 |
| California | -9.199259 | 1 |
| News | -690.775528 | 1 |

Six default fit/sample cells failed under their frozen time caps and remain explicit null outcomes. Artifact bytes include the fitted generator and projection map. All 24 scored Copula metrics replayed exactly from their verified samples, excluding elapsed timing. Native KPI values are used only within GaussianCopula datasets. This three-dataset, one-fit-seed panel is descriptive and does not establish superiority or production certification. Official tests stayed sealed; MFS-v2 and PTF-v1 are null.
