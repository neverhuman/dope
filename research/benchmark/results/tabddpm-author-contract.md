# Author TabDDPM contract probe

Two author-core GPU fits on the same 72-row S3 training-derived regression cohort (210_cloud). This is a fit/sample/native-objective contract probe. It is not a matched common-outcome benchmark or native tuning.

| Fit seed | Native five-sample validation R² mean | Charged bytes | GPU training seconds | Fit entrypoint seconds | Host |
|---:|---:|---:|---:|---:|---|
| 11 | 0.8035797920 | 500946 | 2.816294 | 107.366174 | xbabe3 |
| 23 | 0.7893576197 | 500979 | 2.830357 | 105.083254 | xbabe3 |

Both artifacts exceed the 10,240-byte L3 cap. Sample-seed 0 replay is byte-exact for both fits. All five native synthetic seeds are accounted for each fit; native values are never ranked across methods.

Physical operation costs include all four failed fit entrypoints: 818.082278s fit total (605.632850s failed preflight), 1261.569109s sample/repeat and 183.854202s native evaluation. Three inherited sample-operation aliases are verified and charged once. Energy is unavailable.

## Scope and limitations

- One S3 train-derived 72-row regression cohort, five projected features; not California benchmark rows.
- Only fit rows enter preprocessing/training; official tests never staged.
- Fit seeds11/23; author numeric base settings, feature width follows projected data.
- Sample n rows with min(n,2000) batch, versus author52800 rows/batch8192.
- Tensor-only weights and numeric quantile/discrete preprocessing arrays plus projection; no Python-object deserialization.
- Author-core contract probe, not two reported-experiment reproductions, native tuning or a matched common-outcome benchmark.
- Historical runtime lock omitted executable bytecode and directory aliases: fit-time executable closure is unverified. Current caches match pinned source; this cannot establish their historical bytes.
- The original lib64-to-lib ABI alias and interpreter aliases are explicitly pinned; every additional source/runtime symlink is rejected.
- Regression only; classification/categorical and generic final-runner integration remain pending.
- Learned quantiles/discrete state are charged; no claim that preprocessing is free of source observations.
- Both artifacts exceed L3; copy, leakage, attacks, real-vs-real and production profile gates are not complete.
- Native values are not ranked across methods; no tuned configuration or winning fit seed selected.

Official tests remain sealed. MFS-v2, PTF-v1 and release-safe scores are null; no DOPE win, privacy or certification claim.

Author source and objective: [TabDDPM](https://github.com/yandex-research/tab-ddpm), bibliography key `kotelnikov2023tabddpm`.
