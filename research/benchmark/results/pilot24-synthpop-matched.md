# DOPE versus synthpop CART: paired pilot validation

Synthpop configurations were selected by their own frozen CART pMSE on training-derived validation rows. Native pMSE values compare synthpop configurations within a dataset only. The common outcome is CatBoost TSTR/TRTR retention at fit seed 23 and three paired sample seeds.

| Dataset | Synthpop | Size | Native pMSE | Synthpop retention | DOPE retention | Paired DOPE − synthpop | Synthpop bytes | Copy-gate failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Adult | default | 1n | 0.023349 | 0.9269 | 0.8275 | -0.1022 | 58,346,388 | 3/3 |
| Adult | default | 4n | 0.023349 | 0.9356 | 0.8293 | -0.1053 | 58,346,388 | 3/3 |
| Adult | tuned | 1n | 0.021222 | 0.9255 | 0.8275 | -0.0980 | 56,353,264 | 3/3 |
| Adult | tuned | 4n | 0.021222 | 0.9303 | 0.8293 | -0.1018 | 56,353,264 | 3/3 |
| California | default | 1n | 0.035828 | 0.9588 | 0.7008 | -0.2609 | 2,126,519 | 2/3 |
| California | default | 4n | 0.035828 | 0.9651 | 0.6948 | -0.2716 | 2,126,519 | 3/3 |
| California | tuned | 1n | 0.035828 | 0.9588 | 0.7008 | -0.2609 | 2,126,519 | 2/3 |
| California | tuned | 4n | 0.035828 | 0.9651 | 0.6948 | -0.2716 | 2,126,519 | 3/3 |
| News | default | 1n | 0.145160 | 0.7312 | 0.3287 | -0.3951 | 139,417,230 | 0/3 |
| News | default | 4n | 0.145160 | 0.5328 | 0.3109 | -0.1968 | 139,417,230 | 0/3 |
| News | tuned | 1n | 0.135756 | 0.6419 | 0.3287 | -0.3132 | 127,647,802 | 0/3 |
| News | tuned | 4n | 0.135756 | 0.7421 | 0.3109 | -0.4281 | 127,647,802 | 0/3 |

All synthpop artifacts exceed the 10,240-byte L3 limit. The JSON retains each paired seed, receipt hash, copy count, and failed release gate. Official tests remain sealed; MFS-v2, PTF-v1, and production certification are null. These three datasets and one fit seed do not support a paper superiority claim.
