# News validation: packed DOPE q10 against native-tuned CTGAN and TVAE

The lossless packed projection and GPU-trained q10 model total 9,205 bytes, under the 10,240-byte L3 artifact limit. All 12 n/2n/4n/8n sample hashes replayed exactly. This remains validation research: the packed projection is restricted, the official tests are sealed, and MFS-v2/PTF-v1 are null.

| Method | n median retention | 4n median retention | Artifact bytes | L3 bytes |
|---|---:|---:|---:|---:|
| DOPE | 0.3287 | 0.3109 | 9,205 | yes |
| CTGAN | -0.8818 | -0.8440 | 2,922,575 | no |
| TVAE | -1.0013 | -0.9956 | 1,485,221 | no |

| Comparator | Median paired DOPE difference at n | At 4n |
|---|---:|---:|
| CTGAN | +1.2128 | +1.1549 |
| TVAE | +1.3374 | +1.2785 |

Values are null-normalized CatBoost TSTR–TRTR retention on the same training-derived News validation rows, fit seed 23 and sample seeds 101/211/307. Native author KPI values are recorded in JSON and used only within each method's tuning search. The v1 failed receipt attempt is retained. There is no paired superiority or production certification claim.
