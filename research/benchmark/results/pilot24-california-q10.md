# California DOPE autoregressive pilot validation

Two frozen fit seeds completed on GPU hosts, each with three sample seeds at n and 4n. All 12 samples and validation metrics are receipt verified; no exact or near training-row copies were observed.

| Fit seed | Size | Candidate median CatBoost retention | q8 fit-23 median | Median paired difference |
|---:|---:|---:|---:|---:|
| 11 | 1n | 0.7071 | — | — |
| 11 | 4n | 0.7010 | — | — |
| 23 | 1n | 0.6934 | 0.7008 | -0.0087 |
| 23 | 4n | 0.6910 | 0.6948 | -0.0038 |

The candidate was chosen after seeing California validation utility, so this is descriptive research. Fit seed 23 is lower than q8 at both sizes. The compiler selected a symbolic artifact on GPU hosts; no neural GPU training is claimed. Both 2,525-byte artifacts failed utility and driver gates. Official tests remain sealed; MFS-v2, PTF-v1, and certification are null.
