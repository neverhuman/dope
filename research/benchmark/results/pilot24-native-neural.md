# Adult / California / News native-neural validation comparison

Training-derived validation only. Official tests remain sealed. MFS-v2 and production PTF-v1 are null. Medians use fixed fit seed 23, sample seeds 101/211/307, and the same CatBoost auditor on real validation rows.

| Dataset | Method | Configuration | n retention | 4n retention | Charged bytes | L3 bytes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Adult | CTGAN | tuned | 0.0139 | 0.0171 | 3,032,190 | no |
| Adult | DOPE | q8_symbolic | 0.8275 | 0.8293 | 6,337 | yes |
| Adult | TVAE | default_and_tuned | 0.7951 | 0.7788 | 1,665,556 | no |
| California | CTGAN | default_and_tuned | 0.6652 | 0.6748 | 941,787 | no |
| California | DOPE | q8_symbolic | 0.7008 | 0.6948 | 2,525 | yes |
| California | TVAE | default | 0.8750 | 0.8756 | 373,361 | no |
| California | TVAE | tuned | 0.9154 | 0.9163 | 340,593 | no |
| News | CTGAN | tuned | -0.8818 | -0.8440 | 2,922,575 | no |
| News | DOPE | q10_overcap_quality | 0.3287 | 0.3109 | 13,713 | no |
| News | DOPE | q8_symbolic | -3.9536 | -4.0242 | 13,603 | no |
| News | TVAE | default_and_tuned | -1.0013 | -0.9956 | 1,485,221 | no |

A byte-eligible artifact is not production certified: generator, privacy, coverage, and release gates remain incomplete. News q10 is shown for unconstrained quality and exceeds the complete 10,240-byte L3 cap.

## Native validation objectives

Values are compared only within one method and dataset. Adult uses mean binary F1; California and News use regression R². A timeout has no KPI.

| Dataset | Method | Native objective | Default | Selected | Tuning attempts |
| --- | --- | --- | ---: | ---: | ---: |
| Adult | CTGAN | sdmetrics_mean_binary_f1 | timeout | 0.3376 | 8 |
| Adult | TVAE | sdmetrics_mean_binary_f1 | 0.6532 | 0.6532 | 4 |
| California | CTGAN | sdmetrics_mean_regression_r2 | 0.4138 | 0.4138 | 4 |
| California | TVAE | sdmetrics_mean_regression_r2 | 0.5642 | 0.5854 | 4 |
| News | CTGAN | sdmetrics_mean_regression_r2 | timeout | -4.3629 | 8 |
| News | TVAE | sdmetrics_mean_regression_r2 | -29.0100 | -29.0100 | 4 |

## Failed fixed fits

- Adult CTGAN default: timeout (fit seed 23).
- News CTGAN default: timeout (fit seed 23).

All 66 common metric cells, 42 SDV fixed-seed metric receipts, native selections, 24 DOPE conditional tuning trials, artifact charges, copy counts, and immutable scratch hashes are in the JSON. The News real-model gain over the null is small, so its retention ratio is unstable. This is not the preregistered public-test paired analysis.
