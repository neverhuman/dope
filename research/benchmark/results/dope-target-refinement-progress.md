# DOPE target refinement progress

One discovery lineage; three completed GPU profiles; eighteen completed validation cells. Full24-fit discovery is pending; this is a progress batch, not a final benchmark.

Comparators retain their frozen native likelihood selection. Retention is descriptive validation; no method selection or production score is inferred.

| Method | Configuration | Size | Charged bytes | CatBoost median retention |
| --- | --- | ---: | ---: | ---: |
| Chow-Liu | default | 1n | 3342 | 0.741066 |
| Chow-Liu | default | 4n | 3342 | 0.732036 |
| Chow-Liu | native_selected | 1n | 34152 | 0.179296 |
| Chow-Liu | native_selected | 4n | 34152 | 0.061108 |
| DOPE | features12_steps2048 | 1n | 1014 | 0.978144 |
| DOPE | features12_steps2048 | 4n | 1014 | 0.993285 |
| DOPE | features12_steps512 | 1n | 1016 | 0.983858 |
| DOPE | features12_steps512 | 4n | 1016 | 0.996699 |
| DOPE | features12_steps8192 | 1n | 1014 | 0.984921 |
| DOPE | features12_steps8192 | 4n | 1014 | 0.999614 |
| DOPE | features12_width8_steps8192 | 1n | 911 | 0.961544 |
| DOPE | features12_width8_steps8192 | 4n | 911 | 0.976152 |
| DOPE | features16_steps8192 | 1n | 1014 | 0.984921 |
| DOPE | features16_steps8192 | 4n | 1014 | 0.999614 |
| DOPE | features24_steps2048 | 1n | 1014 | 0.978144 |
| DOPE | features24_steps2048 | 4n | 1014 | 0.993285 |
| DOPE | features24_steps512 | 1n | 1016 | 0.983858 |
| DOPE | features24_steps512 | 4n | 1016 | 0.996699 |
| GaussianCopula | default | 1n | 3294 | 0.925531 |
| GaussianCopula | default | 4n | 3294 | 0.941450 |
| GaussianCopula | native_selected | 1n | 1216 | 0.952900 |
| GaussianCopula | native_selected | 4n | 1216 | 0.984326 |
| independent_marginals | default | 1n | 972 | -0.144029 |
| independent_marginals | default | 4n | 972 | 0.020517 |
| independent_marginals | native_selected | 1n | 1202 | -0.065376 |
| independent_marginals | native_selected | 4n | 1202 | -0.017006 |

The infrastructure-interrupted attempt remains recorded and charged; attempt2 is queued with the original job identity. Twenty unstarted jobs continue under the original limits. All official tests remain sealed; MFS-v2/PTF-v1/release/superiority stay null.
