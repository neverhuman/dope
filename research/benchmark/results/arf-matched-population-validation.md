# Complete original ARF / DOPE matched S3 validation

100 planned bounded views from official S3 training partitions; official tests remain sealed.
ARF uses immutable author defaults and separate held-out FORDE native selections [@watson2023adversarial].
Common scores never retune ARF. Native values are reported per lineage and never ranked across methods.

| DOPE research profile | ARF configuration | Paired informative lineages | DOPE retention | ARF retention | Median paired difference |
|---|---|---:|---:|---:|---:|
| features12_steps2048 | author_default | 97 | 0.939876353161328 | 0.8404116242122822 | 0.07672480231706813 |
| features12_steps2048 | native_selected | 97 | 0.939876353161328 | 0.8255067785547131 | 0.09541472594147016 |
| features12_steps8192 | author_default | 68 | 0.9703479637731172 | 0.811160037892046 | 0.10471267248754867 |
| features12_steps8192 | native_selected | 68 | 0.9703479637731172 | 0.7498425755126212 | 0.12462484115045253 |
| features12_width8_steps8192 | author_default | 67 | 0.9832508299676646 | 0.843496605501471 | 0.1213900998844738 |
| features12_width8_steps8192 | native_selected | 67 | 0.9832508299676646 | 0.825773060929091 | 0.1752954544101063 |

Sample seeds reduce within lineage before dataset summaries; each pair may have a different available cohort.
Charged ARF model and projection bytes remain visible even above10240; this is unconstrained quality, not release-safe L3 evidence.
One fit seed; final five-fit/privacy/product coverage incomplete. No family selection or superiority inference.
MFS-v2/PTF-v1/release-safe/superiority are null; every unavailable cell remains visible.
