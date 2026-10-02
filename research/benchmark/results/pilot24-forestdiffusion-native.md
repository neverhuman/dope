# ForestDiffusion native/default validation pilot

One fit seed (23), three sample seeds, official-training-derived validation. Native ML KPIs selected before common evaluation. Official tests remain sealed; MFS-v2, PTF-v1 and release-safe scores are null.

| Dataset | Native trial | Author ML KPI | Attempts |
|---|---:|---:|---:|
| Adult | null | null | 8 |
| California | 6 | 0.629042 | 8 |
| News | null | null | 8 |

Native values use macro F1 for Adult and R² for California/News; they are not ranked across targets or against other methods’ native objectives.

| Dataset | Configuration | n CatBoost retention | 4n CatBoost retention | Charged bytes |
|---|---|---:|---:|---:|
| Adult | cpu_author_default | null | null | null |
| Adult | gpu_author_default | null | null | null |
| Adult | native_selected | null | null | null |
| California | cpu_author_default | null | null | null |
| California | gpu_author_default | 0.954911 | 0.958861 | 396599816 |
| California | native_selected | 0.954911 | 0.958861 | 396599816 |
| News | cpu_author_default | null | null | null |
| News | gpu_author_default | null | null | null |
| News | native_selected | null | null | null |

Native attempts: `{"native_timeout": 1, "ok": 6, "timeout": 17}`. Physical samples: `{"ok": 12}`. All 108 logical cells: `{"fit_unavailable": 84, "ok": 24}`.

GPU default and native-selected California cells share unchanged artifacts/samples; aliases do not count as extra fits or repetitions. Failed defaults and unavailable native selections remain visible.

- Original MIT Python ForestDiffusion generator; study wrapper implements author ML objective.
- CPU author default trial0 and GPU author default trial6 remain separate resource profiles.
- Six CPU plus two GPU native trials per dataset, failures counted; no common-KPI selection.
- California GPU default and native-selected tracks alias the same trial6 artifact and sample receipts.
- Adult/News timeouts are executable-method failures, not source-unavailable exclusions or DOPE wins.
- All artifact bytes charged; eligible ForestDiffusion artifacts exceed the 10240-byte L3 cap.
- This single-fit validation pilot is not five-fit stability, public-core paired superiority or certification.
- Operation times exclude common controller admission/hash/startup overhead; complete historical campaign R&D accounting remains unavailable.
- Official tests sealed; privacy attacks, projection-only utility cost and full campaign coverage remain incomplete.

## Matched California comparison

Same fit seed23 and three sample seeds on identical projected official-training-derived inputs. All DOPE research profiles are retained; no winning family is selected here. Reference compute is already reported in the earlier panels. The CSV retains all three auditors and all datasets.

| Method | Configuration | n CatBoost retention | 4n CatBoost retention | Charged bytes |
|---|---|---:|---:|---:|
| CTGAN | default_and_tuned | 0.665194 | 0.674769 | 941787 |
| DOPE | features12_steps2048 | 0.682756 | 0.684870 | 2772 |
| DOPE | features12_steps512 | 0.694349 | 0.692840 | 2758 |
| DOPE | features24_steps2048 | 0.682756 | 0.684870 | 2772 |
| DOPE | features24_steps512 | 0.694349 | 0.692840 | 2758 |
| DOPE | q8_selected | 0.700830 | 0.694773 | 2525 |
| TVAE | default | 0.875007 | 0.875598 | 373361 |
| TVAE | tuned | 0.915410 | 0.916260 | 340593 |

- Same projected official-training-derived inputs, fit seed23, sample seeds101/211/307 and n/4n sizes.
- All four DOPE research profiles retained; no family selected by this comparison.
- Earlier CTGAN/TVAE seed23 metrics retain prior_immutable_receipt_verified status; no fresh replay claim.
- Default and native-selected baseline configurations frozen before common evaluation; native values never ranked across methods.
- Previously reported reference compute is not counted again as new fits or new sample operations.
