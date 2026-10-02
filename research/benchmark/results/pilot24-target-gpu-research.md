# Bounded DOPE GPU target research

Training-derived validation only. Four fixed profiles, fit seeds11/23, sample seeds101/211/307, three shared auditors at n/4n. Official tests remain sealed; MFS-v2/PTF-v1 and release safety are null.

## DOPE profiles

| Dataset | Profile | CatBoost n | CatBoost 4n | Charged bytes |
|---|---|---:|---:|---:|
| Adult | features12_steps512 | 0.8113 | 0.8204 | 6515–6517 |
| Adult | features12_steps2048 | 0.8443 | 0.8617 | 6527–6541 |
| Adult | features24_steps512 | 0.8004 | 0.8166 | 7143–7147 |
| Adult | features24_steps2048 | 0.8263 | 0.8436 | 7160–7182 |
| California | features12_steps512 | 0.6952 | 0.6905 | 2758–2764 |
| California | features12_steps2048 | 0.6815 | 0.6879 | 2772–2785 |
| California | features24_steps512 | 0.6952 | 0.6905 | 2758–2764 |
| California | features24_steps2048 | 0.6815 | 0.6879 | 2772–2785 |
| News | features12_steps512 | -14.3533 | -14.4945 | 9247–9249 |
| News | features12_steps2048 | -0.0037 | -0.0938 | 9250–9252 |
| News | features24_steps512 | -2.3804 | -2.3807 | 9848–9862 |
| News | features24_steps2048 | -20.7561 | -20.6427 | 9844–9867 |

## Matched author-native/default references

These references use the identical Adult/California workers and matching fit seeds11/23. Their configs retain the author-native selections from the complete five-fit panel; the shared KPI did not tune them. This extraction does not replace the original five-fit analysis.

| Dataset | Method | Configuration | CatBoost n | CatBoost 4n |
|---|---|---|---:|---:|
| Adult | CTGAN | default | null | null |
| Adult | CTGAN | tuned | -0.0573 | -0.0569 |
| Adult | DOPE | q8_selected | 0.8253 | 0.8298 |
| Adult | TVAE | default_and_tuned | 0.8031 | 0.7972 |
| California | CTGAN | default_and_tuned | 0.6204 | 0.6279 |
| California | DOPE | q8_selected | 0.7048 | 0.6996 |
| California | TVAE | default | 0.8735 | 0.8782 |
| California | TVAE | tuned | 0.9080 | 0.9091 |

## Accounting

All24GPUfits accounted: {"charged_artifact_cap": 8, "ok": 16}. GPU fit elapsed 356.943s; dispatch process time including preparation 1771.493s. Peak device use 616MiB. Architecture research cost is separate from final per-cell parity; historical R&D cost is incomplete.

Raw validation identities: {"fit_unavailable": 48, "ok": 96}. Separate packed News identities: {"ok": 48}. All eight original raw byte failures are retained; lossless packing changes no learned bytes or samples.

Generated fixture launch failure and unlaunched validation-source repair are preserved. All cells retain immutable receipt/source hashes, costs, three-auditor losses, copy/near and real-vs-real controls. The GPU research used per-job free-RAM admission; complete aggregate reservation enforcement was introduced later and is not established for this round.

Four bounded research profiles, two fit seeds and n/4n validation; no final global family or production configuration selected.

All eight original News raw projection byte failures remain visible. Packed artifacts preserve exact learned bytes and reconstruct the full projection; byte eligibility does not establish release safety.

Adult/California references use the same workers, fit seeds11/23 and three sample seeds. CTGAN/TVAE author-default/native-selected configurations are already frozen; no shared KPI enters comparator selection. News has no matched two-fit neural reference in this panel.

The reference includes complete five-fit results; this panel extracts the two matching fit seeds and does not replace their five-fit estimates. Failed executable cells contribute no DOPE win.

GPU host runtime pins cover Torch/NumPy code and shared objects, not a complete final environment closure. Metric sources/versions and dependency files are frozen separately.

New GPU research cells have exact sample and metric replays. Prior CTGAN/TVAE fit-23 metrics retain their prior_immutable_receipt_verified label; no new exact replay is claimed for those reference metrics. Copy/near screens and real-vs-real controls are retained. Complete privacy attacks, projection-only utility cost and public-core/product coverage remain missing.

Earlier architecture research and baseline native tuning costs remain reported in their separate immutable panels; total historical R&D cost is incomplete. No equal-total-R&D-spend or paired superiority claim.

Native baseline author reference: `xu2019modeling` in the corrected local bibliography.
