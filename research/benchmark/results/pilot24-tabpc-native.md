# TabPC author-native pilot and matched DOPE validation

Complete frozen pilot scope; fit seed23, sample seeds101/211/307. Eight TabPC trials per dataset selected only by author transformed validation mean NLL (minimize). Native NLL values are not ranked against other methods. Official tests remain sealed; MFS-v2/PTF-v1 and release-safe status remain null.

## Common CatBoost retention

| Dataset | Method | Configuration | n | 4n | Charged bytes |
|---|---|---|---:|---:|---:|
| Adult | DOPE | q8_symbolic | 0.8275 | 0.8293 | 6337 |
| Adult | TabPC | default | unavailable | unavailable | unavailable |
| Adult | TabPC | tuned | unavailable | unavailable | unavailable |
| California | DOPE | q8_symbolic | 0.7008 | 0.6948 | 2525 |
| California | TabPC | default | 0.9620 | 0.9670 | 707633 |
| California | TabPC | tuned | 0.9247 | 0.9327 | 16882952 |
| News | DOPE | q10_packed | 0.3287 | 0.3109 | 9205 |
| News | TabPC | default | 0.5105 | 0.6937 | 5785356 |
| News | TabPC | tuned | 0.4220 | 0.6675 | 138577828 |

Group medians require all three informative sample-seed results. The JSON retains every sample, all three auditor losses/retentions, marginal/joint/C2ST screens, exact/near-copy counts and real-vs-real controls. The CSV exports n/4n cell status, bytes and three auditor retentions. Artifact byte eligibility is separate from release safety; all successful TabPC artifacts exceed 10,240 bytes.

## Native validation NLL

| Dataset | Default NLL | Selected NLL | Selected trial |
|---|---:|---:|---:|
| Adult | unavailable | unavailable | unavailable |
| California | 8.9907 | 8.1292 | 6 |
| News | 31.9499 | 9.6899 | 4 |

## Complete failures, sources and costs

Native fit statuses: `{"failed": 8, "ok": 13, "timeout": 3}` across24 attempts. Sample statuses: `{"ok": 45, "sample_timeout": 3}` across48 jobs. All24 planned n/4n metric cells succeeded; the three News tuned8n sample timeouts remain explicit. Adult default and tuned are unavailable after eight native GPU allocation failures. Executable failures confer no DOPE win.

Summed native attempt time: 4441.38s. Common sampling/evaluation/replay elapsed time: 8841.70s. No new common-validation fits or tuning trials. Other DOPE and comparator R&D is reported separately, not claimed equal.

Source/runtime commits, license evidence, compatibility changes and charged fit/sample contracts are in [TABPC.md](../TABPC.md). Original training artifacts and receipts are preserved; V5 changes only integrity/error handling and metadata binding. Bulk data, weights and detailed logs remain on scratch.

This is a complete single-fit-seed pilot panel; final five-fit campaign locks, public-core paired analysis, projection-only utility, full privacy attacks and production coverage remain incomplete. Official tests remain sealed and every gated score is null.
