# Matched fit11 linear-regression retention

Measured descriptive validation on the same official-training-derived inputs. DOPE uses fixed `features12_steps2048`; TabDDPM retains its original author-default/native-selected roles; Forest retains author-default fit11. Official tests remain sealed.

Only the shared linear **regression** auditor is compared. All 23 projection task fields are hash-authenticated as regression. The shared pilot implementation constructs `LinearRegression()` without a seed. DOPE supplied 101/211/307 and the retained evaluator supplied 1729; CatBoost and MLP consume these different seeds and are excluded. Other pilot/A952 diagnostics are not pooled.

Each method first averages its three sample-seed retentions (101/211/307) within each lineage and size. Dataset medians and paired differences use the exact same complete informative cohort on both sides. This is one fit seed, with no fit-variance or superiority claim.

| Comparator | Configuration | Size | Paired lineages | DOPE median | Comparator median | Median paired difference |
|---|---|---:|---:|---:|---:|---:|
| Forest-Flow | author_default | 1n | 11 | 0.989069 | 0.995491 | -0.004390 |
| Forest-Flow | author_default | 4n | 11 | 0.995956 | 0.995869 | -0.004079 |
| TabDDPM | author_default | 1n | 10 | 0.995912 | 0.823612 | 0.318592 |
| TabDDPM | author_default | 4n | 10 | 1.003505 | 0.854121 | 0.166282 |
| TabDDPM | native_selected | 1n | 12 | 0.995912 | 0.885817 | 0.096924 |
| TabDDPM | native_selected | 4n | 12 | 1.003505 | 0.967828 | 0.048069 |

The difference between two aggregate medians need not equal the median of paired lineage differences. For example, Forest 4n has aggregate medians 0.995956 and 0.995869 while its median paired difference is -0.004079. Pairing occurs before taking the difference median.

Cohorts differ across comparator rows. Two Forest lineages (`19f4780b53b3fa41`, `a04f964bc53281e0`) have uninformative linear real-data controls and are excluded at both sizes. Their null outcomes remain in `panel.json`; neither is treated as a win or imputed result. Forest's other fit seeds are not pooled here.

The comparison is unconstrained by the 10,240-byte stored-artifact threshold. The original [TabDDPM panel](../tabddpm-twelve-lineage-retained-validation/README.md) records all 21 retained models above that threshold. The Forest [first two](../forest-first-two-fivefit-v1/README.md), [next six](../forest-next-six-fivefit-v1/README.md) and [third five](../forest-third-five-fivefit-v1/README.md) cohorts likewise record all 65 models above it. Per-cell model-plus-projection byte charges remain in this panel. A stored-byte charge does not establish production eligibility, quality, privacy, or a DOPE win.

There are 210 logical comparator cells, 204 distinct comparator metric receipts and 138 distinct DOPE metric receipts. Default/native aliases and DOPE reuse are not independent repetitions or multiplied costs. No training/campaign cost, hardware, energy or parent clock is inferred by this panel.

The source proof binds the committed input panels, original metric references, the shared pilot and both drivers, 23 task-only projection checks, and relevant declared numerical-provider metadata. It retains all 42 additional comparator runtime references (40 native providers, one stdlib source/extension and one loader). Whole-runtime equivalence and historical runtime certification are not claimed.

`MFS-v2`, `PTF-v1`, release-safe L3 and superiority are null. No rows, headers, transforms, model weights, sample payloads or TEST values are published; no new fit, generation or evaluation was performed.

## Reproduction

Authenticate the committed JSON/schema bytes, recompute every three-sample mean and same-cohort median, and check `figure-kpis.csv` plus this README byte for byte:

```sh
python3 -B -m research.benchmark.render_matched_linear_fit11 --check
```

The command uses only the standard library and committed public files. `--write` regenerates the two rendered files. The private source receipts are needed only to reproduce the original metadata projection, not to check these committed scalar aggregates.
