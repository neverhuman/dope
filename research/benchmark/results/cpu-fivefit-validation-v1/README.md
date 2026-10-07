# CPU replication validation: first complete sample batch

The published 24 ARF fits and eight Forest-Flow fits now have all six n/4n × sample seeds 101/211/307 measurements each: **192/192 cells**. These are additional independent generator fits on the same frozen official-training-derived splits as the earlier panels. Official tests remain sealed.

ARF reports default and previously native-selected configurations at fit seed 23 on 12 lineages; selection still uses held-out FORDE density. Forest-Flow reports author defaults at fit seed 23/37/53/71 on two lineages. No shared metric selects a model. This small, fixed prefix does not complete the population or five-fit matrix. Forest seed 11 and native-selected replication remain incomplete.

## Descriptive 4n results

Sample seeds are averaged within each dataset and generator fit; the table reports dataset medians separately for every fit seed. The two methods have different cohort sizes here, so these rows are not a matched comparison or a ranking.

| Method/configuration | Fit seed | Lineages | Marginal KS/TV error | CatBoost retention | CatBoost C2ST AUC | Distance MIA AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARF/author_default | 23 | 12 | 0.171151 | 0.899571 | 0.632193 | 0.589264 |
| ARF/native_selected | 23 | 12 | 0.173747 | 0.896557 | 0.594494 | 0.575534 |
| Forest-Flow/author_default | 23 | 2 | 0.221281 | 1.037085 | 0.540278 | 0.932099 |
| Forest-Flow/author_default | 37 | 2 | 0.225642 | 1.036828 | 0.577500 | 0.938272 |
| Forest-Flow/author_default | 53 | 2 | 0.222173 | 1.042115 | 0.642163 | 0.946502 |
| Forest-Flow/author_default | 71 | 2 | 0.221273 | 1.046336 | 0.588254 | 0.946502 |

The committed JSON retains per-cell fidelity, alpha-precision/beta-recall, five-fold CatBoost/logistic detection, DCR/NNDR and distance-based/KDE membership attacks, plus CatBoost/linear/MLP utility. Uninformative utility and unavailable metrics remain null. Empirical attacks establish no formal privacy guarantee.

## Replay and costs

`manifest.json` pins the producer, schema, panels, KPI CSVs, original metric input matrices, receipt locks, actual exit records and the earlier per-fit panels. Models, samples and restricted logs stay on their declared scratch hosts. Forest original paths refer to the remote producer host; transported metadata/CSV mirrors preserve original hashes. The first Forest sampling attempt stopped on a metadata field typo before any sample was generated; its failed log and lock remain retained beside the corrected immutable attempt.

Run `python3 -m research.benchmark.publish_retained_metrics --summarize-by-fit-seed` with the input refs in each method’s manifest entry and its scratch `publication-batch1-v1/receipt-mirror` directory. The source change adds separate summaries per generator fit seed; the default single-fit behavior remains unchanged. Read both the original metric and sampling locks when tracing a fit to samples. The summed per-cell numerical evaluation times are 71.113627 seconds for ARF and 19.243407 seconds for Forest-Flow, matching the pinned execution records. Total evaluator-process wall times, including runtime verification and startup, are 120.600844 and 26.747209 seconds respectively; `manifest.json` directly pins both process exit records and reports these two clocks separately. Both processes exited0. Peak evaluator RSS was 362315776 B and 298844160 B respectively. Fit and projection-inclusive bytes remain in the previous fit publication. The clocks describe overlapping work and must not be added.

Real-vs-real controls, projection-only utility cost and full attack/certification coverage are not established by this batch. MFS-v2, PTF-v1, release-safe L3 and superiority are null. No production certification, full-five-fit uncertainty or full-population ranking is asserted.
