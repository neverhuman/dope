# Changelog

## Unreleased

- Block B: TabSyn's scaled arm (200 VAE and at most 1,000 diffusion
  epochs, which the authors' early stop can end sooner) is fit at seeds 23,
  37, 53, and 71 under its seed-11 caps. It is scored by the pinned evaluator,
  which matches bit for bit across xbabe1 and xbabe3 on 200 seeded cells, and
  it enters the v2 panel as a five-seed arm. The paper's TabSyn rows,
  intervals, bytes, and seed text follow the panel.
- The pre-declared all-five-fits lineage sensitivity (`v2_all_five.py`) is
  printed for every five-seed arm and DOPE-minus contrast, outside every
  Holm family; CI recomputes it from the pinned ledger.

- Paper v2 (wave 2): a pre-declared v2 analysis contract
  (`research/benchmark/review_fixes/predeclare_v2.json`) committed before any v2
  result; DOPE refit at five fit seeds with the historical binary plus an
  untuned-default arm; a binary-by-seed diagnostic; nested cluster, lineage,
  and fit-seed bootstrap with Hodges-Lehmann contrasts and a robust
  equivalence rule; the full-panel TabSyn comparison, both controls, and the
  privacy panel in the main paper; a utility-per-byte figure; a rewritten
  manuscript and supplement; a README benchmark generated from the same
  ledgers; and guards against orphan tables, figure-text drift, and
  denominator mismatches. On the validation panel the five-seed CatBoost
  retention at 4n is 0.898, against 0.949 for the earlier fit-seed-11
  population: both binaries write identical artifacts, and seed 11, the seed at
  which the profile was chosen, ranks first of five on 47 of 99 lineages.
  The v2 and retained-evidence reducers refuse any undeclared method or
  configuration before grouping, and the statistics text states that the
  Wilcoxon p-values treat lineages as independent. The supplement prints the
  first version's headline contrasts as a labeled traceability table.
  Captions and text now state the loss space, every interval type, and each
  inclusion rule where its count is printed, and the abstract prints the
  interval of every contrast it reports.
  The supplement RMSE table states original target units, as its producer
  computes them, and the compute-cost and publication-row reducers read every
  input through a declared length and digest before decoding it.
  The supplement repeats the source-family sensitivity on the five-seed
  panel (`v2_family.py`): excluding the four simulated families and within
  each family, with fit seeds resampled inside each lineage, as rows outside
  every Holm family; CI recomputes the family file from the pinned ledger.

- Add tiered compact kernels, measured release gates, and versioned MFS-v2
  evidence on the single PR branch. No production release is declared.
