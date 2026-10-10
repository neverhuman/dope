# Changelog

## Unreleased

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

- Add tiered compact kernels, measured release gates, and versioned MFS-v2
  evidence on the single PR branch. No production release is declared.
