# Baseline gap

This memo is the cell list for methods that are not in the manuscript's
measured blocks. It does not admit a campaign. The dope screen owns compute.
Do not create an output directory from this file.

## What the manuscript already measures

Numbers below are the generated macros, traced by
`research/benchmark/verify_paper_numbers.py`.

- Displayed DOPE is `features12_steps2048`: at most 12 inputs, width 16,
  tanh, refit readout, AdamW 2,048 steps, learning rate `2e-3`, clip 5.
  Official tests were not opened. MFS-v2, PTF-v1, and release-safe L3 are
  null. Paired retention tests are descriptive. They are not a production
  superiority claim.
- Density block, size 4, own selection objective for each method: DOPE,
  native-selected Gaussian copula (Copulas 0.14.1), study Chow--Liu (8 bins,
  Laplace 1.0), study independent marginals. CatBoost, linear, and MLP
  auditors. The linear auditor does not separate DOPE from the Gaussian
  copula.
- Neural block, not pooled: CTGAN and TVAE on the lineages whose
  native-selected size-4 group is complete. Unfinished scheduling cutoffs
  are not method failures. No TabDDPM or TabSyn number is filled in.
- Forest-Flow confirmation, not pooled: six native-selected size-4 lineages.
  Charged bytes in the validation summary match hash-checked `fit.json`
  files (115,202,253 through 1,085,155,062). Fit time in the paper is the
  sum of `fit_seconds` over the twelve confirmation `fit.json` files. The
  confirmation JSON's shared-evaluator seconds are not that sum and are not
  reported.
- ARF is not a retention result. The committed receipt
  `research/benchmark/results/arf-native-closure-watch-v1.receipt.json`
  records 799 closed and 799 ok out of 800 planned, then exit 0.
  That is not a scored utility panel.

## Missing strong methods

Each row is absent from the measured blocks. A blank utility cell in the
paper stays the words "not measured".

| Method | Why it is missing | Cells required before a number can be added |
| --- | --- | --- |
| TabDDPM | No 100-lineage native panel on this split | 100 prepared S3 lineages; sizes n and 4n; fit seed 11; sample seeds 101, 211, 307; author objective; validation rows only; charged bytes; official test sealed |
| TabSyn | Same | Same matrix as TabDDPM, author score-based objective, not a DOPE width |
| CTAB-GAN+ | Source and rights receipt is the unavailable record | Do not backfill. A cell exists only after a rights-cleared native fit on the same 100 lineages |
| STaSy, CoDi, TabDiff | Not fitted here | Same 100-lineage matrix and the author's own selection objective |
| GReaT, REaLTabFormer, TabPFGen | Not fitted here | Same matrix. Language-model and prior-fitted generators need their own decoding budget written down before a comparison |
| ARF retention | Density closure was watched; this schedule was not sampled for retention | After the native artifacts are sampled on seeds 101, 211, 307 at n and 4n, score retention on validation only. Do not treat the watch count as utility |
| Forest-Flow, full panel | The paper's block is six lineages | The remaining prepared lineages, same seeds and sizes, author regression objective, bytes from `fit.json`. Do not pool those rows with the density median |
| SMOTE | Not fitted. It is a floor, not a generative model of the joint table | One interpolating baseline on the same regression lineages and auditors, or an explicit statement that the task is not a classification SMOTE task |
| Equal tuning budget | Each method kept its own selector | A written trial cap, the same wall clock, validation-only selection, and no test-split reads. The current paper does not claim that budget |

Gaussian copula is already in the density block. It is not a missing floor.

## What this paper is allowed to say

DOPE's CatBoost median on the density block is higher than the three
density comparators under the descriptive tests, and the lineage record
against the Gaussian copula is not a sweep. The linear auditor does not
separate DOPE from that copula. Forest-Flow on six lineages is ahead of
DOPE on CatBoost and on the MLP. The byte charge is about five orders of
magnitude larger. CTGAN and TVAE have lower medians in their own block, and
DOPE still loses lineages against TVAE. Successful size-4 CTGAN and TVAE
artifacts do not sit under the 10,240-byte cap. None of those sentences is
an MFS-v2 score. The counts are the generated wins/ties/losses macros.
