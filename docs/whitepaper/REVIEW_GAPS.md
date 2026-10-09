# Deep-review gap ledger (paper-review-fixes)

An external adversarial review of `dope-mfs.pdf` at b774807 found the gaps below. Each row closes with a commit or with a stated reason
why it cannot be done. If a new result weakens a claim, the paper weakens the claim.

| ID | Gap | Fix / acceptance | Status |
|---|---|---|---|
| CM-01 | "0 of 388 lineages": 388 counts method×lineage records | Report method–lineage cells; assert n_records = methods × lineages | closed in a4be7ed: scalar text says 0 of 388 method–lineage cells (97 lineages × 4 methods) |
| CM-02 | The abstract refers to "that interval" but never gives one | State the CI inline | closed: abstract states DiffLinGauss with LoDiffLinGauss to HiDiffLinGauss |
| CM-03 | "not measured and not measured" | Distinct macro wording; lint for repeated adjacent phrases | wording closed; measured costs still open under PR-08 |
| CM-04 | Sketch width 10+64×12+6 = 784 ≠ 856 | Document the 72-wide distribution block; test the sum | closed: text is 10+12×6+64×12+6=856; rust test documented_sketch_decomposition_matches_emitted_width in a4be7ed |
| CM-05 | Contribution bullets point at the wrong figures; bullet 1 has no verb | Repoint; rewrite | closed |
| CM-06 | "Each method selected on its own objective" overstates | Per-method selection table | closed: tab:selection |
| CM-07 | Table 5 Holm family undocumented | Declare family_size per table | closed: sec:stats and captions state the family; density-table footnote in a4be7ed is 3 tests per auditor |
| CM-08 | Pre-registered size-n endpoint not reported | Report size n next to 4n | emitted in 3c63eec: `generated/review-size-retention.tex` and `generated/review-paired.tex` list size n beside size 4n. Manuscript placement is Lane A |
| CM-09 | Endpoint is validation; official test sealed | One sealed run of the frozen config | in progress, wave 2. The sealed run has not started. No official-test number is reported |
| CM-10 | Supplement is in IEEE style; no anonymous supplement | ICLR-style supplement + anonymous build + leak grep | open |
| CM-11 | `just paper` does not regenerate everything | Complete build graph + clean-rebuild CI job | open |
| CM-12 | Byte charge includes projection.json but the text omits it | Define bytes as model + projection | closed: encoded model file plus projection.json |
| CM-13 | MFS-v3 can exceed 100 | Clip; perfect profile = 100 test | closed: eq:mfsv3 clips to [0,100]; the main text no longer prints the historical all-ones illustration. Native v3 uses max(c, epsilon) and the clip at f6665d9. The v2 path remains eq:mfs |
| CM-14 | Negative bytes pass the tier gate | 0 < int bytes ≤ 10240; negative tests | closed in a4be7ed: integer bytes with 0 < x ≤ 10240 |
| CM-15 | v3 scorer drops the v2 prerequisites | Port the checks | closed in a4be7ed: four prerequisites are required before a v3 scalar |
| CM-16 | Matched null: band in the contract, median-of-3 in code | Make code and contract agree | closed: contract and panel use median_linear_mse_of_three (a4be7ed); manuscript describes that null. No learnability band |
| CM-17 | Near-copy floor: quantile in the contract, minimum in code | Make code and contract agree | closed: q=0.01 linear squared-L2 nearest-distinct-row floor (a4be7ed); manuscript states that floor |
| CM-18 | Table 5 reducer overwrites duplicate keys | Fail closed | open |
| CM-19 | compute_panel inputs not SHA-pinned | Pinned loader | open |
| CM-20 | C2ST n=98 vs 97; BeyondArena 12 vs 142 | Explicit denominators | closed: fidelity n is the complete three-seed group; BeyondFamilies is the fit check and BeyondInventory is the inventory |
| CM-21 | Related-work taxonomy mixes families and evaluation | Restructure | closed: generators by family, then what was measured, then utility and privacy |
| PR-05 | Size n + real-bootstrap ×4 control | Tables + claims state the size | emitted in 3c63eec and 1a20875. The CatBoost real-bootstrap contrast favors the real-row resample over DOPE. Lane A places the tables and keeps that result |
| PR-06 | Predictor-only control | Control in all retention tables; paired CI | emitted in 1a20875 as `generated/review-controls.tex`. Lane A places the paired intervals |
| PR-07 | One fit seed; no clustering; no final test | ≥5 seeds, hierarchical bootstrap, family sensitivity, one sealed run | family bootstrap and excluding-simulated sensitivity emitted in 3c63eec. Extra DOPE fit seeds and the sealed run are in progress, wave 2, with no number |
| PR-08 | TabSyn on 8 lineages; TabDDPM not run | All 97 lineages + ledger | in progress, wave 2. Units `dope-rf-tabsyn-sample` and `dope-rf-tabddpm-v5` are still running. No median is reported. A failed or missing cell is not a win |
| PR-09 | No privacy/fidelity panel | DCR/NNDR/MIA with CI; fidelity; C2ST | holdout DCR, NNDR, distance MIA, and CatBoost C2ST emitted as `generated/review-privacy.tex`. Forest-Flow `sample_csv_absent` cells are not wins. Empirical only: not formal DP and not HIPAA |
| PR-10 | "Ties" read off non-significant tests | Pre-declared TOST; delta-method variance | tie means a zero difference; no equivalence claimed; delta-method appendix in. TOST table emitted in 3c63eec: the pre-declared margin passes for no published comparator |
| PR-11 | No stress suite or ablations | XOR/parity/product; k, blocks, budget | not started. No number |
| PR-12 | Credentialed data path; profile default | PMLB fetch path; no default profile; artifact statement | open |
| PR-13 | Novelty claimed on the metric; teaser shows marginal medians | Credit TSTR/TRTR; paired-difference teaser | closed: Esteban et al. TSTR/TRTR cited; claim is the encoding and the release gate; teaser draws the DOPE generator minus author-default ARF, the Gaussian copula, and Chow--Liu with stored family-cluster intervals |
| RS-01 | ARF missing from abstract, Fig. 1, and the strongest-baseline sentence | Put the generated paired ARF difference in those places | closed: abstract, the within-block headline sentence, and Fig. 1 use author-default size-$4n$ family-cluster ARF from the pinned review panel. The teaser draws that series beside the copula and Chow--Liu |
| RS-02 | Tables 12–14 lack a DOPE row on the same cohort | Matched-cohort DOPE row from lane B | in progress, wave 2. Those tables are the TabSyn comparisons. The TabSyn unit is still running, so no matched-cohort number is reported. Privacy matched-density rows are a separate cohort |
| RS-03 | C2ST classifier differs by method | One classifier and one n for every method | CatBoost C2ST is the detector in `generated/review-privacy.tex` for every scored method. Lineage counts differ where a sample or a finite triple is absent. Forest-Flow is not scored. The stored logistic table is a different detector |
| RS-04 | DOPE called a measurement in the abstract and a generator in the figures | One definition | closed: title, abstract, and captions name DOPE as the encoding and the generator it writes; retention is the measurement. Legend source files outside this lane still print the generator label |
| 11c | Negation density above 5 per 1,000 words | Keep every caveat and drop spaced " not " | closed in prose: local `pdftotext -layout` of `target/paper-build/dope-mfs.pdf` has 23 spaced " not " in 13705 words (1.678 per 1,000). The utility-score sentence keeps one required "not". Generated cost phrases still print "not measured" |

## Wave 1 / wave 2

Wave 1 keeps every finished generated table. The sentences below are the list for the PR body and for Limitations. Nothing in the wave-2 list has a printed number.

Finished, with the number only in the generated file named here:

- Size n beside size 4n, paired differences, TOST, family and excluding-simulated sensitivity, denominators, RMSE, and the stored logistic fidelity table are in the `review-*.tex` files from 3c63eec.
- Predictor-only and real-bootstrap controls are in `generated/review-controls.tex` from 1a20875. The CatBoost contrast favors the real-row resample over DOPE. That result stands.
- Size-n Holm uses three tests. Real-bootstrap has no size-n arm. The predeclare text says six tests per size. `review-controls.tex` records that deviation. Lane A owns the manuscript sentence.
- Holdout DCR, NNDR, distance MIA, and CatBoost C2ST are in `generated/review-privacy.tex`. `\input{generated/review-privacy.tex}`. Forest-Flow cells whose reason is `sample_csv_absent` are not wins. The metrics are empirical. They are not formal differential privacy and they are not HIPAA de-identification.

In progress, wave 2. No number:

- TabDDPM population v5, unit `dope-rf-tabddpm-v5`, still running on xbabe3 at the pre-registered config. A failed or missing cell is not a win.
- TabSyn sampling and scoring, unit `dope-rf-tabsyn-sample`, still running on xbabe1. The direct-model fallback stores the worker directory name as a dataset id, so a later median has to be restricted to the predeclared lineages before it can be reported.
- DOPE fit seeds beyond published fit seed 11. The features12_steps2048 trainer has no seed argument yet. Lane C owns `rust/`.
- The sealed official-test run. It has not started. A start marker now refuses any later invocation, so an interrupted attempt stays incomplete.

PR-11, the XOR/parity/product stress suite, was not started.
