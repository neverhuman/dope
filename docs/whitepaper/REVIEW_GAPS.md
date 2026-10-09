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
| CM-08 | Pre-registered size-n endpoint not reported | Report size n next to 4n | closed: the main table inputs `generated/review-wave2-size-retention.tex` at size n and size 4n, including TabSyn and same-cohort DOPE rows. Author-default and native-selected ARF remain separate configurations |
| CM-09 | Endpoint is validation; official test sealed | One sealed run of the frozen config | in progress, wave 2. The sealed run has not started. No official-test number is reported |
| CM-10 | Supplement is in IEEE style; no anonymous supplement | ICLR-style supplement + anonymous build + leak grep | closed: ICLR-style supplement, anonymous source build, and leak checks are part of the paper gate. Hosted clean-rebuild passed at 2bf3578; the final wave-2 head needs its own green run |
| CM-11 | `just paper` does not regenerate everything | Complete build graph + clean-rebuild CI job | closed: the complete authenticated scalar graph precedes rendering and source PDF builds. Hosted clean-rebuild passed at 2bf3578; final wave-2 CI is recorded separately |
| CM-12 | Byte charge includes projection.json but the text omits it | Define bytes as model + projection | closed: encoded model file plus projection.json |
| CM-13 | MFS-v3 can exceed 100 | Clip; perfect profile = 100 test | closed: eq:mfsv3 clips to [0,100]; the main text no longer prints the historical all-ones illustration. Native v3 uses max(c, epsilon) and the clip at f6665d9. The v2 path remains eq:mfs |
| CM-14 | Negative bytes pass the tier gate | 0 < int bytes ≤ 10240; negative tests | closed in a4be7ed: integer bytes with 0 < x ≤ 10240 |
| CM-15 | v3 scorer drops the v2 prerequisites | Port the checks | closed in a4be7ed: four prerequisites are required before a v3 scalar |
| CM-16 | Matched null: band in the contract, median-of-3 in code | Make code and contract agree | closed: contract and panel use median_linear_mse_of_three (a4be7ed); manuscript describes that null. No learnability band |
| CM-17 | Near-copy floor: quantile in the contract, minimum in code | Make code and contract agree | closed: q=0.01 linear squared-L2 nearest-distinct-row floor (a4be7ed); manuscript states that floor |
| CM-18 | Table 5 reducer overwrites duplicate keys | Fail closed | closed in a4be7ed: a duplicate method-lineage-auditor key raises |
| CM-19 | compute_panel inputs not SHA-pinned | Pinned loader | closed at d75f10f: registered inputs are SHA-pinned before decode, including the author-default ARF headline rows |
| CM-20 | C2ST n=98 vs 97; BeyondArena 12 vs 142 | Explicit denominators | closed: fidelity n is the complete three-seed group; BeyondFamilies is the fit check and BeyondInventory is the inventory |
| CM-21 | Related-work taxonomy mixes families and evaluation | Restructure | closed: generators by family, then what was measured, then utility and privacy |
| PR-05 | Size n + real-bootstrap ×4 control | Tables + claims state the size | closed: both sizes appear in the main retention table, and the real-bootstrap control is reduced from the bound 5700-cell scalar ledger in the supplement |
| PR-06 | Predictor-only control | Control in all retention tables; paired CI | closed: the supplement inputs generated/review-controls.tex from the bound scalar ledger. Every declared aggregate is reproduced before rendering; equivalence stays unclaimed |
| PR-07 | One fit seed; no clustering; no final test | ≥5 seeds, hierarchical bootstrap, family sensitivity, one sealed run | family bootstrap and excluding-simulated sensitivity emitted in 3c63eec. Extra DOPE fit seeds and the sealed run are in progress, wave 2, with no number |
| PR-08 | TabSyn on 8 lineages; TabDDPM not run | All 97 lineages + ledger | TabSyn closed: independently captured sample hashes and frozen fit receipts admit 594 scalar cells on 99 lineages, reproducing the published aggregate. Both sizes, matched DOPE rows, paired intervals, and actual auditor denominators are generated. All paired family intervals include zero; size-4n MLP favors TabSyn at the median. TabDDPM v5 remains pending with no number; the retained subset remains separate |
| PR-09 | No privacy/fidelity panel | DCR/NNDR/MIA with CI; fidelity; C2ST | partly closed: bound privacy cells reproduce the original DCR, NNDR, and membership-attack panel. Configuration-aware KS, correlation, logistic and common CatBoost detectors, and alpha/beta tables are generated from pinned ledgers. Missing Forest-Flow detector/alpha cells remain unavailable; attribute inference and predictor attacks remain pending. Empirical diagnostics stay outside formal DP and HIPAA de-identification |
| PR-10 | "Ties" read off non-significant tests | Pre-declared TOST; delta-method variance | paired-difference table placed in the supplement; every filled TOST cell is no. Equivalence stays unclaimed. The TOST-only file stays uninput because its ARF rows lack a configuration column. Delta-method appendix remains |
| PR-11 | No stress suite or ablations | XOR/parity/product; k, blocks, budget | not started. No number |
| PR-12 | Credentialed data path; profile default | PMLB fetch path; no default profile; artifact statement | closed: public credential-free PMLB fetch, no default AWS profile, artifact custody statement, and source clean-build gate |
| PR-13 | Novelty claimed on the metric; teaser shows marginal medians | Credit TSTR/TRTR; paired-difference teaser | closed: Esteban et al. TSTR/TRTR cited; claim is the encoding and the release gate. The teaser draws paired generator differences against author-default ARF, the Gaussian copula, Chow--Liu, and matched TabSyn, with family-cluster intervals and separate cohort counts |
| RS-01 | ARF missing from abstract, Fig. 1, and the strongest-baseline sentence | Put the generated paired ARF difference in those places | closed: abstract, Fig. 1, the headline sentence, and `generated/headline-table.tex` use author-default size-$4n$ family-cluster ARF from the pinned review panel. The linear interval contains zero. The teaser draws that series beside the copula and Chow--Liu |
| RS-02 | Tables 12–14 lack a DOPE row on the same cohort | Matched-cohort DOPE row from lane B | closed: both size factors show DOPE and TabSyn on each auditor’s actual informative intersection in the main table and supplement. The matched 99-lineage source cohort stays distinct from the retained subset |
| RS-03 | C2ST classifier differs by method | One classifier and one n for every method | partly closed: the shared grouped CatBoost detector is reported beside distinct historical logistic AUC cells with explicit per-metric n. TabSyn is scored by the same fixed implementation. One TabSyn CatBoost cell is unavailable; Forest-Flow remains unavailable. Unequal cohorts prevent a cross-method ranking |
| RS-04 | DOPE called a measurement in the abstract and a generator in the figures | One definition | closed: title, abstract, and captions name DOPE as the encoding and the generator it writes; retention is the measurement. Legend source files outside this lane still print the generator label |
| 11c | Negation density above 5 per 1,000 words | Keep every caveat and drop spaced " not " | closed: the wave-2 source PDF passes check_paper against the fixed maximum of 5 spaced " not " per 1,000 words |
| 11d | Main body before References over 10 pages | Move long displays into the supplement | closed: both source main PDFs have 10 pages, with References beginning on page 9. The main body passes check_paper; the large privacy and fidelity displays remain in the supplement |

## Wave 2 admission and remaining work

Every displayed result comes from a committed generated file and an authenticated
scalar source. The build authenticates the complete graph before decoding and
replays the controls, privacy, TabSyn utility, and fidelity aggregates before any
render. Hash mismatch, missing pins, duplicate cells, malformed identities, or an
opened official-test flag refuse emission.

Finished public units:

- Controls: `generated/review-controls.tex` is reproduced from the bound scalar ledger. Equivalence remains unclaimed.
- Privacy: `generated/review-privacy.tex` is reproduced from the bound public DCR/NNDR/membership cells. These are empirical diagnostics, outside formal DP and HIPAA de-identification.
- TabSyn: `generated/review-tabsyn.tex`, `review-wave2-size-retention.tex`, and the teaser use the bound full cohort and actual matched auditor intersections. All paired family intervals include zero. The retained subset remains a separate cohort.
- Fidelity: `generated/review-fidelity.tex` separates ARF configurations, historical logistic AUC, and shared CatBoost AUC; `review-alpha-beta.tex` identifies the numeric full cohort and retained mixed-type cohort. Every metric has its own denominator and family-cluster interval. Forest-Flow alpha/beta and detector cells remain unavailable.
- Custody: the frozen sample grid has six unavailable cells; six foreign `worker` operations and a legacy unavailable operation with an unknown official-test flag are quarantined in the original sampling receipt. They never enter the admitted cohort. Missing cells are excluded explicitly, with no imputed result.
- Singleton bootstrap optimization: frozen conformance fixtures preserve the original interval endpoints and RNG state. Draw count, clustering, configuration, and statistical estimands stay unchanged.

Remaining gaps, with no result number:

- Attribute-inference and shipped-predictor attacks require their own committed predeclarations before execution.
- Extra DOPE fit seeds have no admitted result in this historical-validation integration. Existing and separately declared v2 jobs remain untouched; no extra-seed median is printed.
- TabDDPM population v5 lacks an admitted finished unit for this paper. No result is printed and no existing job is touched.
- Selection and scoring still share validation rows. The sealed official-test run stays untouched; no official-test result or disjoint-selection claim is printed.
- Equal tuning budgets are unestablished by the current receipts. No budget-parity claim is printed.
- The full XOR/parity/product stress suite and a complete three-axis ablation suite remain pending. No result is imputed.

The final exact-head local and hosted gate results are recorded in the external
`WAVES_LEDGER.md` and `REVIEW_FIX_PROGRESS.md`; a historical green run never
stands in for the final head.

## Wave 3: literal rubric follow-up

The frozen-rubric review of wave 2 found three publication descriptions that
needed correction. The generated origin paragraph now distinguishes legacy
lineage intervals from the source-family bootstrap and retains the missing
parent-map limitation for other related lineages. The source-family sensitivity
and informative-denominator tables now declare their descriptive multiplicity
scope. Both document titles name encoding and generation. These changes leave
the measured results and statistical procedures unchanged.

Remaining work from that review:

- The standalone retained-evidence reducer still needs an explicit unknown-method
  refusal; the current pinned publication graph does not replace that check.
- The page-1 paired figure still omits the measured Forest-Flow block. Adding it
  requires its own short-cohort counts and the correct existing uncertainty.
- The older retained-classical and retained-eight tables still lack matched DOPE
  rows. The full numeric cohort cannot substitute for those historical cohorts.
- Fit-reference NNDR does not complete holdout-referenced NNDR. Forest-Flow's
  strong-classifier and alpha/beta cells remain unavailable.
- Fit-seed uncertainty, new attribute and shipped-predictor attacks, additional
  mechanism/stress units, equal tuning budgets and original-artifact packaging
  require admitted evidence; prospective declarations alone supply no result.

The official test and existing frozen jobs remain untouched. The external
`WAVE3_PLAN.md` orders these units by points per effort. A wave-3 rubric score
requires a new exact-head review against the unchanged frozen criteria.
