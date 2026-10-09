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
| CM-08 | Pre-registered size-n endpoint not reported | Report size n next to 4n | closed in the main PDF: Table 2 inputs `generated/review-size-retention.tex` at size n and size 4n for the methods in that file. TabSyn stays outside it. Author-default ARF and native-selected ARF are separate rows |
| CM-09 | Endpoint is validation; official test sealed | One sealed run of the frozen config | in progress, wave 2. The sealed run has not started. No official-test number is reported |
| CM-10 | Supplement is in IEEE style; no anonymous supplement | ICLR-style supplement + anonymous build + leak grep | source closed in 62f93e0. The anonymous supplement PDF is in the wave-1 PDF commit. Hosted clean-rebuild is not claimed green |
| CM-11 | `just paper` does not regenerate everything | Complete build graph + clean-rebuild CI job | source closed through b589297: authenticated review outputs precede the teaser, and the clean job restores the pinned historical compute receipt. Hosted CI is not claimed green |
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
| PR-05 | Size n + real-bootstrap ×4 control | Tables + claims state the size | size n is Table 2 of the main PDF. The real-bootstrap comparison remains in progress, wave 2, and this paper prints no number for it |
| PR-06 | Predictor-only control | Control in all retention tables; paired CI | in progress, wave 2. The scalar cells are unbound, so this paper prints no number. Equivalence stays unclaimed |
| PR-07 | One fit seed; no clustering; no final test | ≥5 seeds, hierarchical bootstrap, family sensitivity, one sealed run | family bootstrap and excluding-simulated sensitivity emitted in 3c63eec. Extra DOPE fit seeds and the sealed run are in progress, wave 2, with no number |
| PR-08 | TabSyn on 8 lineages; TabDDPM not run | All 97 lineages + ledger | the matched TabSyn cohort and TabDDPM v5 remain in progress, wave 2. The retained subset stays a separate historical cohort. This paper prints no number for the matched cohort or for TabDDPM v5 |
| PR-09 | No privacy/fidelity panel | DCR/NNDR/MIA with CI; fidelity; C2ST | the historical fidelity table stays in the supplement and stays outside formal DP and outside HIPAA de-identification. The review privacy panel remains in progress, wave 2, and this paper prints no number for it. The stored logistic file stays uninput |
| PR-10 | "Ties" read off non-significant tests | Pre-declared TOST; delta-method variance | paired-difference table placed in the supplement; every filled TOST cell is no. Equivalence stays unclaimed. The TOST-only file stays uninput because its ARF rows lack a configuration column. Delta-method appendix remains |
| PR-11 | No stress suite or ablations | XOR/parity/product; k, blocks, budget | not started. No number |
| PR-12 | Credentialed data path; profile default | PMLB fetch path; no default profile; artifact statement | source closed in 62f93e0: public PMLB fetch, no default AWS profile, artifact custody stated. Hosted clean-rebuild is not claimed green |
| PR-13 | Novelty claimed on the metric; teaser shows marginal medians | Credit TSTR/TRTR; paired-difference teaser | closed: Esteban et al. TSTR/TRTR cited; claim is the encoding and the release gate; teaser draws the DOPE generator minus author-default ARF, the Gaussian copula, and Chow--Liu with stored family-cluster intervals |
| RS-01 | ARF missing from abstract, Fig. 1, and the strongest-baseline sentence | Put the generated paired ARF difference in those places | closed: abstract, Fig. 1, the headline sentence, and `generated/headline-table.tex` use author-default size-$4n$ family-cluster ARF from the pinned review panel. The linear interval contains zero. The teaser draws that series beside the copula and Chow--Liu |
| RS-02 | Tables 12–14 lack a DOPE row on the same cohort | Matched-cohort DOPE row from lane B | in progress, wave 2. The matched TabSyn aggregate has no bound scalar cells, so this paper prints no number. Distinct from the retained subset |
| RS-03 | C2ST classifier differs by method | One classifier and one n for every method | in progress, wave 2. The review privacy panel prints no number. Forest-Flow remains unscored. The stored logistic file stays uninput |
| RS-04 | DOPE called a measurement in the abstract and a generator in the figures | One definition | closed: title, abstract, and captions name DOPE as the encoding and the generator it writes; retention is the measurement. Legend source files outside this lane still print the generator label |
| 11c | Negation density above 5 per 1,000 words | Keep every caveat and drop spaced " not " | closed on the wave-1 PDF: `pdftotext -layout` of `dope-mfs.pdf` has 5 spaced " not " in 5756 words (0.869 per 1,000). The utility-score sentence keeps one required "not" |
| 11d | Main body before References over 10 pages | Move long displays into the supplement | wave-1 `dope-mfs.pdf` is 11 pages and the References heading is the first line of page 11. Body before References is 10 pages |

## Wave 1 / wave 2

Wave 1 keeps every finished and correct generated table. A stored aggregate with no bound scalar cells is not that kind of table. Nothing in the wave-2 list has a printed number.

Finished, with the number only in the generated file named here:

- Size n beside size 4n is `generated/review-size-retention.tex` in the main text. TabSyn stays outside that file.
- Paired differences are `generated/review-paired.tex`. Every filled TOST cell in that table is no. Family and excluding-simulated sensitivity are `generated/review-family.tex`. Denominators are `generated/review-denominator.tex`.
- `review-tost.tex`, `review-rmse.tex`, `review-fidelity-stored.tex`, and `review-seeds-existing.tex` stay uninput because an ARF block repeats in each without an author-default versus native-selected column.

In progress, wave 2. No number:

- Predictor-only and real-bootstrap controls. The public panel stores an aggregate and the scalar cells are unbound.
- The review privacy panel. The same aggregate gap applies. Empirical privacy stays outside formal DP and outside HIPAA de-identification, and this paper prints no privacy-panel number.
- The matched TabSyn cohort. It stays distinct from the retained subset. This paper prints no matched-cohort number.
- TabDDPM population v5, unit `dope-rf-tabddpm-v5`, still running on xbabe3 at the pre-registered config. A failed or missing cell is not a win.
- Extra DOPE fit seeds. Qualification passed. At c6f5ad4 the even lineage-record indexes had finished on xbabe1 and the odd indexes were still fitting, with `dope-rf-seeds-x3` waiting on TabDDPM v5. The hierarchical reducer is not an inclusion path for this cut. This paper prints no extra-seed number.
- The sealed official-test run has not started. An existing `started.json` is refused, so the run cannot resume. No official-test number.
- PR-11, the XOR/parity/product stress suite, was not started.
