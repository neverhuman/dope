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
| CM-08 | Pre-registered size-n endpoint not reported | Report size n next to 4n | open: lane B, PR-05 |
| CM-09 | Endpoint is validation; official test sealed | One sealed run of the frozen config | deviation stated in Limitations; sealed run is lane B, PR-07 |
| CM-10 | Supplement is in IEEE style; no anonymous supplement | ICLR-style supplement + anonymous build + leak grep | open |
| CM-11 | `just paper` does not regenerate everything | Complete build graph + clean-rebuild CI job | open |
| CM-12 | Byte charge includes projection.json but the text omits it | Define bytes as model + projection | closed: encoded model file plus projection.json |
| CM-13 | MFS-v3 can exceed 100 | Clip; perfect profile = 100 test | prose closed: eq:mfsv3 clips to [0,100] and all-ones is 100; Python scorer clips in a4be7ed. Native rust/fitness.rs still adds epsilon (parity open) |
| CM-14 | Negative bytes pass the tier gate | 0 < int bytes ≤ 10240; negative tests | closed in a4be7ed: integer bytes with 0 < x ≤ 10240 |
| CM-15 | v3 scorer drops the v2 prerequisites | Port the checks | closed in a4be7ed: four prerequisites are required before a v3 scalar |
| CM-16 | Matched null: band in the contract, median-of-3 in code | Make code and contract agree | closed: contract and panel use median_linear_mse_of_three (a4be7ed); manuscript describes that null. No learnability band |
| CM-17 | Near-copy floor: quantile in the contract, minimum in code | Make code and contract agree | closed: q=0.01 linear squared-L2 nearest-distinct-row floor (a4be7ed); manuscript states that floor |
| CM-18 | Table 5 reducer overwrites duplicate keys | Fail closed | open |
| CM-19 | compute_panel inputs not SHA-pinned | Pinned loader | open |
| CM-20 | C2ST n=98 vs 97; BeyondArena 12 vs 142 | Explicit denominators | closed: fidelity n is the complete three-seed group; BeyondFamilies is the fit check and BeyondInventory is the inventory |
| CM-21 | Related-work taxonomy mixes families and evaluation | Restructure | closed: generators by family, then what was measured, then utility and privacy |
| PR-05 | Size n + real-bootstrap ×4 control | Tables + claims state the size | open |
| PR-06 | Predictor-only control | Control in all retention tables; paired CI | open |
| PR-07 | One fit seed; no clustering; no final test | ≥5 seeds, hierarchical bootstrap, family sensitivity, one sealed run | open |
| PR-08 | TabSyn on 8 lineages; TabDDPM not run | All 97 lineages + ledger | open |
| PR-09 | No privacy/fidelity panel | DCR/NNDR/MIA with CI; fidelity; C2ST | open |
| PR-10 | "Ties" read off non-significant tests | Pre-declared TOST; delta-method variance | tie means a zero difference; no equivalence claimed; delta-method appendix in; TOST numbers are lane B |
| PR-11 | No stress suite or ablations | XOR/parity/product; k, blocks, budget | open |
| PR-12 | Credentialed data path; profile default | PMLB fetch path; no default profile; artifact statement | open |
| PR-13 | Novelty claimed on the metric; teaser shows marginal medians | Credit TSTR/TRTR; paired-difference teaser | closed: Esteban et al. TSTR/TRTR cited; claim is the encoding and the release gate; teaser is paired differences |
| RS-01 | ARF missing from abstract, Fig. 1, and the strongest-baseline sentence | Put the generated paired ARF difference in those places | open: waiting on lane B generated paired series. The teaser still draws Chow--Liu, the Gaussian copula, and independent marginals |
| RS-02 | Tables 12–14 lack a DOPE row on the same cohort | Matched-cohort DOPE row from lane B | open: waiting on lane B |
| RS-03 | C2ST classifier differs by method | One classifier and one n for every method | open: lane B rerun; manuscript text waits on that output |
| RS-04 | DOPE called a measurement in the abstract and a generator in the figures | One definition | closed in prose: DOPE is the encoding and the generator it writes; retention is the measurement column; legends label that generator DOPE |
