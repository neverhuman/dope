# Deep-review gap ledger (paper-review-fixes)

An external adversarial review of `dope-mfs.pdf` at b774807 found the gaps below. Each row closes with a commit or with a stated reason
why it cannot be done. If a new result weakens a claim, the paper weakens the claim.

| ID | Gap | Fix / acceptance | Status |
|---|---|---|---|
| CM-01 | "0 of 388 lineages": 388 counts method×lineage records | Report method–lineage cells; assert n_records = methods × lineages | open |
| CM-02 | The abstract refers to "that interval" but never gives one | State the CI inline | open |
| CM-03 | "not measured and not measured" | Distinct macro wording; lint for repeated adjacent phrases | open |
| CM-04 | Sketch width 10+64×12+6 = 784 ≠ 856 | Document the 72-wide distribution block; test the sum | open |
| CM-05 | Contribution bullets point at the wrong figures; bullet 1 has no verb | Repoint; rewrite | open |
| CM-06 | "Each method selected on its own objective" overstates | Per-method selection table | open |
| CM-07 | Table 5 Holm family undocumented | Declare family_size per table | open |
| CM-08 | Pre-registered size-n endpoint not reported | Report size n next to 4n | open |
| CM-09 | Endpoint is validation; official test sealed | One sealed run of the frozen config | open |
| CM-10 | Supplement is in IEEE style; no anonymous supplement | ICLR-style supplement + anonymous build + leak grep | open |
| CM-11 | `just paper` does not regenerate everything | Complete build graph + clean-rebuild CI job | open |
| CM-12 | Byte charge includes projection.json but the text omits it | Define bytes as model + projection | open |
| CM-13 | MFS-v3 can exceed 100 | Clip; perfect profile = 100 test | open |
| CM-14 | Negative bytes pass the tier gate | 0 < int bytes ≤ 10240; negative tests | open |
| CM-15 | v3 scorer drops the v2 prerequisites | Port the checks | open |
| CM-16 | Matched null: band in the contract, median-of-3 in code | Make code and contract agree | open |
| CM-17 | Near-copy floor: quantile in the contract, minimum in code | Make code and contract agree | open |
| CM-18 | Table 5 reducer overwrites duplicate keys | Fail closed | open |
| CM-19 | compute_panel inputs not SHA-pinned | Pinned loader | open |
| CM-20 | C2ST n=98 vs 97; BeyondArena 12 vs 142 | Explicit denominators | open |
| CM-21 | Related-work taxonomy mixes families and evaluation | Restructure | open |
| PR-05 | Size n + real-bootstrap ×4 control | Tables + claims state the size | open |
| PR-06 | Predictor-only control | Control in all retention tables; paired CI | open |
| PR-07 | One fit seed; no clustering; no final test | ≥5 seeds, hierarchical bootstrap, family sensitivity, one sealed run | open |
| PR-08 | TabSyn on 8 lineages; TabDDPM not run | All 97 lineages + ledger | open |
| PR-09 | No privacy/fidelity panel | DCR/NNDR/MIA with CI; fidelity; C2ST | open |
| PR-10 | "Ties" read off non-significant tests | Pre-declared TOST; delta-method variance | open |
| PR-11 | No stress suite or ablations | XOR/parity/product; k, blocks, budget | open |
| PR-12 | Credentialed data path; profile default | PMLB fetch path; no default profile; artifact statement | open |
| PR-13 | Novelty claimed on the metric; teaser shows marginal medians | Credit TSTR/TRTR; paired-difference teaser | open |
