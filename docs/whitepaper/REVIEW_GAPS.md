# Deep-review gap ledger

An external adversarial review of `dope-mfs.pdf` at b774807 found the gaps below, and the
later grades and audits added G-01 to G-07. Each row closes with
the change that fixed it or states what is still open and why. When a new result weakens a
claim, the paper weakens the claim; the v2 prose takes every data-dependent word from
`generated/v2-numbers.tex`, so a verdict cannot outrun its interval.

| ID | Gap | Fix / acceptance | Status |
|---|---|---|---|
| CM-01 | "0 of 388 lineages": 388 counts method×lineage records | Report method–lineage cells | closed in a4be7ed |
| CM-02 | The abstract refers to "that interval" but never gives one | State the CI inline | closed: every abstract estimate carries its interval macro |
| CM-03 | "not measured and not measured" | Distinct wording; lint for repeated phrases | closed: `paper_lint.py` refuses a repeated five-word sentence |
| CM-04 | Sketch width 10+64×12+6 = 784 ≠ 856 | Document the 72-wide distribution block; test the sum | closed in a4be7ed; the sketch now lives in supplement A |
| CM-05 | Contribution bullets point at the wrong figures | Repoint; rewrite | closed: four contributions, each with its section |
| CM-06 | "Each method selected on its own objective" overstates | Per-method selection rule | closed: `method_registry.py` holds each arm's selection rule; supplement C prints it |
| CM-07 | Holm family undocumented | Declare the family per table | closed: every caption states its Holm family or that it has none; `review_tex.longtable` refuses a caption without one |
| CM-08 | Pre-registered size-n endpoint not reported | Report size n next to 4n | closed: main Table 2 prints every arm at n and 4n |
| CM-09 | Endpoint is validation; official test sealed | One sealed run of the frozen configuration | authorized 2026-10-09 as one run after the Stage 2 freeze (`predeclare_v2.json`, `sealed_test_v2`); not run, so no official-test number is printed |
| CM-10 | Supplement style; no anonymous supplement | ICLR-style supplement and anonymous build | closed in 62f93e0 |
| CM-11 | `just paper` does not regenerate everything | Complete build graph and a clean-rebuild CI job | closed: `paper-clean` deletes `generated/`, rebuilds, and requires a zero diff; the v2 panel replays from its scalar ledger |
| CM-12 | Byte charge omits projection.json | Bytes are model plus projection | closed; the v2 fit ledger checks each model file's size |
| CM-13 | MFS-v3 can exceed 100 | Clip; perfect profile is 100 | closed |
| CM-14 | Negative bytes pass the tier gate | 0 < bytes ≤ 10240 | closed in a4be7ed |
| CM-15 | v3 scorer drops the v2 prerequisites | Port the checks | closed in a4be7ed |
| CM-16 | Matched null: band in the contract, median-of-3 in code | Agree | closed in a4be7ed |
| CM-17 | Near-copy floor: quantile vs minimum | Agree | closed in a4be7ed |
| CM-18 | Reducer overwrites duplicate keys | Fail closed | closed; every v2 binder refuses a duplicate identity, and `v2_panel.reduce` refuses a duplicate sample cell and any arm outside its declared roster before grouping |
| CM-19 | compute_panel inputs not SHA-pinned | Pinned loader | closed at d75f10f; the v2 ledger, fit ledger, panel, and byte ledgers are in the review input lock |
| CM-20 | C2ST n=98 vs 97; BeyondArena 12 vs 142 | Explicit denominators | closed; `paper_consistency.check_v2` checks every paired n against both level counts |
| CM-21 | Related-work taxonomy mixes families and evaluation | Restructure | closed: generators, compact program models, evaluation |
| G-01 | An uncaptioned MFS-v3 layer table floated into the main body | Captioned float in the supplement; lint | closed: `mfs_v3_table.py` writes a scalar note and a captioned layer table; `paper_lint.py` refuses an uncaptioned tabular |
| G-02 | Text described Fig. 2 clipping that the figure does not do | Figure windows from one spec | closed: `figure_spec.py` holds every window and emits the macros the text uses |
| G-03 | Forest-Flow informative count printed as 0 | Level n from the retention rows | closed: `denominator_view.py` prints the level n beside the complete-loss rows |
| G-04 | Owner audit 2026-10-09 23:09: byte-cap universality, iid seed p-value, missing Forest-Flow abstract contrast, retention versus privacy sample custody | Name the over-cap lineages with a sensitivity; descriptive seed count; qualified six-lineage abstract sentence; route-specific custody wording | closed: §5.4 names both over-cap lineages and gives the headline without them from the bound ledger (no rerun); the seed count carries no p-value; the abstract adds the Forest-Flow block; main §5.5 and supplement E state which runs read the Forest-Flow samples |
| G-05 | Independent frozen-rubric score of f47197b (2026-10-10): the first version's headline set H (Forest-Flow median 1.003, 1/0/5, the 2,067-byte DOPE median on those lineages, linear critical difference 0.489) was generated but no longer printed; `v2_panel.reduce` admitted undeclared arms that never reached a contrast; stale README and RS-03 notes | Print H with its historical label; validate every cell before grouping; correct the notes | closed: supplement `tab:hist-headline` and the historical-block paragraph print every H value from the committed ledgers through `verify_paper_numbers.py` macros; the roster check and its `reduce()` regression are in place; `docs/whitepaper/README.md` names the five-seed panel as the displayed result |
| G-06 | Independent frozen-rubric score of c6647d4 (2026-10-10, 75/100): main Table 3 gave no inclusion rule for its n; supplement G cited the α/β table for logistic AUC, KS, and pair fidelity and left their second interval unexplained; the byte table's stated denominator was wrong for CTGAN and TVAE; the BeyondArena medians named no quantity; the loss space was unstated and the RMSE caption said target units | State each rule and unit where the number is printed | closed: Table 3 states its sample-file rule; supplement G names the lineage-bootstrap interval and points to `tab:review-fidelity` for the family-cluster one; the byte caption gives both counting rules; the BeyondArena sentence gives elapsed seconds per profile; main §4.2 and the RMSE caption state the training-split min–max projection to [0, 1] |
| G-07 | Independent frozen-rubric score of fbf52f4 (2026-10-10, 73/100): abstract verdicts for ARF, Forest-Flow, and both controls printed without their intervals; the baseline-coverage TabSyn row read as the current panel; the architecture figure said the density baselines report no search; the first version's Holm families were declared only in captions | Print every abstract interval; label the coverage row; state the selection rule; declare the families in supplement D | closed: the abstract prints each contrast with its 95% nested interval (199 rendered words); the coverage caption names the earlier run and points to main Tables 1–2; the figure says the native density baselines select by held-out log density; supplement D names the first version's per-size families |
| PR-05 | Size n and a real-row ×4 control | Tables state the size | closed: both controls are in main Tables 1–2, the abstract, and Fig. 1 |
| PR-06 | Predictor-only control | Paired CI in every retention table | closed: five predictor fit seeds in the v2 panel; supplement F keeps the split-seed rows |
| PR-07 | One fit seed; no clustering; no final test | Five seeds, nested bootstrap, family sensitivity, sealed run | five DOPE fit seeds with the historical binary, a cluster→lineage→fit-seed bootstrap, and seed-exchangeability diagnostics are in the paper; the sealed run is CM-09 |
| PR-08 | TabSyn on 8 lineages; TabDDPM not run | Full panel and a budget ledger | TabSyn's scaled-schedule full panel is in the main tables and names its epochs. TabDDPM v5 failed (472 attempt failures, no new success), so TabDDPM v6, TabDiff, GReaT, synthpop, SMOTE, the author-default TabSyn schedule, and Forest-Flow under the common budget are Stage 2 blocks with no number |
| PR-09 | No privacy or fidelity panel | DCR/NNDR/MIA with CI; C2ST | closed for empirical black-box attacks: main Table 3 and supplement G; supplement G also gives KS, pair fidelity, the CatBoost detector, and α-precision/β-recall for every method with samples. Shadow-model, attribute-inference, and artifact attacks (P2–P4) are Stage 2 |
| PR-10 | "Ties" read off non-significant tests | Pre-declared equivalence | closed: the Hodges–Lehmann 90% rule of predeclare_v2 decides equivalence; the v1 mean TOST is a continuity table |
| PR-11 | No stress suite or ablations | XOR/parity/product; k, blocks, budget | two axes are printed (step budget and the untuned default); the grid and stress suite are Stage 2 blocks I and J |
| PR-12 | Credentialed data path | PMLB fetch path; artifact statement | closed in 62f93e0 |
| PR-13 | Novelty claimed on the metric; marginal-median teaser | Credit TSTR/TRTR; paired teaser | closed: Fig. 1 is the paired forest plot of DOPE minus every comparator and control |
| RS-01 | ARF missing from the abstract and Fig. 1 | Paired ARF difference in both | closed |
| RS-02 | TabSyn tables lack a DOPE row on the same lineages | Matched DOPE rows | closed: supplement `tab:review-tabsyn` and every v2 contrast pair on shared lineages |
| RS-03 | C2ST classifier differs by method | One classifier for every method | closed: one grouped CatBoost detector for every method with samples, TabSyn included (supplement G); Forest-Flow's stored samples feed its retention re-score, but the detector and privacy runs record its cells as unavailable, so it has no detector row |
| RS-04 | DOPE called a measurement and a generator | One definition | closed |
| 11c | Negation density above 5 per 1,000 words | Lint | closed: `check_paper.py` enforces the density on the built PDF |
| 11d | Main body over 10 pages | Page budget | closed: `check_paper.py` refuses a body over ten pages |

## Open, with no printed number

- The sealed official-test run (CM-09).
- Publication of the scored DOPE artifacts, the fit and validation partitions, and the synthetic-row bundle
  (Stage 4 of the plan, each step confirmed by the owner). Until then supplement C states that they are
  unpublished and that the committed ledgers reproduce every reduction; it gives no reason for withholding them.
- A five-seed source-family sensitivity table; the printed one is the fit-seed-11 table.
- Stage 2 blocks of `predeclare_v2.json`: TabDDPM v6, TabSyn author-default and four more TabSyn seeds,
  Forest-Flow under the 1,800-second budget, density and synthpop at five seeds, SMOTE, the SDV rerun,
  TabDiff and GReaT adapters, the ablation grid and stress suite, privacy P2–P4, and the OpenML extension
  under its Go rule. A failed or missing cell is never a DOPE win.

## Main wave-3 follow-up (53f6558), reconciled with v2

Main's wave-3 commit edited wave-2 sources that the v2 rewrite replaces. Each of its items, checked
against this tree:

| Wave-3 item | Status in v2 |
|---|---|
| Origin paragraph separates the lineage bootstrap from the family-cluster bootstrap and keeps the parent-map limitation | closed: `\OriginSentence` renders both, the cluster rule from `cluster_text.describe_clusters` |
| Wilcoxon and TOST keep independent-lineage assumptions | closed: main §4 and supplement D say the Wilcoxon, Yuen, and v1 mean tests treat lineages as independent and that every verdict reads the clustered interval |
| Source-family sensitivity and denominator tables declare their multiplicity scope | closed: the supplement keeps the historical fit-seed-11 source-family table (`tab:review-family`, `generated/review-family.tex`), captioned as a sensitivity without Holm p-values, and the denominator table has no Holm family. No five-seed source-family table exists yet |
| Both titles name the encoding and the generator | closed: both v2 titles read "Encoding of Regression Tables into Kilobyte Generators" |
| Unknown-method refusal in the retained-evidence reducer | closed: `retained_evidence.groups_from_rows` refuses any method or selection binding outside the pinned panel's seven arms; `v2_panel.reduce` checks every cell against the v2 roster before grouping, including incomplete, noninformative, and fidelity-only cells |
| Page-1 paired figure omits Forest-Flow | closed: Fig. 1 draws the six-lineage Forest-Flow contrast as a small-cohort row |
| Retained-classical and eight-lineage tables lack matched DOPE rows | open: those historical cohorts print no DOPE row; the matched DOPE contrasts for ARF, the density baselines, and TabSyn are in the v2 panel (main Table 1, supplement `tab:v2-contrasts`), a different cohort |
| Fit-referenced NNDR is not holdout-referenced NNDR | open: main Table 3 names its NNDR as fit-referenced; holdout-referenced NNDR is part of privacy P1 in Stage 2 block L |
| Forest-Flow CatBoost-detector and alpha/beta cells | open: supplement G prints them as unavailable; only the stored logistic detector, KS, and pair fidelity have Forest-Flow values (RS-03) |
| Fit-seed uncertainty, attribute and shipped-predictor attacks, stress units, equal budgets, artifact packaging | fit-seed uncertainty is closed for DOPE and ARF (five seeds, nested bootstrap); the rest are the Stage 2 blocks above and Stage 4 publication |
