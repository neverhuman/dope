# IEEE review readiness

## Retained-evidence update, 2026-10-07

The merged retained publications are now integrated in the manuscript:
TabSyn eight-lineage/48-evaluation scaled-default cohort; ARF/copula/Chow-Liu
100-lineage default/native n/4n diagnostics and utility; and an authenticated
first12 diagnostic appendix. `scripts/retained_evidence.py` regenerates
lineage intervals, exploratory paired statistics, figure data and the
coverage table with conditional ETAs. Receipts remain validation-only;
full-population neural coverage, final five-fit comparisons and release
certification remain incomplete. No GPU use is authorized by this writing
update. The earlier readiness audit below is a historical snapshot; its
TabSyn/source-only and ARF/watch-only statements are superseded by the
current coverage table and `RETAINED_EVIDENCE_REVIEW.md`.

Reconciled 2026-10-05 with the local audit `.agent/PAPER_IEEE_READINESS.md`
(that file stays uncommitted). The PR 124 manuscript was IEEEtran `journal`,
14 letter pages, at `52b187b`. This note lives on `paper-ieee-w1`. Lane W
owns the paper, the statistics below, and the figure scripts. Lane B owns
adapters, harness, campaign rounds, and BeyondArena preparation. Lane W
does not dispatch those runs.

Status on this branch: `dope-mfs.tex` is retitled, the abstract carries
lineage-bootstrap intervals, and every measured number is a macro in
`generated/numbers.tex`. `python3 docs/whitepaper/scripts/compute_panel.py`
rebuilds those macros from the committed ledgers. `just paper` also redraws
the figures and both PDFs when the local replay scratch is mounted.
`just paper-check` is the CI diff. The three figures are Type 42 TeX Gyre
Termes at 7.16 in. Forest-Flow is its own six-lineage block. The built
journal and anonymous PDFs are 13 letter pages. The supplement is 9 pages and holds the
100-row tables. TabSyn, 100-lineage TabDDPM, privacy attacks, fit-seed
variance, and the full ablation grid stay the words "not measured."
The third-pass build log is clean of overfull boxes, underfull boxes,
undefined references, and floats that are too large. `DOI_CHECK.md` records
the doi.org resolution. `BASELINE_GAP.md` names the cells that were not run.

## Verdict

A TKDE-style reviewer would reject the PR 124 PDF for missing strong
generators under an equal tuning budget, for a utility-only metric, for
medians without intervals or paired tests, for nine references and no
related-work section, for a title that names a null scalar, and for Type 3
fonts. Official-test isolation holds on the receipts checked below. Venue
and authorship are closed by Jepson for this pass.

Jepson decisions, applied here:

- Class stays `\documentclass[journal]{IEEEtran}`. Conference mode is not the target.
- Length budget is about 14 pages excluding the supplement. The 100-row lineage tables move to a supplementary PDF.
- Author block is Jepson Taylor and Alton Alexander, NEVERHUMAN Research. The anonymous PDF withholds that byline.

## Gap list

| # | Reviewer question | Evidence on `52b187b` | Owner | This pass |
| --- | --- | --- | --- | --- |
| 1 | IEEEtran conference mode and a proceedings page cap | `journal`, letter, 14 pages. IEEEtran does not set a page cap. A six-page conference cap is a different venue. | Jepson | Journal class. Measured PDF is 13 letter pages, including the appendix after the bibliography. Supplement is 9 pages. |
| 2 | Strongest tabular generators, each with the same documented tuning budget | Executed on the 100-lineage density panel: DOPE `features12_steps2048` (one pre-registered profile), Copulas 0.14.1 Gaussian copula, study Chow–Liu, study independent marginals. Separate neural block: native-selected CTGAN and TVAE. Forest-Flow has a 6-lineage confirmation ledger. TabSyn is source-audit only. TabDDPM is pilot-locked, not on the 100. The ARF watch log is not a retention panel. `methods.lock.json` has `complete: false` and `frozen_for_final_evaluation: false`. | B for new fits; W for the Forest-Flow block already measured | Top scientific gap. Fair-tuning plan is below and is not dispatched. The PDF contains the 6-lineage Forest-Flow block. `BASELINE_GAP.md` lists the missing cells. The manuscript says those methods are not measured. |
| 3 | Did an optimizer or a selector see an official test? | Protocol text says seed 1729, grouped 80/20 of official training rows, workers have no official test file, final evaluation unauthorized. Ledger field `official_tests_opened` is present and false on the lineage record. The proof table is regenerated into `generated/provenance.json` from listings and locks. | W writes the paragraph and table; B must not open tests to fill baselines | Pass, once the proof table is in the paper |
| 4 | Significance tests and confidence intervals | Paper says the pre-registered paired test has not been run. No interval on any median. Figures have no CI. | W, from committed paired records, no new fits | Emitted by `compute_panel.py`: lineage-bootstrap intervals, two-sided Wilcoxon, one-sided sign test, Holm within each family, matched-pairs rank-biserial, Friedman and Nemenyi on the density methods. The linear auditor does not separate DOPE from the Gaussian copula. |
| 5 | Ablations | Six-lineage width/step check only. The 8,192-step profile is a sufficiency budget on the same partitions, not the displayed model. | B for a full grid; W reports the six-lineage check and the 2,048-vs-8,192 sufficiency medians with intervals | Placeholder macro for the full grid |
| 6 | Reproducibility appendix and a rebuild | Ledger section points at files. Numbers in the tex are hand-typed. No `just paper`. | W | `generated/numbers.tex`, `just paper`, and a reproducibility appendix after the bibliography. `verify_paper_numbers.py` exits nonzero on a mismatched macro or an untraced token. |
| 7 | Related work | Methods are cited in the introduction. `references.bib` has nine entries. | W | Related Work section. `DOI_CHECK.md` records that every DOI returned citation JSON from doi.org. |
| 8 | Threats to validity | Section “What this panel does not establish” is a limitations paragraph. | W | Threats to Validity, with internal, external, construct, and statistical paragraphs. |
| 9 | Figures usable at IEEE width | `pdffonts` on `retention-bytes.pdf`, `s3-paired-differences.pdf`, `s3-loss-steps2048.pdf`, and `s3-retention-bytes-catboost.pdf` shows embedded Type 3 DejaVuSans. Paired-difference page size is 801.91 by 1110.35 pt. Loss y-axis says “hidden-basis loss” with a 10th–90th band, which is not a CI. Colors are already Okabe–Ito. | W | Three Type 42 figures, serif, 3.5 in or 7.16 in, text at least 8 pt, lineage-bootstrap intervals |
| 10 | Title and abstract sell MFS-v2 | Title is “A 10,240-Byte Generator, Scored After It Is Selected.” MFS-v2 is null on every row. Abstract is a six-decimal ledger. Line 118 says the paper does not show a training curve; Section VIII shows two. | W | Retitle around retention per byte. Abstract ≤250 words with intervals. Delete the contradiction. |
| 11 | Fidelity, detection, privacy, fit-seed variance | Density cells already store `marginal_ks_mean`, `pair_correlation_fidelity`, and `c2st_auc`. DOMIAS, distance-to-closest-record, and alpha-precision are not in that ledger. One fit seed (11). | W may summarize fields that already exist. B owns new attacks and extra fit seeds. | Existing fields get intervals. Missing attacks stay the placeholder “not measured.” Empirical privacy is not formal DP or HIPAA. |

Displayed generator remains `features12_steps2048`: at most 12 inputs, hidden width 16, tanh, refit linear readout, AdamW 2,048 steps, learning rate 0.002, no embedding. The 8,192-step replay and the width-8 six-lineage check do not replace it. Density, neural, Forest-Flow, and BeyondArena stay unpooled. An unexecuted method is not a DOPE win. A failed byte gate is not a retention of zero. MFS-v2, PTF-v1, and release-safe L3 stay null.

## Fair-tuning plan (top gap, not dispatched)

Lane B runs this. Lane W does not create the directory, does not write requests, and does not start fits. Output root, when B starts it:

`/mnt/fast-scratch/dope-benchmark/ieee-fair-tuning-v1/`

Forbidden write targets, including for that future run:

- `/mnt/fast-scratch/dope-benchmark/arf-s3-population-sampling-v1`
- `/mnt/fast-scratch/dope-benchmark/arf-s3-population-sampling-preparation-v1`
- any other `arf-s3-population-sampling*` directory
- any Codex output directory under the scratch root

Rules from `research/benchmark/PROTOCOL.md` and `methods.lock.json`:

- `methods.lock.json` `complete` is false. `admission.py` fails closed until the source, data, budget, matrix, and evaluator locks are complete. A command listed below is the lock’s recorded string, not an admission to score the 100.
- Two tracks stay separate: pinned defaults, and equal-budget tuning. The paper must show both once they exist.
- Budget for every method–dataset pair: eight tuning trials, 12-hour wall-time cap, every trial and timeout retained. Selection uses validation rows only. Official test files stay unread.
- Fit seeds for a full job are 11, 23, 37, 53, 71. Sampling seeds are 101, 211, 307. Sizes `n` and `4n` are the required cells. The published paper measured fit seed 11 only.
- DOPE’s displayed profile stays the pre-registered `features12_steps2048`. Extra DOPE trials are a sensitivity table. They must not replace the displayed profile after retention is seen.
- Per-fit ceiling already used on this host: 600 seconds and 16 GiB, inside the scratch ceiling of 200,000,000,000 bytes.
- Study-owned Chow–Liu and independent marginals stay labeled study code.
- Native objectives are not ranked across methods. The shared columns are retention and charged bytes.

Recorded commands (placeholders `WORKER`, `TASK`, `SEED`, `ARTIFACT`, `ROWS`, `OUTPUT`, `VERIFIED_REQUEST`, `RECEIPT`, `DATASET`, `GPU` are the lock’s own tokens):

| Method | Lock status | Fit command as recorded | Tuning record |
| --- | --- | --- | --- |
| dope | locked | `dope-kernel compile --dataset-dir WORKER --task TASK --tier l3 --seed SEED --deadline-seconds 600 --out ARTIFACT/model.dpk` | search space `candidate: [default]` only |
| GaussianCopula | locked | `research.benchmark.adapters.fit(GaussianCopula, TRAIN, METADATA, CONFIG, FIT_SEED, ARTIFACT)` | distribution in `{Univariate, GaussianUnivariate}` |
| Chow-Liu | locked, study code | `research.benchmark.adapters.fit(Chow-Liu, TRAIN, METADATA, CONFIG, FIT_SEED, ARTIFACT)` | bins `{8,16,32}`, Laplace alpha `{0.5,1.0}` |
| independent_marginals | locked, study code | `python -m research.benchmark.runner JOB METHODS OUTPUT_ROOT` | bins `{8,16,32}` |
| CTGAN | locked, campaign unadmitted | `research.benchmark.adapter_worker: fit CTGAN` | batch size `{500,100}`, embedding dim `{128,64}` |
| TVAE | locked, campaign unadmitted | `research.benchmark.adapter_worker: fit TVAE` | same family as CTGAN |
| Forest-Diffusion | pilot_locked, five-lock admission pending | `python -m research.benchmark.forestdiffusion_adapter fit --request VERIFIED_REQUEST --receipt RECEIPT` | `combined_trial_cap` 8 |
| TabDDPM | pilot_locked, grid `pending_before_tuning` | `/mnt/fast-scratch/dope-benchmark/envs/tabddpm-py39/bin/python3.9 -S /mnt/fast-scratch/dope-benchmark/tabddpm-contract-v3/source/entry.py fit VERIFIED_REQUEST` | author grid recorded, no tuned config selected |
| TabSyn | source_audit_pending, no campaign admission | `python main.py --dataname DATASET --method vae --mode train; python main.py --dataname DATASET --method tabsyn --mode train` | `tuning_search_space` null. Author sampler ignores `args.steps`. |
| TabDiff | source_audit_pending | `python main.py --dataname DATASET --method tabdiff --mode train --gpu GPU --deterministic --no_wandb` | `tuning_search_space` null |
| ARF | pilot_locked | `fit_command` null, `sampling_command` null | eight-trial cap recorded; no generic command to run |
| AIM | pilot_locked | `research.benchmark.adapters.fit(AIM, TRAIN, METADATA, CONFIG, FIT_SEED, ARTIFACT)` | pilot public set, not the 100 |
| synthpop CART | pilot_locked | R script `research/benchmark/synthpop_cart.R` | four frozen trials; not on the 100 |

`VERIFIED_REQUEST` is not a built request for the 100-lineage equal-budget matrix. Building it is lane B.

Newer generators cited as scope, not as executed baselines: CDTD (ICLR 2025), GATD (ICML 2026) if the PMLR record verifies, and the MIND preprint. Kumo’s tabular model is a predictor, not a generator, and is out of the baseline list. GReaT, REaLTabFormer, and TabPFGen stay out until B admits a subset or the threats section scopes them out. PrivBayes is `unavailable` on license grounds in the lock.

What “done” means for this gap: each admitted method has a default row and an eight-trial validation-selected row, the same wall clock, the same fit seeds, retention and charged bytes on the same 100 lineages, and a written timeout ledger. Until then the comparison does not support a claim that DOPE beats TabSyn, TabDDPM, or Forest-Diffusion on the 100.

## Statistics lane W will compute

No new fits. Sources: `s3-lineage-record.json`, `density-matched-population-validation.json`, `sdv-matched-population-validation.json`, `s3-matched-forest-confirmation-validation.json`, and the loss replay TSVs under `/mnt/fast-scratch/dope-benchmark/dope-s3-loss-log-v1/` (read only).

- Lineage bootstrap, 10,000 resamples, seed 20261005, percentile 95% interval on every median. The three sample seeds stay inside the lineage median. They are not a confidence interval.
- Two-sided paired Wilcoxon, DOPE minus each comparator, per auditor, with Holm adjustment inside one block. Density, neural, and Forest-Flow are separate families.
- Matched-pairs rank-biserial correlation.
- Friedman test and Nemenyi critical difference on the density methods’ common lineage set. Demšar’s q for four methods at alpha 0.05 is 2.569.
- The protocol’s sign test and dataset-cluster bootstrap, reported as validation descriptions. They do not flip PTF-v1, MFS-v2, or release-safe L3.
- Informativeness-threshold sweep at 0, 0.001, 0.01, 0.05, and 0.1, using `null_loss`, `trtr_loss`, and `tstr_loss` already stored in the density ledger. The published column stays the 0.01 rule.

## Figures

Replace the seven current figure files with three, plus a TikZ architecture diagram in the tex:

1. `figures/retention-bytes.pdf` — 7.16 in, three auditor panels, size `4n`, median marker with a lineage-bootstrap interval. No per-point whiskers.
2. `figures/paired-cdf.pdf` — 7.16 in, sorted dots of paired CatBoost differences (no dataset-name axis) and the critical-difference diagram.
3. `figures/loss-curves.pdf` — both step budgets as panels, median and bootstrap band, ylabel “hidden-basis MSE”, 8,192 marked as not the displayed model.

Font setup: `pdf.fonttype` 42, `ps.fonttype` 42, TeX Gyre Termes. Okabe–Ito plus distinct markers. `pdffonts` must show no Type 3 font.

## Placeholders left for lane B

The generator emits these as the words “not measured,” not as numbers: TabSyn on the 100, TabDDPM on the 100, a privacy-attack panel, fit-seed variance, the full DOPE ablation grid, DOMIAS, and alpha-precision / beta-recall. BeyondArena retention stays absent until B runs the auditors. Seven of twelve locked families failed to prepare; that shortfall stays printed.
