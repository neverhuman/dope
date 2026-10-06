# Hostile review

Review of the journal manuscript on `paper-ieee-w1`. Each item is marked
after the text change that answers it. A mark of "out of scope" means this
lane does not own the missing experiment.

## Novelty

The generator is a small residual network with a refit linear readout. That
architecture is not, by itself, a new synthesis model. The manuscript's
claim is a measured comparison of validation retention and charged bytes
under a 10,240-byte cap.

Resolved in the title, the abstract, and the conclusion: the paper does not
claim a new generative family, a DOPE win, or a certification score.

## Baselines

CTGAN, TVAE, and a six-lineage Forest-Flow block are not the strongest
published tabular generators, and they were not tuned under one shared
budget. Gaussian copula is present. SMOTE is not. ARF's watch log is not a
retention panel.

Resolved as far as this lane can go. `BASELINE_GAP.md` names the missing
cells. The manuscript says those methods are not measured. Running TabSyn,
TabDDPM, CTAB-GAN+, or a 100-lineage Forest-Flow panel is out of scope: the
dope screen owns compute, and this lane does not launch fits.

## Statistics

A reviewer will ask for paired tests, a multiplicity correction, an effect
size, and an interval on the median difference. The linear auditor is the
comparison that does not separate DOPE from the Gaussian copula. Pooling
the neural and Forest-Flow blocks into the 97-lineage median would be a
sample-size error.

Resolved. `compute_panel.py` emits lineage-bootstrap intervals, two-sided
Wilcoxon tests, one-sided sign tests, Holm adjustments inside each family,
matched-pairs rank-biserial correlations, and a Friedman test with a Nemenyi
critical difference on the four density methods. The text states the linear
non-separation and forbids pooling.

## Leakage

Any use of the official test in training, selection, or the reported
retention would invalidate the panel.

Resolved for the records this script reads. Forest-Flow `fit.json` files
have `official_tests_opened` false. The BeyondArena prepare summary has
`official_test_opened_by_optimizer` false. The provenance table is file-name
counts. The verifier does not open test files. A content-level audit of
every worker remains out of scope for this writing lane.

## Reproducibility

Hand-typed medians and a Type 3 font will fail a careful reader and PDF
eXpress.

Resolved for the numbers the verifier covers. `just paper` rewrites the
evidence file, the macros, the figures, and the PDF. `verify_paper_numbers.py`
exits nonzero on a mismatched macro or an untraced token in `dope-mfs.tex`.
Forest-Flow bytes and `fit_seconds` come from hash-checked `fit.json` files.
The shared-evaluator seconds in the confirmation JSON are not that sum, and
the cost table no longer reports them. Replay medians come from receipt
`elapsed_seconds`. Loss point medians come from the logged TSV files.
Bootstrap bands are produced by `render_figures.py` with a fixed seed; the
verifier checks the point medians, not every stride of the band.

Out of scope: a second machine without `/mnt/fast-scratch/dope-benchmark`
and the ARF watch log cannot rerun the verifier. The committed
`fit-trace.json` is the reduction those files produced.

## Over-claiming

A small p-value is not MFS-v2, not PTF-v1, and not a release decision.
Forest-Flow matching DOPE on six lineages is not a 100-lineage result.
The ARF sealer's exit code is not a utility score.

Resolved in the claims block, the threats section, and the ARF sentence,
which quotes the watch log's last count rather than calling the closure
complete.

## Form

The previous 14-page draft contradicted itself on the training curve, used
Type 3 fonts, and left the 100-row tables in the main file.

Resolved. Section order is related work, method, setup, results,
discussion, threats to validity, and conclusion. The reproducibility
appendix follows the bibliography. The 100-row tables are in the
supplement. The logged-replay figure replaces the claim that the scored
campaign has no training curve.

`dope-mfs.pdf` is 8 letter pages. `supplement.pdf` is 6 letter pages.
The third-pass logs have no overfull box, underfull box, undefined
reference, or float-too-large warning. `check_paper.py` reads
`/tmp/dope-mfs-3.log` and `/tmp/supplement-3.log` and fails if any of
those lines are present. `research/benchmark/tests/test_paper_log.py`
checks that scanner. The three data figures stay full width because each
one holds several panels; the pipeline diagram is a single column. Fonts
are embedded TeX Gyre Termes, CID Type 0C, not Type 3.
