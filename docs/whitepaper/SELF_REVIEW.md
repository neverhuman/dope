# TKDE reviewer report

Manuscript: *Regression Utility of a 10,240-Byte Tabular Generator*.
Class: IEEEtran `journal`, letter paper. Byline in the journal source:
The DOPE Project. This report reads the source and the third-pass PDFs
on branch `paper-tkde-review`.

Measured build: `dope-mfs.pdf` and `dope-mfs-anonymous.pdf` are 9 letter
pages. `supplement.pdf` is 8 letter pages. The third-pass logs
`/tmp/dope-mfs-3.log`, `/tmp/dope-mfs-anonymous-3.log`, and
`/tmp/supplement-3.log` contain no Warning, undefined reference,
Overfull, or Underfull line. `check_paper.py` passed.
`pdffonts` reports zero Type 3 fonts in those three PDFs and in
`figures/retention-bytes.pdf`, `figures/paired-cdf.pdf`, and
`figures/loss-curves.pdf`. Text is embedded Nimbus Roman; figure text
is embedded Liberation Serif.

## Summary

The paper measures one compact residual, `features12_steps2048`, on
paired PMLB regression lineages. Retention is the share of
real-versus-null validation improvement recovered by an auditor trained
on synthetic rows. Charged bytes are the other reported column. On 97
CatBoost lineages the DOPE median is 0.940 against 0.656, 0.586, and
−0.012 for a Gaussian copula, a study Chow–Liu tree, and study
independent marginals. Holm-adjusted Wilcoxon tests separate those
three CatBoost comparisons. The linear auditor does not
(`HolmLinGauss` 0.373). A 21-lineage neural block and a 6-lineage
Forest-Flow block stay unpooled. Forest-Flow leads that block on
CatBoost (1/0/5) and the MLP (0/0/4). The text calls those counts a
lead inside the block and refuses a certified win. Ninety-eight of 100
displayed artifacts sit inside the 10,240-byte cap. MFS-v2, PTF-v1, and
release-safe certification stay null. Official tests stay sealed.

A TKDE reader looking for a comparison against current tabular
generators will not find TabSyn, a 100-lineage TabDDPM publication, or
a full Forest-Flow publication. Those cells print the words "not
measured", taken from a receipt reader that has no file to read. The
same words cover the production privacy-attack panel, fit-seed
variance, and the full ablation grid. The prose now says this in the
abstract, the contribution list, and the conclusion. That honesty
makes the draft a scoped measurement note. It does not make the missing
experiments optional for a claim of standing against those generators.

## Major issues

1. The state-of-the-art baseline panel is absent. TabSyn, the
   100-lineage TabDDPM publication, and the full Forest-Flow
   publication are `\TabSynResult`, `\TabDDPMResult`, and
   `\ForestPublication`. Each expands to "not measured" until a
   matched-population receipt in the ARF publication schema is on
   disk. Table `tab:publication` repeats that sentence as a row. The
   cost rows `\TabSynPanel` and `\TabDDPMPanel` are a different claim
   and also stay "not measured"; a later utility receipt must not be
   copied into them. The measured subsets are real and narrower:
   Table `tab:density` (copula, Chow–Liu, independent marginals),
   Table `tab:neural` (CTGAN, TVAE), and Table `tab:forest` (six
   Forest-Flow lineages). Related work cites STaSy, CoDi, TabDiff,
   CTAB-GAN+, GReaT, REaLTabFormer, and TabPFGen and states that they
   were not run. An equal tuning budget is not measured.

2. The production privacy-attack panel is `\PrivacyAttackPanel`
   ("not measured"). Table `tab:fidelity` reports grouped-validation
   fidelity and an empirical distance ledger, including DOMIAS. The
   caption and the threats section say this is not differential
   privacy and not HIPAA de-identification. One density DCR contrast
   against independent marginals clears Holm with a sign that is worse
   for DOPE. That ledger is not a membership-inference certification
   and does not fill `\PrivacyAttackPanel`.

3. Fit-seed variance and the ablation grid are open experiments.
   `\FitSeedVariance` and `\FullAblationGrid` are "not measured".
   Section Ablations records an 8,192-step sufficiency budget on the
   same displayed residual. It is not a width-and-step grid. Threats
   states the limit directly: coresets, fit seed 11 only, a
   training-derived cut of the official training rows, and a scored
   panel that is regression only. Other tasks are not a result of
   these tables. Generalization past the recorded split seed is
   `\FitSeedVariance`.

These three are blocked on the benchmark lane that owns the fits. This
writing pass does not invent the numbers and does not launch the jobs.

## Minor issues

1. `DOI_CHECK.md` (2026-10-06) lists 50 DOIs. The bibliography that
   this PDF cites has 46 entries and 36 unique DOIs. All 36 appear in
   that log with status 200. Fourteen log rows belong to entries that
   were later dropped. The log was left as a historical record. A
   fresh resolution pass was not run for this review.

2. On this branch the supplement inventory caption still contains the
   integer 142 (`generated/beyondarena-inventory.tex`). The journal
   sentence uses `\BeyondInventory`. The caption trace that removes
   the typed integer is PR #144 and is not part of this diff.

3. The journal grew from 8 letter pages to 9 when the publication
   table and the scoped sentences were added. `IEEE_READINESS.md`
   budgets about 14 pages excluding the supplement. IEEEtran journal
   mode sets no conference page cap. The supplement is 8 letter pages.

## Recommendation

Major revision. The regression panel that was actually run is careful:
intervals, paired Wilcoxon tests, Holm correction inside each family,
rank-biserial correlations, a Friedman–Nemenyi diagram on the common
density set, and a generator that rewrites every measured number from
committed receipts. The abstract, the contribution list, and the
conclusion now name each unmeasured cell instead of letting a reader
infer a comparison. Acceptance still requires either the missing
publications or a venue decision that this scoped regression note is
the paper. I would not accept it as a comparison against TabSyn,
TabDDPM, or a full Forest-Flow publication.

## Verdict map

Each item from the original verdict has one status.

| Item | Status | Where it stands |
| --- | --- | --- |
| SOTA baselines | blocked-on-B-lane | `\TabSynResult`, `\TabDDPMResult`, `\ForestPublication`. Cost rows `\TabSynPanel` and `\TabDDPMPanel` stay "not measured" as well. Measured subsets: `tab:density`, `tab:neural`, `tab:forest`. |
| Fidelity, utility, privacy, and detection metrics | blocked-on-B-lane | `\PrivacyAttackPanel`. Utility is the retention tables. Grouped-validation fidelity and empirical DOMIAS/DCR are `tab:fidelity`. Those rows do not fill the attack panel and are not DP or HIPAA. |
| CIs, paired tests, Holm, effect sizes | closed | Results, `tab:headline`, `tab:density`, `fig:pairs`. Rank-biserial correlation is the effect size. Families are not pooled. Linear Holm against the copula is 0.373 and the interval contains 0. |
| Number regeneration | closed | `just paper-check`, `verify_paper_numbers.py`, and `scripts/publication_rows.py`. Table `tab:publication` is generated. A missing receipt prints "not measured". A receipt that fails the ARF publication schema fails the build. |
| Split provenance and leakage | closed | Method and `tab:provenance`. The official test is not an input to training, selection, or the reported numbers. Opening that seal is the decision row below. |
| Ablations | blocked-on-B-lane | `\FullAblationGrid` and `\FitSeedVariance`. Section Ablations is the 8,192-step sufficiency budget only. |
| Figures | closed | `fig:arch`, `fig:bytes`, `fig:pairs`, and supplement `figures/loss-curves.pdf`. Figure width is 7.16 in. Type 3 count is 0. |
| IEEEtran and zero warnings | closed | `\documentclass[journal]{IEEEtran}`. Third-pass logs cited above. Journal 9 letter pages, supplement 8. |
| References and DOIs | closed | 46 cited keys, 46 bibliography entries, no missing key. 36 DOIs, each logged as HTTP 200 in `DOI_CHECK.md`. Minor issue 1 records the 14 dropped-key rows still in that log. |
| Threats to validity | closed | `sec:threats`. Regression only, fit seed 11 only, coresets, and the training-derived cut are stated in the internal paragraph. |
| Claims versus evidence | closed | Abstract, contribution items 4 and 5, and the conclusion each name the unmeasured macros. No DOPE win is claimed. Forest-Flow's lead is the six-lineage W/T/L, with a non-significant Holm test. |
| BeyondArena | closed | BeyondArena subsection and `tab:beyond`. Revision `2ecfe882ccfb814fc27c4de10a64ceefd5d7655c`. No retention column. Not pooled with the density block. |
| Venue | blocked-on-decision | Journal class is set. Which IEEE journal receives the PDF is an author decision. |
| Byline | blocked-on-decision | Journal source stays The DOPE Project. `dope-mfs-anonymous.pdf` rewrites the byline and the running header only. |
| Sealed official test | blocked-on-decision | The receipt flag stays closed. Scoring the official test is not a writing change. |

No item is open-W. The experiments that would change the recommendation are the B-lane receipts and the three decisions above.

## IEEE figure pass

Read on the merged PDFs at `2016fb8`, then checked again after the axis-label edit. Grayscale renders of `fig:arch`, `fig:bytes`, `fig:pairs`, and the supplement loss curves keep each series distinct. The pipeline is black rules on a white ground at column width. The retention figure uses circles, squares, triangles, and diamonds, and the loss figure uses a solid line and a dashed line. Type 3 count remains 0. Fonts in the span figures are set so 8.1 pt stays at least 8 pt at the 7.16 in placement.

The paired-difference axis previously said "DOPE minus comparator" and did not name the unit. It now says "retention difference (dimensionless)", and the vertical axis says "sorted lineage index". The density section names the four marks. The supplement names the solid and dashed loss lines. The numbers printed in the paired-figure titles match `density-table.tex`: 0.224 [0.149, 0.290], 0.337 [0.282, 0.451], and 0.956 [0.918, 0.980]. `just paper-check` on `2016fb8` passed before this edit. Venue, byline, and the sealed test are unchanged.
