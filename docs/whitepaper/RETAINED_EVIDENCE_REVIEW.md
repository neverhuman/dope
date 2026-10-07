# Retained-evidence red-team review, 2026-10-07

## Findings repaired

1. **Stale coverage:** TabSyn was labelled entirely unmeasured and ARF absent
   from diagnostics. The new tables use the merged eight-lineage TabSyn
   and 100-lineage classical receipts, keeping full-population neural
   coverage incomplete and separating each aggregation rule.
2. **Small-sample interval overclaim:** finite Bonferroni-tail bootstrap
   extrema on eight lineages cannot provide the advertised simultaneous
   median coverage. The tables now use conservative binomial/order-statistic
   intervals with Bonferroni correction under independent-lineage assumptions.
   All matched-eight corrected intervals are unbounded. Finite bootstrap
   ranges remain explicitly exploratory in JSON; no point estimate or p-value
   was changed. The ARF six-test family has finite corrected intervals.
3. **Blanket byte claim:** the abstract previously said the generator stays
   under the cap despite two recorded overruns. It now says it is evaluated
   against the byte budget; the original failed-cap count remains visible.
4. **Post-selection inference:** threats now state that validation also
   supplies native tuning signals, bootstrap intervals condition on the
   split/artifact, and related simulation lineages can violate independence.
5. **First-batch pooling:** the deterministic first12 is a per-physical-job
   appendix. Mixed research profiles, aliases and incomplete seed groups are
   not independent method replicates; no population CI or ranking is added.
6. **Storage and private names:** TeX build logs/auxiliaries stay in the
   checkout target directory. Manuscript/PDF scans reject private host names
   and scratch paths; original receipt paths remain in source JSON rather
   than the manuscript. No real row, header, extrema, sample or weight was read.

## Reproducible checks

- `scripts/test_retained_evidence.py` checks source-pin rejection, dataset
  pairing, three-seed completeness and missing outcomes, exact original ARF
  point estimates, small-n unbounded intervals, binomial coverage tails,
  regeneration and forecast arithmetic.
- `generated/retained-evidence.json` traces groups through source digests and
  JSON pointers to the original metric receipt references.
- `generated/baseline-coverage.json` identifies measured scope and the
  conditional 156-slot-hour forecast. Forecasts grant no GPU lease, and
  final five-fit baseline matrices remain unscheduled.
- `verify_paper_numbers.py`, `check_paper.py`, the deterministic renderer,
  the public-output leak scan and repository required gates are run before
  the writer commit; actual reports remain in the checkout target directory.

Official tests remain sealed. MFS-v2, PTF-v1, release-safe L3, formal privacy
and production-superiority claims are not emitted. A better validation median
or a small p-value cannot change those gates. The paper still lacks complete
strong-neural default/native and final five-fit matrices; this integration
alone does not make the full scientific campaign IEEE-ready.
