# TabDDPM matched receipts and remaining fit seeds

This reduction adds no fits, samples, or auditor evaluations. It joins the
existing 132 logical TabDDPM evaluations (126 physical receipts, 21 retained
models) to ARF and TabSyn on identical training-derived validation inputs,
fit seed 11, sample seeds 101/211/307 and sizes n/4n. Original publications
are authenticated before JSON decoding. Every paired observation retains
both original metric receipts and both publication digests/JSON pointers.
Current model weights and runtime environments are not recertified.

## Measured comparisons

`matched-summary.md` has the descriptive paired dataset medians and paired
differences for utility, marginal fidelity, detection and empirical privacy.
Three sample seeds are averaged inside each lineage. Only complete groups
with both finite outcomes contribute. Missing sample groups are never averaged.
`panel.json#/matched_cells` traces every constituent metric to its receipt.
Default/native ARF overlaps have 10/12 lineages at both sizes. TabSyn overlaps
have 10/11 lineages at n and eight at 4n. Cohorts are kept separate.

The shared evaluator source is frozen at A952; detection uses five grouped
folds and splitter seed 1729. Identical *fold membership* across generators is
not certified: their synthetic rows and duplicate groups differ. Real-data
auditor losses and null controls match exactly on every joined sample.
This is matched validation evidence, not independent official-test evidence.

TabDDPM configurations remain the original author default and native-selected
snapshots. Native selection used its own five-synthetic-seed CatBoost
validation R². ARF uses its own density objective. TabSyn is a scaled author
default, not native-tuned. Nine of the 12 TabDDPM native-selected lineages have
only 2–4 successful receipts out of their five declared configurations; three
have all five. This reduction does not select again, certify equal tuning
spend, or rank different native KPIs. Every retained TabDDPM model is above
the L3 byte cap. There is no DOPE win, confidence interval or superiority claim.

## Exact gaps and conditional cell ETAs

`cell-coverage.csv` lists all 6,000 **logical comparison slots**, with lineage,
selection, fit seed, sample seed, size, and receipt pointer for measured cells.
Only fit seed 11 has measurements (132); seeds 23/37/53/71 each have zero.
The remaining 5,868 slots are pending, not method failures or zero-valued KPIs.
They cannot finish tonight under the current CPU-only order.

Each missing slot has a numeric capacity-conditional ETA, assuming one admitted
GPU slot starts Oct 8 at 08:00 MDT and progresses continuously in lineage order.
The conservative allowance reserves at most eight total native trials per
lineage at 600 seconds, skips the ten already measured default fits, and
allows four further native-selected seeds per lineage at 600 seconds.
Historical trials count toward the eight-trial/12-hour cell cap: this grants
no additional trials and historical spend must be reconciled before admission.
Native seed11 measurements can arise from that native search; they are not
charged as another fit in the envelope. The envelope is 16,900 fit/tuning
minutes plus a provisional, unmeasured six-hour evaluation allowance:
287.67 GPU-slot hours, ending Oct 20 at 07:40 MDT under that hypothetical start.

The ledger is not an executable frozen job matrix. Missing configurations,
author-native winners, available capacity and selection accounting still need
admission. No GPU is granted or used by this publication. These dates are
planning bounds rather than observed runtimes or commitments; the ETA columns
never enter a scientific KPI. All current receipts, failed attempts and their
costs stay retained. No existing job is duplicated or reclassified.

## Reproduction

```sh
python3 -B -m research.benchmark.publish_tabddpm_matched_coverage --check
python3 -B -m unittest research.benchmark.tests.test_tabddpm_matched_coverage -v
```

Official tests remain sealed. MFS-v2, PTF-v1, release-safe L3 and superiority
are null. Full-population and five-fit TabDDPM coverage remain incomplete.
Empirical attack AUCs are not formal DP or a HIPAA claim. Raw rows, models,
samples and detailed logs remain restricted; only scalar reductions and
rights-safe references are published.
