# Benchmark campaign status — 2026-09-30

The final public-core and matched S3 comparisons are **not admitted**. The
public and S3 test partitions have not been used for model selection or common
outcome evaluation. Production PTF-v1 and generator MFS-v2 are null. This file
records validation-only research and the remaining admission work; it is not a
release or paper result.

## Rights-cleared S3 data

The pinned catalog SHA-256 is
`ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`.
All 104 PMLB regression objects with recorded MIT evidence were fetched with
the coordinator profile and checked against catalog blob hashes. The 3,755
unknown-rights entries were excluded. Four of the 104 had an identical source
row in the official S3 training and test partitions; their exclusion receipts
are retained. The remaining 100 use grouped 80/20 fit/validation assignments
derived only from official training rows. Worker directories contain no
official test rows; raw official tests remain in the coordinator cache and
projected tests in the evaluator directories. [The S3 data lock](results/s3-data.lock.json) records
object, manifest, partition, projection, assignment, and receipt hashes, plus
all four exclusions; [its schema](results/s3-data-lock.schema.json) validates
the published form.

The count of 100 meets the candidate-lineage count threshold exactly. The
required count of at least 30 informative groups is still unmeasured under
the final profile, so no production coverage claim follows from that count.

## DOPE L3 validation research

The frozen discovery and confirmation cohorts each contain six datasets from
the 100-lineage S3 panel, stratified by training-row and projected-feature
rank. The three-host research queue recorded 60 neural fits: 18 first-pass
architecture fits, 18 micro-TVAE loss-grid refinements, and 24 disjoint
confirmation fits. Forty-four produced artifacts within the charged 10,240 B
L3 limit; all 16 failures were byte overruns. The queue used one GPU fit per
host, 16 CPU cores per fit, a 600-second neural fit deadline, and a 16-GiB VRAM
limit. Attempt receipts record host, elapsed time, peak GPU usage, approximate
power-sample energy, artifact bytes, and immutable log hashes. Summed observed
GPU fit time was about 955 seconds across three hosts. The 14-day checkpoint
remains a reporting milestone.

On the six disjoint confirmation datasets, the symbolic default had higher
single-seed validation CatBoost retention than each of the four tested
micro-TVAE target-weight/structural-loss configurations. This is a diagnostic
comparison, not the preregistered paired paper analysis. The subsequent
100-lineage symbolic validation survey completed 100 fits; 98 artifacts met
the L3 byte limit. CatBoost retention was defined for 99 lineages, with median
0.788 and maximum 1.144. Nine individual estimates were at least 0.99, but
that number is not a PTF-v1 pass rate. One survey lineage had an exact row
copy in the validation privacy screen. The full attacks, five fit seeds, three
sample seeds, required sizes, and paired lower bound remain unmeasured.

[The validation research JSON](results/gpu-validation-research.json),
[schema](results/gpu-validation-research.schema.json),
[per-cell table](results/gpu-validation-research.csv), and
[SVG](results/gpu-validation-retention.svg)/[PDF](results/gpu-validation-retention.pdf)
point to immutable scratch receipt hashes. Regenerate the S3 lock with
`python3 -m research.benchmark.publish_s3_data` and the research table and
figure with `python3 -m research.benchmark.publish_validation_research`.
Bulk inputs, weights, logs, and generated samples remain on benchmark scratch.

An initial micro-TVAE fit under 10,240 B generated 244 rows with nonfinite
features from a source with no missing values. The neural sampler now respects
observed missingness support. The original failed sample and the repaired
sample have separate scratch hashes; the repaired 800-row sample had zero
nonfinite values. A regression test covers the changed rule, including
permuted feature order.

## Compact native validation wave

A separate 24-cell discovery and confirmation matrix tuned independent
marginals and the study-owned Chow–Liu implementation on the same twelve
training-derived S3 validation partitions. The matrix was frozen on scratch
before fitting, with dataset, projection, method, trial grid, fit seed, and
12-hour per-cell budget hashes. Each method used its locked held-out mean log
density objective; these native KPI values are meaningful only within that
method's search. The 108 planned trials all completed: 36 independent-marginal
and 72 Chow–Liu trials, with no failures. Selected artifacts met the charged
10,240 B L3 limit in 11/12 and 4/12 cells, respectively. The other selected
artifacts remain visible as byte overruns; no production score or comparison
win is assigned from this validation research. These are single-fit-seed
reference results, and the Chow–Liu code is labeled as a study implementation.

[The compact validation JSON](results/compact-native-validation.json),
[schema](results/compact-native-validation.schema.json), and
[trial table](results/compact-native-validation.csv) point to the frozen
round and immutable scratch attempt and KPI receipts. Regenerate them with
`python3 -m research.benchmark.publish_compact_native`. Only the coordinator's
training-derived worker partitions were used; official S3 tests remain closed.

GaussianCopula used a separate frozen 12-cell validation matrix and its locked
Copulas 0.14.1 environment. Both author-library distribution configurations
completed on each dataset: 24 trials, no failures, and 7/12 selected artifacts
within L3. The other five exceed 10,240 B. [Its validation JSON](results/copula-native-validation.json)
and [schema](results/copula-native-validation.schema.json) carry per-trial
status, KPI, bytes, and immutable scratch receipt hashes. Regenerate with
`python3 -m research.benchmark.publish_copula`. This remains an exploratory
single-fit-seed common-numeric result and is not a production comparison.

## Final admission still blocked

The 29-method source inventory currently has four locked entries (DOPE and
three compact baselines), 22 pending source audits, one pilot-only AIM entry,
and two unavailable entries (TabKDE and GEM-T). Unavailable methods contribute
no DOPE win. The density KPI implementation now gives GaussianCopula,
independent marginals, and Chow-Liu held-out validation mean log density;
final tuned selection checks that a frozen native objective actually chose
the winning trial within the eight-trial, 12-hour cap. The remaining native
objectives, adapters, reproduction checks, DP accounting, author-faithful
track, public-core datasets, evaluator gates, budget lock, and exact final job
matrix are not complete. `admission.py` therefore keeps sealed-test common
evaluation closed. The six-fit pilot and this single-seed research survey are
cost and validation evidence only.
