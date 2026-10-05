# Benchmark campaign status — 2026-10-05

The final public-core and matched S3 comparisons are **not admitted**. The
public and S3 test partitions have not been used for model selection or common
outcome evaluation. Production PTF-v1 and generator MFS-v2 are null. This file
records validation-only research and the remaining admission work; it is not a
release or completed public-core paper comparison.

## Completed matched research benchmarks

Two complete receipt-accounting panels now compare DOPE with native-selected
methods on the same rights-cleared S3 worker views. These views are bounded
to 72–800 official-training-derived rows per lineage; they do not represent
full official training partitions or official-test results. DOPE's fixed
`features12_steps2048` reference is shown below. Every frozen DOPE profile,
baseline default, native selection and unavailable cell remains in the linked
reports. No global DOPE family has been selected.

| Native-selected comparator | Complete informative paired lineages | DOPE median retention | Comparator median retention |
| --- | ---: | ---: | ---: |
| GaussianCopula | 97 | 0.939876 | 0.655681 |
| Chow–Liu, study implementation | 97 | 0.939876 | 0.585986 |
| Independent marginals, study implementation | 97 | 0.939876 | −0.011971 |
| CTGAN | 21 | 0.953366 | −0.081259 |
| TVAE | 21 | 0.953366 | 0.669033 |

These are dataset medians of three sample-seed CatBoost null-normalized
retention values at `4n`, using fit seed 11 and training-derived validation.
Negative values are retained. The density and neural rows use different
availability subsets; their values must not be combined into one ranking.
Other auditors and `n` outcomes are in the complete reports. This descriptive
median is not the production PTF-v1 formula or a paired superiority decision.

- [Density report](results/density-matched-population-validation.json),
  [table](results/density-matched-population-validation.csv) and
  [figure](results/density-matched-population-validation.svg): all 6,000 logical
  cells accounted for, 5,904 measured and 96 DOPE cells unavailable. Each
  baseline selected its own held-out density objective.
- [CTGAN/TVAE report](results/sdv-matched-population-validation.json),
  [table](results/sdv-matched-population-validation.csv) and
  [figure](results/sdv-matched-population-validation.svg): all 4,800 logical
  cells accounted for, including 486 measured baseline cells and 1,914
  unavailable baseline cells. Baselines selected their own frozen synthetic-data
  ML efficacy objective. Scheduling affected availability, so this subset
  cannot establish an overall method conclusion.
- [Disjoint GPU refinement confirmation](results/dope-target-refinement-confirmation.json),
  [table](results/dope-target-refinement-confirmation.csv) and
  [figure](results/dope-target-refinement-confirmation.svg): 12 successful GPU
  fits, 72 measured cells and 360 unchanged reference cells on six lineages
  disjoint from discovery. At `4n`, CatBoost median retention is 0.939543 for
  the prior DOPE reference, 0.937693 for the full-width 8,192-step profile and
  0.949384 for the narrow 8,192-step profile. Combined discovery/confirmation
  fit operations total 859.788 seconds within the declared 21,600-second
  research ceiling; earlier architecture research and admission waiting are
  separate costs. Both profiles remain research candidates.

The sealed SDV native-v2 ledger contains 154 successful trials, 525 scheduling
cutoffs (524 unstarted, one truncated) and 25 infrastructure interruptions.
Those cutoffs are not method failures. A new SDV fit round requires Jepson's
explicit authorization; none is admitted by this publication. All shared
comparisons retain byte charges, copy checks, real-vs-real controls, failed
cells and immutable receipt hashes. MFS-v2, PTF-v1, release-safe results and
superiority remain null. The public-core headline, five-fit final schedules
and full evaluator gates are unfinished.

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
within L3. The other five exceed 10,240 B. [Its validation JSON](results/copula-native-validation.json),
[schema](results/copula-native-validation.schema.json), and
[trial table](results/copula-native-validation.csv) carry per-trial
status, KPI, bytes, and immutable scratch receipt hashes. Regenerate with
`python3 -m research.benchmark.publish_copula`. This remains an exploratory
single-fit-seed common-numeric result and is not a production comparison.

The two study-owned density baselines also completed their frozen
100-lineage S3 validation matrix. All 900 native tuning trials completed
without failure: 300 independent-marginal and 600 Chow–Liu trials. Native
likelihood improved over each method's own default in 65 and 75 cells,
respectively. The selected charged artifacts meet L3 in 94/100 and 31/100
cells, respectively; defaults meet L3 in 86/100 and 57/100. These counts are
artifact checks, not release-safe generator scores, because the shared
test, utility, and privacy gates have not run. [The all-lineage JSON](results/compact-native-all-validation.json),
[schema](results/compact-native-all-validation.schema.json), and
[trial table](results/compact-native-all-validation.csv) link every selection
and attempt to immutable scratch receipt hashes. Regenerate the matrix lock
with `python3 -m research.benchmark.freeze_all_native` and the publication
with `python3 -m research.benchmark.publish_compact_native --round-lock
/mnt/fast-scratch/dope-benchmark/compact-native-all-v1/round.lock.json
--expected-jobs 200 --basename compact-native-all-validation`.

GaussianCopula also completed its frozen 100-lineage S3 validation matrix:
200 author-library trials, no failures, and a native likelihood improvement
over its own default in 62 cells. The selected artifacts meet L3 in 62/100
cells, compared with 41/100 defaults. [The all-lineage Copula JSON](results/copula-native-all-validation.json),
[schema](results/copula-native-all-validation.schema.json), and
[trial table](results/copula-native-all-validation.csv) reconcile the source,
environment, trial, metric, timing, artifact, and log hashes. Regenerate the
matrix lock with `python3 -m research.benchmark.freeze_all_copula` and the
publication with `python3 -m research.benchmark.publish_copula --round-lock
/mnt/fast-scratch/dope-benchmark/copula-native-all-v1/round.lock.json
--expected-jobs 100 --basename copula-native-all-validation`. The native
likelihood numbers are never ranked across methods.
The [SVG](results/native-validation-l3.svg) and
[PDF](results/native-validation-l3.pdf) summarize default versus native-tuned
artifact counts for the three methods on the same 100 lineages. Regenerate
both from the committed JSON with
`python3 -m research.benchmark.publish_native_figure`.

## Final admission still blocked

The 29-method source inventory currently has six locked entries (DOPE,
three compact baselines, CTGAN, and TVAE), six pilot-only entries (AIM, ARF,
synthpop CART, TabPC, ForestDiffusion and TabDDPM), 11 entries with pending
execution readiness, and six unavailable entries (CTAB-GAN+, GEM-T, TaEGAN,
PrivBayes, TabKDE and TabPFGen). Source audits alone do not admit execution.
Unavailable methods contribute
no DOPE win. The density KPI implementation now gives GaussianCopula,
independent marginals, and Chow-Liu held-out validation mean log density;
final tuned selection checks that a frozen native objective actually chose
the winning trial within the eight-trial, 12-hour cap. The remaining native
objectives, adapters, reproduction checks, DP accounting, author-faithful
track, public-core datasets, evaluator gates, budget lock, and exact final job
matrix are not complete. `admission.py` therefore keeps sealed-test common
evaluation closed. The six-fit pilot and this single-seed research survey are
cost and validation evidence only.

## Complete S3 matched confirmation validation

[The matched confirmation JSON](results/s3-matched-confirmation-validation.json),
[table](results/s3-matched-confirmation-validation.csv),
[SVG](results/s3-matched-confirmation-validation.svg)/[PDF](results/s3-matched-confirmation-validation.pdf)
and [manifest](results/s3-matched-confirmation-validation.manifest.json) record
six disjoint confirmation lineages with the same bounded official-training-derived
partitions for DOPE, CTGAN and TVAE. All 288 logical cells are accounted for:
258 successful physical cells, 12 unavailable sample cells from two failed DOPE
GPU admissions, and 18 default/native logical aliases with no additional compute.
All four declared DOPE profiles remain visible; no production family is selected.
The 22 successful DOPE fits charged 984–6,307 bytes including projection state.
The retained CTGAN/TVAE configurations exceed the L3 cap and are unconstrained
quality comparisons. Their original four-trial native efficacy searches and
winners were verified; shared outcomes did not select them.

For fit seed 11 and three sample seeds at 4n, the CatBoost median retention
on `503_wind` is 0.996 for DOPE `features12_steps2048` (3,846 bytes) and
0.802 for native-selected TVAE (471,140 bytes). On `1193_BNG_lowbwt`, the
same DOPE profile is 0.970 (2,436 bytes), while native-selected TVAE is higher
at 1.029 (293,868 bytes). These selected examples illustrate mixed outcomes;
the complete table retains every configuration and failure. They are
null-normalized utility estimates, without fit uncertainty or paired superiority.

Successful DOPE fit entrypoints totaled 111.043 seconds on xbabe1/xbabe3.
Dispatch, hash and admission overhead totaled 2,119.258 seconds, including
130.203 seconds for the two failed admissions. The declared GPU fit ceiling
was 14,400 seconds; the original frozen grid, fit deadlines and VRAM cap were
preserved. Sampling and metric replays use xbabe2. Device-wide energy estimates
include idle and do not provide full campaign or baseline-attributed energy.
Original native tuning and earlier DOPE architecture research are separate costs.

Both controllers closed with exit code zero. All successful sample/metric
replays match, 5,085 declared source/input files were verified, and a pinned
receipt lock rejects rewritten evidence before metric reads. Full current or
historical executable closures are not certified. Official tests remain sealed;
MFS-v2, PTF-v1, release-safe L3 and paired superiority remain null. This panel
does not replace public-core paper analysis or the full campaign.

## Completed CTGAN/TVAE GPU comparison and DOPE readout repair

The research queue completed **96 author-library GPU fits** on the same twelve
S3 discovery/confirmation datasets: CTGAN and TVAE, four frozen configurations
each, including their 300-epoch author defaults. Every fit and deterministic
sampling check succeeded. Native selection maximized the prespecified mean
of SDMetrics LinearRegression and MLPRegressor raw R² efficacy, with bytes and
configuration hash as tie-breaks. The component metrics are from the author
library; the aggregation and four-point grid are study choices. DOPE's KPI
did not select comparator configurations. See [the source and objective audit](SDV_METHOD_AUDIT.md).

Shared validation CatBoost retention is reported below. The discovery and
confirmation columns each contain six datasets; these are descriptive medians,
not paired confidence bounds. Comparator tuning uses each dataset's validation
partition, including datasets called confirmation in the DOPE architecture
research. Thus comparator selected estimates can be optimistic.

| Configuration | Discovery median | Confirmation median | All 12 median | Artifacts within L3 |
|---|---:|---:|---:|---:|
| DOPE symbolic default | 0.837 | 0.873 | 0.837 | 12/12 |
| DOPE repaired GPU conditional q8 | 0.964 | 0.638 | 0.959 | 12/12 |
| DOPE repaired GPU conditional q10 | 0.970 | 0.645 | 0.965 | 12/12 |
| CTGAN author default | -0.399 | -0.237 | -0.237 | 0/12 |
| CTGAN native selected | 0.087 | 0.091 | 0.089 | 0/12 |
| TVAE author default | 0.327 | 0.272 | 0.327 | 0/12 |
| TVAE native selected | 0.521 | 0.594 | 0.521 | 0/12 |

Retention is `(null loss − synthetic-trained loss) / (null loss − real-trained
loss)` on real validation rows. Values below zero mean the synthetic-trained
auditor did worse than the constant baseline; values above one can occur in
finite samples. It is not the production PTF-v1 score. Native KPI values remain
in per-trial receipts and are never ranked across methods.

The conditional GPU repair follows a separately recorded 12-fit discovery
failure. The original model exported its readout after 96 AdamW steps; its
prediction bias severely damaged low-variance targets. The repair retains the
GPU-learned hidden basis and refits output coefficients with the existing
regularized solver. A GPU regression test checks a small target signal and
exact repeatability. Twelve repaired discovery fits met the frozen diagnostic
promotion rule, followed by twelve disjoint confirmation fits. All 24 repaired
artifacts meet the byte cap. q8 passed all twelve validation copy screens; q10
had one near-copy failure on confirmation. One native-selected TVAE cell also
had a copy-screen failure. The repaired GPU families beat the symbolic default
on only one of six confirmation datasets, despite their larger pooled medians.
No production family or superiority claim is selected from these results.

CTGAN/TVAE observed fit time sums to 2,927.7 seconds across three hosts, with
maximum single-fit time 108.1 seconds and peak observed device memory 634 MiB.
The 36 original/repaired conditional GPU fits sum to 76.1 seconds, with peak
observed device memory 440 MiB. These costs supplement the earlier DOPE
architecture research and are not evidence of equal total R&D spending.
The repaired probe enforces a process-group 600-second timeout and fails
closed on a resource-monitor error; original frozen attempts retain their
original source and accounting.

[The neural comparison JSON](results/sdv-native-validation.json),
[schema](results/sdv-native-validation.schema.json),
[table](results/sdv-native-validation.csv), and
[SVG](results/sdv-native-validation.svg)/[PDF](results/sdv-native-validation.pdf)
include both defaults, every native tuning trial, DOPE references, common
metrics, failures, and immutable scratch hashes.
[The conditional repair JSON](results/conditional-gpu-validation.json),
[schema](results/conditional-gpu-validation.schema.json),
[table](results/conditional-gpu-validation.csv), and
[SVG](results/conditional-gpu-validation.svg)/[PDF](results/conditional-gpu-validation.pdf)
retain the unsuccessful original models as well as every repaired model.
Regenerate with `python3 -m research.benchmark.publish_conditional_validation`
and `python3 -m research.benchmark.publish_sdv_validation --evaluate --publish`.
Both publishers accept `--render-existing` to rebuild tables and figures from
committed JSON without opening scratch inputs.

The [publication manifest](results/gpu-native-20261001.manifest.json) binds
all ten result artifacts to scratch receipts and archived verification logs.
Four stratified CTGAN/TVAE repeat fits reproduced artifact and sample hashes
exactly, with zero native KPI difference. Two DOPE q8 repeat fits on the
narrowest and widest discovery datasets reproduced artifact hashes, sample
hashes, and all metric values exactly, excluding elapsed measurement time.
Cross-host CTGAN and TVAE sampling hashes also matched. These audit fits are
additional cost, separate from the 96 comparator and 36 conditional research
fits. `python3 -m research.benchmark.publish_result_manifest` verifies and
regenerates the manifest. Repository `just fast`, `just check`, the 45 research
tests, and the focused GPU regression test passed.

Both comparisons use one fit seed, one sample seed, and `n` synthetic rows.
Official tests remain unopened. Full attacks, repeated fits, required sample
sizes, final public-core evaluation, and the other method outcomes remain
outstanding. **PTF-v1 and MFS-v2 remain null.**

## Adult / California / News 24-hour pilot

The separate 24-hour pilot uses training-derived validation partitions. The
official tests remain sealed. CTGAN and TVAE were tuned against their own
locked author-library ML-efficacy objectives: mean binary F1 for Adult and
mean regression R² for California and News. The selected configurations and
their default configurations were evaluated with the same CatBoost
TSTR/TRTR-retention auditor as DOPE. Native KPI values are compared only
within one method and dataset.

At fit seed 23, three sample seeds (101/211/307) and `n`/`4n` sizes produced
66 hash-verified common metric cells. The median `n` retention was 0.8275
for DOPE q8 versus 0.7951 for TVAE and 0.0139 for CTGAN on Adult; 0.7008
for DOPE q8 versus 0.9154 for native-tuned TVAE and 0.6652 for CTGAN on
California. On News, DOPE q8 was −3.9536, TVAE −1.0013, and CTGAN −0.8818;
the over-cap DOPE q10 quality candidate reached 0.3287. News retention has
a small real-model gain over the null and is unstable as a ratio. Two
default CTGAN fixed fits timed out and remain explicit failures.

DOPE q8 charged artifacts are 6,337 B on Adult, 2,525 B on California, and
13,603 B on News. The News q10 quality artifact is 13,713 B. All measured
CTGAN and TVAE artifacts exceed the 10,240 B L3 cap. Byte eligibility alone
does not establish release safety. The q8 compiler candidate is symbolic;
separate neural and conditional GPU training costs are recorded on scratch.
No product PTF-v1 or MFS-v2 score, paper paired claim, or official-test result
is available from this pilot.

[The pilot JSON](results/pilot24-native-neural.json),
[schema](results/pilot24-native-neural.schema.json),
[table](results/pilot24-native-neural.md),
[CSV](results/pilot24-native-neural.csv), and
[SVG](results/pilot24-native-neural.svg)/[PDF](results/pilot24-native-neural.pdf)
retain native selections, all common metric cells, failures, charged bytes,
and immutable scratch receipt hashes. Regenerate with
`python3 -m research.benchmark.pilot24_neural_comparison`; its
`--render-existing` mode reproduces the table and figures from committed
JSON.

The frozen GaussianCopula fixed-seed matrix completed on the same three
datasets. Its native tuning selected only by held-out mean log density. The
paired fit-seed-23 DOPE q8 minus tuned-Copula median retention differences
were +0.119 on Adult, −0.024 on California, and −3.309 on News. Four default
fit cells timed out and two default cells exhausted all eight sample attempts;
those failures have null outcomes. All 24 scored metrics were replayed exactly
from verified validation samples except elapsed timing. The
[paired Copula report](results/pilot24-copula-matched.md),
[JSON](results/pilot24-copula-matched.json), and
[schema](results/pilot24-copula-matched.schema.json) retain every failure and
hash. These three datasets and one paired fit seed do not establish a paper
superiority claim.

A separate bounded lossless projection refinement repackaged the GPU-trained
News q10 model with the repository's compact projection codec. The complete
model and packed map charge **9,205 B**. All twelve frozen sample identities
at `n/2n/4n/8n` reproduced their original hashes. The first frozen attempt
failed while writing a verification receipt and remains an explicit failure;
the versioned second attempt succeeded. On the common News validation rows,
the packed q10 candidate retains the original median CatBoost values of
0.3287 at `n` and 0.3109 at `4n`; native-tuned CTGAN recorded −0.8818 and
−0.8440, and TVAE −1.0013 and −0.9956. This is a single-dataset descriptive
comparison. The packed projection map remains restricted on scratch, public
sidecar safety and other release gates are unverified, and official tests are
sealed. The [packed comparison](results/pilot24-news-q10-packed.md),
[JSON](results/pilot24-news-q10-packed.json),
[schema](results/pilot24-news-q10-packed.schema.json), and
[CSV](results/pilot24-news-q10-packed.csv) regenerate with
`python3 -m research.benchmark.publish_news_q10_packed`;
`--from-json` rebuilds the table without scratch data. PTF-v1 and MFS-v2 are
null.

ARF completed four original and four bounded-refinement native trials per
dataset, within eight trials and twelve hours per dataset. All four original
Adult and News trials timed out at the unchanged 900-second limit; all four
California originals and all twelve refinements succeeded. A labeled study
implementation of held-out FORDE mean log density selected Adult short trial
1, California original trial 3, and News short trial 2 using only the
training-derived validation partition. Their charged artifacts are 19,187,442,
85,471,471, and 55,316,440 bytes, respectively, so none is L3 byte eligible.
The native density values compare configurations within a dataset only.
[The complete ARF native report](results/arf-combined-native-validation.md),
[JSON](results/arf-combined-native-validation.json),
[schema](results/arf-combined-native-validation.schema.json), and
[CSV](results/arf-combined-native-validation.csv) retain all 24 attempts and
immutable receipt hashes. Regenerate with
`python3 -m research.benchmark.publish_arf_combined`; `--from-json` rebuilds
the table. Fit-seed-23 common validation outcomes and release gates are still
pending; ARF has no PTF-v1 or MFS-v2 score from this grid.

The News seed-11 symbolic autoregressive candidate had a one-seed retention
lead, so a frozen single-fit GPU confirmation tested target weight 4 at seed
23. Its model bytes exactly matched the existing target-weight-2 q10 fit.
The first confirmation attempt was blocked by CPU-affinity admission before
fitting; its receipt is retained, and the repaired attempt kept the same
600-second and 16-GiB limits. Six duplicate samples were skipped because the
same model, binary, projection, fit seed, and deterministic sample seeds had
already been replayed. This adds research cost and no independent validation
win. The [identity report](results/pilot24-news-autoreg-identity.md),
[JSON](results/pilot24-news-autoreg-identity.json),
[schema](results/pilot24-news-autoreg-identity.schema.json), and
[CSV](results/pilot24-news-autoreg-identity.csv) regenerate with
`python3 -m research.benchmark.publish_news_autoreg_identity`.
Official tests remain sealed and MFS-v2/PTF-v1 are null.

Synthpop CART completed its four-trial-per-dataset native validation grid,
selecting only by held-out CART pMSE. At the fixed fit seed 23, three paired
sample seeds and `n`/`4n` sizes were measured and exactly replayed against the
same training-derived validation partitions as DOPE. Median tuned `n`
CatBoost TSTR/TRTR retention was 0.9255 for synthpop versus 0.8275 for DOPE
on Adult, 0.9588 versus 0.7008 on California, and 0.6419 versus 0.3287 on
News. Native pMSE and shared retention answer different questions; the native
pMSE values are comparable only across synthpop configurations within one
dataset. All synthpop fitted artifacts exceed the 10,240-byte L3 limit; Adult
samples also failed the copy gate. DOPE's byte eligibility does not certify
its other gates. [The paired synthpop report](results/pilot24-synthpop-matched.md),
[JSON](results/pilot24-synthpop-matched.json),
[schema](results/pilot24-synthpop-matched.schema.json), and
[CSV](results/pilot24-synthpop-matched.csv) retain all 36 paired cells, native
selections, bytes, copy counts, and immutable metric/replay hashes. Regenerate
with `python3 -m research.benchmark.publish_synthpop_matched`; `--from-json`
rebuilds the table without restricted scratch. This one-fit-seed pilot does
not support a paper superiority claim; official tests remain sealed and
MFS-v2/PTF-v1 are null.

The News q10 symbolic autoregressive DOPE candidate has now been assessed at
five independent GPU fit seeds (11, 23, 37, 53, 71) under one frozen
configuration and the same training-derived validation partition. Lossless
projection packing charges 9,195–9,205 bytes per fitted artifact, below L3;
all 60 `n`/`2n`/`4n`/`8n` sample identities matched their references and all
30 `n`/`4n` metric payloads replayed exactly. Median CatBoost retention across
the five fit-seed medians was **0.5723 at `n`** and **0.5478 at `4n`**. The
`n` fit-seed medians range from 0.3287 to 0.9782, so the earlier high
single-seed result is not a stable five-fit result. No sample in the 30 common
metric cells was an exact or near training-row copy. The real MLP auditor was
noninformative in all 30 cells, so its retention remains null. These are
validation outcomes, not PTF-v1 scores or a 0.99 release pass: privacy attacks,
official-test utility and other gates are still absent. The one-fit-seed
native-tuned CTGAN/TVAE comparison remains a separate paired panel and is not
treated as a five-fit-seed comparison. [The five-fit report](results/pilot24-news-q10-fivefit.md),
[JSON](results/pilot24-news-q10-fivefit.json),
[schema](results/pilot24-news-q10-fivefit.schema.json),
[CSV](results/pilot24-news-q10-fivefit.csv), and
[SVG](results/pilot24-news-q10-fivefit.svg)/[PDF](results/pilot24-news-q10-fivefit.pdf)
retain every seed, artifact charge, GPU cost, validation metric and immutable
receipt hash. Regenerate with
`python3 -m research.benchmark.publish_news_q10_fivefit` and
`python3 -m research.benchmark.publish_news_q10_fivefit_figure`.

The two study-owned compact references also completed the same fixed-fit-23
pilot validation protocol. Their default and native-likelihood-selected
configurations produced 72 `n`/`4n` common metric cells, all exactly replayed.
At `n`, tuned independent marginals had median CatBoost retention −0.097 on
Adult and 0.008 on California versus DOPE's 0.827 and 0.701. Tuned Chow–Liu
had 0.659 and 0.710 versus those same DOPE values; California is a measured
case where Chow–Liu's median is higher. News reference retention was strongly
negative with a small real-over-null utility denominator, so those values are
retained in the tables with an instability note. Only 12 of 72 compact cells
met the artifact byte cap, and four failed the row-copy screen. No compact or
DOPE cell is certified release safe. These implementations are labeled as
study references throughout, and their native density KPIs are never ranked
against another method's KPI. [The paired compact report](results/pilot24-compact-matched.md),
[JSON](results/pilot24-compact-matched.json),
[schema](results/pilot24-compact-matched.schema.json), and
[CSV](results/pilot24-compact-matched.csv) retain every cell and receipt hash.
Regenerate with `python3 -m research.benchmark.publish_compact_matched`;
`--from-json` rebuilds the table from committed results. Official tests remain
sealed and MFS-v2/PTF-v1 are null.

ARF completed its fixed-fit-23 common validation readout for every available
default and native-tuned configuration. Twenty-three `n`/`4n` metric cells
replayed exactly. One Adult tuned `4n` sample timed out, while original
default fits on Adult and News were unavailable after native timeouts; none
of these missing cells counts as a DOPE win. At `n`, tuned ARF median CatBoost
retention was 0.8322 on Adult, 0.8993 on California, and 0.5743 on News,
versus DOPE 0.8275, 0.7008, and 0.3287. These comparator leads are visible
as measured validation quality. All ARF fitted artifacts exceed the charged
10,240-byte L3 cap, and six measured cells fail the row-copy screen, so no
ARF outcome is release safe. The native FORDE density evaluator is labeled
as a study implementation of the author's fitted factors; its KPI values
select ARF configurations within each dataset only. [The paired ARF report](results/pilot24-arf-matched.md),
[JSON](results/pilot24-arf-matched.json),
[schema](results/pilot24-arf-matched.schema.json), and
[CSV](results/pilot24-arf-matched.csv) retain every success, timeout,
unavailable default, artifact charge, copy count, and immutable receipt hash.
Regenerate with `python3 -m research.benchmark.publish_arf_matched`;
`--from-json` rebuilds the table. Official tests remain sealed and
MFS-v2/PTF-v1 are null.

The full ARF fixed-fit-23 sample schedule is now accounted for at
`n`/`2n`/`4n`/`8n`: 44 successful samples and four 900-second timeouts
across 48 frozen jobs. The timeouts are one Adult tuned `4n` and all three
Adult tuned `8n` cells. California default and tuned, plus News tuned,
completed all twelve sample cells per configuration. Adult and News original
defaults remain unavailable from native fit timeouts. Every attempt, fitted
artifact charge, source snapshot and receipt hash is reconciled in the
[sample schedule](results/pilot24-arf-sample-matrix.md),
[JSON](results/pilot24-arf-sample-matrix.json),
[schema](results/pilot24-arf-sample-matrix.schema.json), and
[CSV](results/pilot24-arf-sample-matrix.csv). Regenerate with
`python3 -m research.benchmark.publish_arf_sample_matrix`. The shared
validation metric panel still covers only `n` and `4n`; no `2n`/`8n` utility
or release score is inferred.

The full synthpop CART fixed-fit-23 schedule is now accounted for at
`n`/`2n`/`4n`/`8n`: all 72 frozen default and native-tuned sample jobs
succeeded. Every sample receipt, fitted artifact charge, source snapshot,
and prior `n`/`4n` paired receipt link is reconciled in the
[sample schedule](results/pilot24-synthpop-sample-matrix.md),
[JSON](results/pilot24-synthpop-sample-matrix.json),
[schema](results/pilot24-synthpop-sample-matrix.schema.json), and
[CSV](results/pilot24-synthpop-sample-matrix.csv). Regenerate with
`python3 -m research.benchmark.publish_synthpop_sample_matrix`. Native
tuning used validation CART pMSE. The shared validation metrics remain
limited to `n` and `4n`; no `2n`/`8n` utility or release score is inferred.
All five distinct fitted synthpop artifacts exceed the 10,240-byte L3 cap;
California default and tuned share the same fit, making six reported
dataset/configuration cells. Official tests remain sealed, and MFS-v2/PTF-v1
remain null.

A bounded California DOPE autoregressive follow-up ran two fit seeds on
GPU hosts and measured three sample seeds at `n` and `4n` on training-derived
validation. Both 2,525-byte artifacts failed compiler utility and driver
gates. Fit seed 23 median CatBoost retention was 0.6934 at `n` and 0.6910
at `4n`, below the q8 seed-23 medians of 0.7008 and 0.6948; all 12 cells
had zero exact and near training-row copies. The compiler selected a symbolic
artifact, so host placement does not imply neural GPU training. The family
was chosen after observing California validation utility, making this a
descriptive follow-up rather than an independent dataset confirmation. The
[report](results/pilot24-california-q10.md),
[JSON](results/pilot24-california-q10.json),
[schema](results/pilot24-california-q10.schema.json), and
[CSV](results/pilot24-california-q10.csv) regenerate with
`python3 -m research.benchmark.publish_california_q10`. Official tests
remain sealed; MFS-v2/PTF-v1 and certification are null.

The fixed DOPE q8 configuration now has five fit seeds on both Adult and
California, with all 120 `n`/`2n`/`4n`/`8n` sample cells complete. All 60
`n`/`4n` validation metric payloads replayed exactly, as did a separately
frozen stratified 16-cell audit. Median CatBoost retention across fit-seed
medians is Adult 0.8232/0.8289 and California 0.7032/0.6991 at `n`/`4n`.
The ten fits consumed 116.88 seconds of summed fit elapsed time, and charged
artifacts range from 2,524 to 6,342 bytes. Host costs, three preflight launch
failures, per-seed outcomes, near-match rates, and real-vs-real controls are
retained in the [stability report](results/pilot24-dope-q8-fivefit.md),
[JSON](results/pilot24-dope-q8-fivefit.json),
[schema](results/pilot24-dope-q8-fivefit.schema.json),
[CSV](results/pilot24-dope-q8-fivefit.csv), and
[figure](results/pilot24-dope-q8-fivefit.svg). Regenerate with
`python3 -m research.benchmark.publish_dope_q8_fivefit` and
`python3 -m research.benchmark.publish_dope_q8_fivefit_figure`;
`--from-json` regenerates tables without scratch access. This is a fixed
configuration replication following validation family selection. The initial
native-tuned neural comparator panel had one fit seed; the completed matched
five-fit panel below adds the fixed comparator replications.
Every compiler receipt still fails utility, driver, and unmeasured production
evidence gates. Some California fits used negligible GPU memory, so GPU
host placement is not interpreted as neural training for every artifact.
Official tests remain sealed, and MFS-v2/PTF-v1 are null.

## Complete matched five-fit CTGAN/TVAE stability panel

Adult and California now have the complete matched five-fit validation matrix:
240 shared `n`/`4n` metric cells, including 210 successful cells and 30 explicit
unavailable Adult default CTGAN cells. All 360 comparator `n`/`2n`/`4n`/`8n`
sample identities are reconciled (300 successful, 60 unavailable); prior
seed-23 fits and samples are reused once. The 30 distinct comparator fits
include 25 successes and five Adult default CTGAN 600-second timeouts.
Defaults and configurations selected on each comparator's frozen SDMetrics
native F1/R² objective remain separate, except when they are identical and
share a physical fit. Replication introduced no tuning.

Median CatBoost retention across the five fit medians at `n`/`4n` is
DOPE 0.8232/0.8289 versus TVAE 0.8180/0.8128 on Adult, and
DOPE 0.7032/0.6991 versus native-selected TVAE 0.8929/0.8964 on California.
These descriptive validation outcomes establish neither paired paper
superiority nor a release score. DOPE's charged artifacts are 2,524–6,342
bytes; every successful CTGAN/TVAE artifact exceeds the 10,240-byte L3 cap.
Unconstrained comparator quality remains visible separately from byte
eligibility. Failed executable fits confer no DOPE win.

The [report](results/pilot24-native-neural-fivefit.md),
[JSON](results/pilot24-native-neural-fivefit.json),
[schema](results/pilot24-native-neural-fivefit.schema.json),
[CSV](results/pilot24-native-neural-fivefit.csv), and
[SVG](results/pilot24-native-neural-fivefit.svg)/
[PDF](results/pilot24-native-neural-fivefit.pdf) retain per-seed outcomes,
charged bytes, native KPI, near-match/real-vs-real controls, failed attempts,
host costs, all three DOPE preflight failures, and immutable receipt hashes.
Prior pilot R&D is charged separately; total campaign R&D remains incomplete,
and equal total research spending is not claimed. Regenerate with
`python3 -m research.benchmark.publish_native_neural_fivefit` and
`python3 -m research.benchmark.publish_native_neural_fivefit_figure`;
`--from-json` regenerates the tables without scratch access. Official tests
remain sealed; MFS-v2/PTF-v1 and release-safe status remain null.

## TabPC author contracts

The pinned author TabPC/Cirkit runtime now passes regression and binary
generated-input fit/sample contracts. Native tuning is frozen to author
transformed validation mean NLL and the declared eight-trial grid; the CPU
memory-stat compatibility guard, retained empty-row metadata tensor and full
artifact charges are explicit in [TABPC.md](TABPC.md) and
`tabpc-source.lock.json`. No real-data TabPC result or release score is inferred
from these probes. Final campaign locks and official tests remain sealed.

The later complete frozen TabPC pilot has 24 native trials (13 successes,
eight Adult GPU allocation failures and three News fit timeouts), selected
only by author transformed validation NLL. All 48 fixed-fit-23 common sample
jobs are accounted for: 45 successful and three News tuned `8n` timeouts.
All 24 planned `n`/`4n` metric cells completed. Median CatBoost retention
across three sample seeds is California default 0.9620/0.9670 and
native-selected 0.9247/0.9327 at `n`/`4n`, versus DOPE 0.7008/0.6948.
News default is 0.5105/0.6937 and native-selected 0.4220/0.6675,
versus the prior losslessly packed DOPE q10 result 0.3287/0.3109.
All successful TabPC artifacts exceed L3; Adult unavailable tracks confer
no DOPE win. This is a descriptive single-fit-seed pilot, not the final
five-fit or public-core paired analysis. [The complete report](results/pilot24-tabpc-native.md),
[JSON](results/pilot24-tabpc-native.json),
[schema](results/pilot24-tabpc-native.schema.json),
[CSV](results/pilot24-tabpc-native.csv), and
[figure](results/pilot24-tabpc-native.svg) retain all outcomes and immutable
receipt hashes. Original learned bytes are preserved through the sampling
runtime metadata repair; no new common-validation fits or tuning occurred.
Official tests remain sealed; MFS-v2/PTF-v1 and release-safe status remain null.

## Closed ARF native search

[The complete ARF native ledger](results/arf-s3-population-native.md) retains
800/800 successful fits: eight frozen configurations on each of the 100 bounded
S3 lineages, one fit seed (11). Native winners maximize held-out FORDE density,
independently of DOPE's KPI or shared retention. Default and selected artifacts
include projection charges. These are completed native fits, not a shared
quality comparison; ARF sampling, five-fit uncertainty and generator gates
remain pending. Native density values are not aggregated across datasets or
ranked against another method. No production score or DOPE win is inferred.
