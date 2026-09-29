# DOPE generator benchmark protocol v1

Drafted 2026-09-29 before any public-cohort test partition was opened. The
version 1 generator thresholds and study design are frozen here; source,
budget, and evaluator locks are still pending. This protocol is a research
comparison, not a DOPE production release decision.
Its source base is merged DOPE `581fd69`; the generator contract pins the
SHA-256 of `production/kpi-contract.json`. The literature inventory and
corrected BibTeX are in `DOPE_SYNTH_LITERATURE_SUMMARY.md` and
`DOPE_SYNTH_REFERENCES.bib` beside the published copy of this protocol.

## Question and endpoints

The hypothesis is that DOPE passes more L3 generator gates than each executable
compact baseline on paired public-core datasets. The first endpoint is the
paired dataset gate-pass-rate difference. Among datasets where both methods
pass, the second endpoint is MFS-v2 difference. No win is presumed. PTF-v1
remains DOPE's frozen production KPI and is reported separately with its
informative-group count, interval, and sparse-profile folding. A small public
cohort cannot establish production-scale profile coverage.

The generator-only gate is `contract.json` version 1. It inherits the frozen
MFS-v2 scalar from production: weights 0.30 utility, 0.20 driver, 0.20
distribution, 0.15 structure, 0.10 coverage, 0.05 compactness, privacy soft
weight zero, and `100 exp(sum(weight log(1e-6 + component)))`. A component is
in `[0,1]`. Any applicable failed or unmeasured gate yields a null score.
Raw metric vectors remain available. DOPE's router and campaign release gates
are displayed separately and never applied to external generators.

## Population and exclusions

The public core is the union of the 16 named TabDDPM datasets and six TabSyn
datasets in `study-design.json`. Adult and Default are deduplicated by source
identity plus sorted source-row SHA-256, leaving at most 20 unique datasets.
Only binary and regression tasks enter; every other task receives an exclusion
receipt. The compact panel runs on all applicable public-core datasets. Neural,
LLM, and DP primary panels use only applicable TabSyn-six datasets. Additional
core results are secondary. Rankings never pool different dataset sets.

The public extension selects up to ten binary OpenML-CC18 and ten regression
CTR23 tasks. Require a defined target, 500–100,000 rows, 1–2,000 input
features, recorded redistribution rights, and deduplication. Sort eligible
official task IDs by SHA-256 and take the first ten per task. Record shortfall.

The JopeDime robustness panel is separate. The pinned catalog hash is
`ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`.
It has 104 regression entries with recorded MIT evidence. The 3,755 entries
with unknown license status are ineligible for publication. Blob and dataset
manifest hashes must match the catalog. S3 presence is not a rights grant.

Use an auditable official train/validation/test split when one exists. Otherwise
group identical source rows and split deterministically 60/20/20 with seed
1729, stratified for binary tasks. Reject overlaps across partitions. Fit
projection maps on training rows only. Workers receive training and validation;
the evaluator alone receives test. The author-faithful track preserves each
method's documented preprocessing and data types. DOPE is inapplicable to
native mixed-type input there. The common-numeric track gives every method
the same headerless `[0,1]` projection, with inverse maps and projection-only
utility cost reported. The current projection is train-min/max with clipping,
train-observed categorical one-hot levels, all-zero unknown levels, and
train-fitted target scaling. Its map is charged to artifact bytes if sampling
or inverse transformation requires it. The research runner conservatively
charges the projection map for every common-numeric artifact.

## Method and budget locks

`methods.lock.json` lists the compact, neural/flow, LLM, and DP methods. A
method may enter final evaluation only with an audited source commit,
dependency or container digest, license, literal default configuration, tuning
search space, and fit/sample commands. The file is currently `complete: false`:
DOPE, study-owned independent-marginals and discretized Chow–Liu adapters,
and the pinned Copulas GaussianMultivariate adapter are the only runnable
methods. The study-owned adapters are labeled as reference implementations,
never represented as another author's code. GEM-T remains an availability check. If author code
cannot be located, a paper implementation requires validation against at
least two reported experiments before comparison. An unavailable or
unvalidated method is not a DOPE win.

Run pinned defaults and equal-budget tuning as separate tracks. Published
search spaces take precedence; otherwise the audited adapter search space is
frozen before test. Select with validation data only. The 24-hour maximum
pilot uses Adult, California, and News; DOPE, GaussianCopula, TabPC, TabKDE,
TabSyn, TabDiff, and AIM; two fit seeds and two sample seeds. Measure fit,
tuning, sample, attack, storage, timeout, and failure costs. Freeze
`budget.lock.json` and the final method–dataset matrix after the pilot and
before opening the public test set. The final evaluator also requires frozen
`datasets.lock.json` and `evaluator.lock.json`; `admission.py` fails closed until
all five source, data, budget, matrix, and evaluator locks are complete. A
three-host full campaign has a 14-day
ceiling. Protect compact public-core first; reduce extension, then secondary
methods. If compact public-core exceeds the ceiling, stop after the pilot and
issue a quantified feasibility report.

## Jobs, artifacts, and metrics

Full jobs use fit seeds `11,23,37,53,71`, sampling seeds `101,211,307`, and
sample sizes `n,2n,4n,8n`; `n` and `4n` are required PTF cells. The run key
hashes dataset, split, track, source, configuration, fit seed, sampling seed,
evaluator contract, and output rows. Receipts are immutable and idempotent;
failures and timeouts remain in the ledger. Count all dataset-specific files
needed for sampling, including preprocessing maps, coresets, fine-tuned
weights, and metadata. L3 raw limit is 10,240 B; L2 is 32,768 B. Gzip is
descriptive. Report shared pretrained weights and runtime bytes separately.
Verify repeated sampling from the recorded artifact; source-row dependence
prevents a compact-tier pass.

The generator thresholds in `contract.json` are exact. Feature-importance
Spearman and top-k Jaccard apply only with at least three measured informative
features; the informative count itself is required. PTF-v1 uses production's
null-normalized `(null-TSTR)/(null-TRTR)` for informative lineages, its
low-signal non-inferiority rule, equal lineage weighting, paired one-sided
Student-t lower bound, and sparse-profile folding. The profile bounds and
required sample sizes are inherited from the production contract. The shared
evaluator must freeze code and dependency digests for calibration, joint and
driver fidelity, query p95, tails, type-I error, 95% coverage, shadow-model
membership, attribute inference, exact/near copies, and feature importance
before final evaluation. Until then, these are missing evidence and MFS is
null. The current `pilot_metrics.py` is validation-only descriptive evidence;
it is not a complete gate evaluator.

Independent metrics include TSTR with CatBoost, logistic or linear regression,
and MLP; marginal and pair fidelity, C2ST, precision/recall/authenticity,
feature-importance agreement, query error, time, peak RAM/VRAM, and raw bytes.
Cross-check a stratified subset against SDMetrics and SynthEval. Publish
per-dataset vectors, seed variation, Pareto fronts, task and size strata,
real-vs-real false-positive rates, weight and component sensitivity, and rank
concordance. A validation-only smoke result is never a paper result.

## Privacy threat model

Every releasable method faces one released sample, repeated black-box sampling
at `n`, `10n`, and `100n` (capped at one million rows), plus downloadable
artifact access. Repeat attacks with only released rows. Run shadow-model
membership, attribute inference, rare-record strata, exact and near copies,
and static artifact inspection. Calibrate false positives on real-vs-real
controls. Apply the same access to all methods. For DP methods, report
epsilon 1, 4, and 10; delta `min(1e-5,1/n^2)`; preprocessing and accountant
assumptions; and the exact implementation. An unaudited implementation gets
no formal guarantee. Empirical attack evidence is distinct from formal DP
and from HIPAA de-identification. [HHS describes Expert Determination and
Safe Harbor](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)
as the two de-identification methods.

## Analysis and handoff

The paired L3 public-core comparison is primary. Missing and timed-out
applicable runs count as no pass and are reported separately. A source-unavailable
method is excluded from the comparison. The analysis script uses dataset
cluster bootstrap familywise 95% intervals and Holm-adjusted paired sign
tests; the conservative Bonferroni intervals must be wholly positive against
every executable compact comparator for a gate-pass superiority claim. MFS-v2
superiority requires at least ten paired passing datasets and a positive
adjusted interval. Otherwise report inconclusive. L2, unconstrained quality,
neural/LLM/DP, and JopeDime are distinct analyses.

Keep host inventories, source/container and dataset digests, scratch usage,
immutable receipts, failure ledger, cost report, sensitivity outputs, and
paper tables. Run the adversarial row-copy, leaky-generator, missing-evidence,
byte-reconciliation, and resume tests, then independently rerun a stratified
subset. Publish only rights-compatible data and artifacts.

## Formula-level novelty check

The 2026-09-29 search queried PTF-v1, null-normalized TSTR/TRTR retention,
paired lower bounds, and related formulas. [TSTR/TRTR utility comparison](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1576290/full)
and [TSTR-to-TRTR ratios](https://cran.nics.utk.edu/cran/web/packages/riskutility/vignettes/riskutility.html)
are prior art. This search did not establish priority for DOPE's exact combined
formula and is not an exhaustive proof of novelty. The defensible contribution
to test is the measured combination of bounded executable size, no shipped
source rows, deterministic sampling, and non-compensable gates. Compactness
alone does not improve privacy.
