# Forest-Flow: first two complete five-fit validation cohorts

Ten actual author-default fits on two frozen rights-cleared lineages use fit
seeds 11, 23, 37, 53 and 71. Each fit has all six common evaluations: n and 4n,
with sample seeds 101, 211 and 307. This publication contains all sixty cells
and ten native author-ML objective audits. It is a two-lineage cohort; the
100-lineage default/native-selected five-fit matrix is incomplete.

Samples within one fit are averaged first. The reported variation is the
unbiased sample standard deviation of the five independent fit means.
The fifteen samples are not fifteen independent fits. Error bars are SD,
not confidence intervals. No population superiority or production score is
reported; MFS-v2, PTF-v1, release-safe L3 and superiority remain null.

## Measured 4n outcomes

| Frozen lineage | Marginal KS/TV mean | CatBoost C2ST AUC mean | CatBoost retention mean | Between-fit SD | Utility support | Native mean R² | Native between-fit SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0d482c1dd7cc6592 | 0.240856 | 0.599524 | 1.036965 | 0.009010 | 5/5 | 0.809161 | 0.043892 |
| 19f4780b53b3fa41 | 0.204986 | 0.567143 | null | null | 0/5 | -1.051174 | 0.353522 |

Common outcomes use the same frozen evaluator as the earlier matched panels.
The native objective is the study implementation of the author four-model,
five-auditor-seed mean R² formula. Native values are reported without comparing
objectives across methods. There is no new tuning or shared-KPI selection.
Negative native R² and unclipped utility retention are retained. The second
lineage has uninformative utility; its missing utility mean and SD stay null,
with 0/5 measured fits. Fidelity, detection and empirical privacy remain visible.

## Checkpoints, bytes and costs

`checkpoint-kpis.csv` lists each original sampler hash, its bytes, charged
model plus projection and adapter bytes, fit receipt, native R² and CatBoost
TSTR loss and retention at n/4n. No weights or samples are committed.
The ten checkpoints charge 1,803,510,426 bytes in total; all exceed
10,240 bytes and are unconstrained comparison artifacts.

Summed generator fit seconds: 422.644759. Summed whole-fit process
seconds: 549.828167. Fit time is nested inside whole-fit time;
these clocks are not additive. Summed native-audit seconds: 69.343535.
Summed common per-cell numerical seconds: 29.609683. The two actual common
evaluator process wall times are 26.747209 and 24.075352 seconds. Process wall includes
startup and runtime checks; it must not be added to per-cell numerical time.
These are measured cohort costs, not full-population throughput forecasts.

## Evidence and reproduction

`manifest.json` pins the complete publication input, both metric receipt locks
and process exits, ten fit/native receipts through `panel.json`, source versions
and all output files. `original-verification.json` records a read-only producer-
host hash check of 84 original files, including all ten models. It did not load
pickle or rerun a fit. The check includes declared source files and runtime
inventory JSON, not a fresh recursive check of every dependency binary.

The publisher verifies frozen receipt digests before metric reads, frozen job,
fit, split, configuration and native-audit joins, complete byte inventory and
the original checkpoint hash proof. Replay with the pinned input SHA:

```sh
PYTHONPATH=. python3 -B research/benchmark/publish_forest_fit_variance.py \
  --inputs /mnt/fast-scratch/dope-benchmark/forest-first-two-fivefit-publication-v1/inputs-v6.lock.json \
  --inputs-sha256 675ecf57d3950f71dcb1af0acef9916201352c90b2565aebcb6e73dcd8119ef1 \
  --output target/forest-fivefit-replay
python3 -B research/benchmark/plot_forest_fit_variance.py target/forest-fivefit-replay
python3 -B -m unittest research.benchmark.tests.test_forest_fit_variance -v
```

The public job identity names the worker partition digest `worker_partition_sha256`;
restore its original name `worker_key` before checking the canonical job digest.
This naming projection changes no frozen receipt or job identity.

The source input also pins durable metadata mirrors. Raw data, samples and model
checkpoints remain on their original storage roots. The initial failed sampling
and transport launches are retained in the earlier scratch histories and do not
contribute fits or scores. Official tests remain sealed. Real-vs-real controls,
projection-only utility cost and complete certification privacy evidence are
not supplied by this cohort; no formal DP or release claim follows from its
empirical privacy metrics.
