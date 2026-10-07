# Forest-Flow: six additional complete five-fit validation cohorts

Thirty measured author-default checkpoints on six frozen rights-cleared S3
lineages have fit seeds 11, 23, 37, 53 and 71. All 180 common sample cells
are complete: n and 4n, with sample seeds 101, 211 and 307 per fit.
Thirty native author-ML efficacy audits are also complete. The six cohorts
were selected by checkpoint size before their common outcomes were read;
they are not a representative estimate of the complete population.

The 24 existing new-seed checkpoints were reused. A scoped, canonical-job
lookup of 31 earlier fit receipts found no matching seed-11 checkpoint in
the declared roots, so six missing seed-11 author-default fits were run.
No existing OK fit was duplicated. There was no tuning or common-KPI selection.
Official training-derived validation partitions are used throughout; official
tests remain sealed. The full default/native-selected population is incomplete.

Three sample outcomes are averaged within each fit first. The reported mean
and unbiased sample SD then use the five independent fit means. SD is not
a confidence interval; fifteen samples are not fifteen independent fits.
MFS-v2, PTF-v1, release-safe L3 and superiority remain null.

## Measured 4n outcomes

| Frozen lineage | Marginal KS/TV mean | CatBoost C2ST AUC mean | CatBoost retention mean | Between-fit SD | Native mean R² | Native between-fit SD |
|---|---:|---:|---:|---:|---:|---:|
| 04c80944e971cb3e | 0.174996 | 0.540491 | 0.994113 | 0.007586 | 0.907587 | 0.004578 |
| 0779d3fd5b3133a2 | 0.133158 | 0.589525 | 1.002084 | 0.000654 | 0.721580 | 0.003935 |
| 585fef2a1cb6b33f | 0.190648 | 0.690905 | 1.169079 | 0.056461 | 0.519521 | 0.149122 |
| 693890aae9dc860b | 0.165360 | 0.590770 | 1.003079 | 0.001539 | 0.876528 | 0.003074 |
| a04f964bc53281e0 | 0.157589 | 0.601334 | 1.022075 | 0.203821 | -0.502585 | 0.812460 |
| a863cd42be4d05d4 | 0.219861 | 0.548817 | 1.005806 | 0.001405 | 0.944271 | 0.008114 |

The native objective is the study implementation of the author's four-model,
five-auditor-seed mean R² formula. Its values are not ranked against other
methods' native KPIs. Negative R² and unclipped retention are preserved.
These are author-default outcomes, not native-selected configurations.
The common evaluator is the same A952 fidelity/utility/detection/empirical-
privacy implementation used for the matched panels. Missing or uninformative
utility stays null; none is silently dropped from a fit mean or SD.

## Checkpoints and measured costs

`checkpoint-kpis.csv` lists every sampler SHA-256, sampler bytes, charged model
plus projection and adapter bytes, original fit receipt, native R², and common
CatBoost TSTR loss and retention at n/4n. The checkpoints charge 2,738,956,952 bytes.
All exceed the 10,240-byte production cap and are unconstrained baselines.
Weights, samples and restricted rows remain on their original storage roots.

Summed generator fit seconds: 514.524763. Whole-fit seconds: 874.233495.
Native-audit seconds: 180.722104. Common numerical seconds: 98.996145.
Common process wall times: 88.284476, 23.207538 seconds.
Nested numerical and enclosing process clocks are not additive.
The six new seed-11 fits took 181.677683 coordinator wall seconds; this is
not an estimate of full-population throughput. The initial empty-attempts-
directory launch failed before any fit; the initial metric launch failed
its CPU-affinity assertion before any metric cell. Both histories remain
on scratch and are not counted as method failures or successful work.

## Timeout accounting

The original 400-fit CPU queue is still running. Its eight earlier 600-second
fit timeouts span lineages 297b7688a045c583 and 48b7a3bf81e5f3e5 at seeds
23, 37, 53 and 71. `timeout-dispositions.json` is the frozen 12:05Z snapshot:
128 closed attempts, 120 OK and eight timeouts. It is not a current total.
Each timed-out attempt remains individually visible with worker exit 124,
original fit/job/closure/admission receipt digests, no accepted artifact,
and no native/common score. Measured elapsed time is null because those
receipts do not contain it; the 600-second cap is not substituted for a
measurement. There is no automatic retry or method-quality conclusion.
Any admitted continuation must retain the logical identity, use a new physical
attempt and preserve the eight-trial/12-hour budget and 600-second fit cap.
These timeout lineages are outside this complete six-lineage cohort.

## Reproduction

`manifest.json` pins the immutable input, metric receipt locks and exits,
producer versions and all outputs. The original checkpoint replay re-hashed
225 producer-host files, including all 30 models, source files and runtime
inventory JSON, without loading pickle or rerunning fits. It does not claim
a fresh recursive verification of every dependency binary.

```sh
PYTHONPATH=. python3 -B research/benchmark/publish_forest_fit_variance.py \
  --inputs /mnt/fast-scratch/dope-benchmark/forest-next-six-fivefit-publication-v1/inputs.lock.json \
  --inputs-sha256 3d86aac009db3133997a61f6f2d8a436e90d2c26acee2669d433bd1a01f91dc8 \
  --output target/forest-next-six-replay
python3 -B research/benchmark/plot_forest_fit_variance.py target/forest-next-six-replay
python3 -B -m unittest research.benchmark.tests.test_forest_complete_cohort -v
```

Receipt hashes are checked before metric reads, with exact frozen job, fit,
split, configuration, byte-inventory and native-audit joins. The public job
identity projects `worker_key` as `worker_partition_sha256`; restore the
original field name to check its canonical job digest. This changes no receipt.
Real-vs-real controls, projection-only utility cost and full certification
privacy evidence remain missing. Empirical privacy metrics are not formal DP.
