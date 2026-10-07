# Five additional Forest-Flow fit-seed cohorts

This research receipt reports 25 author-default fits on five additional S3
lineages, with fit seeds 11/23/37/53/71, sample seeds 101/211/307 and n/4n.
All 150 common-metric cells and all 25 native audits are measured. Official
S3 tests remain sealed. This is a complete five-fit cohort, not the unfinished
100-lineage population, native-selected track or production certification.

## Between-fit utility

Within each fit, average the three sample replicates. Across five independent
fit seeds, report the mean and sample standard deviation. The standard
deviation is not a confidence interval; the 15 sample replicates are not
15 independent fits. All fidelity, detection and empirical privacy metrics
retain the same nested aggregation and explicit null/support counts.

| Frozen lineage | 4n CatBoost retention mean | Between-fit SD | Measured fits |
| --- | ---: | ---: | ---: |
| `552665868f8f72ed` | 1.000559 | 0.000112 | 5/5 |
| `7423783f7e38bb9b` | 0.999988 | 0.017442 | 5/5 |
| `ce837ccd24a3a18c` | 1.003756 | 0.000482 | 5/5 |
| `e18350fb51a9d1b0` | 0.854074 | 0.098261 | 5/5 |
| `f652ea96eff62654` | 1.029175 | 0.044354 | 5/5 |

These availability/resource-defined cohorts were not selected by common
outcomes. They do not establish a population ranking or superiority. Values
are not clipped. All five lineages share the current CPU author-default
implementation. A sixth lineage in the original metric batch has a legacy
GPU seed-11 fit; that lineage remains in its original immutable batch and is
excluded from this homogeneous five-fit projection. No known-OK fit was
repeated to replace it.

## Native objective and receipts

Forest-Flow's native audit is the study implementation of the author's
four-model, five-auditor-seed mean validation R² formula, pinned in the
original source audit. It is reported independently of common retention.
No tuning trial, native winner or default configuration was changed for
this publication. `checkpoint-kpis.csv` lists each retained model's native
R², charged bytes, CatBoost n/4n validation MSE and receipt/hash.

The publisher verifies the complete original metric batches before the
explicit five-lineage projection, including the preserved excluded lineage.
Original native locks remain unchanged. Original-host proof hashes all 25
checkpoint inventories without loading pickles, plus frozen fit/job/close/
source and native references. It does not certify dynamic runtime closure.

## Costs and scope

Total charged artifact bytes: 2,886,122,766; every fit exceeds 10,240 bytes.
Measured generator fit seconds: 626.661868; whole fit-worker seconds:
930.078232; native audit seconds: 152.866496. These clocks overlap and must
not be added. Sampling/common process clocks remain separately traced to
their actual closures. No timeout receives an inferred measured duration.

Restricted rows, models, samples and detailed logs remain on scratch. JSON,
scalar CSVs, SVG/PDF and hashes are the public evidence. MFS-v2, PTF-v1,
release-safe L3 and superiority remain null. No release-safe DOPE wins or
full-population/native-selected completion is asserted. Paper sources are
unchanged.

## Reproduction

The frozen input is `inputs.lock.json` (SHA-256
`abde27da0caa541c2477ce1d36cce172ee03a1c68e86cac00935db1149088a62`).
The original metric batches contain 174 physical cells; all are verified before
projecting these 150 reporting cells. Whole process clocks cover the original
batches, including excluded cells, and are not cohort-only elapsed time.

```sh
PYTHONPATH=. python3 -B research/benchmark/publish_forest_fit_variance.py \
  --inputs /mnt/fast-scratch/dope-benchmark/forest-third-cohort-five-fivefit-publication-v1/inputs.lock.json \
  --inputs-sha256 abde27da0caa541c2477ce1d36cce172ee03a1c68e86cac00935db1149088a62 \
  --output target/forest-third-five-replay
python3 -B research/benchmark/plot_forest_fit_variance.py target/forest-third-five-replay
python3 -B -m unittest research.benchmark.tests.test_forest_projected_cohort -v
```

Previous two-lineage and six-lineage numerical panels remain exactly
reproducible. Their manifest producer pins identify the current compatible
publisher. Original fits, immutable source rounds and receipt histories
remain unchanged. Real-vs-real controls, projection-only utility costs and
full certification privacy evidence remain incomplete; no gated score is filled.
