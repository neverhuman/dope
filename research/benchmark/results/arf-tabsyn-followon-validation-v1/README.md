# ARF five-fit and retained TabSyn follow-on validation

`panel.json` publishes 6,000 ARF logical cells on the same 100 lineages,
default/native-selected configurations, fit seeds 11/23/37/53/71, n/4n,
and sample seeds 101/211/307. There are 5,892 distinct physical metric
receipts; 108 logical aliases are not independent replicates. Configuration
selection remains the earlier native validation-density selection.

The additional 95 TabSyn cells use scaled author defaults, fit seed 11,
and n-sized samples on 41 other lineages. Nineteen groups have all three
sample seeds; 22 groups have only one or two. Their individual measurements
are published, but incomplete groups have null means. Together with the
earlier eight-lineage/48-cell publication, this is 143 measured cells on
49 lineages, 35 complete three-sample groups, and 22 partial groups. Only
the earlier eight lineages have complete n/4n schedules. No native TabSyn
winner or five-fit/full-population TabSyn completion is implied.

Each scalar row carries its original metric receipt path, byte count,
SHA-256, canonical metric job, input hashes, charged model-plus-projection
bytes, and logical configuration identity. Seed-11 ARF aliases retain
their original alias manifest and pointer. Export rechecked original
metric bytes before decoding and independently rechecked those aliases'
fit/sample receipts and shared numeric inputs. The import lock pins the
ARF closure and coverage verification and TabSyn input/metric/exit locks.
It records the common evaluator source hashes. Sampling backends remain
per-cell provenance, with the new TabSyn samples generated on CPU.

The upstream job field `worker_key` contains a SHA-256 lineage digest, not
a credential. Public scalar exports spell it `worker_sha256`; the reducer
restores the original field name before checking the canonical job hash.
This lossless projection changes no original receipt, job digest or metric.

`scalar-cells.jsonl` is a rights-safe projection, not a copy of raw
training/validation rows, generated samples, model weights, or logs.
Portable regeneration authenticates this frozen projection, validates the
exact schedules and aliases, and reduces the already measured scalars:

```sh
python3 -B -m research.benchmark.publish_arf_tabsyn_followon --check
python3 -B -m unittest research.benchmark.tests.test_arf_tabsyn_followon
```

Raw metric receipt hashes cannot be reconstructed from these excerpts.
Original receipts remain at the immutable paths in the result; current
bulk model weights were not rehashed during this publication. Regeneration
does not run scientific auditors or load generators. The CSV is an exact
per-cell reduction of the same frozen scalars.

For ARF, samples are averaged within each fit, then unbiased sample SD is
computed across five independent fit means. Missing auditor support yields
null aggregate mean/SD rather than a partial-seed average. Fit SD is not a
confidence interval or a paired superiority test. Dataset cohorts and n/4n
sizes remain separate in the paper's descriptive figures.

This publication performs no new fitting, sampling, or auditor evaluation.
Earlier failures and retries remain in original attempt receipts; this
deduplicated scientific matrix does not erase their costs or count their
duplicate samples as independent observations. The TabSyn evaluator wall
time is separately pinned; overlapping clocks are not added together.

All observations are on training-derived validation. Official tests remain
sealed, empirical attacks establish no formal DP claim, and MFS-v2, PTF-v1,
release-safe L3, and superiority remain null.

`operations-checkpoint.json` separately freezes two Forest-Flow queue
observations and their conditional disposition/evaluation forecasts. These
are operational forecasts, excluded from scientific KPI rows and compute
cost totals. The append-only source log is bound by a verified byte-prefix
hash; later observations cannot change this published checkpoint.
