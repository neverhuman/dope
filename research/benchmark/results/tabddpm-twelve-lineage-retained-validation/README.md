# Retained TabDDPM validation samples

This measured research cohort contains 21 retained model records on 12 S3
lineages at fit seed 11. All 126 physical comparison samples have common
fidelity, utility, detection and empirical privacy receipts: `n` and `4n`,
sample seeds 101, 211 and 307. No generator was fitted or sampled again.
The official S3 tests remain sealed. This is separate from the unfinished
100-lineage population publication and supplies no production certification.

## Configurations and native objectives

The original author-default and native-selected configurations are retained.
TabDDPM selection maximized its frozen author objective: CatBoost validation
R² averaged over synthetic sample seeds 0–4. Common outcomes selected nothing.
The five-configuration grid, original selections and native values are traced
to immutable receipt hashes. Native R² values are not ranked against another
method's native KPI.

The current reporting snapshot contains 10 default and 12 native-selected
lineages. Nine native-selected lineages have only 2–4 successful native
receipts out of the five declared configurations. Three have all five.
This is selection from the originally available successful pool, not evidence
that every tuning trial succeeded or that the whole campaign finished.

One lineage has two historical native-selection snapshots. Both physical
models and their measured cells remain visible. The reporting frontier uses
the latest **original frozen round** for that lineage, independent of common
outcomes. Its earlier selected-default alias remains a default observation.
Two originally selected-default aliases reuse a physical model; they add no
fits. After resolving the reporting frontier, 132 logical metric rows join
126 physical measurements. No sample replicate is an independent generator fit.

The default and native-selected cohorts differ. Their unpaired dataset medians
cannot establish a tuning benefit, cross-method ranking or superiority.
Within each dataset and size, the three sample values are averaged before
taking a dataset median. Missing or uninformative outcomes retain null values
and explicit support counts. No value is clipped to create a passing score.

## Measured outputs

| Reporting cohort | Size | Lineages | CatBoost retention | Marginal KS/TV | CatBoost C2ST AUC | Distance MIA AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Author default | n | 10 | 0.875401 | 0.128015 | 0.578937 | 0.531279 |
| Author default | 4n | 10 | 0.943636 | 0.117544 | 0.595042 | 0.525868 |
| Original native selected | n | 12 | 0.914907 | 0.103867 | 0.584347 | 0.551466 |
| Original native selected | 4n | 12 | 0.973885 | 0.099022 | 0.579738 | 0.571253 |

These are descriptive training-derived validation outcomes, each with full
support for the table's cohort. Alpha precision/beta recall, pairwise numeric
and categorical fidelity, logistic detection, DCR, NNDR, DOMIAS and all three
utility auditors are in the per-cell JSON and scalar CSV. Unsupported outcomes
remain null. Attack AUCs are empirical measurements, not formal privacy claims.

`checkpoint-kpis.csv` lists every model record's native R², charged bytes,
native receipt and digest, and CatBoost synthetic-trained validation MSE at
n/4n. `figure-kpis.csv` and `validation.svg` regenerate from `panel.json`.
All 21 artifacts exceed the L3 10,240-byte cap; their recorded total is
10,873,137 bytes. These results supply no release-safe DOPE wins.

The new common evaluator process exited 0 in 165.673494 seconds. Summed
per-cell numerical evaluation time was 159.291646 seconds. Sampling and native
audit clocks are separately recorded; overlapping clocks are not added.
The CPU job used cores 58–59, a 4 GiB memory limit, zero swap, one thread per
library, low CPU/I/O priority, and the 15% available-RAM floor. Measured peak
process RSS was 487,759,872 bytes. No GPU operation ran for this publication.

## Provenance and reproduction

`inputs.lock.json` hashes the completed metric lock, actual process closure,
pre-metric original-host verification, and mirrored original requests,
operations, rounds and native selections. The original verification read
artifact payloads for hashing without loading models and found no source drift.
Historical operation schemas that omitted a test flag remain labeled; their
training-derived worker/request/round hashes establish the input partition.
The original interpreter/runtime chain is not recertified by this publication.

Weights, generated samples, raw rows and detailed logs stay on scratch.
Only opaque identities, metric scalars, hashes, byte counts and receipt paths
are committed. Earlier prototype verification failures remain on scratch;
no original round, request or receipt was overwritten.

From an environment with the immutable receipt mirrors available:

```sh
python3 -B -m research.benchmark.publish_tabddpm_retained \
  --inputs /mnt/fast-scratch/dope-benchmark/tabddpm-retained-common-publication-v1/inputs.lock.json \
  --sha256 e245b125ffbb29b1685356b9b58659645c890fcbb4fb023c10f1137167a669e8 \
  --output target/tabddpm-retained-replay
python3 -B -m unittest research.benchmark.tests.test_tabddpm_retained -v
```

The publisher verifies original metadata before metric reads, joins every
fit/configuration/split/sample identity, preserves native-selection versions,
and regenerates all scalar outputs. MFS-v2, PTF-v1, release-safe L3 and
superiority stay null. Full-population, five-fit and native-search completion
remain false; source-unavailable and unfinished cells contribute no DOPE win.
