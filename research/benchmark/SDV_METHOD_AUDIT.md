# CTGAN and TVAE author-library audit — 2026-10-01

Both adapters use CTGAN 0.12.1 at author release commit
`826da23f8f9385ad15fd206ecad691e04cb0ccdc`. The installed Python sources were
compared byte for byte with the pinned author archive before the S3 round
was frozen. The archive SHA-256 is
`ca63518ec09ee1d70e0d8343ad71c75de0c6e610c00299f95d0f0da9d426683c`.
The [author license](https://github.com/sdv-dev/CTGAN/blob/826da23f8f9385ad15fd206ecad691e04cb0ccdc/LICENSE)
is BUSL-1.1; this use is non-production benchmark research. The bibliography
key for both models is `xu2019modeling`, from the corrected local bibliography.
These are wrappers around the author implementation, not independent model
reimplementations.

`sdv-runtime.lock.json` pins the runtime package versions. The immutable round
lock additionally hashes installed dependency files, adapter and runner source,
the dataset cohort, each worker partition, and every configuration. Dependencies
are shared from benchmark scratch; the three hosts retain their local Torch
installation and record their GPU/driver inventory at each admission.

## Inputs and native objective

The admitted representation is common-numeric regression. Every method sees
the same fit partition derived from the official S3 training data. Validation
also comes only from that official training partition. The entry point rejects
evaluator directories and files named `test.csv`. Both partition hashes and the
projection hash must match the frozen cohort before a fit starts.

The native selection objective is the prespecified arithmetic mean of the
author-library SDMetrics `LinearRegression` and `MLPRegressor` raw R² efficacy
scores on real validation rows. The two component metrics and their defaults
come from [SDMetrics regression efficacy](https://docs.sdv.dev/sdmetrics/data-metrics/metrics-in-beta/ml-efficacy-single-table/regression),
version 0.32.0, author commit `6e75fb1ac4469152abe4dabebeb87eb6828d10a5`.
The arithmetic mean is the study's aggregation choice. It is not claimed to
be an author-provided tuning rule. The auditor RNG seed is 1729. Selection
maximizes that value, then minimizes artifact bytes and configuration SHA-256.
DOPE retention and privacy diagnostics do not enter the comparator selection.

The grid retains the literal author defaults, including 300 epochs, and varies
only author-supported `batch_size` (500 or 100) and `embedding_dim` (128 or 64).
That gives four configurations, including the default, per method and dataset.
The grid is a study choice; no author-published optimum is asserted.

## Fit, export, and sampling

Fitting requires CUDA and seeds Python, NumPy, and Torch. The child process has
a 600-second deadline, a 16-GiB Torch allocator limit, and an independent device
memory monitor. Each host reserves one GPU fit and 16 CPU cores, checks other
GPU owners and available RAM, and checks the 200,000,000,000-byte scratch limit
before each attempt. This queue is separate from the weekly scheduled GPU CI.

The artifact contains `model.pt`, `model.json`, and the charged projection map.
CTGAN 0.12.1 does not retain its original training matrix in `DataSampler`, but
it retains row-index lists for training. The wrapper removes those lists and
both models' loss histories before saving. Sampling uses the saved transformer
and generator on CPU with a fixed sample seed. It clips numeric values to the
common [0,1] representation and reports the clipping count. The generic names
inside the model are study column identifiers, not original source headers.

Generated-fixture GPU checks passed for both models. Repeated sampling from
each saved artifact produced identical CSV hashes. Additional real S3 artifact
checks reproduced an xbabe3 CTGAN sample on xbabe2 and an xbabe2 TVAE sample on
xbabe3 with identical hashes. Detailed receipts remain under
`target/sdv-adapter-smoke-20261001/` and scratch `neural-native-v1/cross-host/`.
This is artifact contract evidence; it does not establish all privacy gates.

The completed matrix has 96 successful fits. A subsequent audit froze four
same-host repetitions: both author defaults on the narrowest and widest cohort
datasets, selected by feature count, then row count and dataset hash. Every
repeat reproduced the complete artifact inventory and sample hashes exactly;
all native KPI differences were zero (prespecified tolerance `1e-9`). These
four audit fits are separate cost evidence and do not replace selection trials.
`reproduce_sdv --freeze` writes the audit lock; execute its frozen script with
the original round package and dependencies to reproduce it. The original
selections remain unchanged.

## Frozen research scope and reproducibility

The round has 96 planned fits: two methods, the existing twelve discovery and
confirmation datasets, and four configurations. It uses fit seed 11, sample
seed 101, and `n` rows for native validation. It is not the final five-fit-seed,
three-sample-seed, four-size matrix. All failures and timeouts remain attempts.
Retries retain the job identity and count toward the eight-attempt, 12-hour
cell cap. Native selections are also exported in the shared runner's selection
format and checked by its native-objective admission logic.

The older method lock is preserved at `method-locks/2026-09-30.json` so the four
previous compact validation result sets regenerate unchanged after adding the
two neural methods. Their underlying adapter and objective source is unchanged.

Run research checks with `python3 -m unittest discover -s research/benchmark/tests -v`.
Use `python3 -m research.benchmark.publish_sdv_validation --evaluate --publish`
to verify scratch receipts and publish a complete validation comparison.
Use `python3 -m research.benchmark.publish_sdv_validation --render-existing`
to regenerate its CSV and SVG/PDF from the committed JSON alone.

The selected models are evaluated on the validation data used for their native
selection, so these estimates can be optimistic. Official public and S3 tests
remain unopened. Production PTF-v1 and MFS-v2 remain null, and this round cannot
support a paper superiority claim or a production certification.
