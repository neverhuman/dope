# Offline generator benchmark (research only)

This package does not enter the Rust production binary or alter its release decision.
`contract.json` pins a generator-only gate profile and the SHA-256 of the frozen
production KPI contract. A future contract change requires a new version.

`PROTOCOL.md` is the pre-registration. `study-design.json` fixes the cohorts;
`pilot-datasets.lock.json` pins three rights-cleared pilot inputs and their
splits; `methods.lock.json` records audited methods and pending source checks.
The latter is incomplete, so no final evaluation or headline comparison is
authorized by these files. `score.py` requires complete measured gate evidence
and keeps MFS-v2 null otherwise.

The worker receives only training and validation data. The evaluator owns the
test partition. Final evaluation is disabled until the data manifest, method
source/config lock, budget lock, and method–dataset matrix are frozen. Missing
privacy, utility, or attack evidence is a failed gate with a null MFS-v2 score.
An unavailable method is reported as unavailable; it is never a DOPE win.

Use `python3 -m unittest discover -s research/benchmark/tests -v` for the
research package checks. Run the repository's `just fast` and `just check`
before committing. Results and raw inputs belong outside worktrees and `/tmp`;
the worktree's `target/` may hold build and test evidence.

Entry points:

```text
python3 -m research.benchmark.inventory_hosts --output HOSTS.json
python3 -m research.benchmark.fetch_jope DATASET_HASH DATA_ROOT
python3 -m research.benchmark.public_sources Adult ADULT.zip ADULT.csv
python3 -m research.benchmark.california_source CAL_HOUSING.tgz CALIFORNIA.csv
python3 -m research.benchmark.manifest SELECTION.json DATA_ROOT/prepared
python3 -m research.benchmark.select_extension REVIEWED_TASK_CANDIDATES.json
python3 -m research.benchmark.runner JOB.json research/benchmark/methods.lock.json RESULTS_ROOT
python3 -m research.benchmark.pilot_metrics WORKER_DIR SAMPLE.csv
python3 -m research.benchmark.pilot_queue --deadline-utc 2026-09-30T17:13:00Z --execute
python3 -m research.benchmark.pilot_queue --deadline-utc 2026-09-30T17:13:00Z --execute --repair-digest-mismatch
python3 -m research.benchmark.pilot_compact_supplement --deadline-utc 2026-09-30T17:13:00Z --execute
python3 -m research.benchmark.pilot_evaluate Adult dope --fit-seed 11 --sample-seed 101 --multiplier 1
python3 -m research.benchmark.pilot_report --output /mnt/fast-scratch/dope-benchmark/pilot-4h/reconciled-report.json
python3 -m research.benchmark.analysis MATRIX.json GATE_REPORTS.jsonl research/benchmark/methods.lock.json
python3 -m research.benchmark.admission REPO_ROOT LOCK_ROOT
python3 -m research.benchmark.full_queue --repo-root REPO_ROOT --lock-root LOCK_ROOT
python3 -m research.benchmark.real_test_utility WORKER_DIR EVALUATOR_DIR SAMPLE_N RECEIPT_N SAMPLE_4N RECEIPT_4N OUTPUT
```

The runner enforces a 16-core affinity and respects the launcher's GPU
visibility setting (defaulting to GPU 0), hashes worker
inputs, verifies repeated artifact sampling, and resumes receipts without
refitting. The coordinator's JopeDime fetch reads only requested catalog,
manifest, and blob objects through the named AWS profile. Credentials stay on
`xbabe2`. `pilot_metrics` consumes validation, never the test partition.
On `xbabe2`, each fit and sample cell reserves space against the 200 GB scratch
ceiling, counting the whole scratch root (the output root's parent by default).
`admission` exits nonzero while any final source, data, budget, method matrix,
or evaluator lock is absent or incomplete; public test evaluation must call it
before reading test data.
The current worker runs only the common-numeric track. Author-faithful input
preparation and external adapters are still pending and fail closed.
The full queue also fails closed until admission passes. It checks each host's
CPU affinity, load, available memory, benchmark scratch filesystem, GPU
processes, and free VRAM before each dispatch. CPU jobs occupy distinct
16-core slots, up to four per host; GPU jobs use one such slot and one visible
GPU. Attempts, logs, and admission inventories are immutable under benchmark
scratch. A failed attempt remains visible and needs `--retry-failed` for another
attempt. Worker environments disable AWS credential discovery. The real-test
utility entry point checks admission before reading the test partition and
requires matching frozen dataset manifests, fit artifacts, and sample receipts
for `n` and `4n`. Final cells require the complete three-seed, four-size
sample matrix. Each fit identity binds that matrix, the selected configuration,
and any DP budget. Tuned configurations require an eight-trial validation
selection receipt on scratch with a matching SHA-256 and a total wall time of
at most 12 hours. Failed fit and sample attempts remain in attempt directories;
only successful attempts occupy canonical receipt paths, so retries retain the
earlier failure evidence.
It is one endpoint of the unfinished shared evaluator, not a complete gate
report.

`PILOT_STATUS.md` records the four-hour dispatch. `PILOT_METHOD_AUDIT.md`
records source and dependency findings. `pilot_queue.py` freezes all 42 fit
cells, writes blocked receipts, and runs the 12 cells supported by the current
source lock. `pilot_evaluate.py` writes validation-only descriptive vectors.
`pilot_report.py` reconciles original and infrastructure-repair receipts while
retaining every failed attempt in its ledger. The frozen queue package and
all bulk receipts live on benchmark scratch. Pilot-only AIM compatibility jobs
are supplemental and do not alter the original 42-cell matrix.
`pilot_compact_supplement.py` runs the two already locked reference compact
methods on the pilot datasets with separate receipts and a separate matrix.
`PILOT_COST_REPORT.md` records measured costs, validation-only vectors, the
historical 14-day capacity bound, and the full-campaign admission decision.
The later protocol amendment treats day 14 as a reporting milestone and sets
the new full-campaign scratch ceiling to 200 GB. Historical pilot usage and
its original 100 GB limit remain as recorded in the pilot cost report.
The final public test remains closed until `admission.py` passes.

`PREPILOT_COST_REPORT.md` records the three-host DOPE probe and the first
GaussianCopula cost cell. It is not the seven-method pilot or a benchmark
comparison. The runner charges the projection map in raw artifact bytes and
archives DOPE compile sidecars separately as fit evidence.
