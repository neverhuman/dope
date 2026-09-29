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
python3 -m research.benchmark.runner JOB.json research/benchmark/methods.lock.json RESULTS_ROOT
python3 -m research.benchmark.pilot_metrics WORKER_DIR SAMPLE.csv
python3 -m research.benchmark.analysis MATRIX.json GATE_REPORTS.jsonl research/benchmark/methods.lock.json
python3 -m research.benchmark.admission REPO_ROOT LOCK_ROOT
```

The runner enforces a 16-core affinity and one visible GPU, hashes worker
inputs, verifies repeated artifact sampling, and resumes receipts without
refitting. The coordinator's JopeDime fetch reads only requested catalog,
manifest, and blob objects through the named AWS profile. Credentials stay on
`xbabe2`. `pilot_metrics` consumes validation, never the test partition.
On `xbabe2`, each fit and sample cell reserves space against the 100 GB scratch
ceiling, counting the whole scratch root (the output root's parent by default).
`admission` exits nonzero while any final source, data, budget, method matrix,
or evaluator lock is absent or incomplete; public test evaluation must call it
before reading test data.

`PREPILOT_COST_REPORT.md` records the three-host DOPE probe and the first
GaussianCopula cost cell. It is not the seven-method pilot or a benchmark
comparison. The runner charges the projection map in raw artifact bytes and
archives DOPE compile sidecars separately as fit evidence.
