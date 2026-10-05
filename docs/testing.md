# Testing, evidence, and budgets

`just fast` runs formatting, locked Rust unit tests, and V1 Python tests.
`just focus codec::tests` runs a reusable narrow Rust check and test filter
while editing; it never replaces the full CPU PR lane.
The CPU workflow pins sccache and preserves the lane's Cargo target cache to
shorten repeated locked checks. Local builds reuse their worktree target.
`just check` adds Clippy with warnings denied and the Rust CLI fixtures.
`just security` scans secrets, dependencies, workflow syntax, and writes an
SPDX SBOM. `just score` regenerates Python boundary evidence, then runs the
full pinned Jankurai audit; `diff-audit` is only a faster local aid.
The audit workflow caches uv downloads in its disposable LAN guest. Audit
dependencies use the pinned uv installer and V1 version constraints; cache
restoration never replaces fresh boundary tests or changes the job quota.
`just fast-audit` writes a changed-source advisory under `target/jankurai/`;
its partial scope can produce boundary or release false positives and never
supersedes the clean full gate.

The CPU PR lane is required. The scheduled GPU lane uses CUDA 12.8, libtorch
2.7, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; it checks repeated training,
resource measurement, and sealed corpus validation. A failed or absent GPU
lane blocks promotion and claims for its affected backend.

The bounded compiler tries at most two micro-TVAE profiles for L3 and three
compact profiles for L2 after a symbolic baseline. A caller can set a shorter
deadline or lower byte cap. Standard neural training is capped at ten minutes
per explicit candidate; GPU admission rejects an estimated footprint above
16 GiB. Fitness diagnostics use at most 128 rows and 32 dimensions per sample.
External learner smoke tests use fixed seeds and one thread per learner; they
do not change Rust certification.

## Cost and time budgets

The CPU PR job has a 35 minute quota; the security and full-audit jobs each
have 30 minutes. The GPU compute spend cap is one scheduled 120-minute job per
week with no automatic retries. A single neural
fit is limited to 10 minutes and 16 GiB of estimated GPU memory. Abort a run
when its timer, memory quota, or sealed-corpus authorization fails; the job
must retain the partial log and mark the affected backend ineligible. The
operator kill switch is cancellation of the GitHub job or interruption of the
local `ops/ci/gpu.sh` process. Do not retry a paid GPU job without recording
the previous failure and an explicit repair in the campaign ledger.
The stop condition is any exceeded timer, memory cap, or missing sealed input.
`agent/cost-budget.toml` is the machine-readable budget; `just check` and the
CPU CI lane run `ops/ci/check-budgets.sh` to compare it with workflow timeouts,
the 600-second compiler cap, GPU memory admission, and required sealed inputs.

## Launch gate evidence

Security evidence is the CI secret scan, dependency advisory scan, workflow lint,
and SPDX SBOM. Backup evidence is the signed prior passing manifest and its
hash-verified bundle. Monitoring evidence is the campaign ledger, certification
JSON, and scheduled GPU job result. Rollback evidence is the restore procedure
in [release.md](release.md#rollback). Abuse controls are the strict input
boundary, byte limits, privacy tests, and allowlisted release inventory. A
production promotion needs all five evidence classes plus the exact source
commit, contract digest, and sealed holdout receipt; missing evidence blocks
promotion. GitHub Dependency Review is unavailable on this repository, so the
independent Cargo advisory scan and strict security wrapper provide the required
dependency gate. A PR can establish the code and fixture gates without claiming a
sealed-corpus release.

Test reports and repair receipts live under `target/`. A failing gate should
record a typed reason and the narrow rerun command. Stop a GPU experiment at
its deadline or when it exceeds its frozen memory budget. No campaign job
opens sealed holdout data outside certification. Do not rerun expensive
campaign matrices merely to repair a local source or fixture failure.

## Local repair

Command errors print a JSON `repair_receipt` with `purpose`, `reason`,
`common_fixes`, `docs_url`, and `repair_hint`. The fields describe a repair
without copying the input value into the receipt; the error line supplies the
row and column when input validation fails.
For data failures, use the reported row and column to fix numeric input;
avoid copying a source field into an issue or report. For an artifact failure,
compare its recorded digest and regenerate it from the same source commit.
For a backend failure, check the build feature and the scheduled GPU report.
Rerun the narrow failing command before repeating a whole campaign.
