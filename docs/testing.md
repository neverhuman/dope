# Testing, evidence, and budgets

`just fast` runs formatting, locked Rust unit tests, and V1 Python tests.
`just focus codec::tests` runs a reusable narrow Rust check and test filter
while editing; it never replaces the full CPU PR lane.
The CPU workflow pins sccache and preserves the lane's Cargo target cache to
shorten repeated locked checks. Local builds reuse their worktree target.
CPU guest setup installs Rust, Just, TeX, Poppler, and ripgrep in its existing
early package transaction, retaining the 300-second package-lock wait and
35-minute job budget. `cpu.sh` checks all required tools before tests and never
starts a late package transaction. `test-cpu-tool-readiness.sh` exercises each
missing-tool refusal against the real CPU entry, before any build or replay.
`just check` adds Clippy with warnings denied, the Rust CLI fixtures, and the
raw-schema, raw-workflow, artifact-only, installer, byte-allocation, native
successor, public-demo, and frozen-geometry binding Rust integration suites.
Hosted CPU retains full
locked Cargo test discovery, which also executes every one of these suites.
Fast/check and hosted CPU verify the shipped native legal-input hashes before
product checks. The full installer suite covers license delivery, fixed-member
uninstall validation and legacy two-file compatibility. Offline notice generation
uses cached locked crate archives and pinned Rust legal inputs; it is separate
from checking the shipped pins.
Local fast/check and hosted CPU run the six bounded public-demo regressions in
`research.benchmark.tests.test_public_demo`: frozen source admission, disjoint
physical groups, query-label independence, integer producer/reader compatibility,
safe native diagnostics, and honest failed-step/report output.
The actual native raw-schema, raw-workflow, artifact-only, and installer suites
remain auto-discovered by hosted CPU's existing locked Cargo test invocation.
The native guard's required fast/check and CPU routes share
`ops/ci/prepare-merge-guard.sh`: explicit locked full-graph metadata preparation
for both manifests, at most 120 seconds each with no retries. Setup may fetch
current locked public crates; the subsequent feature-boundary proof stays
offline and unfiltered. Metadata JSON lives under ignored target output, and
Cargo failures retain bounded stderr without exposing it in GitHub receipts.
Hosted CPU also runs the complete offline `research/transfer/tests/test_*.py`
discovery once, using its provisioned V1 interpreter after all fixture packages
are installed. Temporary files stay in `target/transfer-test-tmp`, and native
learner fixtures use one CPU thread. These fixtures preserve identity, coverage,
statistics and adapter contracts; they do not qualify official vendor weights
or a protected scientific campaign. Local fast/check retain their full discovery.
Local fast/check also run the bounded Rust include-closure fixture against the
actual fitter. Old and split module layouts must produce the same architecture
macros; missing, escaping, symlinked or conflicting constants fail closed.
Hosted paper-script discovery retains that fixture, and the CPU review-fix
entry includes the unknown-method and duplicate-identity receipt regressions.
`just security` scans secrets, dependencies, workflow syntax, and writes an
SPDX SBOM. `just score` regenerates Python boundary evidence, then runs the
full pinned Jankurai audit.
The audit workflow caches uv downloads in its disposable LAN guest. Audit
dependencies use the pinned uv installer and V1 version constraints; cache
restoration never replaces fresh boundary tests or changes the job quota.
`just fast-audit` writes a changed-source advisory under `target/jankurai/`;
its partial scope can produce boundary or release false positives and never
supersedes the clean full gate.

`just paper` regenerates every generated paper input, figure, README benchmark
block, and PDF from committed, SHA-pinned ledgers; `just paper-check` verifies
macros against those ledgers, figure hashes, the README block, and the build.
Both recipes enter `paper_build.py` once after their existing legacy preparation
and drift checks. The direct `build_pdf.sh` entry creates its own fresh admission.
The independent unit gate is `just paper-tests`: it retains the entire former
paper-script discovery suite, including the expensive committed-input replay,
and adds the bounded pipeline/recipe regressions. Full paper validation requires
both `just paper-tests` and `just paper-check`; moving the suite does not waive it.
CPU CI runs that same required named gate with its provisioned V1 interpreter:
`target/ci-python/bin/python -B -m unittest discover -s docs/whitepaper/scripts -p 'test_*.py'`
followed by
`target/ci-python/bin/python -B -m unittest discover -s research/benchmark/tests -p test_paper_build_pipeline.py -v`.
The later paper-number and paper-log unittests also use that provisioned V1
interpreter, preserving their NumPy-dependent claim-registry imports.
These replace the selective paper-script unittest calls in `ops/ci/cpu.sh`; the
independent committed-input replay remains exactly once in that suite.
The recipe fixture executes Just itself. CPU CI provisions the repository's
existing Just tool with the other tools before the test stage.
The real recipe regression lives in `test_paper_packaging.py`, so that required
paper-script discovery invokes it exactly once after CPU provisions Just.
The benchmark boundary's full discovery retains the bounded driver tests;
it does not require Just to run the paper recipe regression. Moving that
regression preserves its actual Just execution, reducer replay, old-path
rejection, and four fixture PDF checks. The focused invocation is
`PYTHONPATH=docs/whitepaper/scripts python3 -B -m unittest test_paper_packaging -v`.
`bash ops/ci/paper-clean.sh` is the CI form: it deletes `docs/whitepaper/generated`,
runs `just paper`, and requires a zero diff. `bash ops/ci/check-review-fixes.sh`
runs the review-fix and predeclare_v2 statistics, binder, and sealed-gate fixtures.
`bash scripts/ci-local.sh cpu|security|audit` reproduces the hosted lanes locally.

Admission captures the verified lock bytes without rereading them as a new pin.
The driver replays reducers and renders expected bytes from captured code and
inputs in one private snapshot child; different reducer contracts cannot share
a completion merely because they use the same panel. TeX, anonymization, the
architecture sidecar, and final paper checks consume a captured input tree under
the lane's reused `target/paper-build`. Source identities are checked around
children, including restored writes. This is trusted-driver control flow with
one writer per private tree, not an OS sandbox against a principal able to modify
that tree or substitute the interpreter, TeX installation, or loaded libraries.

Fresh `--check` compares review/auxiliary bytes without rewriting them, then
compiles all four PDFs and checks drift including the architecture outputs.
A local receipt may report `status=passed` with `cold_budget=unmeasured`.
CPU CI still has two independent full replays: the committed-input unit test
and its standalone authenticated PDF entry. Broader unit coverage and stronger
snapshot admission may affect that lane's cost. Canonical-driver fixture proof
does not qualify either full CPU or paper job inside the unchanged 35-minute
cap; setup, unit tests, compilation and drift need separately admitted measured
job proof. The historical cancellation remains failed, without unchanged retry.

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
have 30 minutes. Disposable guest package installs wait at most 300 seconds
for an existing dpkg lock, within those job quotas, then fail if it remains held.
The GPU compute spend cap is one scheduled 120-minute job per
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
