# J08 — Expose bounded JopeDime search as a DOPE downstream auditor

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P2**. Dependencies: J05, J07, D06. Cross-repository prerequisites: D06 frozen DVS roster/receipt and FIT-TUNE-TEST contract.

## Objective and non-goals

Let corrected DOPE evaluation assess synthetic artifacts with JopeDime as a fixed downstream learner family. Search/fit on the declared training arm, then score unchanged on common real test rows. This is an auditor integration, not artifact-search product rollout, retraining, live serving adoption or a SOTA/privacy claim.

## Ordered reads and ownership

Read D06 DVS metric/receipt/selection definitions; `crates/jope-bench/src/eval.rs` and `docs/same-cut-bench.md`; `crates/jope-pipeline/src/{run.rs,prepare.rs,receipt.rs}`; J05 NC1 schema/view loader; J07 SDK boundaries; current external command timeout/output/resource patterns. Read exposed-cohort registry before any future qualification.

Jope worker owns new `crates/jope-cli/src/dope_auditor.rs`, new `crates/jope-cli/tests/dope_auditor.rs`, minimal typed prepared-input bridge and a versioned process contract. CLI/JSON-schema registration belongs to integrator. DOPE's adapter lives in DOPE under its own owner; no cross-repository source dependency on JopeDime product crates. Runtime archives, study outcomes, weight files and sealed targets remain read-only.

## Implementation steps

1. Freeze a strict auditor request: protocol/roster ID, source/model/evaluator identity, task, public grouped feature schema, explicitly prepared input space, FIT/TUNE membership hashes, seed, search/fit/evaluation budget and timeout. Separate prediction input from outcome scoring; no API should provide real TEST labels to search or selection.
2. Implement a proposed `jope ... dope-auditor` handler (exact verb determined by existing CLI layout) consuming local admitted matrices/manifest only. Reuse current pipeline evaluator/search; no archive/bank/network lookup, implicit pretrained retrieval or automatic S3 access. Disable J02 retrieval by default for a clean named learner unless roster explicitly includes it with full cost/role binding.
3. Fit/search on real FIT for the real-trained reference and synthetic FIT for the synthetic-trained arm under the same declared budget/recipe. Learner-specific preprocessing must fit only that arm's training data; frozen NC1 coordinate map is shared. TUNE selection must follow DVS; common real TEST is assessment only.
4. Emit a bounded fitted-model/prediction receipt with program SHA, actual fits/evaluations/failures, input/view hashes, chosen method and resource time. Use structured failure status for timeout, unsupported task, invalid probabilities/nonfinite predictions and empty output. A crash cannot be counted as a zero-error learner or silently removed from denominator.
5. Agree a process/JSON exchange adapter with DOPE owner. DOPE remains independent of private Jope crates; source/version identity is explicit and invocation is optional. Prevent shell interpolation of data and bound input/output allocations. Preserve no-follow/symlink protections from existing paths.
6. Test synthetic-vs-real role separation using tiny analytic fixtures; freeze selected program before real TEST prediction/scoring. Grouped D permutes one-hot groups together; S_select uses synthetic validation selection and untouched real test regret. Do not conflate it with S_transfer ranking synthetic-trained and real-trained models on real test.

## Named success criteria and validation

- **J08-S1:** Tiny known-function fixtures complete real-trained and synthetic-trained auditor arms through the actual handler and DVS-compatible receipt.
- **J08-S2:** Poisoning real TEST targets changes scores only, never search, fit, chosen program or predictions; forbidden role/network opens are zero.
- **J08-S3:** Timeouts/failures/unsupported tasks remain explicit with complete denominators and consumed cost.
- **J08-S4:** Protocol pins, grouping and evaluator identities agree across process boundary; legacy Jope commands and DOPE metric readers remain intact.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-cli --test dope_auditor
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --lib
rtk git diff --check
```

First target is proposed; confirm feature selection after rebase. Both repos' adapter fixture tests and strict JSON-schema validation must run at their exact final heads. Expected artifacts: request/response contract, analytic prediction comparison, open-boundary audit, timeout/failure inventory and cost reconciliation. Passing fixtures qualifies integration only; native dataset/value campaigns need separate admission and preregistration.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store durable receipts at `/home/ubuntu/jope_plan/evidence/next-version/J08/<full-head>/`: `RECEIPT.json`, `commands.log`, and `artifacts.json` map each named success criterion to evidence; unknown outcomes/cost remain null.
