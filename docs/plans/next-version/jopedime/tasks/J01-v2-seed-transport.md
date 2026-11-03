# J01 — Carry existing v2 seed provenance into real generation zero

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P0**. Dependencies: J00. Cross-repository prerequisites: none; retain existing DOPE engine/model pin.

## Objective and non-goals

Connect the existing `SeedSetV2`/`admit_v2` contracts to the production pipeline without archive access or retrieval selection. Preserve ancestry, requested/realized source mixture and effort through actual generation-zero consumption. Keep v1 input support and flag-off output bytes unchanged. Do not add DOPE inference, new model weights or policy proposals here.

## Ordered reads and ownership

Read `crates/jope-pipeline/src/{seeds.rs,run.rs,receipt.rs,search.rs,evidence.rs}`; current receipt schemas; `crates/jope-learning/tests/retrieval_exposure.rs`; `crates/jope-cli/src/main.rs::Command::Pipeline`, which delegates to `jope_pipeline::cli::dispatch`; `crates/jope-pipeline/src/cli.rs::{SearchArgs::options,run}`; generation-zero tests in `run.rs`. Resolve split modules introduced by #2071 first.

Pipeline worker owns focused seed/run/receipt/evidence changes and new `crates/jope-pipeline/tests/seed_v2_product.rs`. Integrator owns schema registration and CLI argument plumbing. Do not edit learning retrieval selection, context producers, accepted policy slots or historical receipt fixtures. `jope-learning` already depends on `jope-pipeline`; adding the reverse dependency is forbidden.

## Implementation steps

1. Trace `resolve_seeding`, which currently reads v1, and `SeedSetV2::check`, `admit_v2`, `AdmittedV2`, `SeedAncestryV2`, `SeedSourceV2`. Reuse those types; do not introduce a third seed interchange format.
2. Add `run_with_seed_provider` and a neutral `CutSeedProvider` callback returning `SeedSetV2`. Invoke it after existing `prepare::partition`, before output creation. Supply only that cut’s TRAIN features/targets, physical TRAIN IDs and cut identity; never holdout fields. Preserve `run()` without a provider. New retrieval requires `cuts=1`; refuse >1 before encoding. Do not duplicate partition logic in CLI. Define compatibility/mutual exclusion with legacy seeds/incumbent.
3. Retain existing canonicalization, bootstrap removal, deduplication, source-mixture counts, cap and complete ancestry. Preserve raw offered effort and post-admission realized counts separately. Preserve CPU `run::resolve_seeding` cap `search::ELITES` (currently4). Neural `neural_search::MAX_INCUMBENTS=16` is a different path, deferred. Do not raise either budget implicitly.
4. Carry v2 provenance into a versioned optional receipt extension and evidence verifier. Absence must serialize identically to the old path. Recompute canonical program hashes after canonicalization while retaining original immutable draw identity; never discard source provenance by mapping everything to a generic seeds-file label.
5. Feed that single cut’s admitted genomes into `SearchOptions.seeds`; never union per-cut seeds into the existing global vector; each seed gets ordinary fitting/remeasurement before ranking. Prove an admitted bank-style genome appears in generation zero after bootstrap and before later evolution.
6. Test malformed ancestry, corrupt program hashes, duplicated draw IDs, all-bootstrap input, cap overflow, partial realization and mixed legacy/v2 arguments. A refusal leaves no partial output. Define explicit result when admission yields zero usable seeds.

## Named success criteria and validation

For replay comparisons, inject a deterministic fixture clock or compare an explicitly versioned stable projection excluding only declared timestamps/durations. Seed/program/provenance/role/identity fields remain exact; do not drop a changed semantic field to obtain equality. Live wall-clock receipt bytes are not expected to match.

- **J01-S1:** A v2 fixture reaches actual generation zero, with the same canonical program SHA and complete ancestry visible in verified receipts.
- **J01-S2:** Offered, duplicate/bootstrap rejected, surplus and realized counts reconcile exactly; source budgets and ordinary reevaluation remain enforced.
- **J01-S3:** Existing v1 runs replay byte-identically with v2 absent. Refused inputs leave no pipeline output.
- **J01-S4:** Cargo graph has no pipeline→learning edge. Retrieval refuses >1 cut; held-row/target poisoning under frozen split/physical-row manifests leaves provider input and seeds unchanged; receipts bind cut and TRAIN IDs.

Existing checks plus proposed integration target:

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning --test retrieval_exposure
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --lib run::tests
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --test seed_v2_product
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo metadata --locked --format-version 1 --no-deps
rtk git diff --check
```

The new target must be created. Expected artifacts: flag-off receipt digest comparison, positive generation-zero trace, refusal/output-tree comparison, seed-count reconciliation and dependency receipt. A nonempty imported list alone does not satisfy search consumption. If current run/evidence schemas cannot preserve v1 bytes, split an explicitly versioned successor rather than silently altering historical semantics.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J01/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
