# J06 — Emit owner-side ScaleCard stages for raw-row prediction

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P1**. Dependencies: J05, D05. Cross-repository prerequisites: D05 frozen NC1 apply/inverse/group contracts.

## Objective and non-goals

Programs trained in NC1/synthetic unit space must predict owner raw rows through the exact frozen private transform. Add one composable emission stage and target inverse across supported languages. Never place private ScaleCard contents in public `.dope` packets or treat a byte-cap claim as privacy proof.

## Ordered reads and ownership

Read NC1 private/public schemas and reference apply/inverse; `crates/feat-math/src/{artifact_manifest.rs,emit_rs.rs,emit_py.rs,emit_cpp.rs,emit_js.rs}`; `advanced_v3/{input.rs,pipeline.rs,emit/}`; existing cross-runtime emission/compatibility tests; J05 view/row/group bindings. #2071 substantially splits emitters; resolve actual generators before claiming. Generated source/registries and frozen compat bodies must be changed only via their owning source generator.

One emitter integrator owns new `crates/feat-math/src/unit_stage.rs`, module/manifest registration and cross-language adapter signatures. Language workers may own disjoint emitter implementation files after interface freeze. They must not race on common schema/templates or historical golden fixtures. New `crates/feat-math/tests/unit_stage_emission.rs` owns focused parity fixtures.

## Implementation steps

1. Define `BoundUnitStage` with private card/schema/FIT recipe identity and an explicit target map; validate it before code generation. Bind expected raw input column order/kinds locally. Public metadata carries only necessary structural identifiers; owner-private code may contain the private transform only in an explicitly private export.
2. Implement reference `apply_features(raw_row)` and declared `inverse_target(unit_prediction)` by reusing NC1 semantics. Never refit at prediction, derive extrema from prediction batches, or transform regression test labels for clipping. Classification preserves explicit class/positive-label semantics.
3. Compose stage before the learned program and target inverse afterward. Avoid adding the map twice when a caller already supplies NC1; require explicit input-space tag. Refuse card/program/schema mismatch before writing artifacts. Keep existing raw/legacy emission byte identities when no stage is present.
4. Render deterministic Rust, Python, C++17/20 and JavaScript implementations for supported paths, using established numerical-policy helpers. Define language-specific tolerances grounded in f32 matrix representation; do not promise raw-f64 four-ULP inversion universally. Unsupported language/feature combinations are explicit exclusions, not emitted stubs labeled success.
5. Exercise numeric missing indicators, constant/all-missing columns, extreme/out-of-range values, missing/unknown categorical tokens and reserved-token collisions. One-hot groups and column ordering must match NC1, including >64 expanded columns. Unsupported/nonfinite input fails consistently without echoing private cells/headers in public errors.
6. Add public/private export separation tests and a canary card. Verify public program/artifact manifests exclude vocabularies/extrema/raw headers and private sidecar paths. Do not regenerate unrelated historical emitter fixtures to hide drift.

## Named success criteria and validation

- **J06-S1:** For each supported language, raw-row emitted predictions equal reference `inverse(program(apply(raw)))` within declared task/dtype tolerance.
- **J06-S2:** The same matrix, group, target and recipe identities are used at fit and prediction; double-application and mismatched cards refuse.
- **J06-S3:** No-stage legacy bytes remain exact; unsupported emissions are recorded as exclusions/failures, never success.
- **J06-S4:** Public canary scan exposes zero private map contents; private owner export is explicitly labeled and locally bound.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p feat-math --test unit_stage_emission
rtk git diff --check
```

The new target must compile/run emitted programs using existing toolchain-admission helpers; retain actual per-language compiler/interpreter commands, versions, exit status and prediction diffs. Missing compiler coverage is NOT_RUN and blocks claiming that language. Expected artifacts: private binding fixture, public canary receipt, cross-language prediction table, emitted source hashes and legacy-byte comparison. Run owning existing emitter/compatibility suites and exact-head required CI once after the final change.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J06/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
