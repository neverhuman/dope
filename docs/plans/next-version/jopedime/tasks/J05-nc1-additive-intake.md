# J05 — Recover raw provenance and add frozen NC1 dataset views

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P1**. Dependencies: J00, D05. Cross-repository prerequisites: D05 qualified NC1 normalization contract/schema; D01 corrected evaluation when measuring retention.

## Objective and non-goals

Add a new normalization recipe from verified raw sources without rewriting legacy catalog data or changing active workers. Fit one private normalization contract per declared FIT scope; preserve physical-source, row, family, target and fold identities. Missing raw provenance blocks clean claims rather than inviting inverse reconstruction.

## Ordered reads and ownership

Read `docs/datasets.md`; `crates/jope-datasets/src/{ingest.rs,cache.rs,meta.rs,regression_release.rs,family_contracts.rs}`; `crates/jope-engine/src/prepare.rs` including `Preprocessing::fit`, `load_bound_view*` and admission checks; current `python/src/jope/intake/` owners; DOPE NC1 contract/tests; current catalog/raw-source manifests. #2071 splits several files: map actual APIs before editing.

Dataset worker owns new `crates/jope-datasets/src/nc1_view.rs`, focused cache/view adapters, new `tests/nc1_view.rs` and a small raw-source manifest under the existing governed catalog area. Integrator owns recipe/schema registration. Engine worker modifies only a new bound-view branch/loader. Current catalog pointers, S3 uploads, raw downloads, source datasets and historical metadata stay read-only.

## Implementation steps

1. Inventory recoverable raw sources and explicit target/kind/license/source hashes. Verify suspected train∪test scaling from provenance/reproducers; do not label every live catalog object leaked because ten fixtures or dated docs share a recipe. Missing raw source becomes `legacy_scaled`/unknown and is excluded from new confirmation.
2. Reuse DOPE's dependency-light NC1 crate/facade; no local normalizer implementation. Proposed `Nc1View::fit(FitView, DeclaredSchema)` returns a private ScaleCard reference and public structural schema plus typed matrix. Resolve target from declaration, then write target-last output; never infer source target from position/cardinality.
3. Fit on the exact declared scope. For CV, each cut's FIT rows determine extrema/vocabulary/imputation; validation/test features or targets cannot influence fit. Preserve ordered physical row IDs and family/group/fold bindings. Canonical physical-row reductions and dtype/tolerance rules come from NC1.
4. Store private contract in owner-private custody; real FIT/TUNE/TEST matrices remain private/custody data. Only structural schemas and separately admitted synthetic matrices may travel; normalization is not publication permission. Cache keys separately bind physical source, FIT membership, transform/recipe, schema, payload and implementation. Use atomic publication, safe bounded parsing and no-follow access consistent with existing cache contracts.
5. Add an opt-in recipe/view loader to `jope-engine::prepare::load_bound_view*`, converting f32 to f64 exactly where required. Preserve legacy loaders, evaluator identity, numeric preparation and old catalog bytes. Do not pass NC1 directly to released-v3 encoding unless a separate equality-qualified shim supports that input class.
6. Test unseen/missing categories, constant/all-missing numeric columns, overflow, >64 expanded columns/group mapping, duplicate IDs and target copies. Regression test tails remain unclipped. Unsupported legacy encoding refuses rather than truncates or relabels data compliant.

## Named success criteria and validation

- **J05-S1:** Every new view has verified raw/target/FIT/row/family/fold provenance; excluded or unknown coverage is reported explicitly.
- **J05-S2:** Mutating TUNE/TEST rows leaves fitted private-contract bytes unchanged; applying the same card is deterministic.
- **J05-S3:** Public-schema/export canary scans reveal zero private headers/vocabulary/extrema; errors expose safe codes/indices only.
- **J05-S4:** Legacy view/evaluator receipts remain unchanged; new recipe is opt-in and no live catalog pointer moves.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-datasets --test nc1_view
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-parallel --lib prepare::tests
rtk git diff --check
```

First target is proposed; verify actual engine package/test module after rebase and record collection. Expected artifacts: raw-source coverage report, exact FIT/cache identity manifest, leakage-poisoning and public-canary receipts, legacy identity comparison. Bulk reingest/publication requires separate curator admission after these gates; this task only makes its future inputs reviewable.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J05/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
