# J07 — Adopt the qualified SDK facade without changing frozen semantics

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P1**. Dependencies: J00, D03. Cross-repository prerequisites: D03 immutable facade-v0 surface with source/parity qualification.

## Objective and non-goals

Move the existing starforge gateway from deep `dope-core` imports to qualified `dope-sdk`, using the one compatibility manifest. Preserve released-v3 vectors/preparation, evaluator identity and NS3 freeze semantics. Produce a conditional retirement assessment for duplicate service code; do not delete fused-ranker behavior merely to reach a line-count target.

## Ordered reads and ownership

Read D03 facade API/parity/release receipts; `crates/jope-encoders/{Cargo.toml,src/dope_encoder.rs,src/dope_encoder/}` and direct imports; J00 manifest/checker; `crates/jope-service/src/ranker/{landmarks.rs,cuda/forward.rs}`; service bundle/projection/normalizer contracts; `docs/qualification/dope-resident512-adoption.md`; freeze guard files listed in J00; existing evaluator transition machinery.

Gateway worker owns starforge import/type adapters and focused tests. Integrator alone owns Cargo/lockfile/pin-check/manifests/module registrations. Service graph, normalizers, frozen r882 checkpoint, generated projections and historical transition fixtures remain read-only. PR #2071 restructures ranker paths; caller map must use the settled source tree.

## Implementation steps

1. Freeze old and new complete engine/SDK/model/preparation/landmark/vector/evaluator identities. D03 compilation alone is insufficient: require an immutable usable facade revision with source/parity evidence and no unreviewed behavioral delta from the old gateway contract.
2. Replace starforge deep calls with narrow SDK exports for preparation, landmark bundles/sessions and resident512. Preserve actual CPU pool, cancellation, source/conditioning digests, validation, device/mutation independence and unsupported-CUDA refusal. Do not mix model registration or landmark-policy changes into this migration.
3. Update `models/dope-compat.json` and Cargo source revision in one integrator-owned transaction. Preserve standalone checkpoint SHA separately from fused params/buffers/normalizer composite. Validate no runtime download or local unpinned path dependency. Only the gateway directly depends on SDK; transitive dope-core in Cargo.lock is expected.
4. Run old/new fixture oracles on identical declared input classes; compare f32 bits, recipes, digests, failures and optional/native session behavior. If a deliberate evaluator identity transition is unavoidable, stop identity-preserving scope and create a separate reviewed transition packet. Never paste new expected hashes into existing goldens.
5. Retain Python `lock_dope_weights_and_eval_mode`, `snapshot_dope_weights`, `assert_optimizer_excludes_dope`, `reject_research_dope_unfreeze` and Rust `DopeGuard::enforce`. Parameters, normalizer buffers and anchor remain frozen; serving/flywheel rejects research-unfreeze environment.
6. Create `docs/coordination/dope-duplicate-retirement-readiness.md`: enumerate remaining generic landmark/router callers, receipt dependencies and exact recipe semantics. Existing heuristic `fit_transform_and_extract_bundle_high` and fused `FusedForward::stages` are not guaranteed equivalent to native DOPE. Mark retirement HOLD until replacement parity/new-lineage training/qualification and owner handoff exist. A GO requires a separate narrow deletion task before editing service code.

## Named success criteria and validation

- **J07-S1:** Gateway imports only the qualified SDK surface; manifest and resolved Cargo revision agree and historical docs remain labeled historical.
- **J07-S2:** Released vectors/digests/failures and required session contracts match old gateway; frozen guard tests still reject every protected mutation.
- **J07-S3:** No service-ranker semantics, model weights, accepted serving state or historical evaluator fixture changes.
- **J07-S4:** Every duplicate caller has an evidence-backed GO/HOLD retirement disposition; no LOC-driven deletion is bundled.

```sh
rtk rg -n 'dope_core::|dope-core|dope-sdk' crates Cargo.toml Cargo.lock
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --test dope_encoding
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --test dope_compat
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-telemetry --lib hooks::dope
rtk proxy python3 -m pytest -q python/tests/test_north_star_flywheel_guards.py python/tests/test_telemetry_smoke.py
rtk git diff --check
```

Search results are reviewed, not blindly required empty: transitive lock entries/historical docs are legitimate. Expected artifacts: exact source/revision inventory, full gateway parity receipt, guard coverage and caller/retirement map. Pinning a different source commit does not justify silently changing tensor identities.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store durable receipts at `/home/ubuntu/jope_plan/evidence/next-version/J07/<full-head>/`: `RECEIPT.json`, `commands.log`, and `artifacts.json` map each named success criterion to evidence; unknown outcomes/cost remain null.
