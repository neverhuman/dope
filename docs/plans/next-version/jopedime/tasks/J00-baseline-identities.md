# J00 — Lock baseline identities and establish the execution checkpoint

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P0**. Dependencies: D00. Cross-repository prerequisites: D00 shared lineage, naming and role decisions; retain current engine/model bytes.

## Objective and non-goals

Create one strict compatibility manifest for the existing released-v3 lane before any repin. Make source, standalone encoder, fused serving, preparation, and evaluator identities independently checkable. This is identity bookkeeping: do not upgrade engine or model, regenerate evaluator baselines, enable encoding, train, or promote serving state.

## Ordered reads and ownership

After universal reads: `crates/jope-encoders/Cargo.toml`; `crates/jope-encoders/src/dope_encoder/{checkpoint.rs,preparation.rs}`; `crates/jope-encoders/src/dope_encoder.rs`; `models/{PINNED,manifest.json}` and relevant bundles; `crates/feat-math/tests/fixtures/compat_freeze/transitions/`; `python/src/jope/north_star/dope_freeze_guard.py`; `python/src/jope/telemetry/dope_guard.py`; `crates/jope-telemetry/src/hooks/dope.rs`. Check existing pin-check tooling after rebase before adding a checker.

Integrator owns new `models/dope-compat.json`, new `crates/jope-encoders/src/dope_lock.rs`, module registration, new `tests/dope_compat.rs`, Cargo test registration and authoritative pin checker. Do not edit historical transition receipts or `models/PINNED` merely to align text.

## Implementation steps

1. Capture branch, remote-main SHA, dirty state, PR queue and active claims. Produce a machine-readable baseline receipt before edits; abort if another writer owns these paths.
2. Define one `deny_unknown_fields` manifest with engine Git revision, SDK contract or explicit legacy status, standalone checkpoint SHA, input/conditioning/landmark recipes, vector layout, evaluator identity and separate fused checkpoint/composite identity. Do not invent a digest when a producer did not record it: encode applicability/unknown explicitly.
3. Preserve current engine rev `15e12aa3dc33851655580f6aa7c9c66353143f23`, standalone SHA `d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`, and fused composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` as different objects. Verify applicability against exact existing manifests before accepting them.
4. Implement `DopeCompat::validate()` and a static checker resolving current source pins/manifests. Product startup validates already-local bytes; never download at runtime. Explicitly mark historical docs as historical instead of rewriting them.
5. Add negative fixtures for one-field drift, unknown schema/field, malformed digest, wrong vector width and conflated standalone/fused identity. Add an unchanged-baseline positive fixture.
6. Preserve NS3 strict serving/flywheel freeze: zero DOPE gradients/optimizer membership, eval mode recursively, normalizer/anchor immutability, and rejection of `JOPE_RESEARCH_DOPE_UNFREEZE != 0`. Manifest validation does not replace actual before/after snapshots.

## Named success criteria and validation

- **J00-S1:** Exactly one canonical compatibility file exists; each asserted identity has a source reference and applicability.
- **J00-S2:** Current released CPU vectors and source/conditioning digests are unchanged; every intentional drift fixture is refused.
- **J00-S3:** Strict freeze guards still reject trainability, optimizer, mode and buffer mutations; no serving/pin/evaluator change occurs.

Existing checks, then proposed new test:

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --test dope_encoding
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-telemetry --lib hooks::dope
rtk proxy python3 -m pytest -q python/tests/test_north_star_flywheel_guards.py python/tests/test_telemetry_smoke.py
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --test dope_compat
rtk git diff --check
```

Last test is proposed and must be created/registered. Expected artifacts: baseline identity inventory, strict manifest, drift-test receipt and old/new vector-bit comparison. Any changed vector/hash fails identity-preserving scope and requires a separately reviewed transition.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J00/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
