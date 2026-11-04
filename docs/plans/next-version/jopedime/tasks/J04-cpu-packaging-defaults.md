# J04 — Qualify CPU packaging and explicit landmark/device semantics

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P1**. Dependencies: J02, J03. Cross-repository prerequisites: no new DOPE source prerequisite; actual separately admitted positive study and measured packaging/latency gates remain external.

## Objective and non-goals

Make CPU DOPE support usable without CUDA while preserving explicit execution choices. Default feature inclusion or default runtime retrieval is conditional on actual packaging, latency and downstream-value gates. Do not assume enabling a Cargo feature makes appliance search consume DOPE; do not make ordinary fits pay unused encoding cost.

## Ordered reads and ownership

Read `crates/{jope-cli,jope-appliance,jope-pipeline,jope-encoders}/Cargo.toml`; `crates/jope-runtime/src/landmark_args.rs`; `crates/jope-pipeline/src/{run.rs,landmark_encoding.rs}` and CLI device parsing; appliance dependency/build/profile manifests; `docs/qualification/dope-resident{512-adoption,-session-bound}.md`; J02/J03 receipts. Resolve current appliance service route after #2071.

Integrator owns Cargo/default features, packaging manifests and command registration. Runtime worker owns narrow argument/device semantics and tests; stage sequentially against one frozen diff. Released signed packages, weights, native timing receipts and CUDA qualification locks remain outside write scope.

## Implementation steps

1. Write a behavior table separating build support, per-run encoding activation, device choice, landmark mode, consumer and weight availability. Existing `LandmarkArgs::enabled()` is optional; weights alone do not activate work. `mode()` currently defaults Full when activated; `RunOptions::new` uses cuda:0. Preserve explicit full/progressive/sample choices.
2. For the opt-in retrieval path, default requested device to CPU and mode to none. Unsupported explicit CUDA fails; no silent CPU fallback. Consumer-only/subsample-only combinations receive documented deterministic semantics and tests; do not globally rewrite Full semantics accidentally.
3. Prove normal CPU build graphs do not select CUDA/LibTorch or runtime-download weights. Verify local authenticated assets, startup refusal when missing/corrupt and actual appliance/CLI routing. Keep encoding feature optional until measured binary/dependency/packaging costs are accepted.
4. Add proposed `tests/dope_cpu_defaults.rs` in both CLI/appliance owning suites. Exercise no encoding, explicit none/full/progressive, explicit CPU/CUDA, unavailable assets, help JSON and feature-unavailable refusal. Verify functional use, not only cfg! values.
5. Prepare a separately admitted timing protocol: cold process/model load, resident encode, preparation and retrieval spans distinct; fixed actual thread/CPU admission and table shapes. The existing session-bound test's <=30ms median and <=5ms maximum post-preparation gate applies only to its supplied numeric FIT fixture/device/build, not arbitrary data or cold start. Reuse the existing harness rather than inventing forward-only proof.
6. Enable CPU build support by default only after owner-accepted package/dependency cost and asset gates. Enable retrieval runtime default only after complete ADVANCE from a separately admitted study registered under J03 plus cost/harm gates. Otherwise ship explicit CPU opt-in with unchanged ordinary behavior and an honest hold receipt.

## Named success criteria and validation

- **J04-S1:** Explicit CPU path works without accelerator libraries; explicit unavailable CUDA refuses.
- **J04-S2:** Argument behavior table matches implementation, including consumer/subsample-only and weight-only cases.
- **J04-S3:** Unrequested fits perform no unused encoding; defaults change only with linked measured value/latency/packaging evidence.
- **J04-S4:** No undocumented fallback, runtime network download, model repin or serving change.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-runtime --lib landmark_args
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --lib dope_encoder::workflow::resident_release_tests -- --skip real_s3_fit_none_session_release_bound
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-cli --features dope-encoding --test dope_cpu_defaults
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-appliance --features dope-encoding --test dope_cpu_defaults
rtk git diff --check
```

Last two targets are proposed. Expected artifacts: build-feature graph, package-size/dependency comparison, behavior matrix and admitted latency receipt or explicit NOT_RUN/HOLD. Missing native evidence blocks default claims; unit tests cannot substitute.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J04/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
