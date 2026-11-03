# D03 — Expose a minimal identity-preserving DOPE SDK

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P0**. Prerequisites: D00, D10.

## Objective and non-goals

Create the narrow Rust surface JopeDime currently needs, preserving numerical preparation, landmarks and encoder outputs. Replace deep consumer imports later through one gateway. This task does not repin JopeDime, redesign caches/sessions, add NC1, publish a tag, change weights or collapse workspaces. SDK compilation is not consumer parity or release qualification.

## Read-first sequence

1. Current `AGENTS.md`, claims/worktrees/open queue, `docs/{architecture,testing,release}.md`, owner/test maps and D00 compatibility decisions.
2. `engine/Cargo.toml`, `engine/Cargo.lock`, `engine/dope-core/src/lib.rs`, `engine/dope-core/src/prepare/mod.rs` exports.
3. `engine/dope-core/src/mlc_v3/{mod,resident,nets}.rs`: `MlcV3Model`, `MlcV3RoutedOutput`, weight inventory/loading, existing feature-forward methods. Read actual signatures before wrapping.
4. `engine/dope-core/src/landmark/{mod,bundle,progressive}.rs`, `engine/dope-core/tests/{prepare_parity_a,prepare_parity_b,mlc_v3_safetensors,landmark_bundle}.rs`; JopeDime `crates/jope-encoders/src/dope_encoder.rs` and direct `dope_core` imports read-only.

## Exact edits and owner boundaries

Create `engine/dope-sdk/Cargo.toml`, `engine/dope-sdk/src/lib.rs`, `engine/dope-sdk/tests/surface.rs`. Integrator alone edits `engine/Cargo.toml`, `engine/Cargo.lock`, owner/test maps and SDK guide. Current canonical dirty files include both workspace files, `mlc_v3/{mod,nets}.rs` and safetensor tests; leave them untouched. The isolated branch still needs stopped-head reconciliation before those edits integrate. No worker may race D05's manifest changes or D07's `lib.rs` registration.

## Ordered implementation

1. Inventory exact consumer methods/types. Start from current `prepare_table_cells*`, `build_conditioning*`, `LandmarkBundle`, `ProgressiveLandmarks`, `MlcV3Model` loading/encoding operations. Expose only needed opaque/wrapper types; do not clone their implementation.
2. Define a small stable `Engine`/encoder contract with separate model/preparation/landmark/vector-layout identity metadata. Preserve domain errors and redact source-bearing detail at the public boundary. Avoid leaking Candle internals if current consumer can use typed arrays and model handles.
3. Preserve engine minimum Rust **1.88**; the SDK public surface cannot require JopeDime's newer toolchain. CPU-first default, optional existing device support only if already needed. SDK must not depend on JopeDime or research Python. Do not imply `no-default-features` means the engine has a nonexistent feature arrangement; inspect actual manifest.
4. Route SDK calls directly into existing implementation. Keep operation order, f32/f64 conversions, masks, source-row scope and recipe identifiers unchanged. No hidden default landmark upgrade or new normalization.
5. Add synthetic/golden surface tests that call direct core and SDK with identical inputs and compare full outputs/digests, not only vector length. Include none/observed/full modes where fixtures support them, three task types, malformed identity, missing weights and unsupported shape.
6. Produce an immutable dependency revision candidate and consumer migration note. Do not publish release/tag or alter JopeDime lockfiles here. JopeDime gateway task adopts it only after its own exact parity/freeze tests.

Suggested API shape: `Engine::from_verified_weights(bytes, WeightIdentity) -> Result<Engine>` and `encode(&PreparedInput, &LandmarkInput) -> EncodingWithIdentity`. Final signatures must minimize conversion around actual existing APIs; this is guidance, not permission to invent parallel model types.

## Success criteria

- **D03-S1:** SDK builds/tests on CPU with no Python or JopeDime dependency.
- **D03-S2:** Direct-core and SDK vectors, conditioning digests and identity fields agree exactly wherever current core promises exactness.
- **D03-S3:** Invalid weights/recipe/schema inputs refuse with stable safe errors and no private canaries.
- **D03-S4:** Existing core/preparation/landmark public behavior and legacy decoders remain unchanged.
- **D03-S5:** Consumer migration instructions list supported methods and required parity checks; no serving/pin transition is implied.

## Validation and receipts

Proposed: `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --manifest-path engine/Cargo.toml --locked -p dope-sdk --test surface`. Existing: same environment with `cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test prepare_parity_a --test prepare_parity_b --test landmark_bundle --test mlc_v3_safetensors`; `rtk proxy just engine`; `rtk git diff --check`. Run nonempty fixture cases and final repository exact-head checks after capacity admission. Retain manifest dependency graph, core/SDK output hashes, tested task/mode matrix and source head. Unavailable released checkpoint is an explicit limitation, never substitute synthetic parity for released-weight qualification.

## Stop, rollback and handoff

Stop on changed output, unintended preparation/recipe transition, cyclical dependency or overlapping owner. Rollback new crate/registration only; preserve core and canonical edits. Handoff full HEAD, public surface inventory, parity receipts, unsupported cases and exact consumer adoption prerequisite. Coordinate one manifest integrator before D05 proceeds.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D03/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D03-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
