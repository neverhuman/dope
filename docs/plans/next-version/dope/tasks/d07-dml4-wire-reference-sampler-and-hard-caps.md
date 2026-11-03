# D07 — Freeze DML4 wire, basis and reference sampler before training

Current status: **SOURCE IMPLEMENTED; VALIDATION IN PROGRESS.** Branch `fast/next-version-dope-20261103` in the same isolated checkout. See [live source/check ledger](../../STATUS.json) and external receipts `/home/ubuntu/jope_plan/evidence/next-version-implementation-20261103/D07/<full-head>/RECEIPT.json`. Source implementation does not establish every success criterion. Exact-head native/full CI is still pending; scientific efficacy/release/serving remains unqualified. One integrator owns repository writes; refresh the current owner claim before any further change.

Historical planning baseline (retained for audit; current assignment and lane above govern):

> Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.


Priority: **P1**. Prerequisites: D00, D03, D05, D10.

## Objective and non-goals

Build the additive v4 packet contract, deterministic reference sampler and hard-cap packer **before D08/D09 training implementation**. Every active training/encoding gate must have defined wire and sampling semantics. This is not a model-quality claim. Do not mutate DML3/CSC1/DPK readers, require prefix nesting, remove 4k compatibility, infer privacy from bytes or publish a new release.

## Ordered read-first files

1. Current `AGENTS.md`, claims, `docs/{architecture,testing}.md`, `contracts/AGENTS.md`, maps and D00/D05 decisions.
2. `research/levels/{strata_audit,budget_planner,multilevel_model,multiclass_head}.py`: `_WIRE_GATE_KEYS`, basis construction/fitting and decoded sampling.
3. `engine/dope-core/src/mlc_v3/{wire,sample,scaler,mod}.rs`, existing `mlc_v3_safetensors` and sampler parity tests. Current four-gate wire and default asinh/row/periodic gates demonstrate why cross-language parity alone is insufficient.
4. D03 SDK types, D05 group/schema contract and D06 receipt API draft; resolve hard-cap names and required metadata overhead first.

## Files, ownership and protected zones

Create `engine/dope-core/src/mlc_v4/{mod.rs,wire.rs,basis.rs,sample.rs,rd_pack.rs}`, `engine/dope-core/tests/mlc_v4_wire.rs`, `engine/dope-core/tests/mlc_v4_sampling.rs` and public synthetic fixtures in a **new** versioned fixture directory. Create `research/levels/v4_wire_reference.py` and its bounded test module. Integrator alone registers `lib.rs`/SDK/maps. Old fixtures/readers, released assets, training data, manifests owned by D03/D05 and generated scientific outputs stay read-only. D08/D09 consume this API, not competing basis code.


Python boundary integration: before adding proposed Python source/tests, read `PYTHON_EXCEPTIONS.md`, `docs/internal/PYTHON_EXCEPTIONS.md` and `agent/boundaries.toml`. Integrator updates the authoritative exception inventory and synchronized record as current rules require; change allowed roots only if genuinely necessary. These are shared, currently owner-sensitive paths: module workers submit proposed entries instead of racing edits. No Python enters the Rust product closure.

## Ordered implementation and edge cases

1. Freeze a machine-readable basis/gate registry: each gate's ID, order, type/range, supported task, applicability, encoding and decoded use. Include version/schema/group/layout/weight identities. Unknown required gates and unsupported versions refuse; absent optional behavior is explicit, never hidden numeric defaults copied from v3.
2. Define framing and complete cap accounting for independent 2k/16k/64k packets. Retain explicit legacy 4k reader semantics. Metadata, checksum, group schema, required per-table decoder state and sidecars count. Impossible mandatory structure returns typed budget refusal before expensive work.
3. Implement deterministic packer with stable candidate/tie order and bounded arithmetic. It either returns `encoded.len() <= cap` or safe refusal. No three-pass best-effort oversized success. Rate-distortion selection uses development/FIT/TUNE evidence only, not real TEST; static synthetic costs suffice for engineering fixtures.
4. Reference sampler consumes decoded packet only plus declared shared weight closure. Categorical group sampling is mutually exclusive with unknown/missing handling; numeric missingness and regression/classification target semantics follow D05. Validate dimensions and allocation limits before multiplication/allocation; reject malformed counts, NaN, truncation, corrupt checksums and inconsistent groups.
5. Add independent **fitted-basis versus decoded-basis** checks using deliberately nondefault gate values. Python/Rust matching the same omitted gate is a failure, not parity success. Use coupled fixed draws to compare decoded outputs and distinguish arithmetic tolerance from semantic mismatch.
6. Test source/encoder/private card physically unavailable for recipient sampling. Keep literal prefix nesting deferred: later framing/rate-distortion evidence must prove independently valid prefixes and equal-total-byte quality, not merely array truncation.
7. Freeze API/fixture hashes and hand them to trainer/objective owners. Any later graph/gate change increments contract identity and invalidates dependent parity/training evidence deliberately.

## Success criteria

- **D07-S1:** Every active gate has one authoritative fitted/encoded/decoded definition, tested with nondefault values.
- **D07-S2:** All admitted packets meet complete hard caps exactly; impossible caps return typed refusal and no oversize output.
- **D07-S3:** Cross-language fixed-draw and independent basis equality pass within frozen tolerance on all task/group fixture classes.
- **D07-S4:** Malformed/framing/allocation attacks refuse before unbounded allocation or private-value output.
- **D07-S5:** Sampling works without source/encoder/private card and preserves one-hot exclusivity/missing semantics.
- **D07-S6:** DML3/CSC1/DPK compatibility bytes/readers are unchanged; contract hash is frozen before D08/D09 begin.

## Validation and evidence

Proposed new Rust commands: `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test mlc_v4_wire --test mlc_v4_sampling`; Python `rtk proxy python3 -B -m unittest research.levels.tests.test_v4_wire -v`. Existing v3 safe-tensor/preparation suites, `just engine`, required final checks/review remain. Receipt records registry/schema hash, cap matrix, malformed refusal classes, legacy fixture hashes and coupled draws/results. No quality, privacy or actual speed claim follows.

## Stop, rollback and handoff

Stop on unresolved required bytes, silent gate omission, semantic tolerance relaxation, impossible width or changed legacy output. Rollback new opt-in exports only. Handoff stopped HEAD, frozen API/registry/fixtures, cap refusal behavior and mandatory downstream invalidation rules. D08/D09 cannot train against a draft sampler.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D07/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D07-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.

## Individual engineering verdicts

Verdicts reflect scoped engineering receipts; the exact stopped HEAD is recorded externally and in the live ledger. Native compilation PASS belongs to 84a55cf9; later source has native execution/full CI pending. `PASS` never implies efficacy or release.

| Criterion | Verdict | Evidence and limitation |
| :--- | :--- | :--- |
| D07-S1 | **NOT_RUN** | All six gates serialized in authoritative basis registry; nondefault native basis fixture compiled; execution pending. |
| D07-S2 | **NOT_RUN** | Typed cap/framing validator compiled; actual boundary and allocation refusal execution pending. |
| D07-S3 | **PASS** | Default source-bound actual Rust regression and grouped-classification packets/fixed draws replay in Python test_v4_wire with 0 SKIP; engine native execution pending separately. |
| D07-S4 | **NOT_RUN** | Malformed wire/native bounds fixtures compiled; actual refusal execution pending. |
| D07-S5 | **NOT_RUN** | Reference source-free sampling passes; actual native and CLI absent-source/card/encoder journeys pending. |
| D07-S6 | **NOT_RUN** | D07 graph contract frozen before producer/trainer. Complete DML3/CSC1/DPK compatibility replay pending. |
