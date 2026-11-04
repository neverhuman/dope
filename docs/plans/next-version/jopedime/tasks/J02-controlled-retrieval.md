# J02 — Add identity-bound DOPE bank retrieval and decisive controls

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P0**. Dependencies: J01. Cross-repository prerequisites: none; released-v3 lane only.

## Objective and non-goals

Make released `Dope512` affect search through an opt-in, provenance-bound retrieval path using the unchanged released-v3 preparation/checkpoint. Include truthful dataset-shuffle and random-eligible controls. Do not retrain, repin, enable runtime encoding globally, or claim lift from a functional fixture.

## Ordered reads and ownership

Read `crates/jope-learning/src/retrieval.rs` and `tests/retrieval_exposure.rs`; `crates/jope-encoders/src/dope_encoder/{preparation.rs,checkpoint.rs,session.rs}`; `dope_encoder.rs`; J01's v2 seam and verifier; `crates/jope-cli/src/main.rs::Command::Pipeline` dispatch, `crates/jope-pipeline/src/cli.rs::{SearchArgs::options,run}`, and bank reader. There is no `jope-cli/src/run.rs`.

Learning worker owns retrieval identity/control logic and tests. CLI worker owns new `crates/jope-cli/src/{dope_precompute.rs,dope_retrieval.rs}` after confirming no existing equivalent; integrator alone owns registration/Cargo/schema edits. These scopes are sequential where they share API changes. Pipeline receives data only; never add pipeline→learning dependency. Bank snapshots and vectors are generated outside Git by their owning admitted producer.

## Implementation steps

1. Define a strict context identity binding checkpoint, engine/preparation/conditioning recipe, landmark mode `none`,512-coordinate layout, distance/normalization rule, FIT source/scope, role manifest and snapshot. Reject mismatched recipe or width before distance computation. Do not confuse vector512 with full629 or fused policy2048.
2. Add an offline bank precompute tool that reads only already-admitted TRAIN sources with complete flattened ancestry. Reuse `prepare_product_training_input` and resident encoder; preserve legacy input semantics. Publish an immutable local manifest keyed by source, effective conditioning and checkpoint digests. No discovery/download, S3 publication or bulk run belongs to this implementation task.
3. Keep `materialize` eligibility filtering shared across all controls: frozen support/role manifest, own physical-source/family/parent aliases, own program and missing ancestry checks occur before selection. Record eligible support digest once.
4. Preserve historical `ShuffledContext` as its existing one-coordinate rotation. Introduce separately versioned `DatasetShuffled` that permutes query-to-dataset context assignments under a frozen bijection; it must not cross role/eligibility boundaries. Add deterministic `RandomEligible`, uniform without replacement from the same eligible support and cap. Freeze seeds/permutation identities and tie behavior; no outcome-based reassignment.
5. Implement J01’s post-partition provider in new upper-layer `jope-cli/src/dope_retrieval.rs`; wrap main dispatch without moving archive reads into pipeline. Require cuts=1, encode only supplied cut-TRAIN data, and bind cut/physical TRAIN IDs. Keep `--seed-retrieval dope512` off by default. Handle zero eligible support before `materialize`: it calls `seeds.check()` and rejects empty draws. Use explicit fallback/refusal receipt, never empty success. Missing weights/unsupported data follow the same declared contract.
6. Record cold bank construction separately, query encode/retrieval latency inside run cost, requested/eligible/retrieved/admitted/realized counts, vector recipe and control identity. Preserve CPU `search::ELITES=4`; neural16 is separate/deferred. Retain ordinary seed remeasurement.

## Named success criteria and validation

For replay comparisons, inject a deterministic fixture clock or compare an explicitly versioned stable projection excluding only declared timestamps/durations. Seed/program/provenance/role/identity fields remain exact; do not drop a changed semantic field to obtain equality. Live wall-clock receipt bytes are not expected to match.

- **J02-S1:** A positive CPU fixture realizes a retrieved genome in generation zero with exact bank/query identity and ancestry.
- **J02-S2:** True, dataset-shuffled and random controls share eligibility/cap/effort constraints; fixed seeds replay exactly. Historical coordinate rotation remains distinctly named.
- **J02-S3:** Family aliases, role mismatch, mixed modes/checkpoints, stale digest, nonfinite/zero-distance edge cases and empty support cannot silently yield transfer claims.
- **J02-S4:** Disabled path is byte-identical. Heldout feature/label poisoning under fixed split/row manifests leaves query vectors and seeds unchanged; >1 cut refuses before encode. No model, serving or runtime default changes.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning --test retrieval_exposure
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --test dope_encoding
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-cli --features dope-encoding --test dope_retrieval_product
rtk git diff --check
```

Last target is proposed. Expected receipts include control support equality, deterministic permutation/random selection, feature-off identity and actual search consumption. Fake vectors test wiring only; label them fixture-proven, not native quality or performance evidence.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J02/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
