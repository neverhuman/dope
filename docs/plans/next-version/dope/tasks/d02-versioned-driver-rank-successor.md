# D02 — Add informative driver-rank semantics without rewriting history

Current status: **SOURCE IMPLEMENTED; VALIDATION IN PROGRESS.** Branch `fast/next-version-dope-20261103` in the same isolated checkout. See [live source/check ledger](../../STATUS.json) and external receipts `/home/ubuntu/jope_plan/evidence/next-version-implementation-20261103/D02/<full-head>/RECEIPT.json`. Source implementation does not establish every success criterion. Exact-head native/full CI is still pending; scientific efficacy/release/serving remains unqualified. One integrator owns repository writes; refresh the current owner claim before any further change.

Historical planning baseline (retained for audit; current assignment and lane above govern):

> Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.


Priority: **P0**. Prerequisites: D00.

## Objective and non-goals

Add explicitly versioned driver-rank semantics that do not award perfect agreement to uninformative all-tied vectors. Current Python and Rust reducers award one when signs match, including both-zero pair signs; no-pair input also yields one. Fix the prospective protocol and preserve historical v1 **and existing v2** receipts. Do not select a new formula by maximizing prior published scores, alter frozen contracts, or bundle a campaign.

## Exact read-first order

1. Repository `AGENTS.md`, current claims/open PRs, `docs/architecture.md`, `docs/testing.md`, `contracts/AGENTS.md`, owner/test maps and D00's metric decision.
2. `research/transfer/tdl/drivers.py::{rank_agreement,evaluate,_evaluate_v1}` and all current protocol dispatch paths.
3. `rust/metrics/drivers.rs::rank_agreement` and its uses; `rust/transfer_metrics.rs`; locate included modules with `rtk rg rank_agreement rust research/transfer`.
4. `research/transfer/tests/test_drivers.py`, especially `test_explicit_protocol_dispatch_preserves_every_v1_fixture`; `tests/golden/drivers/`, `tests/golden/drivers/v2/`, `research/transfer/tdl/golden.py`, `docs/metric.md` and governed schemas.

## Write boundaries and concurrency

One metric worker owns a new successor function/module in the existing metric areas, successor dispatch/tests/goldens and a new schema/protocol identity. Integrator owns top-level schema registration and maps. Do not modify existing golden expected values, immutable TDL contracts, historical receipt bytes, raw importer inputs, benchmark charts or D01 coordinate code. D06 consumes this implementation instead of adding another rank formula.


Python boundary integration: before adding proposed Python source/tests, read `PYTHON_EXCEPTIONS.md`, `docs/internal/PYTHON_EXCEPTIONS.md` and `agent/boundaries.toml`. Integrator updates the authoritative exception inventory and synchronized record as current rules require; change allowed roots only if genuinely necessary. These are shared, currently owner-sensitive paths: module workers submit proposed entries instead of racing edits. No Python enters the Rust product closure.

## Small implementation steps

1. Enumerate current callers and version defaults in both languages. Freeze fixture hashes before edits. Existing `rank_agreement` may remain unchanged; expose proposed `rank_agreement_informative` or successor reducer through an explicit new protocol rather than changing the default.
2. Encode D00's chosen behavior for zero informative pairs, constant real importances, zero synthetic mass, ties on one side, ties on both sides, one group, empty vectors and non-finite/negative entries. A recommended typed shape is `{status, reason, value, informative_pairs, total_pairs}`; `value=null` for ineligible evidence, never silently one or zero. Final schema follows the repository's existing state vocabulary.
3. Apply grouping before ranks: permute one-hot groups coherently; do not count each dummy as an independent driver. Reuse declared group mapping and positive-importance semantics rather than infer groups from correlated values.
4. Add tiny analytic fixtures: identical strict rank, reversed rank, one-sided ties, joint ties, partially informative ranks, zero vectors, one group and malformed lengths. New expected values must be hand-derivable and shared across Python/Rust.
5. Add version dispatch that fails closed for unsupported identity. An old receipt missing an explicit version must retain its documented old interpretation. Update new result schema without weakening historical readers.
6. Preserve existing v2 C02 simultaneous `full_family_max_t` inference, positive floors and aligned query/groups/perturbation/prepared-view hashes. The successor cannot replace these with naive ratio ceilings. Check callers never use missing/ineligible rank as perfect score, zero loss or dropped denominator; count failed/incomplete states explicitly.

## Success criteria

- **D02-S1:** All existing v1/v2 fixture bytes and expected outputs remain unchanged and replay successfully.
- **D02-S2:** New all-tied and no-informative-pair cases return the specified non-perfect/ineligible state in Python and Rust.
- **D02-S3:** Every analytic successor fixture agrees cross-language within declared arithmetic tolerance and records informative-pair counts.
- **D02-S4:** Duplicating one-hot members within the same declared group cannot inflate successor score; this does not assert invariance to arbitrary duplicate groups. Non-finite, wrong-length and unsupported-version input fails safely.
- **D02-S5:** Prospective result cannot be interpreted by an old reader as an old passing receipt; default legacy dispatch is preserved.

## Validation commands and evidence

Existing: `rtk proxy python3 -B -m unittest research.transfer.tests.test_drivers -v`; `rtk proxy just golden-parity`; `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked --lib metrics::golden`. Proposed new focused module: `rtk proxy python3 -B -m unittest research.transfer.tests.test_driver_rank_successor -v`; Rust `cargo test --locked --lib rank_successor` through `rtk proxy env ...`, only after adding named tests and verifying nonzero discovery. Final repository checks follow current maps, including `just check`/exact-head CI. Log old fixture inventory hashes, new protocol/schema hash, each analytic pair/result and executed counts. A zero-test filter or updated old expected values fails validation.

## Stop, rollback and handoff

Stop if existing v2 is assumed mutable, metric definition remains unresolved, group provenance is missing or old receipt replay changes. Rollback successor registration, not historical content. Handoff exact HEAD, caller inventory, formula/state definitions, schema version, all cross-runtime results, retained compatibility hashes and D06 consumption instructions. No scientific D80/utility claim is authorized by fixture correctness.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D02/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D02-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.

## Individual engineering verdicts

Verdicts reflect scoped engineering receipts; the exact stopped HEAD is recorded externally and in the live ledger. Native compilation PASS belongs to 84a55cf9; later source has native execution/full CI pending. `PASS` never implies efficacy or release.

| Criterion | Verdict | Evidence and limitation |
| :--- | :--- | :--- |
| D02-S1 | **NOT_RUN** | Existing Rust golden/legacy targets passed as scoped subtargets; strict complete129+34 discovery/count/error proof remains pending. Historical expected bytes unchanged. |
| D02-S2 | **PASS** | Accepted completed subtargets at tested synthetic merge07887a13b1f9bc60ac085bde85a502ca2924b599 (head ccd28bb0122425e625e36323091627267a46af39): root Rust rank_successor_shared_analytic_cases and Python shared/all-tied successor tests passed. Every seven-case expected object includes informative/total counts. Overall job41651340409/job125970478639 cancelled later; not whole-job CI PASS. D02 metric/dependency/fixture closure unchanged through a23 and4054; scoped evidence only. |
| D02-S3 | **PASS** | The same completed Rust and Python analytic subtargets compare all seven shared fixture objects at tolerance1e-12, including informative-pair counts; no analytic case is skipped. See completed-root-metrics-ccd.json and external source report hash d6f7508cd8e04c1f2bb0e306c98c45030561284d212cbeb1c0ad3bad84a60581. Overall CI cancelled; current full exact-head CI remains pending. |
| D02-S4 | **NOT_RUN** | Malformed-input refusals passed at ccd. New actual production grouped-permutation same-group member duplication and successor-rank equality fixture authored at4054; execution pending. |
| D02-S5 | **PASS** | test_legacy_joint_ties_and_v2_fixture_are_unchanged and tagged dispatch preserve historical defaults; new protocol ID remains explicit. Full native legacy replay pending separately. |
