# D01 — Repair prospective evaluation coordinate identity

Current status: **SOURCE IMPLEMENTED; VALIDATION IN PROGRESS.** Branch `fast/next-version-dope-20261103` in the same isolated checkout. See [live source/check ledger](../../STATUS.json) and external receipts `/home/ubuntu/jope_plan/evidence/next-version-implementation-20261103/D01/<full-head>/RECEIPT.json`. Source implementation does not establish every success criterion. Exact-head native/full CI is still pending; scientific efficacy/release/serving remains unqualified. One integrator owns repository writes; refresh the current owner claim before any further change.

Historical planning baseline (retained for audit; current assignment and lane above govern):

> Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.


Priority: **P0**. Prerequisites: D00.

## Objective and non-goals

Ensure real-trained and synthetic-trained auditors are scored on identical real-test coordinates derived from declared real FIT data. Current `evaluate_dataset_across_tiers` fits `rec_csc1_fit` on synthetic data, then applies it to real test features; the fp64 diagnostic repeats this mismatch. Correct future evaluation under a new identity. Do not regenerate old headline results, change model weights, eliminate all learner-local scalers or claim engine regression targets are presently clipped.

## Ordered read-first files

1. Current assignment, repository `AGENTS.md`, current claims, `docs/{architecture,testing,data-contract}.md`, owner/test maps and D00 decision hash.
2. `research/levels/v3_canonical_runner.py::_prepare_dataset_split` and `evaluate_dataset_across_tiers`, particularly current lines 75–77 and 95–119.
3. `research/levels/scaling.py::FitOnlyTableTransform`, `research/levels/v3_canonical_roster_eval.py`, and `research/levels/tests/test_dope_mlc_v3.py`.
4. `bench/real-world/auditors.py::{_features,predict}` and `research/transfer/adapters/auditors/conventional.py::ConventionalAdapter`; `engine/dope-core/src/{encode,schema}.rs` and `eval/mod.rs` for semantic comparison only.

## Scope, owners and protected paths

Evaluation worker owns the canonical runner, **new** `research/levels/transfer_evaluation.py`, **new** `research/levels/tests/test_transfer_evaluation.py` and narrow tests/docs. Integrator handles any owner/test-map entry. Do not concurrently edit `scaling.py` with a legacy/parity owner. Frozen ledgers, existing protocol hashes, benchmark render outputs, released weights, DML3 decoder, source datasets and learner recipe identities remain untouched. Conventional training-only StandardScaler use is valid within a declared learner recipe; changing it requires a separate successor recipe.


Python boundary integration: before adding proposed Python source/tests, read `PYTHON_EXCEPTIONS.md`, `docs/internal/PYTHON_EXCEPTIONS.md` and `agent/boundaries.toml`. Integrator updates the authoritative exception inventory and synchronized record as current rules require; change allowed roots only if genuinely necessary. These are shared, currently owner-sensitive paths: module workers submit proposed entries instead of racing edits. No Python enters the Rust product closure.

## Ordered implementation and code shape

1. Capture a failing synthetic fixture: FIT x around 0–10, synthetic x around 100–200, real TEST x beyond FIT support. Record fitted-map hash and real-test view hash for both arms; demonstrate the old mismatched maps without running any real benchmark.
2. Introduce identity-tagged views: `PreparedEvaluationView {space_id, fit_scope, feature_groups, x, y}`. New `evaluate_prepared_transfer(real_fit, synth_fit, real_test, protocol)` validates common schema/preparation/test identity and never calls `fit()` on synthetic or TEST rows.
3. Fit the common receiver transform once from real FIT. Apply it to reconstructed raw synthetic rows and raw real TEST rows. If synthetic rows already occupy that prepared coordinate space, bypass reapplication explicitly. Reject unknown/double-scaled spaces rather than infer from ranges.
4. Preserve a clearly named historical replay route. Corrected outputs receive new preprocessing/protocol identifiers and include raw/scaled arm labels; old ledgers are not relabeled. Target inversion is separate from feature preparation; approximate CSC1 inversion must be reported, not assumed exact.
5. Add poison tests: change TEST extrema, categories and targets and assert fitted state unchanged; change synthetic distribution and assert real-test view unchanged. Test classification class order, missing/unseen categories, width mismatch, empty/non-finite input and malformed target roles.
6. Keep per-learner transformations when fitted only on that learner's training rows and bound into its recipe. Do not conflate comparing two learner pipelines with refitting the common evaluation view.

## Individually verifiable success criteria

- **D01-S1:** Deterministic fixture reproduces old arm coordinate mismatch and passes under the corrected route.
- **D01-S2:** Every scored arm shares the same real-test feature/target digest and real-FIT common-map digest.
- **D01-S3:** Synthetic/TEST mutations cannot alter common fitted state; double-transform and mismatched-space input refuse safely.
- **D01-S4:** Legacy outputs/readers remain replayable under their actual identity; corrected outputs cannot masquerade as old protocol.
- **D01-S5:** A fixture demonstrates unclipped held-out regression target scoring; no unproven current-engine bug is asserted.

## Validation and expected evidence

Proposed focused check: `rtk proxy python3 -B -m unittest research.levels.tests.test_transfer_evaluation -v`. Existing checks: `rtk proxy python3 -B -m unittest research.levels.tests.test_dope_mlc_v3 -v`; `rtk proxy python3 -B -m unittest research.transfer.tests.test_target_last research.transfer.tests.test_pilot_worker_inputs -v`; `rtk git diff --check`. Run required final `rtk proxy just check` only after admitted capacity; engine edits also need `rtk proxy just engine`. Record actual tests executed, failures, protocol IDs, before/after fixture hashes and common-map/test digests under one reused lane `target/` receipt. No new retained-science result follows from these tests.

## Stop, rollback and final handoff

Stop on missing coordinate provenance, unsupported inverse, changed upstream APIs or overlapping owner. Keep legacy route available; rollback only new opt-in route and own edits. Do not increase tolerances or fit from TEST to force parity. Handoff exact stopped HEAD, reproducible failure/fix commands, new protocol ID, unchanged historical identities, remaining scientific replay needs and next D06 dependency.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D01/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D01-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.

## Individual engineering verdicts

Verdicts reflect scoped engineering receipts; the exact stopped HEAD is recorded externally and in the live ledger. Native compilation PASS belongs to 84a55cf9; later source has native execution/full CI pending. `PASS` never implies efficacy or release.

| Criterion | Verdict | Evidence and limitation |
| :--- | :--- | :--- |
| D01-S1 | **PASS** | test_real_fit_map_is_shared_and_synthetic_refit_reproduces_bug demonstrates old mismatch and corrected common map. |
| D01-S2 | **PASS** | Same test and canonical receiver bridge assert shared real-test arrays, common FIT identity and protocol-tagged prepared views. |
| D01-S3 | **PASS** | test_test_and_synthetic_poisoning_never_changes_fitted_metadata plus double-transform/mutation/mismatched-space refusals. |
| D01-S4 | **PASS** | CanonicalReceiverBridgeTests executes actual extracted orchestration with deterministic callbacks under both named routes; default historical replay remains explicit. Full legacy suite pending separately. |
| D01-S5 | **PASS** | Held-out target190 maps to10 without clipping; bridge asserts real query targets exceed1. No current-engine repair is claimed. |
