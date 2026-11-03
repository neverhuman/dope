# D04 — Skip unused High probes while preserving admitted conditioning

Current status: **SOURCE IMPLEMENTED; SCOPED NATIVE FIXTURES PASS; REMAINING VALIDATION IN PROGRESS.** Mapped criteria below include actual testedbb/mergefee51ff5 native fixtures, with referenced core production and immutable fixture closure unchanged. See [bb scoped evidence](../receipts/hosted-native-bb-criteria.json) and [live ledger](../../STATUS.json). Other criteria and current full CI remain pending; no efficacy, privacy certification, scientific qualification or release is implied. One integrator owns the same isolated fast branch.

Historical planning baseline (retained for audit; current assignment and lane above govern):

> Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.


Priority: **P1**. Prerequisites: D00, D10.

## Objective and non-goals

Avoid computing FINE probes unused by High conditioning while preserving its baseline inputs and full-panel work-reservation/spectrum semantics. First deliver an opt-in identity-preserving optimization. Do not change fold mathematics, impute absent probes as successful zeros, switch CLI defaults, choose N=200/300, overhaul caches or claim latency gains from fixture tests.

## Ordered read-first files

1. `AGENTS.md`, current claims/worktrees, `docs/{architecture,testing}.md`, owner/test maps, D00 recipe decisions.
2. `engine/dope-core/src/landmark/{panel,bundle_scope,cost,progressive}.rs`: `ProgressiveLandmarks::{new_with_scope,run_ids,is_complete}` and full-panel reservation.
3. `engine/dope-core/src/prepare/{conditioning,descriptors,mod}.rs`: group constants, level masks and `build_conditioning_at_level`.
4. `engine/dope-core/tests/{landmark_bundle,landmarks,prepare_parity_a,prepare_parity_b}.rs`, optional-landmark fixtures; `engine/dope-cli/src/mlc_v3_encode.rs`; `docs/optional-landmarks-qualification.md`.

## Write scope and concurrency

Landmark worker owns shared slot/group constant extraction and new requirement planner within `engine/dope-core/src/landmark/`, narrow preparation imports, focused tests and implementation docs. Integrator handles any `lib.rs`/maps registration. D03/D05/D07 workers cannot concurrently edit common module registration. Frozen prepare-v1 fixtures and historical bundle expected bytes remain unchanged. Source/caches/S3, released weights, default policies, recipe mathematics and other owners' worktrees are read-only.

## Ordered implementation with hard cases

1. Freeze baseline masks, High `vec256`, column tensor, Dope512/629 and spectrum results on existing public fixtures. Document requested policy, row scope, seed, budget, thread count and legacy recipe for each case.
2. Consolidate identical slot/group constants from `prepare/conditioning.rs`, `prepare/descriptors.rs`, `landmark/bundle_scope.rs` into one canonical table, preserving index/order exactly. Land this no-math change separately if optimization review becomes large.
3. Introduce proposed `ConditioningRequirements::HighRequiredV1` (new API) derived from actual High consumers. Preserve distinction between requested, not-needed, failed and over-budget probes. Do not use an empty mask to mean successfully measured zero.
4. Preserve full-panel cost reservation from `cost::plan_requested_with_costs(..., &[true;23])`. Compute spectrum once its actual data prerequisites are ready, using the same admission work balance the baseline would have after full-panel reservation. Avoid declaring all 23 probes computed solely to unlock `is_complete()`.
5. Separate conditioning-ready from full-report-complete where needed. Progressive `macro→larger→fine` callers must still complete/report exactly once and reuse folds. A later fine request computes only genuinely missing cells; cached readiness must not suppress it.
6. Compare optimized High against the corresponding baseline with identical inputs/policy, including high/low budgets, row subsamples, missing/over-budget cells and requested order. Macro policy is compared with its matching baseline, not required to equal a full panel arbitrarily. If coordinate drift remains, abandon identity-preserving claim and propose a new recipe separately.
7. Expose the optimized plan only behind explicit opt-in until full parity is established. Default macro/subsample decisions belong to admitted DVS development qualification, not this patch.

## Success criteria

- **D04-S1:** Shared constants retain exact legacy index/order and all existing fixture digests.
- **D04-S2:** Baseline/optimized High conditioning tensors and encoder vectors match under identical policies across the test matrix.
- **D04-S3:** Spectrum admitted cells/budget decisions match full-reservation baseline; missing FINE cells are not fabricated.
- **D04-S4:** Progressive order/idempotence and later FINE completion still work, with no duplicate fold/probe work.
- **D04-S5:** Existing full/none paths and legacy report bytes remain unchanged; no default/recipe identity silently changes.

## Validation, receipts and limits

Existing: `rtk proxy bash ops/ci/landmarks.sh`; `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test landmark_bundle --test prepare_parity_a --test prepare_parity_b`; `rtk proxy python3 -B -m unittest research.levels.tests.test_optional_landmarks -v`. Proposed new `high_requirements` test filter must execute nonzero cases. Final `rtk proxy just engine`, repository checks and exact-head CI follow admission. Retain test matrix, full reservation/admission trace, baseline/new tensor hashes and actual omitted-call counts. Timing is NOT_RUN until separately admitted hardware measurement; 43–78% historic costs are not a guaranteed speedup here.

## Stop, rollback and handoff

Stop on any conditioning/admission drift, false coverage state, claimed-owner overlap or unresolved spectrum prerequisites. Disable opt-in and restore only own unmerged implementation if needed; preserve baseline and evidence. Handoff full HEAD, unchanged identities, requirement graph, matrix results, unsupported cases and precise future profiling prerequisites.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D04/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D04-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.

## Individual engineering verdicts

Verdicts reflect source-bound engineering receipts. Named1811 native fixtures are mapped individually below; new SDK/runtime and current full CI require separate evidence. `PASS` never implies scientific efficacy, privacy qualification or release.

| Criterion | Verdict | Evidence and limitation |
| :--- | :--- | :--- |
| D04-S1 | **PASS** | Root independently read 59b conditioning/descriptors old arrays and e5 canonical panel.rs; each Macro/Larger/Fine index and order exact. Read actual constants_have_one_definition_and_legacy_aliases_keep_coordinates assertion body and actual successful six-test binary log973/980. Recomputed raw Git prepare-v1 tree and landmark_bundle_v1 blob at59b/e5 equal. Gitdiff e5->6191 engine+tests/golden empty. GitHub mergeace parentsmain59b/e5 with identical testedtree independently confirmed. Golden/bundle replay assertions and log outcomes mapped in strongreview; allscope fixture-specific. Scoped immutablefixture/coordinate proof only; D04-S5 remains NOT_RUN. See ../receipts/hosted-native-e5-d04-constants.json. |
| D04-S2 | **PASS** | Actual full_high_none_policy_matrix_uses_real_loader_and_forward_for_all_tasks plus budget/subsample matrix PASS in2/0/0 log994-996. Real synthetic checkpoint loader/parameter inventory, three tasks, equal policies, identical High conditioning/column tensors/digests and actual encoder output; nonzero outputs and perturbation-sensitive routed latents. Reservation/mask/private report distinctions retained; no scientific performance claim. See ../receipts/hosted-native-bb-criteria.json; current fullCI and other criteria remain separate. |
| D04-S3 | **PASS** | Hosted native tight_spectrum_boundaries_reserve_unused_full_panel_cost and high_omits_fine_marks_readiness_and_preserves_conditioning passed at451-equivalent tested tree; full reservation and NotRequested FINE asserted. See ../receipts/hosted-native-451-criteria.json. |
| D04-S4 | **PASS** | high_omits_fine_marks_readiness_and_preserves_conditioning actual log981 PASS: FINE no-op before request, progressive Larger/Macro order, report byte/work equality after repeated run_ids/run_required, explicit Full upgrade, later Fine completion and final report/bundle equality. Existing implementation/test closure unchanged; not the new three-task encoder matrix. See ../receipts/hosted-native-1811-criteria.json; current SDK/full required CI remains separate. |
| D04-S5 | **NOT_RUN** | Historical constructors/default full path unchanged in source; actual legacy report replay pending. |
