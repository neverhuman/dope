# D11 — Prepare qualification gates without launching or promoting a model

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P1-gated**. Prerequisites: D00, D01, D02, D03, D04, D05, D06, D07, D08, D09, D10.

## Objective and non-goals

Produce a campaign admission packet after engineering correctness gates pass. Ends at **READY_FOR_OWNER_ADMISSION** or a documented blocker. It does not run a campaign, fetch source data, reserve GPU, dispatch CI/science, publish S3 objects, release/tag weights, repin consumers or change serving. A ready plan is not a measured model success.

## Ordered read-first inputs

1. Current user instruction, repository `AGENTS.md`, live ownership/open PR/experiment registry, `docs/{testing,release,architecture}.md` and current compute admission policy.
2. D00 decision/exposure ledger; exact stopped-head receipts from D01–D10; D06 DVS schema/inference; D07 wire/basis registry and D08/D09 graph/resume proofs.
3. `research/levels/{v3_ablation_and_eval,v3_canonical_runner,train_landmark_dropout}.py` only to understand actual launch/checkpoint interfaces; current model release inventory and independent engine/model asset semantics.
4. JopeDime J00 pin identities, J02 retrieval contract, J03 study-readiness packet and J07 gateway adoption gates. Coordinate read-only; do not take those owners' registry/claim paths.

## Files and ownership boundaries

Campaign planner owns **new** `docs/plans/v4-qualification-readiness.md` and **new** `docs/plans/v4-qualification-manifest.json`. Existing experiment registry receives an entry only through its current owner after agreeing a fresh ID; no reused H228 or invented next number. Preregistered claims/results, sealed families, source datasets, serving pointers, cloud stores, model files and release workflow/protections are read-only. This documentation packet cannot authorize its own launch or infer budget from another study's unused allocation.

## Ordered readiness work

1. Verify engineering receipt hashes bind current full source HEAD, corrected evaluator, successor metrics, NC1, wire, model/trainer graph and checkpoint identity. Record every missing/stale/failed lane explicitly. Final source changes invalidate previous exact-head evidence until rerun/review.
2. Draft a finite cohort with immutable physical-family roles: gradient TRAIN, development/TUNE, independent confirmation and any sealed/custodian role. Exposure ledger records which historical panels already influenced architecture/defaults. Unknown raw provenance is excluded or blocks the relevant claim; transformed values cannot reconstruct it.
3. Freeze experiment hypotheses, comparisons, scheduled cells, seeds, task strata, synthetic row policy, equal-complete-byte raw-row/current-release controls, R/D/S endpoints, harm tails, multiplicity, uncertainty and failure denominators **before outcomes**. Epsilon, macro N, architecture, step count and numerical tolerances are development decisions with a documented freeze, not values selected on confirmation. Do not invent numeric pass thresholds in this task; owner/statistical decision fills them before readiness.
4. Separate engineering profile, v4 training, development model selection and one untouched confirmation into distinct phases with source/checkpoint lineage. Define exact checkpoint selection, termination, no-retry/repair policy, resume semantics and what counts as a fresh scientific attempt. Preserve failures, actual optimizer updates, costs and incomplete status.
5. Specify resources from fresh capability evidence, not this planning snapshot. Name requested host/device/thread/memory/disk/wall limits, admission lock, output custody and owner responsible for containment. Leave ungranted budget null; no inherited campaign clocks or resource claims. CPU fixture success does not establish CUDA availability/latency.
6. Resolve recovery route before launch: durable checkpoint/raw logs, per-cell identity/cost, immutable artifact manifest and restore verification. Model promotion requires separate reviewed release qualification and later JopeDime identity/freeze/efficacy adoption; all remain NOT_RUN.
7. Create an owner-facing final approval packet containing exact commands only after implemented `--help`/schema/fixture dry-run proves they exist. No command is executed here except bounded document validation. Explicitly list actions requiring future assignment/admission and current blockers.

## Individually verifiable success criteria

- **D11-S1:** All engineering prerequisites reference current exact-head passing receipts or readiness remains blocked.
- **D11-S2:** Finite cohort/physical-family/exposure roles and fresh experiment IDs are verified without opening prohibited outcomes.
- **D11-S3:** Hypotheses, statistics, thresholds, comparator byte/compute accounting and failure policy are frozen before any campaign outcome.
- **D11-S4:** Named owner, resource request/admission procedure, containment, custody and resume rules are concrete; absent authorization remains null.
- **D11-S5:** Plan distinguishes profile/training/dev selection/confirmation/release/adoption states and marks every unrun state accurately.
- **D11-S6:** Handoff is reviewable readiness, with no launched process, source download, published model or changed live pointer.

## Validation and expected evidence

Existing bounded checks: `rtk proxy python3 -m json.tool docs/plans/v4-qualification-manifest.json`; `rtk git diff --check`; path/hash/link validation against prior receipts. Proposed schema validation verifies required fields/statuses and refuses `ready=true` with unresolved gates. Retain validation output, exact document hashes, prerequisite receipt inventory and unresolved-owner list. Do not rerun tests/training to fill missing proof under this planning task; route that requirement back to its engineering owner/admitted lane. Unknown costs/timing remain null.

## Stop, rollback and final handoff

Stop readiness on stale source, reused exposure/ID, missing budget, weak denominator/inference, unavailable recovery or resource ownership conflict. Preserve drafts with BLOCKED status and exact reasons; do not weaken gates or silently launch. Handoff stopped HEAD, readiness hash, owner decision list, proposed commands, blockers and explicit next authorization boundary. Actual model qualification is a separate future action.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D11/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D11-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
