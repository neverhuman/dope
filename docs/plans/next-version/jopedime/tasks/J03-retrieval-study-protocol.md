# J03 — Prepare the smallest unanswered matched retrieval study

Status: **PLANNED**. No implementation, build, study, CI run, release, deployment, or serving change has been performed by this task pack. Planning source is JopeDime `4548321f0f6c2baabbbec6c33a4a143fa8242800`; paths below are repository-relative unless absolute. Intended isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`, branch `codex/dope-jope-master-plan-20261102`.

Execution admission: obtain an implementation assignment, settle the active implementation queue with its owner, then refresh verified remote `main` in the private lane. PR #2071 (`v1/fastlane`) was active and restructures many listed paths. Re-resolve every path/API and dependency before editing; do not alter that PR or shared branch. Read `AGENTS.md`, `MASTER_PLAN.md`, `docs/plans/next-version/MASTER_PLAN.md` (linked as [shared master](../../MASTER_PLAN.md)), current relevant `AGENT_CHAT.md` claims and worktree registry first. Claim exact files under canonical `AGENT_CHAT.md.lock` before writing; one implementation PR at a time. All shell commands use `rtk`.

Universal forbidden zones: canonical checkout/index, other owners' lanes, credentials/protection, serving pointers, released weights, sealed/meta-test payloads, historical scientific receipts/goldens, and generated `target/health`. No cleanup or scientific compute is granted here. One reused build target; keep `CARGO_TARGET_DIR` unset globally. Cargo commands below set required environment and require allocated CPU capacity.

Priority: **P0**. Dependencies: J02, D00. Cross-repository prerequisites: D00 exposure/identity decisions; D01 corrected evaluation and D02 versioned rank semantics are required if those DOPE endpoints are used.

## Objective and non-goals

Produce a registered, falsifiable CPU study protocol and data-free runner validation for released-DOPE retrieval. First determine whether existing valid evidence already answers it. Task completion authorizes no execution, new protected read, budget reset, optimizer sweep or policy training. Scientific ADVANCE is a separate outcome, not a required result.

## Ordered reads and ownership

Read `docs/same-cut-bench.md`, existing same-cut driver and `crates/jope-bench/src/eval.rs`; experiment registry and prior matched isolation studies; `research/flywheel_selfplay/h228_family_invariant/{PREREG.md,README.md,metrics.json}`; H231/H233 terminal records; `h236_sfadamw` and `h237_adabelief` results; J02 control/receipt contracts. These are historical records, not permission to reopen their payloads or retry them.

Registered study owner claims a fresh unused ID, new `research/flywheel_selfplay/<fresh_id>_dope_retrieval/` files and exactly assigned index rows. Keep placeholder `<fresh_id>` unresolved until registry allocation; never create `h228_dope_isolation`, because H228 is occupied. Integrator owns shared index appends and current Python exception/boundary inventories if the proposed Python runner/tests are added. Historical studies, sealed test data and serving state are read-only.

## Implementation steps

1. Build an evidence matrix of prior question, source/model/recipe, exposed families, controls, budget, endpoints and verdict. If an existing decisive study matches this claim, write an evidence-adoption/stop receipt and omit duplicate compute. H228 is incomplete58/60; H231/H233 incomplete; H236/H237 NO_ADVANCE concern optimizer continuation and do not settle retrieval.
2. If unanswered, preregister baseline, true warm retrieval, DatasetShuffled and RandomEligible with identical frozen eligible support. Retain coordinate-rotation as historical/exploratory if included. Do not quietly substitute ranker-only isolation for end-to-end search consumption.
3. Freeze data roles/family grouping, source/model/context hashes, seeds, incumbent cap, method roster, missing/failure treatment and both evaluation-budget and wall-clock comparisons. Initial product path supports exactly one cut per run; do not promise five-cut retrieval or globally reuse seeds across folds. A future multi-cut study requires a separate per-cut provider/receipt task first. Charge query encoding/retrieval in product cost; report bank construction separately. Existing test-10 is development/exposed, not clean confirmation by default.
4. Predeclare primary real-CV utility/loss, time-to-quality, family harm threshold, positive effect margin, confidence/bootstrap procedure, multiplicity and power. Select numerical thresholds from admitted development analysis, then freeze them before confirmation. All mandatory fields must be concrete before execution; no post-outcome epsilon or sample-size adjustment.
5. Implement a runner with `--validate-only` that checks pinned manifests/roles/contracts without opening scientific payloads or constructing models. Add data-free `test_protocol.py`, exact receipt schema, missing-cell/guard-stop simulation and deterministic control replay. A full run entry point stays blocked without separate compute/data admission.
6. Optional CPU ranker isolation is a distinct registered hypothesis only when earlier evidence leaves it unanswered. Match information/capacity across program-only, descriptors, true latent, shuffled-latent and landmark arms. Do not automatically open adapter training after a null retrieval study.

## Named success criteria and validation

- **J03-S1:** Evidence matrix proves an unanswered question or documents a justified stop without new science.
- **J03-S2:** Fresh ID, exact protocol/source/model/data/exposure pins, numerical decision/stop rules and non-overlapping ownership exist before any run.
- **J03-S3:** Validation-only mode performs zero model construction, training, scientific payload opens or network writes; malformed/incomplete protocols fail.
- **J03-S4:** Outcome logic distinguishes COMPLETE, INCOMPLETE, NO_ADVANCE and ADVANCE; ties do not advance and missing cells remain unknown.

Proposed commands after ID allocation:

```sh
rtk proxy python3 -m pytest -q research/flywheel_selfplay/<fresh_id>_dope_retrieval/test_protocol.py
rtk proxy python3 research/flywheel_selfplay/<fresh_id>_dope_retrieval/runner.py --validate-only --protocol research/flywheel_selfplay/<fresh_id>_dope_retrieval/protocol.json
rtk git diff --check
```

Angle-bracket path is deliberately non-runnable until allocated. Expected artifacts: evidence matrix, preregistration, exposure ledger, protocol validation and no-payload-open audit. A future study can support defaults only after complete positive effect/harm/cost gates; null, harmful or incomplete keeps retrieval opt-in.

## Stop, rollback, and handoff

Stop on overlapping ownership, changed prerequisite identity, unresolved #2071 API conflicts, guard failure, private-data exposure, or unavailable resource admission. Retain failed evidence; do not update goldens or thresholds to force a pass. Roll back only this task's private-branch changes; preserve historical objects and other owners' edits. Handoff records stopped full HEAD, dirty files, exact scope, command exits and actual collected/executed/skipped counts, receipt paths/hashes, exact-head review/required CI, remaining blockers, and retained/released ownership. A zero-test filter, missing CI, or unrun benchmark is not a pass. Store compact receipts outside bulk source trees under `/home/ubuntu/jope_plan/evidence/next-version/J03/<full-head>/`; retain `RECEIPT.json`, `commands.log`, and `artifacts.json` mapping each named success criterion to evidence; unknown outcomes/cost remain null.
