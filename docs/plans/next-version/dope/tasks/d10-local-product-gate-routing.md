# D10 — Close the local product-check routing gap

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P0**. Prerequisites: D00.

## Objective and non-goals

Make the recommended local precommit path unambiguously include standalone product correctness when product code changes. Current `Justfile::fast` checks root `dope-kernel` and Python; `check: fast` does not depend on `engine`. Existing `just engine` runs the product gate and `ops/ci/cpu.sh` already invokes `ops/ci/engine.sh`. This is a local routing/documentation task, not a claim hosted engine CI is absent, permission to change required contexts, or a request for every edit to run every broad check.

## Ordered read-first files

1. Current repository `AGENTS.md`, live claims/open queue and `ops/AGENTS.md`; `docs/testing.md`, `docs/architecture.md`, maps and D00 decisions.
2. `Justfile::{fast,check,engine}`, `ops/ci/{engine,cpu}.sh` and relevant `.github/workflows/` routing read-only.
3. `agent/test-map.json` entries for `engine/`, `Justfile` and product source subtrees; existing CI/gate script tests.
4. Current canonical dirty-file inventory, especially workflow/shared-levels scripts. Do not conflate pending owner changes with baseline.

## Narrow scope and conflicts

Gate owner may edit `Justfile`, `docs/testing.md` and the `Justfile` route in `agent/test-map.json`. Edit `ops/ci/engine.sh` only if a reproduced problem requires it; otherwise call it unchanged. Workspace/lockfiles, workflows/protection, vendor code, scientific job scripts and held evidence are forbidden. Integrator serializes maps with D03/D05. Read `contracts/AGENTS.md` only if an actual new contract edit is proposed; that is not expected here.

## Small steps and design choice

1. Trace the actual local and hosted invocation graph at the refreshed source pin. Record that `cpu.sh` already runs engine. Verify whether intervening main has already fixed the local gap; if so, deliver evidence-only completion without duplicate recipe.
2. Prefer a named combined recipe such as proposed `product-check: engine` plus the appropriate lightweight repository checks, or make current precommit `check` include engine once if the current owner explicitly chooses that interface. Do not make `fast` silently become a training/benchmark runner or call engine twice through a dependency cycle.
3. Preserve `ops/ci/engine.sh` as the single implementation of format/lint/locked test discovery. Existing CPU entry and `just engine` should still resolve to that script. New CLI/helper is unnecessary unless existing mechanisms cannot express routing.
4. Add only meaningful route validation: a bounded dry-run or script fixture proving engine is selected once and a failing engine exit cannot be swallowed. Do not write a test that merely duplicates the literal implementation string. Use existing script-test style if present; otherwise preserve command trace as receipt and avoid speculative scaffolding.
5. Document exact worker choice: engine edits run focused tests then `just engine` and required final repository checks; unrelated docs use current route. Explicitly exclude model campaigns and hardware latency qualification from correctness CI.
6. Update owner/test map intent so new SDK/norm/DVS routes do not rely on root-only checks. Check docs do not assert every old success proves a new head.

## Individually verifiable success criteria

- **D10-S1:** Invocation graph identifies the existing hosted engine call and the baseline local gap accurately.
- **D10-S2:** Chosen local product/precommit command reaches the same engine script exactly once, without cycles or hidden campaign work.
- **D10-S3:** A product-gate failure propagates as nonzero; no `|| true`, masked exit or zero-test success appears.
- **D10-S4:** Root fast/check behavior changes only as documented; current required hosted workflow contexts and protections are untouched.
- **D10-S5:** Maps/docs give a fresh worker exact commands for engine changes and distinguish correctness from retained-science/hardware qualification.

## Validation and receipts

Existing commands: `rtk proxy just --list`; `rtk proxy just --dry-run engine`; `rtk proxy just --dry-run check`; `rtk git diff --check`. Proposed combined recipe, if created: `rtk proxy just --dry-run product-check`. After current owner admits build capacity, `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only just engine`; required final `just check` remains required by repository policy, not bypassed by a dry-run. Keep dry-run call graph, command exits, actual nonempty engine test counts and full source head. Dry-run is routing proof only; it is not a product correctness PASS.

## Stop, rollback and handoff

Stop if fixing this needs workflow/protection changes, duplicate gate implementation, unauthorized broad computation or conflicting map ownership. Revert only own recipe/docs changes while preserving baseline route. Handoff exact HEAD, invocation graph before/after, measured versus unrun checks, recipe choice and next D03/D04 engine packet. If baseline already supplies the desired route, report verified no-op rather than creating churn.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D10/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D10-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
