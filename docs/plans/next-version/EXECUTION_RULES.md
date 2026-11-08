# Next version execution rules

These rules apply to every task in this planning package. They summarize current constraints; the latest user assignment and current repository instructions win. Planning files authorize no implementation, scientific campaign, deployment or release by themselves.

## Before implementation

1. Read the task, its dependencies, shared master and current repository instructions. Refresh source pins, API paths, active claims, registered worktrees and PR queue. The large JopeDime master merge can invalidate pinned paths. Do not repair or adopt its unrelated work under this assignment.
2. Claim disjoint paths in the existing canonical `AGENT_CHAT.md`, under `flock` on its lock file. Record branch, checkout, full base, owner, scope and intended checks. Read competing claims again while holding the append lock. An old or idle claim is not released by age.
3. Work only in the isolated assigned lane. Keep `.agent/worktrees/` location and `.claude/worktrees` symlink intact. Do not create more implementation lanes to evade the one-PR queue or lane ceiling. Reconcile current over-capacity ownership before implementation.
4. Distinguish existing paths from proposed files. Read nested `AGENTS.md` before entering its area. One integrator owns workspace files, lockfiles, CLI registration, schemas and ownership/test maps. Module workers cannot race on those paths.
5. Reserve compute only under the current owner's admitted limits. One reused build directory per lane; `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; `CARGO_TARGET_DIR` unset globally. No source, build output or custody under `/tmp`.

## Evidence contract

Every task stores its future receipts beneath `/home/ubuntu/jope_plan/evidence/next-version/<TASK_ID>/<full-head>/`, unless the owning repository requires another durable location. Record any substitution explicitly. Do not create a fake full-head directory for a source state that does not exist.

Required small artifacts:

- `RECEIPT.json`: task ID, status, full base/head, branch/worktree, owner, timestamps, exact scope, dependency hashes, each success ID and its evidence or blocker.
- `commands.log`: exact commands, host/environment, exits, duration and tests collected/executed/passed/failed/skipped. Redact secrets and source values before public projection; preserve private raw evidence under its proper boundary.
- `artifacts.json`: durable path/object identity, byte length, SHA-256 and recovery state for each result. Scientific objects also bind physical source, FIT/family/cut, preparation, model, evaluator and preregistration identities.
- `HANDOFF.md`: stopped full HEAD, dirty files, completed/remaining criteria, limitations, ownership disposition, rollback and next task admission.

Use exact named evidence for each criterion. An empty filter, canceled check, unavailable runner, skipped capability or old-head success is not PASS. A local fixture does not establish hardware latency or scientific lift. Missing outcomes/cost remain null rather than zero. Never refresh old goldens or benchmark ledgers merely to make a successor pass.

## Fixed boundaries

Preserve existing DPK/DML readers, v1/v2 metric recipes and released-v3 preparation. Separate engine revision, standalone encoder file hash, fused serving composite, preparation, landmark and evaluator identities. Existing NS3 parameter, buffer, optimizer and eval-mode guards remain enforced.

FIT-only transformation and supervised context apply to each declared cut. Held-out labels never affect context, selection or seed materialization. Retrieval initially uses one cut with a post-partition TRAIN-only provider and existing CPU incumbent capacity; multi-cut support needs an explicit per-cut contract. Private ScaleCards never enter public packet/schema/error output.

Historical scientific records, immutable contracts, released weights, raw datasets, sealed/meta-test data, serving pointers, other workers' targets, and `target/health/` remain read-only. Generated artifacts change only through their owning generator/proof lane. No blanket deletion or foreign job control.

## Stop and final handoff

Stop dependent work on overlap without handoff, changed contract/source identity, missing raw provenance, role/family leakage, private canary exposure, required guard failure, unadmitted compute or unresolved exact-head checks. Preserve evidence and report the responsible owner; do not change thresholds, budgets, credentials or historical expected values to continue.

Follow the current repository PR/review/CI/merge lifecycle at the final head. Historical emergency exceptions do not transfer. A new head requires affected checks and review again. Retain serving adoption and model publication as separate qualified decisions.

Planning lanes stay available for review and follow-up. Eventual lane removal requires preservation and independent restore verification of unique source/evidence, then the supported worktree lifecycle. This planning package grants no cleanup.
