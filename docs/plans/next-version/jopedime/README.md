# Next version JopeDime tasks

First lock current identities, then make existing DOPE embeddings affect search through an opt-in, TRAIN-only retrieval consumer. Preserve the current model and seed budgets. Add shared normalization/emission and gateway consolidation after their DOPE contracts exist. All 10 tasks are **PLANNED**; no implementation or scientific qualification occurred for this assignment.

Read [the shared master](../MASTER_PLAN.md) and [execution rules](../EXECUTION_RULES.md), then assign one task file. Each file names exact existing/proposed paths, ownership, steps, its own success criteria, checks, stops and handoff.

Planning source: `4548321f0f6c2baabbbec6c33a4a143fa8242800`. Isolated lane: `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102`; branch `codex/dope-jope-master-plan-20261102`. PR #2071 was active and restructures thousands of paths; source work waits for its current owner/queue handoff, then refreshes APIs and claims. Do not change shared fastlane or canonical dirty files.

## Recommended order

| Task | Result | Priority | Task dependencies | Success criteria |
| --- | --- | --- | --- | --- |
| [J00](tasks/J00-baseline-identities.md) | Lock baseline identities and establish the execution checkpoint | P0 | D00 | 3 |
| [J01](tasks/J01-v2-seed-transport.md) | Carry existing v2 seed provenance into real generation zero | P0 | J00 | 4 |
| [J02](tasks/J02-controlled-retrieval.md) | Add identity-bound DOPE bank retrieval and decisive controls | P0 | J01 | 4 |
| [J03](tasks/J03-retrieval-study-protocol.md) | Prepare the smallest unanswered matched retrieval study | P0 | J02, D00 | 4 |
| [J04](tasks/J04-cpu-packaging-defaults.md) | Qualify CPU packaging and explicit landmark/device semantics | P1 | J02, J03 | 4 |
| [J05](tasks/J05-nc1-additive-intake.md) | Recover raw provenance and add frozen NC1 dataset views | P1 | J00, D05 | 4 |
| [J06](tasks/J06-private-scalecard-emission.md) | Emit owner-side ScaleCard stages for raw-row prediction | P1 | J05, D05 | 4 |
| [J07](tasks/J07-facade-migration.md) | Adopt the qualified SDK facade without changing frozen semantics | P1 | J00, D03 | 4 |
| [J08](tasks/J08-dope-auditor.md) | Expose bounded JopeDime search as a DOPE downstream auditor | P2 | J05, J07, D06 | 4 |
| [J09](tasks/J09-growth-readiness-routing.md) | Route optional growth through evidence-backed GO or HOLD decisions | P2 | J03, J06, J08, D07 | 4 |

J01/J02 use a data-only post-partition provider from CLI orchestration, initially one cut and the ordinary CPU incumbent cap of four. They do not add a pipeline-to-learning dependency or share supervised seeds across held-out cuts. The separate neural path's sixteen slots are outside initial scope. Historical coordinate rotation remains distinct from new dataset-shuffle/random controls.

J03 completes a study-ready protocol and fixture suite, not scientific ADVANCE. J04 may change runtime defaults only after a separately admitted positive study plus measured packaging/latency gates. J05/J06 and independently useful DOPE artifact work do not require retrieval to win. J07 preserves the frozen fused ranker; duplicate retirement needs its own qualified replacement and task.

## Cross repository handoffs

- D00 supplies J00's shared lineage/role naming decisions; J00 initially locks the current engine/model without a re-pin.
- D03 supplies J07's immutable facade and parity evidence.
- D05 supplies J05/J06's frozen NC1 and private owner-card contract.
- D06 supplies J08's versioned DVS auditor contract.
- D07 and actual qualified model/value evidence constrain J09 optional growth. D11 readiness alone is insufficient.

J09 is a GO/HOLD routing task, not a bundled artifact-search/curriculum/adapter implementation. Each GO needs a new standalone task and separate execution assignment before any work.

## Ownership and evidence

One implementation PR at a time under current repository rules. One integrator owns Cargo/lockfiles, module/CLI/schema registration, source/test inventories and live delivery appends. Do not parallelize J01/J02 plumbing or J05/J06 contracts before their interfaces freeze. Frozen metrics, model/normalizer/anchor guards, historical studies, generated registries, sealed data and serving pointers remain read-only.

[Task inventory](tasks/inventory.json) records hashes and dependencies as a planning snapshot. Future receipts use `/home/ubuntu/jope_plan/evidence/next-version/<TASK_ID>/<full-head>/`, mapping each named criterion to exact evidence or a blocker. The existing project ledger remains the live ownership/status source.
