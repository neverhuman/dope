# Next version DOPE tasks

> **Current execution status:** see [STATUS.md](STATUS.md). The planning text and table below preserve the original planning snapshot; their PLANNED wording is historical, not a current implementation or qualification claim.

DOPE changes are justified, starting with trustworthy measurement and an identity-preserving facade/landmark path. The new normalization/model lane follows those foundations. All 12 tasks are **PLANNED**; none has been implemented or scientifically qualified for this assignment.

Read [the shared master](../MASTER_PLAN.md) and [execution rules](../EXECUTION_RULES.md), then open one task file. Every task has its own objective, exact read-first files, ownership, implementation, named acceptance criteria, tests, failure rules and stopped-head handoff. Do not assign a whole row group to a worker as one unbounded task.

Planning source: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Isolated lane: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch `codex/dope-jope-master-plan-20261102`. Existing canonical workspace/MLC changes belong to other owners. Refresh paths and ownership before execution.

## Recommended order

| Task | Result | Priority | Task dependencies | Success criteria |
| --- | --- | --- | --- | --- |
| [D00](tasks/d00-contracts-lineage-and-decision-record.md) | Freeze contracts, lineage and exposure decisions | P0 | None | 5 |
| [D01](tasks/d01-prospective-evaluation-coordinate-repair.md) | Repair prospective evaluation coordinate identity | P0 | D00 | 5 |
| [D02](tasks/d02-versioned-driver-rank-successor.md) | Add informative driver-rank semantics without rewriting history | P0 | D00 | 5 |
| [D10](tasks/d10-local-product-gate-routing.md) | Close the local product-check routing gap | P0 | D00 | 5 |
| [D03](tasks/d03-minimal-sdk-facade.md) | Expose a minimal identity-preserving DOPE SDK | P0 | D00, D10 | 5 |
| [D04](tasks/d04-high-landmark-dependency-plan.md) | Skip unused High probes while preserving admitted conditioning | P1 | D00, D10 | 5 |
| [D05](tasks/d05-nc1-private-normalization-contract.md) | Implement additive FIT-only normalization with private owner state | P1 | D00, D03, D10 | 6 |
| [D06](tasks/d06-dvs-and-audit-card.md) | Implement one versioned downstream-value receipt and Audit Card | P1 | D00, D01, D02, D05, D10 | 6 |
| [D07](tasks/d07-dml4-wire-reference-sampler-and-hard-caps.md) | Freeze DML4 wire, basis and reference sampler before training | P1 | D00, D03, D05, D10 | 6 |
| [D08](tasks/d08-batched-resumable-trainer.md) | Build bounded batched training and exact resume around frozen wire | P1 | D00, D05, D07, D10 | 5 |
| [D09](tasks/d09-v4-train-what-ships-objective.md) | Implement v4 objective and model against the served graph | P1 | D00, D04, D05, D06, D07, D08 | 6 |
| [D11](tasks/d11-qualification-campaign-readiness.md) | Prepare qualification gates without launching or promoting a model | P1-gated | D00, D01, D02, D10, D03, D04, D05, D06, D07, D08, D09 | 6 |

D10 comes early despite its ID: hosted CPU already tests the engine, while local combined check routing needs an explicit product path. D07 freezes wire/basis/reference sampling before D08/D09 training implementation. D11 produces a qualification-ready manifest and launch handoff; it is not a training run, release or adoption result.

## Cross repository handoffs

- D00 ↔ J00 agree on distinct engine, standalone model, fused checkpoint, preparation and evaluator identities; neither edits the other's live records.
- D03 → J07 supplies the immutable facade and parity evidence before consumer migration.
- D05 → J05/J06 supplies FIT-only NC1 and private-card/public-schema contracts.
- D06 → J08 supplies a versioned DVS adapter contract; selection metrics remain honest about held-out roles.
- D07/D08/D09 → D11 establish engineered v4 readiness. J09 growth still requires actual qualified model/codec value, not D11 readiness alone.

## Parallel and protected paths

Evaluation and JopeDime retrieval can proceed in separate repositories after their queue/claims clear. Within DOPE, one integrator serializes `engine/Cargo.toml`, lockfiles, root/module registration, CLI, schema and owner/test-map updates. D03/D04 and D05/D06 have separable module scopes but shared integration cannot race. Existing v1/v2 metric recipes, v3 preparation, DML3 decoding, released weights, raw source and historical receipts stay unchanged.

[Task inventory](tasks/inventory.json) records per-file hashes and 65 criterion IDs as a planning snapshot. The existing project ledger remains the live ownership/status source. Future receipts use `/home/ubuntu/jope_plan/evidence/next-version/<TASK_ID>/<full-head>/` and map every success ID to concrete evidence or a blocker.
