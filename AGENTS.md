# Dope repository guide

## Current language policy — 2026-11-03

The latest user instruction is **Python ONLY for deep-learning TRAINING; everything else in DOPE must be Rust**. This supersedes historical V1, advanced-data, benchmark and research-Python exceptions as end-state permission. Python may own actual neural model/loss/autograd/optimizer steps and necessary training-coupled batch/RNG/checkpoint state. Standalone preparation, inference, classical evaluator fitting, validation/metrics/Elo, reports/qualification, CLI/orchestration, CI/checkers/examples and new nontraining tests are Rust. A mixed module or use of a neural model is not an exemption.

**Current status: NOT_COMPLIANT.** The a84 snapshot has 890 tracked Python files. Inventory/classification and seven source-only Rust packets are migration work, not a waiver or completion. Preserve all old source, readers, fixtures, models, results, receipts, custody and test intent until reviewed native replacement coverage exists; no deletion/renaming or guard/exclusion weakening. Do not add new nontraining Python or hide it in a training helper. Read [the migration queue and acceptance gates](docs/plans/next-version/dope/language-policy/README.md) before assigning work. Native/full CI/PR admission and scientific/release gates remain separate.

Follow the host worktree and build rules supplied by the user. Use one writer per
checkout. This repository's release branch is developed in
`.agent/worktrees/integration`; keep generated evidence under `target/` and local
tools under the ignored `.agent/` directory.

- [Architecture](docs/architecture.md) identifies Rust product code, the V1
  Python compatibility boundary, and immutable contract inputs.
- [Testing](docs/testing.md) maps changes to deterministic commands, budgets,
  and repair evidence. Run `just fast` while editing and `just check` before a
  commit.
- [Release](docs/release.md) lists the certification and publication gates.
- [Audit](docs/audit.md) explains the pinned Jankurai evidence and exceptions.
- `agent/owner-map.json` and `agent/test-map.json` route files to owners and
  proof commands. Update them when adding a new source area.

Preserve historical DPK3.2 and v1 receipt decoding. New tier claims require
measured certification; an empirical privacy result is never formal DP or HIPAA
de-identification. Errors and public sidecars must not echo source values,
headers, or extrema.

All new canonical training and evaluation tables must follow
[the mandatory target-last contract](docs/data-contract.md). Resolve the target
from its explicit dataset declaration, normalize external CSVs once with the shared
intake, and reject downstream schema/header drift. Never infer the target merely
from the last source column. Keep query targets out of features and context
selection, and preserve original sources and physical row identities.
