# Architecture and ownership

`rust/main.rs` owns CLI argument parsing. `rust/data.rs` validates input before
fitting. `rust/contract.rs` freezes tiers, candidate identities, and auditors.
`rust/compiler.rs` fits candidates; `rust/codec.rs` is the canonical DPK
encoder and decoder. `rust/certification.rs` consumes sealed holdout data,
auditor evidence, structure and privacy diagnostics, and `rust/release.rs`
stages a bundle only after all release gates pass. `rust/production.rs` binds
the v2 contract digest and inventories release files. `rust/fitness.rs` and
`rust/fitness_metrics.rs` compute the measured MFS-v2 diagnostic after hard
gates pass.

The GPU feature links pinned libtorch for training and six-auditor evaluation.
`rust/neural_train.rs` owns training; `rust/neural.rs` samples exported tensors
in Rust. The Rust CUDA sampler records a conservative device-used memory peak
while the seeded training lock is held. This is a resource upper bound, not a
per-process allocator statistic. CPU builds report unavailable GPU auditors.

`src/dope_kernel/` is a bounded V1 compatibility implementation. The V3
compiler, sampler, contract, release decision, and campaign receipts are Rust
product truth. The Python V1 boundary has no database access, product routes,
or subprocess execution. `agent/boundaries.toml` and the fresh file-hash/test
receipt prove this boundary before each full audit.

Large Rust modules keep their public module names and split implementation
items into `rust/<module>/` sections through `include!`. The section files
share their parent module's privacy and preserve item order. Candidate and
auditor implementation hashes include their split source files explicitly,
so a section edit changes the evidence lineage.

`validation/external/` is a second dated advanced-data exception for offline
evidence about exact sklearn, XGBoost, and LightGBM models. It cannot write a
Rust certification result or publish a bundle. Its boundary receipt checks
file hashes, pinned dependencies, fixtures, and the absence of network,
subprocess, database, and product-route code before each full audit.

`production/kpi-contract.json` is the active canonical v2 input embedded at
build time. `production/kpi-contract-v1.json` remains for read-only historical
verification. `production/kpi-contract-v3.json` pre-registers MFS-v3 and
`production/kpi-contract-v4.json` pre-registers the historical MFS-v4 draft
(control-contrastive lift, model-driver attribution agreement, and historical
compactness score), superseded by `contracts/tdl-contract-v1.json`. A v3 or v4
scalar stays null until its components and hard gates are measured. Neither
legacy receipts nor V1 Python output acquire an L3 label.

## Unified-spec authority and incremental extraction

Rust owns product decoding, compilation, metric admission and release decisions.
`rust/lib.rs` retains the public exports; the existing compiler and codec facade
paths remain stable. `rust/metrics/` and `rust/transfer_metrics.rs` are the current
metric boundary, with fixture parity against `research/transfer/tdl/`.
Extraction keeps one implementation and the existing call sites. The development
interaction helper's fold-fitting functions live in
`rust/compiler/interactions/support_fit.rs`, included at their original position
in `rust/compiler/interactions.rs` with the same privacy, arithmetic and order.
This helper remains a term-tariff diagnostic; it does not rank production tiers.
Its extraction does not alter production candidate hash inputs. Changes to a
production source closure must include every new section in its lineage hash.

`contracts/tdl-contract-v1.json` and its schema remain immutable historical
inputs. `contracts/tdl-contract-v2-candidate.json` is an unfrozen successor:
`status=candidate`, `protocol_frozen=false`, `certified=false`. Its protocol and
result schemas are `schemas/tdl-protocol-v1.schema.json` and
`schemas/tdl-result-v1.schema.json`. Unresolved calibration and multiplicity
parameters remain null. Artifact qualification, scheduled cells, campaign
completeness, empirical privacy, formal DP, release authorization and claim
provenance are independent state axes; fixture success does not freeze them.

Python adapters under `research/transfer/adapters/` are bounded evidence workers.
Their locked runtime and model admission cannot grant Rust certification.
`research/transfer/frontier.py` retains historical family-coverage diagnostics;
its `evaluate` entry is a direct alias of `research/transfer/tdl/frontier.py`.
The diagnostic coverage and censoring states do not replace TDL receipt states.
The Python component oracle is checked against Rust through shared goldens;
it is not a second source of product metric semantics.

The proposed `packages/protocol/` surface is for generated TypeScript types
from the versioned schemas. It has no independent scorer or release authority.
`src/dope_kernel/` remains the bounded V1 compatibility boundary. DPK3.2 and
historical v1 receipt decoding continue through their existing readers.

Ownership and proof routing use `agent/owner-map.json` and `agent/test-map.json`.
The existing `rust/` route covers the included fitting section as `rust-core`
with `just check`; the historical frontier paths already have exact test routes.
Custody worktrees and ignored evidence are inventoried read-only under `target/`.
Extraction never implies permission to clean, relocate or discard those assets.
