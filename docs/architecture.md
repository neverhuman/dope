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
verification. `production/kpi-contract-v3.json` pre-registers MFS-v3 and is
not embedded. A v3 scalar stays null until its components and hard gates are
measured. Neither legacy receipts nor V1 Python output acquire an L3 label.
