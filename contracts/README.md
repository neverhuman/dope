# Generated contract seam

The governed TDL successor is `contracts/tdl-contract-v2-candidate.json`.
Rust-core owns it; `schemas/tdl-protocol-v1.schema.json` admits its protocol and
`schemas/tdl-result-v1.schema.json` admits its result shape. The existing metrics
module embeds these inputs into Cargo build output, registered in
`agent/generated-zones.toml`. Its replayable proof is
`cargo test --locked --lib metrics::` plus
`python3 -B -m unittest research.transfer.tests.test_tdl_result_schemas -v`.
The candidate expires as a design state only after admitted calibration,
roster and multiplicity decisions; it grants no scientific certification or
release authority. `docs/decisions/tdl-v1.md` records migration and the single-PR
scope. Historical `contracts/tdl-contract-v1.json` and its schema remain read-only.

`production/kpi-contract.json` is the canonical v2 release contract. `build.rs`
validates its version and frozen L2/L3 byte limits, then generates Rust tier
constants into Cargo's `OUT_DIR`. `rust/contract.rs` imports those generated
constants. The build also embeds SHA-256 and BLAKE3 digests, and
`rust/production.rs` validates the full typed contract at runtime.
`just contract-drift` compares those generated values with the embedded v2
contract, checks the historical v1 fixture, and writes a digest receipt.

`production/kpi-contract-v1.json` is a historical input for read-only receipt
verification. New freezes and releases bind v2. The contract test and the
locked CPU CI lane check the generated seam on every change.

`production/kpi-contract-v3.json` pre-registers MFS-v3 and
`production/kpi-contract-v4.json` pre-registers MFS-v4. Neither replaces the
active v2 release contract that historical receipts pin. A v3 or v4 scalar is
null until its components and hard gates are measured.
