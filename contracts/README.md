# Generated contract seam

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

`production/kpi-contract-v3.json` pre-registers MFS-v3. It is not the active
embedded contract. v2 stays the file historical receipts pin. A v3 scalar is
null until its components and hard gates are measured.
