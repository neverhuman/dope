# Contracts

This directory owns machine-readable protocol contracts. Rust-core owns
`contracts/tdl-contract-v1.json`, validated against the historical
`schemas/tdl-contract-v1.schema.json`; both remain read-only.
The governed successor is `contracts/tdl-contract-v2-candidate.json`, admitted by
`schemas/tdl-protocol-v1.schema.json` and `schemas/tdl-result-v1.schema.json`.
It is an active, unfrozen candidate input to the existing Rust metrics module,
registered in `agent/generated-zones.toml`, not a parked implementation.
Its migration plan is `docs/decisions/tdl-v1.md`: no inherited certification,
no candidate freeze until the coordinator resolves D05/D19 and required rosters.
Forbidden: setting `protocol_frozen: true` while any decision field remains `TBD` or `TBD-D0x`,
or placing source table values, headers, or private hostnames in contract files.
Proof lane: `cargo test --locked --lib metrics::`,
`python3 -B -m unittest research.transfer.tests.test_evidence research.transfer.tests.test_tdl_result_schemas -v`,
and `just fast`. No advisory exception or generated-output exclusion grants a freeze.
