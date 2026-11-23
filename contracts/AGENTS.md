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

Rust-core also owns `contracts/dope-fidelity-v3-candidate.json`, the active unfrozen Fidelity Engine 2.0
candidate contract registered in `agent/generated-zones.toml` (not a parked implementation).
Migration plan: freeze into the v2 release contract once `dope_kpi` confirmation receipts pass
(review expiry `2027-06-01`). Proof lane: `cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test fidelity_v2_e2e`.

Rust-core also owns `contracts/dope-kpi-v2.json` (`dope.kpi-contract/v2`), the unfrozen candidate
scoreboard protocol (TRG statistic, family bootstrap, promotion rule, byte grid, sign-flip null)
that `dope_kpi primary/compare/null` compile in and hash into every receipt; registered in
`agent/generated-zones.toml`. A change to it changes every receipt's contract SHA-256. Freeze
(`protocol_frozen: true`) only by an owner-approved decision before any trained-model outcome is
read. Proof lane: `cargo test --manifest-path engine/Cargo.toml --locked -p dope-arena`; reference
in `engine/dope-arena/docs/kpi-v2.md`.
