# Contracts

This directory owns machine-readable protocol contracts (`contracts/tdl-contract-v1.json`)
validated against `schemas/tdl-contract-v1.schema.json`.
Forbidden: setting `protocol_frozen: true` while any decision field remains `TBD` or `TBD-D0x`,
or placing source table values, headers, or private hostnames in contract files.
Proof lane: `python3 -B -m unittest research.transfer.tests.test_evidence -v` and `just fast`.
