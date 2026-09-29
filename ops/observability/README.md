# Local repair receipts

The CLI writes a `repair_receipt:` JSON line on a failed command. It contains
the stable purpose, reason class, common fixes, local documentation path, and
the narrow rerun hint from `rust/error.rs`. The receipt contains no source
table cell, feature header, or extrema. Keep the human error and this receipt
together in a local issue or CI log; rerun the narrow failing command before
repeating a campaign.
The checked schema is `repair-receipt.schema.json`; the Rust unit test exercises
the serialized fields and source-data omission.

Campaign telemetry and immutable job receipts live in `rust/ledger.rs` and
`rust/campaign.rs`. Certification JSON records gate failures, metric coverage,
privacy attack maxima, contract hash, and candidate lineage. Keep those
content-addressed artifacts under `target/` until review or release storage.
The full audit emits `repair-queue.json` and a hash-only
`report-attestation.json` beside its JSON and Markdown report. Their schemas
are in this directory. The attestation records `signature_kind=none`, so it
proves a local digest and source association without claiming a signature.
It also emits `repair-log.json`, which records a disposition and proof command
for each of the 83 original audit fingerprints. Its schema is in this directory.
