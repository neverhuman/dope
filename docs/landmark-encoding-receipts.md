# Landmark encoding snapshot receipts

`dope mlc-v3 encode --encoding-receipt FILE` writes an optional compact JSONL
companion to the existing stdout or `--out` output. The file must be new and
distinct from the primary output. Existing files, including inputs and symlinks,
are refused. A failed setup can leave an empty new receipt file; choose a new
path for a subsequent invocation.

```sh
dope mlc-v3 encode --weights model.safetensors --csv fit.csv --target target \
  --device cuda:0 --landmark-mode progressive --landmark-subsample 300 --json \
  --encoding-receipt encoding.jsonl --out features.jsonl
```

Each successful encoding produces one snapshot conforming to
[`dope.landmark-encoding/v1`](../schemas/landmark-encoding-receipt-v1.schema.json).
None/fast/full produce one record; progressive flushes the initial zero-landmark
snapshot before starting native landmark work, then appends each completed group.
The companion adds no model forwards or landmark measurements. Omit the flag to
retain the existing output and avoid companion hashing and I/O costs.

The receipt describes the actual **MLC-v3 High629** output: high encoder level,
629 f32 coordinates, and SHA-256 of their little-endian f32 bytes. Its structured
`embedding_identity` names that layout; `weights_sha256` identifies the exact
checkpoint bytes loaded, including custom checkpoints. `conditioning_recipe`
and `conditioning_level` describe the effective input independently of the
encoder level. This schema is distinct from JopeDime's Dope512 product receipt.

`source_digest` hashes the complete prepared FIT partition, including physical
row identities, regardless of landmark sampling. `landmark_source_digest` records
the native bundle scope, or `zero` for the universal empty sentinel. `subsample_n`
is the actual bundle value (null means no subsampling), including null on the
initial zero snapshot. Status is derived from the sanitized 23-slot mask:
`"none"`, `{"partial": n}`, or `"full"`; measured finite zeros remain present.
Native options digest and work budget are null for none/fast and present for
full/progressive. No source values, headers, vocabularies, or embeddings are
included in the companion.

`seq` counts emitted events. `version` starts at zero and advances only when the
existing effective conditioning digest changes; repeated digests retain their
version. Versions are local to this invocation and fixed loaded checkpoint.
They do not identify a new trained recipe or certify model quality.

Receipt write/flush errors fail the command. If native computation or a later
forward fails, earlier flushed snapshots remain valid; the receipt is evidence
of those snapshots, not a completion certificate. Check the command's exit status
and primary event stream for whole-run success. Flushing does not promise disk
durability after a host crash. Existing final stdout and primary event fields
remain unchanged; opted-in I/O and hashing can affect reported elapsed time.

CPU synthetic-checkpoint fixtures exercise the wire contract and compatibility.
They do not qualify GPU performance, retention, cold latency, or a release.
