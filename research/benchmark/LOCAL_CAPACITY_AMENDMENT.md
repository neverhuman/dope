# Owner-authorized local capacity amendment

On 2026-10-05 at 21:09 MT, ENG:watcher authorized TabDDPM and Forest-Flow
population continuations and the TabSyn 100-lineage native/default work to write
new artifacts to **xbabe3 local `/home/ubuntu/dope-scratch-x3`**. The receipt was
recorded at **2026-10-06T03:12:29.579657Z**. This gives those operations storage
capacity on a separate filesystem; each operation still requires admission.

The [public record](results/campaign-local-capacity-amendment.json) is a byte-exact
copy of the owner receipt, SHA-256
`3f099815d53b8d009a3425f510a358ffbff11d383ff48bae412c620de3c0dea9`.
Frozen continuation manifests bind this receipt's digest. Its filesystem/free
space fields are the observation at the receipt timestamp, not current capacity.
The [schema](results/campaign-local-capacity-amendment.schema.json) fixes the
authorized scope, caps, host restrictions and null claim fields.

## Resource accounting

Each storage root has its own **200,000,000,000-byte aggregate ceiling**. Count
the entire new root using the recorded apparent-size/count-links semantics,
including full active/requested output reservations. Check actual filesystem
free space separately. The xbabe3 SSHFS alias of fast-scratch is not local disk.

Existing `/mnt/fast-scratch/dope-benchmark` evidence stays in place under its
original ceiling. Only hashes, byte inventories, immutable receipts and small
summaries for new local work go there. New Forest-Flow models are hashed and
charged in full and retained on xbabe3 local disk within that root's ceiling.
Model and projection bytes remain charged to each sampling artifact.

The operation budgets remain eight tuning trials per cell, 12 total hours per
cell counting earlier failed trials and costs, 600 seconds per neural fit and
16 GiB GPU admission. Use at most four disjoint 16-core study slots, one GPU fit
per host, and the shared local GPU lease. Recheck owners before each launch and
yield to JopeDime DLTrain or Hyperion. Only xbabe1/2/3 are permitted; no foreign
process signals or refits of already successful cases are authorized.

## Evidence and verification

The original receipt is retained at both locations:

- xbabe3: `/home/ubuntu/dope-scratch-x3/capacity-amendment.owner-v1.json`.
- xbabe2: `/mnt/fast-scratch/dope-benchmark/capacity-amendments/work-order-b-local-x3-v1/capacity-amendment.owner-v1.json`.

Verify each copy against the digest above. Validate the public copy with:

```sh
python3 -B -m jsonschema \
  -i research/benchmark/results/campaign-local-capacity-amendment.json \
  research/benchmark/results/campaign-local-capacity-amendment.schema.json
```

This record changes no scientific configuration, native objective, tuning
selection, production artifact tier, method outcome or required evidence gate.
Official tests remain sealed. Full campaign admission is false and
MFS-v2/PTF-v1/release/superiority are null in the receipt. Completed and failed
operation outcomes belong to their separate immutable campaign ledgers.
