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

## Second amendment: xbabe2 local fallback

ENG:watcher separately authorized TabSyn qualification and its 100-lineage run,
TabDDPM continuation when xbabe3 remains held, and parallel DOPE seed-23/37 CPU
refits on the same 100 lineages. The owner message specified 11:42pm MT without
an accompanying date. Its verbatim text and the actual recording timestamp are
in the [xbabe2 receipt](results/campaign-local-capacity-amendment-xbabe2.json),
SHA-256 `f2f3c66bcaa469e58a80195aac9f0612488bf8943b7d37a299e03794f233a506`.
The identical private record is
`/home/ubuntu/dope-scratch-x2/capacity-amendment.owner-v1.json`.

New outputs use **xbabe2 local `/home/ubuntu/dope-scratch-x2`**, with a separate
**100,000,000,000-byte aggregate ceiling**. Count the root's existing contents
and every full active/requested output reservation; check current filesystem
free space before each operation. The recorded free space of 1,046,495,477,760
bytes is a historical observation at 2026-10-06T12:13:27.253385Z. It grants no
current GPU, CPU, RAM or disk admission.

Existing evidence and the copied runtime remain in place. New bulk models and
samples go to the local root; fast-scratch receives hashes, byte counts and
small summaries. The earlier xbabe3 local 200 GB record remains unchanged.
Continuation manifests must bind the appropriate amendment digest and check
the filesystem holding their new output root.

TabSyn qualification precedes the 100-lineage run. GPU workers check actual
owners before each fit and yield to JopeDime/Hyperion at the next idle boundary
without signaling foreign processes. CPU DOPE refits use nice 19 and a disjoint
bounded core slot, reconcile successful seed identities before dispatch, and
record CPU execution separately from historical GPU execution. Existing
successful cases are reused; earlier trial costs and deadlines are carried.
The 600-second fit, 16-GiB GPU, eight-trial and 12-hour cell caps remain intact.

Validate the [xbabe2 schema](results/campaign-local-capacity-amendment-xbabe2.schema.json):

```sh
python3 -B -m jsonschema \
  -i research/benchmark/results/campaign-local-capacity-amendment-xbabe2.json \
  research/benchmark/results/campaign-local-capacity-amendment-xbabe2.schema.json
```

This amendment records authorization and storage evidence. It reports no fit,
CUDA qualification, method outcome or campaign completion. Official tests stay
sealed and MFS-v2/PTF-v1/release/superiority stay null.
