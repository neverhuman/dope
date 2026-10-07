# New sample, qualification and metadata costs

This partial four-row delta follows the preserved [v3 closure
ledger](../work-order-b-cost-closure-delta-v3/README.md) and [v2 compute
ledger](../work-order-b-compute-costs-v2/README.md). The [table](costs.md) and
[CSV](costs.csv) are reproducible from the included [numeric ledger](delta.json).
No campaign total is asserted.

TabSyn's 41 new CPU sample operations cost 1222.4426932735369 seconds:
31 admitted operations cost 926.7678763717413 seconds and 10 unavailable
operations cost 295.67481690179557 seconds. All 41 small exit receipts
(21610 bytes) were authenticated before their elapsed, exit and lifecycle
metadata was read. Every worker exited 0; unavailable admission remains
unavailable. The controller's kernel exit and parent wait remain unknown.
Natural absence twice is retained as a separate lifecycle fact.

At this closure, TabSyn records 174 admitted, 49 unavailable and 377 unstarted
cells of 600. The newly admitted 31 supplement 143 retained admissions.
The original cutoff and nine expired original unstarted cells are preserved.
This queue closure gives no method-completion or full-panel quality claim.

The A952 generated12 qualification whole operation cost 11.803844023030251
seconds and has captured positive qualification and parent exit 0. Its
11.180450793821365-second inner clock overlaps that whole-operation clock and
is not added. Only generated arrays were used; there were no generator fits
or study-data decodes. Sampled summed process RSS reached 255500288 bytes;
this observation is not an internal allocator cap.

Two distinct byte-only transports verified the same 16 metadata files
(344997 bytes each). Their measured walls were 0.258453578222543 and
0.2457813280634582 seconds, totaling 0.5042349062860012 seconds of engineering.
The identical readback digest does not merge separate transport attempts.

Earlier DOPE seven-operation cost 579.6459803101607 seconds, historical
generated research 388.65127065824345 seconds, and previous TabSyn126 cost
3980.3958190921694 seconds are not charged again. The previous TabSyn8 gap
aggregate is outside this delta pending a complete scoped join. Previous
zero-allocation metadata attempts do not invent zero engineering cost.
Hardware, device peaks, energy, GPU hours and historical sharing flags remain
unknown. `co_tenant` and `shared_gpu_cost` stay null when absent from an actual
receipt. MFS-v2, PTF-v1, release safety, superiority and native winners remain
null. Current A952 real metric costs are outside this four-row scope.

## Reproduce

```sh
python3 -I -S -B research/benchmark/results/work-order-b-cost-closure-delta-v4/render.py --check
```

The standard-library renderer checks the included JSON digests before decode
and reproduces tables, finite schemas and publication manifests byte for byte.
Without `--check` it creates missing sidecars and rejects changed existing
bytes. It opens no external scratch input, private receipt, array, row, model,
sample or dependency. [source-proof.json](source-proof.json) records immutable
input digests and the private preparation proof reference; private attempt
payloads stay outside the publication leaf. No learner or probe is started by
publication preparation or reproduction.
