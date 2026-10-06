# Costs from the closed Forest and TabSyn research queues

This partial ledger adds six current phase rows to the separately preserved
[earlier ledger](../work-order-b-compute-costs-v2/README.md). The reproducible
[table](costs.md) and [CSV](costs.csv) come from [delta.json](delta.json).
The earlier scopes have unproved overlap, so no campaign total is asserted.

Forest records 25 complete required cells, 12 unavailable or incomplete, and
163 unstarted of 200. The current 32 operation receipts total 1838.6582904793322
seconds, including five custody operations totaling 33.93066829815507 seconds.
Custody is metadata overhead. Coordinator and watchdog walls overlap these
phase walls and cannot be added. All 32 operation bodies (57102 bytes) were
authenticated locally before their cost and lifecycle fields were decoded.

Forest controller wait exit is 2; supervisor exit is unknown. Job36's worker
exit is 0 and its watchdog exit is 1, so its fit stays lifecycle incomplete.
The worker and watchdog exit fields are distinct. Three such differences are
retained. Watchdog bodies were outside the 32-operation rehash scope; their
exits remain bound by the frozen closure. This resource closure gives no
method failure or full quality result.

TabSyn records 143 admitted sample cells (48 retained plus 95 new), 39
unavailable (8 prior plus 31 new), and 418 unstarted of 600. Only the 126 new
workers are charged: 3980.3958190921694 seconds. Their raw zero exits do not
upgrade unavailable admission. Parent numeric exit is unknown, and no
`sampler.closed` receipt exists. The pinned source and safe failure frame show
a memory/load assertion before the next allocation; the failing conjunct and
historical memory/load readings are unknown. Original fit clocks are preserved.

Forest retains nine model/adapter/projection charges for jobs28–36 totaling
1752023805 bytes. The 941 metadata files total 6426326 bytes. These are receipt
inventories; current physical model hashes and disk use were not remeasured.
No artifact was deleted. Hardware, peak RAM/VRAM, energy, GPU hours and an
additive campaign cost remain unknown. MFS-v2, PTF-v1, release safety and
superiority remain null. Official tests were not opened.

## Reproduce

Run from any checkout containing these committed files:

```sh
python3 -I -S -B research/benchmark/results/work-order-b-cost-closure-delta-v3/render.py --check
```

The standard library renderer checks frozen JSON digests before decoding,
validates phase/count/lifecycle accounting, and regenerates every table and
schema byte for byte. Without `--check` it creates missing sidecars and rejects
existing sidecars with changed bytes. It starts no learner and reads no rows,
models, samples or external scratch receipts.

[source-proof.json](source-proof.json) pins the original receipts and their
scope. [publication-manifest.json](publication-manifest.json) binds every
generated file and points to private, hash-verified restore custody:
`/home/ubuntu/dope-scratch-x2/work-order-b-publication-custody/cost-closure-delta-v3/receipt-custody.json`
(SHA256 `82a242912c572e3c886852525c446506b038be59817cc8ab91f05ce66082321d`).
The private archive contains metadata/source receipts; restricted rows,+weights and samples stay outside this publication.
