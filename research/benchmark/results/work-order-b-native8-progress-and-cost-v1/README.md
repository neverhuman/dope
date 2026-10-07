# DOPE native8 progress and measured operation cost

This frozen snapshot has **108 closed logical slots of 500**: 106 `ok` (98 historical seed11 reuse and eight newly measured native fits) and two historical artifact caps. The other 392 slots are unstarted in the frozen continuation snapshot. These counts do not establish completion of the 500-slot campaign or stability across five fits. Metadata GPU refusals are not method failures or scientific fits.

The eight actual new fit operations use seeds 23, 37, 53 and 71 twice each. Their inclusive controller parent timers sum to **694.114947 seconds**. The prior seven contribute 579.645980 seconds and the eighth adds 114.468967 seconds. This is a cumulative native8 snapshot; do not add it to another ledger containing any of the same receipt digests.

| Operation | Seed | Status | Parent elapsed (s) | Nested operation (s) | Charged model + projection (B) | Maximum observed owned GPU residency (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 23 | ok | 116.854228 | 116.661916 | 2576 | 430 |
| 2 | 37 | ok | 63.417309 | 63.267591 | 2580 | 430 |
| 3 | 53 | ok | 63.434600 | 63.282576 | 2576 | 430 |
| 4 | 71 | ok | 63.364475 | 63.214608 | 2579 | 430 |
| 5 | 23 | ok | 143.803178 | 143.657754 | 3321 | 430 |
| 6 | 37 | ok | 64.645603 | 64.490361 | 3318 | 430 |
| 7 | 53 | ok | 64.126587 | 63.936683 | 3315 | 430 |
| 8 | 71 | ok | 114.468967 | 114.251185 | 3316 | 430 |
| **Eight-operation snapshot** | | | **694.114947** | **692.762672** | **23581** | **Observation only** |

The nested operation timers are shown separately and must not be added to parent time. Neither timer is GPU kernel time or a separate fit-only/sample-only clock. Historical reused costs, qualification/R&D clocks, metadata refusal clocks and evaluation ledgers are excluded. A campaign-wide elapsed total is unknown.

The 23,581 charged bytes are the receipts' model-plus-projection accounting; no model or projection body was opened. All eight receipts explicitly record `co_tenant=false` and `shared_gpu_cost=false`. These receipt-scoped observations do not certify future exclusivity or foreign mutex participation.

The 430 MiB values are **maximum observed owned GPU residency**, from the sum of verified owned PID/start-tick GPU-memory observations. They do not measure host RSS. Sampling gaps are recorded per operation in JSON/CSV. They are not true peaks or a certified VRAM cap. True peak RAM/VRAM, hardware models, CPU time, GPU kernel time and energy remain null.

Historical trials and the existing deadline are preserved. Per-cell caps remain eight trials and 43,200 seconds, each new operation retains its 600-second ceiling, and the new-operation budget retains 240,000 seconds. The eighth operation changes one cell's inclusive trial count from four to five. The charged ledger's stored `native_operations_started=0` was not updated by the controller and is preserved as a non-authoritative field; eight actual fits are established by their immutable terminal and parent receipts.

`source-proof.json` lists immutable SHA256/byte references without private paths or payloads. The renderer hashes all four included JSON/schema buffers before decoding, validates strict public schemas and receipt joins, and regenerates this README, `costs.md` and `costs.csv` using only included rights-safe JSON:

```sh
python3 -I -S -B render.py --check
python3 -I -S -B render.py --output-dir PATH
```

Official TEST remains sealed. MFS-v2, PTF-v1, release, superiority, full runtime certification, native winner and five-fit stability claims remain null. Later operations are outside this frozen snapshot.
