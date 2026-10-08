# Native index008: measured GPU operation and frozen progress

This slice records one accepted native GPU operation for lineage `060b9fff8092db17`, fit seed 37, attempt 1. Its whole native parent wait was 424.280335310 seconds. The nested 424.037508198 seconds and operator 425.132649991 seconds overlap it and are not added.

The fixed 500-slot schedule has 111 accepted fits (98 historical seed 11 reuses and 13 new fits), 2 historical artifact caps, 5 distinct unresolved infrastructure slots, and 382 snapshot-unstarted slots. There are 119 paid physical trials; one resolved prior infrastructure attempt remains an extra paid trial for an already accepted slot. Historical reuse retains its original source and runtime evidence.

Two of 100 lineages have all five accepted fit seeds. This counts accepted artifacts only. Sampling completeness, common quality, five-fit stability, MFS-v2, PTF-v1, formal privacy, superiority, method completion, and release safety remain unavailable. An empirical privacy observation does not establish formal DP or de-identification.

New-operation native parents total 2155.040604033 seconds. The selected historical-inclusive cell ledger totals 2949.580656322 seconds and includes 794.540052289 seconds of selected historical seed 11 cost. These are distinct cost bases. Generated qualification (337.200031360 seconds), released sampler metadata hold (1670.322089693 seconds), unpaid admission attempts, and publication work are excluded from native parent time. This is not an exhaustive architecture research ledger.

Four retained GPU-clock observations measured maximum owned GPU residency of 430 MiB and maximum owned RSS of 949,403,648 bytes. These are observations, not true peaks. `co_tenant=false` and `shared_gpu_cost=false` apply to those operation observations only. Neither zero impact nor full-system runtime closure is certified. Hardware model, energy, true RAM and VRAM peaks, and kernel-only fit time are NULL.

The request reserves 16 GiB RAM and CPU 80–95. Its prior snapshot measured 60,002,570,240 bytes of MemAvailable against 52,479,131,648 bytes of aggregate RAM reservation on xbabe2. Admission headroom differs from process RSS and provides no future capacity or peak-memory guarantee. The prospective GPU residency threshold of 11,324 MiB is not a certified internal allocator hard cap.

`source-proof.json` binds immutable receipt digests and sizes without private paths or payloads. `render.py` reads only this rights-safe leaf. Run `python3 -I -S -B render.py --check` to check the projections, or use a fresh `--output-dir` to regenerate the CSV files and table. Validate `result.json` against `schema.json` and `source-proof.json` against `source-proof.schema.json`. Later operations are outside this fixed cutoff.
