# Native indices011 through014: frozen progress and cost delta

This fixed receipt slice follows the [indices009–010 snapshot](../work-order-b-native-indices009-010-progress-and-cost-delta-v1/README.md). It includes four distinct slots for opaque lineage `06b0312a585b436e`: index011/seed23 is paid infrastructure (`qos_measurement_unknown`), while indices012–014/seeds37,53,71 are accepted fits. No failed slot is retried or treated as a method failure. Later live progress is excluded.

The 500-slot snapshot has **115 accepted fits** (98 historical seed11 reuses and 17 new fits), 2 historical artifact caps, 7 distinct unresolved infrastructure slots, and 376 snapshot unstarted slots. There are 124 distinct accounted slots and 125 paid physical trial units: one earlier infrastructure attempt was resolved by a later accepted trial. Two of 100 lineages have all five accepted seeds; this is identity coverage, with quality, sample completeness and stability still unavailable. Historical reuse retains its original runtime evidence.

## Measured cost and observations

The four native parent waits are 54.942895993590355, 112.79932291200384, 66.20583181316033 and 66.85351031227037 seconds. Their additive delta is **300.8015610310249 seconds** over the frozen121-unit snapshot, giving **2852.2506302162074 seconds** of new-operation parent time. The alternate selected historical-inclusive cell basis is 3646.7906825055834 seconds, including 794.540052289376 seconds of selected historical seed11 cost. These are alternate totals. The earlier paid infrastructure cost remains in its historical parent totals and is not added again.

Nested native elapsed times and overlapping per-operation operator waits are nonadditive. The inherited closed v13 operator timer, prior released-sampler hold1670.3220896930434 seconds, and generated qualification337.20003135968 seconds remain separate excluded bases. Current held-service runtime and all architecture R&D completeness are unavailable; unpaid metadata attempts do not add scientific trials.

Indices012–014 each have five retained GPU observations: maximum observed owned residency430MiB and maximum observed RSS888766464/888815616/888766464 bytes. They observed no co-tenant/shared-GPU cost in those clocks. These are observations, not true peaks, a future guarantee, or causal impact evidence. Index011 has an empty GPU clock; raw defaultfalse sharing fields are normalized to unknown. Parent exit0 does not establish training success. CSV empty cells mean unavailable.

Accepted charged model-plus-projection totals are2619/2619/2617 bytes, comprising model984/984/982 and accounted projection1635 bytes each. Projection accounting is derived from the authenticated charged total minus model bytes; no model or projection payload was reopened. The original scientific7 sources/runtime/binary/product source hashes, 16GiB grant, CPU80–95 pool, 600-second fit timeout,8 trials/43200seconds per cell and240000seconds new-operation ceiling are unchanged. No resident4GiB profile is applied.

MFS-v2, PTF-v1, quality, superiority, release safety, true RAM/VRAM peaks, energy, hardware-model attribution and formal privacy remain null. No train/test/sample/model payload, header, private path or public raw value is included.

## Reproduce public metadata

```sh
python3 -I -S -B render.py --check
```

The standard-library publisher hashes its pinned source-proof before decoding, verifies the seven-file public manifest, validates counts/nulls/nonadditive cost, and reproduces both CSVs and the table from committed rights-safe JSON only. Author preparation parsed source AST and projected verified scalar metadata; it did not execute the publisher or controls. ROOT must run the prepared13 focused controls, renderer replay and both strict schema checks before adopting this disabled proposal.
