# Strict16 qualification and first native continuation

This fixed snapshot ends at the first successful native operation under the declared 16 GiB RAM profile. It contains two completed generated qualification operations and one native continuation operation. Later operations are excluded.

The 500 logical slots contain **110 scientific terminals: 108 OK and 2 historical artifact caps**, with **390 pending**. The OK total includes 98 retained seed11 successes and 10 new native successes. These counts do not establish fivefit stability, sample completeness, or method completion.

## Measured operations

| Operation | Seed | Parent seconds | Nested seconds (nonadditive) | Model + projection bytes | Maximum observed own GPU MiB | Co-tenant |
|---|---:|---:|---:|---:|---:|---|
| qualification_1 | 11 | 224.843454 | 224.598725 | 2323 | 430 | false |
| qualification_2 | 11 | 112.356577 | 112.229968 | 2323 | 430 | false |
| first_strict16_native | 37 | 109.859873 | 109.714889 | 762 | 430 | false |

The two generated qualifications each charged 2,323 model plus projection bytes and produced matching model, projection, and inspection hashes. Their **337.20003135968 parent seconds** are separate R&D, outside the final cell budget. Historical qualification cost remains separate.

The first strict16 native operation charged 762 bytes and **109.85987342894077 parent seconds**. Native and prefit infrastructure paid history totals **924.4300867351703 seconds across 111 trials**. This includes the preserved **18.162802021950483 seconds** of historical prefit infrastructure. The successful successor resolves that same logical slot; the failed physical attempt remains in the history and does not add a 501st slot.

## Cost and observation scope

Each row uses its recorded parent controller elapsed time. Nested operation elapsed time is shown for reconciliation and is **nonadditive**. Neither clock is a GPU kernel fit timer. Retained seed11 costs, the earlier native8 slice, closed evaluation ledgers, and qualification clocks are not charged again into this native total. Campaign total wall time, CPU core seconds, energy, hardware model, and true RAM/VRAM peaks are null.

The three displayed receipts record `co_tenant=false` and `shared_GPU_cost=false`. Sharing for the prior nine native operations and foreign DLTrain participation in the campaign mutex remain null. The fixed serial mutex scopes our campaign jobs. No full runtime, production, or zero impact certification is claimed.

The 430 MiB values are maximum **observed** owned process GPU residency, using verified PID/start identities. Recorded maximum sampling gaps are retained in JSON and CSV. These values are not true peaks, complete host RSS measurements, or capacity qualification for another dataset.

The profile retains CPU80–95 on xbabe2, GPU0, a declared 16 GiB RAM envelope, 8 trials per cell, 43,200 inclusive seconds per cell, a 600 second operation ceiling, and a 240,000 second new operation budget. The stored `native_operations_started=0` counter is explicitly non-authoritative; receipt and charged trial joins determine progress.

Official TEST data remains sealed. MFS v2, PTF v1, release safety, superiority, native winner, sample completeness, fivefit stability, and method completion remain null.

## Reproduction

`source-proof.json` contains only immutable metadata digests, byte counts, source pins, and scalar joins. It includes no private receipt paths, model/sample payloads, or source values. The standard library renderer verifies the four committed JSON/schema buffers before decoding and never reopens private inputs.

```sh
python3 -I -S -B research/benchmark/results/work-order-b-strict16-qualification-progress-v1/render.py --check
python3 -I -S -B research/benchmark/results/work-order-b-strict16-qualification-progress-v1/test_render.py
```

Running the renderer without `--check` regenerates this README, `costs.md`, and `costs.csv` from the committed scalar ledger.
