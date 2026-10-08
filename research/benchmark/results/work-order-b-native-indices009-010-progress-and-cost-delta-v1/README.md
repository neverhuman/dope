# Native indices009 and010: frozen progress and cost delta

This receipt-scoped update follows the [index008 snapshot](../work-order-b-native-index008-real-GPU-progress-and-cost-v1/README.md). Index009 accepted one new fit for lineage `060b9fff8092db17`, seed53. Index010, seed71, ended with an infrastructure failure before any GPU observation or model was retained. Its generic failure reason does not identify the underlying assertion. Neither paid operation is automatically retried or removed from cost accounting.

The frozen 500-slot matrix contains 112 accepted fits (98 historical seed11 reuses and 14 new fits), 2 historical artifact caps, 6 distinct unresolved infrastructure slots, and 380 unstarted slots. There are 120 distinct accounted slots and 121 paid physical trials because one earlier infrastructure attempt was resolved by a later accepted trial. Two lineages have all five accepted seeds. These counts do not establish sample completeness or fivefit stability. Historical reuse is not an upgrade of its runtime evidence.

## Cost and observation scope

Whole native parent waits are 299.15787934092805 s and 97.25058581121266 s: an increment of 396.4084651521407 s over the preserved 119-trial snapshot. New-operation parent time totals 2551.4490691851825 s. The alternate selected historical-inclusive basis is 3345.9891214745585 s, including 794.540052289376 s of selected historical seed11 cost. These are alternate totals, not additive costs. Nested elapsed times, per-operation operator waits, and the 795.9687294093892 s overlapping v13 operator timer are separate and nonadditive. No authenticated whole held-service runtime is added.

Index009 has five retained GPU observations: maximum observed owned residency 430 MiB and maximum observed owned RSS 884449280 bytes. These are observations, not true peaks. Its sharing fields are false for those observations only; no zero-impact or causal claim follows. Index010 has zero retained GPU observations. Its default raw false sharing flags do not establish solo operation, so public observed sharing fields are null. CSV empty cells mean unavailable, not false or zero.

Index009 charges 1970 model bytes plus 3744 projection bytes, totaling 5714. Index010 has no retained model or charged artifact total. No row, sample, model, projection payload, header, or private path is included. The original 16 GiB scientific grant, 16-CPU pool, and 600 s timeout remain the measured profile; a separate resident 4 GiB proposal was not applied.

MFS-v2, PTF-v1, utility, stability, empirical or formal privacy, production certification, superiority, and release safety remain unavailable. This is a partial progress and measured-cost ledger.

## Reproduce public metadata

From this directory:

```sh
python3 -I -S -B render.py --check
```

The standard-library renderer authenticates pinned source-proof bytes before JSON decoding, verifies its seven-file manifest, validates counts/nulls/nonadditive costs, and reproduces both CSVs and `table.md` from `result.json`. It reads no private receipts or scientific payloads. Proposal replay and focused controls are prepared for ROOT execution; no candidate controls or renderer were run by the author.
