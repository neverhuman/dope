# Closed retained evaluation costs: v5

Four authenticated closed CPU evaluation scopes add no generator fit or sampling costs. TabSyn covers 48 retained evaluations across eight lineages; ARF covers 1,116 physical evaluations / 1,200 logical aliases; GaussianCopula and Chow-Liu cover 2,017 new physical evaluations plus five retained diagnostics / 2,400 logical aliases. Five utility-only actions use those retained diagnostic job identities with auditor seed 1729.

| Evaluation scope | New physical actions | Full-panel logical aliases | Metric timer wall seconds | Outer runner seconds | Process high-water RSS bytes | Host / CPU cores |
|---|---:|---:|---:|---:|---:|---|
| TabSyn | 48 | 48 | 49.574226 | null | 318398464 | xbabe1 / 96–103 |
| ARF | 1116 | 1200 | 1396.094902 | null | 768512000 | xbabe2 / 104–111 |
| GaussianCopula+Chow-Liu | 2017 | 2400 | 2239.001908 | null | 845471744 | xbabe1 / 96–103 |
| GaussianCopula+Chow-Liu utility only | 5 | — | 2.723124 | 7.970948 | 297795584 | xbabe2 / 104–111 |

The new per-job timers sum to **3687.3941588359885 seconds**. These are wall durations from `time.perf_counter`, not CPU time or elapsed campaign time. The diagnostic driver timers include input verification/decoding, A952 diagnostics, utility auditors and final verification/capacity checks; preliminary capacity checks, module setup and receipt writing are outside those timers. The utility-only timer covers input verification/decoding and utility; its separate 7.970948356203735-second outer runner includes setup and postchecks. That outer clock is not added to the timer sum.

Logical aliases never multiply physical costs. The five retained A952 diagnostic timers (31.6338860578835 seconds) are excluded entirely; three overlap ROOT's frozen first12 batch and all five overlap its completed16. ROOT's first12 publication was merged as [PR169](https://github.com/neverhuman/dope/pull/169) at `774c72f16ebae006ad627eddeacf2f5c727fa70b` after exact-head review of `c01048e58a698d9ff4eab1b47cec2d963eb75f0a`. It is a separate committed slice; its evaluator clocks remain excluded from this delta. The previous v4 sample/qualification/engineering clocks are excluded. No cross-ledger campaign total is claimed.

RSS is the process lifetime `RUSAGE_SELF.ru_maxrss` high-water mark in bytes, not summed family RSS, an allocation cap or an additive cost. Each execution records a 4 GiB RAM bound and zero swap; the utility-only bound is retained accepted-launch evidence, not a later live kernel measurement. Host total RAM comes from each frozen receipt lock. CPU/GPU models, GPU device use/time, energy, core-seconds, co-tenancy and shared-GPU cost are unknown and remain null. Parent/kernel exit codes remain unknown; collected-unit defaults and PID absence are not exit0 evidence.

Evaluation receipts are closed; this does not certify full method training or a native winner. Official TEST stays sealed. MFS-v2, PTF-v1, release-safe and superiority remain null. The immutable source and receipt locks remain in private custody. This public leaf contains aggregate costs and digests; no rows, models, samples, headers or source extrema.

## Reproduce

Run `python3 -I -S -B research/benchmark/results/work-order-b-cost-closure-delta-v5/render.py --check` after adoption. It authenticates the included JSON/proof/schemas before decoding and compares CSV, table and README in memory. It needs no private receipt payload, numerical dependency or host call.

- `delta.json` and `delta.schema.json`: measured cost rows and strict metadata shape.
- `costs.csv` / `costs.md`: exact CSV numbers / rounded readable table; CSV `null` means unmeasured.
- `source-proof.json` and its schema: input digests, scoped verification, exclusions and remaining gaps.
- `render.py`: committed-input regenerator.
