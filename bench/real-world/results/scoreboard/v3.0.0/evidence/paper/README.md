# Re-reduced `dope-v3b-auto-v8` vs. External Methods Receipts (No New Data)

These four receipts re-reduce the already-scored frozen report-look rows (`schedule` `14238f1d…`, `contract` `200ea0ed…`, data seeds 5–8, `44` informative cells across `11` lineage families) in the paper's claimed comparison direction:

```bash
dope_kpi compare --candidate dope-v3b-auto-v8 --baseline <method> ...
```

## Provenance & Sanity Replay

- **No new fits, emissions, or evaluations** were performed. The per-cell score ledgers are exact copies of the committed report-look run (`bench/real-world/results/scoreboard/v3.0.0/primary-methods.json`).
- Reduced with the report run's own `dope_kpi` binary (`c01fd919`), which reproduces the committed [`compare-methods.json`](../../compare-methods.json) (`36f00d467dcb56b7…`) byte-for-byte when run in the original direction (`--candidate <method> --baseline dope-v3b-auto-v8`).
- Because this comparison direction was reduced after the methods were scored, the white paper discloses that ordering explicitly alongside Holm-Bonferroni adjustment ($m = 4$) across the four one-sided $p$-values.

## Files (`SHA256SUMS`)

- `v3-vs-arfpy-arf.json` — `DOPE v3` vs. `arfpy-arf` (`+10.8 [−3.5, +29.5]` pts at 4×, `HOLD`)
- `v3-vs-sdv-gaussian-copula.json` — `DOPE v3` vs. `sdv-gaussian-copula` (`+47.8 [+33.7, +62.6]` pts at 4×, `HOLD`)
- `v3-vs-sdv-tvae.json` — `DOPE v3` vs. `sdv-tvae` (`+91.9 [+71.9, +112.3]` pts at 4×, `PROMOTE`)
- `v3-vs-sdv-ctgan.json` — `DOPE v3` vs. `sdv-ctgan` (`+157.1 [+141.8, +173.6]` pts at 4×, `HOLD`)
