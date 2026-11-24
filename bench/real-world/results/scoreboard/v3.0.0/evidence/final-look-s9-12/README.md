# v3.0.0 final look (DEV data seeds 9-12, Addendum F)

The single final look that Addendum F of [`../selection/V3_SELECTION_PREREG.md`](../selection/V3_SELECTION_PREREG.md)
froze before any of these cells was scored. The files are copied unchanged from
xbabe2 `~/dope-data/kpi-runs/` and are listed with their SHA-256 in [`SHA256SUMS`](SHA256SUMS).

- **DEV-SELECT only.** Fresh row splits of the 22 scheduled DEV tables; no CONFIRM, HOLDOUT or
  SEALED panel. Schedule `beaa2a0fa3e2…`, contract `dope-kpi/v2` `200ea0ed1ba2…`, scored by
  `dope_kpi` 3.0.0 at `bc92df0b`.
- **Holm family, m = 2:** the candidate `rc-auto-v8` (build `bc92df0b`, `v3c-cand6b`) against
  `dope-2.0.0-auto` (`compare-a.json`) and against `served-auto-v8`, the released engine
  `b8a0037c` (`compare-b.json`). `family.json` applies Holm over both.
- **Ship rule:** the candidate ships only if it PROMOTEs against `dope-2.0.0-auto` and its point
  delta against `served-auto-v8` is at least zero. It PROMOTEs against v2.0.0, but its delta
  against the released engine is below zero, so the released engine stays `b8a0037c`.
- **Disclosure only, no decision role:** the per-family decompositions with exact family
  sign-flip p values (`family-disclosure-*.json`), the contract's sign-flip null (`null-a.json`)
  and, outside the Holm family, the released engine against v2.0.0 (`served-vs-v2-descriptive.json`).
  The descriptive receipt was scored without the families file, so its panel lacks
  `families_sha256` and reports 8 unmatched DEV tables; its two arms equal those of `primary.json`
  field for field.

| File | Content |
| --- | --- |
| `primary.json` | `dope_kpi primary`: the candidate, the released engine, v2.0.0 and the five controls |
| `compare-a.json` | `dope_kpi compare`: candidate vs `dope-2.0.0-auto` |
| `compare-b.json` | `dope_kpi compare`: candidate vs `served-auto-v8` |
| `family.json` | the two comparisons with Holm over the family (m = 2) |
| `null-a.json` | `dope_kpi null` for compare-a: the contract's sign-flip calibration |
| `arms.json` | the three external arms: binaries (SHA-256), fit and sample commands |
| `family-disclosure-rc-vs-v2.json` | per-family decomposition of compare-a, exact sign-flip p |
| `family-disclosure-rc-vs-served.json` | per-family decomposition of compare-b, exact sign-flip p |
| `served-vs-v2-descriptive.json` | descriptive: the released engine vs `dope-2.0.0-auto` on this look |

The v3 white paper renders its final-look numbers from these files and refuses to build if any of
them disagrees with `SHA256SUMS`, the frozen schedule, contract or `dope_kpi` build, or the
pre-registration (`docs/whitepaper-v3/scripts/v3_paper_final_look.py`).
