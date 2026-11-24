# Addendum H part (a): second look of DOPE v3 against the four third-party generators

These receipts repeat the paper's comparison of DOPE v3 with SDV GaussianCopula, CTGAN and TVAE and
arfpy ARF on a second set of DEV row splits: data seeds 9–12 of the same 22 DEV tables, the splits of
the final selection look in [`../final-look-s9-12/`](../final-look-s9-12/README.md). They are copied
unchanged from xbabe2 `~/dope-data/kpi-runs/v3-h-s9-12/`. Every number in them is a `dope_kpi` output.

- **Pre-registered and descriptive.** [`ADDENDUM_H.md`](ADDENDUM_H.md) is the logged text, and
  [`PREREG_LOG-H.txt`](PREREG_LOG-H.txt) is its single `PREREG_LOG.txt` line (logged
  2026-11-24T07:15:07Z, text SHA-256 `c7e446a8…`), written before anything under H was emitted or
  scored. H has no role in model selection or release qualification, and it ships nothing.
- **Not new splits for DOPE.** Addendum F had already scored the released engine (`served-auto-v8`)
  and the five controls on these splits, and those numbers were known when H was written. Their six
  rows in `primary-h.json` equal the committed `../final-look-s9-12/primary.json` field for field.
  Only the four third-party generators were new to these splits.
- **One `dope_kpi` build for every stage.** `ea855a89…` (version 3.0.0, commit `bc92df0b…`), the
  build that scored `served-auto-v8` on these splits. The report look used another build, so the two
  looks are reduced separately and never pooled.
- **Same recipes as the report look.** The arm specs, wrapper, arms file, `make_arms.py` and venv
  lock equal the report run's pins, and the emit, score and guards argv differ from the committed
  report `../../MANIFEST.json` only as H permits (binary path, run directory and schedule path,
  `--timeout-s 14400`, and the report's per-method thread counts). `MANIFEST-h.json` records them.
  It holds paths, argv, digests and versions only, the same kind of fields as the report manifest.
- **DEV-SELECT only.** No CONFIRM, HOLDOUT, SEALED or FMGAP data. Score directories, packets,
  synthetic rows and worker logs stay in research custody.
- **Verified.** Codex re-reduced the family and re-checked every receipt byte for byte
  (2026-11-24T08:03Z). The paper's renderer (`docs/whitepaper-v3/scripts/render_v3_paper.py
  --check`) and a `dope_kpi` cargo test re-check the bindings below on every build.

## Reading (Addendum H's own rule)

The transfer advantage over a method **replicates** on this look iff the 4× lower bound of
`DOPE v3 − method` is above zero and the Holm (m = 4) p is below 0.05.

| Method | DOPE v3 − method, 4× [95 % CI] (points) | Holm p (m = 4) | Replicates | Rule with Holm p |
| --- | --- | --- | --- | --- |
| SDV GaussianCopula | +46.9 [+36.6, +58.2] | < 0.001 | yes | HOLD (tasks, privacy) |
| SDV TVAE | +89.8 [+56.4, +121.3] | < 0.001 | yes | PROMOTE |
| SDV CTGAN | +139.3 [+121.4, +158.5] | < 0.001 | yes | PROMOTE |
| arfpy ARF | +17.8 [−6.5, +43.6] | 0.081 | no | HOLD (interval, tasks, mlp, realism) |

At the headline 4× sampling, DOPE is not shown to transfer better than ARF on either look. At matched
size (1×) this look's ARF difference, +33.7 [+14.4, +56.6], has an interval above zero, unlike the report
look's; H's reading and the promotion rule use 4× only. The paper's Table 3 shows both looks as separate blocks (`dope_kpi render-paper-tables --second-primary … --second-compare …`).

## Files (`SHA256SUMS`)

| File | Content |
| --- | --- |
| `primary-h.json` | `dope_kpi primary` over `served-auto-v8`, the five controls and the four methods (108 scheduled cells; 41 informative in 10 families for DOPE) |
| `v3-vs-<method>-h.json` | `dope_kpi compare --candidate served-auto-v8 --baseline <method>`, one per method |
| `family-h.json` | the Holm (m = 4) family over the four pair receipts, bound to their SHA-256s, with singleton and Holm outcomes |
| `MANIFEST-h.json` | provenance: argv of every stage, binary, input and pin digests, package versions |
| `ADDENDUM_H.md`, `PREREG_LOG-H.txt` | the logged pre-registration text and its hash-log line |

## Additional split set

[Part (b)](part-b/) contains the independently reproduced baseline results on DEV data seeds 13–16.
