# v3.0.0 selection evidence

The receipts behind the v3.0.0 selection, copied unchanged from the scoring hosts. With them,
the claims in [`../README.md`](../README.md) can be checked from the repository alone. Every
number in them is a `dope_kpi` output (`dope.kpi-receipt/v2`).

- **DEV-SELECT only.** No CONFIRM, HOLDOUT or SEALED panel was scored for the v3.0.0 selection.
- **Pre-registered.** The order and rules of the selection are in
  [`selection/V3_SELECTION_PREREG.md`](selection/V3_SELECTION_PREREG.md). Each addendum and
  deviation was hashed into [`selection/PREREG_LOG.txt`](selection/PREREG_LOG.txt) before the
  step it governs was scored.
- **Neural screens are information only.** The `SingleEncV8` checkpoint screens under
  `selection/dev-lite/` were not selection steps. None of them beat the analytic `DSE8` packet,
  and v3.0.0 ships no V8 weights as the default.
- **Final look.** Addendum F's single final look (DEV data seeds 9-12, Holm m = 2) and its
  disclosures are in [`final-look-s9-12/`](final-look-s9-12/README.md), with their own
  `SHA256SUMS`. The candidate PROMOTEs against v2.0.0 but is below the released engine
  `b8a0037c`, so `b8a0037c` ships; the wide-DEV eligibility receipts of the builds tried are
  the `selection/wide-dev/cmp-v3c-*` and `family-disclosure-cand6b-vs-v2.json` rows below.
- **Second look at the third-party generators.** Addendum H part (a) repeats the paper's
  comparison with the four third-party generators on the final look's DEV splits (data seeds 9-12),
  descriptively, with its own Holm family. Its receipts, logged text and `SHA256SUMS` are in
  [`addendum-h/`](addendum-h/README.md); the white paper reports it as the second block of Table 3.

| File | SHA-256 | Origin | Content |
| --- | --- | --- | --- |
| `dev-tables.txt` | `6fe722304bdef26b…` | the DEV-SELECT table list (dataset hashes only), for `dope_kpi --panel dev --dev-tables` |  |
| `selection/PREREG_LOG.txt` | `b719db03bf92c706…` | the pre-registration and its append-only hash log |  |
| `selection/V3_SELECTION_PREREG.md` | `f711447b500cba4a…` | the pre-registration and its append-only hash log |  |
| `selection/dev-lite/primary-dope-1.3.0-small.json` | `30f0be76386cacd4…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-1.3.0-small` TRG -0.2386 [-0.443, -0.073] |
| `selection/dev-lite/primary-dope-2.0.0-auto.json` | `30e5ecd8ab032276…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-2.0.0-auto` TRG -0.2246 [-0.422, -0.087] |
| `selection/dev-lite/primary-dope-2.0.0-small.json` | `e288a5f05d2910ac…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-2.0.0-small` TRG -0.2386 [-0.443, -0.073] |
| `selection/dev-lite/primary-dope-v3-auto-v8.json` | `a548961e8b6f7fe7…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-v3-auto-v8` TRG -0.1851 [-0.305, -0.080] |
| `selection/dev-lite/primary-dope-v3-auto.json` | `4a02f28053907531…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-v3-auto` TRG -0.2246 [-0.422, -0.087] |
| `selection/dev-lite/primary-dope-v3-small.json` | `27322b188bd6b758…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-v3-small` TRG -0.2386 [-0.443, -0.073] |
| `selection/dev-lite/primary-dope-v3-v8-analytic.json` | `9a63154ccc60d46d…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-v3-v8-analytic` TRG -0.6406 [-1.006, -0.365] |
| `selection/dev-lite/primary-dope-v3b-auto-v8.json` | `fda33820558ef9b9…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dope-v3b-auto-v8` TRG -0.1187 [-0.188, -0.053] |
| `selection/dev-lite/primary-dse8-p0-nometa.json` | `322d252fcc0f7e32…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dse8-p0-nometa` TRG -0.5569 [-0.929, -0.252] |
| `selection/dev-lite/primary-dse8-p0-v3b.json` | `7e01e04938f07974…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dse8-p0-v3b` TRG -0.2776 [-0.465, -0.126] |
| `selection/dev-lite/primary-dse8-p0.json` | `d9b63e7d6045c508…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `dse8-p0` TRG -0.5555 [-0.927, -0.249] |
| `selection/dev-lite/primary-pr390-v7-step0.json` | `32f41cdee54f3917…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `pr390-v7-step0` TRG -0.4108 [-0.581, -0.248] |
| `selection/dev-lite/primary-s7w-step0.json` | `97d0581807aeff45…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `s7w-step0` TRG -0.2649 [-0.445, -0.120] |
| `selection/dev-lite/primary-v2.0.0-v7.json` | `041715ac89fd3f73…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v2.0.0-v7` TRG -0.3825 [-0.562, -0.224] |
| `selection/dev-lite/primary-v8-r1-k002000.json` | `a7051c4c7ba46960…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r1-k002000` TRG -0.6035 [-0.963, -0.294] |
| `selection/dev-lite/primary-v8-r1-k004000.json` | `174f9d896f6ba848…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r1-k004000` TRG -0.5478 [-0.910, -0.245] |
| `selection/dev-lite/primary-v8-r1-k006000.json` | `4b83425c571a180f…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r1-k006000` TRG -0.5398 [-0.900, -0.238] |
| `selection/dev-lite/primary-v8-r1-k008000.json` | `87692e7cbb6ffa45…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r1-k008000` TRG -0.5445 [-0.902, -0.245] |
| `selection/dev-lite/primary-v8-r1-k010000.json` | `d0b641e8eaa80ed4…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r1-k010000` TRG -0.5288 [-0.892, -0.226] |
| `selection/dev-lite/primary-v8-r1-k012000.json` | `199f44081d17caa7…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r1-k012000` TRG -0.5457 [-0.906, -0.241] |
| `selection/dev-lite/primary-v8-r2-k001000.json` | `27074a874589143f…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r2-k001000` TRG -0.2537 [-0.428, -0.117] |
| `selection/dev-lite/primary-v8-r2-k002000.json` | `00d274a54c6f1767…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r2-k002000` TRG -0.2853 [-0.446, -0.155] |
| `selection/dev-lite/primary-v8-r2-k003000.json` | `0ec3d846e7a0d3bb…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r2-k003000` TRG -0.2743 [-0.436, -0.148] |
| `selection/dev-lite/primary-v8-r3-surg-r2d128-k001000.json` | `b829438b10f0cd6b…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r3-surg-r2d128-k001000` TRG -0.4966 [-0.727, -0.286] |
| `selection/dev-lite/primary-v8-r3-surg-r2d128-k002000.json` | `4fcf02b6ab346391…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-r3-surg-r2d128-k002000` TRG -0.3572 [-0.523, -0.208] |
| `selection/dev-lite/primary-v8-s7w-k001000.json` | `8e017eee03bcf3d8…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-lite/`: 27-cell DEV-lite (data seed 0; 12 cells informative at 4×), schedule sha256 `9207d1575415…` | `v8-s7w-k001000` TRG -0.2596 [-0.440, -0.118] |
| `selection/wide-dev/cmp-v3c-cand6-vs-dope-2.0.0-auto.json` | `5dfca6bc4bd3a7d0…` | xbabe2 `~/dope-data/kpi-runs/v3c-route-compose/`: Addendum F eligibility, wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `v3c-cand6` vs `dope-2.0.0-auto` +0.0444 [+0.002, +0.119] HOLD |
| `selection/wide-dev/cmp-v3c-cand6-vs-dope-v3b-auto-v8.json` | `f8702b510839e0ae…` | xbabe2 `~/dope-data/kpi-runs/v3c-route-compose/`: Addendum F eligibility, wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `v3c-cand6` vs `dope-v3b-auto-v8` +0.0076 [-0.043, +0.067] HOLD |
| `selection/wide-dev/cmp-v3c-cand6b-vs-dope-2.0.0-auto.json` | `82429777eb9b1487…` | xbabe2 `~/dope-data/kpi-runs/v3c-route-compose/`: Addendum F eligibility, wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `v3c-cand6b` vs `dope-2.0.0-auto` +0.0489 [+0.003, +0.119] PROMOTE |
| `selection/wide-dev/cmp-v3c-cand6b-vs-dope-v3b-auto-v8.json` | `e6675a4e8036d131…` | xbabe2 `~/dope-data/kpi-runs/v3c-route-compose/`: Addendum F eligibility, wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `v3c-cand6b` vs `dope-v3b-auto-v8` +0.0121 [-0.037, +0.072] HOLD |
| `selection/wide-dev/compare-auto-v8-vs-2.0.0-auto.json` | `24a09277d37d7533…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-wide/`: 108-cell wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `dope-v3-auto-v8` − `dope-2.0.0-auto` -0.0267 [-0.0679, +0.0075], Holm p 1.0000, HOLD |
| `selection/wide-dev/compare-v3b-guards.json` | `782376d394cb069e…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-wide/`: 108-cell wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `dope-v3b-auto-v8` − `dope-2.0.0-auto` +0.0368 [+0.0084, +0.0623], Holm p 0.0047, HOLD |
| `selection/wide-dev/compare-v3b-vs-2.0.0-auto.json` | `72a5347de5ecd8b5…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-wide/`: 108-cell wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `dope-v3b-auto-v8` − `dope-2.0.0-auto` +0.0368 [+0.0084, +0.0623], Holm p 0.0094, HOLD |
| `selection/wide-dev/family-disclosure-cand6b-vs-v2.json` | `f8f2e145d41f91a9…` | xbabe2 `~/dope-data/kpi-runs/v3-disclose-dev-cand6b-vs-v2/family-disclosure.json` | disclosure only: `v3c-cand6b` vs `dope-2.0.0-auto` gain from 3 of 11 families, exact family sign-flip p 0.125 |
| `selection/wide-dev/primary-dope-2.0.0-auto.json` | `84c814fc6bff625a…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-wide/`: 108-cell wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `dope-2.0.0-auto` TRG -0.2073 [-0.312, -0.117] |
| `selection/wide-dev/primary-dope-v3b-auto-v8.json` | `8375116c7553687c…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-wide/`: 108-cell wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `dope-v3b-auto-v8` TRG -0.1705 [-0.265, -0.084] |
| `selection/wide-dev/primary-dse8-p0-v3b.json` | `c63b2fd234533b86…` | xbabe6 `~/dope-data/kpi-runs/v3-dev-wide/`: 108-cell wide DEV, data seeds 1-4, schedule sha256 `97439e8a978b…` | `dse8-p0-v3b` TRG -0.1897 [-0.362, -0.047] |
