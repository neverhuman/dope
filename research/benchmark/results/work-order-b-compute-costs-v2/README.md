# Completed research compute and artifact costs

This is a **partial, not final** cost report. It preserves the published Work Order B cost core and adds completed ARF sampling/measurement, Forest's 20-job prefix and two negative job22 attempts. TabSyn is a frozen 25-fit checkpoint. TabDDPM is a closed resource boundary whose scientific operation timing is unavailable in the retained local capture.

## New completed scopes

| Method and phase | Operations | Recorded seconds |
|---|---:|---:|
| ARF native fit wrapper, existing charge | 800 | 14412.990776 |
| ARF sampling | 199 | 9901.332473 |
| ARF matched measurement | 199 | 16043.791480 |
| Forest prefix fit/native/sample | 20 each | 1997.059431 / 443.997977 / 736.546316 |
| Forest prefix measurement core | 20 | 115.817620 |
| Forest job22 infrastructure attempts | 2 | 77.847373 / 600.327437 |
| TabSyn fit prefix, whole operation | 25 | 1321.381350 |
| TabDDPM closed boundary | 20 | unavailable |

Seconds are recorded operation walls, including wrapper overhead, except Forest measurement's explicitly labeled worker core. Scheduler walls include waits and are separate. ARF's 800 fits are charged once. Forest's historical 12-job confirmation and newer prefix have unresolved overlap; no combined total is asserted. Job22 retains actual exit -15, an unproved initializer phase and typed infrastructure/deadline status; it is not a method loss or win.

The DOPE research fit report covers 400 cells, with 354 new operations and 46 immutable prior reuses. Refinement covers 200 cells, with 176 new operations and 24 reuses. Prior architecture research remains separate. The density combined run aliases phase rows. CTGAN/TVAE retain scheduling classifications, 96 historical reused trials and two disjoint historical infrastructure failures. Cuts are not method failures. Operator publication/audit/review work is excluded from scientific totals.

TabSyn's checkpoint at 2026-10-06T14:09:48.131601+00:00 records 24 new fits plus one retained first fit, zero failures, and no native or sampling results. It uses the recorded VAE200/diffusion1000 budget, not the author's full epoch defaults. Three completed fit/exit pairs are directly mirrored; the 24-new-fit aggregate is independently pinned by the checkpoint. Whole-operation timing differs from entry operation timing and inner training time.

Hardware models, historical co-tenancy, attributed energy, total operation time and GPU-hours remain null. Unknown does not imply exclusive use or zero cost. Historical device-used VRAM is a whole-device observation. TabSyn's 326508544B Torch allocation maximum covers three mirrored completed fits. Forest's 20 model/adapter/projection charges total 4881024355B; TabSyn's checkpoint inventory is 2120422296B including exported checkpoints. These are artifact inventories, not current disk use. Runtime/source/worker inputs and sample storage are separate. TD14's metadata holder time is not fit cost; its coordinator numeric exit remains null.

## Reproduce from committed JSON

The six-file publication contains no raw cost constructor. `render.py` uses the existing publisher as the single serializer owner. It checks that exact source path and hash before a static import, rejects cached module identities, and redirects bytecode reads to a required absent cache with bytecode writes disabled. It then checks the report hash before decoding metadata, rejects duplicate row aliases and unsupported null/status claims, and regenerates the CSV/schema byte for byte. Import settings are restored afterward. It never opens scientific rows, models or samples, or initializes a learner.

Committed directory: `research/benchmark/results/work-order-b-compute-costs-v2/`.

```sh
python3 -I -S -B research/benchmark/results/work-order-b-compute-costs-v2/render.py
```

Report SHA256: `9ffb69a46f509fc9c2a0da8efdf81bc6ccf1bbfe97c8491a5439f910cb1f42a4`.
Renderer SHA256: `e3b1acb7021c317273035e4762cf67451596de19b520ac47be9ff196cc8c47fa`.
Existing publisher SHA256: `d90e275318b8eb766fb340334103060de8aeb3c81f65728e28314b2da4f34e01`.

`source-proof.json` retains exact frozen source receipts and anchored indexes. The 21KiB raw preparer remains private, with original input hashing, source/result pins and observed operator provenance in the immutable custody receipt:

`/home/ubuntu/dope/.agent/worktrees/integration/target/work-order-b-compute-cost-report-item6-v1/private-receipt-custody-v1/receipt-custody.json`.

Custody receipt SHA256: `3213cf6f1252f523f3ba47f59040deec4e9d92263c9481dfea7529c4fbe4d9a2`.

The successful metadata preparation and renderer exited 0 at 0.79s/94744KiB and 0.06s/19456KiB respectively. Thirteen focused metadata controls cover exact replay, repeat bytes, identical/conflicting row aliases, scheduling classification, null unknowns, Boolean/nonfinite cost rejection, density alias charging and digest checks. Their private receipt is `renderer-controls.json`, SHA256 `fe662f0ffc4751b0ec247a843947c1844aafc06bc026ad9661e8fb6f79f39384`. None is a new scientific operation. Historical native scalar fields were decoded within metadata documents but were not used for selection or this report. No official tests were opened.
