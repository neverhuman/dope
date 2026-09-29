# Prepilot cost and feasibility record

Observed 2026-09-29. This record precedes the preregistered 24-hour,
seven-method pilot. It measures worker and validation costs; it makes no MFS-v2,
privacy, or superiority claim. The public test partitions remain sealed.

## Inputs and execution

The three source archives, transformations, rights evidence, 60/20/20 grouped
split hashes, and projected file hashes are frozen in `pilot-datasets.lock.json`.
Workers received only the train and validation projections. `xbabe1` ran Adult
on an RTX 4090, `xbabe2` ran California on an RTX 3090, and `xbabe3` ran News on
an RTX 3090. `host-inventory-20260929.json` under the external data root records
the observed drivers, free GPU memory, active workloads, memory, and disk. The
prepilot worker source tar SHA-256 was
`d55569df18932f69185cb906ab3bc75c02b64b3a4c4d76669b6f607ef60cbdbc`;
the DOPE probe binary SHA-256 was
`1e4425fb04aed4215018cc52a2c0179e05217321c2b9691e230cce786ffdd2b3`.
No AWS credentials were copied to workers. The coordinator read a selected
JopeDime catalog entry and verified its catalog, manifest, and blob hashes,
separate from the public pilot.

## Measured DOPE default cells

The probe used fit seeds 11 and 23, sample seeds 101 and 211, and `n`, `2n`,
`4n`, and `8n`. Sample time includes an independent repeat used to check
determinism. All six fits and all 48 sample cells completed; repeated samples
matched their recorded SHA-256. Timing is wall-clock seconds.

| Dataset / host | Train rows | Fit seconds, seeds 11 / 23 | Sampling seconds for eight cells, seeds 11 / 23 | Model-only bytes | Projection bytes | Charged bytes under revised runner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult / xbabe1 | 29,306 | 27.28 / 27.11 | 31.46 / 31.74 | 737 | 4,534 | 5,271 |
| California / xbabe2 | 12,384 | 4.49 / 4.48 | 1.89 / 1.84 | 1,162 | 1,591 | 2,753 |
| News / xbabe3 | 23,787 | 14.26 / 14.52 | 17.51 / 17.44 | 2,731 | 9,770 | 12,501 |

The first probe recorded model-only bytes. After the byte-accounting review,
the worker was changed to charge the dataset-specific projection map for every
method. A fresh California fit verified 2,753 charged bytes and archived its
compile diagnostics outside the sampling artifact; all eight California
sample cells passed the artifact-only deterministic repeat check under the
current worker. Adult and News charged
figures above are the exact sum of the existing model and immutable map file
sizes; they have not been refit with the revised runner. On these observations,
News does not meet the L3 byte cap of 10,240 B if the map is required, and it
remains below the L2 cap of 32,768 B. That is a byte observation, not a full
generator-gate outcome.

Validation-only descriptive checks took 11.49 seconds on Adult, 4.55 on
California, and 9.25 on News. CatBoost null-normalized retention estimates
were 0.815, 0.724, and 0.578, respectively. These single-split values are
neither production PTF-v1 nor the paired lower-bound gate. The metric JSON
explicitly sets `gate_profile_complete: false` and `mfs_v2: null`.

## First GaussianCopula cost cell

The pinned Copulas 0.14.1 GaussianMultivariate author-default distribution
fit California seed 11 in 361.13 seconds. Its serialized model was 1,617,839 B;
the additional 1,591 B projection map would make a charged artifact of
1,619,430 B with the revised runner. Either count exceeds both compact byte
caps. The first `n = 12,384` sample cell was still running beyond eight minutes
on `xbabe2`; its result and operator cap are recorded in the external failure
ledger. This cell ran with the pre-timeout worker source, so it is cost evidence
and does not satisfy the current method lock.

## Admission decision

The seven-method pilot is not complete. TabPC, TabSyn, TabDiff, and AIM have no
locked executable adapter or isolated dependency image yet. The identified
TabKDE repository had no license file at the audited commit, so its terms
require resolution before runnable status. GaussianCopula needs a bounded
sampling strategy and a current-lock rerun. A container digest and permitted
GPU allocation for final dispatch also remain to be pinned. The method lock
therefore stays incomplete; no `budget.lock.json` or final method–dataset
matrix is frozen, and no public test partition or full campaign is opened. The
final-analysis CLI checks the admission locks before reading result reports.

Outside the fixed seven-method pilot, the study-owned Chow–Liu compact
adapter is source/config locked for common-numeric runs. Its California seed
11 default fit took 0.063 seconds. The numeric tree file was 10,192 B; adding
the required 1,591 B projection map yields 11,783 B, above L3 and below L2.
The 12,384-row sample and deterministic repeat took 0.148 seconds together.
Validation-only CatBoost retention was 0.645 and C2ST AUC was 0.986, with no
exact or near copies observed. These diagnostics are not a gate report, and
this extra cost cell does not complete the seven-method pilot or method lock.

These observations cannot justify a 14-day estimate for the protected compact
panel: training and audit costs for most compact comparators remain unknown.
The next admission decision requires the full seven-method pilot or an
explicit quantified feasibility stop after it, as set out in the protocol.
