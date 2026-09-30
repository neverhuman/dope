# Four-hour generator benchmark pilot: cost and admission decision

Observed 2026-09-30 on `xbabe1`, `xbabe2`, and `xbabe3`. The pilot started at
13:13 UTC and closed at 15:47 UTC, within its 17:13 UTC hard deadline. The public test
partition has not been opened. Each worker directory contains only train,
validation, projection, and its manifest.

## Decision

**Do not open the full campaign or public test set.** The fixed pilot cannot
establish a 14-day budget for the protected compact panel: only three of ten
compact baseline sources are runnable in the current lock, and one of these
has not produced a completed pilot sample. The evaluator and attack contract,
complete public-core manifest, tuning budgets, author-faithful adapters, and
final runtime images remain incomplete. `budget.lock.json` and the final
method–dataset matrix remain absent so `admission.py` fails closed. No MFS-v2
score or DOPE superiority claim is available. An actual admission check exited
2 and named six blockers: missing budget, datasets, evaluator, and final
matrix locks, plus an unfrozen method lock with source/config gaps. Its JSON
receipt is archived on benchmark scratch.

The paper's primary comparison remains DOPE against each executable compact
baseline at L3 on paired public-core datasets. BeyondArena is a separate
context panel. The real-data utility endpoint is null-normalized TSTR/TRTR
retention for CatBoost, linear/logistic regression, and MLP on untouched real
test rows at `n` and `4n`. It directly measures whether generator output
supports prediction on real data; the test calculation waits for final locks.

## Frozen pilot and observed execution

The frozen matrix has 42 fit cells: Adult, California, News × DOPE,
GaussianCopula, TabPC, TabKDE, TabSyn, TabDiff, AIM × fit seeds 11 and 23.
Two sample seeds and four sizes (`n,2n,4n,8n`) would give 336 sample cells if
every fit were runnable. The queue admitted 12 fit cells under the frozen
source lock; 30 received explicit blocked receipts. The queue package and
matrix are preserved on benchmark scratch. Infrastructure retries never
overwrite failed first attempts. Training `n` was 29,306 Adult rows, 12,384
California rows, and 23,787 News rows; source identity, rights evidence,
transformation, and split hashes are in `pilot-datasets.lock.json`.

| Fixed matrix method | Fit cells | Status and measurable cost |
| --- | ---: | --- |
| DOPE | 6 | All fits and 48 deterministic sample cells completed. Fit 4.69–31.62 s; eight samples per fit 6.23–72.29 s. Charged artifacts: Adult 5,271 B; California 2,753 B; News 12,501 B. News exceeds L3, though below L2. |
| GaussianCopula | 6 | Both Adult cells hit 2,700 s outer caps without a fit artifact. California's two fits took 54.3–54.5 s and produced 1,619,430 B artifacts; eight attempted sample cells timed out at approximately 600 s, and both jobs hit the outer cap. News fits took 2,327–2,329 s and produced 11,720,459 B artifacts; neither completed a sample receipt before its 2,700 s outer cap. |
| TabPC, TabKDE, TabSyn, TabDiff, AIM | 30 | Blocked in the frozen queue by adapter, dependency, release, or license checks; each has a receipt. Later compatibility probes remain separate. |

The first queue pass also had four California/News GaussianCopula infrastructure
failures because a subprocess imported the live adapter instead of the frozen
package. Attempt-2 receipts preserve the original failure and use the frozen
working directory. The first queue summary's elapsed-time sum includes host
slot waiting and is **not** a compute-cost total; fit and sample receipt times
are used here. The repair queue has its own receipt and log directories. The
reconciled main matrix is 6 successful DOPE jobs, 6 GaussianCopula timeouts,
and 30 source-blocked jobs. Its failure ledger retains 40 blocked, failed,
and timed-out attempts, including the four original infrastructure failures.

## Separate compatibility and compact reference evidence

| Method / dataset | Fit, seconds | Eight sample cells per fit | Charged artifact bytes | L3 byte screen |
| --- | ---: | ---: | ---: | --- |
| Independent marginals / Adult | 0.52 | 184–185 s | 19,100 | Over |
| Independent marginals / California | 0.06–0.10 | 6.4–6.5 s | 4,296 | Within |
| Independent marginals / News | 0.46–0.47 | 79–81 s | 25,937 | Over |
| Chow–Liu / Adult | 1.70–1.77 | 47–48 s | 27,896 | Over |
| Chow–Liu / California | 0.06 | 6.7 s | 11,783 | Over |
| Chow–Liu / News | 0.76 | 61–62 s | 71,671 | Over |

This separate 12-fit matrix completed all 96 artifact-only deterministic
sample cells on xbabe1. A fresh host rerun of California Chow–Liu seed 11
returned the same eight sample receipts, and all fit/sample receipt SHA-256
values remained identical. The methods are study-owned reference
implementations, labeled as such in the source lock. A byte screen is only one
gate; the L3 cap is 10,240 B and L2 cap is 32,768 B. No listed cell is
declared a generator pass. An independent inventory check reconciled file
hashes and raw byte totals for all 23 completed runner artifacts, including
the four California and News GaussianCopula fits.

The TabPC News author-configuration compatibility fit took 545.1 s on the
available RTX 4090. Its 555,381,583-byte artifact includes a saved
8,087,986-byte training tensor that upstream reloads for sampling; the model
has 136,729,600 reported parameters. The fit
used installed Torch 2.7.0, whereas the author source requests 2.9.1, and the
receipt's peak VRAM measurement reflects sampling only. This probe cannot be
scored or released as an L3 comparison. A full `n` sample was not measured;
the 100-row sample took 6.62 s.

A pilot-only AIM adapter with a fixed public eight-bin domain completed a
California fit in 67.16 s, produced a 14,554-byte charged artifact, and
completed eight deterministic sample cells in 7.78 s total. Adult and News
fits each reached their 1,200 s child-process hard cap, with peak resident
sets of 14,858,676 and 18,902,724 KiB respectively. The first Adult attempt
failed because xbabe3 lacked `cycler`; that failure and the dependency-fixed
timeout are both retained. Train-fitted common projection bounds prevent an
end-to-end formal DP claim for this AIM path.

## Validation-only observations

Single-seed, `n`-size validation metrics are saved as raw JSON. Every vector
has `gate_profile_complete: false` and `mfs_v2: null`. The table below is
descriptive; neither a production PTF-v1 estimate nor a final generator gate
is computed. Large negative News retention reflects the null-normalized
denominator and must be interpreted with the production low-signal rule in
the final evaluator.

| Dataset | Method | CatBoost retention | Linear retention | MLP retention | C2ST AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Adult | DOPE | 0.815 | 0.963 | 0.923 | 0.726 |
| Adult | Independent marginals | -0.064 | -0.009 | -0.082 | 1.000 |
| Adult | Chow–Liu | 0.700 | 0.721 | 0.661 | 0.925 |
| California | DOPE | 0.724 | 0.575 | 0.661 | 0.800 |
| California | Independent marginals | -0.013 | 0.027 | 0.032 | 0.959 |
| California | Chow–Liu | 0.645 | 0.819 | 0.743 | 0.986 |
| California | AIM, pilot-only | 0.698 | 0.917 | 0.748 | 0.994 |
| News | DOPE | 0.578 | 0.501 | undefined | 0.972 |
| News | Independent marginals | -101.146 | -108.692 | undefined | 0.993 |
| News | Chow–Liu | -490.417 | -517.865 | undefined | 0.998 |

The three DOPE CatBoost validation estimates are below the registered 0.99
retention gate threshold; the full paired lower-bound and profile-folding
calculation has not run. This is a reason to test the claim carefully, not a
paper conclusion. Copy and C2ST checks are descriptive and do not substitute
for shadow-model attacks, attribute inference, rare strata, or real-vs-real
privacy calibration.

## Fourteen-day and storage budget

The public-core name union has at most 20 unique datasets before task
exclusions. If all 20 are eligible, the protected default common-numeric
panel of DOPE plus ten compact baselines requires 1,100 fits and 13,200
sampling cells across five fit seeds, three sampling seeds, and four sizes.
Across 14 calendar days this requires 3.274 completed fits and 39.286
completed sample cells per hour **before** tuning, validation, privacy
attacks, the author-faithful track, and all secondary panels. The remaining
seven compact baselines have no measured runnable fit/sample distribution;
the GaussianCopula and AIM timeout evidence rules out a simple extrapolation
from DOPE speed. A credible equal-budget tuning cap cannot yet be frozen.

The completed six DOPE and twelve compact-reference fit directories already
consume about 10.5 GB of sample/results space, before the TabPC, AIM, and
dependency snapshots. A naive per-fit extrapolation to 1,100 fits would be
roughly 640 GB, not a forecast because datasets and methods vary, but enough
to show that full CSV retention requires streaming evaluation and an explicit
archival policy under the 100 GB xbabe2 benchmark-scratch ceiling. The
reconciled pilot snapshot is 13,178,599,395 bytes, below that ceiling, with about 265 GB
free on the filesystem at closure. All bulk inputs, fits, dependencies,
samples, and results are under `/mnt/fast-scratch/dope-benchmark`; no bulk
data was written into a worktree or `/tmp`.

At pilot admission xbabe1's RTX 4090 was available and was used for the TabPC
probe. The RTX 3090s on xbabe2 and xbabe3 had unrelated owners' jobs; the
pilot did not preempt them. CPU jobs used the three hosts with a 16-core cap
per job, and xbabe1/xbabe3 accessed the coordinator's benchmark scratch via
SSHFS. The host inventory records GPU models, drivers, memory, disk, and
active workloads. Full runtime/container digests still need qualification.
The frozen queue's two simultaneous GaussianCopula workers both selected CPU
affinity 0–15, so their wall-clock time includes contention for the same 16
cores. The reusable launcher has since been corrected to allocate distinct
16-core slots, but the frozen pilot attempts were not rewritten. Their timeout
costs are bounds for this actual schedule, not standalone single-job timings.

| Host | GPU / driver at admission | Pilot use |
| --- | --- | --- |
| xbabe1 | RTX 4090 / 570.153.02 | TabPC GPU probe; DOPE and reference compact CPU jobs |
| xbabe2 | RTX 3090 / 595.99.02 | GaussianCopula and AIM CPU jobs; owner GPU workload left in place |
| xbabe3 | RTX 3090 / 580.126.09 | DOPE and AIM CPU jobs; owner GPU workload left in place |

## Admission work left

The compact lock currently has three runnable baselines of ten. Five require
source/config/dependency and adapter completion (synthpop CART, ARF,
PrivBayes, TVineSynth, and TabPC), TabKDE lacks a license grant
in its pinned source, and the GEM-T author implementation was not located in
the documented availability search. TabPC's author artifact stores training
rows, so a releasable variant needs separate implementation and validation.
TabSyn and TabDiff need train/validation-only staging that prevents their
upstream `test`-named paths from reading the sealed partition. AIM needs a
preprocessing/accounting audit before any formal DP statement. The complete
generator gate evaluator, attack calibration, source/data/container locks,
projection-only utility cost, tuning trial budget, and final eligible
method–dataset matrix remain required. The protocol's 14-day stop rule and
fail-closed admission remain in force.

The machine-readable reconciled receipt report, failure ledger, frozen
matrix, host inventory, validation vectors, compatibility receipts, and
resume check are under `/mnt/fast-scratch/dope-benchmark/pilot-4h`. The
frozen matrix SHA-256 is
`2c07f8e7b0793284fd61bed918fe35a397cdccc769dff9700731249358ba2b6d`;
the frozen queue package's method-lock SHA-256 is
`c82a5433d3726b4bd70ebf87c5baa2e341d18ee9637c8cb850d3f6adb7460b14`.
The live method lock gained source-audit notes and a pilot-only AIM entry
after admission; it does not retroactively change the fixed matrix. The
immutable reconciled report SHA-256 is
`312c2321e75344ebdd6c297d892054cf87cc808d0609bb71eef343941ca2bacc`.

## Verification

The 25 benchmark unit tests, `just fast`, and `just check` passed. The
reconciled 42-cell status counts, 40-entry failure ledger, 23 matching
artifact inventories, ten null-MFS validation vectors, and the unchanged
host-resume hashes were checked against the scratch receipts. The full-run
admission command exited 2 as intended; no public test evaluation ran.
