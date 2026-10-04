# Offline generator benchmark (research only)

[`CONTAINER_VALIDATION_WORKER.md`](CONTAINER_VALIDATION_WORKER.md) describes the
source-only compressed sampling worker, owned member inputs and exact replay
controls. Its 32-batch validation preview remains unadmitted; live rounds and
production scores are unchanged.
`container_python_custody.py` checks the externally pinned interpreter/import
inventories before a future worker startup. It starts no process and leaves full
runtime closure and dispatch admission unverified.
`container_bootstrap_invocation.py` prepares restricted startup arguments and
environment with an unchanged outer-attempt budget. Its immutable proposal starts
no process and proves no filesystem, ELF, transport or capacity admission.
`container_elf_inputs.py` extracts externally pinned ELF declarations from owned
bytes without resolving or loading libraries. Its supported-profile inspection
leaves executable closure and admission unverified.
`container_system_custody.py` binds the declared ELF inventory to frozen runtime
manifests, verifies exact members and aliases, and replays owned-byte declarations.
It proves selected file custody, with loader resolution and dispatch still unadmitted.
`container_loader_custody.py` rechecks an externally pinned proposal of candidate
provider files, alias chains, cache/configuration files, directory entries and
absence records. These snapshots leave actual provider selection, complete loader
search, runtime closure and execution admission unverified.
`container_loader_catalog.py` replays the frozen package-basename/cache candidate
enumeration, including ordered root and recursive provider dependencies. It
retains competing provider identities and establishes no actual loader choice.
`container_loader_configuration.py` hashes selected configuration bytes before
inspecting a bounded declaration profile, binds include file membership and
checks the additional directory/absence snapshots named by those declarations.
`container_search_path_custody.py` reconstructs ordered root/provider declared
path components and binds their literal pathname projections to those snapshots.
It establishes no actual loader origin, search, provider choice or execution.
`container_loader_environment.py` binds the restricted invocation's declared
library directory and preload absence to those snapshots, retaining the original
outer timer and all unadmitted execution and null score fields.
It proves neither actual cache construction nor loader search or execution.
`container_bootstrap_filesystem.py` checks the proposed six-file source tree,
owned request identity and empty cache before any startup. Its staged descriptor
is unadmitted and supplies no final round or execution lease.
`container_native_sampler_custody.py` adds the separate sampler executable root,
binding its owned ELF declarations to the original fit binary and proposed job.
Actual child library/provider resolution and execution remain unverified.
`container_native_library_custody.py` checks the original native directory's
selected files and literal alias snapshots against owned historical records.
Actual provider selection, child environment, runtime and execution admission
remain separate gates; see the worker custody contract.
`container_native_child_environment.py` binds a separate restricted proposal to
the original native directory and selected preload absence record. It verifies
only proposed environment declarations; actual child/runtime/lease/admission
remain unverified.
`container_native_runtime_inventory.py` rechecks the full recorded original
native file set, including paths outside the selected directory. This does not
replay the recorded linker transcript or establish complete runtime closure.
`container_startup.py` binds those original recorded stages and replays them on
one original whole-batch timer. This remains an unadmitted startup preparation.

This package does not enter the Rust production binary or alter its release decision.
`contract.json` pins a generator-only gate profile and the SHA-256 of the frozen
production KPI contract. A future contract change requires a new version.

`PROTOCOL.md` is the pre-registration. `study-design.json` fixes the cohorts;
`pilot-datasets.lock.json` preserves the historical four-hour pilot inputs.
Its Adult partition mixed UCI's official training and test files before splitting;
all Adult quality metrics from that pilot are withdrawn. The replacement
`pilot-24h-datasets.lock.json` pins the official Adult test as evaluator-only,
with 25 overlapping training-row occurrences excluded and receipted.
`methods.lock.json` records audited methods and pending source checks.
The latter is incomplete, so no final evaluation or headline comparison is
authorized by these files. `score.py` requires complete measured gate evidence
and keeps MFS-v2 null otherwise.

The worker receives only training and validation data. The evaluator owns the
test partition. Final evaluation is disabled until the data manifest, method
source/config lock, budget lock, and method–dataset matrix are frozen. Missing
privacy, utility, or attack evidence is a failed gate with a null MFS-v2 score.

`RESULTS_STATUS.md` records the 2026-10-01 rights-cleared S3 preparation,
DOPE GPU refinement, and native-tuned CTGAN/TVAE validation comparison.
Its published JSON, tables, and figures live in
`results/`; the final comparison remains blocked by admission.
The compact native likelihood study freezes its twelve-dataset, two-method
matrix with `freeze_native_round`, runs `tune_density` on training-derived
worker partitions, and publishes validated trial receipts with
`publish_compact_native`.
The GaussianCopula validation wave uses `tune_copula freeze`, followed by
`tune_copula tune DATASET_ID` for each frozen job, and `publish_copula`.
The study-owned density baselines' complete S3 validation matrix is frozen by
`freeze_all_native` and published by `publish_compact_native --expected-jobs 200
--basename compact-native-all-validation` with its scratch round lock.
The separate 100-lineage GaussianCopula matrix is frozen by
`freeze_all_copula` and published by `publish_copula --expected-jobs 100
--basename copula-native-all-validation` with its scratch round lock.
An unavailable method is reported as unavailable; it is never a DOPE win.

`SDV_METHOD_AUDIT.md` records the CTGAN/TVAE author source, license, environment,
native objective, grid, and cross-host sampler checks. `freeze_sdv_round` and
`sdv_round` freeze and execute the 96-fit research matrix. The frozen package
on scratch must be used to resume that matrix: current sources may have moved
on. `publish_sdv_validation --evaluate --publish` verifies and publishes its
native selections and common validation outcomes. `conditional_gpu_round`
supports bounded DOPE conditional GPU refinement with a source snapshot and
binary hash; `publish_conditional_validation` retains original and repaired
results. Both publishers support `--render-existing` for offline regeneration.

The matched five-fit pilot panel uses `publish_native_neural_fivefit` to verify
DOPE q8, CTGAN and TVAE against the same Adult/California training-derived
validation inputs. All five fit seeds and three sample seeds must be accounted
for, including failures. Native selections are retained without new tuning;
shared `n`/`4n` outcomes never select comparator configurations. Generate the
SVG/PDF with `publish_native_neural_fivefit_figure`; `--from-json` regenerates
tables using committed rights-safe results alone. This descriptive stability
panel keeps official tests sealed and all gated release scores null.

[`TABSYN_SOURCE_AUDIT.md`](TABSYN_SOURCE_AUDIT.md) records the pinned author
defaults, native regression evaluator behavior and the remaining representation,
runtime and sampling gaps. Its 96 source/documentation files match the archive
and original Git blobs. This source-only audit keeps TabSyn pending and does
not authorize fits, tuning or final evaluation.

[`GPU_TARGET_REFINEMENT.md`](GPU_TARGET_REFINEMENT.md) documents the separately
bounded GPU target research feature, its finite training profiles, and the
required source, runtime, artifact and resource evidence before fit admission.

The complete bounded GPU target research panel is
[`pilot24-target-gpu-research.md`](results/pilot24-target-gpu-research.md).
It accounts for 24 fits and 192 validation identities across Adult, California
and News, including the eight original raw byte failures and separate lossless
News packing. Adult/California references retain the prior CTGAN/TVAE native
selections and the two matching fit seeds. These exploratory profiles are not
a frozen production family; gated scores stay null. Recreate tables with
`python3 -m research.benchmark.publish_target_research --from-json --result
research/benchmark/results/pilot24-target-gpu-research.json`, then regenerate
SVG, PDF and the publication manifest with
`python3 -m research.benchmark.publish_target_research_figure`.

The complete single-fit TabPC pilot publishes all native trials and fixed
sample/metric jobs with `publish_tabpc_native`, with SVG/PDF generated by
`publish_tabpc_native_figure`. It retains native NLL selection independently
from common utility, all failed tracks, artifact charges and immutable receipt
hashes. `--from-json` regenerates tables from committed JSON alone.

Use `python3 -m unittest discover -s research/benchmark/tests -v` for the
research package checks. Run the repository's `just fast` and `just check`
before committing. Results and raw inputs belong outside worktrees and `/tmp`;
the worktree's `target/` may hold build and test evidence.

Entry points:

```text
python3 -m research.benchmark.inventory_hosts --output HOSTS.json
python3 -m research.benchmark.fetch_jope DATASET_HASH DATA_ROOT
python3 -m research.benchmark.public_sources Adult ADULT.zip ADULT_TRAIN.csv --test-output ADULT_TEST.csv
python3 -m research.benchmark.california_source CAL_HOUSING.tgz CALIFORNIA.csv
python3 -m research.benchmark.manifest SELECTION.json DATA_ROOT/prepared
python3 -m research.benchmark.select_extension REVIEWED_TASK_CANDIDATES.json
python3 -m research.benchmark.runner JOB.json research/benchmark/methods.lock.json RESULTS_ROOT
python3 -m research.benchmark.pilot_metrics WORKER_DIR SAMPLE.csv
python3 -m research.benchmark.pilot_queue --deadline-utc 2026-09-30T17:13:00Z --execute
python3 -m research.benchmark.pilot_queue --deadline-utc 2026-09-30T17:13:00Z --execute --repair-digest-mismatch
python3 -m research.benchmark.pilot_compact_supplement --deadline-utc 2026-09-30T17:13:00Z --execute
python3 -m research.benchmark.pilot_evaluate Adult dope --fit-seed 11 --sample-seed 101 --multiplier 1
python3 -m research.benchmark.pilot_report --output /mnt/fast-scratch/dope-benchmark/pilot-4h/reconciled-report.json
python3 -m research.benchmark.analysis MATRIX.json GATE_REPORTS.jsonl research/benchmark/methods.lock.json
python3 -m research.benchmark.admission REPO_ROOT LOCK_ROOT
python3 -m research.benchmark.full_queue --repo-root REPO_ROOT --lock-root LOCK_ROOT
python3 -m research.benchmark.real_test_utility WORKER_DIR EVALUATOR_DIR SAMPLE_N RECEIPT_N SAMPLE_4N RECEIPT_4N OUTPUT
```

The runner enforces a 16-core affinity and respects the launcher's GPU
visibility setting (defaulting to GPU 0), hashes worker
inputs, verifies repeated artifact sampling, and resumes receipts without
refitting. The coordinator's JopeDime fetch reads only requested catalog,
manifest, and blob objects through the named AWS profile. Credentials stay on
`xbabe2`. `pilot_metrics` consumes validation, never the test partition.
On `xbabe2`, each fit and sample cell reserves space against the 200 GB scratch
ceiling, counting the whole scratch root (the output root's parent by default).
`admission` exits nonzero while any final source, data, budget, method matrix,
or evaluator lock is absent or incomplete; public test evaluation must call it
before reading test data.
All five final locks must carry the same `freeze_set_sha256`, computed by
`admission.frozen_set_digest` from their complete payloads with only that field
removed. Set this binding after every payload is final; any later change
invalidates the set. Digest placeholders and version-only dependency pins do
not satisfy final admission.
Every declared method–dataset pair needs either an applicable schedule or an
explicit exclusion. Applicable schedules include all five fit seeds, defaults
and native-tuned configurations when tuning applies, and all three epsilon
budgets for DP methods. Each cell binds `method_source_sha256`, `adapter_sha256`,
`dependency_or_container_digest`, `projection_sha256`, `train_sha256`,
`validation_sha256` and `evaluator_sha256`
to the corresponding locks. Dataset locks include the projected `train_rows`.
Native selections are checked against immutable attempt, artifact and metric
receipts before admission; admission never opens test rows or fits a model.
Resolved native receipt paths must be JSON files outside evaluator storage,
including their symlink targets. Compact NumPy methods bind a portable source
and shared-library manifest in `numpy-runtime.lock.json`; final admission and
execution verify the same digest, versions and installed files before import.
Historical validation jobs retain their recorded version identities.
The current worker runs only the common-numeric track. Author-faithful input
preparation and external adapters are still pending and fail closed.
The full queue also fails closed until admission passes. It checks each host's
CPU affinity, load, available memory, benchmark scratch filesystem, GPU
processes, and free VRAM before each dispatch. CPU jobs occupy distinct
16-core slots, up to four per host; GPU jobs use one such slot and one visible
GPU. Attempts, logs, and admission inventories are immutable under benchmark
scratch. A failed attempt remains visible and needs `--retry-failed` for another
attempt. Worker environments disable AWS credential discovery. The real-test
utility entry point checks admission before reading the test partition and
requires matching frozen dataset manifests, fit artifacts, and sample receipts
for `n` and `4n`. Final cells require the complete three-seed, four-size
sample matrix. Each fit identity binds that matrix, the selected configuration,
and any DP budget. Tuned configurations require an eight-trial validation
selection receipt on scratch with a matching SHA-256 and a total wall time of
at most 12 hours. Failed fit and sample attempts remain in attempt directories;
only successful attempts occupy canonical receipt paths, so retries retain the
earlier failure evidence.
It is one endpoint of the unfinished shared evaluator, not a complete gate
report.

`PILOT_STATUS.md` records the four-hour dispatch. `PILOT_METHOD_AUDIT.md`
records source and dependency findings. `pilot_queue.py` freezes all 42 fit
cells, writes blocked receipts, and runs the 12 cells supported by the current
source lock. `pilot_evaluate.py` writes validation-only descriptive vectors.
`pilot_report.py` reconciles original and infrastructure-repair receipts while
retaining every failed attempt in its ledger. The frozen queue package and
all bulk receipts live on benchmark scratch. Pilot-only AIM compatibility jobs
are supplemental and do not alter the original 42-cell matrix.
`pilot_compact_supplement.py` runs the two already locked reference compact
methods on the pilot datasets with separate receipts and a separate matrix.
`PILOT_COST_REPORT.md` records measured costs, validation-only vectors, the
historical 14-day capacity bound, and the full-campaign admission decision.
The later protocol amendment treats day 14 as a reporting milestone and sets
the new full-campaign scratch ceiling to 200 GB. Historical pilot usage and
its original 100 GB limit remain as recorded in the pilot cost report.
The final public test remains closed until `admission.py` passes.

`PREPILOT_COST_REPORT.md` records the three-host DOPE probe and the first
GaussianCopula cost cell. It is not the seven-method pilot or a benchmark
comparison. The runner charges the projection map in raw artifact bytes and
archives DOPE compile sidecars separately as fit evidence.
# ForestDiffusion original-package pilot contract

`forestdiffusion-source.lock.json` pins the MIT Python package at upstream
`818ac3b9c8df7b7c763470c9a0691dacca8d8b37`, the complete runtime inventory,
portable interpreter, and the unchanged frozen GPU adapter. Repository-root
experiment code has unresolved license coverage and is not imported. The study
wrapper implements the author's mean ML efficacy over linear, AdaBoost, random
forest and XGBoost auditors, each at seeds 0–4: macro F1 for binary classification
and R² for regression, maximized with artifact bytes and configuration hash as
tie breaks. Common retention never selects its configuration.

The wrapper validates runtime/source/dependency bytes before initializing optional
libraries and validates the complete artifact inventory before loading its locally
produced pickle. All sampler, metadata and projection bytes are charged. Removing
retained row containers passed exact original and serialized sample parity on
generated binary/regression inputs. These contracts are not reported-experiment
reproductions or an author-faithful paper dependency reproduction. Four generated
CPU/GPU fits and all 14 contract operations are cost evidence. The earlier GPU
probe's incomplete aggregate RAM admission remains explicit in the source lock.

Regenerate and verify the rights-safe source lock using verified local scratch:

```sh
python3 -m research.benchmark.publish_forestdiffusion_source \
  --schema research/benchmark/forestdiffusion-source.lock.schema.json
python3 -m unittest research.benchmark.tests.test_forestdiffusion_adapter \
  research.benchmark.tests.test_forestdiffusion_source -v
```

The method inventory records `pilot_locked`. Full campaign admission remains
closed, official tests remain sealed, and MFS-v2, PTF-v1 and release-safe scores
remain null. The historical CPU adapter/runtime and frozen pilot sources are
retained unchanged on scratch.

## Complete ForestDiffusion native/common panel

`results/pilot24-forestdiffusion-native.json` accounts for all 24 native attempts
(six CPU and two GPU trials per dataset), all 12 physical sampling jobs and all
108 logical default/selected sample cells. California's GPU author default and
native winner share identical fit and sample receipts; aliases add no compute.
Adult/News failures remain visible and contribute no DOPE wins. Successful
generator artifacts charge every retained file, including the projection.

The matched references keep fit seed 23, the same projected training/validation
partitions, three sample seeds and n/4n sizes for DOPE q8, all four DOPE GPU
research profiles, and default/native CTGAN/TVAE. Native tuning remains specific
to each author's objective. All three shared utility auditors are reported.
Sample min/max whiskers are not confidence intervals; this single-fit panel does
not replace the earlier five-fit stability analysis or public-core paired tests.

```sh
python3 -m research.benchmark.publish_forestdiffusion_native \
  --repo "$PWD" --output research/benchmark/results/pilot24-forestdiffusion-native.json \
  --schema research/benchmark/results/pilot24-forestdiffusion-native.schema.json
python3 -m research.benchmark.publish_forestdiffusion_native_figure \
  research/benchmark/results/pilot24-forestdiffusion-native.json
python3 -m research.benchmark.publish_forestdiffusion_manifest \
  --schema research/benchmark/results/pilot24-forestdiffusion-native.manifest.schema.json
```

The publisher fails closed before the complete frozen matrix is receipted. The
figure compares unconstrained common quality; the baseline artifacts exceed L3.
Official tests remain sealed, and MFS-v2/PTF-v1/release-safe scores remain null.

## Complete S3 matched discovery validation

`results/s3-matched-discovery-validation.json` compares four frozen DOPE GPU
research profiles with CTGAN and TVAE author defaults and native-selected
configurations on six rights-cleared S3 regression lineages. Each method uses
the same bounded official-training-derived fit/validation cohorts, fit seed 11,
three sample seeds and n/4n sizes. The panel accounts for 288 logical cells:
258 successful physical cells, 12 unavailable cells from two DOPE artifact-cap
failures, and 18 additional logical cells sharing default/native sample receipts.
Shared receipts add no compute. All retained artifact files and projections are
charged; baseline quality is reported independently of the L3 byte cap.

CTGAN/TVAE selections reproduce their frozen author-library native objective
(mean LinearRegression/MLPRegressor validation R2), with bytes/configuration
digest tie-breaks, across four original trials per method and dataset. Shared
CatBoost, linear and MLP utility never selects those baseline configurations.
Native KPI values remain descriptive and are not ranked across methods.

```sh
python3 -m research.benchmark.publish_s3_matched
python3 -m research.benchmark.publish_s3_matched_figure \
  research/benchmark/results/s3-matched-discovery-validation.json
python3 -m research.benchmark.publish_s3_matched_manifest
python3 -m unittest research.benchmark.tests.test_s3_matched_panel -v
```

Regeneration requires the immutable local scratch receipts. The publisher
verifies the complete matrix, native winners, sample/metric replays, charged
bytes and the full frozen source/runtime/binary/partition closures before
writing outputs. A separately pinned receipt lock preserves the original
published hashes; replacement receipts cannot acquire new accepted hashes.
The manifest repeats the full evidence reconciliation before binding
all tables, SVG/PDF figures, schemas and receipts. A separately locked capacity
continuation retains the original pilot deadline and metric/sample sources.

This single-fit discovery panel does not select a production DOPE family or
establish paired superiority. All profiles and failed cells remain visible;
sample min/max whiskers do not estimate fit uncertainty. Official tests remain
sealed, and MFS-v2/PTF-v1/release-safe scores remain null pending the complete
campaign and production gates.

## S3 matched confirmation validation

`results/s3-matched-confirmation-validation.json` is a separate panel on six
disjoint confirmation lineages. It retains all four frozen DOPE GPU research
profiles and the CTGAN/TVAE author defaults and native-selected configurations.
The same bounded S3 training-derived partitions, fit seed 11, three sample
seeds and n/4n sizes are used throughout. Native selections reuse the original
four-trial author-library efficacy searches; no shared KPI selects a baseline
and no new baseline fits or tuning trials are run for this panel.

```sh
python3 -m research.benchmark.publish_s3_confirmation
python3 -m research.benchmark.publish_s3_confirmation_figure \
  research/benchmark/results/s3-matched-confirmation-validation.json
python3 -m research.benchmark.publish_s3_confirmation_manifest
python3 -m unittest discover -s research/benchmark/tests \
  -p test_s3_confirmation_panel.py -v
```

The evidence publisher requires closed controllers and the complete frozen
matrix. It checks a separately pinned receipt lock before reading metrics,
sample/metric replays, original native winners, successful operation receipts,
charged model and projection bytes, and the declared source, dependency,
runtime and worker inventories. It rejects directory aliases. Current and
historical full executable closures remain unverified; inventories alone do
not certify them. GPU admission failures retain unavailable sample cells and
no artifact or DOPE win. Baseline quality is unconstrained where bytes exceed
10,240. Host costs and prior tuning costs are reported separately; earlier
architecture research and device-wide energy are not equal per-cell spend.

For scratch-independent regeneration of the tables and figures:

```sh
python3 -m research.benchmark.publish_s3_confirmation --from-json
python3 -m research.benchmark.publish_s3_confirmation_figure \
  research/benchmark/results/s3-matched-confirmation-validation.json
```

This panel does not choose a production family, estimate five-fit uncertainty,
replace the public-core paired analysis or open official tests. MFS-v2,
PTF-v1, release-safe L3 and paired superiority remain null.

## Author TabDDPM contract evidence

`tabddpm-source.lock.json` and `results/tabddpm-author-contract.json` record the
original MIT author Trainer, diffusion and MLP implementation at commit
`b476257dd460b778ba09eb97f7a51d6490fa17f8`. Two GPU fits use the author's
California numeric base settings on the same 72-row, five-feature S3
training-derived cohort. The train-only loader, tensor-only artifact export,
sampling size/batch changes and Python patch variation are disclosed. This is
a regression contract probe; native tuning, classification/categorical support
and the generic final runner remain pending.

The frozen native objective is mean validation CatBoost R2 over synthetic seeds
0–4, as in the author's tuning script. Fit seeds 11/23 measured
0.8035797920/0.7893576197 on that objective. These values are not ranked against
other methods' native KPIs. Both fits replay sample seed 0 exactly. Charged
artifacts are 500,946/500,979 bytes, including learned preprocessing and the
projection; both exceed L3. Quantile/discrete preprocessing is learned state,
and no claim that it contains no source observations is made.

```sh
python3 -m research.benchmark.publish_tabddpm_contract
python3 -m unittest research.benchmark.tests.test_tabddpm_contract -v
```

Publication verifies a pinned custody lock covering all 10,774 original source,
runtime, partition, receipt and artifact files before consuming results. A
separate immutable executable-custody supplement pins 65 existing bytecode
caches (17 dependency and 48 interpreter caches) and checks their code against
the pinned sources without executing either. The parent's hash/inventory checks
cover all 2,228 resolved interpreter-library files, extension modules and archive
search locations before invoking Python. An empty cache prefix with `-B` prevents
bootstrap from reading old bytecode before equivalence verification.
It pins the interpreter aliases and the original `lib64 -> lib` ABI alias; added
or changed runtime/author links and executable caches are rejected. Historical
fit-time executable closure remains unverified because the original locks omitted
bytecode and directory aliases. Current equivalence cannot prove historical cache
bytes. System shared libraries and drivers are outside the declared inventory;
complete current runtime closure also remains unverified. Artifact directory
links are rejected. Twenty physical
operations include four failed fit entrypoints and 16 successful operations;
three inherited sample-operation aliases are charged once. Total fit entrypoint
time including failures is 818.082278 seconds under the frozen 1,200-second cap.
Historical sources, failures and receipts remain immutable on scratch.
Official tests remain sealed; MFS-v2/PTF-v1/release-safe remain null.

## Central native method inventory

`methods.lock.json` binds the completed ARF, synthpop CART, TabPC and ForestDiffusion source and
native-validation evidence as `pilot_locked`. Their selection objectives are:

| Method | Native validation objective | Pilot search |
| --- | --- | --- |
| ARF | Maximize held-out FORDE mean log density | Four original and four refinement trials |
| synthpop CART | Minimize author `utility.gen` CART pMSE | Four trials |
| TabPC | Minimize author transformed validation NLL, with author early stopping | Eight trials |
| ForestDiffusion | Maximize author mean ML efficacy across four models and five auditor seeds | Six CPU and two GPU trials |

The entries pin adapter/source identities, configurations and existing pilot
panels. Native values are never ranked across methods. Artifact-only sampling,
charged learned state and source-rights limits remain explicit: synthpop uses a
hash-pinned CRAN release, its sampler is GPL research code, and TabPC's Cirkit
dependency is GPL-3.0-or-later. ARF's dependency digest and generic fit/sample
commands remain null pending an executable inventory and final adapter. The R
package lock and TabPC source/version lock do not establish complete current or
historical executable closure.

ForestDiffusion's native entry binds the existing study implementation to its
adapter hash and the author metric reference. Its regression objective is mean
R2; its binary objective is mean macro F1. Both average linear, AdaBoost, random
forest and XGBoost auditors over seeds 0–4, using synthetic sample seed 101.
The central tie-break names express the same existing bytes-then-configuration
order. The published pilot panel and manifest are hash-bound; failed trials and
unavailable native selections remain visible. The original MIT generator is
used, while the author ML formula is implemented separately by the study.
Repository-root experiment code with unresolved rights is not imported, and
the paper dependency environment has not been reproduced.

This reconciliation does not admit final jobs. Generic final runner integration,
the remaining method audits and the complete five-lock set are pending. The
central lock remains incomplete and unfrozen for final evaluation; official
tests stay sealed and gated scores stay null.

### AIM fixed-budget query objective

`aim_native.measure` implements the pinned author's synthetic marginal error:
the mean, over all column pairs including the target, of half the L1 distance
between independently normalized reference and synthetic histograms. The
reference is training-derived validation. The existing common-numeric adapter
uses eight public bins; synthetic seed 101 is fixed. Epsilon 1, 4 and 10 are
separate comparison budgets. Selection may never choose between them.

The current allowed algorithm settings remain the author's defaults at each
budget. A native-selected alias still needs frozen fit, sample and objective
receipts; this source audit provides no new fits, selections or measured values.
The train-fitted projection and validation selection lack end-to-end privacy
accounting, so formal DP remains false. Author-faithful categorical preprocessing
and its reproduction remain pending; binary-target adapter postprocessing is
part of this common-numeric representation. AIM stays `pilot_locked`, the final
five-lock admission remains closed, and MFS-v2/PTF-v1 remain null.

### Matched S3 confirmation with original Forest-Flow

`publish_s3_forest` extends the disjoint six-lineage S3 confirmation panel with
the original MIT Forest-Flow generator. The two declared configurations are
selected by the author's four-model, five-auditor-seed mean regression R².
The study implements that author formula; its paper dependency environment has
not been reproduced. Common validation outcomes never select Forest-Flow.
Default and native-selected aliases share one physical fit and sample schedule.

The panel retains all four DOPE research profiles and the prior default and
native-selected CTGAN/TVAE configurations, on identical projected inputs and
the same auditor implementation and package versions. Fit seed 11 and sample
seeds 101/211/307 are reported at n and 4n. Tables show all three auditors;
figures show CatBoost sample medians and ranges, with hatching for artifacts
over the 10,240-byte cap. Every retained sampler and projection byte is charged.

The publisher requires complete frozen receipt sets and checks their external
digests before reading metrics. It rehashes source, executable bytecode, runtime,
artifact, worker and sample inventories, rejects directory aliases, and verifies
the declared deterministic replays. Its manifests reproduce tables and figures
from the committed JSON. Runtime inventories do not certify the complete system
or driver closure. Single-fit-seed validation is not five-fit stability, the
public-core paired paper analysis or production certification. Official tests
stay sealed; MFS-v2, PTF-v1, release-safe L3 and paired superiority stay null.

### All 100 S3 DOPE GPU fit custody

`publish_dope_population_fits` publishes the complete 400-cell architecture
research round: 100 usable rights-cleared S3 lineages, four unchanged profiles,
fit seed 11. There are 384 fits within the 10,240-byte charged cap and 16
byte-cap failures. The charge includes the generator and projection. 354 new
GPU fits and 46 immutable prior successes are reported separately; all failures
and earlier repair/research costs remain visible. New fit operations consumed
3121.035606 seconds; this excludes admission waiting and earlier research.

This panel reports training and artifact accounting. Shared validation is a
separate running study; no global family, five-fit stability, public-core
paired outcome, privacy certification or final campaign admission is claimed.
Official tests remain sealed. MFS-v2, PTF-v1, release-safe and superiority are
null. Source and declared runtime identities are recorded; full system and
dynamic-library closure is not certified. No cell contributes a DOPE win.

```sh
python3 -m research.benchmark.publish_dope_population_fits
python3 -m research.benchmark.publish_dope_population_fits_figure \
  research/benchmark/results/dope-s3-population-gpu-fits.json
python3 -m research.benchmark.publish_dope_population_fits_manifest
python3 -m unittest discover -s research/benchmark/tests \
  -p test_dope_population_fits.py -v
```

The publisher verifies the externally pinned receipt lock before consuming
fit records, rejects changed source/data/model/evidence inventories and
reconciles every dataset/profile cell. Bulk artifacts and rows remain on
scratch; committed outputs contain only rights-safe metadata and hashes.

## Complete all100 DOPE GPU validation

`results/dope-s3-population-validation.json` accounts for every sample cell
from the four unchanged population research profiles: 2,400 logical cells,
2,304 measured and 96 unavailable because their fitted generators exceeded
10,240 bytes including projection. Identical generator and worker lineages
share 232 physical six-sample batches; all 232 completed successfully.
This is one fit seed (11), three sample seeds (101/211/307), and n/4n on
bounded cohorts derived from the official S3 training partitions.
Official tests remain sealed; no global family or production score is selected.

The JSON retains per-cell copy/near-match counts, real-versus-real controls,
fidelity diagnostics, null-normalized TSTR/TRTR retention and immutable receipt
hashes. Dataset/profile medians require all three samples to be informative;
missing or low-signal groups stay null. Tables and curves retain the denominator
of 100 lineages, negative retention, and all four profiles. No fit uncertainty
or paired superiority is inferred from sample repetitions. MFS-v2/PTF-v1,
release-safe L3 and superiority remain null until their complete gates pass.
The prior fit custody remains a separate historical publication.

Regenerate from the externally frozen receipt and reconciliation anchors:

```sh
python3 -m research.benchmark.publish_dope_population_validation \
  --receipt-sha256 ecd88832cb9c8c0018bb55de67ded9c3eba0936eba541e799ded0def34e173bc \
  --reconciliation-sha256 51e48fb3227588c0a8b1e49a64b94c1dd8b391063f94411aa33fc6bdac96004a
python3 -m research.benchmark.publish_dope_population_validation_figure \
  research/benchmark/results/dope-s3-population-validation.json
python3 -m research.benchmark.publish_dope_population_validation_manifest \
  --receipt-sha256 ecd88832cb9c8c0018bb55de67ded9c3eba0936eba541e799ded0def34e173bc \
  --reconciliation-sha256 51e48fb3227588c0a8b1e49a64b94c1dd8b391063f94411aa33fc6bdac96004a
python3 -m unittest discover -s research/benchmark/tests \
  -p test_dope_population_validation.py -v
```

Publication verifies the receipt-lock digest and complete referenced inventory
before reading metrics. Declared interpreter/package and Rust sampler inventories
are checked; complete system dynamic-library closure is not certified. Shared
sampling/evaluation operations cost 3,826.696542 seconds; coordinator wall time
was 10,178.424520 seconds. Prior GPU fit and repair costs remain explicit and
separate from final per-cell parity. Bulk rows, samples, weights and detailed
logs remain on scratch. Five fit seeds, full n/2n/4n/8n schedules, complete
privacy attacks, projection-only utility cost, public-core paired analysis and
five-lock final campaign admission remain required.

### Frozen SDV population native closure

`reconcile_sdv_population` verifies the frozen CTGAN/TVAE research ledger on
the same 100 S3 workers. The matrix has 704 new trials and 96 immutable prior
trials, with two earlier failed attempts charged separately. Author defaults
and the four-configuration native grid remain fixed. Native selection uses
the pinned mean SDMetrics LinearRegression/MLP regression R² only; byte count
and configuration digest break ties. Negative native values remain visible.
Sampling success and shared utility scores cannot change the native winner.

Run the closure after the frozen coordinator and supervisor have exited:

```sh
python3 -B -m research.benchmark.reconcile_sdv_population
python3 -B -m research.benchmark.reconcile_sdv_population --seal
```

The preview returns accounting without writing a seal. Sealing requires all
704 new job identities, the 96 verified historical trials, a clean coordinator
exit, stopped processes, frozen source and declared runtime inventories,
passing operation admissions, whole-operation quota evidence and exact sample
replay for successful sample phases. Failed fits retain partial artifact byte
charges; failed and timed-out operations retain elapsed compute. A valid native
KPI remains eligible when a later sample phase fails, and its shared outcomes
stay unavailable. No winner is frozen from a partially closed tuning group.
Historical runtime closure is never upgraded by reuse. Closure requires bytecode
writes and Python optimization disabled before loading the frozen verifier.
Transport time must cover monitor time within one microsecond; charged time is
the greater measurement. This applies to failed operations too. Unstarted or
omitted phases cannot contain execution evidence, including partial fit artifacts,
native samples and scheduled sample CSV/JSON outputs. Prelaunch request and
admission receipts remain allowed. The physical inventory permits only the
frozen jobs with one attempt each; extra retries, orphan jobs and aliases block
sealing. Added caches also block verifier loading.

This tool hashes worker and sample files without parsing their rows and does
not initialize ML libraries, fit generators, or open official tests. Its
research receipts keep MFS-v2, PTF-v1, release-safe and superiority claims null.
The source-only closure implementation does not publish a partial comparator
panel or admit the final campaign.

### Native retry and reuse preparation

After immutable SDV native closure, `sdv_native_continuation` prepares the next
retry/reuse inventory. Supply the receipt-lock digest recorded by that closure;
the tool has no inferred or default digest. Keep its output on scratch or under
the lane's `target/`:

```sh
python3 -B -m research.benchmark.sdv_native_continuation \
  --receipt-lock-sha256 "$SDV_NATIVE_RECEIPT_SHA" \
  > target/sdv-native-continuation-preview.json
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_sdv_native_continuation.py -v
```

The tool verifies the externally pinned receipt lock and every referenced file
before reading native results. It requires the complete 704 new/96 historical
trial ledger, clean stopped coordinator, frozen source and declared runtime,
and all 100 matched workers. Twelve fully historical lineages are retained even
though they have no new physical jobs. Native scores remain on the frozen
SDMetrics mean regression R² objective; sampling failures do not invalidate an
otherwise complete native result, and negative scores remain unclipped.

Complete fits can be reused for native scoring; complete native results can
be reused for later default/native-selected common sampling. These adapter
stages run on CPU. A retry requiring a new fit still needs GPU admission.
Retries preserve the original fit identity with a separate second attempt,
charge the four original trials and both earlier failed attempts where present,
and reserve at most 600 seconds each within eight attempts and 12 hours per
method–dataset cell. The author default has first retry priority. Winners stay
deferred until native retries close; shared outcomes cannot select them.

This source-only preparation launches no work, changes no live frozen round,
and certifies no speedup or runtime upgrade for historical trials. A new
source-versioned execution lock, monitor, recognized-owner registry, aggregate
CPU/GPU/RAM/scratch admission and respect for queued density reservations are
still required. Official tests stay sealed and all gated claims remain null.

`sdv_cpu_reuse.cpu_reuse` supplies the CPU operation core for complete physical
fits from SDV native-v2. It does not handle the 96 historical trials or perform
admission. A future entry point must first verify closed predecessor custody
and externally pinned fit/native receipts, verify the full runtime **before
importing the adapter**, hide CUDA, set `LOKY_MAX_CPU_COUNT=1`, and acquire fresh
disjoint CPU/RAM/scratch capacity under the owner registry. Its external monitor
must charge all verification and imports within the 600-second whole-process
limit. The core's elapsed timer includes its own source and artifact checks;
preflight failures before callbacks are accounted by that external monitor.

The core rejects changed source, job identity, worker files, model inventory
or model/projection byte charges before adapter callbacks. Native reuse runs
the unchanged validation SDMetrics objective with seed 1729, preserving
negative R². Common sampling requires complete native evidence for that same
fit, generates n/4n at seeds 101/211/307, and checks seed-101 replay against the
native sample. It checks artifacts after callbacks and worker custody before
writing a successful native or sample-batch receipt. Callback failures retain
typed failure and elapsed-cost evidence; no new fit is called.

```sh
python3 -B -m unittest discover -s research/benchmark/tests \
  -p test_sdv_cpu_reuse.py -v
```

These controls use opaque fixtures without ML initialization or real data.
This component has no launcher, admitted runtime, measured speedup or campaign
results. It leaves the live SDV round and density wait gate unchanged. The
source-versioned execution lock and monitor remain required before use.

The [GReaT source audit](GREAT_SOURCE_AUDIT.md) binds the current MIT author
implementation and identifies its defaults, required pretrained checkpoint,
generic native ML utility API and artifact state. It verifies no runtime or
fit/sample execution and leaves GReaT `source_audit_pending`. A complete author
default, defensible native scalar or tuning-inapplicable receipt, weights and
runtime custody remain required before admission.

The [REaLTabFormer source audit](REALTABFORMER_SOURCE_AUDIT.md) binds the MIT
author tabular implementation, literal defaults and bootstrap sensitivity
checkpoint policy. Its utility API requires a caller-supplied auditor and
returns predictions; the native scalar/search/default runtime remain unfrozen.
It stays `source_audit_pending`; no model or campaign job is admitted.

The [CDTD source audit](CDTD_SOURCE_AUDIT.md) binds the MIT author code and
released noise-schedule variants. It identifies the author absolute RMSE-gap
objective, terminal EMA policy, zero-categorical assumptions and checkpoint
row retention. Native scalar/search, safe inference artifacts and executable
runtime remain unfrozen; CDTD stays `source_audit_pending`.

The [TabCascade source audit](TABCASCADE_SOURCE_AUDIT.md) distinguishes the
current author Python discretizer from the paper's R track and records native
detection quality separately from regression efficacy. Paper/default boosting
iterations differ; runtime, track equivalence, selection and artifact contracts
remain unfrozen. TabCascade stays `source_audit_pending`; no job is admitted.

The [CTAB-GAN+ source availability audit](CTABGAN_PLUS_SOURCE_AUDIT.md) binds
the pinned author snapshot and inspected source-rights gap. Its explicit
unavailable receipt admits no execution or formal DP claim and contributes
no DOPE win. Native tuning requires rights and runtime admission first.

The [MST author-source audit](MST_SOURCE_AUDIT.md) pins licensed current-author
MBI source, defaults and native marginal-error evidence. Runtime, reusable-model
adaptation, artifact-only fit/sample and native selection bindings remain
pending. MST stays `source_audit_pending`; no execution or formal DP is admitted.

The [TAEGAN source availability audit](TAEGAN_SOURCE_AUDIT.md) binds
the pinned author snapshot, inspected source-rights gap and withheld preprocessing
and recovery steps. Its unavailable receipt admits no execution and contributes
no DOPE win. Native selection requires rights, a complete pipeline and runtime first.

The [DP-CTGAN author-source audit](DPCTGAN_SOURCE_AUDIT.md) binds the original
paper-linked DP branch, source defaults and native classification efficacy.
Source privacy ordering, stored training rows, older runtime and artifact/native
selection gates remain pending. No formal DP, execution or scored result is admitted.

The [TabbyFlow author-source audit](TABBYFLOW_SOURCE_AUDIT.md) binds the original
EF-VFM release and native XGBoost RMSE/AUROC outcomes. Epoch/checkpoint/sampler,
validation target-scale, original test-path and runtime/artifact gates remain
pending. No execution, scored comparison or production claim is admitted.

The [original PATE-GAN source audit](PATEGAN_SOURCE_AUDIT.md) distinguishes the
author-lab BSD3 implementation from the later auditing project. Native logistic
AUC selection, generated-array/row-free artifact, preprocessing/selection privacy,
legacy runtime and fresh capacity gates remain pending; no execution or DP claim.

The [TabPFGen source/reproduction readiness receipt](TABPFGEN_SOURCE_READINESS.md)
records scoped unavailability for this campaign. The available Apache2 candidate
is explicitly independent and has no verified two-experiment reproduction;
original-author implementation/defaults and safe model-context artifacts remain
unverified. It contributes no DOPE win or win-denominator entry.

The [TabKDE source-rights receipt](TABKDE_SOURCE_RIGHTS.md) binds the already
recorded original commit and selected rights scope. Execution rights and a
licensed independent implementation with two reproductions remain unverified;
this unavailable method contributes no DOPE win or win-denominator entry.

The [GEM-T source/reproduction readiness receipt](GEMT_SOURCE_READINESS.md)
records the bounded inspected provenance and an unrelated same-name candidate.
Original executable source and a two-experiment independent reproduction remain
unverified; its paper objective is identified without admitting tuning or jobs.
This unavailable method contributes no DOPE win or win-denominator entry.

The [compressed-member contract](CONTAINER_MEMBER_CONTRACT.md) verifies the
complete encoded byte charge and independently frozen kernel/projection digests
before returning immutable owned members. It admits no sampler, runtime or job;
compressed-candidate validation still waits on predecessor closure and capacity.


The [compressed-worker startup source gate](CONTAINER_VALIDATION_WORKER.md)
checks the exact fifteen application guard files before imports and returns
immutable owned source bytes on the original whole-batch timer. This source-only
preparation admits no interpreter, runtime, sampler or dispatch; SDV closure,
density priority and fresh capacity remain required.


The [owned startup code preparation](CONTAINER_VALIDATION_WORKER.md) compiles
verified guard source bytes without running initializers or reading cached
bytecode. Compiler flags, source digests and the original deadline remain bound;
this prepares no executable child, runtime certification or dispatch admission.

The [proposed guard module bindings](CONTAINER_VALIDATION_WORKER.md) pair those
owned code objects with verified source paths and fixed private package names.
Repeated source membership and helper checks retain the original deadline.
No initializer or namespace is installed; closed imports, runtime/lease/capacity
and predecessor/density admission remain required before execution.

The [static guard import plan](CONTAINER_VALIDATION_WORKER.md) records import
declarations from matching owned source bytes and orders declared dependencies.
It rejects unknown or cyclic guard dependencies without resolving a module or
running an initializer. Actual imports, runtime and dispatch remain unadmitted.
