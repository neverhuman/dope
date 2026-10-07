# Baseline coverage and remaining evidence

Updated 2026-10-07 from validation publications and the receipt-backed CPU
replication checkpoint below. This memo itself authorizes no fit, sample
generation, GPU use or official-test access; execution follows the separate
owner order and frozen research rounds.

## Completed evidence

- TabSyn: eight complete lineages, 48 sample evaluations at n/4n and seeds
  101/211/307, fit seed 11, scaled author defaults. Fidelity, utility,
  detection and empirical privacy are measured. This is not a 100-lineage
  or native-tuned TabSyn comparison.
- ARF, GaussianCopula and study-owned Chow-Liu: each has 1,200 logical
  default/native cells on the same 100 lineages, n/4n and three sample
  seeds. The distinct metric records are 1,116, 972 and 1,050 respectively;
  numeric aliases are not extra independent replicates.
- The deterministic first12 diagnostic batch has authenticated physical
  evaluation receipts and incomplete sample groups. It supports per-cell
  diagnostics, not population medians, intervals or rankings.
- The earlier density, CTGAN/TVAE, ARF retention and six-lineage Forest-Flow
  confirmation blocks remain separately measured. Scheduling cutoffs are
  not method failures. Native KPIs are not ranked across methods.

Sources: `research/benchmark/results/tabsyn-eight-lineage-validation/`,
`research/benchmark/results/retained-classical-validation/`,
`research/benchmark/results/common-A952-validation-v1/`,
`research/benchmark/results/arf-matched-population-validation.json` and
`research/benchmark/results/s3-matched-forest-confirmation-validation.json`.
The original scratch receipt paths and their hashes remain in those JSONs.

## Remaining strong-baseline coverage

The executable coverage table and ETA arithmetic are
`generated/baseline-coverage.json` and `generated/baseline-coverage.tex`.
`coverage-plan.json` freezes the earlier forecast assumptions as of 04:55Z.
The generated coverage table remains that snapshot; the new CPU execution
checkpoint and its forecasts are recorded separately below.

| Method | Complete measured scope | Remaining full-population scope | ETA MDT |
| --- | --- | --- | --- |
| TabSyn | Eight-lineage scaled-default cohort | 100 default/native-selected lineages | Oct 14 20:00, conditional |
| TabDDPM | Contracts/pilot only in this population publication interface | Complete 100-lineage default/native panel | Oct 14 20:00, conditional |
| Forest-Flow | Six-lineage native-selected 4n confirmation; eight additional defaults at four fit seeds on two lineages, 48 common evaluations | Complete 100-lineage default/native panel and missing seed11 replication | Oct 10 02:00 new-default fit/metrics forecast; broader scope Oct 14 20:00, conditional |
| Forest-Diffusion | Pilot mode receipts | Separate complete 100-lineage diffusion-mode panel | Oct 14 20:00, conditional |
| ARF | Complete 100-lineage fit11 default/native n/4n panel; 24 additional fit23 receipts on 12 lineages with 144 common evaluations | 796 physical / 800 logical additional fits across seeds 23/37/53/71, then remaining common metrics | Oct 8 02:00 fit close, conditional forecast; metrics follow |

Each future forecast assumes a dedicated eligible slot starting Oct 8
08:00 MDT: 100 default fits plus up to 800 native tuning trials at 600s,
150 slot-hours, plus six hours for sampling/metrics. No slot has been
leased by this table. Encoding-priority work retains priority. Contention,
source/runtime readiness and fresh admission can move the dates. Final
five-fit neural matrices remain separate; the CPU replication rounds below
are running under their own frozen research admission. Native tuning
must use each method's own objective; shared metrics select nothing.

## Active CPU replication checkpoint (2026-10-07)

`research/benchmark/results/cpu-fivefit-progress-v1/manifest.json` pins
24 additional ARF fits and eight Forest-Flow fits, all successful in this
published prefix. The ARF prefix covers 12 lineages at fit seed 23.
Forest-Flow has seeds 23/37/53/71 on two lineages using author defaults.
The same frozen training-derived splits are used; no official test is read.
Each model's checkpoint hash, charged bytes and ARF's native
validation density value appear in the per-fit JSON/CSV.

`research/benchmark/results/cpu-fivefit-validation-v1/manifest.json` now
adds the complete common-metric schedule for those same 32 fits: 144 ARF
and 48 Forest-Flow cells. Each fit has n/4n and sample seeds 101/211/307.
Fidelity, utility, detection and empirical privacy are measured; summaries
keep generator fit seeds separate. ARF evaluation closed with actual exit0
in 120.600844 seconds, Forest-Flow in 26.747209 seconds. These are measured
prefix costs, not full-population throughput forecasts. The initial ARF and
Forest cohorts have different sizes and cannot be ranked against each other.
Full-population and five-fit completion remain false.

The ARF queue replicates the existing default and native-selected winners:
796 physical fits, 800 logical configurations, with no new tuning. The
fit-close forecast is Oct 8 02:00 MDT, conditional on its four-core CPU slot;
remaining common n/4n measurements follow; the first published prefix is complete.
This 20-slot-hour planning allowance is conservative relative to the
previous native round but is not a measured full-round duration.
Forest-Flow's 400 additional default fits have a conditional fit-and-metrics
forecast of Oct 10 02:00 MDT: at most 400 times 600 seconds of fit work plus
six hours provisionally reserved for evaluation. Storage/RAM admission can
extend it. Native-selected replication and missing seed-11 coverage require
complete native-winner receipts and remain unscheduled. The initial small
lineages cannot determine full-population throughput or method ranks.

Both slots use four cores, low CPU/I/O priority, 8 GiB RAM caps and a 15%
available-RAM floor. The CPU allocation change is recorded in the frozen
rounds. GPUs remain subject to encodings-first admission, and no neural
run is authorized by these forecasts. No existing artifact is deleted.

## Scientific limits that remain

Official tests stay sealed; the observations are training-derived validation
results. MFS-v2, PTF-v1, release-safe L3 and production superiority remain
null. TabSyn's completed subset and the six-lineage Forest-Flow confirmation
cannot establish a full-population neural ranking. Fit-seed variance and the
full DOPE ablation grid remain uncompleted. The attacks reported here are
empirical; attribute-inference and full certification coverage are not
inferred. Missing/unavailable source or method cells contribute no DOPE win.
Equal per-cell caps do not imply equal total architecture-research compute.
