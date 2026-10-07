# Baseline coverage and remaining evidence

Updated 2026-10-07 from merged validation publications. This memo authorizes
no fit, sample generation, GPU use or official-test access.

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
`coverage-plan.json` freezes the forecast assumptions as of 04:55Z.

| Method | Complete measured scope | Remaining full-population scope | ETA MDT |
| --- | --- | --- | --- |
| TabSyn | Eight-lineage scaled-default cohort | 100 default/native-selected lineages | Oct 14 20:00, conditional |
| TabDDPM | Contracts/pilot only in this population publication interface | Complete 100-lineage default/native panel | Oct 14 20:00, conditional |
| Forest-Flow | Six-lineage native-selected 4n confirmation | Complete 100-lineage default/native panel | Oct 14 20:00, conditional |
| Forest-Diffusion | Pilot mode receipts | Separate complete 100-lineage diffusion-mode panel | Oct 14 20:00, conditional |
| ARF | Complete 100-lineage fit11 default/native n/4n panel | Final five-fit matrix | Retained panel observed complete Oct 6 21:16; final five-fit unscheduled |

Each future forecast assumes a dedicated eligible slot starting Oct 8
08:00 MDT: 100 default fits plus up to 800 native tuning trials at 600s,
150 slot-hours, plus six hours for sampling/metrics. No slot has been
leased by this table. Encoding-priority work retains priority. Contention,
source/runtime readiness and fresh admission can move the dates. Final
five-fit baseline matrices are separate and unscheduled. Native tuning
must use each method's own objective; shared metrics select nothing.

## Scientific limits that remain

Official tests stay sealed; the observations are training-derived validation
results. MFS-v2, PTF-v1, release-safe L3 and production superiority remain
null. TabSyn's completed subset and the six-lineage Forest-Flow confirmation
cannot establish a full-population neural ranking. Fit-seed variance and the
full DOPE ablation grid remain uncompleted. The attacks reported here are
empirical; attribute-inference and full certification coverage are not
inferred. Missing/unavailable source or method cells contribute no DOPE win.
Equal per-cell caps do not imply equal total architecture-research compute.
