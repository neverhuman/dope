# Benchmark

**Active evidence:** the v2 validation panel, labeled `selection_exposed_validation`.
**Planned:** the TDL benchmark, labeled `specified_unrun`. The official test split is
sealed. Labels and wording rules are in the [claims policy](claims-policy.md). The metric
is defined in [metric](metric.md).

No result number is typed on this page. The panel numbers are rendered into the README's
generated benchmark block from `docs/whitepaper/generated/v2-headline.json`, and the
paper's tables come from the same committed ledgers.

## 1. The v2 validation panel

| Item | Value |
| --- | --- |
| Tables | 100 PMLB regression tables. 65 lineages come from four simulated source families, and fit views hold at most 800 rows. |
| DOPE profile | `features12_steps2048`: compact residual, at most 12 inputs, hidden width 16 |
| Fit seeds | DOPE, ARF and TabSyn use five fit seeds (11, 23, 37, 53, 71); every other comparator has one |
| Auditors | CatBoost, linear regression and an MLP, each trained on synthetic rows and scored on real validation rows |
| Synthetic sizes | `n` and `4n` |
| Quantity | null-normalized retention: the share of a real-trained auditor's gain over the null that the synthetic-trained auditor recovers |
| Comparison | paired Hodges–Lehmann differences on shared lineages, with 95% nested bootstrap intervals (simulated-family cluster, then lineage, then fit seed) |
| Controls | resampled real rows (a row-count diagnostic, not a generator) and a predictor-only control |
| Bytes | the encoded `.dpk` file plus `projection.json`: every byte that sampling reads |
| Analysis contract | [`research/benchmark/review_fixes/predeclare_v2.json`](../research/benchmark/review_fixes/predeclare_v2.json), committed before the five-seed results existed |

**Why the label is `selection_exposed_validation`.** Four DOPE profiles were compared on
validation retention at fit seed 11, and no contemporaneous record states the selecting
rule. Every lineage and all three auditors therefore count as exposed to the profile
choice. Only the pre-declared sealed run (CM-09, decision D-08) can give a selection-free
estimate.

**Cohorts are not one ranking.** The full-panel rows (DOPE, TabSyn scaled schedule,
both ARF configurations, Gaussian copula, Chow–Liu, independent marginals and the
controls) share lineages. The CTGAN and TVAE block and the historical Forest-Flow block
are smaller, separate cohorts, and the README table's lineage column says so. A row from
a small cohort is never read as a win or a loss against a full-panel row.

**Not run.** These are Stage 2 blocks of `predeclare_v2.json` with no number: TabDDPM v6,
TabSyn author-default, Forest-Flow under the common budget, TabDiff, GReaT, synthpop,
SMOTE, the ablation grid and the stress suite. A missing or failed cell is never a DOPE
win.

**What the panel does not show.** It is CatBoost-centered. It has no held-out learner
family and no held-out source family, and it does not measure driver fidelity. Empirical
attacks cover only the stated cells, and measured copy, membership and attribute gates
already fail on some of them. It therefore supports no TDL, cross-model or privacy claim.

## 2. Auditor status (T-065)

| Auditor | Where it appears | Status |
| --- | --- | --- |
| CatBoost, linear regression, MLP | v2 panel auditors | measured under the v2 protocol; selection-exposed |
| CatBoost, LightGBM or XGBoost, a neural tabular learner, elastic net or EBM | TDL core panel (registry, T-060) | `specified`, not yet qualified for TDL |
| Kumo Tabular | MFS-v4 contract (historical); TDL frontier panel | **planned, unqualified** |
| Mitra-v2 | MFS-v4 contract (historical); TDL frontier panel | **planned, unqualified** |
| TabICLv2 | MFS-v4 contract (historical); TDL frontier panel | **planned, unqualified** |
| TabPFN | TDL frontier panel | **planned, unqualified** |
| LimiX, TabDPT, TabFM | TDL frontier panel | **planned, unqualified** |

An auditor counts toward TDL only after its lock reaches `qualified`. The lock path is
`specified`, `source_verified`, `weights_verified`, `environment_verified`,
`parity_verified`, `resource_verified`, `qualified`. Qualification needs owner decision
D-07 (licenses and GPU budget) and is follow-on work (FOLLOW-AUDQ).

## 3. Reproduce the v2 numbers

```bash
just setup          # Python environment for the paper scripts
just paper          # regenerate every generated/ file, figure, the README block and both PDFs
just paper-check    # verify macros against ledgers, figure hashes, and the build
bash ops/ci/paper-clean.sh   # what CI runs: delete generated/, rebuild, require a zero diff
```

Every number in the historical paper is a macro emitted from a committed, SHA-pinned
ledger. `research/benchmark/verify_paper_numbers.py` fails if a printed value disagrees
with its ledger. The public data path needs no credentials:
`python3 -m research.benchmark.fetch_pmlb --list` lists the tables, and
`python3 -m research.benchmark.fetch_pmlb DATASET_ID OUTPUT_DIRECTORY` fetches one.

## 4. The TDL benchmark (specified, not run)

The design comes from the 2026-10-15 specifications. Execution is follow-on work that
needs compute, licenses or owner decisions.

- **Tracks.**
  - Author-default: each method in its published configuration.
  - Equal-compute: matched selection budget.
  - Rate-constrained: methods compiled to the same byte grid as DOPE.
- **Cohorts (T-064, FOLLOW-DATA).**
  - RealCore: real tables, the headline majority.
  - ShiftCore: grouped and temporal shift.
  - MechanismLab: PMLB and procedural tasks, mechanism tests only.
  - StressCore: known interactions and hard cases.
  - These are tiered as Quick-12, Open-60 and Sealed-24. Rights and contamination fields
    are recorded for every dataset.
- **Budgets and seeds.** The byte grid is `{0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256}` KiB.
  Each artifact uses five fit seeds by three sample seeds.
- **Controls.** Copied rows, the gzip-compressed original, a row-subset (coreset) frontier,
  predictor-only and independent marginals. A compressed original is a size reference,
  never a generator.
- **Generator panel (FOLLOW-BASELINES).**
  - Compression controls.
  - Classical generators: copula, CART or synthpop, ARF, SMOTE.
  - Neural generators: CTGAN, TVAE, TabDDPM, TabSyn, TabDiff, GReaT, Forest-Flow.
  - Every generator is admitted through the same source, license and parity checks.
- **Failure accounting.**
  - Every scheduled cell ends in a terminal disposition.
  - Failures, timeouts and refusals stay in every denominator.
  - A figure never mixes cohorts without a prominent disclosure.
- **Sealed test.** Opened once, through a frozen release manifest (FOLLOW-SEAL, D-08).

## 5. Where the operational detail lives

The benchmark tooling overview is in
[`research/benchmark/README.md`](../research/benchmark/README.md). The dated campaign
diary moved out of that file is in
[`docs/archive/benchmark-operations.md`](archive/benchmark-operations.md). The README
material that described MFS-v4, the artifact format and the full command catalog is in
[`docs/archive/mfs-history.md`](archive/mfs-history.md).
