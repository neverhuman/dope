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

## 2. Frozen TDL panel edition (2026-10-18)

The immutable `tdl-panel-2026-10-18-edition-1` role roster is specified in the example in [`panel-v1.schema.json`](../schemas/panel-v1.schema.json) and bound to each checkpoint in `research/transfer/registries/models.json`. The edition fixes regression and classification as mandatory workloads at equal weight (1.0 each), uses real-only references for eligibility, and lists CatBoost, LightGBM, XGBoost, RealMLP, linear, independent boosting, and independent neural as mandatory models. These commitments precede generator outcomes.

For every checkpoint and workload, the panel roster declares `compiler_teacher`, `selection`, `required_confirmatory`, `challenger_only`, and `native_recipe_sensitivity`. CatBoost, LightGBM, XGBoost, RealMLP, and linear are compiler teachers; Kumo Tabular, Mitra-v2, and TabICLv2 are selection exposed. Independent boosting and independent neural are mandatory confirmation-only families, held out from compiler-teacher and selection roles. KNN and kernel are challenger-only. Teacher lineage overlap is explicit per checkpoint (`known_none` or `unknown`); unknown foundation-model overlap remains a conflict and cannot be read as clean independence.

The panel is specified and unqualified. Role, conflict, and rights signatures are SHA-256 content digests over the roster assertions; they provide integrity fingerprints, not external approval or proof of rights. Rights and checkpoint identities remain unresolved where the registry has no evidence. Recursive resolution and all three signatures are prerequisites for qualification. Missing mandatory native qualification blocks a panel-qualified pass; family averages cannot rescue it.

Future challenge panels append a new immutable edition against the same artifact zoo. They do not recompile that zoo or overwrite historical scores. A new model edition schedules only its newly required evaluation cells. Historical panel scores retain their original edition and interpretation.

## 3. Auditor status (T-065)

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

## 4. Reproduce the v2 numbers

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

## 5. The TDL benchmark (specified, not run)

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

### 4.1 Source-balanced roster plan (E00)

`research/transfer/manifests/datasets.json` is a planning denominator, not an
acquisition receipt. Its 116 rows are target slots: Quick-12 (12), Open-60 (60),
Sealed-24 (24), and MechanismLab stress (20). The first three groups are split
80/20 regression/binary classification (Quick-12: 8/4, Open-60: 40/20,
Sealed-24: 16/8); multiclass is outside the initial admission and requires a
separate protocol and power review. The current task labels are candidate task
types, not verified properties of acquired tables. Horizons for time-shift
tasks are defined per source before outcomes as the earliest contiguous 60%
for fit, next 20% for selection, and latest 20% for external audit; non-temporal
sources use entity-group 60/20/20. Site holdout is an additional external audit
partition wherever multiple sites exist. If a source cannot support its frozen
split, it is excluded before outcomes and remains in the denominator with a
reason; it is never replaced after outcomes.

Candidate pools currently name six possible source organizations/families for
Quick-12 and Open-60 (UCI, OpenML, StatLib, PMLB, PhysioNet, and World Bank),
and twelve additional candidate families for Sealed-24 (NHANES, USDA ERS, NOAA,
Eurostat, NCES, WHO, US Census ACS, CDC, FAOSTAT, OECD, EIA, and BTS). These are
discovery leads only. The repeated Open-60 slots do not assert 60 independent
sources, and repeated entries from one family count once. Sealed-24 requires
at least 12 independently owned, non-overlapping families new to development
and selection. A dataset and every alias, version, alternate target, repeated
entity extract, or simulated derivative inherit one parent family; missing
identity or unresolved overlap blocks admission rather than increasing counts.
PMLB's 20 MechanismLab stress slots are task cases from one task pool, not 20
independent sources and never contribute to RealCore or ShiftCore counts.

Before any outcome access, freeze a row for every candidate with exact source
identity and SHA-256, task and prediction horizon, entity/time/site and source
partition hashes, duplicate/contamination review digest, license decisions,
and an attrition/exclusion disposition. All rights, exposure and source hashes
are currently unresolved, so the roster is not complete for execution and all
headline eligibility stays false. Fit and selection may use only their frozen
views; final labels stay with the external evaluator. Regression and binary
results are reported first. Multiclass rows remain separately pending and
cannot be counted as initial-benchmark successes.

The primary finite-roster estimand is the equal-weight mean of source-family
means over the licensed, independent families admitted in the frozen roster;
within each family, canonical tasks, fits, samples and learners receive equal
weight. Each family therefore receives weight `1/F` and weights sum to one.
The future-source estimand is the expected family-level result for a draw from
the explicitly named source-population frame, with equal family weighting;
it is not inferred from the finite roster without a sampling frame and
probability-based inclusion. Report both only when their populations and
uncertainty procedures are defined before outcomes.

Before admitting any cohort size, attach a per-source cost sheet for rights and
acquisition, normalization/model development, five fits by three samples,
driver evaluations, privacy attacks, and the predeclared power expansion.
Record vendor or compute quote, units, one-time versus marginal cost, wall-time
and hardware assumptions, and a maximum spend; unresolved quotes mean no
admission. Power expansion is estimated from blinded pilot variance and the
predeclared minimum detectable effect, with an explicit cap and no outcome-led
roster enlargement. The present files contain no quotes or measured pilot
variance, so Quick-12/Open-60/Sealed-24 are targets only and no campaign is
authorized by this plan.

## 5. Where the operational detail lives

The benchmark tooling overview is in
[`research/benchmark/README.md`](../research/benchmark/README.md). The dated campaign
diary moved out of that file is in
[`docs/archive/benchmark-operations.md`](archive/benchmark-operations.md). The README
material that described MFS-v4, the artifact format and the full command catalog is in
[`docs/archive/mfs-history.md`](archive/mfs-history.md).

## 6. Comparator execution contract (E01)

`research/transfer/locks/comparators.json` records the method inventory and the
external-runtime dispositions. The portable controls are development-only numeric
reference controls; they are not claims that ARF, synthpop, TabSyn, TabDiff,
TabDDPM, ForestDiffusion, CTGAN or TVAE have executed. Those external entries stay
unresolved until an exact source/license/runtime pin and native parity receipt exist.
The existing neural lock preserves the pinned TabSyn, TabDiff and TabDDPM source
revisions and licenses; it does not qualify their runtime.

Portable artifacts use the versioned canonical JSON closure in
`research.transfer.adapters.generators.comparators`: the manifest, payload digest,
codec and decoder identity are included in the charged artifact bytes. Native,
float32, float16 and int8/pruned encodings use this same closure. A byte-infeasible
artifact is a terminal disposition. `protocol_pin` binds release scope, development
split, auditor/model, loss and byte cap. `ComputeSchedule` divides one total
fit/selection budget and rejects mismatched totals. Native defaults, equal-total
fit/selection compute and rate-constrained artifacts remain separate comparison
tracks; no track borrows budget from another.

`compressed_full_source` and `real_row_copy` are nonrelease controls. They retain
real development rows and cannot be sampled through the release API. The selected,
random, stratified, farthest-first coreset and prototype controls use only the
provided development rows. Any LAD/data-line diagnostic must use a development-only
selector; it does not open the sealed test. The portable numeric reference controls
are not substitutes for a native external runtime or native-objective parity.

DP AIM/MST remain in a separate matched privacy track. Large native states and
methods with 40 MB or larger exports stay visible in results and rate grids; they
are not silently omitted because they miss a small byte rung.

## 7. E03 observational and adverse-mechanism pilot (blocked before launch)

The frozen pilot design is in
[`research/transfer/manifests/e03-pilot.json`](../research/transfer/manifests/e03-pilot.json).
It specifies one observational task and one adverse mechanism task, DOPE and the
Gaussian-copula reference at two budgets, plus compressed real rows as a non-generator
reference. Required traces cover prepare, compile, reload, source-free sampling,
real/null/synthetic model evaluation, driver analysis, Rust receipt, and report. The
manifest's native launch command remains null and execution is unauthorized.

No run is admitted: `manifests/datasets.json` leaves candidate source rights and
contamination review pending, while `locks/comparators.json` records external runtimes
as unresolved and portable controls as unqualified references. Source ownership and
collision review, a protected-owner admission, concrete resource budget, and W03 modern
panel/scheduler admission are also absent. E07's earlier CPU smoke is a separate evidence
scope and does not satisfy E03. No native work was launched and no observational,
privacy, driver, or cost result is claimed.

Quick-12/three-budget/one-fit cost remains unestimated until an admitted smoke provides
observed runtime and uncertainty, driver, and privacy accounting. The suggested next
smoke is three tasks, two budgets, and two models. Private per-query receipts and public
redacted aggregates remain separate requirements. Ordinary fixtures are locked as the
manifest preflight command; they cannot clear the external admission gates. The pilot
must stop on inadequate references, weak attack power, a critical rare slice, unresolved
identity, unbounded cost, or any unqualified required model.
