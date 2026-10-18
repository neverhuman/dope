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

### Source admission (D02; development metadata only)

[`source-manifest-v1.schema.json`](../schemas/source-manifest-v1.schema.json) is a
closed Draft 2020-12 schema for **private** source mappings. It requires acquisition
URI/method, source version/content SHA-256, license and reviewed rights, target and
prediction-time definitions, group/site/entity/time hierarchy, feature availability,
and corpus-overlap lineage for each exact foundation checkpoint (including Kumo and
TabICL). `known_none` and `overlapping` require corpus/review identities; `unknown`
and `overlapping` carry explicit caveats. Unknown contamination never becomes clean.

`research.transfer.lineage.admit_source_manifest(manifest, registry, signing_key=key)`
checks the schema and registry grants before returning a public projection. The
declared split must be `entity_then_time_then_fit`, with operations `entity`, `time`,
`fit_transforms` in that order and transforms fitted only on `train`. Canonical
parent families (including aliases and derivatives), and identical content, cannot
span `train`/`eval` cohorts. Features available after prediction time, target-as-input,
unresolved acquisition/training rights, and unapproved redistribution are refused.
An explicit `owner_only_custody` or `prohibited` redistribution status permits only
owner use; it grants no redistribution right. Absent group/site levels are explicit
nulls; entity/time definitions are mandatory.

Keep private manifests and a random custody key of at least 32 bytes outside public
artifacts. The returned allowlisted projection contains keyed opaque identities,
rights dispositions and checkpoint overlap states, with a domain-separated
`hmac-sha256-custody-v1` signature. `verify_source_projection` authenticates it using
the custody key; this is a MAC, not a publicly verifiable signature. Headers, cells,
extrema, URIs, timestamps, private paths and unkeyed source hashes are never copied.
Unknown/overlapping checkpoints produce `development_only_caveated` admission.

The compatibility inventory in `research/transfer/registries/sources.json` remains
unfrozen with unknown rights; it supplies no admitted TDL source. These checks validate
owner assertions, not source bytes, legal clearance or actual transform execution.
**D18 custody HOLD remains unresolved**: every projection fixes
`confirmation_allowed=false` and `sealed_source_claims=false`. No sealed source is
opened, reconstructed, or qualified by this manifest or its signature.

## 6. Where the operational detail lives

The benchmark tooling overview is in
[`research/benchmark/README.md`](../research/benchmark/README.md). The dated campaign
diary moved out of that file is in
[`docs/archive/benchmark-operations.md`](archive/benchmark-operations.md). The README
material that described MFS-v4, the artifact format and the full command catalog is in
[`docs/archive/mfs-history.md`](archive/mfs-history.md).
