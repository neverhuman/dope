# DOPE 1.0: Transfer Description Length Engineering Specification

**Working subtitle:** From a synthetic-data scorecard to an executable rate–distortion benchmark  
**Repository audited:** `neverhuman/dope`, default branch  
**Audit date:** 2026-10-15  
**Primary audience:** DOPE maintainers, benchmark engineers, research authors, reviewers, and external contributors  
**Status:** Proposed redesign; all current numerical claims remain development/validation evidence until a new frozen protocol and sealed test are complete

---

## 0. Executive verdict

DOPE contains a genuinely important idea, but the repository currently hides it under too many metric generations, release contracts, caveats, compatibility layers, campaign receipts, and internal audit documents.

The profound idea is not “a better synthetic-data score.” It is:

> **How few independently usable bits are required to preserve what powerful learners would have learned from the real table?**

That question should become the entire project.

The project should be rebuilt around three objects:

1. **The transfer–description frontier**: predictive transfer on unseen real data versus exact charged artifact bytes.
2. **Transfer Description Length at a declared threshold**, abbreviated **TDL@τ**: the minimum charged bytes required to preserve at least `τ` of real-data learning across a frozen model-family suite.
3. **A fail-closed certification tuple**, not a weighted “master fitness” number: `{bytes, transfer, driver fidelity, privacy/copy status, scope, evidence status}`.

The public operating point should be:

> **TDL@90: How many bytes are needed to preserve 90% of real-data learning?**

The current development result is already striking enough to motivate this direction: the README reports a DOPE median artifact of about 1.9 KB with CatBoost transfer retention of 0.898, versus about 42.3 MB and 0.924 for the tested scaled TabSyn schedule. That is roughly a 22,000× difference in dataset-specific state, while the paired utility interval does not demonstrate separation. This is a powerful compression result, but it is not yet a cross-model TDL@90 certificate, a sealed-test result, or an equivalence claim.

### Non-negotiable redesign decisions

- **Retire MFS-v2/v3/v4 as public headline metrics.** Preserve them as historical/internal certification experiments.
- **Do not blend utility, feature importance, fidelity, privacy, and compactness into one opaque weighted score.** Use a constrained optimization problem and a transparent tuple.
- **Make exact bytes the primary rate axis.** Encoded scalar count, parameter count, and bits per original cell are diagnostics, not substitutes.
- **Require transfer across model families**, including modern tabular foundation models and strong task-trained baselines.
- **Measure driver behavior on untouched real data**, calibrated against real-versus-real instability.
- **Treat privacy, copying, schema validity, and artifact closure as hard gates.** Never let utility compensate for a failed safety or evidence requirement.
- **Keep the full Pareto frontier.** TDL@80, TDL@90, and TDL@95 are readable operating points, not replacements for the curve.
- **Separate scientific size from deployment size.** Report dataset-specific state, sampling closure, and cold portable bundle.
- **Preserve raw logical types.** Do not present HMAC-bucketed categories as continuous quantities to foundation-model auditors.
- **Replace the root README with a product/research landing page.** Internal campaign commands and audit archaeology move out of the primary path.
- **Run the official test only after every adapter, byte rule, model checkpoint, tuning rule, and figure script is frozen and hash-committed.**

---

## 1. What was reviewed

The redesign is based on a close audit of the current public repository, including:

- Root `README.md`, especially the benchmark table, artifact description, install flow, MFS-v4 section, command reference, and limitations.
- `docs/whitepaper/dope-mfs.tex`, its abstract, first-page figure, method, protocol, controls, baseline comparisons, limitations, and conclusion.
- `docs/whitepaper/README.md`, `REVIEW_GAPS.md`, `BASELINE_GAP.md`, `SELF_REVIEW.md`, `IEEE_READINESS.md`, `COMPREHENSION_v1.md`, and `DECISIONS_PENDING.md`.
- `production/kpi-contract-v4.json`.
- `rust/fitness.rs`, `rust/fitness/v4.rs`, `rust/transfer_metrics/mfs_v4.rs`, `rust/representation.rs`, and related Python mirror code.
- `research/benchmark/mfs_v3_audit.py`, `mfs_v3_panel.py`, `mfs_v3_score.py`, `representation.py`, and current benchmark evidence structure.
- Rust compiler, codec, schema, typed-data, artifact, sampling, release, and library boundaries.
- `Cargo.toml`, `pyproject.toml`, `Justfile`, and the current Rust/Python CLI split.
- Current open-source generator and auditor landscape, with preference for official repositories, official documentation, and primary papers.

The repository has unusually strong evidence discipline: hash-pinned ledgers, generated paper numbers, fail-closed conditions, explicit missing cells, and honest limitations. Those strengths should be kept. The redesign is about making that rigor serve one clean scientific question instead of becoming the visible product.

---

## 2. Diagnosis: why the project currently feels confusing

### 2.1 The strongest idea is buried beneath the certification machinery

The README begins with the right conceptual ingredients—small sampling program, exact byte charge, and train-on-synthetic/test-on-real retention—but quickly expands into multiple artifact profiles, MFS generations, prerequisites, gates, auditor pins, compatibility versions, campaign commands, and internal conformance terminology.

A new reader has to understand all of the following before reaching a stable mental model:

- DOPE the generator.
- DPK versions and compatibility rules.
- L0/L1/L2/L3 tiers.
- MFS-v2, MFS-v3, and MFS-v4.
- Jankurai.
- validation versus official test.
- compile versus certify.
- artifact bytes versus runtime closure.
- current paper versus historical evidence blocks.
- Rust v3 product versus Python v1 compatibility.
- Kumo representation distance versus CatBoost headline utility.

The result is not merely “too much detail.” It is a failure of hierarchy. A world-class project may contain all this machinery, but the first screen should communicate one question, one unit, one figure, and one reproducible command.

### 2.2 MFS-v4 is a dense certification bundle, not a convincing scientific ruler

The current MFS-v4 contract combines seven terms through a weighted geometric mean:

- utility transfer,
- driver fidelity,
- representation closeness,
- knowledge compression,
- distribution fidelity,
- structure fidelity,
- coverage realism.

It also includes six prerequisites and numerous hard release gates. This is defensible as a conservative internal release decision, but weak as the headline scientific metric for several reasons:

1. **The weights are conventional.** A reviewer can reasonably ask why transfer is 0.30, driver fidelity 0.20, representation closeness 0.15, and so on.
2. **Correlated compression ideas are counted multiple times.** Artifact bytes, bits per original cell, encoded-number count, and log-byte score are not independent scientific dimensions.
3. **Bits per original cell changes when the source table has more rows even if the artifact is identical.** That can be useful as a descriptive compression ratio but should not change the intrinsic quality of the artifact.
4. **The 10,240-byte discontinuity is a product tier, not a law of nature.** It should appear as a visible policy line on the frontier, not inside the universal metric.
5. **Encoded-number count is representation-dependent.** One method may store a codebook, another packed integers, another a neural tensor. Exact canonical bytes are the common unit.
6. **Kumo embedding distance creates a model-specific representation term.** Kumo is valuable as an auditor, but one hidden representation should not define the public meaning of synthetic quality.
7. **A weighted score hides failure modes.** Two artifacts can receive similar totals while having very different transfer, driver, and fidelity behavior.
8. **The score is currently null for every method.** A public flagship number that cannot yet be printed does not help readers understand the measured contribution.

The fail-closed philosophy is correct. The scalarization is not the right public abstraction.

### 2.3 The paper and README describe different generations of the project

The paper currently centers MFS-v3 and CatBoost/linear/MLP transfer, while the README introduces MFS-v4 and a stronger foundation-model-oriented contract. The code contains Kumo, Mitra-v2, and TabICL2 auditor paths, but the main published evidence still centers the older auditor panel.

This creates three versions of “what DOPE is”:

- the measured paper,
- the forward-looking README contract,
- the broader compiler/campaign platform in the source tree.

The redesign must choose one scientific identity and make every layer subordinate to it.

### 2.4 The benchmark is careful but not yet broad enough for the strongest claim

The current primary panel is regression-only, fit views are capped, and a majority of lineages come from a small number of simulated source families. Strong baselines are partially measured, measured under different coverage, or still pending. The official test is deliberately sealed.

This is acceptable for development evidence. It is not sufficient for a definitive claim about the best general-purpose synthetic-data methods or model-independent transfer.

### 2.5 The repository reads like an internal evidence archive

The project preserves extensive source audits, campaign contracts, evidence gap ledgers, completion estimates, worker instructions, historical reviews, and compatibility notes. This is excellent for forensics. In the public navigation path, however, it creates the impression of unresolved process rather than a polished open-source system.

The solution is not to delete history. The solution is to separate:

- **product code,**
- **benchmark code,**
- **current frozen evidence,**
- **historical evidence,**
- **internal operational records.**

### 2.6 The current interface exposes implementation history

The Rust CLI is `dope-kernel`, the Python compatibility command is `dope-kernel-v1`, and the Python and Rust packages share naming but not product identity. The root README asks users to understand this split immediately.

A public project should expose one command and one Python import. Compatibility code may remain, but it should not shape the user experience.

### 2.7 The byte claim needs a clearer closure contract

The repository distinguishes a very small dataset-specific artifact from a much larger generic runtime closure. That distinction is scientifically meaningful, but the current messaging allows readers to confuse:

- bytes learned from one dataset,
- bytes required by the method generally,
- bytes required to run on a clean machine.

All three should be measured and named.

---

## 3. North-star definition

### 3.1 One-sentence project definition

> **DOPE compiles a table into the smallest generative program whose samples preserve learning behavior on unseen real data.**

### 3.2 Plain-language question

> **How many bytes are needed to preserve 90% of what strong models learn from the real table?**

### 3.3 Scientific framing

DOPE should be framed as an **operational rate–distortion problem for supervised tables**.

- **Rate:** exact charged description length of the dataset-specific generative state, plus separately reported closure sizes.
- **Distortion:** loss of downstream transfer and driver behavior when learners receive synthetic data instead of the real fit table.
- **Constraints:** no exact/near copies, declared privacy status, valid schema, complete artifact closure, frozen evidence, and no official-test leakage.

This is related to minimum-description and rate–distortion thinking, but DOPE should avoid claiming a universal Shannon rate, Kolmogorov complexity, or formal MDL code unless it actually implements the corresponding coding assumptions. The safe term is **operational transfer description length**.

### 3.4 Public metric name

**Transfer Description Length**, abbreviated **TDL**.

The most readable operating point is:

**TDL@90** — minimum charged bytes needed to preserve at least 90% of real-data learning under the frozen core auditor suite, while all required driver, copy, privacy, and evidence gates pass.

Additional operating points:

- `TDL@80`: useful low-rate regime.
- `TDL@90`: primary README and paper scalar.
- `TDL@95`: high-fidelity regime.
- `TDL@99`: optional, often unresolved or infinite.

The complete result remains the **transfer–description frontier**.

### 3.5 The public result must be a tuple, not a magic number

Every published method/dataset result should expose:

```text
TDL@90 state bytes:        1,944 B
robust transfer LCB:       0.903
robust driver retention:   0.842
copy gate:                 PASS
empirical privacy:         PASS / FAIL / NOT EVALUATED
formal DP:                 none / (epsilon, delta)
sampling closure:          72.1 MB
cold portable bundle:      83.4 MB
scope:                     regression, mixed-type, core-2026.10 auditors
status:                    development / sealed-test / certified
```

The README may display only the first four lines plus status, but the machine-readable report must contain all fields.

---

## 4. Formal metric specification

### 4.1 Objects and splits

For dataset `d`, define:

- `D_fit`: visible to the generator compiler and real-trained auditor.
- `D_select`: visible only to configuration selection and budget selection.
- `D_test`: sealed until all methods, hashes, and analysis rules are frozen.

A compiler `C` receives `D_fit`, a task declaration, and a byte budget `B`, and produces artifact `A`.

A sampler `S` receives only:

- `A`,
- a fixed decoder/runtime identified by hash,
- requested row count `N`,
- sample seed `s`.

It emits a synthetic table `D_syn(A, N, s)`.

An auditor `m` trains or conditions on either `D_fit` or `D_syn` and is evaluated on the same untouched `D_test`.

### 4.2 Null-normalized transfer

For a declared proper loss where lower is better:

- `L_null(d,m)`: loss of a frozen null predictor.
- `L_real(d,m)`: loss after training/conditioning auditor `m` on `D_fit`.
- `L_syn(d,m,A)`: loss after training/conditioning the same auditor on synthetic rows from `A`.

Define the real-data gain:

```math
g_real(d,m) = L_null(d,m) - L_real(d,m)
```

Define the synthetic-data gain:

```math
g_syn(d,m,A) = L_null(d,m) - L_syn(d,m,A)
```

Define raw transfer:

```math
T_raw(d,m,A) = g_syn(d,m,A) / g_real(d,m)
```

Interpretation:

- `1.0`: synthetic-trained auditor recovers all measured real-training improvement over the null.
- `0.9`: recovers 90%.
- `0.0`: no improvement over null.
- `<0`: worse than null.
- `>1`: synthetic training outperforms the measured real-training baseline for this run.

The report stores `T_raw` without clipping. Certification uses:

```math
T_cert = clamp(T_raw, 0, 1)
```

A value above one must never compensate for a failure by another auditor or gate.

### 4.3 Informative-task rule

Transfer is undefined when the real-trained auditor does not meaningfully beat the null. Define:

```math
g_real >= max(g_abs_min, g_rel_min * |L_null|)
```

Recommended defaults:

- `g_abs_min` selected per loss family from numerical precision and benchmark scale.
- `g_rel_min = 0.01` for continuity with the current protocol, then sensitivity analysis at `0`, `0.001`, `0.01`, `0.05`, and `0.10`.

If the task/auditor pair is uninformative:

- mark the cell `uninformative`,
- exclude it from that auditor’s paired estimate,
- do not assign zero,
- do not treat it as a method failure or win.

### 4.4 Seed uncertainty

For each generator configuration:

- five fit seeds where the method is stochastic,
- three sample seeds per fitted artifact,
- declared training seeds for task-trained auditors,
- deterministic in-context auditors with a frozen inference seed or deterministic mode.

For every dataset/auditor/artifact budget, compute a one-sided 95% lower confidence bound:

```math
T_LCB(d,m,A)
```

Recommended hierarchy:

1. sample seed within fit seed,
2. fit seed within dataset,
3. optional auditor seed within auditor.

For certification, use the lower bound, never the mean or best seed.

### 4.5 Model-family aggregation

A single model can be gamed. A flat average can hide catastrophic transfer for one model family. Freeze auditor families:

1. **Tabular foundation models**
2. **Boosted trees**
3. **Task-trained neural tabular models**
4. **Linear/generalized linear sanity models**

For family `f`:

```math
T_family(d,f,A) = median_{m in f}(T_LCB(d,m,A))
```

For the core certificate:

```math
T_robust(d,A) = min_{required families f}(T_family(d,f,A))
```

This produces an understandable promise:

> The artifact preserves at least this much learning in every required model family.

Do not use a weighted mean as the gate. Also publish the complete per-auditor matrix.

### 4.6 Transfer Description Length

Let `R(A)` be the canonical charged size in bytes. Let `G(A)` be the conjunction of mandatory gates. Let `D_robust(A)` be the robust driver-fidelity measure.

For transfer threshold `τ` and driver threshold `δ`:

```math
TDL@τ(d) = min_A R(A)
           subject to
           T_robust(d,A) >= τ,
           D_robust(d,A) >= δ,
           G(d,A) = PASS.
```

Recommended primary thresholds:

- `τ = 0.90`
- `δ = 0.80`, calibrated against real-versus-real stability and revisited only through a versioned contract.

If no tested artifact satisfies the constraints:

```text
TDL@90 = unresolved (> largest tested budget)
```

Do not print zero, null-as-win, or the smallest failed artifact.

### 4.7 Benchmark-level aggregation

TDL is first defined per dataset. Across a benchmark, publish:

- pass coverage at each byte budget,
- median TDL among resolved datasets,
- geometric mean TDL among resolved datasets,
- 25th/75th/90th percentiles,
- unresolved fraction,
- domain-stratified distributions,
- task-type-stratified distributions,
- empirical cumulative distribution: fraction of datasets certified by `x` bytes.

Never report only the median among successes. The unresolved fraction is part of the result.

### 4.8 Frontier dominance

For each method, retain all non-dominated configurations in `(rate, transfer)` space after gates.

Artifact `A1` dominates `A2` when:

```text
R(A1) <= R(A2)
T_robust(A1) >= T_robust(A2)
D_robust(A1) >= D_robust(A2)
```

with at least one strict inequality and equal gate status.

The principal paper figure is the frontier, not a ranking by one arbitrary weight vector.

### 4.9 Secondary efficiency measures

Keep these as diagnostics:

- `transfer_per_kib = max(0, T_raw) / (bytes / 1024)`
- `driver_per_kib`
- `stored_scalars`
- `program_nodes`
- `bits_per_source_cell`
- `state_to_raw_ratio`
- `state_to_zstd_ratio`
- compile wall time and energy
- sample throughput and peak memory

Do not use these as the primary ranking because ratios can explode near zero bytes and can reward an unusably weak artifact.

---

## 5. Exact byte and closure contract

### 5.1 Why one byte number is insufficient

A 1.9 KB dataset-specific state can still require a 72 MB runtime. A 42 MB neural checkpoint may be dataset-specific while its Python environment is much larger. Both facts matter, but they answer different questions.

Report three rates.

### 5.2 Rate A: dataset-specific state bytes

`state_bytes` is the primary scientific x-axis.

It includes every byte whose value depends on the fitted dataset or fitted configuration and that sampling reads or requires, including:

- learned weights,
- selected architecture identifiers,
- fitted transforms,
- category dictionaries/codebooks required for reconstruction,
- missingness state,
- fitted schema information,
- target encodings,
- noise models,
- quantization scales,
- permutations,
- calibration constants,
- method-specific fitted preprocessors,
- compressed source rows if the method stores them,
- external dataset-specific caches,
- metadata necessary to interpret generated columns.

It excludes only the frozen method implementation and generic runtime shared across every dataset for that exact method version.

### 5.3 Rate B: sampling closure bytes

`sampling_closure_bytes` includes everything needed to sample when the base operating system is present:

- dataset-specific state,
- fixed decoder binary or package,
- method-specific source/runtime,
- method-shared pretrained checkpoints,
- shared lookup tables,
- dynamic libraries not guaranteed by the declared base image.

If a method uses a pretrained generator checkpoint, that checkpoint is charged here even if it is shared across datasets.

### 5.4 Rate C: cold portable bundle bytes

`cold_bundle_bytes` is the size of a deterministic OCI image, archive, or portable executable bundle that samples successfully on a clean runner with no network.

This is a deployment measure. It must not replace `state_bytes`, but it prevents a tiny fitted file from being marketed as a fully standalone 1.9 KB product.

### 5.5 Optional amortized closure

For a collection of `K` datasets using the same method implementation:

```math
amortized_bytes(K) = state_bytes + shared_method_bytes / K
```

Publish this only as a sensitivity curve for `K ∈ {1, 10, 100, 1000}`. Do not use an unspecified amortization assumption.

### 5.6 Canonical packaging

Create a deterministic artifact package:

```text
artifact.dope/
  manifest.json
  state.bin
  schema.public.json
  decoder.lock.json
  LICENSES/
```

The canonical archive must have:

- sorted members,
- normalized paths,
- fixed timestamps,
- fixed permissions,
- no user/group names,
- no nondeterministic compression metadata,
- content hashes for every member,
- a hash of the decoder/runtime contract,
- explicit uncompressed and compressed lengths.

`state_bytes` is the sum specified by the contract, not the host filesystem allocation and not a human-readable report size.

### 5.7 Charge manifest

Proposed schema fields:

```json
{
  "format": "dope-charge-manifest",
  "version": 1,
  "method_id": "dope-1.0",
  "dataset_id": "...",
  "artifact_sha256": "...",
  "state_bytes": 1944,
  "sampling_closure_bytes": 72050710,
  "cold_bundle_bytes": 83411200,
  "shared_method_bytes": 72048766,
  "charged_members": [
    {"path": "state.bin", "bytes": 1422, "class": "dataset_state"},
    {"path": "schema.public.json", "bytes": 522, "class": "dataset_state"}
  ],
  "external_reads": [],
  "network_required": false,
  "verified_by_trace": true
}
```

### 5.8 Runtime trace verification

For every published artifact:

- sample inside a network-disabled sandbox,
- trace file opens with eBPF, `strace`, or platform equivalent,
- fail if sampling reads any undeclared path,
- fail if sampling opens source data,
- fail if sampling contacts the network,
- fail if environment variables contain hidden fitted state,
- bind every read file to the manifest digest.

### 5.9 Encoded values

The user-facing intuition that fewer stored values is profound should be retained, but made precise.

Report:

- `stored_scalar_count`: scalar numerical payloads after decoding the canonical state.
- `stored_symbol_count`: discrete symbols, branches, program opcodes, and codebook entries.
- `effective_payload_bits`: compressed payload bits excluding fixed framing.
- `program_node_count`: interpretable program operations.

These are method-specific diagnostics. Exact canonical bytes remain the cross-method rate.

### 5.10 Literal-storage controls

Every dataset must include:

- raw CSV/Parquet bytes,
- gzip bytes,
- zstd bytes,
- canonical quantized-table bytes,
- deduplicated row-store bytes,
- smallest exact-lossless archive,
- row subset/coreset frontier under the same byte budgets.

This makes the claim concrete:

> DOPE is not merely smaller than neural generators; it preserves more learning per bit than storing a compressed subset of the table.


---

## 6. Auditor suite: powerful, diverse, frozen, and license-aware

### 6.1 Principle

A synthetic artifact should not be rewarded because it happens to fit CatBoost’s inductive bias. It should transfer across multiple high-value learning systems.

The suite must contain:

- frontier tabular foundation models,
- conventional boosted trees,
- task-trained neural tabular models,
- linear/generalized linear sanity models.

No one model defines “truth.”

### 6.2 Versioned auditor profiles

Create versioned profiles rather than one permanently growing list.

#### `TDL-Core-2026.10`

Required for the principal certificate and chosen for reproducibility, open implementation, manageable licensing, and coverage:

| Family | Proposed auditor | Role |
|---|---|---|
| Foundation | NVIDIA KumoTabular Large or Medium | Frontier mixed-type in-context/structured model |
| Foundation | TabICLv2 | Open high-value classification/regression foundation auditor |
| Foundation | Mitra-v2, if weights and license are locally pinned | Independent foundation architecture and prior |
| Boosted tree | CatBoost | Strong categorical/native tree baseline and continuity with current paper |
| Boosted tree | LightGBM | Different boosting implementation and preprocessing sensitivity |
| Boosted tree | XGBoost | Widely used gradient-boosting control |
| Neural | RealMLP or TabM | Strong task-trained neural tabular control |
| Linear | Elastic net / logistic regression | Low-capacity structural sanity check |

The exact final roster must be frozen only after adapter qualification and license review.

#### `TDL-Frontier-2026.10`

Research-only extension:

- TabPFN 3.5,
- TabFM,
- EXAONE Tabular,
- additional Mitra-v2 configurations,
- Kumo sizes Small/Medium/Large,
- TabICLv2 scale variants,
- other current open models that pass source, weight, and license review.

The frontier suite may contain non-commercial or research-only checkpoints, but the report must label that status. A method must not be called commercially reproducible merely because its source code is open.

### 6.3 Why core and frontier must be separate

Weights, licenses, hardware requirements, and model limits differ. Mixing every available foundation model into one mandatory certificate could make the metric impossible for external users to reproduce.

The core profile should be:

- available offline,
- hash-pinnable,
- runnable on documented hardware,
- legally redistributable or independently downloadable under clear terms,
- stable for at least one benchmark release.

The frontier profile exists to test whether the conclusion survives the strongest current models.

### 6.4 Exact model locks

For every auditor, record:

```yaml
id: kumo_tabular_large
family: foundation
source_repository: NVIDIA/structured-data-models
source_commit: <sha>
package_lock_sha256: <sha256>
checkpoint_files:
  - path: ...
    sha256: ...
license_code: ...
license_weights: ...
commercial_use: allowed|restricted|unknown
precision: fp16
inference_device: cuda
context_limit: ...
feature_limit: ...
preprocessing_id: typed-v1
prediction_seed: 0
```

No benchmark run may silently adopt a newer default model.

### 6.5 Foundation-model contamination and prior disclosure

For each auditor, record whether pretraining used:

- only synthetic tables,
- public real tables,
- unknown/proprietary data,
- benchmark datasets or close derivatives.

Use three labels:

- `clean-by-construction`: declared synthetic-only prior and no benchmark overlap.
- `screened`: real-data pretraining but benchmark overlap checked.
- `unknown`: no complete disclosure.

Do not exclude useful models automatically, but stratify results and never imply contamination-free evidence when it is unknown.

### 6.6 Typed data must reach typed auditors

The Rust code already contains a typed schema with numerical, integer, ordinal, nominal, Boolean, timestamp, and role information. Promote this path.

For the new protocol:

- categoricals remain categoricals,
- ordinals remain ordered,
- booleans remain booleans,
- missing states remain distinct from unknown categories,
- timestamps are supported or explicitly excluded by profile,
- target type is explicit,
- source-derived transforms are fit on `D_fit` only.

Do not use a HMAC bucket mapped to one floating-point scalar as the primary input representation for categorical foundation-model auditors. HMAC identifiers are useful for privacy-preserving names; they are not a semantic numeric encoding.

### 6.7 Preprocessing contract

Each auditor receives one of two frozen paths:

1. **Native typed path** — the auditor receives the declared logical types.
2. **Shared conventional path** — a training-fit-only numerical transformation for models requiring matrices.

The benchmark records both. No method receives a preprocessing advantage selected on test.

### 6.8 Row and feature scale tracks

Foundation models have context and feature limits. Create two benchmark tracks.

#### Small-context track

- all core auditors participate,
- fixed maximum fit rows compatible with every required foundation model,
- ideal for apples-to-apples cross-model TDL.

#### Scalable track

- boosted trees, task-trained neural models, and foundation models that support larger tables,
- datasets up to the declared scale limit,
- separate certificate and figure.

Do not silently coreset a large dataset for one auditor and call it the same task. Any coreset is part of that auditor’s declared protocol.

### 6.9 Losses

Use proper or task-appropriate losses.

#### Regression

Primary:

- normalized MSE or RMSE in training-derived target scale.

Secondary:

- MAE,
- pinball loss for quantile tasks,
- raw-unit RMSE for interpretability.

#### Binary classification

Primary:

- log loss.

Secondary:

- Brier score,
- AUROC,
- AUPRC for imbalanced tasks,
- calibration error.

#### Multiclass

Primary:

- multiclass log loss.

Secondary:

- balanced accuracy,
- macro F1,
- calibration.

The transfer ratio uses the primary proper loss. Rank metrics remain descriptive because they do not measure calibrated predictive information.

### 6.10 Controls per auditor

Every auditor must be scored on:

- null predictor,
- real fit data,
- real rows resampled to each synthetic size,
- compressed row subset at matched byte budget,
- predictor-only synthetic control,
- independent-marginals control,
- feature-only generator with target permutation,
- DOPE and all generator baselines.

These controls answer distinct questions:

- Is the task learnable?
- Does extra row count alone help?
- Could a tiny fitted predictor explain the result?
- Does joint feature structure matter?
- Does the generator retain target dependence rather than marginal appearance?
- Does a program beat simply storing rows?

---

## 7. Driver fidelity: preserve what powerful models rely on

### 7.1 Goal

A synthetic dataset can preserve aggregate predictive performance while teaching a different or spurious rule. DOPE should require that synthetic-trained models rely on similar stable drivers when evaluated on unseen real rows.

The current feature-importance work is directionally correct, but the new metric must address:

- correlated-feature substitution,
- unstable importance ranks,
- model-specific attribution scales,
- foundation models without native SHAP support,
- nonlinear effect shapes,
- differences between prediction agreement and driver agreement.

### 7.2 Evaluate attributions on the real test set

For each auditor `m`, fit/condition two instances:

- `m_real` on `D_fit`,
- `m_syn` on synthetic rows.

Evaluate both on exactly the same `D_test` rows.

Attributions must explain real-world predictions, not synthetic rows.

### 7.3 Use model-agnostic grouped conditional permutation importance

Primary importance method:

1. Group mutually substitutable features using training-only dependence estimates.
2. Perturb one group conditionally on the remaining groups, using a frozen conditional resampler or matched-neighbor procedure.
3. Measure loss increase on `D_test`.
4. Repeat across perturbation seeds and test bootstraps.

Why grouped conditional permutation:

- works for foundation models and conventional models,
- evaluates downstream behavior rather than internal representation,
- reduces the false penalty when two correlated features swap rank,
- uses a common meaning across model families.

Native SHAP, integrated gradients, and model coefficients may be published as sensitivity analyses, not the primary cross-model contract.

### 7.4 Real-versus-real stability ceiling

Driver importance is not perfectly stable even when both models are trained on real data.

Create two matched real controls:

- `m_real_a` trained on a bootstrap or disjoint fit view,
- `m_real_b` trained on another matched real view.

Compute raw attribution agreement:

```math
F_syn = agreement(m_real, m_syn)
F_rr  = agreement(m_real_a, m_real_b)
```

Normalize:

```math
DriverRetention = clamp(F_syn / max(F_rr, epsilon), 0, 1)
```

If `F_rr` is below a preregistered stability floor, mark the driver cell `uninformative`. Do not punish synthetic data for a driver ranking that is not reproducible on real data.

### 7.5 Feature grouping

Build groups from `D_fit` only using a deterministic graph:

- absolute Spearman or distance correlation for numeric pairs,
- Cramér’s V or mutual information for categorical pairs,
- mixed-type mutual information for mixed pairs,
- threshold and linkage rules frozen in the protocol.

Report both:

- group-level driver fidelity as the primary metric,
- feature-level fidelity as a secondary diagnostic.

### 7.6 Driver components

For stable groups, compute:

#### A. Importance-mass agreement

Normalize nonnegative importance vectors to distributions `p_real` and `p_syn`.

Recommended score:

```math
M = 1 - JS(p_real, p_syn) / ln(2)
```

This asks whether predictive importance mass is distributed similarly.

#### B. Rank agreement

Use:

- NDCG@k,
- rank-biased overlap or weighted Spearman,
- top-k recall of real drivers.

Recommended `k`:

```text
max(1, min(number of stable groups, ceil(0.10 * groups), 10))
```

Print `k` explicitly.

#### C. Direction and effect-shape agreement

For each stable high-importance group:

- estimate accumulated local effects or a model-agnostic partial dependence curve on `D_test`,
- normalize y-range by real effect magnitude,
- compare curves using normalized integrated absolute error,
- compare monotonic direction only when the real effect’s confidence interval excludes zero over a declared region.

Do not require a nonzero sign for every active feature. That rule is too brittle for interactions and non-monotonic effects.

#### D. Interaction agreement

Secondary in v1, promotable later:

- top interaction pairs by conditional joint permutation or H-statistic,
- overlap and strength agreement,
- synthetic stress tasks with known interactions.

### 7.7 Driver certificate

For each auditor:

```text
importance_mass >= 0.80
rank_agreement  >= 0.80
effect_shape    >= 0.70
real_real_stability >= 0.70
```

These are proposed starting values. Freeze them only after calibration on real-versus-real controls, not after looking at DOPE versus competitors.

Define auditor driver score as the minimum normalized component. Define family score as the median, and robust driver score as the minimum family score.

No weighted sum is required for the pass decision.

### 7.8 Prediction-behavior agreement

Driver fidelity is not prediction identity. Publish both.

Regression:

- prediction correlation,
- normalized prediction RMSE between `m_real` and `m_syn`,
- subgroup residual agreement.

Classification:

- probability Jensen–Shannon agreement,
- decision agreement,
- calibration-curve distance,
- confusion-pattern agreement.

These are diagnostics and can help explain why equal utility has different drivers.

### 7.9 Driver output schema

```json
{
  "auditor": "kumo_tabular_large",
  "real_real_stability": 0.91,
  "importance_mass_agreement": 0.87,
  "ndcg_at_k": 0.92,
  "top_k_recall": 0.80,
  "effect_shape_agreement": 0.84,
  "interaction_agreement": null,
  "driver_retention": 0.87,
  "informative": true,
  "test_split_hash": "..."
}
```

---

## 8. Distribution, representation, and fidelity metrics

### 8.1 Demote them from universal scalar terms

Distributional fidelity remains important, but it is not interchangeable with downstream transfer. Two methods can have similar marginal fidelity and very different utility; a target-aware method may intentionally spend bits on predictive structure rather than reproducing every irrelevant detail.

Therefore:

- do not blend distribution fidelity into TDL,
- retain it as a diagnostic panel,
- require only task-critical validity gates unless the release profile explicitly promises general-purpose synthesis.

### 8.2 Diagnostic fidelity panel

Recommended metrics:

- numerical marginal Wasserstein/KS distance,
- categorical total variation,
- missingness-rate error,
- pairwise association fidelity by type,
- higher-order dependency tests,
- classifier two-sample test with grouped folds,
- alpha-precision and beta-recall,
- nearest-neighbor distance distributions,
- support violation rate,
- logical constraint violation rate,
- rare-category coverage,
- tail and extreme-value fidelity.

### 8.3 Task-conditional fidelity

Add metrics that focus on what affects the target:

- conditional target distribution within stable driver bins,
- residual distribution conditional on prediction quantile,
- calibration of synthetic-trained models on real test,
- subgroup transfer by target and driver strata.

### 8.4 Representation metrics

Kumo/Mitra/TabICL embeddings may be useful exploratory diagnostics, but they should not define the public score.

If retained:

- report each encoder separately,
- calibrate against matched real-versus-real and independent-marginal nulls,
- do not use one encoder as a hard universal gate,
- disclose extraction layer and vector pooling rule,
- freeze checkpoint and implementation hash.

---

## 9. Privacy, copying, and release safety

### 9.1 Privacy is not a soft score

Privacy must never be traded against utility or compression. Publish explicit statuses.

```text
formal_dp: none | (epsilon, delta, accountant, adjacency)
empirical_sample_attacks: pass | fail | not_evaluated
artifact_attacks: pass | fail | not_evaluated
copy_gate: pass | fail
```

### 9.2 Exact and near-copy gates

Retain and strengthen the existing fail-closed approach:

- zero exact row matches unless explicitly allowed by a non-private benchmark profile,
- nearest-neighbor distances calibrated against distinct real-to-real distances,
- mixed-type distance with training-only scaling,
- duplicate-aware grouping,
- separate feature-only and full-row copy checks,
- check every sample seed and fit seed,
- check artifacts for cleartext source values, column names, rare category strings, and serialized rows.

### 9.3 Empirical attacks

Minimum empirical panel:

- membership inference against samples,
- shadow-model membership inference,
- attribute inference,
- nearest-neighbor disclosure,
- property inference,
- canary exposure,
- artifact inversion/reconstruction,
- target leakage and sensitive-identifier reconstruction.

Report attacker access assumptions:

- black-box samples only,
- unlimited sampling,
- artifact bytes available,
- source schema known or unknown,
- auxiliary population data available.

### 9.4 Formal DP track

Formal differential privacy must be a separate benchmark track with:

- explicit adjacency definition,
- `(epsilon, delta)`,
- accountant implementation and hash,
- clipping/noise parameters,
- composition across training and selection,
- privacy cost of hyperparameter tuning,
- no “privacy” claim from attack failure alone.

Include AIM, MST/Private-PGM, and other reproducible DP baselines in this track.

### 9.5 TDL certification and privacy

Offer two named certificates:

- `TDL-Utility`: copy gate and empirical attack panel required; no formal DP promise.
- `TDL-DP(epsilon,delta)`: all utility requirements plus formal DP.

Do not call the first “private synthetic data.” Call it “empirically audited synthetic data.”

---

## 10. Why compression is a complement to intelligence

The project can make a profound claim without overreaching:

> A compact artifact that teaches many strong learners on unseen real data has captured reusable regularity rather than merely reproducing records.

The evidence comes from the conjunction of:

- very small exact dataset-specific state,
- cross-model transfer,
- stable driver behavior,
- absence of row copies,
- performance on unseen real rows,
- stress tests that defeat memorization and target-only shortcuts.

Do not claim that TDL is a universal measure of intelligence. It is a workload-relative measure of **how compactly predictive regularities are encoded**.

This distinction is both scientifically safer and more compelling.


---

## 11. Benchmark redesign

### 11.1 Benchmark name

Use a distinct benchmark identity:

**DOPE-Bench: Transfer–Description Evaluation for Synthetic Tables**

The generator remains DOPE. The benchmark must not be named as though it exists only to prove DOPE wins.

### 11.2 Benchmark tracks

#### Track A — compact regression

Purpose:

- continuity with current work,
- strongest near-term paper,
- full foundation-model panel,
- exact TDL frontier.

Recommended primary set:

- 30–40 real-world regression datasets,
- diverse domains,
- mixed numerical/categorical features,
- at least six temporal/group/site-shift datasets,
- no more than 20% from one source portal or benchmark family,
- no simulated datasets in the headline aggregate.

The current PMLB panel remains a legacy robustness appendix and stress resource.

#### Track B — classification

- 20 binary datasets,
- 10 multiclass datasets,
- class imbalance strata,
- probability calibration as a required diagnostic.

#### Track C — scalable tables

- larger row counts and feature counts,
- excludes auditors that cannot support the declared scale,
- separately versioned profile.

#### Track D — privacy

- sensitive or privacy-relevant public datasets with legally permitted use,
- formal-DP methods and attack panel,
- no accidental implication that utility-only artifacts are private.

#### Track E — controlled science

Synthetic or semi-synthetic tasks with known ground truth:

- XOR and parity,
- sparse linear rules,
- nonlinear interactions,
- heteroscedastic noise,
- causal confounding,
- rare subgroup effect,
- categorical hierarchy,
- extrapolation,
- tail dependence,
- missing-not-at-random mechanisms,
- target leakage traps,
- duplicated-row traps.

This track explains *why* methods work. It is not allowed to dominate the real-world headline aggregate.

### 11.3 Dataset selection requirements

Every dataset card must specify:

- stable dataset ID and version,
- original source and license,
- task and exact target,
- row and column counts,
- logical types,
- missingness,
- domain,
- split type,
- group/time/site keys,
- known leakage risks,
- duplication audit,
- applicable auditor tracks,
- source and processed hashes,
- any legal redistribution limits.

### 11.4 Split design

Use three layers:

1. `fit`: generator fitting and real-trained auditor fitting.
2. `selection`: generator configuration and budget selection.
3. `test`: sealed final score.

For ordinary IID tasks:

- group-aware split where entity IDs exist,
- stratified split for classification,
- fixed split seed committed before benchmark execution.

For shift tasks:

- later time period,
- held-out site,
- held-out geography,
- held-out product or entity group,
- declared natural shift.

The strongest claim is transfer to unseen real conditions, not merely another random sample from the same rows.

### 11.5 Dataset scale bins

Publish stratified results for:

- rows: `200–2k`, `2k–20k`, `20k–200k`, `>200k` scalable track,
- features: `2–20`, `21–100`, `101–1000`, `>1000` scalable track,
- categorical share,
- missingness share,
- target signal strength,
- domain.

### 11.6 Synthetic row counts

Primary row counts:

- `n`: same number of rows as `D_fit`,
- `4n`: continuity and saturation test.

Secondary:

- `0.25n` and `0.5n` for low-sample efficiency,
- `16n` only as a saturation sensitivity.

TDL must not silently improve simply by sampling unlimited rows. Publish transfer as a function of both state bytes and generated row count. Recommended headline fixes `4n`; the paper also reports `n`.

### 11.7 Generated-row cost

Generation output rows are not part of the persistent artifact rate, because the artifact can generate them on demand. Still report:

- output CSV/Parquet bytes,
- generation time,
- energy,
- peak memory,
- rows per second.

For methods that *store* a synthetic dataset or coreset rather than a sampler, the stored synthetic rows are the artifact and are fully charged.

### 11.8 Task eligibility and denominators

Every table and figure must state:

- total datasets in registry,
- datasets applicable to method,
- successful fits,
- informative auditor cells,
- complete paired cells,
- certified cells,
- unresolved cells.

No denominator should be inferable only from supplementary notes.

### 11.9 Statistical analysis

Primary analysis:

- paired by dataset,
- one-sided lower confidence bounds for TDL qualification,
- cluster bootstrap by source family/domain when applicable,
- fit and sample seeds nested within dataset,
- explicit unresolved outcomes.

Method comparisons:

- paired Hodges–Lehmann differences for transfer at fixed budgets,
- paired differences in log2 TDL for datasets where both resolve,
- bootstrap interval on coverage difference at each budget,
- predeclared equivalence margins when claiming practical equivalence,
- familywise multiplicity control only for declared hypothesis families.

Never equate “not statistically different” with equivalence.

### 11.10 Sealed-test policy

Before test access:

- freeze dataset registry and split hashes,
- freeze every method adapter and source commit,
- freeze model checkpoints,
- freeze tuning spaces and budgets,
- freeze artifact charge code,
- freeze transfer and driver metric code,
- freeze missing-cell rules,
- freeze figure and table scripts,
- write all generated artifacts and sample hashes,
- sign the manifest.

Crash rule:

- one logical run,
- resume only the exact signed manifest,
- fill only missing cells,
- preserve every session boundary,
- never recompute a completed cell unless the entire sealed run is invalidated and publicly restarted under a new benchmark version.

---

## 12. Baseline matrix

### 12.1 Principles

The paper needs a baseline panel that a skeptical reviewer cannot dismiss as outdated or selectively weak.

Each baseline must have:

- official or author-approved implementation when available,
- pinned source and environment,
- explicit license,
- deterministic adapter contract,
- default and equal-budget tracks,
- exact artifact charge,
- timeout/failure ledger,
- no selective retries that erase failures.

### 12.2 Required generator families

#### Compact probabilistic and classical

- independent marginals,
- Gaussian copula,
- Chow–Liu tree,
- vine or sparse copula where reproducible,
- ARF default and selected,
- synthpop CART,
- Bayesian-network or graphical-model baseline.

#### GAN/VAE

- CTGAN,
- TVAE,
- CTAB-GAN+ if source/adaptation passes,
- one modern adversarial baseline where reproducible.

#### Diffusion and flow

- TabDDPM,
- TabSyn,
- TabDiff,
- CDTD,
- ForestDiffusion/Forest-Flow,
- GATD after source and paper qualification,
- one additional current mixed-type diffusion/flow model if it materially expands coverage.

#### Autoregressive/LLM

- GReaT,
- REaLTabFormer,
- TabularARGN or another strong open implementation.

#### Privacy-preserving

- AIM,
- MST/Private-PGM,
- a reproducible DP GAN/diffusion method if it passes source audit,
- non-DP counterpart for cost comparison.

#### Non-generative compact-learning controls

- predictor-only artifact,
- compressed real-row subset,
- stratified/random coreset,
- k-medoids or prototype set,
- label-aware coreset,
- tabular dataset-distillation method when a reproducible implementation is available,
- compressed exact table.

These controls are scientifically essential. They show whether a generative program is better than storing examples or a predictor.

### 12.3 Three evaluation tracks per method

#### A. Author default

Run the documented default without method-specific tuning, except adaptations required by dataset schema.

#### B. Equal search budget

Recommended starting envelope:

- maximum eight trials per method/dataset,
- fixed wall-clock and accelerator budget,
- every trial, timeout, OOM, and failure retained,
- selection only on `D_select`,
- method-native selection objective plus common transfer reported afterward.

A method may not use common test transfer to select its configuration.

#### C. Rate frontier

Obtain multiple artifacts across a declared size-control path:

- native width/depth/latent-size choices,
- quantization,
- pruning,
- low-rank factorization,
- reduced codebooks,
- fewer trees/components,
- budget-aware training where supported.

Do not post-compress a method in a way that changes sampling without documenting it as a configuration.

### 12.4 Budget grid

Recommended target budgets:

```text
512 B
1 KiB
2 KiB
4 KiB
8 KiB
16 KiB
32 KiB
64 KiB
256 KiB
1 MiB
4 MiB
16 MiB
64 MiB
unconstrained default
```

Methods need not hit each target exactly. Plot actual charged bytes.

### 12.5 Failure semantics

- timeout: missing cell, retained as timeout,
- OOM: missing cell, retained as OOM,
- invalid sample: failed gate,
- over budget: valid point at its actual size, not a zero score,
- unavailable license: excluded with explicit reason,
- partial auditor support: no core certificate; descriptive partial result only.

No missing or failed comparator cell counts as a DOPE win.

### 12.6 Literal compression baseline

For every dataset, include a visible point for:

- exact zstd-compressed training table,
- quantized zstd table at multiple precisions,
- compressed row subsets at matched byte budgets.

This establishes the intuitive lower-level reference: how much learning can be retained simply by storing data.

---

## 13. DOPE compiler redesign around the frontier

### 13.1 Compiler objective

The compiler should search for Pareto-optimal artifacts, not one weighted MFS score.

At compile time, optimize on `D_select`:

```math
maximize transfer and driver proxies
minimize exact state bytes
subject to compile budget and hard validity constraints.
```

The compiler emits a set of non-dominated candidates or one candidate for a requested byte budget.

### 13.2 User modes

```bash
dope compile train.parquet --target price --budget 2KiB -o price.dope

dope frontier train.parquet --target price \
  --budgets 512B,1KiB,2KiB,4KiB,8KiB,16KiB \
  --out-dir frontier/
```

### 13.3 Candidate families

Keep the current compact structure but expose the candidate program clearly:

- typed marginal encoders,
- missingness model,
- compact joint/dependence model,
- target conditional model,
- residual/noise model,
- optional interactions,
- quantization and entropy coding.

Candidate families should be compositional, each with:

- exact encoded cost before full training where possible,
- deterministic canonical serialization,
- declared expressive scope,
- validation transfer proxy,
- interpretability summary.

### 13.4 Explicit bit allocation

A profound extension is to make bit allocation visible:

```text
schema and type state       12%
marginals                   21%
dependence                  28%
target conditional mean     31%
noise and tails               6%
framing                       2%
```

The compiler should record marginal value of each block:

```math
Delta transfer / Delta byte
```

This supports ablation, interpretability, and rate allocation research.

### 13.5 Budget-aware search

Recommended search algorithm:

1. Generate simple candidate blocks.
2. Estimate exact serialized size.
3. Evaluate cheap transfer proxies on selection folds.
4. Maintain an epsilon-Pareto archive over `(bytes, proxy transfer, proxy driver)`.
5. Promote promising candidates to stronger auditors.
6. Refit selected candidates on `fit + selection` only after final configuration freeze, if the protocol allows.
7. Serialize and remeasure exact bytes.
8. Never select on sealed test.

### 13.6 Progressive auditor cascade

To keep compile cost manageable:

- Stage 0: linear and tree proxy.
- Stage 1: CatBoost/LightGBM.
- Stage 2: one foundation auditor on a selection fold.
- Stage 3: full auditor suite for benchmark evaluation only.

The public artifact is not allowed to encode or depend on auditor weights.

### 13.7 Compression mechanisms

Implement and ablate:

- scalar quantization with learned precision,
- shared scale/zero-point groups,
- entropy coding of sparse/discrete parameters,
- dictionary sharing across columns,
- low-rank dependence factors,
- sparse tree structures,
- piecewise-linear marginal maps,
- symbolic basis selection,
- residual pruning,
- mixed-precision bit allocation,
- canonical integer encoding,
- optional arithmetic/rANS coding.

A compressed byte stream must decode deterministically to the same sampling state.

### 13.8 Do not optimize stored scalar count alone

Fewer scalars are attractive, but one high-precision scalar can cost more than many low-precision symbols. The compiler may use scalar count as a structural regularizer, while exact encoded bits decide the rate.

### 13.9 Conditional target preservation

The target-aware design is DOPE’s differentiator. Strengthen it with:

- cross-fitting so residual models do not overfit fit rows,
- monotonic and interaction candidates,
- robust loss for heavy tails,
- calibrated conditional variance,
- target-transform selection,
- classification link functions,
- quantile or distributional heads when justified.

### 13.10 Joint feature model

The predictor-only control shows why the feature distribution matters. Improve candidate families for:

- mixed-type conditional dependencies,
- categorical interactions,
- tail dependence,
- missingness dependence,
- rare categories,
- constraints and logical rules.

Keep each family byte-accountable and ablatable.

### 13.11 Internal optimization metric versus public metric

The compiler may use cheap surrogates. The public benchmark must use the frozen TDL protocol.

Never call the internal search fitness “TDL” unless it is computed from the full required evidence.

---

## 14. Artifact format and public API

### 14.1 One public package identity

Recommended naming:

- PyPI package: `dope-synth` if `dope` is unavailable or ambiguous.
- Python import: `dope`.
- CLI: `dope`.
- Rust crates: internal names may remain `dope-core`, `dope-cli`, etc.

Remove `dope-kernel-v1` from the primary documentation. Preserve it under `legacy` or an explicit compatibility extra.

### 14.2 Python API

```python
from dope import Compiler, Artifact

compiler = Compiler(
    task="regression",
    target="price",
    budget_bytes=2048,
    random_state=11,
)
artifact = compiler.fit(table)
artifact.save("price.dope")

synthetic = artifact.sample(10_000, random_state=101)
report = artifact.inspect()
```

### 14.3 Evaluation API

```python
from dope_bench import evaluate

report = evaluate(
    real_fit="train.parquet",
    real_test="test.parquet",
    artifact="price.dope",
    profile="tdl-core-2026.10",
)
```

### 14.4 CLI

```bash
# Compile
dope compile train.parquet --target price --budget 2KiB -o price.dope

# Inspect exact charge and structure
dope inspect price.dope

# Sample offline
dope sample price.dope --rows 10000 --seed 101 -o synthetic.parquet

# Verify closure
dope verify price.dope --offline --trace-reads

# Evaluate development transfer
dope evaluate price.dope \
  --real-fit train.parquet \
  --real-test validation.parquet \
  --profile tdl-core-2026.10
```

### 14.5 `inspect` output

The first output should be human-readable:

```text
DOPE artifact
  format                   1.0
  task                     regression
  rows fitted              12,480
  features                 27
  state bytes              1,944
  stored scalars           63
  program nodes            117
  decoder                  dope-runtime 1.0 (sha256:...)
  external reads           none
  source rows embedded     no
  copy scan                not run
  certification            development only
```

JSON output is available with `--json`.

### 14.6 Rust/Python boundary

Recommended implementation:

- Rust owns parsing, schema, compiler, canonical codec, sampling, charge, and verification.
- Python uses PyO3/maturin bindings for ergonomic APIs and benchmark adapters.
- One canonical metric implementation in Rust, with a Python reference used only for parity tests.
- Avoid two independently evolving products.

### 14.7 Format versioning

Use semantic format versions:

- `dope-artifact/1.0`
- `dope-charge/1`
- `dope-tdl-evidence/1`
- `dope-tdl-report/1`

Compatibility rules:

- old artifacts remain readable where promised,
- new claims require current evidence version,
- reading an old format never upgrades its certification status.


---

## 15. Root README redesign

### 15.1 The first screen

The first screen should contain only:

```markdown
# DOPE — Downstream Objective-Preserving Encoding

**Compile a table into the smallest program that still teaches models.**

What is the minimum number of bytes needed for synthetic rows to preserve
what strong learners would have learned from the real table?

[frontier figure]

**Development result:** 1.9 KB at 0.898 CatBoost transfer versus 42.3 MB at
0.924 for the tested TabSyn schedule. The paired interval includes zero;
o equivalence or sealed-test claim is made. Cross-model TDL@90 is pending.
```

That is enough to make the project memorable.

### 15.2 Recommended README order

1. Hero and one-sentence definition.
2. Transfer–description figure.
3. Three headline facts and honest evidence status.
4. 60-second install and quick start.
5. “What TDL@90 means” with one equation and one example.
6. Compact benchmark table.
7. Artifact anatomy and byte charge.
8. How to reproduce current results.
9. Scope, privacy, and limitations.
10. Paper, documentation, benchmark registry, and contribution links.

Target length: roughly 250–450 lines. Everything else goes to docs.

### 15.3 Recommended status block

```markdown
> **Evidence status — development validation**
>
> The current public numbers use a training-derived validation split and the
> published CatBoost-centered panel. The official test remains sealed. The
> cross-model TDL contract, modern generator panel, driver certificate, and
> artifact attack panel are not yet complete. Missing evidence is never counted
> as a DOPE win.
```

This is clearer than explaining several null MFS generations in the hero.

### 15.4 Benchmark table redesign

The main table should show only comparable full-panel rows.

| Method | Transfer | State | TDL@90 | Driver | Status |
|---|---:|---:|---:|---:|---|
| DOPE | ... | ... | pending | pending | development |
| TabSyn | ... | ... | pending | pending | development |
| ARF | ... | ... | pending | pending | development |

Small-cohort or historical rows move to a second table with an explicit cohort label.

Do not mix 99-lineage, 21-lineage, and six-lineage rows as though the ranking were common.

### 15.5 One formula only

README formula:

```math
transfer = (null loss - synthetic-trained loss)
           / (null loss - real-trained loss)
```

Then:

```text
TDL@90 = minimum charged bytes that achieve at least 0.90 robust transfer
         and pass driver/copy/privacy gates.
```

No seven-term weight table belongs in the root README.

### 15.6 Documentation migration

Move material as follows:

| Current visible material | Destination |
|---|---|
| MFS-v2/v3/v4 history | `docs/legacy/mfs.md` |
| Full CLI command reference | `docs/reference/cli.md` |
| Artifact wire format | `docs/reference/artifact-format.md` |
| Tier/release contracts | `docs/reference/certification.md` |
| Benchmark campaign commands | `benchmarks/README.md` and `docs/research/` |
| Internal ownership/worktree rules | `CONTRIBUTING.md` and maintainer docs |
| Jankurai internals | `docs/maintainers/audit.md` |
| Evidence gap ledgers | `evidence/legacy/<release>/` |
| Paper build internals | `paper/REPRODUCING.md` |

### 15.7 Badges

Keep badges that help external users:

- CI,
- package version,
- license,
- paper/evidence release,
- reproducibility status.

Remove the internal Jankurai score badge from the hero. It is not meaningful to a first-time user.

### 15.8 README language rules

- Use “development validation” or “sealed test” on every result block.
- Use “state bytes” rather than “artifact is standalone” unless closure is included.
- Use “empirical privacy audit” rather than “private.”
- Use “paired interval contains zero” rather than “ties.”
- Use “unresolved” rather than “failed” when no configuration meets TDL.
- Never say “world’s best” or “state of the art” without a complete matched panel.
- Never describe a null metric as though it were measured.

---

## 16. Proposed README draft

The following is a working top section, not final copy.

```markdown
# DOPE — Downstream Objective-Preserving Encoding

**Compile a table into the smallest program that still teaches models.**

DOPE asks one question: *How many bytes are needed for synthetic rows to
preserve what strong learners would have learned from the real table?*

It compiles a supervised table into a compact generative program, samples new
rows from that program alone, trains powerful tabular models on those rows, and
tests them on untouched real data.

[transfer–description frontier]

### Current development result

On the current regression validation panel, DOPE stores a median **1.9 KB**
and reaches **0.898 CatBoost transfer** at `4n` synthetic rows. The tested
scaled TabSyn schedule stores **42.3 MB** and reaches **0.924**. The paired
DOPE-minus-TabSyn interval includes zero, so this is a compression result—not
a superiority or equivalence claim. The official test and cross-model TDL@90
certificate remain pending.

### The ruler: TDL@90

`TDL@90` is the minimum charged state needed to recover at least 90% of
real-data learning across the frozen auditor families, while preserving stable
model drivers and passing copy, privacy, and artifact-closure gates.

Smaller is better. If no tested artifact passes, TDL@90 is unresolved.

### Quick start

    pip install dope-synth
    dope compile train.parquet --target price --budget 2KiB -o price.dope
    dope inspect price.dope
    dope sample price.dope --rows 10000 -o synthetic.parquet
```

---

## 17. Paper redesign

### 17.1 Recommended title

Primary recommendation:

> **DOPE: Transfer Description Length for Synthetic Tables**

Subtitle or opening question:

> **How Few Bytes Can Preserve a Dataset’s Predictive Behavior?**

Alternative titles:

- “DOPE: Minimum-Description Synthetic Data for Cross-Model Transfer”
- “How Few Bytes Can Teach a Model? Generative Compression of Tabular Data”
- “Transfer–Description Frontiers for Synthetic Tabular Data”

Avoid leading with an internal release score or fixed 10,240-byte tier.

### 17.2 Paper thesis

The paper should make one claim:

> Synthetic data should be evaluated by the shortest independently usable description that preserves downstream behavior on unseen real data across model families.

DOPE is then one compiler that performs well under that ruler.

### 17.3 Contributions

Use four contributions only.

1. **Transfer Description Length.** A rate–distortion-style, model-family-aware metric for the minimum exact bytes that preserve downstream learning, with unresolved outcomes and hard validity gates.
2. **DOPE compiler.** A compact typed generative program with explicit bit allocation across marginals, dependencies, target structure, and noise.
3. **Cross-model benchmark.** Modern tabular foundation models, boosted trees, neural and linear auditors; current open generators; literal-storage and coreset controls; unseen real test splits.
4. **Executable evidence.** Deterministic charge manifests, offline read tracing, hash-pinned model and dataset registries, generated tables/figures, and one sealed test.

Privacy can be a fifth contribution only after the full attack panel or formal-DP track is complete.

### 17.4 Proposed abstract

> How many bits of a dataset are actually needed to teach a model? We introduce Transfer Description Length (TDL), an operational rate–distortion measure for synthetic tables. A generator is charged for every dataset-specific byte required to sample, and its output is judged by the fraction of real-data learning recovered by multiple frozen model families on untouched real rows. TDL@90 is the smallest artifact that recovers at least 90% of real-data gain while preserving stable model drivers and passing copy, privacy, and closure gates. We present DOPE, a compiler that encodes typed marginals, compact dependence structure, a target-conditional program, and residual noise into a byte-bounded sampler. We evaluate DOPE against classical, adversarial, diffusion, flow, autoregressive, coreset, and literal-storage baselines under matched splits and documented tuning budgets. [Insert sealed results only after completion.] The study exposes full transfer–description frontiers rather than a single weighted score and releases deterministic artifacts, byte manifests, benchmark adapters, and the complete reduction pipeline.

Until sealed results exist, the abstract must label all numbers as development validation and avoid “outperforms,” “equivalent,” or “certified.”

### 17.5 First-page figure

One plot. No multi-panel metric dashboard.

#### Axes

- x: exact dataset-specific state bytes, log scale, smaller is better.
- y: robust cross-model transfer lower bound, higher is better.

#### Reference lines

- horizontal `0.90` TDL threshold,
- vertical `10 KiB` product tier,
- optional vertical exact compressed-table median.

#### Marks

- one method family per marker shape,
- filled: driver/copy/privacy gates pass,
- hollow: utility measured but a gate is incomplete or failed,
- crossed: partial cohort,
- arrows: unresolved beyond plot range.

#### Caption

State cohort, row count, auditor profile, interval, and evidence status in fewer than 90 words.

### 17.6 Paper figure set

1. **Transfer–description frontier** — the main idea.
2. **TDL coverage curve** — fraction of datasets certified by byte budget.
3. **Cross-model transfer matrix** — auditors × methods, with lower bounds.
4. **Driver retention matrix** — importance/effect agreement by auditor.
5. **Bit-allocation ablation** — where DOPE spends bytes and marginal transfer per byte.
6. **Privacy/copy panel** — only after complete.
7. **Runtime/closure panel** — state versus closure and sample speed.

Move detailed distributions, dozens of tests, and legacy MFS tables to the supplement.

### 17.7 Main paper structure

1. Introduction: one question and one figure.
2. TDL: definitions, transfer, rate, constraints.
3. DOPE compiler: artifact and budget-aware search.
4. Benchmark: datasets, auditors, generators, controls, splits.
5. Results: frontier, TDL, cross-model transfer, drivers.
6. Analysis: bit allocation, controls, stress tests.
7. Privacy and closure.
8. Limitations and threats.
9. Conclusion.

### 17.8 Remove version soup from the main narrative

Do not make readers track MFS-v2/v3/v4. The paper may say:

> Earlier repository releases explored weighted certification scores. We replace them with a constrained transfer–description definition; historical contracts remain reproducible in the supplement.

### 17.9 Claims table

Before submission, maintain a generated table:

| Claim | Required evidence | Status |
|---|---|---|
| DOPE has lower TDL than TabSyn | matched sealed TDL cells, paired interval | pending |
| DOPE matches utility at much smaller state | predeclared equivalence + size ratio | pending |
| Cross-model preservation | all core families complete | pending |
| Driver preservation | stable driver panel complete | pending |
| Empirical privacy | attack panel complete | pending |
| Formal privacy | DP proof/accountant | not claimed |

The paper build must fail if prose contains a claim whose status is not satisfied.

---

## 18. Figure generation specification

### 18.1 README prototype

The attached prototype is intentionally based on current validation numbers and CatBoost only. It demonstrates the visual grammar, not the final scientific claim.

Final figure data must come only from a frozen machine-readable ledger.

### 18.2 Data schema

```json
{
  "method": "dope",
  "configuration": "...",
  "cohort": "regression-core-v1",
  "auditor_profile": "tdl-core-2026.10",
  "synthetic_size": "4n",
  "state_bytes_median": 1944,
  "transfer_lcb_median": 0.903,
  "transfer_interval": [0.87, 0.93],
  "driver_pass_rate": 0.91,
  "gate_status": "pass",
  "evidence_status": "sealed"
}
```

### 18.3 Final plot rules

- log x-axis with byte labels `512 B`, `1 KiB`, `10 KiB`, `1 MiB`, `1 GiB`.
- transfer y-axis begins below zero when negative methods exist; do not truncate failures.
- no unexplained colors.
- direct labels for the principal methods.
- cohort size on every partial mark.
- no connecting lines across incomparable cohorts.
- no confidence interval that combines missing methods as failures.
- vector PDF/SVG for paper; optimized PNG/WebP for README.
- accessible grayscale/shape encoding.

### 18.4 TDL coverage figure

For byte budget `b`:

```math
coverage(b) = resolved datasets with TDL@90 <= b / all eligible datasets
```

Plot step curves with bootstrap intervals. This is often more informative than a median because it includes unresolved datasets.

### 18.5 Bit allocation figure

For each DOPE artifact:

- state bytes by block,
- transfer loss when block removed,
- marginal transfer per byte.

Aggregate by dataset class and show examples. This figure can become one of the most novel parts of the paper.


---

## 19. Repository restructuring

### 19.1 Preserve history before moving anything

Before the cleanup:

1. Create a signed tag such as `evidence-legacy-2026.10`.
2. Generate a manifest of every evidence file, digest, current path, and future path.
3. Publish the current PDFs and immutable evidence bundle as a release asset.
4. Verify that all current numbers rebuild from the tagged tree.
5. Only then begin path migration.

### 19.2 Proposed top-level tree

```text
/
  README.md
  LICENSE
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  SECURITY.md
  GOVERNANCE.md
  ROADMAP.md
  CITATION.cff

  crates/
    dope-core/
    dope-cli/
    dope-python/
    dope-bench-core/

  python/
    dope/
    dope_bench/

  benchmarks/
    README.md
    registry/
      datasets-v1.yaml
      generators-v1.yaml
      auditors-2026.10.yaml
      environments.lock.json
    protocols/
      tdl-v1.yaml
      privacy-v1.yaml
    adapters/
      generators/
      auditors/
    metrics/
    figures/
    tests/

  docs/
    index.md
    quickstart.md
    concepts/
      transfer-description-length.md
      artifact-charge.md
      driver-fidelity.md
    reference/
      cli.md
      python-api.md
      artifact-format.md
      evidence-schema.md
    privacy.md
    reproducibility.md
    limitations.md
    legacy/
      mfs.md
      v1-python.md
    maintainers/
      audit.md
      release.md

  paper/
    main.tex
    supplement.tex
    figures/
    generated/
    scripts/
    REPRODUCING.md

  evidence/
    current/
      manifest.json
      reports/
      figures/
    legacy/
      2026-10-mfs/

  examples/
    quickstart-regression/
    mixed-type-classification/
    offline-verification/

  schemas/
    dope-artifact-v1.schema.json
    dope-charge-v1.schema.json
    tdl-evidence-v1.schema.json
    tdl-report-v1.schema.json
```

### 19.3 What stays out of the primary path

Move or archive:

- operational ETA memos,
- worker-specific capacity amendments,
- one-off source audit narratives,
- stale review snapshots,
- superseded paper classes,
- campaign queue instructions,
- worktree ownership details.

They may remain in `evidence/legacy` or the tagged historical branch.

### 19.4 Results storage

Do not keep an indefinitely growing universe of large campaign receipts on `main`.

Recommended structure:

- scalar, small, immutable benchmark summaries in Git,
- large artifacts/samples in versioned release assets or a content-addressed object store,
- manifest and hashes in Git,
- downloader verifies every byte,
- no paper number depends on an unpinned scratch path.

### 19.5 Public contribution surface

Add:

- `CONTRIBUTING.md` with one development setup,
- adapter template,
- dataset-card template,
- metric-change policy,
- benchmark versioning policy,
- issue templates for generator adapter, auditor adapter, dataset proposal, and reproducibility bug,
- `SECURITY.md` for artifact/parser vulnerabilities,
- governance and maintainer policy.

### 19.6 Documentation site

Generate a small docs site from Markdown. The navigation should be:

- What is DOPE?
- Quick start
- TDL metric
- Artifact and bytes
- Benchmark
- Privacy
- Reproduce the paper
- API reference
- Contribute

Do not expose the entire internal document tree as top-level navigation.

---

## 20. File-level implementation plan

### 20.1 P0 — freeze and separate the current evidence

#### Create

- `evidence/legacy/2026-10-mfs/manifest.json`
- `docs/legacy/mfs.md`
- `docs/status.md`

#### Modify

- `README.md`: label current numbers development-only and remove MFS-v4 from the hero path.
- `docs/whitepaper/README.md`: mark current paper as the historical validation paper once the new paper starts.
- `Justfile`: add `legacy-paper-check` and reserve `paper` for the new pipeline after migration.

#### Acceptance

- Current paper rebuilds byte-for-byte or with a documented deterministic-equivalence exception from the tag.
- Every moved file appears in the path mapping manifest.
- No current claim changes during cleanup.

### 20.2 P0 — implement TDL contract

#### Create

- `production/tdl-contract-v1.json`
- `schemas/tdl-evidence-v1.schema.json`
- `schemas/tdl-report-v1.schema.json`
- `rust/tdl.rs` or `rust/fitness/tdl.rs`
- `research/benchmark/tdl_score.py`
- `research/benchmark/tests/test_tdl.py`
- `tests/tdl_contract.rs`
- `fixtures/tdl/golden-v1.json`

#### Modify

- `rust/lib.rs`: export TDL types and evaluator.
- `rust/main.rs`: add development `evaluate-tdl` command only after schema is frozen.
- `ops/ci/check-contracts.sh`: validate contract digest and Rust/Python parity.

#### Required Rust structures

```rust
pub struct TransferObservation {
    pub null_loss: f64,
    pub real_loss: f64,
    pub synthetic_loss: f64,
    pub informative: bool,
}

pub struct AuditorTransfer {
    pub auditor_id: String,
    pub family_id: String,
    pub raw_transfer: f64,
    pub certified_transfer: f64,
    pub lower_bound: f64,
}

pub struct TdlCandidateReport {
    pub state_bytes: u64,
    pub closure_bytes: u64,
    pub cold_bundle_bytes: Option<u64>,
    pub family_transfer: BTreeMap<String, f64>,
    pub robust_transfer: Option<f64>,
    pub robust_driver: Option<f64>,
    pub gates: BTreeMap<String, GateStatus>,
    pub qualifies_at: Vec<f64>,
}
```

#### Acceptance

- Python and Rust produce bit-identical JSON for golden fixtures.
- uninformative cells remain null, not zero.
- transfer above one is preserved in raw output and capped only for certification.
- one family failure prevents qualification.
- missing required auditor prevents qualification.
- smaller bytes with identical metrics is always preferred.
- source row count does not change candidate quality unless it changes measured transfer or the artifact.

### 20.3 P0 — implement charge contract

#### Create

- `rust/artifact/charge.rs`
- `rust/artifact/manifest.rs`
- `research/benchmark/charge_artifact.py`
- `schemas/dope-charge-v1.schema.json`
- `tests/artifact_charge.rs`
- `tests/offline_closure.rs`

#### Modify

- `rust/codec.rs`: canonical member ordering and deterministic framing.
- `rust/inspect.rs`: print state, closure, and cold-bundle sizes separately.
- `rust/sample.rs`: optional read-trace verification mode.
- `scripts/install-native.sh`: emit decoder manifest and digest.

#### Acceptance

- same state serializes byte-identically on two supported systems.
- changing filename or directory does not change artifact hash.
- sampling fails if an undeclared external file is required.
- network-disabled sampling succeeds for qualified artifacts.
- trace contains no source-table reads.
- all benchmark methods emit the same charge schema.

### 20.4 P0 — typed auditor bridge

#### Create

- `benchmarks/adapters/auditors/base.py`
- `benchmarks/adapters/auditors/kumo.py`
- `benchmarks/adapters/auditors/tabiclv2.py`
- `benchmarks/adapters/auditors/mitra_v2.py`
- `benchmarks/adapters/auditors/catboost.py`
- `benchmarks/adapters/auditors/lightgbm.py`
- `benchmarks/adapters/auditors/xgboost.py`
- `benchmarks/adapters/auditors/realmlp.py`
- `benchmarks/adapters/auditors/linear.py`
- `benchmarks/registry/auditors-2026.10.yaml`

#### Modify

- `research/benchmark/representation.py`: remove scalar HMAC category coding from the primary typed path; retain only as a legacy privacy-safe identifier mapping.
- `rust/schema.rs`: complete declared support matrix or fail with explicit profile restrictions.
- `rust/typed_data.rs`: export Arrow/Parquet-compatible typed batches through bindings.

#### Adapter interface

```python
class AuditorAdapter(Protocol):
    id: str
    family: str

    def fit(self, context: TypedTable, seed: int) -> FittedAuditor: ...
    def predict(self, model: FittedAuditor, query: TypedTable) -> Prediction: ...
    def loss(self, prediction: Prediction, target: Series) -> float: ...
    def fingerprint(self) -> dict: ...
```

Foundation in-context models may implement `fit` as context preparation rather than gradient training.

#### Acceptance

- identical typed input reaches real and synthetic paths.
- preprocessing fits only on `D_fit`.
- auditor adapter records exact checkpoint and code hashes.
- output shapes and loss units are tested.
- categorical values are not silently treated as ordered floats.

### 20.5 P0 — driver-fidelity engine

#### Create

- `research/transfer/driver_fidelity.py`
- `research/transfer/conditional_permutation.py`
- `research/transfer/feature_groups.py`
- `research/transfer/effect_shapes.py`
- `research/transfer/tests/test_driver_fidelity.py`
- `schemas/driver-report-v1.schema.json`

Optional Rust core after the algorithm stabilizes:

- `rust/driver_metrics.rs`

#### Acceptance

- correlated duplicate features are grouped and do not falsely fail top-k overlap.
- unstable real-versus-real drivers become uninformative.
- a synthetic model using the wrong feature fails controlled tests despite equal aggregate accuracy.
- nonlinear sign-changing effects are evaluated by shape, not forced into one sign.
- all importance computations use real test rows.

### 20.6 P0 — benchmark registries

#### Create

- `benchmarks/registry/datasets-v1.yaml`
- `benchmarks/registry/generators-v1.yaml`
- `benchmarks/registry/auditors-2026.10.yaml`
- `benchmarks/protocols/tdl-v1.yaml`
- `benchmarks/protocols/privacy-v1.yaml`
- `benchmarks/registry/schema/*.json`

#### Acceptance

- every benchmark run resolves entirely from registries and content hashes.
- unknown method or dataset IDs fail closed.
- licenses and redistribution status are machine-readable.
- source-family and domain labels are available for clustered intervals.

### 20.7 P1 — generator adapters

One adapter directory per method:

```text
benchmarks/adapters/generators/tabdiff/
  adapter.py
  method.yaml
  environment.lock
  patches/
  tests/
```

Priority order:

1. existing current comparators and controls,
2. TabSyn, TabDDPM, ForestDiffusion/Flow, ARF,
3. TabDiff and CDTD,
4. GATD after qualification,
5. GReaT/REaLTabFormer/TabularARGN,
6. privacy track.

#### Acceptance per adapter

- official source or documented fork,
- source digest,
- environment lock,
- smoke fit/sample test,
- deterministic charge manifest,
- timeout handling,
- schema round-trip,
- no test access,
- default and tuning definitions,
- license status.

### 20.8 P1 — new paper pipeline

#### Create

- `paper/main.tex`
- `paper/supplement.tex`
- `paper/scripts/reduce.py`
- `paper/scripts/render.py`
- `paper/scripts/claims.py`
- `paper/generated/`
- `paper/REPRODUCING.md`

#### Modify

- `Justfile`: add `bench`, `paper`, `paper-check`, `sealed-freeze`, `sealed-score`.

#### Acceptance

- one command regenerates every number and figure from frozen summaries.
- paper prose claims are checked against the claim registry.
- no hand-typed benchmark number exists in TeX or README.
- README benchmark block and paper figure use the same ledger.

---

## 21. Proposed TDL contract

A compact human-readable version:

```json
{
  "format": "dope-tdl-contract",
  "version": 1,
  "primary_threshold": 0.90,
  "secondary_thresholds": [0.80, 0.95, 0.99],
  "transfer": {
    "formula": "(null_loss - synthetic_loss) / (null_loss - real_loss)",
    "raw_unclipped": true,
    "certification_clip": [0.0, 1.0],
    "uncertainty": "one_sided_95_lcb",
    "uninformative_rule": "real_gain_below_declared_floor"
  },
  "aggregation": {
    "within_family": "median_required_auditors",
    "across_required_families": "minimum",
    "missing_required_family": "ineligible"
  },
  "driver": {
    "threshold": 0.80,
    "real_real_calibrated": true,
    "primary_method": "grouped_conditional_permutation",
    "components": [
      "importance_mass_agreement",
      "rank_agreement",
      "effect_shape_agreement"
    ]
  },
  "rate": {
    "primary": "state_bytes",
    "also_report": [
      "sampling_closure_bytes",
      "cold_bundle_bytes",
      "stored_scalar_count",
      "program_node_count"
    ]
  },
  "hard_gates": [
    "canonical_artifact",
    "offline_sampling",
    "declared_reads_only",
    "schema_valid",
    "no_exact_copy",
    "near_copy_pass",
    "empirical_privacy_complete",
    "evidence_complete",
    "sealed_split_integrity"
  ]
}
```

Whether empirical privacy is mandatory for the first scientific TDL paper or only the release certificate should be decided before freezing v1. The recommendation is:

- mandatory copy/closure gates for TDL scientific measurement,
- privacy status always reported,
- complete empirical privacy required for a “release-certified” badge.

This prevents an incomplete attack campaign from blocking the core rate–transfer paper while avoiding any privacy overclaim.

---

## 22. Testing and CI specification

### 22.1 Metric properties

Property tests must prove:

- lower synthetic loss monotonically improves raw transfer.
- identical real and synthetic loss gives transfer `1`.
- synthetic equal to null gives `0`.
- worse than null gives negative raw transfer.
- real equal to null is uninformative, not divide-by-zero.
- raw transfer above one is retained but certificate-capped.
- a failed family cannot be compensated by another family.
- an absent required family makes TDL unresolved.
- a failed hard gate makes TDL unresolved.
- TDL is non-increasing as the tested candidate set gains smaller qualifying artifacts.
- TDL is never inferred between untested budgets without a declared interpolation rule.

### 22.2 Rate properties

- canonical serialization deterministic.
- no platform timestamps.
- exact byte counts match manifest.
- state bytes exclude only declared shared runtime.
- hidden checkpoint read causes failure.
- source-table read causes failure.
- archive path traversal is rejected.
- malformed/oversized fields are rejected before allocation.
- old artifacts cannot acquire new certification by being decoded with new software.

### 22.3 Benchmark integrity

- split hashes stable.
- generator never receives test path.
- selection script cannot import test loader.
- sample and artifact hashes match manifest.
- duplicate identities fail closed.
- unknown arms fail closed.
- partial cohorts never enter full-panel aggregate.
- missing comparator cells never count as losses.
- every figure point traces to report IDs and JSON pointers.

### 22.4 Driver tests

Controlled fixtures:

1. Same rule, rescaled feature — pass.
2. Same accuracy, wrong feature shortcut — fail.
3. Duplicate correlated features swapped — group-level pass.
4. Unstable noise features — mark uninformative.
5. Non-monotonic U-shaped effect — shape pass, simple sign not used.
6. Interaction-only target — interaction diagnostic identifies the pair.

### 22.5 CI lanes

#### Fast PR lane

- formatting and lint,
- Rust/Python unit tests,
- metric golden fixtures,
- one tiny artifact compile/sample/charge test,
- one auditor smoke test with no large weights.

#### Extended CPU lane

- all classical auditors,
- driver fixtures,
- deterministic paper reduction from small fixtures,
- closure trace.

#### Scheduled GPU lane

- foundation auditor adapter tests,
- exact checkpoint hash checks,
- small benchmark replay,
- memory/yield rules.

#### Release lane

- full registry validation,
- source/license lock,
- reproducible wheels and containers,
- signed artifacts,
- paper and README zero-diff regeneration.

---

## 23. Reproducibility contract

### 23.1 One-command development replay

```bash
just reproduce-development
```

It should:

- verify source and model hashes,
- download only declared public artifacts when absent,
- build the native binary and Python bindings,
- replay frozen scalar reductions,
- regenerate figures and paper,
- verify zero diff.

### 23.2 Full rerun command

```bash
just benchmark --profile tdl-core-2026.10 --cohort regression-v1
```

This may be expensive but must be structurally complete.

### 23.3 Offline evidence bundle

Publish:

- registry snapshots,
- environment locks,
- scalar reports,
- source and sample hashes,
- representative artifacts where legally allowed,
- no private source rows,
- exact commands,
- expected hardware range.

### 23.4 Build provenance

Use SLSA-style provenance or equivalent:

- source commit,
- builder identity,
- build command,
- dependency lock hashes,
- artifact digest,
- signature.

### 23.5 Paper number policy

All numbers must be generated from JSON/Parquet reports. A lint should reject:

- benchmark decimals typed directly in TeX/Markdown outside approved generated blocks,
- claims not present in the registry,
- missing evidence status,
- unsupported “best,” “equivalent,” “private,” or “certified” language.

---

## 24. Experiment campaign plan

### Phase 0 — freeze and simplify without changing science

Deliverables:

- historical evidence release/tag,
- rewritten root README structure,
- status page,
- MFS moved to legacy docs,
- current plot labeled development-only,
- one CLI story documented.

No new result claims.

### Phase 1 — metric and charge conformance

Deliverables:

- TDL contract v1,
- Rust/Python metric parity,
- canonical charge manifests,
- offline read tracing,
- literal-storage and row-subset controls,
- current DOPE/TabSyn/ARF results re-expressed as development frontier points.

Exit criterion:

- exact bytes and transfer are generated from one ledger,
- no MFS scalar is needed to understand the result.

### Phase 2 — modern auditor qualification

Deliverables:

- Kumo, TabICLv2, Mitra-v2 qualification,
- CatBoost/LightGBM/XGBoost/RealMLP/linear adapters,
- typed data bridge,
- model license and contamination registry,
- small pilot on 10 datasets.

Exit criterion:

- full core auditor matrix runs offline and deterministically enough for intervals.

### Phase 3 — driver fidelity

Deliverables:

- grouped conditional permutation engine,
- real-real calibration,
- effect-shape agreement,
- controlled tests,
- pilot driver reports.

Exit criterion:

- wrong-rule synthetic controls fail while correlated-feature substitutions do not fail spuriously.

### Phase 4 — baseline completion

Deliverables:

- matched default and tuned runs for mandatory generator panel,
- rate frontier configurations,
- timeout/failure ledger,
- exact charge for every method.

Exit criterion:

- no headline comparison relies on a partial unmatched cohort.

### Phase 5 — benchmark expansion

Deliverables:

- real-world regression and classification cohorts,
- shift tasks,
- controlled stress suite,
- data cards and license audit.

Exit criterion:

- primary aggregate is not simulation-dominated and covers mixed types.

### Phase 6 — freeze and sealed test

Deliverables:

- signed manifest,
- generated artifacts and sample hashes,
- freeze tag,
- one logical sealed run,
- final paper and README reduction.

Exit criterion:

- all primary claims derive from sealed results and complete matched evidence.

### Phase 7 — open-source release

Deliverables:

- wheels and native binaries,
- docs site,
- benchmark adapters,
- evidence bundle,
- paper preprint,
- contribution guide,
- reproducible container.

---

## 25. Prioritized issue backlog

### P0: stop public confusion

- **P0-01:** Rewrite README hero around transfer versus bytes.
- **P0-02:** Add explicit development-evidence status.
- **P0-03:** Move MFS version history out of README.
- **P0-04:** Add one-page `docs/status.md` listing measured, pending, and sealed work.
- **P0-05:** Rename current plot/caption to avoid implying cross-model certification.
- **P0-06:** Remove “start with AGENTS.md” as external onboarding.
- **P0-07:** Add `CONTRIBUTING.md`, `SECURITY.md`, and `CITATION.cff`.

### P0: establish the ruler

- **P0-10:** Freeze `tdl-contract-v1.json` draft.
- **P0-11:** Implement transfer edge cases and golden fixtures.
- **P0-12:** Implement family aggregation.
- **P0-13:** Implement unresolved TDL and coverage curves.
- **P0-14:** Add literal compression and matched row-subset controls.
- **P0-15:** Generate one canonical frontier JSON.

### P0: fix byte semantics

- **P0-20:** Define `state_bytes`, `sampling_closure_bytes`, `cold_bundle_bytes`.
- **P0-21:** Canonical archive and manifest.
- **P0-22:** Offline file-read trace.
- **P0-23:** Resolve all README examples to the new three-size taxonomy.
- **P0-24:** Add method-shared checkpoint charging.

### P1: strongest models

- **P1-01:** Kumo typed adapter and pin.
- **P1-02:** TabICLv2 typed adapter and pin.
- **P1-03:** Mitra-v2 adapter/license qualification.
- **P1-04:** TabPFN 3.5 frontier adapter.
- **P1-05:** TabFM/EXAONE frontier qualification.
- **P1-06:** RealMLP/TabM task-trained neural adapter.
- **P1-07:** Foundation-model contamination registry.

### P1: driver fidelity

- **P1-10:** training-only feature grouping.
- **P1-11:** conditional permutation.
- **P1-12:** real-real stability calibration.
- **P1-13:** ALE/effect-shape agreement.
- **P1-14:** controlled wrong-rule tests.

### P1: modern generators

- **P1-20:** TabDiff.
- **P1-21:** CDTD.
- **P1-22:** complete TabSyn default/tuned.
- **P1-23:** complete TabDDPM default/tuned.
- **P1-24:** complete ForestDiffusion/Flow matched panel.
- **P1-25:** ARF matched rate frontier.
- **P1-26:** GATD qualification.
- **P1-27:** one LLM/autoregressive baseline.

### P2: broad benchmark and release

- classification,
- shift tasks,
- privacy track,
- scalable track,
- full artifact attack panel,
- formal-DP TDL,
- docs site and package release.

---

## 26. Acceptance criteria for the redesign

The redesign is complete only when a fresh technical reader can answer these questions from page one and the README:

1. What does DOPE produce?
2. What exactly is charged in bytes?
3. What does transfer `0.90` mean?
4. Which powerful models must agree?
5. What is TDL@90?
6. Did the result use untouched real test data?
7. Are driver behavior and copying checked?
8. Is privacy empirical or formal?
9. Which result is development, sealed, or certified?
10. Can the full number be reproduced from a public ledger?

Quantitative acceptance checks:

- page-one comprehension: at least 8/10 correct from a fresh reader in five minutes,
- root README first meaningful result visible without scrolling through implementation details,
- one install path and one CLI,
- all benchmark numbers generated,
- all full-panel methods share the same cohort and split,
- core auditor families complete,
- exact three-level byte reporting,
- no public weighted master fitness score,
- no unresolved result rendered as zero,
- no missing baseline rendered as a DOPE win,
- no privacy language without status,
- no test access before freeze.

---

## 27. Risk register

### Risk 1 — TDL@90 may be unresolved on many datasets

This is scientifically acceptable. Publish coverage curves and lower thresholds. Do not lower the threshold after seeing results.

### Risk 2 — Cross-model minimum is noisy

Mitigation:

- aggregate within model families,
- use lower bounds,
- calibrate real-real variability,
- publish sensitivity using family quantiles,
- never replace the frozen primary rule after test access.

### Risk 3 — Foundation models have restrictive licenses

Mitigation:

- small reproducible core suite,
- research-only frontier suite,
- exact code/weight license fields,
- do not package restricted weights.

### Risk 4 — Large generators cannot expose a smooth byte frontier

Mitigation:

- report actual native configurations,
- use quantization/pruning only as declared configurations,
- include default points,
- unresolved small-budget cells are meaningful.

### Risk 5 — Driver metrics are unstable

Mitigation:

- grouped features,
- conditional permutation,
- real-real calibration,
- stability floor,
- effect-shape analysis,
- uninformative status.

### Risk 6 — Exact byte accounting is challenged as unfair

Mitigation:

- report state and closure separately,
- publish charge manifests and traces,
- use the same taxonomy for every method,
- add amortized closure sensitivity.

### Risk 7 — DOPE overfits the auditor suite

Mitigation:

- multiple model families,
- hidden/surprise sensitivity auditors frozen before test but not used for compiler selection,
- shift datasets,
- controlled wrong-rule tasks,
- no benchmark-test feedback during development.

### Risk 8 — Repository cleanup breaks evidence hashes

Mitigation:

- signed legacy tag,
- path map,
- immutable release asset,
- legacy rebuild before migration,
- do not rewrite historical claims.

### Risk 9 — “description length” is mistaken for formal MDL

Mitigation:

- use “operational transfer description length,”
- state exact canonical bytes, not universal code length,
- discuss MDL/rate–distortion as motivation,
- avoid theoretical optimality claims.

### Risk 10 — A compact predictor masquerades as synthetic data

Mitigation:

- predictor-only control,
- feature-distribution fidelity,
- downstream models trained on sampled tables,
- driver and subgroup transfer,
- multiple targets or target-withheld sensitivity where applicable.

---

## 28. Decisions that should be made immediately

### Decision A — public metric

**Recommend:** Adopt Transfer Description Length and TDL@90. Retire MFS as public headline.

### Decision B — primary rate

**Recommend:** `state_bytes` primary; closure and cold bundle mandatory secondary fields.

### Decision C — 10 KiB

**Recommend:** keep as a DOPE product tier and vertical reference line, not a universal metric boundary.

### Decision D — privacy gate scope

**Recommend:** copy/closure gates required for scientific TDL; full empirical privacy required for release certification; formal DP is a separate profile.

### Decision E — first new paper scope

**Recommend:** cross-model regression first only if the modern baseline and auditor panels are complete; otherwise delay the flagship claim rather than publish another partial benchmark. Classification can follow in the same benchmark release or a clearly named extension.

### Decision F — naming

**Recommend:**

- project: DOPE — Downstream Objective-Preserving Encoding,
- metric: Transfer Description Length,
- plot: transfer–description frontier,
- certificate: TDL-Core-2026.10 or later frozen date,
- product tier: DOPE-L3 / 10 KiB only in technical docs.

### Decision G — current paper

**Recommend:** freeze the current paper as a rigorous development/validation paper and begin a clean TDL paper rather than continuing to layer MFS-v4 onto the existing manuscript.

---

## 29. What not to do

- Do not add MFS-v5 with a different set of weights.
- Do not call encoded scalar count the universal size metric.
- Do not let Kumo representation distance define synthetic quality.
- Do not use CatBoost alone as the headline auditor.
- Do not add more caveats to the current first page instead of changing the thesis.
- Do not put every evidence memo in the root documentation navigation.
- Do not mix partial cohorts into one ranking table.
- Do not infer equivalence from an interval containing zero.
- Do not treat a timeout or missing baseline as zero utility.
- Do not hide shared pretrained checkpoints outside the charge manifest.
- Do not claim a 1.9 KB standalone deployment when a runtime closure is required; say 1.9 KB dataset-specific state and report closure.
- Do not claim privacy from no row copies or failed empirical attacks.
- Do not open the official test while the metric, baseline roster, or figure code is changing.

---

## 30. Final recommended public story

The final project should be understandable in under 30 seconds:

> **DOPE compresses a supervised table into a tiny generative program. We measure the smallest exact artifact whose synthetic rows preserve at least 90% of real-data learning across modern tabular foundation models, boosted trees, neural models, and linear controls on unseen real data. That number is TDL@90. The complete result is a transfer-versus-bytes frontier, with driver, copy, privacy, and artifact-closure status shown explicitly.**

That story is cleaner, more defensible, and more profound than a weighted synthetic-data score.

It turns DOPE from “another generator with a complicated benchmark” into a research program around a foundational question:

> **How much predictive structure can be packed into a few thousand bytes?**

---

# Appendix A — Proposed machine-readable candidate report

```json
{
  "format": "dope-tdl-candidate-report",
  "version": 1,
  "benchmark_version": "dope-bench-regression-v1",
  "dataset_id": "airfoil-self-noise",
  "method_id": "dope-1.0",
  "configuration_id": "budget-2k-seed-11",
  "task": "regression",
  "synthetic_size": "4n",
  "fit_seed": 11,
  "sample_seeds": [101, 211, 307],
  "rate": {
    "state_bytes": 1944,
    "sampling_closure_bytes": 72050710,
    "cold_bundle_bytes": 83411200,
    "stored_scalar_count": 63,
    "stored_symbol_count": 117,
    "program_node_count": 117
  },
  "transfer": {
    "auditors": [
      {
        "id": "kumo_tabular_large",
        "family": "foundation",
        "null_loss": 1.0,
        "real_loss": 0.3,
        "synthetic_loss": 0.37,
        "raw_transfer": 0.90,
        "lcb": 0.86,
        "informative": true
      }
    ],
    "families": {
      "foundation": 0.86,
      "boosted_tree": 0.93,
      "neural": 0.91,
      "linear": 0.95
    },
    "robust": 0.86
  },
  "driver": {
    "robust": 0.84,
    "complete": true
  },
  "gates": {
    "canonical_artifact": "pass",
    "offline_sampling": "pass",
    "declared_reads_only": "pass",
    "schema_valid": "pass",
    "exact_copy": "pass",
    "near_copy": "pass",
    "empirical_privacy": "not_evaluated",
    "sealed_split_integrity": "pass"
  },
  "qualification": {
    "tdl80": true,
    "tdl90": false,
    "tdl95": false,
    "release_certified": false
  },
  "evidence": {
    "status": "development",
    "source_commit": "...",
    "contract_sha256": "...",
    "artifact_sha256": "...",
    "test_split_sha256": "..."
  }
}
```

# Appendix B — Proposed benchmark registry excerpt

```yaml
format: dope-benchmark-registry
version: 1

datasets:
  - id: airfoil-self-noise
    task: regression
    domain: aerospace_acoustics
    target: sound_pressure_level
    license: CC-BY-4.0
    split:
      kind: grouped_or_iid
      seed: 1729
    logical_types:
      frequency: continuous
      angle: continuous
      chord: continuous
      velocity: continuous
      displacement: continuous
      sound_pressure_level: target_continuous
    source_sha256: ...
    processed_sha256: ...
    tracks: [compact_regression, scalable]
```

# Appendix C — Proposed auditor registry excerpt

```yaml
format: dope-auditor-registry
version: 1
profile: tdl-core-2026.10

auditors:
  - id: kumo_tabular_large
    family: foundation
    required: true
    implementation: NVIDIA/structured-data-models
    commit: ...
    checkpoint_sha256: ...
    code_license: ...
    weights_license: ...
    task_support: [regression, binary, multiclass]
    typed_input: true
    contamination_status: clean-by-construction
    device: cuda
    precision: fp16
```

# Appendix D — Proposed first-page result table

| Method | Robust transfer LCB | State bytes | TDL@90 coverage | Driver pass | Evidence |
|---|---:|---:|---:|---:|---|
| DOPE | — | — | — | — | sealed result required |
| TabSyn | — | — | — | — | sealed result required |
| TabDiff | — | — | — | — | sealed result required |
| CDTD | — | — | — | — | sealed result required |
| ForestDiffusion | — | — | — | — | sealed result required |
| ARF | — | — | — | — | sealed result required |
| compressed rows | — | — | — | — | sealed result required |

The paper must not prepopulate this table from unmatched development cohorts.

# Appendix E — Recommended first implementation sequence

1. Freeze/tag current evidence.
2. Rewrite README top 150 lines without changing benchmark numbers.
3. Implement three-level byte manifest.
4. Implement TDL transfer mathematics and golden tests.
5. Add compressed-row and exact-table controls.
6. Qualify Kumo, TabICLv2, CatBoost, LightGBM, RealMLP, and elastic net on ten datasets.
7. Implement driver real-real calibration.
8. Produce a pilot cross-model frontier.
9. Complete mandatory generator adapters.
10. Freeze benchmark v1 and run the full development panel.
11. Review every result and claim without opening test.
12. Freeze artifacts, adapters, and analysis.
13. Run one logical sealed test.
14. Publish paper, code, manifests, and evidence.


# Appendix F — Audit evidence map

## Current DOPE repository

| Finding used in this specification | Primary repository location |
|---|---|
| Current 1.9 KB/0.898 DOPE and 42.3 MB/0.924 TabSyn development results | `README.md`, generated benchmark block |
| Current first-page thesis and CatBoost-centered paper | `docs/whitepaper/dope-mfs.tex` |
| Current paper build and evidence rules | `docs/whitepaper/README.md`, `Justfile` |
| Missing sealed test and incomplete baseline/privacy blocks | `docs/whitepaper/REVIEW_GAPS.md`, `BASELINE_GAP.md`, `DECISIONS_PENDING.md` |
| Historical reviewer concerns | `docs/whitepaper/SELF_REVIEW.md`, `IEEE_READINESS.md` |
| MFS-v4 weights, prerequisites, gates | `production/kpi-contract-v4.json` |
| MFS-v4 aggregation and seed handling | `rust/fitness/v4.rs` |
| Current knowledge-compression formula | `rust/transfer_metrics/mfs_v4.rs` |
| Existing Kumo/Mitra-v2/TabICL2 benchmark hooks | `research/benchmark/mfs_v3_audit.py`, `research/benchmark/representation.py` |
| Rust/Python generation split | `Cargo.toml`, `pyproject.toml`, `README.md` |
| Typed schema and logical roles | `rust/schema.rs`, `rust/typed_data.rs` |
| Internal module and evidence architecture | `docs/architecture.md`, `AGENTS.md` |

## External auditor implementations considered

- `NVIDIA/structured-data-models` — KumoTabular, TabICLv2 and unified structured-data interfaces.
- `soda-inria/tabicl` — official TabICLv2 implementation.
- `PriorLabs/tabpfn` — official TabPFN implementation and current model releases.
- `google-research/tabfm` — official TabFM implementation.
- Mitra-v2 technical report and released implementation/weights.
- `LGAI-Research/EXAONE-Tabular` — official EXAONE Tabular implementation.

## External generator implementations considered

- `amazon-science/tabsyn`
- `yandex-research/tab-ddpm`
- `minkaixu/tabdiff`
- `muellermarkus/cdtd`
- `SamsungSAILMontreal/ForestDiffusion`
- `dzagardo/gatd`
- official GReaT and REaLTabFormer repositories
- open ARF, CTGAN, TVAE, synthpop, AIM and Private-PGM implementations

Every method admitted to the final benchmark still requires its own source, license, environment, and artifact-charge qualification. Inclusion in this design document is not equivalent to benchmark admission.
