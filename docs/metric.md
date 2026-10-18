# TDL: Transfer Description Length

**Status: specified, not measured.** No method has a TDL result. The contract keeps
`protocol_frozen=false` until every field is concrete. The name (decision D-01) and the
headline thresholds (D-02) are working defaults pending Jepson's decision; see
[decisions](decisions.md). The code uses the single constant `METRIC_NAME`, so a rename
is one line.

This page states the definitions that the Rust metric core (`rust/metrics/`, T-030 to
T-034) and its Python conformance mirror (`research/transfer/tdl/`, T-036) implement.
The golden corpus (`tests/golden/`, T-037) holds the two to identical outputs. The
exploratory predecessor is [`docs/vnext/LEARNING_DENSITY.md`](vnext/LEARNING_DENSITY.md).
The retired weighted scores are in [the MFS history](archive/mfs-history.md).

## Candidate successor (I00 / C00)

The unified workplan introduces the unfrozen
[`tdl-contract-v2.candidate`](../contracts/tdl-contract-v2-candidate.json).
Its identities, schema admission and independent state axes are defined in
[the candidate protocol decision](decisions/tdl-v1.md). Sections 1–10 below retain
the historical v1 definitions implemented by the existing Rust/Python conformance
core; their defaults and receipt semantics do not freeze or certify the successor.

Candidate transfer uses paired harm `z = L_S - 0.10*L_0 - 0.90*L_R` at tau = 0.90
in original target units, with a simultaneous one-sided upper bound at most zero.
It does not silently reinterpret historical generic `Margin` or opposite-sign
benefit receipts. Candidate admission rejects historical receipts unless explicit
legacy replay is requested; replay preserves v1 semantics and grants no candidate
qualification. Candidate inference and scoring remain later workplan tasks.

The three informative real-reference parameters (`absolute`, `relative`,
`lcb_required`), driver stability, absolute importance, synthetic noncollapse and
the D19 multiplicity procedure remain **null**, pending calibration/decision.
The plan distinguishes 10,000 bootstrap draws from 2,000 independent calibration
replications. Unresolved rosters, rights profiles and applicability are explicit;
the candidate stays `status="candidate"`, `protocol_frozen=false`, `certified=false`.

Seven axes remain independent: `artifact_qualification_state`,
`scheduled_cell_state`, `campaign_completeness_state`, `empirical_privacy_state`,
`formal_dp_state`, `release_authorization_state`, and `claim_provenance_state`.
Schema admission establishes no scientific result or release permission. Candidate
receipts remain noncitable and uncertified; empirical privacy is never formal DP
or HIPAA de-identification.

## 1. The question

> How few dataset-specific bytes does a sampling artifact need before models trained or
> conditioned only on its samples recover the real data's predictive gain, and rely on the
> same drivers, when tested on untouched real rows?

Bytes are the quantity being reduced. Transfer and driver fidelity are constraints that
must both pass. No weight trades one against another, and no diagnostic can rescue a
failed constraint.

## 2. Objects

- `D_fit` is the real fitting table. The compiler and the real-trained auditors see it.
- `D_select` is used for configuration and budget selection only.
- `D_test` holds the untouched real query rows. It stays sealed until the protocol, the
  adapters and the analysis code are frozen and hash-committed.
- A compiler receives `D_fit`, a declared target and a byte budget `B`, and emits an
  artifact `A`.
- The sampler reads only `A`, a decoder identified by hash, a row count and a sample seed.
- An auditor `m` from family `f` is trained or conditioned on `D_fit` or on samples from
  `A`. Both versions are scored on the same `D_test` rows.

## 3. Transfer

### 3.1 Paired losses and retention

For each informative cell (dataset `d`, auditor `m`, artifact `A`, seeds), the evaluator
records paired per-query losses in original target units: `null_losses`, `real_losses` and
`synthetic_losses`. With `L` the mean loss over the query rows:

```math
T_{d,m}(A) = \frac{L_{\mathrm{null}} - L_{\mathrm{syn}}(A)}{L_{\mathrm{null}} - L_{\mathrm{real}}}
```

`T = 1` means synthetic training recovers all of the real gain over the null, and `T = 0`
means none of it. `T < 0` means synthetic training is harmful. `T > 1` means synthetic
training beat the real-trained reference in that run. Receipts store `T` unclipped. Only
a display may clip it, and its caption must say so. A value above 1 never offsets another
auditor's failure.

The null is the strongest legitimate baseline that learns nothing from the table. For an
in-context foundation model, this is the strongest of no-context, label-shuffled-context
and the conventional task null. DOPE therefore gets no credit for what the model's prior
already knows.

### 3.2 Informative-task gate

A cell is informative only when the real data teaches the auditor something measurable:

```math
L_{\mathrm{null}} - L_{\mathrm{real}} \ge \max(\epsilon_{\mathrm{abs}},\ \epsilon_{\mathrm{rel}} \lvert L_{\mathrm{null}} \rvert)
\quad\text{and}\quad \mathrm{real\_gain\_lcb} > 0 .
```

An uninformative cell is marked `uninformative` and left out of that auditor's estimate.
It is not scored as zero, as a failure or as a win. Low-signal coverage is reported
separately, and a method cannot improve its result by creating more excluded cells. A
constant target is refused outright. The conformance defaults are `absolute_gain_floor = 1e-12`
and `relative_gain_floor = 0.01`; the contract fixes the final values.

### 3.3 Family aggregation

The contract's required families are boosted trees, linear or additive models, and
task-trained neural models. Tabular foundation models are the held-out family: the compiler
never sees them, they confirm transfer, and their panel qualification waits on decision D-07.
Every member must be `qualified` in the auditor registry before it can count (see
[claims policy](claims-policy.md) §1).

```math
T_f(A) = \operatorname{mean}_{m \in f} T_{d,m}(A), \qquad
T_{\mathrm{robust}}(A) = \min_{f \in \mathcal{F}_{\mathrm{required}}} T_f(A).
```

The working default takes the mean within a family; S4 and S5 propose the median, and the
choice is pending in T-020. Across families the result is always the worst family. The
cross-family mean is reported beside it but never certifies anything. Adding a worse
family can never raise `T_robust` (a property test in T-031).

### 3.4 Uncertainty

The seed design is five generator fit seeds by three sample seeds, plus declared auditor
seeds for stochastic task-trained models. A cluster bootstrap resamples source family,
then dataset, then seed. It uses a fixed RNG seed (xorshift64) and the same resample
indices for every auditor family. The one-sided 95% lower bound `LCB(T_robust)` is a lower
order statistic of the bootstrap replicates; the contract fixes the generator, seed, replicate
count and quantile rule (`aggregation_policy` in `contracts/tdl-contract-v1.json`). Qualification reads the lower bound only, never the mean
and never the best seed.

## 4. Driver fidelity

### 4.1 Importance on the real test rows

For each qualified auditor, the evaluator fits a real-trained instance and a
synthetic-trained instance of the same frozen recipe. It computes grouped permutation
importance for both on the same `D_test` rows. Groups are the original semantic
features: one-hot columns are recombined, and missingness indicators join their source
feature. Correlated groups get a conditional permutation pass. Raw SHAP magnitudes are
never compared across model families; native attributions are sensitivity analyses only.

### 4.2 Components

With `p_real` and `p_syn` the normalized non-negative importance vectors over groups:

| Component | Definition | Role |
| --- | --- | --- |
| Mass agreement | `1 − ½‖p_real − p_syn‖₁` (one minus total variation) | required |
| Rank agreement | tie-aware pairwise rank agreement of the two orders (NDCG@k is reported beside it) | required |
| Sign agreement | direction agreement on drivers whose real-trained sign is stable across real replicates | required where defined |
| Interaction agreement | mass agreement over the top pairwise interactions, times the strength ratio `min(total_real, total_syn) / max(total_real, total_syn)`, so losing an interaction cannot pass on its rank alone | listed in `required_components`; S4 proposes it as secondary in v1 |

No sign is required for unordered categoricals, inactive groups, or drivers whose real
sign is unstable.

### 4.3 Real-versus-real ceiling

Feature importance is noisy even between two models trained on real data. The evaluator
therefore fits two matched real instances and records each component's real-versus-real
agreement as the ceiling `c`. Each synthetic component is divided by its declared
ceiling and clipped to `[0, 1]`. When a ceiling is below the stability floor (conformance
default 0.70), the cell is `uninformative` for drivers.

### 4.4 Driver score

The contract lists the `required_components`; the conformance default is mass, rank, sign
and interaction. The auditor driver score is the smallest normalized required component. The family
score is the mean over the family's auditors. `D_robust` is the worst family score. A
missing required component makes the cell `incomplete`, never zero.

## 5. Byte scopes

| Scope | Field | Contents | Role |
| --- | --- | --- | --- |
| Dataset-specific payload | `payload_bytes` | every byte that depends on the fitted data and that sampling reads: parameters, codebooks, quantiles, schema, transforms, category dictionaries, missingness, seeds needed for replay, framing and sidecars | primary rate axis |
| Sampling closure | `sampling_closure_bytes` | payload plus decoder, runtime, shared checkpoints and libraries that a cold sampler opens | required secondary axis |
| Amortized | `amortized_bytes(K)` | payload plus shared method bytes divided by a declared `K` | optional sensitivity |

The charge comes from one canonical container with deterministic section order and a
charge manifest (T-022, T-034). The compiler's count, `inspect`'s count, the file length,
the sum of the charged sections and the bytes a clean-room sampler reads must all agree.
Encoded scalar counts, bits per source cell and section sizes are diagnostics. They never
rank anything, because packing changes them without changing the information stored.

## 6. TDL and the frontier

The byte grid is `{0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256}` KiB. Each method compiles
an artifact under every grid budget it supports. An artifact **qualifies** when all of
these hold:

```math
\mathrm{LCB}_{95}\bigl(T_{\mathrm{robust}}(A)\bigr) \ge \tau_T, \qquad
D_{\mathrm{robust}}(A) \ge \tau_D, \qquad G(A) = \text{pass}.
```

`G` is the conjunction of the hard gates in section 8. **TDL** at thresholds `τ_T/τ_D`
is the charged `payload_bytes` of the qualifying artifact at the smallest tested grid
budget. It is reported together with that grid interval `(B_{j−1}, B_j]`. It is the
*smallest tested* size, not an optimum: no budget between grid points is claimed, and
nothing is interpolated.

The public shorthand is `KiB@T90/D80`. The frontier, which is the best qualified transfer
at each budget, is always published beside it.

| Operating point | `τ_T` | `τ_D` | Role |
| --- | --- | --- | --- |
| T80/D80 | 0.80 | 0.80 | low-rate regime |
| **T90/D80** | **0.90** | **0.80** | **headline (working default, D-02)** |
| T90/D90 | 0.90 | 0.90 | stricter drivers |
| T95/D95 | 0.95 | 0.95 | high-fidelity regime |

`transfer@{1, 4, 10, 64} KiB` is reported as a secondary fixed-budget view (D-03). It is
never the headline.

Across a benchmark with frozen task weights `w_t`, coverage at budget `B` is the weighted
share of tasks whose artifact qualifies:

```math
C(B) = \sum_{t \in W} w_t\, \mathbf{1}\bigl[\text{the task's artifact at } B \text{ qualifies}\bigr].
```

Failed, unsupported, timed-out, missing, over-budget and unqualified tasks stay in the
denominator. `B*_tested(q)` is the smallest tested budget with `C(B) ≥ q`; the working
coverage floor is `q = 0.90`.

## 7. Result states

| State | Condition | Reported |
| --- | --- | --- |
| `qualified` | some tested budget qualifies | TDL with its grid interval; at the first grid point, "at most `B_1`" |
| `not_reached_in_tested_range` | no tested budget qualifies, and every required cell is present | "> `B_k`" (this is `B*_tested` beyond the grid) |
| `unqualified` | a hard gate failed, or a required family has no qualified auditor | the failing gate; no byte value |
| `incomplete` | a required cell, family, component or gate is missing | nothing; the metric emits no number |
| `failed` | a terminal failure, timeout, refusal or over-budget compile | the disposition; the cell stays in every denominator |

A coverage curve that decreases somewhere is flagged `non_monotone`. It then reports
every measured point and the first crossing, with no monotone summary. These states
replace the exploratory ones in LEARNING_DENSITY: `measured` and `censored_below_grid`
become `qualified`, `no_feasible_tested_artifact` becomes `not_reached_in_tested_range`,
and `pending` becomes `incomplete`.

## 8. Hard gates

Gates are conjunctive, and none has a weight:

- exact-copy and near-copy checks on the shared map, and the cleartext scan;
- positive lift over the predictor-only and independent-marginals controls;
- a reconciled byte ledger and verified clean-room sampling;
- every required auditor family present and qualified.

The empirical attack panel (membership and attribute inference) is required for release
certification, which is reported separately from the scientific TDL. A formal-DP result
would be its own profile. An empirical attack result never establishes formal differential
privacy or HIPAA de-identification.

## 9. What TDL is not

TDL is an operational rate–distortion measure for one declared supervised workload. It
does not estimate Shannon information, Kolmogorov complexity or mutual information. It
is not a code-length optimum, a causal claim, or a measure of general dataset quality.
It is not a weighted score: distribution, structure, coverage, representation distance,
runtime and encoded-value counts are diagnostics.

## 10. Field names

These are the names in the Python conformance package (`research/transfer/tdl/`, T-036) and
the golden corpus (`tests/golden/`, T-037). T-008 requires the notation to match the T-020 and
T-023 schemas, so this table is checked again when the Rust core and schemas land.

| Quantity | Fields |
| --- | --- |
| Transfer inputs | `null_losses`, `real_losses`, `synthetic_losses` (per query row), `real_gain_lcb`, `constant_target`, `absolute_gain_floor`, `relative_gain_floor` |
| Transfer outputs | `status` (`informative`, `uninformative`, `incomplete`), `null_loss`, `real_loss`, `synthetic_loss`, `real_gain`, `raw_transfer` (unclipped `T`), `clipped_transfer` (display only) |
| Aggregation | `family_means`, `robust_transfer` (worst family), `mean_transfer` (reported, never certifying), `lcb` |
| Drivers | inputs `real_importances`, `synthetic_importances`, `real_signs`, `synthetic_signs`, `real_interactions`, `synthetic_interactions`, `real_real_ceiling` (per component), `stability_floor`, `driver_threshold`, `required_components`; outputs `raw_components`, `normalized_components`, `driver_fidelity` (`D`), `driver_pass` |
| Bytes | `payload_bytes`, `section_bytes`, `framing_bytes`, `shared_bytes`, `sampling_closure_bytes`, `cold_bundle_bytes`, `amortized_bytes` |
| Frontier | `status` (`qualified`, `not_reached_in_tested_range`, `unqualified`, `incomplete`, `failed`), `tdl_bytes`, `largest_tested_bytes`, and per budget `artifact_bytes`, `lcb_transfer`, `driver_fidelity`, `tdl_pass`, `frontier_transfer` |
| Contract | `transfer_threshold` (`τ_T`), `driver_threshold` (`τ_D`), `protocol_frozen` |
