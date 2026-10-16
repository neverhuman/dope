# DOPE Scientific and Engineering Rebuild Specification

**Project:** DOPE — Downstream Objective-Preserving Encoding  
**Repository audited:** `neverhuman/dope`  
**Audit basis:** `main` at commit `ee4a3d0f640088a894926aaa2a268bdc008f0db7`  
**Date:** 2026-10-15 (America/Los_Angeles)  
**Status:** Proposed engineering and scientific specification; no new benchmark claims are made here.

---

## 0. Executive directive

DOPE has a genuinely powerful scientific idea:

> **A synthetic dataset should be judged by the smallest executable description that preserves what strong models learn from the real data.**

The repository already contains many of the necessary pieces: charged artifact bytes, train-on-synthetic/test-on-real transfer, compact sampling programs, explicit controls, privacy gates, multiple tabular foundation-model auditors, and feature-driver agreement. The problem is not an absence of serious work. The problem is that the public scientific story is fragmented across historical contracts, partially measured metrics, validation-only results, implementation detail, operational ledgers, and multiple generations of terminology.

The rebuild must make one concept dominant:

> **How many serialized bytes are required before models trained on synthetic data recover both the predictive skill and the important feature drivers learned from real data, when evaluated on untouched real rows?**

The recommended public north-star metric is:

## Learning-Preserving Description Length

\[
\operatorname{LPDL}_{\tau_T,\tau_D}
=
\min_A |\operatorname{serialize}(A)|
\]

subject to:

\[
T(A)\ge \tau_T,\qquad
D(A)\ge \tau_D,\qquad
G(A)=1
\]

where:

- \(A\) is a self-contained dataset-specific sampling artifact;
- \(T(A)\) is conservative downstream transfer preservation on unseen real data;
- \(D(A)\) is conservative feature-driver preservation on the same unseen real data;
- \(G(A)\) is the conjunction of replay, evidence, byte-accounting, anti-copy, and privacy gates;
- \(|\operatorname{serialize}(A)|\) is the exact deterministic byte count of all dataset-specific state needed for sampling.

The default headline should be:

\[
\boxed{\operatorname{LPDL}_{90/90}}
\]

meaning:

> **The minimum number of charged bytes needed to preserve at least 90% of predictive skill and 90% of model-driver structure.**

The complementary fixed-budget KPI should be:

\[
\boxed{\operatorname{CLP}@10\text{KB}
=
\min(T_{10\text{KB}},D_{10\text{KB}})}
\]

where **CLP** means **Certified Learning Preservation**.

This is simpler, harder to game, and more scientifically profound than a weighted “master fitness” score. It puts the scarce resource—bytes—on the axis being minimized, and makes quality requirements explicit constraints rather than ingredients that can compensate for one another.

---

# Part I — Current-state diagnosis

## 1. What is already excellent

The project should preserve the following ideas and engineering assets:

1. **The artifact is a program, not a CSV dump.**  
   DOPE’s central object is a compact sampler that can emit fresh rows.

2. **Every dataset-specific byte is intended to be charged.**  
   This is much stronger than merely counting neural-network parameters.

3. **Utility is measured by synthetic-to-real transfer.**  
   Models are trained or conditioned on synthetic rows and evaluated on unseen real rows.

4. **The project already distinguishes useful controls.**  
   Independent marginals, predictor-only sampling, real-row resampling, and null prediction expose different failure modes.

5. **The release logic fails closed.**  
   Missing evidence yields a null score rather than an optimistic number.

6. **The current v4 direction recognizes that feature-driver preservation matters.**  
   This is essential. A synthetic dataset that reaches similar error while teaching models the wrong causal or operational signals is not equivalent.

7. **The project has unusually strong evidence-traceability instincts.**  
   Hash-pinned receipts, generated tables, sealed test partitions, and explicit selection exposure are all valuable.

8. **The compact generator is technically interesting.**  
   Quantized marginals, compact dependence families, small target models, deterministic serialization, and Rust sampling are well aligned with the thesis.

The rebuild should not erase this work. It should remove the layers that prevent a new reader from seeing it.

---

## 2. The five critical defects

### P0.1 — The repository exposes three incompatible metric truths

The current public surface simultaneously implies:

- an active historical v2 release contract;
- a white paper centered on MFS-v3;
- a root README and current implementation centered on MFS-v4.

That is fatal to comprehension. A reader cannot determine which metric is canonical, which has measured evidence, which is historical, and which is only preregistered.

**Required action:** establish one public metric contract and archive every predecessor.

Recommended structure:

```text
production/
  lpdl-contract-v1.json              # sole active public scientific contract

docs/archive/metrics/
  mfs-v2/
  mfs-v3/
  mfs-v4/

evidence/status.json                 # generated current truth
STATUS.md                            # generated human-readable truth
```

Public prose should use one status sentence:

> “LPDL-v1 is the active scientific contract. Current public results are validation-only; no method has an LPDL-certified release result until the sealed protocol and every gate are complete.”

Historical MFS implementations may remain executable for receipt verification, but they must not compete with the active story.

---

### P0.2 — The headline result is still effectively CatBoost retention

The root result table is easy to read, but the visible result is still “CatBoost retention at 4n versus bytes.” That does not yet prove the stronger claim the project wants:

- transfer across powerful model families;
- preservation of model behavior;
- preservation of important feature drivers;
- robustness across seeds, tasks, and real-world dataset families;
- privacy and anti-memorization qualification.

The v4 contract names KumoTabular, Mitra-v2, TabICLv2, and driver attribution, but these are not yet the simple, measured, first-page result.

**Required action:** the headline plot and score must be generated from the multi-auditor, multi-seed, driver-aware protocol—not from a CatBoost-only summary.

---

### P0.3 — The current weighted master score is too complicated and too gameable

The v4 score mixes seven weighted components:

- transfer utility;
- driver fidelity;
- representation closeness;
- knowledge compression;
- distribution fidelity;
- structure fidelity;
- coverage/realism.

Even with a geometric mean and hard gates, this creates multiple problems:

1. **Compensation remains possible.**  
   A method can be weaker on the two outcomes users actually care about and recover score through secondary fidelity measures.

2. **The public explanation is impossible to retain.**  
   A reader has to learn seven components, several nested formulas, weights, clamps, matched nulls, and prerequisites before understanding the result.

3. **Several components overlap.**  
   Transfer, representation distance, structure fidelity, distribution fidelity, and coverage are not independent scientific dimensions.

4. **The weights are editorial choices.**  
   There is no natural reason why representation closeness should count 15% while structure counts 5%.

5. **The score hides the profound compression claim.**  
   Bytes become one weighted subcomponent instead of the resource being minimized.

**Required action:** make transfer and driver preservation explicit constraints; make bytes the optimization target. Keep distribution, structure, coverage, representation, runtime, and sample realism as diagnostics and failure analysis—not as ingredients in the public scalar.

---

### P0.4 — Current compression submetrics admit obvious anti-invariance failures

The current implementation includes:

\[
\text{bits per original cell}
=
\frac{8\,\text{artifact bytes}}{\text{rows}\times\text{columns}}
\]

This can improve merely by duplicating the training table. The artifact may be unchanged while the denominator grows. That is not evidence that more predictive information was compressed.

The current `encoded_numbers` concept is also not method-family invariant. One “number” may be:

- an 8-bit code;
- a 64-bit float;
- a tree threshold;
- a compressed rank table;
- a seed driving a procedural decoder;
- an index into a large shared codebook.

Counting values is useful for inspecting DOPE internally, but it is not a defensible cross-method scientific currency.

The current log-byte score also double-counts the 10 KB threshold: size is both a hard gate and a weighted soft component.

**Required action:**

- make exact serialized bytes authoritative;
- retain bits-per-cell and encoded-value counts only as labeled diagnostics;
- add anti-gaming invariance tests;
- never use raw “number of parameters/values” as the cross-method denominator.

Mandatory invariance tests:

| Test | Required outcome |
|---|---|
| Duplicate every training row 10× | Artifact-size score must not improve merely because the denominator grew |
| Reorder rows | Identical artifact or declared deterministic equivalence |
| Reorder columns with schema remap | Same scientific score and same charged content class |
| Append random noise columns | Score must not improve; bytes should stay equal or increase |
| Permute target | Transfer must collapse |
| Store compressed/encrypted training rows | Copy/reconstruction gates must fail |
| Move data into a sidecar | Sidecar bytes must be charged |
| Move data into a “shared” method checkpoint | Closure accounting must expose it |

---

### P0.5 — The README and paper contain the laboratory notebook instead of the scientific product

The root README currently includes:

- benchmark tables;
- binary format details;
- compatibility history;
- GPU feature instructions;
- campaign commands;
- router and embedding internals;
- deep-campaign accounting totals;
- multiple metric generations;
- release and audit details.

The white-paper directory mixes the manuscript with:

- baseline operational gaps;
- ETAs;
- review notes;
- comprehension notes;
- pending decisions;
- style references;
- readiness memos;
- retained-evidence reviews;
- self-review documents.

All of these may be useful internally. They should not sit in the primary path a reviewer uses to understand the work.

**Required action:** separate:

1. **scientific claim;**
2. **user quick start;**
3. **benchmark protocol;**
4. **implementation reference;**
5. **historical evidence and operations.**

---

# Part II — The new scientific contract

## 3. One-sentence product definition

Use this everywhere:

> **DOPE compiles a tabular dataset into a small executable generator and measures the minimum serialized bytes required for models trained on its samples to recover the same predictive skill and feature drivers on unseen real data.**

A shorter tagline:

> **Preserve what models learn—in kilobytes.**

A non-technical explanation:

> “We do not ask whether synthetic rows merely look real. We ask whether powerful models learn the same useful relationships from them—and how few bytes the generator needs to encode those relationships.”

---

## 4. Formal problem statement

Let:

- \(D_{\mathrm{fit}}\) be the real fitting rows;
- \(D_{\mathrm{select}}\) be an inner selection partition;
- \(D_{\mathrm{holdout}}\) be untouched real evaluation rows;
- \(C\) be a compiler;
- \(A_B=C(D_{\mathrm{fit}},B)\) be an artifact produced under byte budget \(B\);
- \(S(A_B,N,\sigma)\) be \(N\) synthetic rows sampled from \(A_B\) under seed \(\sigma\);
- \(m\) be an auditor;
- \(L_m(X_{\mathrm{context}},D_{\mathrm{holdout}})\) be the auditor’s holdout loss after training or in-context conditioning on \(X_{\mathrm{context}}\).

The project solves:

\[
\min_{A} |\operatorname{serialize}(A)|
\]

subject to:

\[
\operatorname{Transfer}(A)\ge \tau_T
\]

\[
\operatorname{DriverPreservation}(A)\ge \tau_D
\]

\[
\operatorname{PrivacyAndReplayGates}(A)=1
\]

\[
\operatorname{EvidenceComplete}(A)=1.
\]

The paper must explicitly say that this is an **operational rate-distortion problem for learned behavior**, not a claim to estimate:

- Shannon information in the source table;
- Kolmogorov complexity;
- mutual information;
- causal truth;
- human intelligence;
- universal dataset quality.

The profound statement is still available:

> A smaller artifact that preserves the same learned behavior has captured more transferable predictive structure per serialized byte.

That is strong and defensible without overclaiming “intelligence per bit.”

---

## 5. Transfer preservation

### 5.1 Auditor-specific transfer

For a lower-is-better loss:

\[
R_{d,m,s}(A)
=
\frac{L^{\mathrm{null}}_{d,m,s}-L^{\mathrm{syn}}_{d,m,s}(A)}
     {L^{\mathrm{null}}_{d,m,s}-L^{\mathrm{real}}_{d,m,s}}
\]

where:

- \(L^{\mathrm{null}}\) is a predeclared null prediction loss;
- \(L^{\mathrm{real}}\) is the loss using real context/training rows;
- \(L^{\mathrm{syn}}\) is the loss using only sampled synthetic context/training rows.

Rules:

1. Store and analyze **unclamped** retention.
2. Clamp to \([0,1]\) only for the public preservation plot.
3. Report values above 1 separately; they may reflect regularization, sampling effects, or noise.
4. Never silently transform missing or failed cells into zero.
5. A timeout is missing evidence, not a DOPE win.

For higher-is-better metrics, use the symmetric gain formulation:

\[
R
=
\frac{S^{\mathrm{syn}}-S^{\mathrm{null}}}
     {S^{\mathrm{real}}-S^{\mathrm{null}}}.
\]

### 5.2 Low-signal tasks

When the real-trained auditor barely beats the null, the denominator is unstable.

A cell is **informative** only when:

\[
L^{\mathrm{null}}-L^{\mathrm{real}}
\ge
\max(\epsilon_{\mathrm{abs}},
     \epsilon_{\mathrm{rel}}L^{\mathrm{null}}).
\]

For low-signal cells:

- do not report a retention ratio;
- require non-inferiority to the real-trained reference under a preregistered tolerance;
- report low-signal coverage separately;
- do not let a method improve its score by creating more excluded cells.

### 5.3 Transfer aggregation

Auditors are grouped into families:

1. simple/transparent;
2. gradient-boosted trees;
3. neural tabular learners;
4. tabular foundation models.

For each family:

- aggregate repeated sample seeds inside a fit;
- aggregate fit seeds inside a dataset;
- use a hierarchical paired bootstrap over source family → dataset lineage → split → fit seed;
- compute a one-sided lower confidence bound.

Define:

\[
T_B = \min_{f \in \mathcal{F}}
\operatorname{LCB}_{95\%}\left(R_{B,f}\right).
\]

This is intentionally non-compensatory. A method cannot be called learning-preserving if it works for one auditor family and fails for another.

Public reporting should also show:

- median;
- 25th percentile;
- family-level LCB;
- dataset coverage;
- failure and timeout counts.

---

## 6. Driver preservation

Transfer alone is not sufficient. Two models can obtain similar aggregate loss while depending on different features, exploiting shortcuts, or learning a different decision surface.

### 6.1 Primary driver object

For each auditor, compute feature importance on the **same untouched real holdout rows** for:

- a model trained/conditioned on real fit rows;
- a model trained/conditioned on synthetic rows;
- an independent real-trained replicate;
- a null or destroyed-structure control.

The primary importance method should be **holdout permutation loss increase**, grouped by original semantic feature.

Why:

- it is model-agnostic;
- it measures consequences on the actual real holdout;
- it supports conventional and in-context auditors;
- it avoids depending on private internal attention or embedding hooks;
- it is reproducible across models.

Rules:

- one-hot columns are recombined into the original categorical feature;
- missingness indicators are grouped with their source feature unless separately declared;
- correlated features receive a second grouped/conditional permutation analysis;
- SHAP, SAGE, integrated gradients, attention, or model-native importances are sensitivity analyses, not the sole headline definition.

### 6.2 Driver distance

Normalize nonnegative permutation importance into a probability vector:

\[
p_j = \frac{I_j}{\sum_k I_k}.
\]

Compute two complementary distances:

\[
d_{\mathrm{mass}}
=
\frac{1}{2}\sum_j|p^{\mathrm{real}}_j-p^{\mathrm{syn}}_j|
\]

and

\[
d_{\mathrm{rank}} = 1-\operatorname{NDCG}@k.
\]

Use:

\[
d_{\mathrm{driver}}
=
\max(d_{\mathrm{mass}},d_{\mathrm{rank}}).
\]

The maximum prevents a method from hiding poor rank recovery behind a similar diffuse mass distribution, or vice versa.

Choose:

\[
k=\max\left(1,
\min\left(d_{\mathrm{active}},
\min(20,\max(5,\lceil0.1d\rceil))\right)\right).
\]

### 6.3 Normalize against real-real stability

Feature importance is noisy. A synthetic-trained model should not be held to a standard that two real-trained models cannot meet.

For each auditor and dataset estimate:

- \(d_{\mathrm{rr}}\): real-trained versus independent real-trained driver distance;
- \(d_{\mathrm{null}}\): real-trained versus null/destroyed-structure driver distance;
- \(d_{\mathrm{syn}}\): real-trained versus synthetic-trained driver distance.

Define:

\[
D_{d,m,s}(A)
=
\operatorname{clip}_{[0,1]}
\left(
\frac{d_{\mathrm{null}}-d_{\mathrm{syn}}}
     {d_{\mathrm{null}}-d_{\mathrm{rr}}}
\right).
\]

Interpretation:

- 1.0 means synthetic-to-real driver agreement is as good as real-to-real reproducibility;
- 0.0 means it is no better than a destroyed-structure control.

This is substantially more defensible than an uncalibrated weighted blend of NDCG, Jaccard, L1 agreement, and sign agreement.

### 6.4 Directional effects

For applicable numeric features, report direction agreement using a signed holdout effect:

- average local effect slope;
- monotonic response slope;
- or centered finite-difference response.

Do not require a sign for:

- unordered categoricals;
- inactive features;
- features whose real-trained sign is unstable across real replicates.

Directional agreement should be:

- a gate on stable top drivers;
- a diagnostic table;
- not an unconditional weighted term.

### 6.5 Driver aggregation

Define:

\[
D_B = \min_{f \in \mathcal{F}}
\operatorname{LCB}_{95\%}\left(D_{B,f}\right).
\]

Also report:

- top-k overlap;
- mass fidelity;
- rank fidelity;
- directional agreement;
- real-real ceiling;
- null floor.

---

## 7. Certified Learning Preservation

At each byte budget \(B\):

\[
\operatorname{CLP}(B)=\min(T_B,D_B)
\]

only when all gates pass.

If any required gate is unknown or failed:

\[
\operatorname{CLP}(B)=\text{null}.
\]

This one line communicates the entire idea:

> **CLP is the weaker of “what models can predict” and “which drivers they rely on.”**

A method with 95% transfer and 62% driver preservation has CLP = 62%, not a weighted average that disguises the failure.

---

## 8. Learning-Preserving Description Length

Use a canonical budget ladder:

```text
1 KiB
2 KiB
4 KiB
8 KiB
10 KiB
16 KiB
32 KiB
64 KiB
128 KiB
256 KiB
512 KiB
1 MiB
2 MiB
4 MiB
8 MiB
16 MiB
32 MiB
64 MiB
128 MiB
256 MiB
512 MiB
1 GiB
```

For each method, fit or compress an artifact under each supported budget.

Define:

\[
\operatorname{LPDL}_{90/90}
=
\min\{|A|:T(A)\ge0.90,\ D(A)\ge0.90,\ G(A)=1\}.
\]

Rules:

1. Use the **actual charged artifact size**, not the nominal budget ceiling.
2. Do not interpolate between unmeasured budgets for the headline.
3. If the threshold is not reached, report `> maximum measured budget`.
4. If gates are incomplete, report `not certified`, not a numeric LPDL.
5. Report LPDL-80/80, LPDL-90/90, and LPDL-95/95 as a threshold table.
6. Use LPDL-90/90 as the primary public result.

---

## 9. Secondary metrics

The public hierarchy should be:

### Primary

- `LPDL-90/90` — lower is better.

### Complementary fixed-budget KPI

- `CLP@10KiB` — higher is better.

### Curve summary

- area under CLP versus log2 bytes, with a predeclared byte interval.

### Diagnostics

- transfer by auditor;
- driver fidelity by auditor;
- distribution fidelity;
- structural fidelity;
- support precision/recall;
- sample-size sensitivity;
- generation latency;
- compile cost;
- memory;
- artifact stability;
- empirical privacy attacks;
- formal-DP parameters where applicable;
- lossless-source compression ratio;
- bytes per feature;
- bytes per stable active driver;
- encoded-value counts;
- quantization breakdown.

No diagnostic should be allowed to rescue a failed primary constraint.

---

# Part III — Byte accounting

## 10. Two byte scopes

A world-class benchmark must prevent both obvious and subtle size hiding.

### 10.1 Dataset-state bytes — primary x-axis

Charge every byte that varies with the fitted dataset and is needed to sample:

- learned parameters;
- codebooks;
- quantiles;
- marginal tables;
- category dictionaries or hashes;
- feature schema;
- missingness metadata;
- normalization and inverse transforms;
- target transforms;
- dependency structures;
- random seeds required for exact replay;
- sampler configuration;
- dataset-specific generated code;
- sidecars;
- checksums and framing;
- compressed payloads;
- calibration state.

Call this:

\[
B_{\mathrm{instance}}.
\]

This is the primary rate axis because it measures how compactly the method encodes the source table’s learned structure.

### 10.2 Cold-start closure bytes — required secondary axis

Charge everything needed to sample on a clean, declared base system:

- dataset-state bytes;
- method runtime;
- nonstandard shared libraries;
- pretrained checkpoints needed by the sampler;
- decoder code;
- external dictionaries;
- tokenizer or encoder assets;
- model configuration assets.

Call this:

\[
B_{\mathrm{closure}}.
\]

Compiler-only models and auditor checkpoints are not charged when they are unnecessary for sampling. A pretrained checkpoint required only during compilation is compile cost, not sampler state. A pretrained checkpoint required during sampling belongs in closure bytes.

### 10.3 Amortized deployment bytes

For systems that share a large runtime or checkpoint across many datasets, report:

\[
B_K
=
B_{\mathrm{instance}}
+
\frac{B_{\mathrm{shared}}}{K}
\]

for:

```text
K = 1, 10, 100, 1,000, 10,000 datasets
```

This prevents two opposite distortions:

- pretending a 2 GB shared checkpoint is free;
- charging the entire generic checkpoint to every small table forever.

The paper’s primary plot should use instance bytes. A secondary panel should show cold-start or amortized deployment bytes.

---

## 11. Canonical serialization contract

Create `production/lpdl-byte-profile-v1.json`.

Requirements:

1. deterministic section ordering;
2. deterministic numeric endianness;
3. explicit quantization width;
4. explicit entropy-coder version;
5. no filesystem metadata;
6. no nondeterministic timestamps;
7. no uncharged sidecars;
8. a complete byte ledger;
9. a content hash;
10. a clean-room sampling test.

Recommended ledger:

```json
{
  "format": "dope-byte-ledger-v1",
  "artifact_sha256": "...",
  "instance_bytes": 1944,
  "closure_bytes": 4213376,
  "sections": [
    {
      "name": "schema",
      "bytes": 184,
      "dataset_specific": true,
      "required_for_sampling": true
    },
    {
      "name": "marginals",
      "bytes": 704,
      "dataset_specific": true,
      "required_for_sampling": true
    }
  ],
  "sidecars": [],
  "external_assets": [],
  "decoder_manifest_sha256": "...",
  "replay_verified": true
}
```

Acceptance test:

```text
bytes reported by compiler
= bytes reported by inspect
= filesystem length
= sum of charged sections
= bytes read by clean-room sampler
```

All five must match exactly.

---

## 12. Encoded values

Keep encoded-value counts as engineering diagnostics, with stronger definitions:

```text
encoded_scalar_count
effective_parameter_bits
nonzero_parameter_count
codebook_entry_count
structural_edge_count
literal_payload_bytes
procedural_payload_bytes
```

Do not use `encoded_numbers` as a cross-method quality score.

A fair “effective parameter bits” measure must be derived from actual serialized sections:

\[
\text{effective parameter bits}
=
8\times \text{serialized parameter-section bytes}.
\]

A one-byte quantized coefficient and an eight-byte float must not count equally.

---

## 13. Raw-table controls

For each dataset report:

- canonical uncompressed table bytes;
- canonical lossless-compressed table bytes;
- artifact-to-lossless-table ratio;
- memorizer/replay control size;
- predictor-only model size.

This protects the claim from pathological cases where:

- the artifact is larger than the source;
- a tiny table can simply be memorized;
- the predictor carries all utility and the sampler adds no value.

The compressed-original table is a **size reference**, never a valid synthetic generator, because it fails anti-copy gates.

---

# Part IV — Auditor architecture

## 14. Core and frontier auditor panels

Do not hard-code one model forever as “the most powerful.” Tabular foundation models are moving too rapidly.

Use two panels.

### 14.1 Core reproducibility panel

This panel changes only at major metric versions.

Recommended families:

| Family | Required auditors |
|---|---|
| Transparent | Ridge/elastic net; GAM or Explainable Boosting Machine |
| Boosted trees | CatBoost; LightGBM; XGBoost |
| Neural tabular | RealMLP or TabM; one transformer-style learner |
| Foundation/in-context | KumoTabular; TabICLv2; Mitra-v2 |

A release may use one declared representative per family for the strict scalar and run the remaining models as robustness checks, but the mapping must be preregistered.

### 14.2 Frontier sensitivity panel

Update this panel more frequently without silently changing the primary score.

Candidate models include:

- TabFM;
- TabPFN;
- LimiX-2;
- EXAONE-Tabular;
- newly released open-weight tabular foundation models;
- any later model that materially changes the public accuracy frontier.

For every model pin:

```text
repository
commit
checkpoint digest
checkpoint license
code license
dependency lock
task support
context limit
preprocessing
inference configuration
hardware
determinism status
```

The README may say:

> “Core certification uses the frozen auditor panel. A rolling frontier panel tests whether the result survives the strongest newly available open models.”

This avoids metric drift while still respecting the user’s requirement to evaluate the best current models.

---

## 15. Conventional and in-context protocols

### 15.1 Conventional learners

For each conventional learner:

1. train only on synthetic rows;
2. tune only on synthetic inner folds or fixed author defaults;
3. never read real holdout labels;
4. evaluate once on untouched real holdout;
5. compare against the same learner trained on real fit rows;
6. use identical feature schema and preprocessing;
7. record all seeds and failures.

### 15.2 In-context tabular foundation models

For an in-context auditor:

- synthetic rows are context;
- real holdout rows are queries;
- real rows are the reference context;
- the same query rows and estimator settings are used;
- context subsampling is deterministic and identical in size;
- no real holdout label enters context;
- any context-size limit is declared.

For large datasets, define a deterministic context selection protocol:

- stratified reservoir;
- diversity-preserving coresets selected without holdout labels;
- fixed maximum context count per model.

Report context sensitivity separately.

---

## 16. Auditor adapter interface

Create:

```text
research/benchmark/auditors/
  base.py
  registry.yaml
  linear.py
  gam.py
  catboost.py
  lightgbm.py
  xgboost.py
  realmlp.py
  tabm.py
  kumo_tabular.py
  tabiclv2.py
  mitra_v2.py
  tabfm.py
  tabpfn.py
  limix2.py
```

Required interface:

```python
class Auditor:
    def fit_or_condition(self, X_context, y_context, *, seed): ...
    def predict(self, X_query): ...
    def loss(self, X_query, y_query): ...
    def permutation_importance(
        self,
        X_query,
        y_query,
        *,
        feature_groups,
        repeats,
        seed,
    ): ...
    def signed_effects(self, X_query, *, feature_groups): ...
    def close(self): ...
```

Each adapter must emit a receipt:

```json
{
  "auditor_id": "kumo_tabular_l",
  "adapter_version": 1,
  "source_commit": "...",
  "checkpoint_sha256": "...",
  "code_license": "...",
  "weights_license": "...",
  "seed": 101,
  "context_rows": 800,
  "query_rows": 200,
  "loss": 0.123,
  "driver_receipt_sha256": "...",
  "runtime_seconds": 12.4,
  "peak_memory_bytes": 1234567890,
  "status": "ok"
}
```

Remove machine-local absolute paths and private wrapper assumptions from the scientific path. Every auditor must run from:

- a lockfile;
- a container;
- or a declared environment manifest.

---

## 17. Representation distance

The current idea of comparing internal foundation-model embeddings is scientifically interesting but should not be a 15% public score component.

Reasons:

- hidden layers are implementation-specific;
- a forward-hook location may change across releases;
- row representations may not be semantically comparable between model versions;
- a close hidden mean does not directly establish downstream transfer;
- a method could optimize toward a particular encoder representation.

Reclassify representation distance as:

- an exploratory diagnostic;
- an ablation;
- a possible early-stop proxy;
- not a primary release score.

If retained:

1. pin the exact layer and preprocessing;
2. validate against transfer and driver fidelity;
3. use multiple encoders;
4. report sensitivity;
5. never call it general “semantic closeness.”

---

# Part V — Benchmark redesign

## 18. Dataset cohorts

The current PMLB-heavy regression panel is useful as a development artifact, but it cannot carry the complete public claim.

Create four explicit cohorts.

### 18.1 RealCore

A license-cleared, deduplicated real-world suite spanning:

- regression;
- binary classification;
- multiclass classification;
- mixed numeric/categorical data;
- missing values;
- small, medium, and large row counts;
- low and high dimensionality;
- multiple industries and scientific domains.

Real datasets should dominate the headline.

### 18.2 ShiftCore

Tasks with realistic non-IID evaluation:

- temporal forward splits;
- grouped entity splits;
- geographic splits;
- site/hospital/customer splits;
- distribution-shifted holdouts.

This tests whether the artifact preserved generalizable structure rather than row-level interpolation.

### 18.3 MechanismLab

Controlled synthetic datasets should remain, but their role changes:

- interaction order;
- sparse drivers;
- nonlinear monotonic effects;
- XOR and parity;
- heteroscedasticity;
- multimodality;
- missing-not-at-random;
- rare categories;
- confounding;
- redundant/correlated features;
- concept shift.

These are mechanistic probes, not the majority of the headline benchmark.

### 18.4 StressCore

Adversarial and pathological tables:

- \(n \ll d\);
- heavy tails;
- severe imbalance;
- high-cardinality categoricals;
- duplicated rows;
- near-duplicate rows;
- irrelevant columns;
- unstable target signal;
- tiny datasets;
- large datasets;
- mixed scales and units.

---

## 19. Dataset manifest

Create `research/benchmark/datasets/manifest-v1.json`.

Each dataset must include:

```json
{
  "dataset_id": "stable-id",
  "name": "Human-readable name",
  "source_family": "portal-or-study",
  "task": "regression",
  "target": "target_name",
  "rows": 10000,
  "features": 24,
  "numeric_features": 18,
  "categorical_features": 6,
  "license": "SPDX-or-explicit",
  "source_digest": "...",
  "schema_digest": "...",
  "split_type": "grouped",
  "split_seed": 1729,
  "group_column": "restricted-or-hashed",
  "holdout_digest": "...",
  "allowed_for_public_release": true
}
```

Requirements:

- deduplicate derived copies;
- cluster related datasets by source family;
- preserve provenance;
- never average 65 simulations from four generators as 65 independent real-world families;
- publish excluded datasets and reasons;
- lock the suite before final evaluation.

Public benchmark candidates may be sourced from broad inventories such as TALENT and TabArena, but DOPE must independently verify license, provenance, task framing, and duplication.

---

## 20. Splits and selection discipline

Each dataset needs:

```text
fit
inner selection
public validation
sealed final test
```

Rules:

1. compiler architecture research uses development datasets only;
2. byte-budget and hyperparameter selection uses fit/inner selection;
3. public validation may be used for paper iteration only when labeled selection-exposed;
4. sealed final test is opened once per named release;
5. opening the sealed test increments the metric/release version;
6. failed test releases are preserved, not overwritten;
7. no method receives extra tuning after seeing another method’s final result.

A final claim requires a frozen release manifest containing:

```text
code commit
metric contract digest
dataset manifest digest
split digests
auditor pins
generator pins
budget ladder
seeds
timeouts
hardware class
statistical plan
allowed claims
```

---

## 21. Synthetic row counts

Bytes and generated row count are different axes.

Primary protocol:

```text
N_syn = 4 × N_fit
```

subject to declared caps for extremely large tables and foundation-model context limits.

Sensitivity:

```text
1n, 2n, 4n, 8n, 16n
```

Report:

- transfer versus generated rows;
- driver fidelity versus generated rows;
- generation time;
- whether a method benefits only by generating enormous samples.

The primary LPDL figure must hold synthetic row count constant across methods.

---

## 22. Controls

Mandatory controls:

| Control | Purpose |
|---|---|
| Real-trained | Reference predictive skill and driver structure |
| Null predictor | Defines available predictive gain |
| Real-row bootstrap | Diagnoses sample-count effects |
| Independent marginals | Tests whether dependencies matter |
| Predictor-only sampler | Tests whether the feature generator adds value |
| Label-shuffled sampler | Detects target leakage and metric bugs |
| Row memorizer | Exposes anti-copy and reconstruction gates |
| Losslessly compressed source | Size reference only |
| Random compact artifact | Detects byte-only scoring failure |
| Oracle mechanism | MechanismLab ceiling where available |

Positive lift over predictor-only and independent-marginal controls should remain gates.

---

## 23. Generator baseline matrix

Benchmark methods in scientifically meaningful lanes.

### 23.1 Compact explicit models

- independent marginals;
- Gaussian copula;
- vine copula where feasible;
- Chow-Liu/Bayesian-network models;
- ARF;
- synthpop/CART-style sequential synthesis;
- compact probabilistic programs.

### 23.2 GAN and VAE models

- CTGAN;
- TVAE;
- CTAB-GAN+;
- relevant conditional variants.

### 23.3 Diffusion, score, and flow models

- TabDDPM;
- TabSyn;
- TabDiff or equivalent current open implementation;
- STaSy/CoDi where reproducible;
- ForestDiffusion/ForestFlow;
- current tabular flow-matching methods such as TabSynFlow/TabVFM when stable.

### 23.4 Language-model generators

- GReaT;
- REaLTabFormer;
- newer open tabular-language generators when reproducible.

### 23.5 Formal-privacy lane

- MST;
- AIM;
- PrivBayes or equivalent;
- DP-CTGAN/PATE-style methods where code and accounting are valid.

Formal-DP methods must report \((\epsilon,\delta)\) and should not be mixed with empirical privacy as if they were identical guarantees.

### 23.6 DOPE ablations

At minimum:

```text
marginals only
marginals + dependence
predictor only
dependence + predictor
no heteroscedastic noise
no interaction terms
no neural residual
no quantization
no entropy coding
fixed dependence family
fixed target family
training-selected family
byte-aware search disabled
```

Every ablation must report both quality lost and bytes saved.

---

## 24. Fairness modes

Run two comparison tracks.

### Track A — Author-faithful

- official implementation;
- author defaults or official recommended search;
- actual artifact size;
- no artificial shrinking merely to fit DOPE’s budget.

This answers:

> “What quality and size does the method normally ship?”

### Track B — Budget-matched

- same byte ceilings;
- method-specific compression or architecture search permitted;
- no access to test data;
- actual post-serialization bytes verified.

This answers:

> “What is the best each method can do at the same rate?”

Both are necessary. A scaled-down neural baseline must not be presented as the method’s canonical quality without a clear label.

---

## 25. Compute fairness

Track:

```text
compile wall time
GPU/CPU hours
peak RAM/VRAM
tuning trials
failed trials
sampling time
evaluation time
energy estimate where available
```

Do not force equal compute if it destroys author-faithful evaluation. Instead:

- cap per-cell search;
- disclose total research spend;
- provide a compute-matched sensitivity;
- never selectively rerun only weak baseline cells after results are visible.

---

## 26. Statistical analysis

Primary inferential unit: dataset lineage.

Recommended hierarchy:

```text
source family
  dataset lineage
    split
      fit seed
        sample seed
```

Use:

- paired comparisons on shared cells;
- hierarchical cluster bootstrap;
- one-sided LCBs for certification;
- two-sided intervals for descriptive figures;
- predeclared multiplicity handling for confirmatory pairwise claims;
- equivalence tests when claiming “same quality at fewer bytes”;
- failure/timeout accounting;
- no cross-cohort ranking when coverage differs.

Report both:

- macro average across lineages;
- stratified results by task, \(n\), \(d\), missingness, categoricals, and shift type.

---

# Part VI — The first-page figure

## 27. Required hero plot

### Title

**Learning preserved versus serialized bytes**

### X-axis

```text
Dataset-specific sampling artifact bytes
log2 scale
```

### Y-axis

```text
Certified Learning Preservation
min(transfer LCB, driver-fidelity LCB)
0% to 100%
```

### Reference lines

- vertical at 10 KiB;
- horizontal at 90%;
- optional horizontal at 95%.

### Target region

Shade the top-left region:

```text
≥90% learning preserved
≤10 KiB
```

### Marks

- one connected rate-distortion curve per method;
- actual serialized sizes, not requested budgets;
- filled point: all gates complete and pass;
- hollow point: scientific quality measured but release evidence incomplete;
- crossed point: privacy/copy gate failed;
- gray point: incomplete method coverage;
- error ribbon: hierarchical uncertainty;
- direct labels at endpoints.

### Anti-clutter rule

The first-page panel should show:

- DOPE;
- best compact explicit comparator;
- strongest forest comparator;
- strongest neural/diffusion comparator;
- strongest language-model comparator if complete;
- real-row and predictor-only controls as reference lines.

The complete method matrix belongs in the next figure or supplement.

### Caption

Use a caption close to:

> **Minimum serialized state required to preserve what models learn.** Each curve shows the conservative lower bound on the weaker of downstream transfer and feature-driver preservation across the frozen auditor families. The vertical line is the 10 KiB target; the horizontal line is 90% preservation. Filled points pass replay, byte-accounting, anti-copy, and empirical privacy gates. Hollow points remain uncertified. Lower and farther left is better.

### Headline sentence

Only after sealed evidence exists:

> “DOPE reaches 90/90 learning preservation in **X KiB**, versus **Y MiB** for the strongest qualifying open-source baseline.”

Until then:

> “Current validation establishes a promising transfer-per-byte frontier; LPDL certification remains pending.”

---

## 28. Supporting figure

A compact two-column panel:

### Left

Transfer preservation by auditor family at the selected artifact.

### Right

Driver preservation by auditor family at the selected artifact.

This prevents the minimum operator from hiding which constraint is limiting.

---

## 29. Figure data contract

Create:

```text
evidence/releases/<release>/figures/figure1-frontier.csv
evidence/releases/<release>/figures/figure1-frontier.json
```

Required columns:

```text
release_id
method_id
configuration_id
budget_bytes
actual_instance_bytes
closure_bytes
task_scope
auditor_family
transfer_mean
transfer_lcb
driver_mean
driver_lcb
clp
coverage
gate_status
gate_failures
source_manifest_sha256
```

The README PNG, paper PDF/SVG, and interactive web version must all be generated from the same file.

---

# Part VII — Generator and compiler engineering

## 30. Change the compiler objective

Current search logic should move toward constrained Pareto optimization.

At each byte budget:

\[
\max_A \operatorname{ProxyCLP}(A)
\quad
\text{subject to}
\quad
|A|\le B.
\]

Do not optimize the final sealed auditors directly on public validation. Use:

- training-only proxies;
- inner-validation auditors;
- inexpensive surrogate models;
- cross-validation inside fit data.

The full frozen auditor panel is for evaluation, not architecture search.

---

## 31. Candidate artifact families

A world-class compact compiler should search a diverse, byte-aware library.

### Feature marginals

- quantile knots;
- compact parametric distributions;
- histogram codebooks;
- spline CDFs;
- count-sketch category tables;
- missingness models;
- shared codebooks across features.

### Dependence

- independence;
- Chow-Liu trees;
- sparse Gaussian copulas;
- factor copulas;
- low-rank Gaussian latent models;
- truncated vines;
- sparse autoregression;
- mixture components;
- dependency graphs;
- compact conditional trees.

### Target models

- constant/null;
- sparse linear and logistic;
- GAM;
- GA²M;
- MARS;
- monotonic splines;
- oblivious trees;
- tiny boosted ensembles;
- low-rank polynomial interactions;
- compact residual MLPs;
- tiny mixture-of-experts;
- task-specific classification heads.

### Noise and uncertainty

- homoscedastic residual;
- heteroscedastic scale;
- quantile residual;
- class-conditional noise;
- calibrated label sampling;
- rare-event tail components.

The search should select the smallest structure that meets inner transfer and driver criteria, not assume one fixed 12-input residual is universally sufficient.

---

## 32. Byte-aware model selection

Each candidate must be serialized before selection.

Never estimate size from parameter counts alone.

Candidate receipt:

```json
{
  "candidate_id": "...",
  "family": "ga2m",
  "inner_transfer": 0.91,
  "inner_driver": 0.88,
  "proxy_clp": 0.88,
  "actual_bytes": 1824,
  "sections": {
    "schema": 160,
    "marginals": 640,
    "dependence": 424,
    "target": 520,
    "noise": 80
  },
  "eligible_under_budget": true
}
```

Selection should use:

1. gate feasibility;
2. highest proxy CLP;
3. smallest actual bytes inside an equivalence margin;
4. lowest decoder complexity;
5. deterministic tie-break.

---

## 33. Compression techniques

Prioritize methods that reduce actual sampler bytes:

- post-training quantization;
- quantization-aware training;
- shared scales;
- delta coding;
- canonical variable-length integers;
- sparse index coding;
- low-rank factorization;
- coefficient pruning;
- knot merging;
- codebook sharing;
- entropy coding;
- symmetry reuse;
- procedural generation from charged seeds;
- section-level rate allocation;
- joint optimization of marginal/dependence/target budget.

Every technique requires an ablation:

```text
bytes saved
transfer change
driver change
privacy change
decode latency change
```

---

## 34. Rate allocation

Treat artifact sections as a constrained resource allocation problem.

For total budget \(B\):

\[
B =
B_{\mathrm{schema}}
+
B_{\mathrm{marginal}}
+
B_{\mathrm{dependence}}
+
B_{\mathrm{target}}
+
B_{\mathrm{noise}}
+
B_{\mathrm{framing}}.
\]

Learn or search section allocations rather than using fixed quotas.

Possible approach:

1. generate a candidate ladder per section;
2. estimate marginal CLP gain per added byte;
3. solve a knapsack or beam-search allocation;
4. serialize;
5. evaluate inner proxy;
6. refine around the Pareto frontier.

This is a direct engineering realization of “more useful learned structure per bit.”

---

## 35. Task support

The flagship release should support:

- numeric regression;
- binary classification;
- multiclass classification;
- mixed numeric/categorical features;
- missing values.

Later:

- multi-target tasks;
- ordinal outcomes;
- count outcomes;
- censored/survival outcomes;
- grouped and relational tables;
- time series.

Do not broaden the paper claim ahead of measured support.

---

## 36. Decoder and artifact robustness

Required tests:

- property-based parsing;
- fuzzing;
- checksum corruption;
- truncated file handling;
- unknown section handling;
- version migration;
- deterministic sampling;
- cross-platform replay;
- endianness;
- denial-of-service bounds;
- malformed categorical codebooks;
- NaN/Inf refusal;
- integer overflow;
- decompression bombs;
- memory ceilings;
- no source-path leakage;
- no header leakage;
- no cleartext category leakage.

A corrupted artifact must fail closed.

---

# Part VIII — Privacy and anti-memorization

## 37. Privacy is a gate, not a soft bonus

A method should not earn a high score because it is useful and compact while leaking rows.

Required gates:

1. exact row matches;
2. near-copy distance;
3. duplicate-aware nearest-neighbor analysis;
4. membership inference;
5. attribute inference;
6. cleartext scan;
7. category-dictionary scan;
8. canary exposure;
9. reconstruction attack;
10. artifact extraction attack;
11. sampler-only attack;
12. seed leakage and deterministic replay analysis.

Evaluate attacks against:

- the serialized artifact;
- generated samples;
- repeated adaptive queries to the sampler.

Report empirical privacy honestly:

> Passing these attacks is not differential privacy and is not a HIPAA de-identification claim.

---

## 38. Formal privacy lane

Where a method is formally differentially private:

- report \(\epsilon\), \(\delta\), adjacency definition, accountant, clipping, and composition;
- verify the implementation;
- keep formal-DP results in a distinct panel;
- compare utility/bytes at multiple privacy budgets.

DOPE should not imply formal privacy without a formal mechanism and proof/accounting.

---

# Part IX — README rebuild

## 39. README target

Target:

```text
180–250 lines
one dominant idea
one hero plot
one quick start
one status block
```

Everything else moves to docs.

### Recommended README order

1. title and one-sentence thesis;
2. hero plot;
3. one measured headline and one honest status line;
4. three-command quick start;
5. what the artifact contains;
6. LPDL in one equation;
7. benchmark protocol in five bullets;
8. scope and limitations;
9. links to paper, docs, and reproduction;
10. citation and license.

---

## 40. Proposed README opening

```markdown
# DOPE

### Preserve what models learn—in kilobytes.

DOPE compiles a tabular dataset into a small executable generator. We measure
the minimum serialized bytes required for models trained on its samples to
recover both the predictive skill and the important feature drivers learned
from real data, evaluated on untouched real rows.

[hero plot]

**Primary metric:** LPDL-90/90 — the smallest artifact that preserves at least
90% of downstream transfer and 90% of feature-driver structure while passing
replay, anti-copy, and privacy gates.

**Current status:** validation-only. The sealed LPDL release protocol has not
yet produced a certified score. Historical CatBoost transfer results remain
available in the evidence archive.
```

Do not lead with:

- an internal audit badge;
- binary codec details;
- historical version compatibility;
- campaign accounting;
- a seven-component formula.

---

## 41. Quick start

The quick start must run on a clean machine and produce:

```text
artifact
synthetic sample
inspection report
```

Ideal interface:

```bash
uv tool install dope-kernel
dope compile train.csv --target outcome --budget 10KiB --out model.dpk
dope sample model.dpk --rows 10000 --out synthetic.csv
dope inspect model.dpk
```

If package installation is not ready, provide a release binary or container. “Clone, configure CUDA, build multiple components, and prepare research dependencies” is not a first-run experience.

---

## 42. Move out of README

Move these to dedicated references:

| Current material | Destination |
|---|---|
| Binary format history | `docs/reference/artifact-format.md` |
| All CLI commands | `docs/reference/cli.md` |
| GPU training | `docs/reference/gpu.md` |
| Deep-campaign commands | `docs/research/deep-campaign.md` |
| Router/embedding internals | `docs/research/router.md` |
| Historical metrics | `docs/archive/metrics/` |
| Full benchmark tables | `docs/benchmark/results/` |
| Audit implementation | `docs/reproducibility/conformance.md` |
| Operational ETAs | issue tracker or private operations log |

---

## 43. Badges

Recommended badges:

- CI;
- reproducible release status;
- current metric version;
- license;
- paper;
- latest release.

Avoid a project-specific badge that a new reader cannot interpret before understanding the project.

---

# Part X — Paper rebuild

## 44. Recommended title

Strongest option:

# Learning-Preserving Description Length for Synthetic Tables

Subtitle or method line:

> **DOPE: executable tabular generators measured in bytes**

Alternative:

# DOPE: Learning-Preserving Tabular Generators in Kilobytes

The first title makes the scientific contribution broader than one implementation.

---

## 45. Abstract structure

The abstract must contain exactly five moves:

1. **Problem:** synthetic-data quality is usually evaluated without charging the size of the learned generator.
2. **Metric:** LPDL is the minimum serialized sampler state needed to preserve predictive transfer and model-driver structure on unseen real rows.
3. **Method:** DOPE compiles tables into compact marginal, dependence, target, and noise programs under explicit byte budgets.
4. **Evidence:** one sealed, fully qualified headline result across declared datasets, auditors, and baselines.
5. **Limitations:** tasks covered, validation/test status, and privacy scope.

Do not put operational caveats, multiple historical metrics, or a paragraph of comparator verdict macros into the abstract.

---

## 46. Paper outline

### 1. Introduction

- why “looks real” is insufficient;
- why TSTR alone is insufficient;
- why generator size matters;
- one rate-distortion equation;
- contributions.

### 2. Learning-Preserving Description Length

- transfer normalization;
- driver normalization;
- byte scopes;
- hard gates;
- statistical aggregation.

### 3. DOPE compiler

- artifact structure;
- candidate families;
- byte-aware search;
- deterministic serialization.

### 4. Experimental protocol

- datasets;
- splits;
- auditors;
- baselines;
- budgets;
- controls;
- privacy;
- statistics.

### 5. Main results

- Figure 1 rate-distortion frontier;
- LPDL threshold table;
- auditor-family breakdown;
- dataset-strata breakdown.

### 6. Why it works

- section-byte ablations;
- dependence and target ablations;
- quantization;
- sample-size sensitivity;
- failure analysis.

### 7. Privacy, limits, and responsible interpretation

- empirical versus formal privacy;
- selection exposure;
- task limitations;
- foundation-model licensing and drift;
- causal limitations.

### 8. Related work

- synthetic data;
- dataset distillation;
- knowledge distillation;
- MDL;
- rate-distortion;
- tabular foundation models;
- privacy.

### 9. Conclusion

One clear statement, not a second results section.

---

## 47. Contributions

Use contributions like:

1. **A rate-distortion metric for synthetic tables.**  
   LPDL measures the smallest serialized sampler that preserves both predictive transfer and feature-driver structure.

2. **A reproducible multi-auditor protocol.**  
   Conventional and in-context tabular learners are evaluated on the same unseen real rows with real-real and null calibration.

3. **A compact executable generator.**  
   DOPE searches quantized marginal, dependence, target, and noise programs under explicit byte budgets.

4. **An open benchmark and evidence ledger.**  
   Every figure point is linked to a frozen dataset, method, artifact, auditor, and gate receipt.

Do not claim state of the art until the frozen comparison supports the exact wording.

---

## 48. Template and visual style

Unless the paper is actively submitted to a named venue, use a neutral preprint template. A venue-branded class can imply a status that does not exist.

Design:

- one visual idea per figure;
- direct labeling;
- restrained colors;
- color-blind-safe palette;
- DOPE emphasized, controls gray, comparators grouped;
- no dense tables on page one;
- no internal operational terminology;
- no “master fitness” language.

---

## 49. Supplement

Move into the supplement:

- all dataset rows;
- all method configurations;
- full gate matrix;
- licensing table;
- exact statistical plan;
- all auditor pins;
- complete byte ledgers;
- failure and timeout tables;
- historical metric mapping;
- detailed privacy attacks;
- additional figures;
- reproducibility instructions.

Move operational planning and ETAs out of both paper and supplement.

---

# Part XI — Repository architecture

## 50. Proposed top-level structure

```text
README.md
STATUS.md
CHANGELOG.md
CONTRIBUTING.md
SECURITY.md
CODE_OF_CONDUCT.md
CITATION.cff
LICENSE

production/
  lpdl-contract-v1.json
  lpdl-byte-profile-v1.json
  auditor-panel-v1.json
  privacy-gates-v1.json

rust/
  transfer_metrics/
    transfer_v1.rs
    driver_v1.rs
    lpdl_v1.rs
    privacy_v1.rs
  fitness/
    lpdl_v1.rs
  byte_ledger.rs

research/
  benchmark/
    auditors/
    generators/
    datasets/
    metrics/
    statistics/
    manifests/
    tests/
  results/
    development/
    validation/
    sealed/

evidence/
  status.json
  claims/
  releases/
    v0.4.0/
      manifest.json
      artifacts/
      receipts/
      figures/
      tables/

docs/
  concepts/
    learning-preserving-description-length.md
    predictive-transfer.md
    driver-preservation.md
  benchmark/
    protocol.md
    datasets.md
    methods.md
    statistics.md
    privacy.md
  reproducibility/
    quick-reproduce.md
    environments.md
    evidence-ledger.md
    sealed-release.md
  reference/
    cli.md
    artifact-format.md
    byte-accounting.md
    gpu.md
  research/
    deep-campaign.md
    router.md
  archive/
    metrics/
      mfs-v2/
      mfs-v3/
      mfs-v4/
    papers/
    historical-results/
  whitepaper/
    main.tex
    supplement.tex
    references.bib
    figures/
    generated/
```

---

## 51. One source of public truth

Generate all public numbers from one release manifest.

Create:

```text
scripts/build_status.py
scripts/verify_claims.py
scripts/reproduce_figure1.py
scripts/reproduce_paper.py
```

`README.md`, `STATUS.md`, and the paper should contain generated blocks:

```markdown
<!-- BEGIN GENERATED RELEASE STATUS -->
...
<!-- END GENERATED RELEASE STATUS -->
```

CI must fail when:

- a displayed number is not in the claim registry;
- README and paper disagree;
- a claim points to a different metric version;
- a figure digest is stale;
- a method label changes without a manifest change;
- a null score is rendered as numeric;
- a validation result is labeled final;
- a test partition was opened outside a release manifest.

---

## 52. Claim registry

Create `evidence/claims/registry.json`.

Example:

```json
{
  "claim_id": "figure1.dope.lpdl90",
  "release_id": "v0.4.0",
  "metric": "lpdl-v1",
  "value": 8192,
  "unit": "bytes",
  "status": "sealed-test",
  "source_pointer": "releases/v0.4.0/results.json#/methods/dope/lpdl_90_90",
  "source_sha256": "...",
  "allowed_surfaces": ["README", "paper", "website"],
  "wording": "DOPE reaches LPDL-90/90 in 8 KiB"
}
```

No hand-edited public numbers.

---

## 53. Fixtures versus evidence

Synthetic conformance fixtures must be unmistakable.

Use:

```text
tests/fixtures/
  PERFECT_SCORE_SYNTHETIC_FIXTURE.json
  ZERO_SCORE_SYNTHETIC_FIXTURE.json
```

Each file must contain:

```json
{
  "synthetic_fixture": true,
  "eligible_for_public_claims": false
}
```

CI must reject fixtures from any evidence or paper-generation path.

---

# Part XII — Reproducibility and open-source quality

## 54. Reproduction tiers

### Tier 0 — Read

- README;
- paper;
- static evidence;
- no special environment.

### Tier 1 — Smoke

- tiny public dataset;
- compile;
- sample;
- inspect;
- local transfer test;
- CPU where possible.

### Tier 2 — Figure reproduction

- download frozen scalar receipts;
- regenerate Figure 1 and tables;
- no full retraining.

### Tier 3 — Benchmark replay

- run selected datasets and auditors;
- containerized;
- pinned model weights.

### Tier 4 — Full campaign

- all methods;
- all datasets;
- declared compute;
- sealed-test credentials controlled.

Each tier should have one command.

---

## 55. Commands

Recommended:

```bash
just smoke
just verify-contract
just verify-bytes
just benchmark-mini
just reproduce-figure1
just paper
just verify-public-claims
just release-audit
```

`just paper` should never silently download live mutable inputs.

---

## 56. Environment

Provide:

- lockfile;
- OCI container digest;
- CUDA and CPU variants;
- model-weight manifest;
- Software Bill of Materials;
- third-party license inventory;
- hardware metadata;
- deterministic settings;
- offline mode.

Absolute local paths must not appear in release code.

---

## 57. Governance

Ensure or strengthen:

- contribution guide;
- code of conduct;
- security policy;
- issue templates;
- benchmark-method request template;
- dataset addition template;
- result dispute process;
- responsible disclosure for privacy failures;
- release signing;
- provenance attestations.

---

# Part XIII — Detailed implementation backlog

## 58. P0 — Establish one truth

### METRIC-001 — Freeze LPDL-v1 terminology

**Deliverables**

- `production/lpdl-contract-v1.json`
- `docs/concepts/learning-preserving-description-length.md`
- metric glossary.

**Acceptance**

- one active metric name across README, CLI, paper, and status;
- all MFS references outside archive are historical labels only.

---

### STATUS-001 — Generated status

**Deliverables**

- `evidence/status.json`
- generated `STATUS.md`
- README status block.

**Acceptance**

A new reader can answer in under 30 seconds:

- What is measured?
- What is only proposed?
- Are tests sealed?
- Which scores are null?
- Which result is validation-only?
- What is the current release metric?

---

### CLAIM-001 — Claim registry

**Deliverables**

- registry schema;
- claim verifier;
- generated README/paper blocks.

**Acceptance**

CI rejects any public numeric claim without a source digest and evidence status.

---

### DOCS-001 — README reduction

**Deliverables**

- rewritten README;
- moved reference material;
- stable links.

**Acceptance**

- no more than 250 lines excluding generated table/figure markup;
- first screen explains the thesis;
- quick start works;
- only one status statement;
- no internal campaign accounting.

---

### PAPER-001 — Paper metric migration

**Deliverables**

- new title;
- LPDL section;
- new abstract;
- new Figure 1 specification;
- MFS history moved to supplement/archive.

**Acceptance**

README and paper define the same metric with the same notation.

---

### EVIDENCE-001 — Separate fixtures

**Deliverables**

- test fixtures moved;
- metadata marking them non-citable;
- CI exclusion.

**Acceptance**

No synthetic conformance example can enter a figure or claim.

---

## 59. P1 — Implement the measurement core

### BYTES-001 — Canonical byte ledger

**Deliverables**

- Rust byte-ledger implementation;
- JSON schema;
- section accounting;
- clean-room sampler trace.

**Acceptance**

Compiler, inspect, filesystem, ledger sum, and sampler-read count agree exactly.

---

### BYTES-002 — Closure accounting

**Deliverables**

- runtime/checkpoint asset manifest;
- instance, closure, and amortized byte reports.

**Acceptance**

A method cannot hide required sampling state in an external checkpoint or sidecar.

---

### METRIC-002 — Transfer v1

**Deliverables**

- loss/score normalization;
- low-signal handling;
- unclamped receipts;
- family aggregation;
- LCB computation.

**Acceptance**

Golden tests match in Rust and Python.

---

### METRIC-003 — Driver v1

**Deliverables**

- grouped permutation importance;
- real-real calibration;
- null calibration;
- rank and mass distance;
- signed effects.

**Acceptance**

Known synthetic mechanisms recover the correct drivers; label-shuffled controls fail.

---

### METRIC-004 — CLP and LPDL

**Deliverables**

- CLP calculation;
- threshold search;
- null propagation;
- budget-ladder reports.

**Acceptance**

No score is emitted when a required family or gate is incomplete.

---

### AUDITOR-001 — Adapter ABI

**Deliverables**

- base adapter;
- receipts;
- registry;
- environment pins.

**Acceptance**

Every auditor runs through the same interface without machine-local paths.

---

### AUDITOR-002 — Core panel

**Deliverables**

- transparent;
- boosted-tree;
- neural;
- foundation-model adapters.

**Acceptance**

A smoke cohort completes with identical holdout rows and valid driver receipts.

---

## 60. P2 — Build the rate-distortion benchmark

### DATA-001 — RealCore manifest

**Acceptance**

- task diversity;
- license verification;
- source-family deduplication;
- real datasets dominate headline.

### DATA-002 — ShiftCore and MechanismLab

**Acceptance**

- at least one temporal, grouped, and geographic split;
- controlled driver-ground-truth cases.

### BASELINE-001 — Author-faithful lane

**Acceptance**

- exact source commits;
- complete artifact accounting;
- no unlabeled scaled schedules.

### BASELINE-002 — Budget-matched lane

**Acceptance**

- post-serialization size enforcement;
- same budget ladder;
- no holdout leakage.

### FIGURE-001 — Hero frontier

**Acceptance**

- one source CSV;
- README PNG;
- paper vector graphic;
- uncertainty and gates visible;
- incomplete methods cannot appear as certified.

### STATS-001 — Hierarchical inference

**Acceptance**

- family clustering;
- paired cells;
- one-sided certification bounds;
- failure accounting;
- reproducible bootstrap seeds.

---

## 61. P3 — Improve the compiler

### COMPILER-001 — Section-level rate allocation

### COMPILER-002 — Quantization-aware target models

### COMPILER-003 — Shared codebooks and sparse dependence

### COMPILER-004 — Classification support

### COMPILER-005 — Pareto beam search

### COMPILER-006 — Artifact hardening and fuzzing

Each compiler change must prove:

```text
quality change
byte change
privacy change
runtime change
```

No architecture is promoted solely because it improves an internal proxy.

---

## 62. P4 — Sealed release

### RELEASE-001 — Freeze manifest

### RELEASE-002 — Independent replay

### RELEASE-003 — Open sealed test once

### RELEASE-004 — Publish signed artifacts

### RELEASE-005 — Submit paper only after claim freeze

Acceptance:

- every public result is traceable;
- no post-test tuning;
- independent reproduction matches within declared tolerance;
- all gate failures remain visible;
- release assets are signed.

---

# Part XIV — Risk register

## 63. Scientific risks

| Risk | Mitigation |
|---|---|
| LPDL threshold is too strict and most methods are null | This is informative; report CLP curves and lower thresholds without changing the frozen primary threshold |
| Driver importance is unstable | Normalize against real-real stability and report ceilings |
| Foundation models change rapidly | Frozen core panel plus rolling frontier panel |
| Dataset suite overrepresents easy tasks | Stratified cohorts and source-family weighting |
| DOPE selection overfits validation | Separate development, validation, and sealed release |
| A method optimizes directly to the auditor | Multiple model families and unseen frontier sensitivity |
| Compact artifact hides data in a seed or decoder | Charge all state, reconstruction attacks, closure accounting |
| Large pretrained samplers appear artificially free | Instance plus closure/amortized byte reporting |
| Similar feature importance hides different interactions | Interaction-specific MechanismLab and pair/group importance diagnostics |
| Equal error hides different predictions | Add prediction-agreement diagnostics on real holdout |
| Privacy attacks give false comfort | Explicitly distinguish empirical tests from formal DP |

---

## 64. Engineering risks

| Risk | Mitigation |
|---|---|
| Benchmark complexity explodes | Stable adapter ABI, manifests, staged cohorts |
| GPU models are hard to reproduce | offline weight manifest, container, exact hashes |
| Baseline coverage remains asymmetric | certification requires complete shared cells; partial methods shown separately |
| README regrows | line budget and docs linter |
| Historical receipts break | immutable archive and compatibility verifier |
| Rust/Python metric drift | shared golden corpus and cross-language property tests |
| Figure values drift | one generated data file and claim registry |
| Sidecars escape accounting | syscall/file-open tracing in clean-room sampler |
| Timeouts bias rankings | explicit missingness, fixed caps, no selective retries |

---

# Part XV — Decisions to make once, now

## 65. Recommended decisions

1. **Adopt LPDL-v1 as the public metric.**
2. **Use LPDL-90/90 as the headline threshold.**
3. **Use CLP@10KiB as the fixed-budget companion.**
4. **Make transfer and driver fidelity non-compensatory.**
5. **Demote representation distance to a diagnostic.**
6. **Use exact bytes, not encoded-number counts, as the cross-method rate.**
7. **Report both instance and closure bytes.**
8. **Use a frozen core auditor panel plus a rolling frontier panel.**
9. **Require real-real calibration for driver importance.**
10. **Make real datasets the majority of the headline benchmark.**
11. **Run author-faithful and budget-matched baseline tracks.**
12. **Archive MFS-v2/v3/v4 rather than continuing public version accretion.**
13. **Generate README and paper numbers from one release ledger.**
14. **Do not open the sealed test until the entire release manifest is frozen.**
15. **Do not claim privacy beyond measured empirical gates or formal DP.**

---

# Part XVI — Things not to do

## 66. Prohibited shortcuts

- Do not create MFS-v5 by adding more weighted components.
- Do not use one foundation-model hidden layer as a universal semantic truth.
- Do not call a partial validation panel “state of the art.”
- Do not count baseline failures as wins.
- Do not compare methods across different dataset cohorts without explicit qualification.
- Do not call a scaled-down baseline its canonical configuration.
- Do not use artifact bytes without charging dataset-specific transforms and sidecars.
- Do not call empirical attacks “anonymous,” “de-identified,” “HIPAA safe,” or “differentially private.”
- Do not normalize bytes by duplicated rows and call the result more informative.
- Do not let generated row count vary in the headline comparison.
- Do not tune on the official holdout.
- Do not put ETAs, queue state, or machine-local paths into the scientific paper.
- Do not expose three active metric versions.
- Do not bury the strongest idea below binary-format details.
- Do not publish a heroic plot until its points are generated from the frozen evidence ledger.

---

# Part XVII — Definition of done

## 67. Public comprehension

A qualified reader can answer these questions after viewing the first page:

1. What does DOPE produce?
2. What is being minimized?
3. What must be preserved?
4. On what data is preservation measured?
5. Which models audit it?
6. What are the privacy limitations?
7. Which result is final versus validation-only?

---

## 68. Scientific completeness

A release is complete only when:

- all required datasets have dispositions;
- all required auditor families have measured transfer;
- all required driver measurements exist;
- byte ledgers reconcile;
- sampling replay passes;
- anti-copy and privacy gates pass;
- baseline configurations are source-pinned;
- the statistical plan is frozen;
- coverage and failures are visible;
- the sealed test was opened only through the release manifest;
- the claim registry validates.

---

## 69. Engineering completeness

- clean install;
- three-command quick start;
- deterministic artifact;
- Rust/Python metric parity;
- no machine-local paths;
- offline auditor execution;
- signed release;
- SBOM;
- license inventory;
- reproducible Figure 1;
- reproducible paper;
- archived historical contracts;
- fuzzed decoder;
- CI claim verification.

---

# Part XVIII — The final story

The project should ultimately be explainable in six sentences:

> DOPE compiles a real table into a tiny executable generator.  
> Models learn only from rows emitted by that generator.  
> We evaluate those models on untouched real data.  
> We measure both how much predictive skill transfers and whether the same feature drivers remain important.  
> We then ask for the smallest serialized artifact that preserves both while passing anti-copy, replay, and privacy gates.  
> That minimum size is Learning-Preserving Description Length.

That is the profound paper.

Not a giant quality checklist.  
Not a CatBoost benchmark with a byte column.  
Not a weighted synthetic-data score.  
Not “the rows look realistic.”

The core scientific object is a **rate-distortion frontier for transferable learned structure**.

And the clearest possible claim is:

> **How little code and data can carry what the original table taught the world’s strongest models?**

---

# Appendix A — Draft LPDL contract

```json
{
  "format": "dope-lpdl-contract",
  "version": 1,
  "headline": {
    "metric": "LPDL-90/90",
    "transfer_threshold": 0.90,
    "driver_threshold": 0.90,
    "fixed_budget_companion_bytes": 10240
  },
  "budgets_bytes": [
    1024, 2048, 4096, 8192, 10240, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304,
    8388608, 16777216, 33554432, 67108864, 134217728,
    268435456, 536870912, 1073741824
  ],
  "transfer": {
    "protocol": "train_or_condition_on_synthetic_score_on_real",
    "null_normalized": true,
    "unclamped_receipts": true,
    "certification_statistic": "one_sided_hierarchical_lcb",
    "family_aggregation": "minimum"
  },
  "drivers": {
    "primary": "grouped_holdout_permutation_loss_increase",
    "distance": "max(total_variation, one_minus_ndcg_at_k)",
    "calibration": "real_real_and_destroyed_structure_null",
    "certification_statistic": "one_sided_hierarchical_lcb",
    "family_aggregation": "minimum"
  },
  "rate": {
    "primary": "dataset_specific_instance_bytes",
    "secondary": [
      "cold_start_closure_bytes",
      "amortized_closure_bytes"
    ],
    "actual_serialized_bytes_only": true
  },
  "gates": [
    "evidence_complete",
    "byte_ledger_reconciled",
    "clean_room_sampling_verified",
    "exact_copy_pass",
    "near_copy_pass",
    "cleartext_scan_pass",
    "membership_attack_pass",
    "attribute_attack_pass",
    "reconstruction_attack_pass",
    "positive_lift_over_predictor_only",
    "positive_lift_over_independent_marginals",
    "required_auditor_families_complete"
  ],
  "public_score_when_gate_unknown": null
}
```

---

# Appendix B — Draft release result object

```json
{
  "format": "dope-lpdl-release-result-v1",
  "release_id": "v0.4.0",
  "status": "validation-only",
  "metric_contract_sha256": "...",
  "dataset_manifest_sha256": "...",
  "auditor_manifest_sha256": "...",
  "method": "dope",
  "thresholds": {
    "transfer": 0.9,
    "driver": 0.9
  },
  "lpdl_90_90": null,
  "clp_at_10kib": null,
  "frontier": [
    {
      "instance_bytes": 1944,
      "transfer_lcb": 0.81,
      "driver_lcb": null,
      "clp": null,
      "gate_status": "incomplete",
      "missing": ["driver_panel", "sealed_test"]
    }
  ],
  "claims_allowed": [
    "validation_transfer_per_byte"
  ],
  "claims_forbidden": [
    "LPDL-certified",
    "state-of-the-art",
    "anonymous",
    "differentially-private"
  ]
}
```

---

# Appendix C — Proposed first-page result table

| Method | LPDL-80/80 | LPDL-90/90 | LPDL-95/95 | CLP@10 KiB | Instance bytes | Closure bytes | Gate status |
|---|---:|---:|---:|---:|---:|---:|---|
| DOPE | pending | pending | pending | pending | measured | measured | incomplete |
| Compact statistical best | pending | pending | pending | pending | measured | measured | incomplete |
| ARF | pending | pending | pending | pending | measured | measured | incomplete |
| Best diffusion/flow | pending | pending | pending | pending | measured | measured | incomplete |
| Best language generator | pending | pending | pending | pending | measured | measured | incomplete |

Never populate a threshold with an estimate derived from an incomplete panel.

---

# Appendix D — Audit snapshot informing this specification

The specification was developed after inspecting:

- the root README and current benchmark presentation;
- the MFS-v4 JSON contract;
- the Rust transfer, driver, and compression implementation;
- the MFS-v3 paper source and white-paper guide;
- baseline coverage and evidence-gap documents;
- paper-generation scripts;
- the current auditor implementation for KumoTabular, Mitra-v2, and TabICLv2;
- the repository’s historical contract seam;
- current open implementations and reports for major tabular foundation models;
- current tabular benchmark suites;
- current open-source synthetic-data generators.

The most important observed mismatch is not a missing feature. It is a mismatch of **public abstraction level**:

- the code is trying to become a rigorous scientific system;
- the README reads like a merged research notebook and reference manual;
- the paper still presents an earlier metric generation;
- the public headline does not yet express the strongest idea already latent in the code.

The rebuild should therefore prioritize **semantic compression of the project itself**:

> one metric, one figure, one status, one evidence ledger, one sentence.
