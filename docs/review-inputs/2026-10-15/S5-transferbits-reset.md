# DOPE: TransferBits Research, Product, and Engineering Reset

**Status:** Proposed engineering specification  
**Repository reviewed:** `neverhuman/dope`, `main`, current head observed during review: `ee4a3d0f640088a894926aaa2a268bdc008f0db7`  
**Primary source areas reviewed:** `README.md`, `docs/whitepaper/`, `production/kpi-contract-v4.json`, `rust/fitness/v4.rs`, `rust/transfer_metrics/`, `research/transfer/`, `research/benchmark/`, model locks, baseline ledgers, and paper-generation code.

---

## 1. Executive decision

DOPE should stop presenting itself as a large collection of synthetic-data metrics, campaign machinery, format variants, and release gates.

It should become one clean scientific and product idea:

> **DOPE compiles a table into the smallest generative program that preserves machine-learnable behavior on unseen real data.**

The public question should be:

> **How few bits preserve 90% of the learning?**

The formal research program should be **predictive rate–distortion for tabular data**. The public metric should be **TransferBits**. The compiler should search the bit–transfer frontier. The README and paper should show one unmistakable plot: robust real-world transfer versus complete charged artifact bytes.

The present repository already contains much of the difficult machinery: artifact-only sampling, byte accounting, train-on-synthetic/test-on-real retention, controls, paired evidence, privacy checks, receipts, multi-seed evaluation, modern auditor adapters, and paper regeneration. The problem is that these ingredients are currently exposed as the product. They should instead become the invisible proof system beneath a dramatically simpler claim.

### The central repositioning

Current impression:

> A complicated synthetic-data system with MFS-v2, MFS-v3, MFS-v4, multiple artifact definitions, many gates, campaign commands, auditors, release tiers, and partially populated evidence.

Target impression:

> A compiler that preserves learning in astonishingly few bits, with a benchmark that proves it across the strongest model families.

### The most valuable existing evidence

The current validation evidence is already potentially remarkable: DOPE’s median artifact is approximately 1.9 KiB and its CatBoost transfer is close to TabSyn’s while TabSyn’s median artifact is tens of megabytes. That is approximately four orders of magnitude less storage. However, this is not yet the final headline because:

1. the result is CatBoost-centered rather than cross-model;
2. the paired interval does not establish superiority;
3. the selected DOPE profile is validation-exposed;
4. the official test remains sealed;
5. the modern foundation-model adapters are specified but not yet qualified;
6. driver preservation is not yet measured as a transparent, per-auditor result;
7. privacy and release status must remain separate from scientific utility.

The immediate goal is not to invent a more complicated score. It is to turn the existing insight into a defensible rate–distortion result.

---

## 2. What is currently wrong

### 2.1 The README is six documents pretending to be one

The README currently acts as:

1. a research abstract;
2. a benchmark report;
3. a release and privacy policy;
4. an artifact-format specification;
5. a campaign operator manual;
6. a CLI reference.

This causes the core insight to disappear beneath implementation archaeology.

The first screen should explain the question, the evidence, and the artifact. Everything else should be linked documentation.

### 2.2 The KPI is a weighted score rather than a natural unit

MFS-v4 combines:

- downstream transfer;
- lift over controls;
- feature-attribution agreement;
- representation distance;
- byte size;
- encoded-number count;
- distribution fidelity;
- structure fidelity;
- coverage;
- privacy gates.

Many of these are useful measurements. The problem is the scalarization.

The current formulas contain several sets of manually chosen weights and scales. Those constants are not a scientific unit. A reader can reasonably ask why knowledge compression is worth 0.15, why one attribution component is worth 0.35, or why a 0.20 lift is the saturation point for one control. The resulting score is hard to interpret, hard to compare, and easy to debate.

The profound quantity is not “83.7 MFS.”

The profound quantity is:

> **This artifact needs 1,895 bytes to preserve at least 90% of the real-data learning across independent model families.**

### 2.3 Bytes are treated as one component, but bytes should be the independent variable

Compression should not receive a modest weight inside a utility score. It should define the rate axis.

The scientific object is a frontier:

- **rate:** complete charged bits;
- **distortion:** lost transfer, changed predictions, and changed drivers.

This gives DOPE a direct connection to rate–distortion, minimum description length, sufficient statistics, model compression, dataset distillation, and information bottlenecks without pretending to estimate Kolmogorov complexity or mutual information.

### 2.4 The modern auditor story is ahead of the evidence

The current contract names Kumo Tabular, TabICLv2, Mitra-v2, and CatBoost. That direction is correct. However, the current Kumo, Mitra, and TabICL locks are explicitly unqualified: checkpoint bytes are not locally verified, exact dependency closures are absent, offline parity has not passed, and admission is unresolved.

These methods must be shown as **specified, not measured** until qualification closes.

### 2.5 The paper and README describe different generations of the metric

The README foregrounds MFS-v4, while the main manuscript still explains MFS-v3. This creates immediate reviewer uncertainty about which metric, protocol, and claim are authoritative.

There must be one current benchmark contract, one current paper, and one current public metric.

### 2.6 Artifact identity is ambiguous

The repository currently exposes several objects:

- benchmark `.dpk + projection.json`;
- a tiny toy artifact;
- a native one-file artifact;
- a cold runtime closure that can be much larger;
- manifests and certification sidecars.

These may all be legitimate engineering objects, but the README does not give the reader one stable answer to “How large is a DOPE artifact?”

A canonical artifact and a two-tier byte-accounting policy are required.

### 2.7 The project is target-specific but sometimes sounds table-general

The current generator explicitly stores a target model and screens features using target association. That is a valid and interesting object: a **target-conditioned generative program**.

It is not automatically a general-purpose compressed model of the entire table.

The paper must distinguish:

1. **target-conditioned transfer:** preserve one declared supervised workload;
2. **workload-conditioned transfer:** preserve a portfolio of declared targets and learners;
3. **target-agnostic table synthesis:** preserve arbitrary future uses.

DOPE can dominate the first category without overclaiming the third.

---

## 3. The new scientific object

## 3.1 Formal concept: transfer-sufficient generative description

Let a dataset \(d\) be split into:

- \(D_{\mathrm{fit}}\): available to the generator and real-trained reference models;
- \(D_{\mathrm{select}}\): available only for method selection and tuning;
- \(D_{\mathrm{test}}\): sealed until the benchmark contract is frozen.

Let \(A\) be a generative artifact compiled from \(D_{\mathrm{fit}}\). Let \(G(A,s,n)\) generate \(n\) synthetic rows using sample seed \(s\). Let \(\mathcal{M}\) be a preregistered set of auditor model families.

An artifact is transfer-sufficient for a declared workload when models trained or conditioned on generated rows recover the behavior of corresponding models trained or conditioned on the real fit rows when both are evaluated on untouched real test rows.

The term “behavior” must include:

1. predictive gain;
2. prediction agreement;
3. model-driver agreement;
4. uncertainty or calibration where supported.

This definition is operational. It does not claim that the artifact contains all information in the original table.

## 3.2 Public metric name

### **TransferBits**

Formal name:

> **Transfer-Sufficient Description Length**

Public sentence:

> **TransferBits is the minimum complete artifact size needed to preserve a declared amount of learning on unseen real data.**

Default operating point:

> **TB90/80:** minimum bits needed to preserve at least 90% of predictive gain and at least 80% driver fidelity across the declared auditor families.

The thresholds must be preregistered, but the complete frontier must always be published. The curve is primary; the thresholded scalar is a readable operating point.

## 3.3 Transfer retention

For dataset \(d\), auditor \(m\), artifact \(A\), and sample seed \(s\), define:

- \(L_{\mathrm{base}}\): loss of the strongest no-dataset-information baseline available to that auditor;
- \(L_{\mathrm{real}}\): loss when trained or conditioned on real fit rows;
- \(L_{\mathrm{syn}}\): loss when trained or conditioned on synthetic rows;
- all losses measured on the same untouched real test rows.

Then:

\[
T_{d,m}(A,s)
=
\frac{L_{\mathrm{base}} - L_{\mathrm{syn}}}
     {L_{\mathrm{base}} - L_{\mathrm{real}}}.
\]

Interpretation:

- \(T=1\): synthetic rows recover the full measured gain of real rows;
- \(T=0\): synthetic rows add no value over the auditor-specific baseline;
- \(T<0\): synthetic rows harm performance;
- \(T>1\): synthetic rows outperform the real-fit reference under that auditor.

Do not clamp the value before statistical analysis.

### Critical change for foundation models

A mean predictor is not a sufficient baseline for a powerful pretrained tabular foundation model. A foundation model may perform well from its prior even when the synthetic context contains little useful information.

For each foundation-model auditor, measure:

1. no-context or minimum-context prediction where supported;
2. label-shuffled context with identical features;
3. feature-only or target-masked context where supported;
4. conventional task null.

Use the strongest legitimate no-table-learning baseline as \(L_{\mathrm{base}}\). This prevents DOPE from receiving credit for knowledge already present in Kumo, TabFM, TabPFN, TabICL, LimiX, or Mitra.

## 3.4 Informative-cell rule

A cell is eligible only when the real table provides a measurable gain over the auditor-specific baseline.

Replace a fixed denominator rule alone with both:

1. a minimum practical effect:
   \[
   L_{\mathrm{base}}-L_{\mathrm{real}}
   \ge \epsilon_{\mathrm{effect}};
   \]
2. a lower confidence bound above zero for the real-data gain.

Publish the threshold sensitivity. Do not silently drop uninformative cells.

## 3.5 Auditor-family aggregation

Do not average a large number of correlated models. Otherwise, adding three nearly identical boosted-tree models changes the score.

Create model families:

1. **tabular foundation models**;
2. **boosted trees**;
3. **task-trained neural models**;
4. **linear/additive interpretable models**.

For each family:

\[
T_f = \operatorname{median}_{m \in f,\,s} T_{d,m}(A,s).
\]

Define robust transfer as:

\[
T_{\mathrm{robust}} = \min_f T_f.
\]

This means the artifact must transfer across different inductive biases rather than win by specializing to one model family.

A sensitivity report may also show:

- mean across families;
- 10th percentile across individual auditors;
- per-auditor values;
- win/tie/loss versus real-trained references.

The public headline should use the conservative family-robust value.

## 3.6 Uncertainty

For every dataset and artifact budget:

- five independent generator fit seeds;
- three independent sample seeds;
- multiple auditor seeds for stochastic task-trained models;
- fixed pretrained-auditor configurations;
- one sealed real test partition.

Use a hierarchical bootstrap:

1. source family;
2. dataset;
3. generator fit seed;
4. sample seed;
5. auditor seed where applicable.

For a single dataset, use seed-level confidence bounds or a valid finite-seed bound. For the population benchmark, use source-family-clustered intervals.

The TransferBits threshold uses the **lower confidence bound**, not the point estimate.

---

## 4. Driver and behavioral fidelity

Predictive accuracy alone is insufficient. Two synthetic datasets can produce equal loss while teaching models different relationships.

The new benchmark should measure model behavior on the same real holdout.

## 4.1 Prediction agreement

For regression:

- normalized RMSE between real-trained and synthetic-trained predictions;
- rank correlation;
- calibration or interval-coverage agreement for probabilistic auditors.

For classification:

- Jensen–Shannon divergence between predicted probabilities;
- agreement rate;
- calibration-curve distance;
- rank/AUROC agreement as a secondary metric.

Prediction agreement is diagnostic and may become a gate after the first benchmark establishes stable thresholds.

## 4.2 Model-conditional driver fidelity

Never compare Kumo importance directly with CatBoost importance as though one is ground truth.

For each auditor \(m\):

1. train or condition \(m\) on real rows;
2. train or condition the same frozen recipe on synthetic rows;
3. evaluate both on the identical real holdout;
4. compute model-agnostic feature effects for both;
5. compare synthetic-trained behavior to that auditor’s real-trained behavior.

This asks the right question:

> Did the synthetic table teach this model the same drivers that the real table taught it?

## 4.3 Primary attribution method

Use **grouped holdout permutation importance** as the primary method because it:

- works for trees, neural models, and in-context models;
- does not require gradients;
- can preserve original feature groups;
- evaluates actual predictive dependence on real test rows.

For categorical variables, permute the original feature as one group. For missingness, treat the value path and missingness indicator as a declared feature family.

Raw gradient attribution should not be a primary metric. It can be badly distorted by preprocessing, imputation, missingness routing, clipping, and heavy-tailed features.

## 4.4 Driver measurements

For each auditor, calculate:

1. **importance mass fidelity**
   \[
   D_{\mathrm{mass}}
   =
   1-\frac{1}{2}\|p_{\mathrm{real}}-p_{\mathrm{syn}}\|_1;
   \]

2. **rank fidelity**
   \[
   D_{\mathrm{rank}}=\mathrm{NDCG@k};
   \]

3. **top-driver overlap**
   \[
   D_{\mathrm{top}}=\mathrm{Jaccard}(\mathrm{TopK}_{\mathrm{real}},
                                    \mathrm{TopK}_{\mathrm{syn}});
   \]

4. **effect-shape fidelity**
   - compute ALE curves on the same real holdout;
   - compare signed and normalized curves for the real-trained and synthetic-trained models;
   - use robust correlation or normalized area distance;

5. **interaction fidelity**, initially secondary
   - compare top pairwise interaction rankings using permutation interaction or H-statistics.

Avoid another arbitrary weighted sum. Define the per-auditor driver gate as a conjunction:

- \(D_{\mathrm{mass}}\ge 0.80\);
- \(D_{\mathrm{rank}}\ge 0.80\);
- \(D_{\mathrm{top}}\ge 0.60\);
- median top-feature effect-shape agreement \(\ge 0.70\).

The thresholds are starting preregistration values and must be validated on real-versus-real replicates.

## 4.5 Real-versus-real calibration

Driver fidelity cannot be interpreted without a noise floor.

Fit the same auditor on two independent real subsamples and measure driver agreement. This establishes the maximum realistic agreement under finite data.

Report synthetic-to-real fidelity relative to that real-to-real stability.

## 4.6 Required schema change

The current observation shape stores one real-importance vector and one synthetic-importance vector while the contract names multiple auditors. Replace that structure with explicit per-auditor reports:

```json
{
  "auditor_reports": [
    {
      "auditor_id": "kumo_tabular_l",
      "family": "tabular_foundation_model",
      "recipe_sha256": "...",
      "base_loss": 0.0,
      "real_loss": 0.0,
      "synthetic_loss": 0.0,
      "transfer_retention": 0.0,
      "prediction_agreement": {},
      "driver": {
        "importance_real": [],
        "importance_synthetic": [],
        "ndcg_at_k": 0.0,
        "top_k_jaccard": 0.0,
        "mass_fidelity": 0.0,
        "effect_shape_fidelity": 0.0
      }
    }
  ]
}
```

No aggregate value may be emitted unless every required auditor report is present and qualified.

---

## 5. Compression and encoding accounting

## 5.1 Bits are canonical; encoded-value count is secondary

The number of encoded values is insightful but not sufficient. One arbitrary-precision number can contain an entire dataset.

The canonical measure must be actual bits in a deterministic artifact.

Report all of:

- artifact bytes;
- artifact bits;
- encoded scalar symbols;
- encoded categorical/dictionary symbols;
- numbers per original cell;
- bits per original cell;
- equivalent source rows;
- byte breakdown by artifact section.

## 5.2 Two byte tiers

### Tier A: dataset-specific description bytes

Count every dataset-specific byte required to sample:

- fitted transforms;
- schema required by the sampler;
- category dictionaries or pseudonymous mappings;
- marginal parameters;
- dependence parameters;
- target model;
- residual/noise model;
- quantization codebooks;
- dataset-specific seeds;
- data-specific weights;
- required sidecars.

This is the primary rate axis.

### Tier B: cold sampling closure bytes

Count:

- Tier A;
- sampler executable/runtime;
- required shared model checkpoints;
- dynamic libraries and data files opened by a cold sampler;
- configuration required for sampling.

This prevents a method from claiming a tiny artifact while hiding gigabytes in an external checkpoint.

The README should headline Tier A and display Tier B directly beside it.

## 5.3 Canonical byte measurement

Create `byte-accounting-v1` with these rules:

1. one canonical artifact archive;
2. deterministic ordering;
3. deterministic metadata encoding;
4. no timestamps;
5. no filesystem-dependent padding;
6. fixed compression implementation and level;
7. SHA-256 over exact bytes;
8. syscall or sandbox proof of every file opened during sampling;
9. dataset-specific and shared files classified explicitly;
10. sampler must fail when an undeclared file is opened.

## 5.4 Canonical artifact

Replace the public ambiguity among `.dpk`, `projection.json`, `.raw`, and sidecar combinations with one artifact:

`model.dope`

Suggested layout:

```text
DOPE magic/version
manifest
schema
transforms
marginals
dependence
target
noise
codebooks
checksums
```

The external certification receipt is not part of the sampling artifact unless sampling reads it.

Commands:

```bash
dope fit data.csv --target outcome --budget 10KiB -o model.dope
dope inspect model.dope
dope sample model.dope --rows 10000 -o synthetic.csv
dope verify model.dope --real-holdout test.csv
```

`dope inspect` must print:

```text
Artifact: 1,895 bytes
Encoded fitted scalars: 47
Bits per source cell: 0.031
Sections:
  schema/transforms       311 B
  marginals               504 B
  dependence              296 B
  target model            612 B
  noise                    92 B
Cold sampling closure: 68.7 MiB
```

## 5.5 TransferBits definition

For dataset \(d\), threshold \(\tau_T\), driver threshold vector \(\tau_D\), and confidence level \(1-\alpha\):

\[
\mathrm{TransferBits}_{d}(\tau_T,\tau_D,\alpha)
=
\min_A
\left\{
8C(A):
\operatorname{LCB}_{1-\alpha}(T_{\mathrm{robust}}(A))\ge\tau_T,
\;
D_{\mathrm{robust}}(A)\succeq\tau_D
\right\}.
\]

The default public operating point is:

\[
\mathrm{TB90/80}
=
\mathrm{TransferBits}(0.90,\tau_D,0.05).
\]

If no artifact reaches the threshold within the tested budget range, report:

`> B_max`

Never omit failed datasets from the denominator.

## 5.6 Population metrics

Publish:

1. **Coverage@10KiB**
   - fraction of eligible datasets for which TB90/80 is at most 10 KiB;

2. **TransferBits survival curve**
   - fraction of datasets reaching the threshold as the byte budget increases;

3. **median and quartiles of TB90/80**
   - failures right-censored at the largest tested budget and shown explicitly;

4. **bits per original cell**
   - secondary normalization for dataset scale;

5. **cold-closure bytes**
   - secondary operational cost.

Avoid a “utility per byte” headline ratio before a utility threshold is met. Tiny useless artifacts can otherwise look artificially efficient.

---

## 6. Scientific KPI versus release certification

Do not make one scalar perform two jobs.

## 6.1 Scientific result

Scientific TransferBits requires:

- artifact-only sampling;
- byte accounting;
- cross-model transfer;
- driver fidelity;
- frozen evaluation;
- uncertainty.

It does not automatically claim privacy.

## 6.2 Release result

Release TransferBits additionally requires:

- zero exact row copies;
- calibrated near-copy threshold;
- cleartext scan;
- membership-inference upper confidence bound;
- attribute-inference upper confidence bound;
- artifact attack, not only sample attack;
- declared categorical-label policy;
- formal DP parameters when formal DP is claimed.

Privacy failure should mark the artifact **not release-qualified**. It should not erase the scientific transfer measurement.

Output both:

```text
Scientific TB90/80: 15,160 bits
Release TB90/80: not established
```

## 6.3 Formal-DP lane

Methods with formal differential privacy should be benchmarked in a separate lane with fixed \((\epsilon,\delta)\) settings.

Required methods include, subject to reproducibility admission:

- AIM/MST-style marginal methods;
- Tab-PE / Private Evolution for tabular data;
- any admitted DP neural synthesizer.

Do not compare empirical attack resistance with formal DP as though they were the same guarantee.

---

## 7. Benchmark design

## 7.1 Three benchmark tracks

### Track A: author-default

Use the authors’ documented default or recommended configuration.

Purpose: practical out-of-box comparison.

### Track B: equal-selection budget

Give each method:

- the same maximum tuning trials;
- the same wall-clock envelope;
- the same data partitions;
- the same number of retained failures and timeouts;
- method-native selection objectives, frozen before evaluation.

Purpose: methodological fairness.

### Track C: equal-artifact budget

Evaluate each method at common dataset-specific byte ceilings:

- 256 B;
- 512 B;
- 1 KiB;
- 2 KiB;
- 4 KiB;
- 8 KiB;
- 10 KiB;
- 16 KiB;
- 32 KiB;
- 64 KiB.

Purpose: the core compression claim.

A method that cannot produce an artifact at a budget receives “not feasible,” not a silently compressed or omitted point.

## 7.2 Dataset suite

Keep the current PMLB regression panel as a legacy replication suite, not the sole headline benchmark.

Create a new versioned suite with four blocks:

### Block 1: real regression

Public, rights-cleared datasets with:

- continuous targets;
- numeric and mixed-type features;
- small, medium, and large row counts;
- low and high dimensionality;
- missingness;
- skew and heavy tails;
- group and temporal splits where meaningful.

### Block 2: real classification

Binary and multiclass datasets with:

- imbalance;
- mixed types;
- calibration-sensitive tasks;
- sufficient rows for independent test partitions.

### Block 3: novel/procedural challenge

Generate post-freeze challenge families with hidden parameters and held-out mechanisms. This reduces foundation-model benchmark contamination and provides known causal/driver structure.

Entire generator families must be held out together.

### Block 4: OOD transfer

Use temporal, geographic, institutional, or grouped test splits where available.

This tests the user’s actual claim: learning from synthetic data should transfer to genuinely unseen real conditions, not merely an IID row split.

## 7.3 Workload modes

### Mode 1: declared-target compression

The current DOPE mode. One target is declared before compilation.

### Mode 2: task-portfolio compression

One artifact must support multiple declared targets or losses.

This is a stronger and more interesting measure of reusable table knowledge.

### Mode 3: target-rotation stress test

Compile without selecting on the final target set, then evaluate multiple eligible columns as targets.

This directly tests whether DOPE preserves general table relationships rather than only a compact predictor for one target.

The first paper may headline Mode 1, but it must explicitly label the scope and include at least one Mode 2 or Mode 3 experiment.

## 7.4 Auditor panel

### Headline panel

After qualification:

1. NVIDIA Kumo Tabular, large or declared headline configuration;
2. TabICLv2;
3. LimiX-2 or LimiX-2M;
4. Mitra-v2;
5. CatBoost.

This gives multiple foundation-model designs plus the strongest conventional anchor already used in the project.

### Extended sensitivity panel

Subject to licenses and admission:

- TabFM;
- TabPFN-3/3.5;
- TabDPT;
- XGBoost;
- LightGBM;
- Explainable Boosting Machine or GAM;
- ridge/linear model;
- MLP;
- a modern task-trained neural tabular model.

### License lanes

Every auditor must be marked:

- permissive/open-commercial;
- open code with restricted weights;
- research-only;
- unavailable.

Do not quietly mix license classes.

## 7.5 Foundation-model adapter qualification

An auditor is not admitted until all gates pass:

1. exact source revision;
2. exact checkpoint revision;
3. local byte verification;
4. license text retained;
5. exact dependency closure;
6. offline load with network denied;
7. official minimal-example parity;
8. deterministic fixture parity;
9. declared context-row and feature limits;
10. actual GPU/CPU resource observations;
11. failure behavior tested;
12. no random-weight fallback;
13. no silent context truncation;
14. preprocessing receipt;
15. model version printed into every result.

The current locks should remain visible as “unqualified” until this process closes.

## 7.6 Generator baseline panel

### Required compact/statistical baselines

- independent marginals;
- Gaussian copula;
- Chow–Liu or Bayesian network;
- ARF;
- synthpop CART;
- quantized histogram sampler.

### Required neural/flow/diffusion baselines

Subject to admission:

- TabDDPM;
- TabSyn;
- TabDiff;
- CDTD;
- ForestFlow/ForestDiffusion;
- GATD or another verified 2026 geometry/flow method;
- TabCascade if its official implementation passes admission.

### Language-model baseline

Admit at least one:

- REaLTabFormer;
- GReaT;
- another reproducible table-language generator.

Keep it in a separate compute tier if necessary.

### Legacy anchors

- CTGAN;
- TVAE.

These should remain for historical continuity, not be presented as the strongest modern competition.

### Formal-DP methods

- AIM/MST;
- Tab-PE;
- other admitted formal-DP baselines.

## 7.7 Compression controls reviewers will demand

The project must compare against compact non-generator representations:

1. predictor-only artifact;
2. independent feature marginals plus predictor;
3. byte-matched quantized real-row sample;
4. k-medoids/prototype set;
5. coreset;
6. dataset-distillation baseline where feasible;
7. tiny decision tree/GAM/linear model;
8. raw table compressed with a fixed lossless codec;
9. label-shuffled artifact;
10. real-row bootstrap.

These controls answer the central skeptical question:

> Is DOPE discovering a generative representation, or is it merely hiding a compact predictor?

## 7.8 Sample sizes

Use:

- \(n\);
- \(4n\), primary for continuity;
- \(16n\), sensitivity;
- fixed-row caps for very large datasets.

Because generation can create arbitrarily many rows, publish the transfer-versus-generated-row curve. This separates artifact knowledge from Monte Carlo sample count.

## 7.9 Failure accounting

Every fit attempt must terminate in one disposition:

- success;
- timeout;
- OOM;
- numerical failure;
- unsupported schema;
- invalid output;
- admission failure.

Failures remain in the denominator. A missing baseline cell is not a DOPE win.

## 7.10 Statistical outputs

For every headline comparison:

- paired dataset differences;
- source-family-clustered 95% interval;
- win/tie/loss;
- multiplicity-adjusted test;
- practical equivalence interval;
- all-seeds sensitivity;
- real-only sensitivity;
- non-simulated sensitivity;
- failure-inclusive result.

Do not use unpaired medians to make superiority claims.

---

## 8. Compiler engineering

## 8.1 Make byte budget a first-class input

The compiler API should be:

```bash
dope fit data.csv \
  --target outcome \
  --budget 2048 \
  --objective transferbits-v1 \
  --out model-2k.dope
```

The compiler should be able to emit a frontier:

```bash
dope frontier data.csv \
  --target outcome \
  --budgets 256,512,1024,2048,4096,8192,10240,16384,32768 \
  --out frontier/
```

## 8.2 Replace gate-shortfall selection with constrained rate–distortion search

The formal optimization is:

\[
\min_A C(A)
\quad\text{subject to}\quad
\widehat{T}_{\mathrm{inner}}(A)\ge\tau_T
\quad\text{and declared hard constraints}.
\]

For frontier construction:

\[
\max_A \widehat{T}_{\mathrm{inner}}(A)
\quad\text{subject to}\quad C(A)\le B.
\]

All selection must use training-only inner folds. The benchmark validation and test rows must not choose the profile.

## 8.3 Candidate dimensions

Search over:

- number of retained features;
- transform family per feature;
- marginal bin count and precision;
- shared versus per-feature codebooks;
- dependence family;
- number of dependence edges/factors/rank;
- target basis width;
- target support size;
- quantization bit width;
- sparsity pattern;
- residual/noise family;
- heteroscedastic feature count;
- entropy-code choice.

## 8.4 Compression methods

Implement and ablate:

- quantization-aware training;
- mixed-precision parameter allocation;
- group-\(L_0\) or hard-concrete sparsification;
- low-rank factorization;
- shared codebooks;
- delta coding;
- rANS/arithmetic coding;
- integer fixed-point sampling;
- transform sharing;
- edge pruning in the dependence graph;
- section-level checksum deduplication where legitimate.

## 8.5 Bit-value analysis

A novel paper result should show the marginal value of bits.

For each artifact section:

1. remove or coarsen the section;
2. measure transfer loss;
3. compute \(\Delta T/\Delta\text{bit}\);
4. show the allocation of the final artifact.

This makes the project interpretable as a learned information budget:

> Which bits are carrying the predictive structure?

## 8.6 Freshness and support

A generator must demonstrate that it is not a row store:

- exact-copy rate;
- near-copy rate;
- support coverage;
- unique-row rate;
- conditional query fidelity;
- ability to draw more rows than the source;
- stable transfer as generated row count rises.

These are diagnostics and release gates, not substitutes for transfer.

---

## 9. Public product behavior

## 9.1 Accept ordinary tables

The public quickstart should not require users to manually:

- remove headers;
- normalize everything to \([0,1]\);
- place the target last;
- encode categoricals;
- understand HMAC bucket projection.

The CLI should accept a normal CSV or Parquet file with a target name.

Internal canonicalization can remain strict and receipt-backed.

## 9.2 Explicit categorical modes

Provide two modes:

### Usability mode

- preserve original category labels;
- count dictionaries and strings in artifact bytes;
- cleartext scan is expected to find declared schema tokens;
- no privacy claim.

### Pseudonymous release mode

- output stable pseudonymous category identifiers;
- no raw category strings in the public artifact;
- mapping held separately if the user needs reversal;
- mapping size and handling stated explicitly.

The current fixed-salt HMAC behavior should not be described as secrecy.

## 9.3 Add classification

The core research can remain regression-first, but a world-class open-source project needs a defined classification contract.

Required:

- binary classification;
- multiclass classification;
- class-probability calibration;
- categorical target support;
- task-specific null losses;
- classification foundation-model adapters.

## 9.4 Four-command public CLI

Public:

```text
dope fit
dope sample
dope inspect
dope verify
```

Advanced/research commands move under:

```text
dope research ...
dope dev ...
```

Do not place campaign assembly, deep-language calibration, shard operations, and internal promotion commands in the README quickstart.

---

## 10. README rewrite specification

## 10.1 New opening

Recommended copy:

```markdown
# DOPE

## Preserve the learning. Minimize the description.

DOPE compiles a table into a tiny generative program.

The test is simple: train powerful models on generated rows, score them on
untouched real rows, and measure how few bits are needed to recover the
learning and its drivers.

**North-star metric: TransferBits — how few bits preserve 90% of the learning?**
```

## 10.2 First-screen content

The first screen should contain only:

1. title and one-line thesis;
2. hero figure;
3. three-number scorecard;
4. one caveat sentence;
5. install/quickstart link.

Example scorecard structure:

```text
Legacy validation evidence
1.9 KiB          median dataset artifact
0.898            CatBoost retained gain at 4n
~22,800×         smaller than the measured TabSyn artifact

Cross-model TB90/80: pending qualified benchmark
Official sealed test: unopened
```

Do not imply finality.

## 10.3 Hero figure

Title:

> **How few bytes preserve the learning?**

Axes:

- x: complete dataset-specific artifact bytes, log scale;
- y: lower 95% bound of robust cross-model transfer.

Reference lines:

- vertical: 10 KiB;
- horizontal: 90% transfer.

Markers:

- filled: driver gate passed;
- hollow: driver not measured;
- crossed: release/privacy gate failed;
- direct labels, no dense legend.

Caption:

> Upper-left is better. Transfer is the fraction of a real-trained model’s gain recovered by the same model trained or conditioned on generated rows and evaluated on untouched real rows. Filled points also preserve the real-trained model’s drivers.

Only methods evaluated on the same dataset cohort, partitions, sizes, seeds, and auditor panel may appear together.

## 10.4 README order

1. Thesis.
2. Hero figure and current status.
3. What TransferBits means.
4. Sixty-second quickstart.
5. How DOPE works, one compact diagram.
6. Benchmark table.
7. Artifact and byte-accounting definition.
8. Reproducibility.
9. Scope and limitations.
10. Documentation links.
11. Citation and license.

## 10.5 Move out of the README

Move to dedicated documents:

- complete MFS-v4 equations;
- all release gates;
- categorical preprocessing internals;
- native runtime closure example;
- deep campaign workflow;
- long command reference;
- DPK v1/v2 compatibility;
- language calibration;
- cluster/shard operations;
- conformance-auditor internals.

## 10.6 Remove or demote

- Jankurai score badge from the top;
- toy artifact as the first numerical result;
- multiple incompatible artifact-size examples before the benchmark;
- “null score for every method” as a headline;
- unqualified model names presented as measured;
- internal campaign totals in the product story.

---

## 11. Paper rewrite specification

## 11.1 Recommended title

> **DOPE: How Few Bits Preserve the Learning in a Table?**

Subtitle or formal framing:

> **Task-Preserving Generative Compression for Tabular Data**

## 11.2 Abstract structure

Five sentences:

1. Synthetic tables are usually judged without asking how much representation they require.
2. Define TransferBits and the cross-model/driver-preserving protocol.
3. Describe DOPE as a byte-budgeted compiler.
4. State one complete sealed-test result.
5. State the open artifacts, scope, and limits.

Do not fill the abstract with method-by-method confidence intervals.

## 11.3 Page-one structure

1. Abstract.
2. One-line TransferBits equation.
3. Hero rate–distortion figure.
4. Three contributions.

Recommended contributions:

1. transfer-sufficient description length;
2. a byte-budgeted generative compiler;
3. a cross-model, driver-preserving benchmark;
4. an open, receipt-backed reproducibility system.

## 11.4 Main paper outline

1. Introduction.
2. Transfer-sufficient generative descriptions.
3. DOPE compiler.
4. Benchmark protocol.
5. Rate–distortion results.
6. Cross-model transfer.
7. Driver preservation.
8. Compression ablations and bit allocation.
9. Privacy/release diagnostics.
10. Limitations.
11. Conclusion.

## 11.5 Move to supplement

- complete gate definitions;
- every method lock;
- adapter qualification;
- all per-dataset tables;
- seed diagnostics;
- byte-conformance proof;
- CI/paper rebuild details;
- internal conformance auditor;
- extensive privacy attack definitions;
- legacy MFS compatibility.

## 11.6 Resolve the venue presentation

Do not ship a manuscript that uses a final-copy venue class while suppressing its venue header unless it is genuinely a submission build.

Maintain:

- neutral public preprint;
- anonymous submission build;
- venue-specific build generated from the same source.

## 11.7 Essential experiments

A serious paper should include:

1. full TransferBits frontier;
2. cross-model transfer matrix;
3. driver fidelity;
4. predictor-only and coreset controls;
5. byte-matched generator comparison;
6. target-portfolio or target-rotation test;
7. real-only and non-simulated sensitivity;
8. OOD split sensitivity;
9. artifact-section bit ablation;
10. fit/sample-seed variance;
11. sampling-row-count curve;
12. formal-DP lane where feasible.

---

## 12. Repository architecture

## 12.1 Separate product, benchmark, and paper

Target structure:

```text
crates/
  dope-core/
  dope-format/
  dope-cli/
python/
  dope_bench/
benchmarks/
  contracts/
  datasets/
  adapters/
  registries/
  results/
paper/
  main.tex
  supplement.tex
  figures/
docs/
  concepts/
  benchmark/
  format/
  privacy/
  development/
examples/
archive/
  mfs-v2/
  mfs-v3/
  mfs-v4/
```

A full move can be staged. The first objective is conceptual separation, not filesystem churn.

## 12.2 One source of truth

Create:

```text
contracts/transferbits-v1.json
contracts/byte-accounting-v1.json
contracts/auditor-panel-v1.json
contracts/benchmark-v1.json
schemas/transferbits-report-v1.schema.json
schemas/artifact-manifest-v1.schema.json
```

Rust and Python must read the same contract.

## 12.3 Retire MFS from the headline

Keep MFS-v2/v3/v4 for historical reproduction under `archive/` or compatibility modules.

Rename the current public role:

- MFS components become a `ReleaseReport`;
- TransferBits becomes the scientific KPI;
- privacy remains a gate report;
- representation closeness becomes an optional diagnostic.

## 12.4 File-level action plan

### `README.md`

Replace almost entirely with the structure in Section 10.

### `docs/whitepaper/dope-mfs.tex`

Rewrite around TransferBits. Remove MFS-v3 as the main scientific contribution.

### `production/kpi-contract-v4.json`

Freeze as legacy. Add `contracts/transferbits-v1.json`.

### `rust/fitness/v4.rs`

Keep compatibility tests. Do not extend it further. Add:

```text
rust/metrics/transfer_bits.rs
rust/metrics/driver_fidelity.rs
rust/metrics/byte_accounting.rs
```

### `research/transfer/objectives.py`

Move MFS-v4 functions to a legacy module. Add:

```text
research/transfer/transfer_bits.py
research/transfer/driver_fidelity.py
research/transfer/behavior_fidelity.py
```

### `research/benchmark/mfs_v3_assemble.py`

Split and rename. A v3-named file should not implement v4 lineage scoring.

Suggested:

```text
research/benchmark/assemble_transferbits.py
research/benchmark/aggregate_population.py
research/benchmark/render_scorecard.py
```

### `research/transfer/adapters/auditors/`

Add adapters and qualification for:

```text
limix.py
tabfm.py
tabpfn.py
tabdpt.py
```

Keep model-family metadata in a registry rather than hard-coded scorer constants.

### `research/transfer/locks/`

Convert “lock” into a state machine:

```text
specified
source_verified
weights_verified
environment_verified
parity_verified
resource_verified
qualified
```

### `docs/readme/`

Replace current images with:

```text
transferbits-hero.svg
transferbits-hero.png
architecture.svg
```

### Public result generation

README numbers and plots must be generated from one frozen result JSON. Hand-entered benchmark numbers are forbidden.

---

## 13. Result and receipt schema

Minimum report:

```json
{
  "format": "dope-transferbits-report",
  "version": 1,
  "dataset": {
    "id": "...",
    "schema_sha256": "...",
    "fit_sha256": "...",
    "select_sha256": "...",
    "test_sha256": "...",
    "rows": 0,
    "columns": 0,
    "task": "regression"
  },
  "artifact": {
    "sha256": "...",
    "format_version": 1,
    "dataset_specific_bytes": 0,
    "cold_closure_bytes": 0,
    "encoded_scalars": 0,
    "encoded_symbols": 0,
    "section_bytes": {}
  },
  "generation": {
    "fit_seed": 0,
    "sample_seeds": [],
    "sample_sizes": []
  },
  "auditor_reports": [],
  "aggregate": {
    "robust_transfer": 0.0,
    "robust_transfer_lcb": 0.0,
    "driver_gate": false,
    "tb90_80_pass": false
  },
  "release": {
    "exact_copy_pass": false,
    "near_copy_pass": false,
    "membership_pass": false,
    "attribute_pass": false,
    "cleartext_pass": false,
    "formal_dp": null
  },
  "provenance": {
    "contract_sha256": "...",
    "code_revision": "...",
    "environment_sha256": "...",
    "created_utc": "..."
  }
}
```

All aggregate values must be reproducible from raw auditor reports.

---

## 14. CI, reproducibility, and claims firewall

## 14.1 CI tiers

### Per pull request

- Rust/Python unit tests;
- schema validation;
- golden metric parity;
- deterministic artifact bytes;
- tiny benchmark fixtures;
- README and paper number regeneration;
- no stale generated files;
- no unqualified method in headline outputs.

### Nightly

- CPU benchmark subset;
- artifact-only syscall proof;
- adversarial format tests;
- privacy smoke tests.

### Scheduled GPU

- qualified foundation-model fixture matrix;
- checkpoint/environment verification;
- numerical parity;
- resource regression.

## 14.2 Golden parity

For every metric:

- canonical JSON fixtures;
- expected Rust output;
- expected Python output;
- exact or tightly bounded parity;
- malformed-input failures;
- NaN/overflow failures;
- missing-cell failures.

## 14.3 Claims status

Every result row must have one status:

- `sealed_test`;
- `validation`;
- `selection_exposed_validation`;
- `historical`;
- `specified_unrun`;
- `unqualified`;
- `failed`.

Renderers must refuse to present a non-sealed result as a final headline.

## 14.4 One-command reproduction

Target:

```bash
just paper
just benchmark-smoke
just verify-release
```

The paper and README should rebuild from committed scalar ledgers without access to restricted source rows or model weights.

---

## 15. Phased implementation plan

## Phase 0: stop the narrative damage

### Actions

- freeze new MFS-v4 feature additions;
- declare MFS-v4 legacy/secondary;
- rewrite the README top third;
- label every current result by evidence status;
- remove unqualified auditors from measured claims;
- state one canonical artifact definition;
- move campaign/operator material out of README;
- add this roadmap.

### Exit criteria

A first-time reader can answer in 30 seconds:

1. What is DOPE?
2. What is measured?
3. How large is the artifact?
4. What is not yet proven?
5. What command should I run?

## Phase 1: TransferBits core

### Actions

- freeze `transferbits-v1`;
- implement auditor-specific baselines;
- implement robust family aggregation;
- implement threshold and frontier logic;
- implement byte-accounting-v1;
- implement per-auditor driver schema;
- add Rust/Python golden parity;
- generate a small fixture plot.

### Exit criteria

A complete report can answer:

> What is the smallest artifact in this budget sweep whose lower-confidence transfer and driver fidelity pass the declared thresholds?

## Phase 2: auditor qualification

### Actions

Qualify:

1. Kumo Tabular;
2. TabICLv2;
3. LimiX;
4. Mitra-v2;
5. CatBoost reference.

Then admit research-only sensitivities:

- TabFM;
- TabPFN;
- TabDPT.

### Exit criteria

Every headline auditor runs offline from pinned bytes, passes a published fixture, reports resource use, and produces per-auditor transfer and driver evidence.

## Phase 3: benchmark completion

### Actions

- freeze dataset suite;
- freeze split hashes;
- freeze generator registry;
- run default, equal-selection, and equal-byte tracks;
- retain all failures;
- complete five fit seeds and three sample seeds;
- freeze final configuration before opening the official test.

### Exit criteria

Every headline point uses the same cohort, splits, sample sizes, seeds, and auditor families.

## Phase 4: compiler frontier

### Actions

- add budget-aware search;
- add quantization/sparsity/codebook sweeps;
- emit frontier artifacts;
- implement bit-allocation report;
- add target-portfolio mode.

### Exit criteria

DOPE can produce a monotone, validated byte–transfer frontier and select the smallest passing artifact without reading benchmark validation/test rows.

## Phase 5: paper and release

### Actions

- run sealed test once;
- regenerate hero figure and tables;
- write neutral preprint;
- publish artifact registry and reproducibility bundle;
- cut a clean release.

### Exit criteria

The first page supports one complete claim with no “pending,” “unqualified,” or cross-cohort comparison in the headline.

---

## 16. Priority backlog

### P0 — required before any public push

1. README rewrite.
2. One artifact definition.
3. Evidence-status labels.
4. MFS-v3/v4 paper mismatch resolved.
5. Kumo/Mitra/TabICL marked unqualified until admission.
6. Current CatBoost result labeled selection-exposed validation.
7. Public CLI reduced to four commands.

### P1 — scientific core

8. TransferBits contract.
9. Auditor-specific prior baseline.
10. Family-robust transfer.
11. Per-auditor driver evidence.
12. Exact byte-accounting contract.
13. Real-versus-real driver calibration.
14. Golden Rust/Python parity.

### P2 — benchmark credibility

15. Kumo qualification.
16. TabICLv2 qualification.
17. LimiX qualification.
18. Complete modern generator panel.
19. Compact predictor/coreset controls.
20. Full same-cohort matrix.
21. Target-portfolio experiment.
22. OOD split experiment.

### P3 — profound innovation

23. Learned bit allocation.
24. Bit-value ablation.
25. TransferBits coverage curves.
26. multi-target/workload-conditioned artifacts;
27. release TransferBits;
28. formal-DP frontier;
29. public artifact explorer;
30. interactive per-dataset rate–distortion dashboard.

---

## 17. Acceptance criteria

The reset is complete only when all are true:

### Scientific clarity

- one primary metric;
- one formal unit: bits;
- one default operating point;
- full frontier published;
- no hidden weighted master score in the headline.

### Model breadth

- at least three qualified foundation-model families;
- at least one boosted-tree family;
- per-family robust aggregation;
- no CatBoost-only central claim.

### Driver preservation

- per-auditor permutation importance;
- effect-shape comparison;
- real-versus-real calibration;
- driver thresholds frozen before final test.

### Compression integrity

- deterministic canonical artifact;
- complete dataset-specific byte count;
- cold closure reported;
- encoded-symbol count reported;
- sampling file accesses proven.

### Benchmark integrity

- same cohort for headline methods;
- all failures retained;
- paired statistics;
- complete seeds;
- official sealed test;
- no validation-selected profile scored as unexposed.

### Product quality

- ordinary CSV quickstart;
- coherent artifact output;
- four-command CLI;
- generated README numbers;
- clean documentation hierarchy.

### Paper quality

- page-one question understandable without project history;
- one hero plot;
- no venue-class ambiguity;
- main text contains completed evidence;
- governance machinery moved to supplement.

---

## 18. Claims that must not be made yet

Do not currently claim:

- that DOPE beats TabSyn overall;
- that DOPE is validated across Kumo, Mitra, and TabICLv2;
- that MFS-v4 is an empirically established master score;
- that the artifact is private, DP, HIPAA-safe, or de-identified;
- that 1.9 KiB represents a complete cold runtime closure;
- that the system preserves an arbitrary table rather than a declared target workload;
- that a validation-exposed result is an official-test result;
- that a missing baseline cell is a loss;
- that categorical hashing is secrecy;
- that representation closeness proves semantic equivalence;
- that encoded-number count alone is description length.

The correct current claim is narrower and still impressive:

> DOPE has validation evidence that a tiny target-conditioned generative program can preserve much of the predictive gain seen by CatBoost, at dramatically lower dataset-specific artifact size than measured neural generators. Cross-model, driver-preserving, release-qualified TransferBits remains to be established.

---

## 19. The desired end state

A visitor opens the repository and sees:

> **DOPE: How Few Bits Preserve the Learning in a Table?**

Then one plot shows DOPE in the upper-left corner.

Then three lines explain:

- models trained on generated rows are tested on untouched real rows;
- the same drivers must be recovered;
- every sampling-required byte is counted.

Then one command produces a tiny artifact and a receipt.

The implementation underneath may remain sophisticated, fail-closed, receipt-backed, multi-seed, privacy-aware, and rigorously versioned. That complexity is valuable. It simply should not be the idea the reader has to decode.

The idea is:

> **Preserve the learning. Minimize the description. Prove every bit.**
