# DOPE Research and Product Reset

## Engineering specification for a world-class synthetic-data metric, benchmark, paper, README, and open-source release

**Repository:** `neverhuman/dope`  
**Evidence reviewed:** root README, Rust metric and artifact code, benchmark registries and predeclarations, generated benchmark evidence, paper source and paper audits, CI and release structure  
**Specification date:** 2026-10-15  
**Status:** Proposed architecture and execution plan  

---

# 0. Executive decision

DOPE is not scientifically empty. Its strongest components are unusually serious:

- an executable, artifact-only generator rather than a synthetic CSV dump;
- exact byte charging of the learned artifact;
- typed Rust encoding and sampling code;
- receipt-backed benchmark reductions;
- paired comparisons, multiple fit and sample seeds, nested uncertainty, and missing-cell refusal;
- an explicit acknowledgment that the official test remains sealed and current results are selection-exposed;
- a reproducible paper build that derives numbers and figures from committed evidence.

The failure is **presentation architecture**. A research lab notebook, product contract, certification system, benchmark ledger, historical audit trail, and public explanation are all exposed at the same level. The profound idea is present, but it is surrounded by several generations of machinery: retention, MFS-v2, MFS-v3, MFS-v4, release tiers, privacy gates, representation distances, campaign status, compatibility layers, and multiple partially overlapping benchmark populations.

The reset should make one scientific question dominant:

> **How few learned bytes are required before synthetic rows teach independent modern models the same useful relationships as the real data?**

Everything else should support that question.

The project should adopt the following hierarchy:

1. **Scientific object:** the **qualified rate-transfer frontier**.
2. **Headline metric:** **Qualified Transfer Description Length (QTDL)**.
3. **Public display:** **`KiB@T90/D80`**.
4. **Evidence card:** bytes, cross-model transfer, driver fidelity, privacy status, benchmark coverage.
5. **Certification:** a separate fail-closed report, never averaged into the public scientific metric.
6. **MFS-v4:** retained as an internal diagnostic/selection vector for compatibility, but removed as the public North Star.

The one-sentence explanation should be:

> **DOPE compiles a table into a tiny generator. We measure the fewest bytes needed for models trained on its samples to generalize to untouched real data and recover the same predictive drivers.**

The current evidence should be described honestly:

> **Development evidence:** DOPE stores a median learned payload of roughly 1.9 KiB and recovers roughly 89.8% CatBoost validation utility on 99 PMLB regression lineages. This is not yet a QTDL result because the modern cross-model suite, driver-fidelity gate, complete privacy panel, and untouched final test are not all complete.

That distinction is essential. The existing evidence is highly promising; presenting it as a completed universal score would weaken it.

---

# 1. What must change

## 1.1 The public metric is currently too composite

MFS-v4 combines seven weighted components, prerequisites, hard gates, multiple auditors, attribution agreement, encoder distance, distribution fidelity, structure fidelity, realism, and a knowledge-compression score. That is useful as an engineering dashboard. It is not a clean scientific claim.

A weighted geometric mean creates six problems:

1. **Weights become the argument.** Reviewers can reasonably ask why utility is 0.30, driver fidelity is 0.20, representation distance is 0.15, and so on.
2. **Averages conceal failure.** A poor driver score can be compensated by distribution fidelity unless it is separately gated.
3. **The score is difficult to explain.** “A gated geometric mean of seven normalized submetrics” is not a first-page idea.
4. **Model-specific choices enter the definition.** Kumo currently appears as a headline predictive auditor and a representation oracle. That makes the score dependent on one rapidly evolving model family.
5. **Version churn becomes unavoidable.** Every new model, threshold, or attack creates pressure for MFS-v5, MFS-v6, and so on.
6. **The score is currently null for the methods that matter.** A public metric that cannot yet be reported should not dominate the README.

The project needs a constrained optimization metric rather than a weighted mixture:

> Minimize learned description length **subject to** transfer, driver, privacy, and reproducibility requirements.

This is simpler and scientifically stronger.

## 1.2 The benchmark currently proves a development result, not the final broad claim

The current headline population is a 100-lineage PMLB regression panel. It is useful for controlled development, but several limitations must be moved to the front of the scientific design:

- the displayed DOPE profile was compared with three alternatives on validation rows;
- the current paper explicitly calls every lineage and auditor selection-exposed;
- many lineages come from repeated simulated source families;
- the official test remains reserved;
- the paper auditor registry is still centered on CatBoost, linear regression, and an MLP;
- several strong generator baselines are incomplete or measured on smaller cohorts;
- classification, mixed-type data, augmentation, distribution shift, and real operational tasks are not yet the headline evidence.

PMLB should remain, but as a **mechanism and stress suite**, not the principal claim of real-world synthetic-data leadership.

## 1.3 “Bytes” needs a rigorous, cross-method definition

DOPE already has exact artifact bytes and an encoded-number count. That is a major advantage. It needs to become a benchmark-wide contract.

A method can otherwise appear artificially small by hiding information in:

- preprocessing files;
- category dictionaries;
- schema metadata;
- a generated Python module;
- a fixed but enormous pretrained base model;
- a nonstandard runtime or decoder;
- external files fetched at sample time;
- hard-coded dataset-specific constants outside the artifact;
- a compressed copy of real rows.

The benchmark must report three distinct byte scopes and prohibit hiding dataset-derived state.

## 1.4 The README is a technical annex masquerading as a landing page

The current root README contains the thesis, a large benchmark table, two plots, artifact anatomy, build instructions, multiple demos, a preprocessing contract, the full MFS-v4 algebra, privacy and certification details, benchmark tooling, and compatibility notes. The result is rigorous but cognitively expensive.

A world-class landing page should let a serious reader answer five questions in 60 seconds:

1. What does DOPE do?
2. Why is it new?
3. What is the one number that matters?
4. What evidence exists today?
5. How do I run it?

Everything else belongs in structured documentation.

## 1.5 The paper directory is an archive, not a canonical manuscript package

`docs/whitepaper/` currently contains the active paper beside baseline gap ledgers, readiness reports, self-reviews, retained-evidence reviews, historical decisions, comprehension notes, DOI checks, style notes, generated evidence, and multiple PDFs. Those files are useful internally, but the directory communicates uncertainty and historical accumulation instead of a clean submission artifact.

The canonical paper package should contain only:

- manuscript source;
- bibliography;
- generated tables/figures;
- paper-specific build scripts;
- final PDF and supplement;
- a short reproduction README.

Historical audits should be retained, but moved to an explicit archive.

---

# 2. North-star scientific definition

## 2.1 Scientific object: the qualified rate-transfer frontier

For a dataset `d`, generator method `g`, and learned payload budget `b`, let the generator produce an artifact `A(g,d,b)` and synthetic rows sampled only from that artifact.

The benchmark measures:

- **rate:** exact learned payload bits or bytes;
- **transfer:** how much real-data learning is recovered when an auditor learns from synthetic rows and is evaluated on untouched real rows;
- **driver fidelity:** whether the synthetic-trained auditor relies on the same predictive variables and directions as its real-trained counterpart;
- **qualification:** whether the artifact passes privacy, novelty, replay, provenance, and coverage gates.

For every method, plot the best qualified transfer achieved at each byte budget. The non-dominated envelope is the **qualified rate-transfer frontier**.

The frontier, not a single weighted score, is the primary scientific result.

## 2.2 Headline metric: Qualified Transfer Description Length

Define **Qualified Transfer Description Length**, abbreviated **QTDL**:

\[
\operatorname{QTDL}_{\tau,\delta}(d,g)
=
\min_A L_{\mathrm{learned}}(A)
\]

subject to:

\[
\operatorname{LCB}_{1-\alpha}\left(T_{\mathrm{robust}}(A,d)\right) \ge \tau,
\]

\[
\operatorname{LCB}_{1-\alpha}\left(D_{\mathrm{robust}}(A,d)\right) \ge \delta,
\]

and:

\[
Q(A,d)=1,
\]

where:

- `L_learned(A)` is the exact dataset-specific learned payload length;
- `T_robust` is robust multi-family synthetic-to-real transfer;
- `D_robust` is robust predictive-driver fidelity;
- `Q` is the conjunction of hard qualification gates;
- `LCB` is a predeclared lower confidence bound;
- lower QTDL is better;
- if no artifact qualifies, QTDL is **infinite/unqualified**, never zero.

### Public shorthand

Display the principal operating point as:

> **`KiB@T90/D80`**

Meaning:

> Minimum learned payload in KiB that achieves a lower-bound transfer of at least 0.90 and driver fidelity of at least 0.80 while passing all qualification gates.

The initial recommended thresholds are:

- `T90`: robust transfer lower bound ≥ 0.90;
- `D80`: robust driver-fidelity lower bound ≥ 0.80;
- confidence level: 95%;
- required model-family coverage: all mandatory families for an eligible task;
- privacy/replay/provenance: all pass.

Thresholds must be frozen before final benchmark results are opened. Sensitivity should show `T80`, `T90`, `T95` and `D70`, `D80`, `D90`, but only one pair is the headline.

## 2.3 Why this is more profound than “accuracy per byte”

A simple utility/byte ratio is easy to game. A nearly empty artifact with weak utility can produce a large ratio, and ratios obscure whether a method crossed a useful quality threshold.

QTDL asks a sharper question:

> **What is the shortest executable description that still teaches independent learners enough of the real task?**

The threshold formulation gives bytes a causal role in the objective while preventing a tiny useless artifact from scoring well.

Compression alone is not intelligence. **Compression under held-out transfer and driver-preservation constraints is evidence that the artifact captured reusable predictive structure rather than merely copied records or optimized one learner.**

## 2.4 Secondary whole-frontier statistic

For statistical comparisons across the full byte range, compute a secondary statistic, **Qualified Rate-Transfer Area (QRTA)**:

\[
\operatorname{QRTA}(g)
=
\frac{1}{\log B_{\max}-\log B_{\min}}
\int_{\log B_{\min}}^{\log B_{\max}}
\widetilde{T}_g(B)\,d\log B,
\]

where `T_tilde(B)` is zero when driver/privacy/replay requirements fail and otherwise is robust transfer clipped only for the area calculation to `[0,1]`.

QRTA is a secondary ranking/tie-breaker. It must not replace the frontier or QTDL in the first sentence.

## 2.5 The public evidence card

Every published method should receive a compact card:

```text
DOPE — benchmark release core-real-reg-v1
Learned payload:          1.9 KiB median [IQR]
Portable closure:         ... MiB
Encoded learned atoms:    ...
Robust transfer:          ... [95% CI]
Driver fidelity:          ... [95% CI]
QTDL T90/D80:             ... KiB or UNQUALIFIED
Privacy:                  PASS / FAIL / UNMEASURED
Replay:                   PASS
Eligible datasets:        x / N
Fit seeds:                5
Auditor suite:             open-2026.10
Result status:             development / sealed-test / release
```

No public table should collapse these into a mysterious 0–100 scalar.

---

# 3. Exact transfer protocol

## 3.1 Data partitions

Each dataset must have immutable, content-hashed partitions:

1. `fit`: visible to generator fitting and auditor real-training controls;
2. `selection`: visible only to method-specific tuning and budget selection;
3. `test`: never visible to generator fitting, model selection, hyperparameter selection, early stopping, threshold selection, or figure design;
4. optional `shift_test`: future time, geography, site, population, or group.

Where an original dataset supplies an official train/test split, preserve it. Otherwise freeze split generation before any benchmark runs.

The runner must enforce filesystem and process isolation so jobs cannot open test paths. A Boolean field stating “test not opened” is not enough; the test mount should be absent from fit containers.

## 3.2 TSTR/TRTR transfer

For auditor `m` on dataset `d`:

- **TRTR:** learn/context on real fit rows, evaluate on real test rows;
- **TSTR:** learn/context on synthetic rows sampled from the artifact, evaluate on the identical real test rows;
- **NULL:** a task-appropriate null trained without predictive feature-target relationships.

For a higher-is-better score `S`:

\[
T_{m,d}
=
\frac{S_{\mathrm{TSTR}}-S_{\mathrm{NULL}}}
{S_{\mathrm{TRTR}}-S_{\mathrm{NULL}}}.
\]

For a lower-is-better loss `L`:

\[
T_{m,d}
=
\frac{L_{\mathrm{NULL}}-L_{\mathrm{TSTR}}}
{L_{\mathrm{NULL}}-L_{\mathrm{TRTR}}}.
\]

Properties:

- `1.0` means synthetic learning recovers the real-trained gain;
- `0.0` means no gain over the null;
- values above `1.0` are retained and indicate synthetic augmentation/regularization can outperform the real-trained control;
- negative values are retained and indicate harm;
- values are never clipped in stored evidence or inferential statistics.

## 3.3 Low-signal tasks

A normalized ratio is unstable when `TRTR - NULL` is near zero. Predeclare an informativeness rule using both:

- an absolute minimum improvement over the null;
- a relative minimum compared with observed seed variation.

For uninformative auditor-dataset cells:

- do not assign transfer zero;
- mark the cell `uninformative`;
- report raw TSTR and TRTR scores/losses;
- exclude it from normalized transfer only according to the frozen rule;
- report the resulting denominator and coverage.

A method may qualify on a dataset only if the required number of model families have informative real controls.

## 3.4 Synthetic sample sizes

Report a learning curve rather than only `4n`:

- `0.25n`;
- `1n`;
- `4n`;
- `16n` where computationally feasible;
- an unlimited streaming condition for executable generators.

The QTDL headline should use `4n` initially for continuity with the current paper, but `1n` and the learning-curve area must remain visible. A generator that needs infinite rows to recover its task signal is different from one that teaches efficiently.

## 3.5 Model-family aggregation

Do not let a leaderboard with many similar tree models dominate the metric. Group auditors into independent families:

1. tabular foundation models;
2. gradient-boosted trees;
3. neural tabular models;
4. transparent/linear models;
5. fixed-budget AutoML or heterogeneous ensemble.

Within each family, aggregate the median transfer across declared models. Across required families, define:

- `T_median`: median family transfer;
- `T_floor`: minimum family transfer;
- `T_robust`: predeclared lower quantile or floor.

Recommended initial qualification:

```text
LCB(T_median) >= 0.90
LCB(T_floor)  >= 0.80
no required family has negative median transfer
```

For the public shorthand, `T90` refers to the median requirement; the result card must also show the floor. If an even simpler single constraint is preferred, use `LCB(T_floor) >= 0.90`; that is maximally strict but may be statistically expensive.

## 3.6 In-context foundation models

For Kumo, Mitra-v2, TabICLv2, TabPFN, TabFM, and related in-context learners:

- “training” means providing labeled context rows;
- TSTR context consists only of synthetic rows;
- TRTR context consists only of real fit rows;
- real test rows are queries and never context;
- context size, row order, ensemble count, preprocessing, and inference seeds are pinned;
- no synthetic and real context mixing occurs in the replacement track;
- a separate augmentation track may mix them under a frozen ratio.

The benchmark must not silently compare a tuned gradient-boosted learner with a differently budgeted foundation-model ensemble. Each auditor gets a frozen, documented inference budget.

---

# 4. Predictive-driver fidelity

## 4.1 Why driver fidelity is a gate

A synthetic dataset can preserve aggregate accuracy while teaching the wrong causal or correlational shortcut. For scientific use, policy, medicine, defense, finance, and debugging, the identity and direction of predictive drivers matter.

Driver agreement must therefore be a qualification requirement rather than a small term that can be averaged away.

## 4.2 Common explanation protocol

For each auditor, fit/context two versions:

- real-trained auditor `m_R`;
- synthetic-trained auditor `m_S`.

Explain both on **the identical untouched real test rows**. This prevents explanation differences from being confounded by different evaluation populations.

Use a model-agnostic primary protocol:

1. group all derived columns back to source features;
2. compute repeated grouped permutation importance on real test rows;
3. normalize positive loss increases into an importance distribution;
4. compute effect direction with accumulated local effects for continuous features and frozen category contrasts for categorical features;
5. use the same metric, rows, permutations, and random seeds for real- and synthetic-trained models.

Conditional permutation or SAGE can be a sensitivity analysis, but should not be the sole primary measure because of cost and conditional-model dependence.

## 4.3 Driver metrics

For each auditor-dataset pair, compute:

- `NDCG@k` for ranked driver recovery;
- `top-k weighted overlap` or Jaccard;
- `importance_mass_similarity = 1 - 0.5 * ||p_R - p_S||_1`;
- `direction_agreement`, weighted by real-model importance;
- optional interaction-driver overlap for the most important pairwise interactions.

Set:

\[
D_{m,d}
=
\min(
\operatorname{NDCG@k},
\operatorname{mass\_similarity},
\operatorname{direction\_agreement}
).
\]

Using the minimum avoids arbitrary weights and ensures the score identifies the weakest driver-preservation property.

The default `k` should be frozen as:

\[
k = \min(d_{\mathrm{active}}, \max(5, \lceil 0.10d \rceil), 20).
\]

If fewer than five real-active features exist, use all active features.

## 4.4 Driver qualification

Recommended initial requirement:

```text
LCB(median family D) >= 0.80
LCB(minimum required-family D) >= 0.70
```

Also publish each component separately. A single `D=0.81` should never conceal that direction agreement is 0.55 while ranking agreement is high.

## 4.5 Interpretation limits

Feature importance is not causality. The paper and README must use “predictive drivers,” “model reliance,” or “attribution agreement,” not “causal mechanisms,” unless a separate interventional benchmark is run.

---

# 5. Byte and encoding accounting

## 5.1 Three mandatory byte scopes

### A. Learned payload bytes — primary rate axis

`learned_payload_bytes` is the exact length of the canonical dataset-specific artifact needed to sample after fitting.

It includes every train-derived or dataset-specific item:

- learned parameters and quantization scales;
- marginals, category maps, bins, knots, codebooks, and feature selectors;
- schema and column-type information required for sampling;
- fitted preprocessing and inverse transforms;
- generated source code or bytecode;
- dataset-specific random tables, dictionaries, or seeds;
- target metadata;
- any sidecar read at sample time.

It excludes only a decoder/runtime that is identical across every dataset and frozen before benchmark access.

### B. Portable sampling closure bytes

`portable_closure_bytes` includes:

- learned payload;
- sampler executable or package;
- fixed decoder assets;
- required model code;
- nonstandard runtime files;
- pinned shared weights required for sampling.

This is the size needed to move the generator to a clean compatible host and sample offline.

### C. Cold execution closure bytes

`cold_closure_bytes` includes the full container or virtual environment, system libraries, interpreter, CUDA/runtime components where required, and all transitive files touched during a clean sample run.

This prevents a 2 KiB learned payload from being misrepresented as a 2 KiB deployable system.

## 5.2 Canonical learned description

The primary rate is:

\[
L_{\mathrm{learned}} = 8 \times \operatorname{len}(A_{\mathrm{canonical}})
\]

in bits, with KiB display using `1024` bytes.

Canonicalization requirements:

- deterministic field order;
- deterministic float/quantized representation;
- no timestamps, paths, UUIDs, or host-specific bytes;
- fixed compression codec and level;
- fixed archive metadata;
- byte-for-byte identical output for identical evidence and seed;
- declared magic/version header;
- full SHA-256 and BLAKE3 digests.

## 5.3 Learned atoms and effective bits

Exact bytes are authoritative. Two diagnostics expose the semantic compactness:

- `learned_atom_count`: number of dataset-derived scalar values or structural codewords;
- `effective_learned_bits`: exact bit width of those atoms after quantization/entropy coding, including structural indices.

The current DOPE encoded-number implementation is a strong starting point, but the benchmark-wide contract must distinguish:

- one FP32 value from one INT8 value;
- a copied source value from a fitted parameter;
- a category code from its probability;
- a structural index from a numeric coefficient;
- a sparse zero not stored from a stored zero.

Also report:

\[
\operatorname{bits\_per\_original\_cell}
=
\frac{L_{\mathrm{learned}}}{n d},
\]

and:

\[
\operatorname{atoms\_per\_original\_cell}
=
\frac{E}{nd}.
\]

These are diagnostics, not the headline score.

## 5.4 Shared priors and base models

A method may use a fixed pretrained base model. It must disclose:

- base model name and exact revision;
- base-weight bytes;
- license;
- whether the base was trained on benchmark datasets or related data;
- dataset-specific adapter/prompt/payload bytes;
- full portable closure.

Do not amortize base weights into an arbitrary per-dataset number in the headline. Report the two parts separately:

```text
learned payload: 8 KiB
shared frozen base: 1.6 GiB
```

Tabular foundation models used only as auditors are evaluation infrastructure and are not charged to generator bytes.

## 5.5 Copying and coreset controls

Every benchmark release must include:

- canonical compressed original table;
- copied-row artifacts at `k` rows;
- random and learned coresets/distilled examples where available;
- row bootstrap control;
- predictor-only control.

A copied-row artifact pays for every stored value, schema field, and index. This makes “copy the table” an explicit point on the frontier rather than an implicit straw man.

## 5.6 Repair current knowledge-compression implementation

The present MFS-v4 implementation computes `numbers_per_original_cell` but does not use it in the parameter-sparsity term; that term divides encoded numbers by `real_columns * 8`, not by original cells. The log-byte score also reaches zero exactly at the allowed 10,240-byte cap. Those choices may be intentional, but they are difficult to defend as a general description-length measure.

Required action:

- freeze MFS-v4 for historical compatibility;
- do not mutate its meaning silently;
- implement QTDL in a new contract;
- publish a migration note explaining that exact learned bytes become the primary rate and encoded atoms become a diagnostic;
- add cross-language golden tests for both old and new metrics.

---

# 6. Auditor suite: modern models without metric brittleness

## 6.1 Registry, not hard-coded formula

The metric contract should require model families and capability properties, not permanent model names. The concrete models live in a versioned registry:

```text
auditor-suite-open-2026.10
```

Each registry entry pins:

- source repository and commit;
- package and model revision;
- model/weight license;
- task support;
- row, feature, class, and context limits;
- preprocessing;
- inference ensemble and seed budget;
- device requirements;
- expected output metric;
- explanation adapter;
- determinism status.

A new model suite creates a new benchmark release, not a silent metric redefinition.

## 6.2 Mandatory open suite

The first modern suite should include at least:

### Tabular foundation family

- **NVIDIA Kumo Tabular Large**, with exact open model revision pinned;
- **Mitra-v2**, open weights and inference path;
- **TabICLv2**, separate classification/regression checkpoints as applicable.

Use a family median so three foundation models count as one scientific family rather than three votes.

### Gradient-boosted tree family

- CatBoost;
- LightGBM or XGBoost;
- one fixed, modest tuning budget or frozen strong defaults.

### Neural tabular family

- one strong modern MLP/tabular neural model with a maintained open implementation;
- optionally a second architecture, aggregated within family.

### Transparent family

- regularized linear/logistic model;
- generalized additive model where practical.

### Heterogeneous performance family

- fixed-budget AutoGluon or an equivalent reproducible ensemble, used for performance transfer rather than feature-direction gating if explanations are unstable.

## 6.3 Extended research suite

Run separately when licensing and compute permit:

- TabPFN current research weights;
- TabFM;
- LimiX, EXAONE Tabular, Xiaomi, or other frontier TFMs with explicit license and reproducible inference;
- new model revisions after suite freeze.

The extended suite is a sensitivity panel, not a requirement for the open benchmark if weights are noncommercial or access-restricted.

## 6.4 Avoid representation-oracle dependence

Do not make Kumo embedding distance a prerequisite of the headline score. A representation distance can be reported as a diagnostic, but it should not define quality when Kumo is also a primary transfer auditor.

If representation analysis remains:

- use at least two independent encoders;
- freeze exact layer, pooling, normalization, and distance;
- validate that representation distance predicts held-out transfer beyond the transfer metrics themselves;
- treat it as explanatory evidence, not a compensating score component.

---

# 7. Generator benchmark

## 7.1 Benchmark tracks

Every method is run in three clearly separated tracks.

### Track A — author/default

- author-recommended configuration;
- no benchmark-specific search except required preprocessing;
- exact source and revision;
- useful for reproducibility and practical adoption.

### Track B — equal-compute selected

- same predeclared wall-time and trial cap per method-dataset cell;
- method-native selection objective on selection rows only;
- all trials, failures, timeouts, and costs retained;
- no final-test access.

### Track C — rate-constrained

- optimize robust transfer subject to learned-payload budgets;
- same search opportunity per method and budget;
- method-specific capacity/quantization/pruning knobs declared before results;
- one point per successful byte budget;
- no artificial zero when a method cannot fit the budget.

Track C produces the QTDL frontier.

## 7.2 Byte budgets

Use a frozen log-scale grid with the existing cap represented exactly:

```text
256 B
512 B
1 KiB
2 KiB
4 KiB
8 KiB
10 KiB
16 KiB
32 KiB
64 KiB
256 KiB
1 MiB
4 MiB
16 MiB
64 MiB
native/unbounded
```

Methods may contribute only the budgets they can honestly realize. Interpolation for the frontier must be conservative and documented; QTDL should normally use observed points or a monotone upper envelope, not optimistic interpolation.

## 7.3 Baseline families

### Controls

- target null;
- independent marginals;
- predictor-only generator;
- copied rows / random coreset;
- compressed original table;
- real-row bootstrap at each synthetic size.

### Compact statistical generators

- Gaussian copula;
- vine/copula variant where stable;
- Chow-Liu or Bayesian network;
- adversarial random forests (ARF);
- synthpop CART.

### GAN/VAE generators

- CTGAN;
- TVAE;
- CTAB-GAN+ if licensing and implementation reproducibility pass admission.

### Diffusion and flow generators

- TabDDPM;
- TabSyn;
- TabDiff;
- CDTD;
- Forest-Diffusion and Forest-Flow;
- open tabular flow-matching implementation;
- later methods only through a documented admission process.

### Autoregressive/language-model generators

- GReaT;
- REaLTabFormer;
- mixed-text generators only in a separate mixed-text track.

### Formal differential-privacy track

At frozen `(epsilon, delta)` values:

- AIM;
- MST/Private-PGM;
- selected SmartNoise synthesizers with complete preprocessing accounting.

Formal-DP results must be visually separated from empirical-privacy methods. A non-DP method never receives a DP label because it passed attacks.

## 7.4 Baseline admission checklist

A baseline is admitted only when:

- source and license are verified;
- artifact export and offline sample paths are deterministic enough to audit;
- all dataset-specific state can be enumerated and charged;
- training and sampling can be containerized;
- method status codes distinguish failure, timeout, unsupported, and missing;
- configuration and tuning space are frozen;
- the adapter passes artifact-only replay tests.

A failed or absent baseline cell is never a DOPE win.

---

# 8. Dataset benchmark architecture

## 8.1 Separate real-world claims from mechanism tests

Create four suites.

### Suite 1 — `core-real-reg-v1`

Near-term paper suite aligned with current DOPE capability:

- at least 32 real-world regression tasks;
- no synthetic-equation families;
- diverse industries and domains;
- numeric and mixed-type source data, with source-aware preprocessing;
- multiple row and feature regimes;
- immutable licenses and hashes;
- no duplicate or near-duplicate datasets across sources.

This should be the first QTDL publication.

### Suite 2 — `core-real-mixed-v1`

After classification support:

- at least 32 regression and 32 classification tasks;
- balanced binary, multiclass, and regression coverage;
- categorical, numerical, missing-value, and mixed-type tasks;
- curated from maintained real-world benchmark registries such as TabArena and TALENT, with independent license verification.

### Suite 3 — `shift-real-v1`

- temporal and group shifts from TabReD, TableShift, or similarly documented sources;
- synthetic generator fits only on the historical/source domain;
- evaluation on future/target-domain real rows;
- separate `QTDL-shift` result.

### Suite 4 — `mechanism-pmlb-v1`

- current PMLB lineages;
- Feynman/Strogatz/BNG/fri-style mechanism families;
- stress tests, ablations, and regression diagnostics;
- never used as the sole basis for broad real-world leadership claims.

## 8.2 Dataset inclusion rules

Freeze before method results:

- valid redistribution and benchmark use;
- target is explicitly identified and not leaked in features;
- identifiers, post-outcome columns, and duplicated target proxies are removed by documented rules;
- minimum informative real-trained performance;
- enough rows for fit/selection/test without reuse;
- missingness and category rules are declared;
- source provenance and checksum are available;
- no benchmark-training contamination for pretrained auditors is known, or the risk is disclosed;
- task belongs to a declared capability envelope.

## 8.3 Size regimes

To test information density rather than only large-data fitting, create nested fit regimes where possible:

```text
128 rows
512 rows
2,048 rows
full capped training set
```

Use the same untouched test split. This reveals whether a generator is efficient in the low-data setting and whether compact artifacts scale.

## 8.4 Augmentation lane

Replacement is not the only high-value use case. Run a separate augmentation experiment:

- real training fractions: 5%, 10%, 25%, 50%;
- add synthetic rows at fixed multiples;
- evaluate on identical real test rows;
- report lift over the same real-only fraction;
- never average augmentation lift into replacement transfer.

The public result can say:

> “At 10% real data, adding DOPE samples improves median test utility by X.”

This directly addresses the user’s desired “show lift” requirement.

## 8.5 Model-selection transfer

A high-value secondary test asks whether synthetic data preserves decisions, not only final accuracy:

- rank a fixed set of downstream models/hyperparameters on synthetic validation;
- rank them on real validation/test under the same protocol;
- measure rank correlation and regret of the synthetic-selected model on real test;
- report this as **selection transfer**.

This shows whether synthetic data can stand in for real data during model development.

---

# 9. Privacy, novelty, and safety qualification

## 9.1 Hard gates

Privacy must never be a small positive term inside a utility score. Qualification should fail closed on:

- exact training-row copies above zero tolerance;
- near-copy rate above a predeclared holdout-referenced threshold;
- membership-inference lower/upper confidence rule failure;
- attribute-inference advantage above threshold;
- artifact reconstruction or extraction attack failure;
- cleartext/source string leakage;
- unaccounted external files or network reads;
- provenance or replay failure.

## 9.2 Relative distance tests

Absolute nearest-neighbor distances are scale-dependent. Compare synthetic-to-train distances with:

- real-holdout-to-train distances;
- nearest-other-distinct training-row distances;
- matched feature representation and missingness handling.

Report distributions and confidence intervals, not only a gate Boolean.

## 9.3 Attack panel

Minimum empirical panel:

- black-box membership inference with several attack families;
- shadow-model membership attack where feasible;
- attribute inference on declared sensitive columns;
- canary insertion and exposure test;
- artifact inspection/extraction test;
- exact/near duplicate analysis;
- train-vs-holdout distance ratio.

## 9.4 Formal DP lane

For DP methods, record:

- `(epsilon, delta)`;
- adjacency definition;
- accountant;
- preprocessing privacy budget;
- public bounds and categories versus privately inferred bounds;
- composition across tuning and fitting;
- whether model selection consumed privacy budget.

Never say “privacy-preserving” without either a formal DP statement or a precisely scoped empirical claim.

---

# 10. Statistical protocol

## 10.1 Unit of inference

The independent scientific unit is the dataset/task, not rows and not sample seeds.

Nested variation:

1. source-family or dataset cluster where needed;
2. dataset;
3. generator fit seed;
4. synthetic sample seed;
5. auditor seed or context order.

## 10.2 Recommended seeds

Maintain at least:

- five independent generator fit seeds;
- three sample seeds per artifact/size;
- multiple auditor seeds for learned auditors;
- deterministic context-order ensembles for foundation models.

The benchmark must report the actual complete count. “Five seeds intended” is not “five seeds measured.”

## 10.3 Confidence intervals

Use a predeclared nested bootstrap that respects dataset clusters, fit seeds, and sample seeds. For each published scalar:

- point estimate;
- 95% interval;
- denominator and eligibility count;
- missing/failure count;
- sensitivity to complete-case-only versus all admitted cells.

## 10.4 Paired comparisons

For method comparisons:

- pair by dataset, split, auditor suite, sample size, and byte budget;
- report median paired difference and Hodges-Lehmann location;
- use predeclared equivalence/noninferiority margins;
- correct multiplicity within declared families;
- report effect sizes, not only p-values.

## 10.5 Final test

The current sealed-test instinct is correct. Strengthen it:

- freeze metric contract, registry revisions, methods, byte budgets, figure specification, and claims before unsealing;
- create a signed pre-unseal manifest;
- execute once through a restricted runner;
- store immutable receipts;
- never overwrite failed or weak final results;
- label all development/validation plots as such until the final run.

---

# 11. Hero figure and result communication

## 11.1 First-page/README hero figure

Create one plot that communicates the complete thesis.

### Axes

- **x-axis:** learned payload bytes, logarithmic;
- **y-axis:** robust multi-family TSTR transfer lower confidence bound;
- vertical line: 10 KiB existing DOPE target;
- horizontal line: `T=0.90`;
- upper-left region: lightly shaded “qualified transfer region.”

### Marks

- color: method/generator family;
- shape: architecture family;
- filled mark: driver-fidelity gate passed;
- hollow mark: driver fidelity measured but failed;
- grey cross: privacy/replay unmeasured or failed;
- error bars: dataset-level nested interval;
- open/low-opacity point: cohort below the minimum coverage threshold;
- label only DOPE and frontier competitors.

### Annotation

For DOPE, annotate:

```text
DOPE
QTDL90/80 = ... KiB
portable closure = ... MiB
N = ... real tasks
```

For the current development figure, annotate plainly:

```text
CatBoost validation only — not QTDL certified
```

### Caption

The caption must define:

- byte scope;
- model suite version;
- dataset suite;
- transfer denominator;
- qualification marks;
- test status.

## 11.2 Supporting figures

1. **Auditor heatmap:** methods × auditor families, transfer and driver fidelity.
2. **Rate curves:** transfer versus bytes for each method with observed budgets.
3. **Driver plot:** real versus synthetic feature-importance mass for selected tasks.
4. **Privacy frontier:** utility versus attack risk, separated by formal-DP status.
5. **Augmentation curve:** real-data fraction versus test performance with/without synthetic data.
6. **Failure map:** timeout/unsupported/uninformative/missing by method and dataset.

The current paired-difference chart remains valuable but moves below the hero result.

---

# 12. README redesign

## 12.1 Target

- approximately 800–1,200 words before reference links;
- under roughly 12 KiB of Markdown excluding generated image assets;
- one hero plot;
- one small evidence table;
- a five-minute path from landing page to first artifact;
- formulas and certification details moved to docs.

## 12.2 Required order

1. logo/name/badges;
2. one-sentence thesis;
3. three-line explanation;
4. hero figure;
5. honest current result card;
6. install;
7. three-command quick start;
8. “how measurement works” in four lines;
9. scope and current limitations;
10. links to paper, benchmark, docs, contributing, security.

## 12.3 Remove from the top-level README

Move to documentation:

- complete MFS-v4 component algebra;
- all hard-gate thresholds;
- preprocessing implementation detail;
- debug runtime conformance narrative;
- historical V1 compatibility details;
- internal campaign and audit commands;
- long baseline ledgers;
- paper build internals;
- agent-specific instructions.

## 12.4 Remove or replace confusing top-level signals

- remove the nonstandard “Jankurai gate” badge from the public top row;
- do not tell new users to start with `AGENTS.md`;
- replace with standard CI, release, license, paper, docs, and reproducibility badges;
- keep internal audit tooling in `ops/` and developer docs.

## 12.5 Proposed first screen

```markdown
# DOPE

**Compile a table into a kilobyte-scale generator — then test whether modern
models trained on its samples still generalize to untouched real data.**

DOPE stands for **Downstream Objective-Preserving Encoding**. It learns an
executable sampling program rather than storing a synthetic table. The central
question is simple: **how few learned bytes can preserve real-world learning and
the same predictive drivers?**

[hero figure]

### Current development evidence

| Learned payload | CatBoost transfer | Benchmark | Status |
|---:|---:|---|---|
| 1.9 KiB median | 0.898 [interval] | 99 PMLB regression lineages | Validation-only |

The modern cross-model QTDL result, complete driver-fidelity gate, privacy panel,
and sealed final test are in progress. We do not report them as completed.
```

## 12.6 Public CLI experience

The native core may retain a strict normalized format, but the public interface should accept ordinary data:

```bash
pipx install dope-tabular

dope fit data.parquet --target sale_price --budget 10KiB -o house.dope
dope inspect house.dope
dope sample house.dope --rows 10000 -o synthetic.parquet
dope evaluate house.dope --real-test test.parquet --suite open-2026.10
```

Python:

```python
from dope import Generator, evaluate

model = Generator(byte_budget="10KiB", task="regression")
model.fit(train_df, target="sale_price")
synthetic = model.sample(10_000)
report = evaluate(model, real_test=test_df, suite="open-2026.10")
```

The fitted preprocessing sidecar is part of learned-payload byte accounting.

---

# 13. Paper redesign

## 13.1 Recommended title

**DOPE: Kilobyte Generators for Synthetic-to-Real Tabular Learning**

Alternative, more conceptual:

**How Few Bytes Can Teach? Generative Compression for Tabular Learning**

Use the first for a conservative research venue; use the second as a talk or project-page title.

## 13.2 Abstract structure

The abstract should have five moves:

1. problem: synthetic data is usually judged by resemblance or one downstream model, while model size is ignored;
2. method: DOPE compiles a table into a bounded executable generator;
3. metric: QTDL measures the minimum learned bytes that pass cross-model transfer, driver, privacy, and replay requirements;
4. benchmark: real-world tasks, strong open generators, and modern tabular foundation models;
5. result: one precise, appropriately qualified statement from the sealed benchmark.

### Draft abstract with placeholders

> Synthetic tabular data is commonly evaluated by distributional resemblance or the performance of one downstream learner, while the amount of dataset-specific information stored by the generator is rarely treated as a first-class variable. We introduce DOPE, Downstream Objective-Preserving Encoding, which compiles a table into a bounded executable sampling program. We also introduce Qualified Transfer Description Length (QTDL): the minimum learned payload needed for models trained only on generated rows to recover a declared fraction of real-trained performance on untouched real data, preserve the same predictive drivers, and pass privacy, novelty, and artifact-only replay gates. We evaluate DOPE and open-source statistical, adversarial, diffusion, flow, and autoregressive generators across [N] real-world [regression/classification] tasks using gradient-boosted trees, neural tabular models, transparent learners, AutoML, and modern tabular foundation models including Kumo Tabular, Mitra-v2, and TabICLv2. At the preregistered `T90/D80` operating point, DOPE requires [X] KiB of learned payload, compared with [Y–Z] for the nearest qualified alternatives, while [sealed-test transfer and driver result]. These results position synthetic generation as a rate-transfer problem: the goal is not merely to imitate a table, but to encode the smallest executable description that still teaches independent learners what matters on real data.

Do not fill placeholders until the final sealed result exists.

## 13.3 Paper organization

1. Introduction: “How many bytes are needed to teach?”
2. Task-conditioned rate-transfer formulation.
3. DOPE compiler and artifact.
4. Byte accounting and qualification.
5. Benchmark design and preregistration.
6. Main qualified frontier and QTDL results.
7. Cross-model and driver transfer.
8. Privacy and memorization.
9. Augmentation and model-selection transfer.
10. Ablations and failure analysis.
11. Limitations and scope.
12. Related work, or earlier depending venue convention.

## 13.4 First-page figure

The first page should contain the hero frontier, not a dense table and not a metric architecture diagram. The figure must answer:

> Who preserves the most robust learning at the fewest learned bytes, and who actually qualifies?

## 13.5 Claim ladder

Use four explicit evidence labels throughout:

- **Development:** validation or benchmark-design evidence;
- **Matched benchmark:** paired, frozen, but not final test;
- **Sealed final:** one preregistered final run;
- **Release-certified:** artifact/privacy/replay gates complete.

Every table and figure should carry one label.

## 13.6 Move to supplement

- complete metric contracts;
- per-dataset rows;
- all seed-level values;
- baseline adapter details;
- exact preprocessing rules;
- privacy attack implementation;
- runtime/container closure;
- historical MFS-v2/v3/v4 continuity;
- paper audit history.

---

# 14. Repository reorganization

## 14.1 Target tree

```text
/
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── GOVERNANCE.md
├── LICENSE
├── CITATION.cff
├── Cargo.toml
├── crates/
│   ├── dope-core/
│   ├── dope-format/
│   ├── dope-metrics/
│   └── dope-cli/
├── python/
│   └── dope/
├── benchmarks/
│   ├── README.md
│   ├── registry/
│   │   ├── datasets.yaml
│   │   ├── generators.yaml
│   │   ├── auditors.yaml
│   │   └── suites.yaml
│   ├── protocols/
│   │   ├── qtdl-v1.yaml
│   │   ├── privacy-v1.yaml
│   │   └── splits-v1.yaml
│   ├── adapters/
│   │   ├── generators/
│   │   └── auditors/
│   ├── runner/
│   ├── metrics/
│   ├── plots/
│   └── results/
├── contracts/
│   ├── artifact-format-v1.json
│   ├── qtdl-contract-v1.json
│   └── result-schema-v1.json
├── docs/
│   ├── concepts/
│   │   ├── predictive-compression.md
│   │   ├── rate-transfer-frontier.md
│   │   └── qtdl.md
│   ├── guides/
│   │   ├── quickstart.md
│   │   ├── ordinary-data.md
│   │   ├── benchmark-reproduction.md
│   │   └── adding-a-generator.md
│   ├── benchmark/
│   │   ├── protocol.md
│   │   ├── datasets.md
│   │   ├── methods.md
│   │   ├── auditors.md
│   │   └── latest-results.md
│   ├── security/
│   │   ├── threat-model.md
│   │   └── privacy-evaluation.md
│   └── reference/
│       ├── artifact-format.md
│       ├── byte-accounting.md
│       ├── cli.md
│       └── result-schema.md
├── paper/
│   ├── main.tex
│   ├── supplement.tex
│   ├── references.bib
│   ├── figures/
│   ├── generated/
│   ├── scripts/
│   └── README.md
├── research/
│   ├── experiments/
│   ├── receipts/
│   └── audits/
│       └── archive/2026-10/
├── examples/
├── ops/
│   ├── ci/
│   ├── agent/
│   └── release/
└── .github/
    ├── workflows/
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

## 14.2 Migration rules

Do not begin by moving every Rust file. That creates noise and breaks evidence hashes. Use two stages:

### Stage 1 — documentation and benchmark cleanup

- create new contracts and benchmark registry;
- move paper audits and historical notes to archive;
- shorten README;
- preserve current Rust paths and historical contracts;
- add redirects/index files where needed.

### Stage 2 — crate/package normalization

After benchmark release is frozen:

- split artifact format, metric, core compiler, and CLI into crates;
- create stable Python bindings/wheels;
- retain compatibility shims for old commands;
- update evidence hashes deliberately with a migration receipt.

## 14.3 Paper archive move

Move these classes of files out of the canonical paper directory:

- `BASELINE_GAP.md`;
- `IEEE_READINESS.md`;
- `REVIEW_GAPS.md`;
- `SELF_REVIEW.md`;
- `RETAINED_EVIDENCE_REVIEW.md`;
- historical decision and comprehension notes;
- obsolete paper versions.

Destination:

```text
research/audits/archive/2026-10/paper/
```

Retain a short `paper/README.md` explaining exactly how to build and verify the current manuscript.

---

# 15. Contracts and implementation interfaces

## 15.1 New contract files

Create:

```text
contracts/qtdl-contract-v1.json
contracts/byte-accounting-v1.json
contracts/driver-fidelity-v1.json
contracts/privacy-qualification-v1.json
contracts/benchmark-result-v1.schema.json
```

Do not call the new contract MFS-v5. A new name is necessary to signal a different scientific object rather than another weighted-score revision.

## 15.2 Generator adapter

```python
class GeneratorAdapter(Protocol):
    identity: MethodIdentity

    def fit(
        self,
        train: TableRef,
        schema: Schema,
        seed: int,
        compute_budget: ComputeBudget,
        byte_budget: int | None,
        output_dir: Path,
    ) -> FitReceipt: ...

    def sample(
        self,
        artifact: ArtifactRef,
        rows: int,
        seed: int,
        output: Path,
    ) -> SampleReceipt: ...

    def inspect(self, artifact: ArtifactRef) -> ArtifactInventory: ...

    def enumerate_learned_state(self, artifact: ArtifactRef) -> LearnedStateLedger: ...
```

The sampler process receives only:

- artifact;
- fixed method runtime image;
- row count;
- sample seed;
- output path.

It must not receive training data, selection data, hidden sidecars, or network access.

## 15.3 Auditor adapter

```python
class AuditorAdapter(Protocol):
    identity: AuditorIdentity
    family: str

    def fit_or_context(
        self,
        x: TableRef,
        y: VectorRef,
        seed: int,
        budget: AuditorBudget,
    ) -> AuditorState: ...

    def predict(self, state: AuditorState, x_query: TableRef) -> PredictionRef: ...

    def explain(
        self,
        state: AuditorState,
        real_test: TableRef,
        protocol: DriverProtocol,
    ) -> DriverEvidence: ...
```

## 15.4 Result record

Every cell should include at least:

```text
benchmark_release
contract_digest
dataset_id / version / source / license
dataset_hashes
split_hashes
task / target / metric
method / revision / license / container digest
selection_track
fit_seed / sample_seed / auditor_seed
requested_byte_budget
learned_payload_bytes
portable_closure_bytes
cold_closure_bytes
learned_atom_count
effective_learned_bits
fit status / failure reason / timeout
sample status / output digest
TRTR / TSTR / NULL raw values
normalized transfer
driver metrics and raw vectors
privacy metrics
qualification gates
compute time / CPU / GPU / peak memory
provenance and replay receipt
```

## 15.5 One source of truth

The README, paper tables, project website, and release notes must all be generated from one signed benchmark release manifest. Hand-entered benchmark numbers should fail CI.

---

# 16. CLI and product requirements

## 16.1 Stable commands

```text
dope fit
dope sample
dope inspect
dope evaluate
dope benchmark
dope doctor
dope verify
dope convert
```

Avoid public versioned command names such as `mfs-v3-holdout`. Historical tools can remain internal.

## 16.2 Ordinary data support

The public path must support:

- CSV and Parquet;
- headers;
- target by name;
- numerical, categorical, Boolean, datetime, and missing values;
- fitted transform export;
- inverse transformation;
- clear errors and schema report;
- all transform bytes charged.

The strict headerless `[0,1]` native contract can remain an internal L3 boundary, but should not be the first user experience.

## 16.3 Inspect output

`dope inspect` should print a one-screen card:

```text
Artifact format:       DOPE/1
Task:                  regression
Rows fitted:           2,048
Learned payload:       1,924 B
Learned atoms:         312
Bits/original cell:    0.47
Generator:             symbolic + residual
Sampling dependencies: dope runtime 0.3.x
Training rows embedded: no
Cleartext strings:     no
Benchmark qualification: not evaluated
Digest:                ...
```

## 16.4 Release packaging

Provide:

- GitHub release binaries for Linux, macOS, and Windows where supported;
- signed checksums;
- SBOM;
- container image for benchmark reproduction;
- Python wheels or a thin Python wrapper that installs the native binary;
- Homebrew/cargo-binstall path if maintainable;
- semver and migration notes.

---

# 17. CI, reproducibility, and open-source quality

## 17.1 Public CI

The current full CI is tied to self-hosted infrastructure. Add a GitHub-hosted public smoke lane so forks and contributors can verify:

- formatting;
- core Rust unit tests;
- Python unit tests;
- contract/schema validation;
- toy artifact fit/sample/replay;
- documentation link checks;
- deterministic artifact golden test.

Keep GPU, paper full rebuild, security scans, and large benchmark replay on trusted self-hosted runners.

## 17.2 Required workflows

```text
ci-public.yml
ci-trusted.yml
paper.yml
benchmark-schema.yml
release.yml
security.yml
docs.yml
```

## 17.3 Governance files

Add:

- `CONTRIBUTING.md`;
- `SECURITY.md` with private reporting process;
- `CODE_OF_CONDUCT.md`;
- `GOVERNANCE.md` or maintainer policy;
- issue templates for bug, method adapter, dataset proposal, and benchmark challenge;
- PR template with evidence/contract checklist.

## 17.4 Benchmark reproducibility bundle

Each benchmark release publishes:

- registry files;
- exact containers and digests;
- dataset acquisition manifests without redistributing restricted data;
- split manifests and hashes;
- result records;
- failure/timeout ledger;
- figure and table source data;
- signed release manifest;
- script to regenerate every public number and plot.

---

# 18. Test and acceptance matrix

## 18.1 Metric tests

- Rust and Python QTDL calculations agree bit-for-bit or within declared floating tolerance.
- Unqualified artifacts return `null/infinite`, not zero.
- Missing auditor families fail closed.
- Negative and >1 transfer values survive raw evidence.
- Low-signal denominators become `uninformative`, not arbitrary values.
- Threshold and confidence-bound behavior has golden fixtures.
- Suite version changes cannot mutate old results.

## 18.2 Byte-accounting tests

- every file opened by sampler is present in access trace;
- every dataset-specific opened file is charged;
- canonical archive is deterministic;
- changed preprocessing changes payload digest and bytes;
- copied-row baseline is fully charged;
- network access is denied;
- symlinks/path traversal cannot escape artifact root;
- runtime and base-weight sizes are measured separately;
- semantic atom ledger reconciles with serialized fields.

## 18.3 Leakage tests

- fit container cannot resolve or open test paths;
- selection process cannot read final test;
- auditor real-test rows never enter context/training;
- target proxies and identifiers are checked by a dataset audit;
- split hashes are immutable after preregistration.

## 18.4 Driver tests

- column reordering does not change grouped driver evidence;
- one-hot/category grouping maps back to the source feature;
- constant features produce zero importance;
- effect-direction protocol is deterministic;
- explanation rows are identical for real- and synthetic-trained models;
- driver score exposes each component and cannot be passed by one high component alone.

## 18.5 Result-generation tests

- README and paper numbers come from one release manifest;
- a hand-edited number causes CI failure;
- plots contain benchmark release and contract digest in metadata;
- method labels and colors come from one registry;
- missing/failure statuses are visible;
- old releases rebuild after new suite versions are added.

## 18.6 Open-source tests

- clean clone can build core on a public runner;
- toy example completes without private paths;
- docs quick start is executed in CI;
- release binary runs `doctor`, `fit`, `sample`, and `inspect` on supported platforms;
- security and dependency scans are documented.

---

# 19. Execution program

## P0 — Narrative freeze

### Actions

- approve QTDL and `KiB@T90/D80` as public North Star;
- freeze MFS-v4 as historical/internal;
- adopt one exact sentence defining DOPE;
- label current numbers as development validation evidence;
- stop adding public metric versions until the new contract lands.

### Exit gate

- signed decision record;
- no README/paper conflict about the active public metric;
- approved terminology and claim ladder.

## P1 — Metric and accounting contract

### Actions

- implement QTDL contract and schema;
- implement three byte scopes;
- implement learned-atom/effective-bit ledger;
- implement family-balanced transfer and driver gates;
- add Rust/Python golden fixtures;
- create result-card renderer.

### Exit gate

- synthetic fixtures pass cross-language parity;
- at least three radically different artifact types reconcile their byte ledgers;
- no current benchmark result is yet promoted to QTDL unless all fields exist.

## P2 — Benchmark substrate

### Actions

- create dataset/generator/auditor registries;
- define adapter APIs;
- containerize fit and artifact-only sampling;
- implement access tracing and byte accounting;
- implement status/failure ledger;
- implement test isolation;
- implement release manifest.

### Exit gate

- DOPE, copied rows, Gaussian copula, ARF, and one neural baseline run end-to-end on five public smoke datasets;
- every number and byte is traceable to a receipt.

## P3 — Modern auditor and driver suite

### Actions

- integrate Kumo Tabular, Mitra-v2, TabICLv2;
- integrate tree, neural, transparent, and AutoML families;
- implement common explanation protocol;
- validate eligibility/capability matrix;
- freeze `auditor-suite-open-2026.10` or successor.

### Exit gate

- all mandatory families produce TSTR/TRTR/NULL and driver evidence on smoke suite;
- model revisions and licenses are pinned;
- no model-specific hidden preprocessing remains.

## P4 — Baseline and dataset evidence

### Actions

- curate `core-real-reg-v1`;
- run author-default and equal-compute tracks;
- implement rate-constrained variants;
- complete privacy panel;
- run augmentation and selection-transfer lanes;
- freeze and execute sealed final test.

### Exit gate

- at least 90% of admitted primary cells have terminal dispositions;
- missing cells are accounted, never imputed as losses;
- QTDL can be computed or declared unqualified for every admitted method;
- final result is immutable and signed.

## P5 — README, paper, and project site

### Actions

- replace README with the short public narrative;
- generate hero figure from final manifest;
- rewrite paper around rate-transfer/QTDL;
- move audits to archive;
- publish docs site and benchmark result explorer.

### Exit gate

- a new reader can explain the project after the first screen;
- README and abstract use the same question, metric, dataset suite, and result;
- all public claims link to evidence.

## P6 — Open-source release

### Actions

- add contribution/security/governance files;
- add public CI;
- ship binaries/wheels/container;
- publish signed benchmark bundle;
- tag stable research release.

### Exit gate

- clean-clone reproduction succeeds on public infrastructure for the smoke suite;
- external contributor can add a generator adapter from documented interfaces;
- release has checksums, SBOM, licenses, and citation metadata.

---

# 20. First ten pull requests

1. **`docs: freeze public thesis and QTDL decision`**  
   Add decision record, terminology, and migration note; no metric code yet.

2. **`docs: replace root README with focused landing page`**  
   Move MFS algebra and preprocessing detail into docs; retain current results with development label.

3. **`contracts: add qtdl-v1 and byte-accounting-v1 schemas`**  
   Include fixtures and validation tests.

4. **`metrics: implement transfer and QTDL parity in Rust and Python`**  
   Add low-signal handling, LCB input contract, and unqualified semantics.

5. **`artifacts: add canonical learned-state inventory and access trace`**  
   Reconcile payload, portable closure, cold closure, and learned atoms.

6. **`benchmark: introduce versioned dataset/generator/auditor registries`**  
   Replace scattered labels and method status with one source of truth.

7. **`benchmark: add generator and auditor adapter protocols`**  
   Migrate DOPE and core controls first.

8. **`benchmark: add modern open auditor smoke suite`**  
   Kumo, Mitra-v2, TabICLv2, CatBoost, tree peer, transparent model.

9. **`paper: archive historical audits and create canonical paper package`**  
   Keep history, remove it from submission surface.

10. **`ci: add public smoke workflow and generated-claim drift gate`**  
    Execute quick start; fail on hand-edited README/paper numbers.

---

# 21. Definition of done

The reset is complete only when all of the following are true:

## Scientific clarity

- one sentence defines the project;
- one frontier is the primary result;
- one fixed-threshold metric is the headline;
- metric quality is constrained, not hidden in weights;
- current claims distinguish development, sealed test, and release certification.

## Benchmark strength

- strong open generators are measured under author-default, equal-compute, and rate-constrained tracks;
- modern foundation models and conventional strong learners audit transfer;
- real-world datasets are the headline population;
- PMLB is a mechanism suite;
- driver fidelity, augmentation, privacy, and model-selection transfer are measured;
- all missing/failed cells are visible.

## Information accounting

- learned payload is exact and canonical;
- preprocessing and sidecars are charged;
- learned atoms/effective bits are available;
- portable and cold closures are reported;
- external priors are disclosed;
- copied-row and compressed-original controls are present.

## Communication

- README first screen explains the project without formulas;
- hero plot shows the entire claim;
- paper title and abstract match the benchmark;
- no null metric dominates the story;
- no historical audit clutter appears as the canonical paper.

## Open source

- a clean clone runs the quick start;
- public CI works for forks;
- releases are signed and packaged;
- contributor, security, and governance paths exist;
- every result is reproducible from a signed manifest.

---

# 22. Risks and controls

| Risk | Consequence | Control |
|---|---|---|
| QTDL thresholds appear arbitrary | Reviewer resistance | Freeze before final results; show sensitivity grid; make frontier primary |
| Foundation models change rapidly | Benchmark instability | Versioned auditor suites; never mutate old releases |
| One auditor family fails on some tasks | Coverage bias | Capability matrix, family-balanced coverage, explicit eligible denominators |
| Byte accounting favors fixed-prior models | Misleading compression claims | Report learned payload and full shared-base/portable closure separately |
| Driver explanations are unstable | False fidelity conclusions | Common real-test rows, repeated permutation, bootstrap, component-level reporting |
| Current PMLB result is overinterpreted | Credibility loss | Label development-only; real-world suite becomes headline |
| Strong baselines are expensive | Incomplete matrix | Admission priority, terminal status ledger, staged release, no missing-as-win |
| Privacy attacks are mistaken for guarantees | Unsafe messaging | Formal-DP track separated; empirical claims precisely scoped |
| Repository move invalidates evidence hashes | Reproducibility break | Archive first; structural code migration only after benchmark freeze, with migration receipts |
| README becomes long again | Narrative decay | README size/section lint and docs routing policy |

---

# 23. Anti-goals

Do **not**:

- create MFS-v5 as another weighted sum;
- call compression itself intelligence;
- claim a universal result from CatBoost alone;
- use Kumo both as the sole transfer judge and sole representation oracle;
- treat a missing or timed-out baseline as a loss;
- hide model weights, preprocessing, or category dictionaries outside byte accounting;
- call empirical attack resistance differential privacy;
- put the complete research ledger in the root README;
- rewrite historical contracts in place;
- open the final test before metric, methods, plots, and claims are frozen.

---

# 24. Final positioning

The profound version of DOPE is not “another synthetic data generator” and not “a 10 KiB model.” It is a research program around a stronger objective:

> **Find the shortest executable encoding of a dataset that can still teach independent modern models what the real data teaches—on real unseen cases, with the same predictive drivers, without memorizing the source.**

That framing unifies compression, transfer, model diversity, interpretability, privacy, and open benchmarking without burying the idea inside an arbitrary composite score.

The project already has much of the difficult engineering discipline. The next step is subtraction: one claim, one contract, one frontier, one hero figure, one canonical benchmark release, and one clean path for a reader to verify it.

---

# Appendix A. Recommended naming

| Object | Name | Public display |
|---|---|---|
| Project | Downstream Objective-Preserving Encoding | DOPE |
| Scientific object | Qualified rate-transfer frontier | frontier plot |
| Headline metric | Qualified Transfer Description Length | QTDL |
| Fixed operating point | QTDL at 90% transfer / 80% driver fidelity | `KiB@T90/D80` |
| Whole-curve secondary | Qualified Rate-Transfer Area | QRTA |
| Payload | dataset-specific learned payload | learned KiB |
| Semantic count | learned atom count | atoms |
| Deployment size | portable sampling closure | closure MiB |
| Current evidence | development validation result | explicitly labeled |

# Appendix B. Suggested source files to preserve as historical evidence

Do not delete:

- existing `production/kpi-contract*.json` files;
- current MFS-v2/v3/v4 implementations and golden fixtures;
- predeclarations and sealed-test authorization records;
- benchmark result receipts;
- paper review/gap ledgers;
- current generated figures and PDFs;
- method registries and source digests.

Move historical narrative artifacts to a dated archive and add a machine-readable index explaining which contract and paper release each file belongs to.

# Appendix C. External benchmark/model references for implementation planning

The implementation team should pin primary sources and exact revisions before admission. Current relevant projects include:

- NVIDIA Structured Data Models / Kumo Tabular / TabICLv2 / TabFM;
- Mitra-v2 through its open release and AutoGluon integration;
- TabArena and BeyondArena;
- TALENT;
- TableShift;
- TabReD;
- TabSyn;
- TabDiff;
- CDTD;
- TabDDPM;
- ARF/arfpy;
- ForestDiffusion/Forest-Flow;
- REaLTabFormer;
- tabular flow matching;
- OpenDP SmartNoise AIM and MST.

Licenses, weight terms, capability limits, and commits must be recorded in the registry rather than assumed from method names.
