# DOPE vNext
## Engineering, Scientific, Benchmark, README, and Paper Overhaul Specification

**Status:** Proposed  
**Evidence cutoff:** 2026-10-15  
**Scope:** `neverhuman/dope`, especially `README.md`, `docs/whitepaper/`, `docs/vnext/`, the MFS scoring code, transfer adapters, artifact accounting, and benchmark machinery.

---

# 0. Executive decision

DOPE has a genuinely strong idea and unusually serious reproducibility machinery. The public project is nevertheless inverted:

- The **one profound idea** is buried.
- Historical scoring versions, audit machinery, source-custody records, partial benchmark campaigns, private operational details, and future contracts dominate the public surface.
- The README presents **MFS-v4** as the conceptual center even though the active production contract is still v2, the paper and generated assets remain largely v3-oriented, v4 is not measured, and `docs/vnext/` is already moving toward a different and better endpoint.
- The headline evidence remains substantially **CatBoost-centered**, while the repository itself says the next claim requires held-out learner and source families.
- The current compactness term is not a defensible information measure. In particular, `encoded_numbers` can be changed by packing values into different scalar containers without changing the information content.

The project should make one decisive transition:

> **DOPE should measure the smallest complete, charged sampling artifact that preserves learning on untouched real data across frozen modern model families.**

The public explanation should be:

> **Train or condition models on synthetic data. Test them on untouched real data. Count every byte required to preserve the learning.**

The theoretical framework should be **task-conditioned rate–distortion**. The practical metric should be **bytes to qualified transfer**. The benchmark should publish a **qualified-coverage-versus-bytes frontier**, not an opaque weighted master score.

Feature-driver agreement, exact-row copying, privacy evidence, downstream compute, model-family coverage, and complete artifact closure should be **qualification constraints and gates**, not small weighted terms that can compensate for one another.

---

# 1. Blunt assessment of the current project

## 1.1 What is excellent and should be preserved

DOPE already contains several unusually strong foundations:

1. **The artifact is executable.** A small sampling program is more interesting than another table-to-table generator.
2. **The repository attempts exact artifact charging.** This is central to the intellectual contribution.
3. **The evaluation direction is correct.** Train-on-synthetic/test-on-real, paired real controls, and sealed partitions are the right scientific structure.
4. **The code is fail-closed.** Missing evidence becomes unavailable rather than silently becoming a win.
5. **The project records hashes, seeds, identities, receipts, and dependency closure.**
6. **The repository is already questioning its own prior metric.** `docs/vnext/` correctly says bytes should be a constraint/frontier axis rather than a score weight.
7. **The current historical size result is potentially striking.** The committed historical headline data reports a median DOPE artifact around 1.9 KB, while the scaled TabSyn artifact is around 42.3 MB. That is a compelling hypothesis to confirm under a modern, held-out transfer protocol.
8. **The vNext adapter design is serious.** It distinguishes scratch training, frozen context, fine-tuning, query access, context caps, exact checkpoint identity, resource limits, and evaluator custody.

The overhaul should not throw away this rigor. It should hide the rigor behind a much simpler scientific and product interface.

## 1.2 What is currently broken

### A. There are too many public truths

The public repository simultaneously exposes:

- MFS-v2 as the active embedded production contract.
- MFS-v3 in paper filenames, paper build machinery, binaries, generated assets, and historical scoring.
- MFS-v4 in the README and newer fitness code.
- A different vNext behavior contract in `docs/vnext/` and `production/behavior-contract-v1.json`.
- A Rust current path and a Python V1 compatibility path.
- A `mfs-v3-holdout` binary even while the README foregrounds v4.
- Multiple claims ledgers, audit reports, readiness reports, gap reports, and decision logs.

This is scientifically careful internally, but disastrous as a public product story.

### B. The public metric is more complicated than the idea

MFS-v4 combines:

- utility transfer,
- driver fidelity,
- representation closeness,
- knowledge compression,
- distribution fidelity,
- structure fidelity,
- coverage realism,

with manually selected weights and a geometric mean.

That creates four problems:

1. A reader cannot explain the score in one sentence.
2. A method can compensate for weakness in one dimension with strength in another.
3. Weight changes can reorder methods.
4. A single number hides the actual utility-versus-size frontier.

### C. The current compression component is gameable

The current `knowledge_compression_metrics` uses:

- bits per original cell,
- number of encoded numbers,
- a parameter sparsity term normalized by `columns × 8`,
- a logarithmic byte score that reaches zero at the valid 10,240-byte cap.

Concrete examples demonstrate the problem.

For the same unchanged 1,895-byte artifact:

- On an `800 × 13` source table, the current bits-per-cell subscore is approximately **0.407**.
- If the source rows are merely duplicated to `8,000 × 13`, the same artifact receives approximately **0.873**.

No new knowledge was compressed. Only the denominator changed.

Likewise, for 13 columns:

- `encoded_numbers = 100` gives a sparsity subscore of approximately **0.510**.
- Packing those values into one large integer gives approximately **0.990**.

Again, no information disappeared. The serialization vocabulary changed.

Finally, the logarithmic byte term is exactly zero at 10,240 bytes, despite 10,240 bytes being an allowed artifact. Compactness is therefore penalized both by a hard cap and by a second continuous term.

These behaviors disqualify `encoded_numbers` and the current compactness score as the headline scientific measure.

### D. The strongest claim is not held out

The historical paper currently says the profile comparison is selection-exposed across the lineages and three auditors. The vNext claim ledger correctly withholds a transferable-knowledge claim until there are:

- held-out learner families,
- held-out source families,
- frozen original-unit paired losses,
- complete required cells,
- simultaneous uncertainty,
- a sealed confirmation run.

The public README should not visually outrun that disclosure.

### E. The paper's first-page plot asks the wrong question

The current first-page figure is essentially:

> CatBoost retention at `4n` versus median charged bytes.

That plot is useful historical evidence, but it is not the project’s ultimate claim. It should be replaced by:

> **What fraction of frozen tasks qualifies at each complete charged byte budget across tree, neural, and tabular-foundation-model auditors?**

### F. The benchmark surface is an operational ledger, not a benchmark product

`research/benchmark/README.md` reads like a long-running lab notebook:

- dated fit counts,
- partial cohorts,
- host-specific scratch constraints,
- private machine names,
- unresolved loader custody,
- incomplete source audits,
- many null scores,
- dozens of internal commands.

That material can be valuable evidence. It should not be the primary public benchmark documentation.

### G. Model names appear before model qualification

The v4 contract requires Kumo, CatBoost, Mitra, and TabICL, but current locks still record unresolved or unqualified status for several advanced models. A model name in a contract is not an evaluated model. The README must distinguish:

- planned,
- adapter implemented,
- locally reproducible,
- native-parity verified,
- license cleared,
- admitted to headline evaluation,
- measured.

### H. The package experience is not coherent

The project currently exposes:

- Rust package and binaries under `dope-kernel`,
- a Python package also named `dope-kernel`,
- a legacy `dope-kernel-v1` script,
- a separate benchmark binary,
- an `mfs-v3-holdout` binary,
- advanced command sequences that are unsuitable as the first user experience.

There must be one public command: `dope`.

---

# 2. North-star product and scientific thesis

## 2.1 One-sentence product thesis

> **DOPE compiles a table into a tiny sampling program whose output can still teach modern models to predict on unseen real data.**

## 2.2 One-sentence metric thesis

> **The score is the smallest tested number of charged bytes required to meet a frozen transfer standard.**

## 2.3 One-sentence benchmark thesis

> **At each byte budget, measure the fraction of real tasks for which every required learner family preserves predictive gain and feature drivers on untouched real rows.**

## 2.4 What DOPE is not

DOPE should not claim that one task-specific artifact is automatically:

- a reusable replacement for every analysis on the table,
- a population-faithful synthetic dataset,
- causally faithful,
- differentially private,
- legally de-identified,
- a universal measure of dataset intelligence,
- an information-theoretic global minimum,
- a replacement for relational or temporal data without separate evidence.

## 2.5 The strongest defensible language

Before held-out confirmation:

> **Generative compression of predictive structure.**

After held-out learner and source-family confirmation:

> **Transfer-preserving generative compression.**

Only after the full confirmation campaign succeeds should the project use:

> **Compressing data into transferable predictive knowledge.**

Avoid “intelligence per byte.” The scientifically defensible concept is:

> **predictive behavior retained per charged bit under a declared task and evaluator family.**

---

# 3. Replace MFS with task-conditioned rate–distortion

## 3.1 Formal framework

Let a frozen workload cell be:

\[
t = (d, y, m, s, \ell, \pi)
\]

where:

- \(d\) is the dataset and frozen split,
- \(y\) is the target,
- \(m\) is the learner/auditor,
- \(s\) is the synthetic sample or context-size policy,
- \(\ell\) is the loss in its declared original or probabilistic unit,
- \(\pi\) is the full recipe, preprocessing, resource, seed, and query policy.

For an artifact \(A\), define:

- \(L_{0,t}\): loss of a frozen null or no-context control on untouched real query rows,
- \(L_{R,t}\): loss of the auditor trained or conditioned on the real training view,
- \(L_{A,t}\): loss of the same auditor trained or conditioned only on samples from \(A\).

For lower-is-better losses, define the real-data learnable gain:

\[
G_t = L_{0,t} - L_{R,t}
\]

A cell is informative only when \(G_t\) exceeds a pre-registered minimum signal threshold.

Define normalized transfer distortion:

\[
D_t(A) =
\frac{L_{A,t} - L_{R,t}}
{\max(G_t, \epsilon_t)}
\]

and gain retention:

\[
R_t(A) = 1 - D_t(A)
       = \frac{L_{0,t} - L_{A,t}}
              {\max(L_{0,t} - L_{R,t}, \epsilon_t)}
\]

Interpretation:

- \(R = 1\): synthetic-conditioned learning matches real-conditioned learning.
- \(R = 0\): synthetic-conditioned learning is no better than the null.
- \(R < 0\): synthetic data is harmful relative to the null.
- \(R > 1\): synthetic conditioning outperforms the real baseline under this finite experiment.

Do not clip the stored value. Clipping may be used only for a display whose caption says so.

## 3.2 Statistical qualification

A cell qualifies only if the **one-sided lower confidence bound** on transfer retention exceeds the frozen threshold:

\[
\underline{R}_t(A) \ge \tau
\]

Recommended primary threshold:

\[
\tau = 0.90
\]

Sensitivity panels should also report 0.80 and 0.95.

A non-significant difference is not equivalence. The protocol is a noninferiority/qualification protocol with:

- a declared margin,
- paired query losses,
- cluster-aware resampling,
- frozen multiplicity control,
- all required failures retained in the denominator.

## 3.3 Transfer-Preserving Description Length

The theoretical object is:

\[
\mathrm{TPDL}_{\tau,\phi}(t)
=
\min_A b(A)
\]

subject to:

\[
\underline{R}_t(A) \ge \tau
\]

\[
\underline{F}_t(A) \ge \phi
\]

and all artifact, copy, privacy, resource, identity, and completeness gates passing.

Where:

- \(b(A)\) is the complete charged bit length,
- \(F_t(A)\) is driver fidelity,
- \(\phi\) is the frozen driver-fidelity threshold.

The project must not claim the true minimum unless it has a valid global argument. The measurable endpoint is:

\[
\mathrm{TPDL}^{\text{tested}}_{\tau,\phi}(t)
\]

the smallest **tested** budget that qualifies.

## 3.4 Public metric name

Use two names at two levels:

- **Technical name:** Transfer-Preserving Description Length.
- **README name:** **Bytes to qualified transfer.**

Do not lead with another unexplained acronym.

A result card should read like:

> **Bytes to qualified transfer:** 4,096 bytes  
> Smallest tested complete artifact budget preserving at least 90% of real-data gain and the frozen driver standard across every required learner family.

## 3.5 Benchmark-level coverage frontier

Freeze a task set \(W\), task/family weights \(w_t\), budget grid \(B_1 < \dots < B_k\), and quality contract before outcomes are observed.

At budget \(B\):

\[
C_M(B)
=
\sum_{t \in W}
w_t \,
\mathbf{1}
\left[
Q_t(A_{M,t,B}) = 1
\right]
\]

where \(A_{M,t,B}\) is the artifact selected using development evidence only.

Tasks that are:

- failed,
- unsupported,
- timed out,
- OOM,
- missing,
- incomplete,
- over budget,
- privacy-failed,
- copy-failed,
- unqualified,

remain in the denominator.

The benchmark summary is:

\[
B^*_{M,\text{tested}}(q)
=
\min \{B_j : C_M(B_j) \ge q\}
\]

Recommended primary benchmark coverage:

\[
q = 0.90
\]

The public label can be:

> **Smallest tested byte budget qualifying 90% of the benchmark.**

## 3.6 Required result states

Keep and promote the good semantics already present in `docs/vnext/LEARNING_DENSITY.md` and `research/transfer/frontier.py`:

- `measured`
- `censored_below_grid`
- `no_feasible_tested_artifact`
- `non_monotone`
- `pending`

Never interpolate between untested budgets in the headline.

## 3.7 Why not utility per KB

Do not use `utility / KB` or qualified coverage per byte as the primary ranking.

A ratio can:

- reward a tiny but barely useful artifact,
- explode near zero bytes,
- hide absolute failure to meet a useful quality floor,
- behave badly when aggregated across heterogeneous workloads,
- encourage optimizing the denominator instead of satisfying the task.

The frontier plus smallest qualifying tested budget answers the real question directly.

## 3.8 Why not a weighted master score

Retire MFS from the public interface because:

- model transfer must not compensate for privacy failure,
- feature-driver failure must not be canceled by marginal fidelity,
- a huge artifact must not receive a high score merely because it is accurate,
- a tiny useless artifact must not receive a high score merely because it is small,
- arbitrary weights obscure scientific meaning.

A weighted development objective may exist internally. It must never be the paper’s primary endpoint.

---

# 4. Complete bit accounting

## 4.1 Replace “encoded numbers” with canonical charged bits

The only headline size unit should be:

\[
b(A) = 8 \times \text{canonical complete artifact bytes}
\]

A semantic count such as number of values can be reported for interpretability, but it must not affect scientific ranking.

## 4.2 Define three byte views

Every artifact report should include:

### 1. Dataset-specific payload bytes

All dataset-dependent bytes required to sample after a frozen decoder/runtime is installed.

Use this for the main “how much dataset-specific information is required?” frontier.

### 2. Cold-start closure bytes

Payload plus every method-specific decoder, checkpoint, code bundle, vocabulary, and sidecar required on a clean machine.

Use this to prevent a 1 KB payload backed by a hidden 400-million-parameter checkpoint from being presented as a 1 KB deployable system.

### 3. Amortized deployment bytes

\[
b_{\text{amortized}} =
b_{\text{payload}} +
\frac{b_{\text{shared closure}}}{N_{\text{declared deployments}}}
\]

Report only when the deployment count is frozen and justified.

## 4.3 What must be charged

Charge every dataset-dependent byte read by sampling, including:

- fitted weights,
- quantiles,
- category dictionaries,
- schema state,
- feature transforms,
- target transforms,
- dependency graph,
- entropy tables,
- codebooks,
- residual tables,
- calibration state,
- selected seed,
- filenames or metadata if the sampler uses them,
- sidecar files,
- method-specific checkpoints,
- data-dependent generated code,
- lookup tables,
- cached examples,
- hidden raw rows.

## 4.4 What may be shared

A decoder may be excluded from payload bytes only when:

- its exact bytes are frozen before the benchmark,
- it is dataset-independent,
- it is available to every artifact in the declared method,
- it cannot be modified per dataset,
- it has no access to original rows,
- the full cold-start size is separately reported.

## 4.5 Canonical container

Define one canonical container, for example `dope-artifact-v1`:

- sorted entry roles,
- canonical UTF-8 names,
- fixed integer encoding,
- fixed endianness,
- fixed metadata,
- no timestamps,
- no ownership fields,
- no symlinks,
- no path traversal,
- no duplicate roles,
- explicit byte lengths,
- content hashes,
- optional fixed versioned compression codec,
- deterministic output bytes.

Do not charge a method’s arbitrary ZIP or TAR metadata differently.

## 4.6 Sampling closure test

A clean sampler process receives only:

1. the frozen shared runtime,
2. the candidate artifact,
3. a public sample request containing row count and sample seed.

The process must run:

- offline,
- without original data,
- without environment secrets,
- without unlisted files,
- without network,
- without mutable caches,
- without hidden checkpoints.

The executor records every opened file. Any unmanifested read invalidates the artifact.

## 4.7 Adversarial accounting tests

The metric test suite must include:

- packing 100 values into one integer,
- duplicating source rows,
- padding the artifact,
- encoding data in filenames,
- encoding data in entry order,
- encoding data in floating-point NaN payloads,
- using a dataset-specific executable,
- using an external checkpoint,
- using a pre-populated cache,
- using a selected random seed as a covert channel,
- storing rows in “metadata,”
- best-of-k seed selection using final data,
- decompression-bomb behavior,
- hidden network fetch.

All must fail or produce unchanged charged bits.

---

# 5. Transfer protocol

## 5.1 Separate compiler, selector, and evaluator

Use three trust domains.

### Compiler domain

Can access:

- real training rows,
- training-only transforms,
- development auditor outputs if allowed by the protocol.

Cannot access:

- final real labels,
- held-out learner outputs,
- sealed source families,
- evaluator secrets.

### Selector domain

Can access:

- development tasks,
- development losses,
- development auditor families,
- budget feasibility.

Selects:

- one artifact/configuration per method, task, budget, and fit seed.

Cannot select using:

- sealed test outcomes,
- held-out learner families,
- private confirmation source families.

### Evaluator domain

Owns:

- untouched real query labels,
- held-out learner checkpoints,
- final metrics,
- confidence intervals,
- claim release.

## 5.2 Required split structure

For each dataset:

- `train`: generator fit and real-trained learner fit.
- `development`: profile/configuration selection.
- `confirmation`: untouched final paired loss.

For temporal or grouped data:

- use entity- or time-respecting splits,
- preserve group IDs for cluster bootstrap,
- prevent entity leakage.

## 5.3 Required source-family holdout

A transferable claim requires source families never used to:

- design the artifact language,
- choose profiles,
- tune thresholds,
- tune compiler hyperparameters,
- select publication settings.

Recommended structure:

- **development families:** public and reusable,
- **confirmation families:** frozen and sealed until final run,
- **challenge families:** released after publication for ongoing evaluation.

## 5.4 Required learner-family holdout

The compiler must not optimize against every model later used in the claim.

Recommended split:

### Development auditors

- CatBoost
- LightGBM
- RealMLP or TabM
- TabICLv2

### Sealed confirmation auditors

- NVIDIA Kumo Tabular
- TabPFN-3.5
- Google TabFM

The exact roster may change only before the protocol is frozen.

At least two genuinely different held-out learner architectures must remain unseen to compiler search.

## 5.5 Synthetic-conditioned terminology

Use a single umbrella phrase:

> **synthetic-conditioned, tested on real**

This includes:

- scratch models trained on generated rows,
- fine-tuned foundation models updated on generated rows,
- in-context models receiving generated rows as context.

Do not describe in-context use as gradient training.

## 5.6 Matched real and synthetic recipes

For every auditor:

- same preprocessing policy,
- same row/context count policy,
- same query batch policy,
- same compute ceiling,
- same ensemble setting,
- same precision,
- same seed schedule,
- same target inverse transform,
- same loss.

The only intended difference is real versus synthetic training/context.

## 5.7 Null and contrast controls

Every cell should include:

- no-feature or target-mean null,
- no-context foundation-model control where applicable,
- predictor-only control,
- independent-marginals control,
- real bootstrap/context control,
- real full-training control.

These distinguish:

- target leakage,
- model prior,
- feature marginal value,
- dependency value,
- genuine synthetic transfer.

## 5.8 Fixed downstream compute and sample policy

Artifact bits alone are insufficient if one method may emit billions of rows.

Freeze:

- requested synthetic rows,
- consumed synthetic rows,
- training steps or wall/GPU budget,
- in-context cap,
- query batch size,
- ensemble count,
- precision.

Report:

- artifact bytes,
- generation wall time,
- generated rows,
- learner preparation time,
- prediction time,
- peak CPU/GPU memory.

The main frontier remains bytes, conditional on the frozen compute contract.

## 5.9 Required metrics

Primary:

- normalized paired transfer retention,
- qualification status,
- complete charged bytes,
- qualified coverage at each budget.

Secondary:

- raw original-unit MSE/RMSE/MAE for regression,
- log loss and Brier score for classification,
- calibration,
- worst-group or tail loss,
- runtime and memory,
- feature-driver fidelity,
- empirical privacy diagnostics,
- distribution and structure diagnostics.

Distributional resemblance must remain secondary. It is not a substitute for transfer.

---

# 6. Feature-driver fidelity

## 6.1 Scientific question

For the same learner architecture and the same untouched real query rows:

> Does the model conditioned on synthetic data rely on the same real-world feature drivers as the model conditioned on real data?

This is stronger than comparing one CatBoost SHAP vector.

## 6.2 Do not compare raw SHAP magnitudes across model families

Raw importance values are not directly comparable across:

- trees,
- neural networks,
- in-context transformers,
- different preprocessing,
- correlated features.

Instead, compare real-conditioned and synthetic-conditioned variants **within each model**, then aggregate qualification across model families.

## 6.3 Evaluator-owned perturbation protocol

Use a frozen evaluator-owned driver protocol:

1. Define feature groups before outcomes.
2. Fit any conditional perturbation model on real training data only.
3. Evaluate both real-conditioned and synthetic-conditioned learners on the same real query rows.
4. Measure loss increase under the same perturbations.
5. Normalize positive contribution mass within each learner.
6. Bootstrap query groups and learner seeds.

Feature grouping is essential for:

- correlated variables,
- one-hot blocks,
- text-derived columns,
- hierarchical or domain-linked features.

## 6.4 Driver components

For each auditor, compute:

### Rank fidelity

- NDCG@k
- rank-biased overlap
- top-k Jaccard

### Contribution-mass fidelity

- total variation similarity
- Jensen–Shannon similarity
- Wasserstein distance over ranked contribution mass

### Direction fidelity

Use fixed-grid accumulated local effects or another frozen perturbation curve:

- sign agreement,
- monotonic trend agreement,
- curve correlation.

### Interaction fidelity

For the top declared interactions:

- conditional pairwise permutation interaction,
- Friedman H-statistic,
- top-pair overlap,
- interaction-mass similarity.

## 6.5 Qualification rule

Avoid a weighted average that permits compensation.

Define normalized subcomponents:

\[
F_{\text{rank}},\;
F_{\text{mass}},\;
F_{\text{direction}},\;
F_{\text{interaction}}
\]

and set:

\[
F_t(A) =
\min(
F_{\text{rank}},
F_{\text{mass}},
F_{\text{direction}},
F_{\text{interaction}}
)
\]

where unsupported interaction fidelity may be governed by a separate frozen applicability rule rather than silently omitted.

Recommended primary driver threshold:

\[
\phi = 0.80
\]

Report sensitivity at 0.70 and 0.90.

## 6.6 Model-family rule

A task qualifies only when the lower confidence bound passes for every required family.

Do not average a strong CatBoost match with a weak foundation-model match.

## 6.7 Consensus-driver diagnostic

Additionally report:

- rank aggregation of real-conditioned auditors,
- rank aggregation of synthetic-conditioned auditors,
- consensus overlap,
- disagreement among real auditors,
- disagreement among synthetic auditors.

This reveals whether synthetic data preserves genuine cross-model agreement or merely one model’s inductive bias.

## 6.8 Causal disclaimer

Feature-driver fidelity is explanatory and predictive, not causal.

Never label importance agreement as causal-structure preservation without a separate causal intervention benchmark.

---

# 7. Auditor portfolio

## 7.1 Headline portfolio

The headline benchmark should include distinct inductive-bias families.

| Family | Required candidates | Role |
|---|---|---|
| Gradient-boosted trees | CatBoost plus LightGBM or XGBoost | Stable strong classical baseline |
| Modern supervised neural | RealMLP, TabM, or an equally qualified replacement | Non-tree trained neural family |
| Open tabular foundation model | TabICLv2 | Development foundation auditor |
| Frontier foundation model | NVIDIA Kumo Tabular | Held-out confirmation auditor |
| Frontier foundation model | TabPFN-3.5 base/Fast | Held-out confirmation auditor |
| Frontier foundation model | Google TabFM | Held-out confirmation auditor |
| Portfolio ceiling | AutoGluon best/extreme preset, frozen | Secondary real-world ceiling |

## 7.2 Research-only frontier lane

Qualify separately, subject to exact code, weights, and license review:

- Mitra-v2
- Xiaomi-TabLDM
- LimiX-2
- EXAONE-Tabular
- other newly released models that pass the adapter contract.

Do not make the core benchmark impossible to reproduce commercially because one research-only model has restrictive weights.

## 7.3 Required adapter states

Every model registry entry must have one of:

- `planned`
- `adapter_implemented`
- `local_smoke_passed`
- `native_parity_passed`
- `dependency_closure_locked`
- `checkpoint_locked`
- `license_cleared`
- `headline_admitted`
- `measured`
- `unavailable`
- `rejected`

A string such as “unqualified” is too coarse.

## 7.4 Admission requirements

No model enters the headline until:

- exact source revision is locked,
- checkpoint bytes are hashed,
- dependency closure is frozen,
- code and weight licenses are reviewed,
- regression/classification support is verified,
- preprocessing is native-parity checked,
- real/context row count is measured,
- hidden truncation is ruled out,
- query order/batching semantics are frozen,
- resource peaks are observed externally,
- clean-machine offline reproduction succeeds,
- prediction parity with a native reference fixture passes.

## 7.5 Foundation-model-specific controls

For every in-context foundation model:

- record full supplied context count,
- record consumed context count,
- reject hidden truncation,
- bind row order,
- bind column order policy,
- bind query batching/transductive behavior,
- include a no-context control,
- preserve exactly the same context policy for real and synthetic conditions,
- separate checkpoint prior from dataset-specific contribution.

---

# 8. Generator benchmark portfolio

## 8.1 Compression controls

These are mandatory because DOPE’s central claim concerns bytes.

- raw CSV plus canonical Zstandard
- canonical Parquet/Arrow compression
- compressed training rows
- random row subset at each budget
- stratified subset
- k-center or medoid coreset
- leverage/sensitivity coreset where applicable
- prototype/teaching-row baseline
- target-only and predictor-only controls

These answer the brutal baseline question:

> Is DOPE better than simply storing the most useful rows?

## 8.2 Classical generators

- independent marginals
- Gaussian copula
- Chow–Liu tree
- Bayesian network baseline
- synthpop CART
- adversarial random forest
- other compact structured density estimators already implemented and reproducible

## 8.3 Established neural generators

- CTGAN
- TVAE
- TabDDPM
- TabSyn
- TabDiff
- Forest-Diffusion / Forest-Flow where the exact implementation is clear
- GReaT or REaLTabFormer only when the task, license, and resource profile are appropriate

## 8.4 Current frontier open-source generators

Prioritize exact qualification for:

- CDTD
- GATD
- TabRep
- TabCascade
- TabularARGN

Add TabFORGE only after exact public code, weights, and license closure are independently verified.

## 8.5 Separate population and teaching/context tracks

This distinction is critical.

### Population track

Claims to approximate a data-generating distribution.

Requires:

- joint support,
- missingness,
- tails,
- marginal and dependency diagnostics,
- population sampling semantics.

### Teaching/context track

Optimizes examples for downstream learning or in-context prediction.

Requires:

- transfer evidence,
- model-family generalization,
- explicit selection policy,
- no population-fidelity claim.

Never compare a teaching set and a population generator as though they make the same claim.

DOPE may support both, but each artifact must declare its mode.

## 8.6 Baseline configuration lanes

For each generator, distinguish:

1. **Author-default**
2. **Native-validation-selected**
3. **Resource-matched**
4. **Byte-constrained**, if the method has a legitimate compression path

Do not invent a severely truncated training recipe and label it as the original method.

## 8.7 Comparator completeness

A method is never counted as a DOPE loss merely because:

- its code failed to install,
- a checkpoint was missing,
- an adapter was incomplete,
- a cohort was partial,
- the method timed out due to an unfair environment.

Such cells remain failures in coverage only when the frozen protocol legitimately attempted the admitted method under the declared resource contract. Otherwise, label them `unavailable` and do not claim superiority.

---

# 9. Dataset benchmark design

## 9.1 Immediate paper scope

Do not expand the core algorithm into every data type before the first clean paper.

The fastest credible path is:

- numeric regression,
- real-world tables,
- modern held-out auditors,
- exact bit accounting,
- sealed source-family confirmation.

State “numeric regression” in the title page, abstract, README, and benchmark card if that remains the scope.

## 9.2 Next capability wave

After the regression claim is frozen:

- binary classification,
- multiclass classification,
- mixed categorical/numeric features,
- missingness,
- grouped and temporal splits.

## 9.3 Proposed benchmark tiers

### Quick-12

- public,
- small,
- runs on a single workstation,
- used for contributor CI and examples.

### Open-60

- public development suite,
- broad domains,
- frozen splits,
- used for method development,
- no final claim of unseen source transfer.

### Sealed-24

- source-family confirmation,
- not used for compiler/profile design,
- opened only under a frozen protocol.

The exact counts may be changed before freeze, but the tier structure should not.

## 9.4 Dataset strata

Include:

- row-count strata,
- width strata,
- low/high target signal,
- linear/nonlinear relationships,
- low/high feature interaction,
- low/high cardinality,
- missingness,
- skew and heavy tails,
- grouped data,
- temporal data,
- engineering/scientific/financial/health/operations domains where rights permit.

## 9.5 PMLB’s future role

Keep the existing 100 PMLB regression lineages as:

- mechanistic stress tests,
- regression tests,
- compiler ablation fixtures,
- continuity with the historical paper.

Do not use a panel dominated by simulated source families as the sole headline evidence for real-world transfer.

## 9.6 Rights and provenance

Every dataset manifest must include:

- landing source,
- immutable digest,
- license or explicit permission,
- target definition,
- feature-use restrictions,
- split policy,
- protected groups if applicable,
- entity/time keys,
- row and column counts,
- duplicate policy,
- leakage review,
- whether the dataset may be redistributed.

Unknown rights block public release even when utility succeeds.

## 9.7 Foundation-model contamination review

For each held-out foundation model, record whether:

- pretraining was synthetic-only,
- public benchmark datasets may have been used for checkpoint selection,
- the exact dataset was potentially exposed,
- a private/sealed confirmation dataset is required.

The strongest confirmation suite should contain datasets unlikely to have appeared in model selection.

---

# 10. Statistical analysis contract

## 10.1 Unit hierarchy

Use the hierarchy:

1. source family,
2. dataset,
3. split/target,
4. generator fit seed,
5. generator sample seed,
6. learner seed,
7. query cluster.

Do not treat sample draws from one fitted artifact as independent generator fits.

## 10.2 Required seed accounting

Record separately:

- compiler seed,
- fit seed,
- artifact/deployment seed,
- sample seed,
- learner seed,
- query order seed,
- perturbation seed,
- bootstrap seed.

Aliases and retries do not create new independent replicates.

## 10.3 Confidence procedure

Use:

- paired query losses,
- cluster-aware bootstrap or an equivalent declared method,
- hierarchy-aware resampling,
- one-sided intervals for qualification,
- simultaneous control across required learner families and primary budgets,
- family-level macro weighting.

## 10.4 Failure handling

All scheduled cells remain in the denominator.

Never use:

- complete-case-only averages,
- best successful seed,
- best artifact selected on final results,
- dropping OOM or timeout methods after outcomes,
- replacing a failure with a smaller cohort.

## 10.5 Aggregation order

Recommended:

1. estimate per-cell paired loss,
2. aggregate seeds within task and learner under the frozen rule,
3. apply confidence qualification,
4. require all mandatory learner families,
5. compute weighted task coverage,
6. bootstrap at source-family and task levels.

Do not pool millions of query rows across datasets and let one large dataset dominate.

## 10.6 Sensitivity analyses

Publish:

- transfer thresholds 0.80, 0.90, 0.95,
- driver thresholds 0.70, 0.80, 0.90,
- payload versus cold-start bytes,
- sample/context sizes,
- development versus held-out learners,
- public versus sealed source families,
- inclusion/exclusion of each foundation-model family,
- regression loss variants.

The primary contract remains frozen; sensitivity panels do not redefine the winner.

---

# 11. Compiler objective and algorithmic innovation

## 11.1 Optimize a frontier, not one scalar

The compiler should maintain a Pareto archive over:

- charged payload bits,
- worst-development-family transfer distortion,
- driver fidelity,
- privacy/copy feasibility,
- runtime.

Do not collapse these into one master score during search unless the scalar is purely an internal acquisition function.

## 11.2 Constrained development objective

A practical internal objective is:

\[
\min_A b(A)
\]

subject to:

\[
\max_{m \in M_{\text{dev}}}
\widehat{D}_m(A) \le \delta
\]

\[
\min_{m \in M_{\text{dev}}}
\widehat{F}_m(A) \ge \phi
\]

and all hard constraints.

A smooth optimization surrogate may be:

\[
J(A)
=
b(A)
+
\sum_m \lambda_m [\widehat{D}_m(A)-\delta]_+
+
\lambda_F[\phi-\widehat{F}(A)]_+
+
\lambda_P P(A)
\]

but the published endpoint remains the constrained tested frontier.

## 11.3 Multi-auditor development

Each candidate should be evaluated against a rotating development portfolio, not only CatBoost.

Use multi-fidelity scheduling:

1. linear and cheap structural checks,
2. CatBoost/LightGBM,
3. modern neural learner,
4. development foundation model,
5. expensive driver checks for frontier candidates.

Sealed auditors are never queried during search.

## 11.4 Optimize worst-family performance

Use:

- worst-family loss,
- lower quantile across development auditors,
- or robust max-risk.

Do not optimize mean performance that allows one learner family to collapse.

## 11.5 Progressive artifact format

A major innovation opportunity is a **progressive DOPE bitstream**.

The artifact is layered:

1. schema/base marginals,
2. coarse target relation,
3. dependency edges,
4. nonlinear residuals,
5. tail corrections,
6. interaction refinements,
7. calibration refinements.

Any valid prefix is sampleable.

Benefits:

- one fit produces a natural rate–distortion curve,
- exact budget truncation,
- direct bit attribution,
- fewer independent profile-selection decisions,
- better explanation of marginal transfer gain per bit.

The benchmark must still report non-monotonic outcomes honestly.

## 11.6 Gain-per-bit allocation

For every candidate component \(c\), estimate:

\[
\frac{\Delta \text{worst-family transfer}}
     {\Delta \text{charged bits}}
\]

Allocate the next bits to components with the highest robust gain, subject to diversity and driver constraints.

## 11.7 Bit attribution

Every artifact inspection should show:

- schema bits,
- marginal bits,
- dependency bits,
- target-relation bits,
- interaction bits,
- tail/calibration bits,
- container overhead,
- total.

This will make the paper materially more profound.

## 11.8 Auditor-disagreement-guided search

Use development auditor disagreement to locate structure the artifact is missing:

- tree succeeds, neural fails,
- neural succeeds, foundation model fails,
- rank fidelity succeeds, direction fails,
- mean loss succeeds, tail loss fails.

Mutations should target the failing structural class rather than blindly increasing artifact size.

## 11.9 Explicit artifact modes

The compiler emits one of:

- `population_sampler`
- `teaching_sampler`
- `foundation_context`
- future `reusable_table`

The mode determines allowed claims and evaluation contracts.

---

# 12. Repository architecture

## 12.1 Recommended repository split

### `neverhuman/dope`

Keep:

- product/compiler source,
- artifact format,
- canonical metric implementation,
- simple CLI,
- public docs,
- Quick-12,
- paper source,
- small reproducible examples.

### `neverhuman/dope-bench`

Move:

- full benchmark harness,
- generator adapters,
- learner adapters,
- dataset manifests,
- full receipts,
- campaign orchestration,
- Open-60/Sealed-24 protocol,
- result aggregation.

### Release/archival storage

Store:

- large immutable result bundles,
- containers,
- checkpoints where redistribution is allowed,
- historical campaign archives,
- rendered paper artifacts.

Cross-pin repositories and result bundles by digest.

## 12.2 If a split is temporarily impossible

Use this clean monorepo structure:

```text
README.md
LICENSE
CITATION.cff
CONTRIBUTING.md
SECURITY.md
CODE_OF_CONDUCT.md

crates/
  dope-core/
  dope-artifact/
  dope-metrics/
  dope-cli/

python/
  dope/
  dope_bench/

spec/
  behavior-v1.schema.json
  artifact-v1.schema.json
  auditor-v1.schema.json
  evaluation-v1.schema.json
  frontier-v1.schema.json

docs/
  concept.md
  metric.md
  artifact.md
  benchmark.md
  transfer.md
  drivers.md
  privacy.md
  reproducibility.md
  limitations.md

bench/
  quick/
  datasets/
  generators/
  auditors/
  protocols/
  reports/

paper/
  main.tex
  supplement.tex
  figures/
  generated/

examples/
  regression/
```

Do not keep private operational ledgers at the public top level.

## 12.3 Historical preservation

Before cleanup:

1. Create an immutable tag such as `historical-mfs-2026-10-15`.
2. Publish the historical paper/result bundle with hashes.
3. Preserve old contract decoders.
4. Stop editing old scorer semantics.
5. Remove superseded narrative files from `main`; Git history and the tag preserve them.
6. Do not create a giant visible `archive/` folder that reproduces the same clutter.

---

# 13. Exact file-level actions

## 13.1 `README.md`

**Replace completely.**

Remove from the first screen:

- MFS-v4 weight table,
- internal audit language,
- long limitation mechanics,
- private campaign detail,
- dozens of commands,
- incomplete model promises.

Add:

- one-sentence thesis,
- one hero frontier plot,
- one metric explanation,
- three-command quick start,
- one historical/current result card with disclosure,
- benchmark link,
- limitations,
- citation.

## 13.2 `docs/whitepaper/dope-mfs.tex`

Freeze as historical after the current artifact is archived.

Create:

- `paper/main.tex`
- `paper/supplement.tex`

Rewrite around:

- task-conditioned rate–distortion,
- exact complete bytes,
- held-out learner transfer,
- held-out source transfer,
- driver fidelity,
- coverage frontier.

## 13.3 `docs/whitepaper/README.md`

Replace the current version-specific description with:

- exact build command,
- required toolchain,
- input result manifest,
- figure provenance,
- expected hashes,
- artifact packaging command.

## 13.4 `docs/vnext/LEARNING_DENSITY.md`

Promote the strongest material into `docs/metric.md`.

Changes:

- retain the frontier and `B*_tested` states,
- replace “nothing is adopted” with the frozen v1 metric after review,
- add transfer and driver qualification equations,
- add payload/cold-start accounting,
- add primary thresholds,
- add confidence semantics.

## 13.5 `docs/vnext/DECISIONS.md`

Split into:

- concise active `docs/decisions.md`,
- historical tagged decision record outside current narrative.

Do not make users read migration hashes and parallel-lane notes to understand the project.

## 13.6 `production/kpi-contract-v4.json`

Mark:

```json
{
  "status": "historical_draft",
  "headline": false,
  "superseded_by": "behavior-contract-v1"
}
```

Do not change its historical score semantics.

Add a new frozen protocol only after the fields are complete.

## 13.7 `production/behavior-contract-v1.json`

This is the correct foundation but currently incomplete.

Complete:

- noninferiority margin,
- catastrophic-harm margin,
- learner portfolio,
- held-out families,
- byte grid,
- payload/cold-start policy,
- context/sample policy,
- dataset manifests,
- rights manifests,
- checkpoint manifests,
- retry/resource policy,
- multiplicity procedure,
- evaluator isolation,
- operator authority.

Change `protocol_frozen` only after all are concrete.

## 13.8 `research/transfer/frontier.py`

Keep the core semantics.

Refactor into the canonical metric package:

- Rust implementation as product truth,
- Python bindings or a generated conformance wrapper,
- confidence intervals,
- driver qualification,
- artifact view selection,
- explicit method/budget cells,
- complete failure taxonomy.

Rename away from exploratory terminology once frozen.

## 13.9 `research/transfer/objectives.py`

Retire from headline scoring:

- `encoded_numbers`,
- arbitrary sparsity normalization,
- logarithmic byte score,
- weighted knowledge-compression score,
- utility-per-encoded-number.

Keep only historical compatibility.

Add:

- normalized paired transfer,
- informative-task gate,
- driver components,
- exact charged-bit manifest consumption,
- qualification report.

## 13.10 `research/benchmark/mfs_v3_score.py`

Stop combining v3 and v4 in one historical module.

Freeze it under the historical tag.

New active code:

```text
bench/metrics/transfer.py
bench/metrics/drivers.py
bench/metrics/qualification.py
bench/metrics/frontier.py
```

## 13.11 `rust/fitness/v4.rs`

Do not evolve it into v5.

Create:

```text
crates/dope-metrics/src/transfer.rs
crates/dope-metrics/src/drivers.rs
crates/dope-metrics/src/frontier.rs
crates/dope-metrics/src/artifact_charge.rs
```

Use generated types from versioned schemas.

## 13.12 `rust/representation.rs`

Move representation-distance metrics to diagnostics.

Representation closeness must not be a required headline component unless a separate hypothesis proves it adds predictive validity beyond transfer.

Remove v3 naming from active paths.

## 13.13 `research/transfer/locks/*.json`

Replace ad hoc status fields with one typed registry schema.

Every entry should include:

- model/version,
- source revision,
- code digest,
- checkpoint digest,
- code license,
- weight license,
- dependency closure digest,
- supported tasks,
- mode,
- context cap,
- preprocessing identity,
- native parity evidence,
- resource evidence,
- admission status,
- rejection reason,
- last verified date.

## 13.14 `research/benchmark/README.md`

Replace with a concise benchmark overview.

Move operational details to:

- machine-readable receipts,
- release bundles,
- internal campaign docs,
- a separate benchmark repository.

Remove:

- private host names,
- scratch-volume narratives,
- partial-fit diary entries,
- long command catalogs.

## 13.15 `Cargo.toml` and `pyproject.toml`

Expose one coherent product:

- package family: `dope`
- CLI: `dope`
- current version: one source of truth
- prebuilt wheels/binaries
- no public `dope-kernel-v1` command
- historical compatibility accessible only under an explicit legacy module.

## 13.16 Generated figures and numbers

Create one machine-readable source:

```text
paper/generated/headline-results.json
```

Both README and paper figures must be generated from it.

CI must fail when:

- a displayed number is hand-typed,
- the figure hash differs,
- a claim references an incomplete cohort,
- method cohorts differ without disclosure,
- a headline model is unqualified.

---

# 14. Public CLI

## 14.1 Required commands

```bash
dope fit data.csv --target y --budget 10KiB --output data.dope
dope sample data.dope --rows 10000 --output synthetic.parquet
dope inspect data.dope
dope verify data.dope
dope evaluate data.dope --real-test test.parquet --suite quick
dope benchmark --suite quick
```

## 14.2 `dope inspect`

Example output:

```text
Artifact format       dope-artifact-v1
Mode                  population_sampler
Payload               1,895 bytes
Cold-start closure    4.8 MiB
Columns               13
Target                median_house_value
Progressive layers    5
Exact-row copies      not evaluated
Sampler replay        verified
Artifact SHA-256      ...
```

## 14.3 Input usability

Support:

- CSV,
- Parquet,
- named target,
- header inference,
- explicit schema overrides,
- numeric and categorical types as capabilities land,
- missing values,
- train/dev split generation with a reproducible manifest.

Do not lead with “headerless numeric values already normalized to [0,1].” Keep that as an internal expert mode.

## 14.4 Install experience

Target:

```bash
pipx install dope-data
```

or:

```bash
uv tool install dope-data
```

Ship:

- Linux x86_64 wheel,
- macOS arm64 wheel,
- macOS x86_64 wheel if practical,
- container image,
- signed checksums,
- SBOM.

The exact package name can change, but the command remains `dope`.

---

# 15. README redesign

## 15.1 First-screen draft

```markdown
# DOPE

### Compress a table into a tiny program that still teaches models to predict.

DOPE compiles a real table into a small sampling artifact. Models are trained
or conditioned only on samples from that artifact, then scored on untouched
real rows.

**The metric is simple: how many complete charged bytes are required to preserve
at least 90% of real-data learning across modern model families?**

[hero figure: qualified benchmark coverage versus charged bytes]

Train on synthetic. Test on untouched real data. Count every byte.
```

## 15.2 Result card

Before the sealed campaign:

```markdown
> Historical validation result — selection-exposed, numeric regression only.
> Median DOPE payload: 1.9 KB on the historical 100-lineage panel.
> This is not yet a held-out learner/source-family transfer claim.
```

After confirmation:

```markdown
> At 10 KiB, DOPE qualified X% of sealed tasks across tree, neural, and
> foundation-model auditors. The next-best admitted open-source generator
> qualified Y%. Full protocol and confidence intervals.
```

Never write the second form until the full matrix exists.

## 15.3 README sections

Keep the root README to approximately:

1. Thesis
2. Hero plot
3. How the metric works
4. Quick start
5. Current evidence
6. Reproduce
7. Limitations
8. Citation

Everything else belongs in docs.

## 15.4 Visual design

Use:

- white or restrained light background,
- near-black typography,
- one DOPE accent color,
- muted comparator colors,
- no neon dashboard aesthetic,
- large explanatory axis labels,
- direct labels instead of a giant legend,
- no dense badge wall,
- no internal project codenames.

---

# 16. Hero figure specification

## 16.1 Main plot

**Title:** `How many bytes are needed to preserve learning?`

### X-axis

- complete dataset-specific artifact bytes,
- logarithmic scale,
- labeled at 512 B, 1 KiB, 4 KiB, 10 KiB, 64 KiB, 1 MiB, 64 MiB.

### Y-axis

- qualified weighted task coverage,
- 0% to 100%.

### Curves

- one step curve per admitted method,
- DOPE highlighted,
- best two competitors clearly visible,
- other methods muted,
- confidence band or interval markers,
- failed/unavailable methods explicitly marked.

### Reference lines

- horizontal primary coverage target, e.g. 90%,
- vertical 10 KiB product budget,
- crossing label with `B*tested`.

## 16.2 Qualification annotation

The figure caption should state:

> A task qualifies only when every required learner family retains at least 90% of real-data gain on untouched real rows, passes the frozen driver-fidelity threshold, and satisfies complete artifact, copy, privacy, and resource gates. Failures remain in the denominator.

## 16.3 Secondary panel

A compact matrix can show, at the selected budget:

- Tree
- Neural
- TabICLv2
- Kumo
- TabPFN-3.5
- TabFM

with:

- median transfer,
- 10th percentile,
- driver fidelity,
- completion rate.

## 16.4 What not to show as the hero

Do not use:

- CatBoost-only retention,
- MFS-v4 score,
- disjoint method cohorts,
- partial baseline fits,
- mean success-only utility,
- a plot where artifact size is only a tiny bubble/annotation,
- a table of seven weighted score components.

---

# 17. Paper redesign

## 17.1 Recommended confirmed-paper title

> **DOPE: How Many Bits Preserve a Table’s Predictive Behavior?**

Alternative:

> **DOPE: Transfer-Preserving Generative Compression for Tabular Data**

Keep `Downstream Objective-Preserving Encoding` as the acronym expansion in the abstract, not as the main conceptual burden.

## 17.2 Abstract structure

The abstract should contain exactly five moves:

1. **Problem:** synthetic data is usually judged by fidelity or one learner, while artifact size is ignored.
2. **Metric:** introduce smallest tested complete bit budget satisfying transfer and driver constraints.
3. **System:** DOPE compiles a table into a bounded sampling program.
4. **Protocol:** held-out real rows, held-out learner families, held-out source families, exact closure.
5. **Result:** one or two confirmed numbers, only after the sealed run.

## 17.3 Contributions

Use four contributions:

1. **Metric:** task-conditioned transfer rate–distortion with exact artifact charging.
2. **System:** a progressive, bounded sampling program compiler.
3. **Benchmark:** held-out model/source-family transfer with modern tabular foundation models.
4. **Evidence:** full frontier against admitted open-source generators and copy/coreset controls.

## 17.4 Main paper structure

1. Introduction
2. Problem formulation
3. Transfer-preserving description length
4. DOPE artifact and compiler
5. Experimental protocol
6. Main frontier results
7. Held-out learner transfer
8. Driver and interaction fidelity
9. Bit allocation and ablations
10. Privacy, copying, and limitations
11. Related work
12. Conclusion

## 17.5 First page

The first page must contain:

- the question,
- the one-sentence answer,
- the hero frontier,
- the main confirmed result.

Do not begin with contract versions, audit tiers, or a long method inventory.

## 17.6 Theory positioning

Use MDL and rate–distortion as framing:

- the artifact is a description conditioned on a fixed decoder,
- distortion is downstream predictive loss,
- qualification prevents tiny useless descriptions,
- exact measured bytes replace a proxy parameter count.

Do not claim an information-theoretic optimum.

## 17.7 Related work categories

Cover:

- synthetic tabular generation,
- train-on-synthetic/test-on-real evaluation,
- dataset distillation and teaching sets,
- coresets,
- generative compression,
- MDL/rate–distortion,
- tabular foundation models,
- synthetic-data privacy,
- feature-attribution stability.

## 17.8 Results tables

Primary table columns:

- method,
- smallest tested qualifying budget,
- coverage at 1 KiB / 10 KiB / 1 MiB,
- payload bytes,
- cold-start bytes,
- worst-family transfer,
- driver fidelity,
- completion rate,
- empirical copy/privacy status.

Do not put a seven-term MFS scalar in the first table.

## 17.9 Required ablations

- no target-relation layer,
- no dependency layer,
- no progressive refinement,
- no bit-aware search,
- CatBoost-only search,
- multi-auditor search,
- no driver regularization,
- no tail correction,
- different quantization,
- payload versus cold-start charge,
- population versus teaching mode,
- held-out auditor results by family.

## 17.10 Supplement

Move to the supplement:

- full source locks,
- dependency hashes,
- per-dataset tables,
- all failure dispositions,
- statistical details,
- adapter parity evidence,
- privacy attacks,
- extended figures,
- exact commands.

Operational campaign diaries should not be typeset as scientific narrative.

---

# 18. Open-source quality bar

## 18.1 Required top-level files

Add or clean:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `GOVERNANCE.md`
- `ROADMAP.md`
- issue templates
- pull-request template
- release checklist

## 18.2 Reproducible environments

Publish:

- lockfiles,
- container digest,
- GPU/CPU lanes,
- clean-machine smoke test,
- offline model cache instructions,
- benchmark resource estimates,
- exact checkpoint acquisition procedure where redistribution is not allowed.

## 18.3 CI lanes

### Fast

- formatting,
- lint,
- unit tests,
- schema tests,
- golden metric fixtures,
- artifact determinism,
- Quick-12 smoke.

### Full CPU

- Rust/Python conformance,
- sampler closure,
- baseline adapters,
- figure regeneration,
- paper build.

### Qualified GPU

- selected model adapter smoke,
- native parity fixtures,
- deterministic checkpoint loading,
- resource observation.

### Release

- signed binaries,
- SBOM,
- artifact hashes,
- paper/result regeneration,
- claim lint,
- README number lint.

## 18.4 Contributor experience

A new contributor should be able to:

1. install,
2. run one example,
3. fit an artifact,
4. sample it,
5. inspect bytes,
6. reproduce one Quick-12 benchmark cell,

without reading internal custody documents.

## 18.5 Result provenance

Every figure point links to:

- method identity,
- dataset identity,
- artifact hash,
- evaluator identity,
- receipt bundle,
- failure state,
- source code revision.

---

# 19. Proposed schemas

## 19.1 Artifact manifest

```json
{
  "format": "dope-artifact-manifest",
  "version": 1,
  "artifact_sha256": "...",
  "mode": "population_sampler",
  "decoder": {
    "name": "dope-runtime",
    "version": "1.0.0",
    "sha256": "...",
    "shared": true,
    "bytes": 0
  },
  "entries": [
    {
      "role": "model",
      "path": "model.bin",
      "sha256": "...",
      "bytes": 1895,
      "dataset_dependent": true
    }
  ],
  "payload_bytes": 1895,
  "cold_start_bytes": 5030000,
  "schema_bits": 240,
  "transform_bits": 128,
  "model_bits": 14400,
  "container_bits": 392,
  "sampling_replay_sha256": "...",
  "external_reads_allowed": false
}
```

## 19.2 Auditor manifest

```json
{
  "format": "dope-auditor-manifest",
  "version": 1,
  "name": "kumo-tabular-large",
  "source_revision": "...",
  "checkpoint_sha256": "...",
  "dependency_closure_sha256": "...",
  "code_license": "...",
  "weight_license": "...",
  "task_support": ["regression", "classification"],
  "mode": "frozen_context",
  "context_policy": "all_up_to_cap",
  "context_cap": 8192,
  "query_policy": "batch",
  "preprocessing_sha256": "...",
  "native_parity_sha256": "...",
  "resource_policy_sha256": "...",
  "admission": "headline_admitted"
}
```

## 19.3 Evaluation receipt

```json
{
  "format": "dope-transfer-evaluation",
  "version": 1,
  "dataset_id": "...",
  "source_family": "...",
  "target_id": "...",
  "split_sha256": "...",
  "artifact_sha256": "...",
  "artifact_payload_bytes": 1895,
  "auditor_sha256": "...",
  "fit_seed": 11,
  "sample_seed": 101,
  "learner_seed": 7,
  "requested_rows": 4000,
  "consumed_rows": 4000,
  "loss_space": "original_mse",
  "real_loss": 0.72,
  "synthetic_loss": 0.80,
  "null_loss": 1.80,
  "transfer_retention": 0.9259,
  "driver": {
    "rank": 0.91,
    "mass": 0.88,
    "direction": 0.86,
    "interaction": 0.82,
    "minimum": 0.82
  },
  "copy_gate": true,
  "privacy_gate": null,
  "resource_gate": true,
  "status": "measured"
}
```

## 19.4 Frontier report

```json
{
  "format": "dope-qualified-frontier",
  "version": 1,
  "quality_contract_sha256": "...",
  "method_sha256": "...",
  "budgets_bytes": [512, 1024, 2048, 4096, 10240, 65536, 1048576],
  "points": [
    {"budget_bytes": 512, "coverage": 0.18},
    {"budget_bytes": 1024, "coverage": 0.42},
    {"budget_bytes": 2048, "coverage": 0.71},
    {"budget_bytes": 4096, "coverage": 0.91}
  ],
  "b_star_tested": {
    "coverage_target": 0.90,
    "state": "measured",
    "budget_bytes": 4096,
    "minimum_lies_in": {
      "lower_exclusive": 2048,
      "upper_inclusive": 4096
    }
  }
}
```

---

# 20. PR and workstream plan

## PR 00 — Freeze and archive historical state

Actions:

- tag current MFS history,
- produce immutable historical bundle,
- mark current results as historical validation,
- stop changing old score semantics,
- add an overhaul tracking issue.

Acceptance:

- every historical result remains reproducible by hash,
- no public page implies v4 is measured.

## PR 01 — Public narrative reset

Actions:

- replace README first screen,
- remove MFS-v4 weights from headline,
- expose the one-sentence thesis,
- link vNext status.

Acceptance:

- a new reader can explain DOPE in 30 seconds,
- no stale active-version contradiction appears above the fold.

## PR 02 — Metric contract

Actions:

- freeze transfer equation,
- freeze quality thresholds,
- freeze budget grid,
- freeze task weights,
- freeze required and held-out learners,
- freeze states.

Acceptance:

- complete schema,
- adversarial review,
- no null required field,
- `protocol_frozen=true` only after review.

## PR 03 — Canonical artifact accounting

Actions:

- implement canonical container,
- payload/cold-start views,
- syscall/file-open closure,
- adversarial accounting tests.

Acceptance:

- packing and denominator attacks do not improve size,
- hidden file/checkpoint reads fail.

## PR 04 — Canonical transfer metrics

Actions:

- paired losses,
- normalized retention,
- informative-task gate,
- cluster bootstrap,
- simultaneous qualification.

Acceptance:

- Rust/Python golden parity,
- original-unit fixtures,
- constant-target refusal,
- failure denominator tests.

## PR 05 — Driver fidelity

Actions:

- feature groups,
- conditional perturbation,
- rank/mass/direction/interaction metrics,
- confidence bounds,
- consensus diagnostics.

Acceptance:

- correlated-feature fixtures,
- sign reversal fixture,
- interaction-loss fixture,
- no raw cross-model SHAP comparison.

## PR 06 — Auditor qualification

Actions:

- qualify classical, neural, and development foundation models,
- qualify held-out models without exposing outcomes to compiler developers.

Acceptance:

- native parity,
- exact checkpoint locks,
- clean-machine offline run,
- resource receipt,
- license decision.

## PR 07 — Generator baseline qualification

Actions:

- compression controls,
- classical generators,
- established neural generators,
- frontier open-source generators.

Acceptance:

- standalone export/reload,
- complete artifact charge,
- author-default/native-selected distinction,
- no incomplete cohort in headline.

## PR 08 — Dataset manifests and sealed splits

Actions:

- Quick-12,
- Open-60,
- Sealed-24,
- rights and contamination review.

Acceptance:

- immutable manifests,
- group/time split tests,
- target leakage review,
- source-family boundaries.

## PR 09 — Pareto compiler

Actions:

- exact bit objective,
- progressive stream,
- multi-auditor development,
- worst-family selection,
- gain-per-bit mutation.

Acceptance:

- produces valid artifacts at each budget,
- no sealed-auditor access,
- deterministic replay,
- Pareto archive export.

## PR 10 — Full benchmark campaign

Actions:

- freeze matrix,
- run all cells,
- preserve failures,
- compute frontiers and uncertainty.

Acceptance:

- at least 95% intended execution completion or a documented frozen failure disposition for every cell,
- no pending cells in headline,
- no best-of-k test selection.

## PR 11 — Figure and report generator

Actions:

- generate hero plot,
- generate model heatmap,
- generate tables,
- link every point to receipts.

Acceptance:

- README and paper use identical result source,
- CI catches hand-edited values.

## PR 12 — Paper rewrite and release

Actions:

- new title/abstract,
- first-page frontier,
- full limitations,
- release artifact,
- citation metadata.

Acceptance:

- every headline claim traceable,
- sealed results only,
- no MFS headline,
- paper builds from clean environment.

---

# 21. Stop-ship conditions

Do not publish a new SOTA or transferable-knowledge claim while any of the following is true:

1. The README says v4 while the active evidence is v2/v3.
2. Kumo, TabPFN-3.5, TabFM, or another named headline model is unqualified.
3. The compiler has seen every learner later called “held out.”
4. The source-family confirmation set was used for profile design.
5. Artifact closure excludes a method-specific checkpoint or sidecar.
6. `encoded_numbers` affects ranking.
7. Incomplete comparator cohorts appear as losses.
8. The final figure mixes different dataset cohorts without a prominent disclosure.
9. Failures are removed from denominators.
10. A best seed is selected using final outcomes.
11. A non-significant difference is described as equivalence.
12. Feature importance is compared by raw cross-model SHAP magnitude.
13. The paper says “privacy-preserving” without a formal or clearly scoped empirical claim.
14. The paper says “minimum” without “smallest tested” or a genuine proof.
15. The paper says “knowledge” without held-out learner and source-family evidence.
16. The first-page figure remains CatBoost-only.
17. The benchmark cannot be regenerated from committed machine-readable results.

---

# 22. Acceptance criteria for v1.0

## Scientific

- One frozen transfer equation.
- One frozen driver qualification rule.
- One frozen byte policy.
- At least three distinct learner families.
- At least two held-out learner architectures.
- A sealed source-family confirmation set.
- All failures in denominators.
- Simultaneous one-sided uncertainty.
- No headline score based on arbitrary weighted components.
- Full payload and cold-start reporting.

## Metric integrity

- Packing values into one scalar does not improve charged size.
- Duplicating source rows does not improve intrinsic artifact bytes.
- Padding cannot improve a result.
- A hidden checkpoint makes the artifact incomplete.
- A tiny useless artifact receives zero qualification.
- A large accurate artifact may qualify but appears at its true byte budget.
- No untested interpolation is presented as measured.

## Product

- One `dope` CLI.
- Three-command quick start.
- Clean CSV/Parquet input.
- Named target.
- Prebuilt installation path.
- `inspect` and `verify`.
- Deterministic sample replay.
- Clear numeric-regression scope if that remains the capability.

## Benchmark

- Compression/copy controls.
- Classical generators.
- Strong neural generators.
- Current open-source frontier generators.
- Modern tabular foundation-model auditors.
- Complete method-by-task matrix or explicit frozen failure dispositions.
- Same cohorts in the hero plot.
- Result provenance per point.

## Paper and README

- Same hero data source.
- Same terminology.
- Same limitations.
- First page explains the project in under one minute.
- No stale MFS language in active narrative.
- No hand-typed headline numbers.
- Reproducible paper build.

---

# 23. Risks and mitigations

## Risk: Foundation-model prior dominates the synthetic context

Mitigation:

- no-context control,
- predictor-only control,
- real-context baseline,
- normalized gain retention,
- held-out models,
- context-size sensitivity.

## Risk: Tiny payload hides a giant decoder

Mitigation:

- payload and cold-start bytes,
- fixed decoder identity,
- amortized view only with declared deployment count.

## Risk: Compiler overfits development auditors

Mitigation:

- rotate development families,
- optimize worst-family performance,
- seal Kumo/TabPFN/TabFM or another independent set,
- use source-family confirmation.

## Risk: Driver fidelity is unstable

Mitigation:

- grouped conditional perturbations,
- within-model comparison,
- confidence bounds,
- real-auditor disagreement baseline,
- multiple driver components.

## Risk: Budget frontier is expensive

Mitigation:

- progressive artifact stream,
- multi-fidelity search,
- Quick-12,
- coarse-to-fine budget grid,
- reuse fit state only when the protocol permits and identity records it.

## Risk: Benchmark scope explodes

Mitigation:

- first confirmed paper remains numeric regression,
- classification and mixed-type support are a later capability claim,
- distinguish core and research-only model lanes.

## Risk: Open-source baseline failures become misleading wins

Mitigation:

- pre-admission qualification,
- explicit `unavailable`,
- no superiority claim against an unexecuted method,
- frozen resource policy.

## Risk: “Synthetic data” means different things

Mitigation:

- explicit artifact mode,
- separate population and teaching tracks,
- claim ledger per capability.

---

# 24. Recommended final public narrative

## The idea

A dataset is valuable because it changes what a learner can do.

DOPE asks:

> **How little executable information can preserve that change?**

## The experiment

1. Compile real training data into a tiny sampler.
2. Generate synthetic training/context rows.
3. Give only those rows to modern learners.
4. Test on untouched real data.
5. Compare against the same learners using real training rows and null controls.
6. Count every required byte.
7. Require the same important drivers.
8. Repeat across unseen model and source families.

## The result format

Not:

> MFS-v4 = 83.7

Instead:

> At 4 KiB, DOPE qualifies 91% of frozen tasks.  
> The smallest tested budget reaching 90% coverage is 4 KiB.  
> Qualification requires ≥90% transfer, ≥80% driver fidelity, complete artifact closure, and all required learner families.

That is understandable, falsifiable, and difficult to game.

---

# 25. Final recommendation

Do not build MFS-v5.

Promote the best work already present in vNext:

- exact bytes as a frontier,
- original-unit paired losses,
- held-out learner/source families,
- all failures in the denominator,
- refusal and censoring states,
- immutable evidence.

Then sharpen it into one world-class idea:

> **Transfer-Preserving Description Length: the smallest tested complete artifact that still teaches modern models to work on unseen real data.**

The project’s public hierarchy should become:

1. **Bytes to qualified transfer**
2. **Coverage-versus-bytes frontier**
3. **Held-out model transfer**
4. **Driver fidelity**
5. **Artifact and privacy gates**
6. **Detailed reproducibility evidence**

Today the hierarchy is almost reversed. Fixing that inversion will make the project feel radically cleaner without sacrificing any of its rigor.

---

# Appendix A — Immediate edits before any new benchmark run

1. Add a prominent historical-evidence banner to the README.
2. Remove MFS-v4 weights from the opening narrative.
3. Replace the opening plot with a clearly labeled historical CatBoost plot or temporarily remove it.
4. Link `docs/vnext/CLAIMS.md` from the README limitations.
5. Add a one-paragraph explanation that v2 is active, v3/v4 are historical/draft, and vNext is unmeasured.
6. Remove internal/private host names from public-facing benchmark docs.
7. Mark Kumo/Mitra/TabICL locks as planned/unqualified in every public table.
8. Stop calling the current compactness term MDL.
9. Stop using `encoded_numbers` in new result copy.
10. Add the one-line thesis and three-command target CLI even before implementation, clearly marked as the vNext interface.
11. Create the historical tag before moving files.
12. Open one master issue containing the PR plan in this document.

# Appendix B — Primary current external model targets

As of the evidence cutoff, the model-qualification queue should begin with official sources for:

- NVIDIA Structured Data Models / Kumo Tabular
- TabICLv2
- TabPFN-3.5
- Google TabFM
- Mitra-v2
- Xiaomi-TabLDM

The generator-qualification queue should begin with official sources for:

- CDTD
- GATD
- TabRep
- TabCascade
- TabularARGN
- TabFORGE, only after exact code/weight release is verified

All names remain candidates until the repository records exact source, checkpoint, dependency, license, native-parity, and clean-machine evidence.

# Appendix C — Suggested paper headline tests

The final paper should be able to answer these in one table:

1. How many bytes does DOPE require to qualify 90% of tasks?
2. How does that compare with compressed raw rows?
3. How does that compare with the best admitted open-source generator?
4. Does the conclusion hold for Kumo?
5. Does it hold for TabPFN-3.5?
6. Does it hold for TabFM?
7. Does it hold for a model family never used in compiler search?
8. Do the top real-world drivers agree?
9. What happens under temporal or grouped shift?
10. How many datasets fail, and why?
11. What changes when cold-start bytes are charged?
12. Which artifact components produce the most transfer gain per bit?

If the project can answer those cleanly, it will no longer feel like accumulated research debris. It will feel like one sharp scientific instrument.
