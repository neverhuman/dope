# DOPE technical review and publication upgrade

Reviewed against main `aac3fdadf93fdbd07811a0e16e53349b4d8eff40`.
This document separates implemented engineering from proposed scientific gains.
No new training campaign, protected dataset access, sealed evaluation or GPU run
is authorized or claimed by this source change.

## 1. The project in one sentence

**Compile a real fitting table into the smallest tested sampling program that
preserves what independent models learn from it.**

The valuable object is the executable carrier of predictive structure. The
research question is how far its byte rate can fall before independent learners
lose predictive gain, driver agreement or required support. Scaling a CSV is an
input operation, not the breakthrough. Generating a million rows is a sampling
operation, not a million observations of new information.

The compiler's structural prior, its byte-aware representation and the
multi-learner test together make this an interesting research program. The
repository already has substantial mechanisms and evidence discipline; the
public narrative obscures them with repeated panels, historical metrics and
overlapping entry points. Preserve those historical artifacts and readers;
simplify the public path around one metric and clearly named profiles.

## 2. What this PR changes

| Area | Implementation | Result and limitation |
| --- | --- | --- |
| Entry point | `README.md` | Plain value proposition, one hero, native quick start, existing pinned public-data walkthrough, expandable detailed evidence. |
| Hero figure | `tools/claims/render.py`, generated hero assets and manifest | Explicit payload axis beside matched transfer. Existing authenticated numbers, intervals and cohort; no fabricated frontier. |
| Active manuscript | `paper/main.tex`, `paper/method.tex`, `paper/supplement.tex` | Shorter first page, compiler factorization, ridge moment bound with proof, corrected evidence scope and numerical-profile disclosure. |
| Preprocessing | `docs/preprocessing.md` | Actual numeric/mixed/prepared boundaries and precise named robust-transform experiments. No silent default transformation change. |
| Native numerical seam | `rust/neural/deployment.rs`, shared native postprocessing and seed functions | Training-consistent GELU helper, shared random streams and marginal/support semantics; historical default unchanged. |
| Deployment | `deploy/` | Separate locked Rust SafeTensors carrier, verified typed TVAE tensors, CPU fallback and batched optional Candle CUDA head execution. No Python/libtorch inference. CUDA needs hardware qualification. |
| Verification | Deployment CPU lane and owner/test routing | Nonzero parity and corruption fixtures, explicit bounds and graph checks. Scientific campaign remains separate. |

The SafeTensors carrier is deliberately a conservative reference export: native
DPK plus typed decoder tensors. Duplicate state and container overhead count.
This creates a working interchange and execution boundary; it is not presented
as an artifact compression gain.

The inference-only `deploy/native` package compiles the existing codec, model,
neural and sampler modules through `rust/runtime_lib.rs`. This avoids importing
the full compiler/campaign/service crate into a recipient runtime. It shares
the implementation rather than copying a second decoder into deployment.

## 3. What existing results do and do not show

The source of public numbers remains the hash-pinned claims renderer, not this
document. The historical foundation-model hero and classical table represent
different cohorts and must remain separate. The classical TabSyn point estimate
is stronger on CatBoost; its full-panel paired interval includes zero. That is
not evidence of equivalence. Smaller Forest-Flow, TVAE and CTGAN cohorts cannot
be pooled into a full-panel ranking. The correlation proxy in the old foundation
panel is not grouped permutation driver preservation.

The existing public evidence supports a compactness/transfer tradeoff in
selection-exposed validation. It cannot establish global first place in
information compression, qualified TDL, privacy-preserving external use, or
foundation-model generalization to genuinely unseen sources. TDL is operational
and task-specific; it is neither Shannon entropy nor Kolmogorov complexity.

The correct route to a strong first-place claim is a frozen comparison of
**qualified coverage at equal complete byte budgets**, with simultaneous
uncertainty, matched recipe/compute tracks and all scheduled failures retained.
If no method qualifies, the public result is the observed frontier and the
failure analysis, not a winner manufactured from a weighted score.

## 4. Current mathematical structure

The symbolic implementation can be understood as

\[
 X\sim Q_\theta(X),\qquad
 Y=g_\beta(X)+\sigma_\gamma(X)\epsilon,\quad\epsilon\sim R_\eta.
\]

`rust/compiler/` fits a bounded tournament of marginals, dependence operators,
target models and noise. `rust/codec/` serializes the actual program. Native
sampling decodes only retained state. Raw-unit carriers additionally retain
schema and fitted inverses, which must be counted in full. Neural joint
candidates use separate architecture and tensor paths. The native mixed
development profile is bounded and uses one-hot coordinates internally.

This separation is useful: predictive structure can survive in a short program
even when exact row values and a generic density model are not preserved. But
the fit-side `stored_target_fit`, structural/correlation proxies and historical
compactness terms are not the downstream objective itself. A generator can
preserve a good predictor and still destroy retraining because it loses feature
support, conditional variance, rare categories or interactions.

### 4.1 A useful exact special case

On one shared frozen feature map, define
`Sigma_P = E_P[xx^T]`, `c_P = E_P[xy]`, and
`w_P = (Sigma_P + lambda I)^-1 c_P`, with `lambda > 0` and finite moments.
For the synthetic distribution Q,

\[
 \|w_Q-w_P\|_2\leq
 \lambda^{-1}\bigl(\|c_Q-c_P\|_2+
 \|\Sigma_Q-\Sigma_P\|_2\|w_P\|_2\bigr).
\]

Subtract the two ridge normal equations and use
`||(Sigma_Q + lambda I)^-1|| <= 1/lambda`. This is a standard perturbation
argument, not a claimed new theorem. It assumes every represented coefficient
is regularized; an unregularized intercept needs separate treatment.
The bound explains a concrete diagnostic target: preserve `X'X` and `X'y`,
including their scaling and semantic grouping, rather than optimize only
feature-wise resemblance. Under a fixed query second moment `Sigma_*`, mean
squared prediction disagreement is at most
`||Sigma_*|| * ||w_Q-w_P||^2`. Query excess risk also has a cross term with the
real model's residual and cannot be equated with that discrepancy bound.

For finite synthetic draws, distinguish generator moment error from Monte Carlo
error. Measure both by comparing analytic moments where tractable, many draws
as a diagnostic and the actual declared `n`/`4n` learner contexts. Independent
rows from a fitted sampler do not remove estimation error in its learned state.

### 4.2 Numerically stable testing of the transfer threshold

The ratio
`T = (L0 - LS)/(L0 - LR)` becomes unstable when the real gain is small.
The candidate successor already addresses this with paired harm. For informative
real references and a fixed threshold tau,

\[
 T\geq\tau\iff z=L_S-(1-\tau)L_0-\tau L_R\leq 0.
\]

Use paired per-query losses in original target units and a simultaneous
one-sided upper bound on `z` for the candidate protocol. Keep the informative
real-gain gate; algebra does not make a near-zero denominator informative.
Pair physical units and resample source families before datasets and seeds.
The candidate calibration/multiplicity fields remain unresolved; this PR does
not freeze them or reinterpret historical ratio receipts.

Before confirmation, calibrate the selected multiplicity procedure under known
nulls and marginal alternatives, including correlated auditors and a search
across budgets. A familywise bootstrap max statistic or conservative correction
is a proposal, not a label to add after looking at the results. Keep all required
families, budgets and driver components in the declared testing family. Confidence
intervals on medians and paired effects answer different questions.

### 4.3 Conditional structure is the likely scientific bottleneck

Dependency models that capture marginal correlations can miss XOR-like pure
interactions, multimodal conditional targets and missingness that depends on a
predictive feature. The current compiler already contains interaction support,
OOF residual alternatives and adaptive byte work; the first task is to ablate
them at equal **complete carrier bytes**, not implement a second competing fitter.

For drivers, normalize stable real/synthetic importance against a real-vs-real
ceiling, group original features and report total importance strength. Identical
normalized rankings of two nearly inactive models should not constitute success.
Permutation importance is observational and can be misleading under correlated
features; add conditional/grouped sensitivity. It does not establish causality.

## 5. A concrete numerical defect and its safe correction

`rust/neural/part_01.rs::gelu` uses `2/sqrt(pi)` in the tanh approximation.
The PyTorch training paths in `rust/neural_train/gpu/` and
`rust/neural_train/gpu_diffusion/` call `gelu("tanh")`, whose conventional
coefficient is `sqrt(2/pi)` (see the official reference below). This is a genuine
formula mismatch. It affects nonlinear forward values even before quantization.
It is not yet evidence that this explains any measured transfer deficit.

The PR keeps historical sampling semantics and adds an explicit `tanh-v2`
deployment profile. CPU scalar golden values and nonzero tensor-head comparisons
make the difference observable. A trained checkpoint must be sampled under both
profiles, with the same seeds, before assigning any downstream benefit.
The old GPU parity test compares the historical native activation to the training
formula; retain its actual result and investigate it rather than weakening its
tolerance. Future transformer/diffusion activation migration needs a separately
versioned export path, not a global replacement of `gelu`.

## 6. Highest-value research changes, in order

| Priority | Comparison | Why it could help | Required falsification |
| --- | --- | --- | --- |
| P0 | Historical vs training-consistent activation, same weights | Removes a known train/infer mismatch | Nonzero scalar/CPU/CUDA forward parity; paired transfer can still show no benefit. |
| P1 | Complete-byte adaptive allocation vs frozen allocation | Spend state on target-sensitive moments, interactions and tails | Compare actual encoded bytes, not term tariffs; same selection budget and held-out sources. |
| P1 | Existing OOF residual profiles vs Gaussian/no-noise controls | Recover uncertainty without interpolation-induced collapse | Original-unit loss, conditional variance, tail loss, support and near-copy tests. |
| P1 | Frozen affine vs robust/rank/hybrid maps | Prevent outliers consuming most quantizer resolution | Equal total bytes, rare/tail cohorts and out-of-support cases; no query-fitted map. |
| P2 | Mixed conditional profile vs independent nominal/mask model | Recover important dependence while preserving category semantics | Rare-level and missingness-driver tests, with all dummy columns grouped. |
| P2 | Moment/gradient-aware selection vs stored-target proxy | Align compact state with retraining behavior | Held-out learner family; never select on the confirmation TFM. |
| P2 | Source-aware splits and external observational sources | Test whether gains survive beyond a reused benchmark family | Freeze source families before selecting profiles; disclose TFM pretraining uncertainty. |

### 6.1 Rate allocation with a useful derivation

Suppose a local distortion approximation for a quantized parameter is
`D_i ~= h_i Delta_i^2 / 12`, with positive sensitivity `h_i` and a uniform
high-resolution quantization error assumption. For range `R_i` and `b_i` bits,
`Delta_i ~= R_i 2^-b_i`. Minimizing the sum under a fixed bit budget gives a
water-filling-like allocation `b_i = const + 0.5 log2(h_i R_i^2)` for active
coordinates. This is only a local approximation: the real codec has headers,
nonuniform supports, discrete bits, shared scales and variable-length sections.

Use it to propose offers, then **serialize every offer**. Rank development
improvement by measured `Delta loss / Delta complete bytes`, preserving the
actual nondominated set. Charge failed fits, selected indices, codebooks and
schema. Sensitivity can come from fit-side cross-validation moment error or
multi-learner gradient sketches, with their compiler-time cost recorded. Any
sketch retained at sampling time counts as payload. Do not use protected query
data to determine bits.

### 6.2 Broader predictive sufficient statistics

For regularized linear learners, preserve covariance and cross moments. For
additive bases, preserve the corresponding basis Gram matrix and target moments.
For shallow trees, compare conditional target moments in a frozen fit-only
partition family. For neural learners, test gradient moments at multiple frozen
initializations or training checkpoints. Multi-auditor gradient matching is a
research hypothesis, and model-specific distillation can overfit its teacher.
Keep one learner family entirely out of selection.

An ambitious next step is a bilevel compiler: the inner loop trains an auditor
on generated rows and the outer loop improves a constrained artifact. Start with
ridge's closed-form inner solution to debug the objective. Move to stochastic
training only after gradient checks, compute accounting and held-out-family
generalization work. An end-to-end differentiable loss does not remove the
discrete codec or make the byte constraint differentiable.

## 7. HPC changes worth implementing and measuring

**Batch useful work before adding hardware.** The original sampler runs small
linear operations per row; moving those individual operations to CUDA can be
slower than CPU execution. The new runtime uploads TVAE layer tensors once,
computes heads for batches, and reuses native row seeds, permutations and
marginal decoding. I8 weights are expanded for F32 matmul; this is not an int8
GPU GEMM implementation or an acceleration claim.

| Hot path | Recommended next implementation | Measurement |
| --- | --- | --- |
| Repeated layer setup | Prepared immutable per-artifact executor, cached transposes/dequantized weights | Load latency vs steady-state rows/s and peak memory; count transient expansion. |
| Many tiny TVAE forwards | Batched row-major matrix products, pinned batch-size crossover | 1/32/256/4096 rows, fixed inputs/seeds, CPU vs GPU; report dispatch/copy costs. |
| Marginal lookup | Batch contiguous feature columns and cache knot-search data | Profile before optimizing; exact endpoint, discrete and missingness behavior. |
| Dependency fitting | Reuse fit-only ranks/covariances across candidate quantizers | End-to-end compile time, identical encoded offers and selected-artifact hashes. |
| Driver tests | Reuse each fitted auditor; batch baseline/permuted query predictions | Same perturbation plan and predictions, actual fit/predict counts and memory. |
| Benchmark scheduling | Dataset/fit-seed cells as independent tasks; one bounded queue per device | Account failed/timeout work, avoid nested Rayon/BLAS oversubscription. |

On CPU, use persistent work buffers and a bounded thread pool. Avoid copying
large datasets into every offer. Keep portable release builds distinct from
`target-cpu=native` experiments. Test whether zstd savings exceed framing and
decompression costs for each size regime. Prefer quantized native symbolic state
for tiny artifacts; tensor-container overhead can dominate kilobyte programs.

On GPU, allocate reusable device tensors and retain model state across batches;
profile host/device transfer and synchronization explicitly. Expand support to
diffusion and transformer graphs only behind numerical conformance tests. More
GPU utilization is not proof of lower compilation or sampling cost. No OpenMP,
Fortran or C++ rewrite is proposed for deployment; Rust/Candle is the inference
boundary, with normal platform GPU libraries outside the application source.

## 8. Benchmark design needed for a publication claim

Keep three distinct tracks: author-default quality, equal selection compute,
and equal complete artifact budgets. Give strong methods appropriate training
schedules; a deliberately shortened schedule needs its own label. Include
TabSyn and TabDDPM, investigate TabDiff and continuous mixed-type diffusion,
and retain copula, dependence-tree, forest, VAE/GAN and simple baselines. Rights,
adapter parity and actual available hardware can make a method unavailable;
that remains visible rather than quietly removing it.

The row/subset frontier and zstd-compressed raw table are essential size controls.
For a subset, charge indices/values, schema and decoder dependencies. Real-row
resampling is a row-count diagnostic, not a generator or universal upper bound.
Predictor-only and independent-marginals controls test whether the compact
generator preserves more than a teacher's outputs or feature histograms.

Use predominantly observational sources. Separate simulated mechanism tests,
known synthetic families and shifted tasks. Group repeated physical entities
before splitting; report unique independent units and source families rather
than treating rows or seeds as independent tasks. Disclose unknown TFM
pretraining overlap and pin actual model revisions, context limits and licenses.

At each budget, publish all terminal cell dispositions, coverage over the fixed
denominator, original-unit losses, unclipped transfer, driver components,
real-vs-real ceilings, copy/near-copy checks, payload and closure bytes, compile
and sample costs. No complete-cell-only denominator. No linearly interpolated
TDL crossing. No best-seed headline. Keep method selection on development data;
freeze the protocol and analysis before the one-time confirmation run.

## 9. Paper layout and scientific readiness

The active TDL manuscript is `paper/main.tex`. The frozen historical paper in
`docs/whitepaper/` remains independently reproducible and linked. They should
not be silently overwritten with a new narrative or made to share incompatible
metrics. The active source now explains the method and an exact mathematical
special case; it remains a preprint with explicit unmeasured confirmation.

| Publication component | Upgrade in this PR | Evidence still needed |
| --- | --- | --- |
| Page one | Shorter title/abstract, readable matched figure, payload axis, compact caption | A qualified prospective headline when the frozen run exists. |
| Method | Factorization, codec/selection distinction, preprocessing and runtime boundaries | Full algorithm/pseudocode linked to the frozen winning profile; complexity and time study. |
| Theory | Ridge perturbation bound and assumptions with proof | A stronger result only if derived; no claim that this elementary bound proves all-model transfer. |
| Results | Separate historical cohorts and explicit strongest comparator | Complete qualified budget curves and simultaneous paired uncertainty on observational confirmation. |
| Ablations | Ordered, falsifiable experiment plan | Measured activation, allocation, residual, preprocessing and mixed-structure ablations. |
| Privacy | Existing release gates preserved | Actual attacks, power/threshold rationale and separate mechanism/accounting if formal DP is proposed. |
| Reproducibility | Generated figures, typed runtime and fixed diagnostics | Independent reproduction, immutable source/model/rights pins and complete cost receipts. |
| References | Primary sources below; existing bibliography preserved | Add verified BibTeX entries and precise novelty comparison for final included methods. |

For the final submission, use a venue-specific build target over the same source
and generated inputs, with anonymization where required. Keep a single readable
type family, normal mathematical typography, short captions and direct axis
labels. Use vector PDF figures, grayscale-distinguishable marks and at least
nine-point text at final print scale. Do not shrink a giant table to fit; split
the question into transfer/bytes, matched paired effects and coverage. Put per-task
failure tables in the supplement, with the failure denominator in the main text.

The decisive blocker to a top-tier submission is evidence, not typography.
No formatting change can convert selection-exposed validation, an unqualified
auditor or an unmeasured driver score into a confirmed result.

## 10. Finite compute request for the PR discussion

Request a **development-only** allocation, with no sealed test access:

1. One provisioned CUDA host, at most **two GPU-hours** and **two CPU-hours** for
   numerical qualification and a four-task timing pilot. Prefer an existing
   project GPU with at least 16 GiB VRAM; record exact device, driver, toolkit,
   compiler and binary/weight hashes. Stop at the cap and retain incomplete work.
2. Run nonzero tensor forward comparisons and native-vs-tensor sampling for the
   named TVAE profiles, both activations and the batch-size grid. Preserve actual
   mismatch counts, absolute/relative errors, masks, integer decisions, latency,
   RSS/VRAM and dependency inventory. Do not turn a CPU fixture into GPU evidence.
3. On four already admitted observational development tasks, compare the old and
   corrected activation with **the same frozen weights and seeds** first. Then
   time a small, explicitly capped allocation/residual/preprocessing screening
   cell. Measure actual fits, predictions and total stage costs before asking
   for a larger benchmark allocation.
4. Propose the next finite cap from those measurements and the existing owner
   campaign protocol. This request grants no retries, override of custody,
   checkpoint download, new rights, protected worker admission or sealed opening.

No job is launched by this document or this PR. A later observational screening
plan can use 16 tasks, six named profiles, three fit seeds and two draw sizes;
that is 288 compiler fits and 576 artifact/size combinations before sample or
auditor seeds. Multiplying by three sample seeds and three auditors yields
5,184 learner arms **per tested byte budget**, before real/null references and
driver perturbations. This arithmetic is a planning count, not measured cost.
Charge reference reuse only where its exact functional identity and protocol
allow it. Do not submit a full ten-budget sweep until the timing pilot supports it.

## 11. Code organization and promotion path

Keep `rust/` as product truth. Preserve facade and historical command names,
DPK3.2/v1 readers, frozen contracts, receipts and hash-pinned results. Do not mix
bulk source moves with mathematical changes. `research/` continues to own Python
training/evaluation evidence; `paper/` owns the active manuscript; `tools/claims/`
owns public evidence rendering; `deploy/` owns the additive inference package.

Next increments should be small and separately reviewable: nonduplicating
SafeTensors graph export, raw/mixed carrier export, all-architecture tensor parity,
then measured backend promotion. In each, update source closure hashing, owner
and test maps, package notices, graph/SBOM checks and actual platform smoke tests.
The CPU facade should not silently switch its numerical profile or claim GPU
availability based on a compile-time flag.

Promotion requires signed/tagged source provenance, checksummed binaries and
models, complete platform notices and SBOM, source-free replay and measured
CPU/GPU parity. Keep compact symbolic artifacts separate from runtime closure
bytes. Do not embed a shared model into every dataset artifact to hide its size;
report cold and explicitly amortized costs beside dataset-specific rate.

## Primary references checked during this review

- [PyTorch GELU definition](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.activation.GELU.html): reference formula for the numerical mismatch.
- [Candle official repository](https://github.com/huggingface/candle): Rust tensor execution and feature-specific accelerator support.
- [SafeTensors format implementation](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs): typed data, metadata and bounds; a tensor format does not define a model graph.
- [TabSyn, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html): VAE latent-space diffusion; an important quality/speed comparator.
- [TabDDPM, ICML 2023](https://proceedings.mlr.press/v202/kotelnikov23a.html): heterogeneous tabular diffusion comparator.
- [TabDiff, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html): mixed-type diffusion with feature-specific processes; add only through an admitted implementation.
- [Continuous Diffusion for Mixed-Type Tabular Data, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html): another representation baseline to evaluate rather than assume away.
- [Dataset Distillation, Wang et al.](https://arxiv.org/abs/1811.10959) and [Matching Training Trajectories](https://arxiv.org/abs/2203.11932): related learner-specific compression objectives. DOPE needs a precise comparison on a reusable sampler and held-out learner families, not a claim that data compression itself is new.
