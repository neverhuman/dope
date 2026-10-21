# The math of DOPE: what it computes, why it works, and what to build next

This page is the technical core of the project in one place. Section 1 states the
problem, section 2 the program DOPE compiles, section 3 the metric, section 4 a short
theory of why a target-explicit program transfers, section 5 where the bytes go
today, and sections 6 to 8 the gaps found in the October 2026 review, a ranked
research roadmap, and the tests and compute it needs. Every measured number names
its evidence label (see [claims policy](claims-policy.md)).

## 1. The problem

A table `D = {(x_i, y_i)}` with features `x ∈ R^d` and a declared target `y` is split
into fit rows and untouched evaluation rows. A compiler maps the fit rows and a byte
budget `B` to an artifact `A` with `|A| ≤ B`. A sampler reads `A` alone and draws
synthetic rows. A learner (an *auditor*) trained on those rows is scored on the real
evaluation rows. We want the smallest artifact whose samples teach every auditor
family most of what the real rows teach it:

```math
\min_A |A| \quad\text{s.t.}\quad T_{\mathrm{robust}}(A) \ge \tau_T,\quad D_{\mathrm{robust}}(A) \ge \tau_D,\quad G(A) = \text{pass}.
```

This is an operational rate–distortion problem: bytes are the rate, and lost
predictive gain is the distortion. Its value at `τ_T = 0.9, τ_D = 0.8` on a tested
byte grid is the project's headline metric, **TDL** (section 3).

## 2. The program DOPE compiles

A DOPE artifact is a small typed program (`rust/model/`, encoded by `rust/codec/`).
`dope-kernel inspect` prints it as an S-expression, for example

```
(dpk v=3 task=regression n=4000 p=9 seed=11
  (schema (continuous i=0) ... (ordinal i=8))
  (marginals (quantile_spline i=0) ... (categorical_grid i=8))
  (dependence kind=chow_liu) (target kind=sparse_linear) (noise kind=homoscedastic))
```

It has four parts.

**Marginals.** For each feature `j` on the unit interval: a missing rate `m_j`, and
either a constant, a Bernoulli, an exact grid over at most 64 observed values, or a
monotone quantile spline `F_j^{-1}` with `K ∈ {5, 9, 17, 33, 65}` knots
(`rust/compiler/part_01.rs`, `fit_column`).

**Dependence.** A latent Gaussian vector `z` with a structured correlation, chosen
from seven families: independence; a Chow–Liu tree (maximum spanning tree on absolute
rank correlations, sampled edge by edge as `z_c = ρ z_p + sqrt(1-ρ²) ε`); a triangular
autoregression; a sparse Gaussian copula (Cholesky factor); a POET factor model with
sparse residual edges; a truncated C-vine; and a two-component copula mixture.
Features are then `x_j = F_j^{-1}(Φ(z_j))`, a Gaussian copula with exact marginals.

**Target.** A conditional-mean model `ŷ(x)` from one of six families: sparse linear or
logistic, sparse GAM, GA2M (pairwise interactions), MARS, an oblivious tree, or the
compact neural residual

```math
\hat y(x) = a + \beta^\top x_S + v^\top \tanh(W x_S + b), \qquad |S| \le 12,
```

where `S` holds the features with the largest absolute rank correlation with `y`.
On a GPU build the hidden layer is trained with AdamW for a fixed step budget and the
read-out `(a, β, v)` is refit in closed form; the CPU build uses fixed, hash-derived
hidden weights and the same closed-form read-out (a random-feature model).

**Noise.** `y = clamp(ŷ(x) + σ(x) ε)`, with a homoscedastic `σ`, a log-linear
heteroscedastic `σ(x)` on up to four screened features (clamped to the residual
deciles), or a Bernoulli/isotonic calibration for binary targets.

Every parameter is quantized to `b ∈ {4, 6, 8, 10, 12, 16}` bits. The compiler runs
a bounded tournament (seven dependence families × six target families × knot and
bit ladders, about 108 candidates) and today selects the candidate with the smallest
*normalized gate shortfall* on train-only proxies: stored-target fit, quantized driver
agreement, a joint-fidelity proxy and a membership proxy
(`rust/compiler/candidate_search/part_01.rs`). Sampling reads only the artifact.

## 3. The metric

**Retention.** For auditor `m` with losses on the real evaluation rows,

```math
T_m(A) = \frac{L_{\mathrm{null}} - L_{\mathrm{syn}}(A)}{L_{\mathrm{null}} - L_{\mathrm{real}}},
```

where the null learns nothing from the table. `T = 1` recovers all of the real gain;
`T < 0` is harmful. A cell is informative only when the real gain clears a floor.

**Robust transfer.** Average within an auditor family, then take the worst family:
`T_robust = min_f T_f`. Adding a family can never raise it.

**Driver fidelity.** Grouped permutation importance of a real-trained and a
synthetic-trained instance of the same recipe on the same real rows; mass, rank and
sign agreement, each divided by the agreement two real-trained instances reach.

**TDL.** The charged dataset-specific bytes of the smallest tested budget at which
`LCB_95(T_robust) ≥ τ_T`, `D_robust ≥ τ_D` and every hard gate (copy, cleartext,
control lift, byte ledger) pass. Unreached budgets are reported as unreached. The full
definition is [metric.md](metric.md); the state machine and gates are not repeated here.

## 4. Why a target-explicit program transfers

The propositions below are elementary, but together they explain the measured
pattern and tell us where bytes should go. Let `P` be the real distribution,
`m_P(x) = E_P[y | x]`, and `Q` the synthetic distribution with conditional mean `m_Q`
and covariate law `Q_X`. Write `f_Q` for what an auditor trained on unlimited synthetic
rows converges to.

**Proposition 1 (retention is conditional-mean error under the real covariates).** Assume finite second moments and a positive real gain.
Under squared loss, with the null predicting `E_P[y]` and the real-trained auditor
reaching the Bayes risk,

```math
T(A) = 1 - \frac{\mathbb{E}_{P_X}\big[(f_Q(X) - m_P(X))^2\big]}{\operatorname{Var}_P\big(m_P(X)\big)}.
```

*Proof.* `E_P[(y - f)^2] = E_P[(y - m_P)^2] + E_{P_X}[(m_P - f)^2]` because
`E_P[y - m_P | X] = 0`. Apply it to `f = f_Q`, `f = m_P` and `f = E_P y`, and note
`Var_P(y) - E_P[(y - m_P)^2] = Var_P(m_P(X))`. ∎

**Proposition 1′ (log loss: normalized excess risk).** Assume a Bayes real-trained comparator, positive mutual information, predictions strictly between zero and one, and finite expected log loss.
For a binary target with the base-rate null and `p(x) = P(y = 1 | x)`,

```math
T(A) = 1 - \frac{\mathbb{E}_{P_X}\big[\mathrm{KL}\big(\mathrm{Ber}(p(X)) \,\|\, \mathrm{Ber}(f_Q(X))\big)\big]}{I(X; Y)} .
```

*Proof.* The excess log loss of `f` over `p` is the expected KL divergence, and the
null's excess over `p` is `H(Y) - H(Y | X) = I(X; Y)`. ∎

At the oracle limit, this is one minus normalized excess log loss, with mutual
information as the denominator. It is not the mutual information stored in the
artifact. Finite-sample retention is not a mutual-information estimator (its
denominator is the auditor's own real gain), and TDL makes no such claim; the identity
says what the quantity converges to and why it is the right thing to compress.

Three consequences follow, each visible in the v2 panel (selection-exposed validation;
numbers from `paper/generated/compression-frontier.json`).

* **Only the conditional mean and the covariate support matter.** Nothing in
  Proposition 1 rewards reproducing parts of `P(x, y)` that do not move `m_P` on the
  support of `P_X`. A joint generator that spends megabytes on such parts buys no
  retention. This is the information-bottleneck intuition made operational, and it is
  why DOPE stores `ŷ` explicitly and keeps the feature model compact.

* **Corollary 1 (linear auditors only see the projection).** If the synthetic
  conditional mean is linear in `x`, ordinary least squares on synthetic rows recovers
  its coefficients for any non-degenerate `Q_X`, whatever the dependence. A
  predictor-only sampler (a ridge fit plus independent features) should therefore keep
  nearly all linear retention and lose most of what a flexible auditor gains from
  nonlinearity. Measured on the panel: predictor-only retention is 0.986 under the
  linear auditor and 0.681 under CatBoost. DOPE's nonlinear residual and dependence
  model close that gap (0.913 under CatBoost on the same lineages).

* **Corollary 2 (an importance-weighted support diagnostic).** Suppose P_X is
  absolutely continuous with respect to Q_X, w = dP_X/dQ_X has finite second
  moment, and h is bounded. For independent Q_X draws, the importance-weighted
  mean has variance at most ||h||_infinity² (1 + chi²(P_X || Q_X)) / n_syn.
  This follows by bounding variance by E_Q[(w h)²] and using
  E_Q[w²] = 1 + chi²(P_X || Q_X). It motivates coverage checks; it is not a
  universal rate for unweighted local learners or a guarantee of improved transfer.

**Retention above one.** Proposition 1 bounds `T ≤ 1` only at the population limit.
At finite `n`, labels drawn from a smooth `ŷ` with modest noise can be *easier* to
learn than the real labels, so an MLP trained on `4n` synthetic rows can beat one
trained on `n` real rows (DOPE's MLP median 1.143 at `4n`; resampled real rows reach
1.174, so row count explains part of it). This is a teacher effect, not leakage, and it
is why receipts keep `T` unclipped and never let a value above one offset a failure.

**Proposition 2 (fixed-resolution kernel storage).** The following bound fixes the feature map, category alphabets, architecture and precision ladders; the complete charge additionally includes encoder/dictionary state, which may grow with cardinality. For `d` features,
`k_j` knots, `|E|` dependence parameters, `s ≤ 12` screened inputs, hidden width `h`
and `b`-bit parameters,

```math
|A| \le c_0 + \sum_j \big(c_1 + k_j b/8\big) + |E|\, b/8 + \big(s h + 2h + s + 1\big)\, b/8 + c_{\mathrm{noise}},
```

which is independent of the row count `n`. Bits per original cell therefore fall like
`1/n`. On the UCI Adult table (32,561 rows, 15 columns; local engineering demo,
`examples/quickstart/results/adult-reference.json`) the selected program and its encoder
take 2,550 bytes against a 3.51 MB CSV (1,378× larger) and a 397 KB `gzip -9` archive
(156× larger), while keeping 0.841 to 0.923 of the real-trained gain under log loss.

**TDL as an operational rate–distortion function.** Define
`R(τ) = inf{|A| : T(A) ≥ τ}`. By Proposition 1, `T(A) ≥ τ` is the statement that the
induced regression function lies within `(1 - τ) Var(m_P)` of `m_P` in `L²(P_X)`, plus
enough covariate support for the auditor to find it. `R(τ)` is thus the operational
description cost of a regression function at a prescribed `L²` distortion, the
quantity Kolmogorov's ε-entropy measures for function classes, with the twist that the
code must be a sampler. TDL estimates `R` on a byte grid; the frontier is the whole
curve.

## 5. Where the bytes go today

Measured on the v2 panel (selection-exposed validation, 86 lineages with every auditor
informative) and on local runs:

| Component | Median share or size | Note |
| --- | --- | --- |
| `projection.json` sidecar (unit-interval map, JSON text with column names) | 60% of charged bytes | charged for every method alike; DOPE's model alone is a median of about 830 bytes |
| DPK header | 56 bytes | fixed framing |
| Marginals section | about 70% of a model | 17-knot splines at 8 bits on a 9-feature table: 481 of 683 bytes |
| Dependence, target, noise | the remaining 30% | a Chow–Liu tree on nine features is 35 bytes |
| Quickstart encoder (`encoder.bin`) | 836 bytes on Adult | 249 bytes of transforms and 587 bytes of compressed names and category labels |

The model itself is already small; most of the rate is framing, text sidecars and
uniform marginal resolution. That is good news: it can be cut without touching
retention (section 7, items R3 and R4).

## 6. What the review found

Items 1 to 3 come from the committed five-seed ablation on UCI Adult and UCI Abalone
(`examples/quickstart/results/ablation.md`; local engineering evidence, mean
worst-auditor retention).

1. **No fixed backend is best, and the built-in selector does not see transfer.** The
   tournament ranks candidates by train-only proxies. Its pick averaged 0.826 on Adult
   but 0.850 ± 0.116 on Abalone; `sparse_gaussian_copula` was best on Adult (0.832) and
   worst on Abalone (0.545); the compact residual was best on Abalone (0.897) and
   middling on Adult (0.803). Transfer-in-the-loop selection (R1 below, implemented in
   `examples/quickstart/dope_select.py`) reached 0.827 and 0.885, within 0.012 of the best
   fixed backend on each table: the best average of every strategy tried.
2. **One-hot columns cost retention on high-cardinality tables.** The benchmark
   projection one-hot encodes every categorical. A target-ordered code (one column per
   category, levels placed by smoothed target mean) reached 0.794 against 0.762 on Adult
   with fewer bytes, and tied on Abalone (0.854 both, with a third of the seed-to-seed
   spread). A full set of dummies is also collinear with the closed-form read-out's
   intercept, so one-hot needs a dropped reference level.
3. **Scaling matters, and quantile maps are expensive.** Robust scaling (log for skewed
   columns, Tukey fences) gave 0.854 and 0.794, min-max 0.833 and 0.800, and a 256-knot
   quantile map 0.846 and 0.749 at four to ten times the encoder bytes.
4. **Bytes are spent on text.** The JSON projection sidecar is 60% of DOPE's median
   charge.
5. **The IR already has transforms the compiler never emits.** `Log1p`, `Power`,
   `Logit` and `Winsorize` decode and sample, but `fit_column` always writes
   `Identity`.
6. **The CPU residual stores weights it could regenerate.** Its hidden weights are a
   fixed hash of `(unit, input)`; the artifact stores them anyway.
7. **Only regression and binary targets.** Multi-class targets are refused, and the
   native mixed path draws categories independently given one anchor split.
8. **A magic-number bug** made `dope-kernel inspect` reject mixed artifacts; it is fixed
   in this change (`DOPMIX01`).

## 7. Roadmap: concepts ranked by expected value per unit of effort

| # | Idea | Why it should work | Expected effect | Cost |
| --- | --- | --- | --- | --- |
| R1 | **Transfer-in-the-loop selection, in Rust.** Score each tournament candidate by K-fold train-on-synthetic, test-on-held-out-fit-rows retention with two cheap auditors (ridge and a small GBDT), entirely inside the fit rows. A four-backend Python prototype runs in the quick start (`dope_select.py`). | Proposition 1: transfer is what we want; proxies only correlate with it. It also removes selection exposure from the profile choice. | the prototype matched the per-table best backend within 0.012 on both ablation tables; moving it into the full ~108-candidate tournament should do better | days; CPU |
| R2 | **Rate–distortion bit allocation.** Spend knots and bits per column in proportion to its importance for `ŷ` (water-filling on a Lagrangian `D + λR`). | Proposition 1 weights errors by their effect on `m_P`; uniform resolution wastes rate on irrelevant columns. | 20–40% fewer marginal bytes at equal retention | days |
| R3 | **Binary, entropy-coded state.** Replace the JSON projection with a binary codec, delta-code monotone knots, and entropy-code (rANS) quantized parameters. | Removes text and redundancy; no model change. | about 2× smaller median charge on the v2 panel; retention unchanged | days |
| R4 | **Seeded random features.** Store a generator id instead of the CPU residual's hidden weights; keep only the read-out. | The weights are already a deterministic function of their index. | `s·h·b/8` bytes saved (144 bytes for `s = h = 12`, 8 bits) | hours |
| R5 | **Teacher-distilled conditional mean.** Fit `ŷ` to out-of-fold predictions of a strong teacher (CatBoost, or a tabular foundation model) on the fit rows, not to raw labels. | Proposition 1 rewards `m_Q ≈ m_P`; a teacher's predictions are a lower-variance target than noisy labels. | higher CatBoost and MLP retention at the same bytes | days; GPU for foundation teachers |
| R6 | **Support-aware dependence choice.** Choose the dependence family by a classifier two-sample estimate of `χ²(P_X ‖ Q_X)` on held-out fit rows. | Corollary 2. | better tree-auditor retention; fewer independence picks | days |
| R7 | **Emit the transforms the IR already has.** `Log1p`/`Logit`/`Winsorize` per column inside the compiler. | Makes the native mixed path as good as the quickstart encoder; heavy tails become linear. | parity with external robust scaling, no sidecar | hours |
| R8 | **Multi-class and multi-target programs.** One shared feature model, one residual head per target. | Bytes for marginals and dependence are amortized. | new tasks; lower amortized bytes per target | weeks |
| R9 | **A formal-DP profile.** Laplace/Gaussian-mechanism marginals and rank correlations, objective-perturbation read-out, with an accountant. | Few parameters means little noise per parameter; compares directly with AIM and MST. | the first byte-charged DP synthesizer with transfer receipts | weeks; needs review |
| R10 | **Two-part code-length selection.** Choose among candidates by `L(A) + L(D_val | A)` (Rissanen's principle) as an alternative to R1. | A principled trade-off of bytes and fit; cheap to compute from the tournament. | a second, parameter-free selector to compare with R1 | days |
| R11 | **Measure the frontier.** Sweep the knot and quantization ladders and the TDL byte grid for DOPE and every comparator; report `R(τ)` curves and the area under them. | The curve, not one point, is the scientific object. | the first TDL measurement | GPU days |
| R12 | **Stabilize the closed-form read-out.** Add a small ridge term (or drop collinear basis columns) where the compiler solves for target and residual read-out weights. | One-hot inputs that sum to one are exactly collinear with the intercept; the quick start's encoder now drops a reference level, but the compiler should not depend on it. | robustness on arbitrary encoded inputs; no change on well-conditioned tables | hours; must keep historical artifact bytes reproducible behind a profile flag |

## 8. Tests that would make the claims sturdier

* **Metric properties as property tests:** TDL is monotone in `τ_T`; retention is
  invariant to affine maps of a regression target; `T_robust` never rises when a family
  is added (exists); column order does not change any number.
* **Adversarial generators:** a lookup-table "generator" that stores rows must fail the
  copy gates at every budget; a predictor-only sampler must fail the control-lift gate.
* **Null strength for foundation models:** the null must include no-context and
  label-shuffled context, so a pretrained prior earns no credit.
* **Seed exchangeability:** fit seed 11 ranks first on far more lineages than chance on
  the v2 panel; report seeds 23–71 alone beside every headline, as the paper does.
* **Selection-free confirmation:** run R1 inside the fit rows only and evaluate once
  on the sealed split; this removes the panel's selection exposure by construction.

The compute these need is listed in the pull request that added this page.
