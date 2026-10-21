# The mathematics of DOPE

What a DOPE program stores, why learners trained on its samples transfer to real
rows, when a program carries more predictive value per byte than the rows it
was compiled from, what the current implementation gets wrong, and which
experiments would move the frontier next. Numbers quoted here come from
`research/benchmark/equal_bytes/results/summary.json` (26 real-world PMLB
tables) and `panel-summary.json` (all 97 lineages), protocol `equal-bytes-v3`,
development validation, unless another source is named; regenerate
them with the commands in `research/benchmark/equal_bytes/README.md`.

## 1. Setup

A table is an i.i.d. sample `D = {(x_i, y_i)}_{i<=n}` from `P` on
`X x Y`, with `x` in the unit cube after the fitted column plan (Section 6).
A compiler maps the fit rows to a program `A` of `|A|` bytes. A sampler draws
rows from `Q_A`, a distribution defined by `A` alone. For a learner `m`, let
`L_null`, `L_real` and `L_syn(A)` be its loss on untouched real test rows when
it predicts the training mean (or prior), when it is trained on the real fit
rows, and when it is trained on samples from `Q_A`. Retention (transfer) is

```
T_m(A) = (L_null - L_syn(A)) / (L_null - L_real).
```

TDL aggregates the worst learner family, `T_robust(A) = min_f T_f(A)`, and
asks for the smallest tested byte budget at which `T_robust` (with its lower
confidence bound) clears a threshold `tau` together with driver and copy gates.
Written as a rate-distortion problem with distortion `1 - T_robust`,

```
TDL(tau) = min { B in grid : max_{|A| <= B} T_robust(A) >= tau }.
```

The frontier `C(B)`, the share of tables that qualify at budget `B`, is what
Figure 1 of the manuscript plots.

## 2. What the program stores

The compiled program factorizes the joint law as

```
Q(x, y) = c_theta(F_1(x_1), ..., F_p(x_p)) * prod_j f_j(x_j) * q(y | x)
```

| Part | Representation | Parameters (headline recipe) |
| --- | --- | --- |
| Marginals `F_j` | quantile spline, `K = 17` knots on the unit interval, plus a missing rate | `p (K + 1)` unit values |
| Dependence `c_theta` | Gaussian copula restricted to a tree: each column has one parent among the previous 16 (triangular) | `p - 1` correlations |
| Conditional mean `m(x)` | `c + beta^T x_S + v^T tanh(W x_S + b)`, `|S| <= 12` screened inputs, 16 hidden units | `12*16 + 16 + 16 + 12 + 1 = 237` |
| Noise `q(y|x)` | Gaussian with `log sigma(x)` linear in 4 features, clamped to the 10th/90th residual percentiles; Bernoulli with a calibration map for binary targets | about 8 |

Every parameter is quantized on a grid fixed by the header bit width `b`:
unit values to `2^b - 1` levels, unbounded values to steps of `2^-(b-2)`.
Sampling draws `z` from the copula, maps `x_j = F_j^{-1}(Phi(z_j))`, computes
`y = m(x) + sigma(x) eps`, and finally masks features at their missing rates.

The `tanh` network is the only part whose size does not grow with `p`: at most
237 numbers regardless of width, rows or noise. That is the source of the
compression.

## 3. Why samples from the program transfer

### 3.1 Squared loss: retention is the program's explained signal

Write `m_P(x) = E_P[y | x]`, `s_P = E_P[Var(y | x)]` (a variance) and similarly
`m_Q`. Consider a learner that is consistent on the support of the distribution
it is trained on, in the limit of many synthetic rows and many real rows, so
that both estimation errors are negligible (Proposition 3 keeps the real sample
finite).

**Proposition 1.** With `L_null = Var_P(y)`, `L_real -> s_P` and
`L_syn -> s_P + E_P[(m_Q(X) - m_P(X))^2]`,

```
T = 1 - E_P[(m_Q(X) - m_P(X))^2] / Var_P(m_P(X)).
```

*Proof.* For any predictor `g`, `E_P[(y - g(X))^2] = s_P + E_P[(m_P - g)^2]`.
The null predicts `E_P y`, giving `s_P + Var(m_P)`; the real learner
converges to `m_P`; the synthetic learner converges to `m_Q` wherever `Q_X`
has support covering `P_X`. Substitute into the definition of `T`. QED.

Retention is the share of the true signal variance that the program's
conditional mean explains, measured under the *real* covariate law. Three
consequences:

1. The noise model and the joint feature model do not enter the population
   limit. They enter through support (3.2) and through finite-sample variance.
2. The quantity a compiler should spend bytes on is `E_P[(m_Q - m_P)^2]`. This
   is why the conditional mean is stored explicitly and why the network gets
   most of the bytes.
3. Retention above one is possible at finite synthetic size: `4n` synthetic
   rows from a smooth `m_Q` can train a data-hungry learner better than `n`
   noisy real rows. The real-row resampling control in the historical paper
   shows the same effect, so values above one measure row count, not fidelity.

### 3.2 Which learners need the joint model

**Corollary 1 (linear learners).** A linear learner trained on `Q` converges to
`beta_Q = Sigma_Q^{-1} Cov_Q(X, m_Q)`. If `Q_X` matches the first two moments of
`P_X` and `m_Q` matches the linear projection of `m_P`, linear retention is one
even when `m_Q` is wrong nonlinearly.

This explains the historical predictor-only control (frozen v2 ledger): a ridge
regressor with independent marginals keeps 0.98 of the linear learner's gain but 0.63 of
CatBoost's.

**Corollary 2 (flexible learners).** Let `w = dP_X / dQ_X`. For a learner whose
error on `Q` is `e(x) = f_Q(x) - m_Q(x)`, Cauchy-Schwarz gives

```
E_P[e^2] = E_Q[w e^2] <= sqrt(E_Q[w^2]) * sqrt(E_Q[e^4]),   E_Q[w^2] = 1 + chi^2(P_X || Q_X).
```

A tree or network learns `m_Q` well only where `Q_X` puts mass, so the
dependence model pays for itself through the chi-square divergence between
real and synthetic covariates (Shimodaira 2000). This motivates measuring
`chi^2` (or a classifier two-sample AUC) at compile time, which the current
selection rule never does (Section 5).

### 3.3 Log loss: retention is delivered label information

For a binary target with `p_P(x) = P(y = 1 | x)`, the null predicts the prior
and the Bayes-optimal real learner reaches the conditional entropy. Under the
same support and limit conditions as Proposition 1 (the synthetic learner
converges to `p_Q` on the support of `P_X`),

**Proposition 2.**

```
T = 1 - E_P[ KL( Ber(p_P(X)) || Ber(p_Q(X)) ) ] / I_P(X; Y).
```

*Proof.* `E_P[-log p_Q(Y|X)] = H(Y|X) + E_P KL(p_P || p_Q)` and
`L_null - L_real = H(Y) - H(Y|X) = I(X; Y)`. QED.

Retention under log loss is the fraction of the feature-label mutual
information the program delivers (the UCI Adult walkthrough reports it for
logistic regression, an MLP and CatBoost; see `examples/adult/reference.json`).

## 4. Why a program can beat the data at equal bytes

The two-part-code view (Rissanen 1978) splits the description of a table into
a model and the residuals given the model. Shipping real rows sends both;
shipping a program sends the model and replaces the residuals by a few bytes of
noise model that regenerate *fresh* residuals.

**Proposition 3 (scaling of the data-equivalence ratio).** Suppose a learner
trained on `k` real rows has excess risk `C/k` with `C = d_eff s_P`, and let
the real reference use `n` rows. The rows needed to reach the program's
retention `T` are

```
k(T) = C / ((1 - T) Var(m_P) + T C / n)  ->  (s_P / Var(m_P)) d_eff / (1 - T)   as n -> infinity,
```

and each row costs `(p + 1)` quantized values, so the data-equivalence ratio is

```
DER = k(T) (p + 1) b / (8 |A|).
```

*Proof.* Solve `(L_null - L_k) / (L_null - L_n) = T` with `L_k = s_P + C/k`.

Predictions and their measured status:

| Prediction | Measurement | Status |
| --- | --- | --- |
| Program bytes do not grow with `n` | 598 to 1,148 B from 250 to 16,000 rows on four tables, while the whole table grows to 88,166 B | holds by construction: a program's size is set by its shape |
| The program beats the data once the table outgrows it | equal-byte gap (worst learner) against whole-table bytes over program bytes: Spearman 0.78 over 26 real tables (0.72 on the 20 whose whole table is larger than the program) and 0.59 over all 97 panel lineages. On the 9 real tables at the 2,000-row cap the program wins 8, loses none, gap +0.38 [0.12, 0.69]; on the 38 panel tables at the cap it wins 36, loses none, gap +0.24 [0.16, 0.29] | confirmed |
| Tiny tables are cheapest to ship whole | on the full panel all 21 equal-byte losses are tables of at most 350 fit rows whose whole table is at most about 2 KB; two of them (the apnea tables) are tables where the program itself fails | confirmed |
| DER grows with `n` | Spearman 0.63 between DER and fit rows over 26 real tables, 0.69 over 97 panel lineages | consistent, but partly mechanical: DER is capped by the whole table's size, which grows with `n` |
| DER grows with noise share over `(1 - T)` | Spearman 0.82 between DER and the proxy `noise/(1 - noise)/(1 - T) * (p + 1)/|A|` on the nine real 2,000-row tables, 0.56 on the 38 large panel tables | ranks, does not predict levels: about half of the panel's large tables are noise-free equations whose learning curves are approximation-limited rather than `C/k` with `C` proportional to noise. A better proxy needs a measured learning-curve exponent per table |
| At strict thresholds the program becomes the bottleneck | to reach T80, real rows need fewer bytes on 14 of 26 real tables (the program on 6); at T90 on 19 (the program on 6). On the panel: 60 against 26 at T80 | confirmed, and already at T80: `1 - T` is bounded below by the network's approximation error |

Numbers are from `research/benchmark/equal_bytes/results/summary.json` and
`panel-summary.json` (protocol `equal-bytes-v3`); the README block and the
manuscript render them from those files.

The last row is the honest limit: a 237-parameter conditional mean is a lossy
channel, so `TDL(0.8)` and `TDL(0.9)` are often unreached by the program while
enough real rows reach them at a few kilobytes. Raising the program's ceiling
at fixed bytes is the main research direction (Section 7).

### 4.1 TDL as an operational rate-distortion function

For a fixed table, `D(B) = 1 - max_{|A|<=B} T_robust(A)` is non-increasing in
`B`, and `TDL(tau)` is its generalized inverse on the grid. Real rows define a
second curve, `D_rows(B)`, the distortion of the trivial code. The program
dominates the data where `D(B) < D_rows(B)`; Figure 1(b) of the manuscript
shows the crossover. No claim is made that either curve estimates a Shannon
rate-distortion function: the learners, losses and grid are operational.

## 5. Audit of the current implementation

Findings from reading the compiler, codec and sampler; references are to
`rust/`.

**Fixed in this release**

1. *Quantization bits did not set bytes.* Unit values were always written as
   u16 and unbounded values as `varint(round(1e6 v))`, so `q6` and `q16`
   programs had nearly the same size. The packed value codec
   (`codec/packed_values.rs`, header flag bit 1) writes grid codes and refuses a
   kernel whose parameters are off its grid. The same programs are 1.85x
   smaller (median over 26 tables), with bit-identical samples for the headline
   shape; shapes with copula edges or bit widths whose grid differs from the
   historical u16 grid agree only to the codec's parameter tolerance.
2. *CPU builds did not train the headline network.* Without libtorch the
   compiler used fixed hash-derived hidden weights with a closed-form readout.
   `compiler/target_fitting/native_training.rs` trains the published shape with
   the published optimizer (full batch, AdamW 2e-3, weight decay 0.01,
   element-wise clipping at 5, ridge readout refit). Over 96 panel lineages its
   CatBoost retention shows no systematic shift from the published GPU run
   (median difference 0.000), but per-table differences are wide (10th to 90th
   percentile -0.40 to 0.22) because the historical splits cannot be rebuilt.
3. *The JSON projection dominated small artifacts.* The historical charge was
   `model.dpk` plus a text `projection.json` (often larger than the model). The
   equal-byte study charges the fit-row scaling as two f32 per column; with the
   packed codec this brings the median charge from 2,666 B to 676 B (4.0x). The
   table artifact stores its full binary plan (per column a kind byte, flags and
   f32/f64 bounds, plus zstd-compressed names and labels), which `inspect-table`
   reports.
4. *A valid full Gaussian copula did not exist.* Family 7
   (`candidate_dependence.rs`) stores the Cholesky factor of rank correlations
   mapped by `2 sin(pi rho / 6)`, so latent variances are one before
   quantization (the quantized factor is not renormalized). It is opt-in: on 26
   real tables it changed robust retention by a median of 0.000 (5 wins, 6
   losses) at a median of 27 more bytes (`results/ablations/gaussian-copula`),
   which is itself a finding, consistent with Proposition 1: dependence is not
   the bottleneck.

**Open (not changed, to keep historical artifacts reproducible)**

5. *Families 3 to 6 are not valid copula samplers.* Sequential edge updates on a
   general graph (sparse copula, POET residual edges, vine) let latent variances
   drift from one, so `Phi(z_j)` is not uniform; the vine's conditioning depth is
   ignored by the sampler; POET uses the first `r` columns as factors instead of
   principal components; the mixture weights are fixed at one half.
6. *Spearman is used as the latent Pearson correlation* without the
   `2 sin(pi rho / 6)` correction, on median-imputed columns, with no polychoric
   treatment of discrete columns. This attenuates dependence.
7. *Selection never measures transfer or joint fidelity.* The shortfall rule
   compares in-sample R-squared against 0.99, fixed fidelity priors per family,
   and a constant membership proxy, so ties go to the fewest bytes.
8. *Missingness is MCAR and applied after the target* is computed; the
   Bernoulli calibration's base rate is stored but unused; the "isotonic" map is
   binned means, not pool-adjacent-violators; the GAM is fitted as a step
   function and evaluated by linear interpolation.
9. *Heavy tails under min-max scaling.* With 8-bit knots, a column whose bulk
   sits in the bottom 1% of its range collapses to a near-constant. Log scaling
   skewed columns in the table plan raised CatBoost retention on 574_house_16H
   from 0.41 to 0.50 in a spot check; the benchmark harness still uses min-max.
10. *The mixed carrier overwrites copula draws for one-hot columns* with
    independent class draws, so category-numeric dependence survives only through
    the target model. The table workflow avoids this with one ordered code per
    category.

## 6. Normalization: why ordered codes and not dummies

The table plan maps every column to the unit interval with an invertible map:

* numeric: `u = (t(x) - lo)/(hi - lo)` with `t` the identity or, when
  `|skewness| > 2`, `t(x) = ln(1 + (x - x_min)/s)` with `s` the median distance
  to the minimum (mirrored for left skew);
* category with levels ordered by smoothed target mean
  `(sum_i y_i + 10 ybar)/(n_l + 10)`: level `i` of `L` sits at `(i + 0.5)/L`;
* dates: day counts; missing numeric cells stay missing (the kernel models a
  missing rate); missing category markers stay literal levels.

Ordering levels by the target statistic makes the code monotone in the level
main effect, so the conditional mean needs one coefficient where one-hot
encoding needs `L - 1`, and every sampled row decodes to exactly one level.
Dummies are an ML-ready *view* for downstream learners (`pandas.get_dummies`),
not a good *representation* for a byte-charged generator: each dummy is a
column with its own marginal and dependence bytes, and independent sampling of
dummies breaks the one-of-k constraint.

## 7. What to try next

Each item names the hypothesis, the test, and the compute it needs.

| # | Idea | Hypothesis | Test | Compute |
| --- | --- | --- | --- | --- |
| R0 | Large-table equal-byte study: 10^4 to 10^6 fit rows through `fit-table`, matched protocol, nested views | the equal-byte gap and the crossover budget grow with `n` (Proposition 3), the regime data owners live in | gap and crossover budget against `n`; `TDL(0.8)` per table | CPU cluster, about a day |
| R1 | Lagrangian rate-distortion compile: choose knots, bits, width, inputs and dependence by `D + lambda R`, with `D` measured by transfer on held-out fit rows | moves the program up its own frontier and removes profile-selection exposure | sweep `lambda`, report `TDL(0.8)` and `TDL(0.9)` on the 97-lineage panel | CPU cluster, about a day |
| R2 | Teacher distillation: fit the network to out-of-fold predictions of a strong learner (CatBoost, TabPFN) instead of noisy `y` | lowers `E_P[(m_Q - m_P)^2]` at equal bytes (Proposition 1) | A/B on the panel, paired by lineage | one GPU for a TabPFN teacher, else CPU |
| R3 | Bit allocation by sensitivity: per-tensor bits from Fisher information or measured loss sensitivity, plus entropy coding | same transfer at fewer bytes; the repository already carries a pre-registered precision pilot | compare against uniform `q6`/`q8` at matched bytes | CPU |
| R4 | Support objective: penalize a classifier two-sample AUC or `chi^2(P_X||Q_X)` estimate during selection | raises CatBoost retention (Corollary 2) | ablation on lineages where CatBoost is the worst learner | CPU |
| R5 | Correct copula families 3 to 6 and discrete-aware correlations | unlocks the dependence families the tournament already pays for | sampler conformance tests (variances, correlations) then panel A/B | CPU |
| R6 | Measure D80 | the driver gate is part of TDL but is unmeasured | grouped permutation importance across learner families on real test rows | CPU, hours |
| R7 | Differentially private programs: DP-SGD or objective perturbation on the 237-parameter head and noisy marginal quantiles | formal privacy at small utility cost because the parameter count is tiny | `(epsilon, delta)` sweep against the non-private frontier | CPU |
| R8 | Shared codebooks across tables (amortized bytes) | per-table bytes fall when a decoder prior is shared | charge amortized and per-table bytes separately, as the contract allows | GPU for the prior |
| R9 | Classification and categorical-heavy panels | external validity beyond PMLB regression | OpenML-CC18 / TabArena subsets through `fit-table` | CPU cluster |
| R10 | Sealed test | the only selection-free estimate | one pre-registered run of the frozen protocol | as R1 |

Tests worth adding to CI now: sampler conformance for every dependence family
(latent variance one, realized correlations within tolerance); packed-codec
round trips on every compiled candidate; table-plan round trips for all column
kinds; and a padded-bytes adversarial test (padding can only move an artifact
to a larger budget).
