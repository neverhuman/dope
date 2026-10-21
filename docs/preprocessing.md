# Fit once, preserve semantics, evaluate in original units

The deployed compiler and sampler are Rust. Python training and external model
evaluation remain research tools. All fitting dictionaries, transforms, ranges,
missingness models, and decoder parameters that sampling needs are charged state.

| Input | Current native path | Important boundary |
| --- | --- | --- |
| Observed numeric regression | `dope compile --csv fit.csv --schema schema.json` | Complete raw artifact retains fit-only affine inverses; sampling restores original units. |
| Nominal predictors and typed masks | `dope-kernel compile-mixed ... --profile numeric_split_v1` | Explicit development profile: 32 observed categories per nominal column, 128 numeric/one-hot coordinates, observed continuous target. Output is tagged cells, not a flat numeric CSV. |
| Prepared model inputs | `typed_data::prepared::prepare` and `PreparedContract` | Private, frozen preparation contract; the current prepared nominal view uses category codes, not a claim that those codes are metric coordinates. |
| Historical numeric benchmarks | Legacy `.dpk` and projection state | A different byte and support profile; do not reinterpret historical receipts as raw-unit results. |

For your own CSV, declare each column's role and type in the schema before
fitting. See the concrete schema producer in
[`examples/public-regression/run.py`](../examples/public-regression/run.py) and
the schema contract in [`rust/schema.rs`](../rust/schema.rs). A nominal column
belongs in the mixed profile; passing arbitrary integer category codes through
the numeric facade makes an unintended ordering available to the model.

## Why global get_dummies plus min/max is insufficient

Split physical entities, repeated configurations and time windows first. Fit the
dictionary and scaling on the fitting partition only. Freeze the same map for
synthetic, real-reference and query inputs. Otherwise category discovery,
extrema, missing-value imputation or target-driven transforms leak information.
The target is explicit at intake and last in every new canonical model table.
Entity IDs and split keys are excluded from predictors, not numerically encoded.

For nominal strings, use exact column-scoped dictionaries and one-hot coordinates
at an interface that needs a numeric matrix. A single categorical variable must
still be sampled as one level: independent Bernoulli dummy columns can produce
impossible multiple-level rows. Combine every dummy and mask coordinate into
its original feature group for driver tests. Unknown category, missing value,
structural absence, padding and censoring are different states. A string whose
literal text is `missing` is an ordinary category unless the schema says otherwise.
Rare-level pooling requires an explicit fit-only policy and disclosure; it can
destroy precisely the small subgroups whose transfer matters.

Affine min/max is compact and invertible. It is a good baseline for a bounded
symbolic language, but a distant outlier can compress most values into very few
quantization bins. Later values can lie outside `[0,1]`; do not silently clip them
and then report preserved tails. `FittedTransforms` preserves its affine inverse;
the separately named frozen numeric embedding transform has historical clamping
semantics. They are distinct APIs.

## Named comparisons to run before changing the default

| Profile proposal | Transformation | Expected value and principal failure mode |
| --- | --- | --- |
| `affine-v1` | `(x - min_fit)/(max_fit - min_fit)`; scale one for a constant | Small, transparent baseline. Outliers waste central resolution; default feature synthesis remains fit-range bounded. |
| `robust-asinh-v1` | `asinh((x - median_fit)/s_fit)`, with a frozen positive robust scale | Signed heavy tails, no arbitrary ordering or irreversible clipping. Requires a new explicit sampler coordinate/support contract; cannot be substituted into a historical `[0,1]` decoder. |
| `ecdf-spline-v1` | Fit-only monotone quantile map with declared knots and inverse | Better central allocation for copulas. Ties need a declared discrete policy; finite knots and their full bytes count. Extrapolation or bounded tails must be named. |
| `hybrid-v1` | Robust raw coordinate for target modeling, rank coordinate for dependence | Can preserve distances and dependence simultaneously. It duplicates fitted state and must win at equal **complete** bytes. |

For robust scale, compare IQR/1.349 and MAD/0.67449 on development data, with a
fixed fallback for zero dispersion. Never choose a transformation using query
targets. Numeric target maps need a frozen inverse; MSE, RMSE and MAE are always
reported in the original target units. Nonlinear target maps change the loss
geometry, so a transformed-space improvement is insufficient.

Run each profile on identical fit/query row IDs, byte budgets and learner recipes.
Measure transfer, original-feature driver agreement, subgroup and tail losses,
out-of-support frequency, compile/sample time, RSS and complete artifact bytes.
Include constant, repeated, missing, rare-category, outlier, OOV and held-out-range
cases. Publish failures and unsupported cells. These proposals are unmeasured
ablations, not a claimed improvement over the current default.
