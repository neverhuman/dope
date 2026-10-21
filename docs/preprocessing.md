# Preprocessing that preserves the task

The generator's coordinates and a downstream learner's coordinates have
different jobs. Fit the generator representation using fit rows only, charge
its retained state, invert it when sampling, and let every auditor apply its
own fixed preprocessing recipe to original-unit rows. An affine transform may
help optimization; a rank transform may help a dependence model. Neither is
automatically the best representation for every learner.

## What is implemented

| Area | Current implementation | Consequence |
| --- | --- | --- |
| Canonical layout | [Explicit target-last intake](data-contract.md), `rust/typed_data.rs`, `research/transfer/tdl/target_last.py` | Resolve the declared target before moving columns; never guess the label from an external file's final position |
| Numeric transform | `rust/typed_data/transforms.rs` | Fit-only offset and range, `(x - offset) / scale`; a constant column uses scale one |
| Nominal strings | `rust/typed_data/mixed.rs` | Column-scoped exact one-hot categories; bounded carrier, not numeric label encoding |
| Missing/unknown state | Typed cells and explicit mask states | Missing, unknown category and structural absence are distinct; do not make them all numeric zero |
| Historical paper dependence | Marginals plus Gaussianized rank-space dependence | The latent coordinate system differs from the user-facing original units |
| Raw numeric successor | `rust/typed_data/workflow.rs`, `docs/native-workflow.md` | Original-unit replay, integer/Boolean repair counts, explicit domain refusal; a different charge profile from the historical paper |

The mixed carrier limits the latent feature width and category cardinality in
its source. It currently stores empirical class/mask probabilities with an
optional split on a numeric feature, rather than a general conditional model
for all nominal dependencies. One-hot support alone does not establish
mixed-type synthesis quality. Unsupported inputs must remain explicit refusals.

## Recommended default and controlled alternatives

Keep the existing versioned affine/one-hot behavior as the reproducible
baseline. Scikit-learn's [MinMaxScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
explains that it does not remove outlier effects. Its
[scaler comparison](https://sklearn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)
also shows how quantile transforms alter distances and saturate extremes.
Use those facts to choose experiments, not to declare a universally better scaler.

| Feature type / problem | Proposed comparison | State and failure checks |
| --- | --- | --- |
| Ordinary continuous measurement | Current affine baseline vs median/IQR scaling with an invertible bounded link | Store median, scale and link; count them; inspect tail amplification after quantization |
| Positive heavy tail | `log1p(x / s)` then affine scaling | Declare support and fit-only scale; preserve structural zero and inverse transform |
| Signed heavy tail | `asinh(x / s)` then affine scaling | Invertible through zero; compare with untransformed physical distances |
| Multimodal marginal | Monotone empirical-CDF knots plus explicit tail policy | Charge knots and tail parameters; measure ties, saturation and extrapolation |
| Low-cardinality nominal | Exact one-hot baseline | Enforce one active category on decoding; permute category names as an invariance test |
| High-cardinality nominal | Fit-only rare pooling with explicit unknown bucket | Pooling changes information content; retain and charge the map, measure rare-class utility and disclosure |
| Ordinal | Declared ordered levels | Never infer order from nominal strings or alphabetical category IDs |
| Time / cyclic variables | Declared physical period and sin/cos features | Store period and phase; group derived columns in attribution and account for support constraints |
| Target | Affine baseline; original-unit evaluation | Nonlinear target transforms change the training loss and conditional estimand; compare as a separate task recipe |

These alternatives are research proposals, not options silently added to a
frozen artifact decoder. Use a new explicit profile and carrier version where
necessary, preserving historical DPK and receipt decoding.

For a Pandas-like numeric matrix, fit **one** category vocabulary on training
rows and reuse its ordering at query time. Independently calling `get_dummies`
on train and test can create different columns. Follow an explicit
[unknown-category policy](https://sklearn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
and group all dummy columns of one original feature for permutation importance.
Supervised target encoding needs out-of-fold fitting inside each training
partition; it is not an unsupervised replacement for one-hot encoding.

## A useful invariance, with a precise limit

If every evaluated target and prediction is transformed by the same
`y' = a*y + b`, with nonzero `a`, every squared error is multiplied by `a*a`.
The retention ratio therefore stays the same because that factor cancels in
numerator and denominator. This proves **evaluation invariance for fixed
predictions**, not invariance of a fitted learner: regularization, stopping,
feature scaling and quantization can change the predictions. A nonlinear
target transform has no such cancellation and must be inverted before scoring
the original task.

Do not clip query observations to the training min/max to make a unit-interval
interface look valid. That changes the task, especially its tails. Historical
projected benchmarks and the native raw successor have different support
contracts; report them separately. Compare feature and target tail errors,
category validity, rare-class lift, groupwise driver agreement and full bytes
beside the mean transfer result.

## Experiment acceptance

Freeze the transform menu and selection rule on development data. Within each
training fold, refit all imputers, category maps, scalers and knots from that
fold's fitting rows only. Cross-fit any label-dependent step. Use shared real
queries, sample counts and learner recipes for all methods. Run both equal
complete-byte and equal-compute tracks; charge the transformed representation's
entire state, including dictionaries, masks and support metadata. Choose a
successor only after paired improvements survive the frozen test procedure;
then run held-out learner and source families once. The experiment matrix and
compute request are in [the publication upgrade review](publication-upgrade.md).
