The auditor ABI lives in `research.transfer.adapters.base.LearnerAdapter`.
`fit(training, seed)` returns an execution receipt, `predict(features, seed)`
receives no query labels, and `importance(real_test, perturbations, score_loss,
protocol_sha256, seed)` returns grouped permutation loss increases. Existing
`prepare` remains supported. CatBoost uses `ConventionalAdapter` with the
`catboost` recipe, one CPU thread and file writing disabled.

The evaluator owns training-only feature groups, conditional resampling, test
labels, repeats, real-real controls, stable directional effects and interaction
evidence. `perturbations` contains opaque group IDs and repeated FeatureBatch
views of the same untouched real-test row IDs. `score_loss` consumes aligned
predictions and returns an original-unit nonnegative loss. Native SHAP is not
required. Raw negative permutation effects are retained; evaluator policy must
freeze any truncation before constructing nonnegative metric input vectors.

Receipts contain commitments, counts, seeds and costs, never machine-local
paths or training values. Adapter success grants no scientific qualification.
Existing identity, resource, alignment, context and error contracts remain in
force. A failed fit invalidates previous prepared state.

Generator conformance adapters live in `research.transfer.tdl.generators` and
support fit -> export -> reload in a new process -> sample, with all sampling
reads traced. Copied rows and gzip-original are literal-storage controls, not
privacy claims. Gaussian copula and ARF are explicitly specified stubs; DOPE
uses the native executable and artifact-only isolation. Public toy smoke data
is in `tests/fixtures/tdl/`, marked citable:false.
