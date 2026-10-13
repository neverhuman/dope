# Portable research adapter conformance

`research/transfer/adapters/base.py` defines the offline interfaces for PR09.
It imports no vendor package, workstation checkout, product route, database or
process launcher. It does not download code or checkpoints. The existing Python
research exception and `research/transfer/` allowlist remain unchanged.

## Admission and exact identity

`ModelIdentity` binds the source revision and artifact hash, optional initial
weights, exact dependency versions and offline artifact hashes, code/weight
license identifiers and text hashes, preprocessing, recipe, aggregation,
environment, precision, context and query policies, determinism and resource
limits. Manifest hashes must cover all native transforms, target inverse
transforms, class order, dependency closure and any ensemble settings. Runtime
and hardware configuration belong in the environment manifest. A changed pin
or policy changes the identity. Ports must verify local artifact bytes through
the executor before constructing an adapter; a hash declaration alone proves
neither availability nor correctness.

`AdapterAdmission.from_registry` checks the custodian's independent registry.
Admission requires an exact identity match, a compatible license decision, an
eligibility-review hash and a native-reference evidence hash. The registry must
come from the executor's trusted policy, never the candidate's exported bundle.
The supplied model and generator registries are prospective and **unresolved**:
none of their entries qualifies an official model or grants evaluation/release
authority. Unavailable models, incompatible/unresolved licenses and identity
drift raise `AdapterUnavailable` with stable codes. Unknown fields, duplicate
entries or malformed pins fail closed. Registry names describe future ports;
they do not certify aliases as independent architectures or checkpoints.

The ten historical refused encoders remain blocked even if a supplied registry
entry says reviewed. A separate eligibility review must explicitly change that
policy before an encoder can be reconsidered. An arbitrary new checkpoint hash
does not override the refusal.

## Fit, predict and sample

`TrainingBatch` holds immutable feature rows and training targets, with role
`development_train` or `synthetic_train`. `FeatureBatch` holds only immutable
feature rows, opaque row commitments and a schema hash. Neither type prints raw
rows in its representation. The query argument accepts the exact feature-only
type; it has no label field. Labels for query rows stay with the scorer.

| Mode | Meaning of learner preparation | Initial checkpoint |
| --- | --- | --- |
| `scratch` | Train fresh parameters on the declared training view | Forbidden |
| `frozen_context` | Attach context without updating checkpoint parameters | Required |
| `fine_tuning` | Update a pinned initial checkpoint on the declared training view | Required |
| `pretraining` | Run a separately identified development pretraining stage | Optional |

Backend ports implement the declared mode and prove its behavior against their
native reference. Declaring a mode does not attest that a vendor obeys it.
Pretraining data rights, source-family exposure and protected-partition custody
remain independent gates. Preparation errors invalidate prior prepared state.
Backend exits become stable errors; keyboard cancellation remains a cancellation
with its diagnostic payload suppressed. Backend messages never become public
adapter diagnostics.

Context policy `all` rejects an input exceeding the declared cap. Policy `head`
selects a declared prefix before backend preparation. Identity records the cap
and policy; the invocation key also records the full provided training hash,
provided count, selected context hash and consumed count. A learner's returned
consumed count must equal the selected count. Hidden internal truncation requires
a different port/policy and reference evidence. Queries and samples must have
the fitted feature width and schema hash and cannot reuse any provided context
row ID, including rows excluded by the cap.

`LearnerAdapter.predict` performs one backend call per supplied query batch.
It rejects batches larger than `query_batch_size`; `rowwise` requires size one.
`batch` permits joint prediction within a batch; `transductive` explicitly
permits interaction among query features. Callers must preserve the declared
batch partition. The interface never silently chunks, sorts or shuffles queries.
Order sensitivity is declared and tested by the port: invariance is not imposed
on methods whose native behavior depends on context/query order. Invocation
identity always includes the actual query order and features.

`PredictionResult` returns exactly aligned, finite original-unit point values,
probabilities with an explicit unique class order, or a distribution vector
whose semantics the aggregation manifest defines. Point width is one;
probability rows must sum to one. Ports must inverse-transform regression
targets using training-only parameters, preserve declared absent class levels
and handle unseen feature levels according to the pinned preprocessing policy.
Raw-unit reference parity is mandatory before official admission. Optional
aligned embeddings can be absent with `not_requested`, `unsupported` or
`unavailable`; valid predictions and their invocation identity remain usable
by the independent receipt writer. Embedding availability does not qualify a
representation and never substitutes for prediction evidence.

`GeneratorAdapter.fit` accepts only development training data. `sample` receives
only requested row count and seed, and returns `SampleResult`: an aligned
synthetic training batch of exactly that count and resource usage. It has no
query-data argument. Generator ports
must additionally prove native sample behavior and source-free bundle
export/reload through PR07; the base interface does not supply a serializer.

## Execution and scientific evidence

Resource limits participate in identity. Preparation returns `ResourceUsage`
with consumed context count and memory/GPU peaks. The wrapper checks cumulative
wall time (including failed calls), reported context consumption, and reported
memory/GPU peaks for preparation, prediction and sampling.
The independent executor must enforce timeouts and observe/enforce peak memory,
GPU, fit, sampling and artifact-byte budgets. Python callbacks cannot preempt a
blocked vendor or establish trustworthy peak usage from a self-report.

The executor must provide offline process isolation, local verified dependencies
and only the authorized training/query-feature views. Role strings, row hashes
and Python type checks do not physically prevent a malicious backend accessing
other files. PR03 custody and the existing audited worker boundary retain that
responsibility. The PR09 fixture's network/process denial and source import
check establish the behavior of this interface, not a new sandbox exemption.

`cache_key` is a private invocation key, not a completed scientific receipt. It
binds complete model identity, full/consumed training data, fit/request seeds,
request hash and requested count. Use the feature-batch hash for prediction;
for sampling use the independently pinned sample request hash. The PR02
executor must also bind the full `EvaluationIdentity`: protocol, family/task,
splits, source, artifact/query data, schema/transforms, generator/learner and
checkpoint/recipe identities, all row/context counts, seeds, environment and
evaluation code. Only PR02 verification can issue a verified completed receipt.
Do not admit this adapter key or a prediction result directly into that cache.

## Validation and remaining gates

Run `python3 -B -m unittest research.transfer.tests.test_adapters -v`, `just fast`
and `just check`. Fixtures use synthetic constants and standard-library code.
They exercise native linear-reference parity after an affine target inverse,
alignment, finite outputs, class order, context cap/counts, cache invalidation,
mode/checkpoint separation, no-query-label inputs, offline operation, typed
unavailability and preservation of predictions without embeddings.

Official code/checkpoint/license verification, clean-machine vendor smoke tests,
GPU resource measurements, author-native parity and generator export/reload are
unmeasured. PR08 schema integration is pending; this interface requires a pinned
schema and width but does not replace native category/missingness constraints.
No empirical privacy result, formal DP claim, legal de-identification claim,
model eligibility or release tier is established by these fixtures.
