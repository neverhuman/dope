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
privacy claims. Gaussian copula retains its specified qualification status,
with a restricted numeric implementation described below; ARF remains a stub.
DOPE uses the native executable with the materialization limitation noted below.
Public toy smoke data
is in `tests/fixtures/tdl/`, marked citable:false.

Literal controls use a fresh `python -S -B` worker under Linux `strace` (required
alongside `ldd`). The trace follows native opens and all threads, rejects
undeclared reads/writes and process/network access, and releases output only
after verification. This checks trusted sampler behavior; it is not an OS
sandbox. The interpreter, stdlib, linked libraries, worker source and fixed
configuration are explicitly inventoried and charged as runtime closure bytes.
The physical TDL1 artifact is checked against the same canonical payload used
by Rust byte accounting. Every readable inventory file is hashed before and
after execution; a missing tracer, changed inventory or incomplete trace fails.
Public receipts contain opaque inventory IDs, charges and stable reasons, with
no raw trace, stderr or machine paths. The Python audit context remains a
diagnostic and reports incomplete coverage, even when no violation occurs.

## Bounded numeric research controls

`GaussianCopulaGenerator` and `CompressedRowsGenerator` implement actual
fit/export/reload/sample routes. They accept continuous feature/target columns
under the explicit v1 table schema, with one target; nominal, count, ordinal and
excluded-key columns are refused rather than reinterpreted. This is a restricted
research control implementation, not a qualified general tabular comparator.
The existing Gaussian registry qualification remains unchanged.

The copula uses tied midranks transformed to Gaussian scores, a correlation
matrix with fixed 1e-6 diagonal shrinkage, and a Cholesky factor. Sampling maps
seeded correlated normal draws through bounded empirical marginal quantiles.
It tries the fixed quantile counts 257/129/65/33/17/9/5/3 under the actual artifact
cap. Constants stay constant. The compressed-row control shuffles fit rows with
the declared uint64 seed, then tries all rows followed by successively halved
prefixes. Each complete gzip/TDL1 frame is measured; gzip size is not assumed
monotone and no maximal row-retention claim is made. Sampling resamples only the
retained rows. Both controls use deterministic gzip metadata and refuse a cap
that cannot fit their minimum representation.

The byte cap includes every dataset-specific artifact and framing byte. The
whole Python/runtime sampling closure is charged separately and reported by
`AccessTrace`; a small payload is not a small cold bundle. New control artifacts
are bounded before parsing, reject duplicate fields and decompression bombs,
and refuse final symlinks and nonregular files. Historical literal-control
artifact decoding remains intact. Exports are exclusive and durable; an uncertain
write is preserved rather than silently retried or overwritten.

Source CSV/schema files can be removed before reloading and sampling. Fresh
workers must pass the real Linux strace inventory check; this proves observed
file access for a trusted worker and is not an OS sandbox. These receipts remain
`citable=false`, and do not establish E03 empirical acceptance, vendor learner
qualification, independent reproduction, sealed custody, or release admission.
`DopeGenerator` now requests an explicit private f64le stdout stream from the
same isolated draw and reads its redacted completion receipt from stderr. It
requires successful exit, exact shape/length, the measured denial markers and
the same-byte sample digest before returning values. There is no second sampler
call or temporary CSV. Default native public output remains redacted JSON.

Control artifacts declare `features_then_target` output order. Fit preserves the
stable input feature order and moves the target last, matching the native raw
sampler even when the source schema/header places the target first.

New numeric controls require the declared CSV header and resolve its permutation
by exact schema names before applying canonical output order. Duplicate, missing,
extra or wrong-width columns are refused; historical literal controls are unchanged.

## Dedicated confined numeric control worker

The explicit research `sample_control` API starts a fresh dedicated worker with
caller-admitted runtime pins. A versioned Rust cdylib ABI invokes the existing
Landlock/confined-read boundary, then the existing sampling-only TSYNC filter.
The artifact is read only inside the private bundle between those phases. The
ABI is argument-free and refuses unsupported platforms; Linux x86_64 is required.
This API is not automatically selected by the legacy generator/tracing interface.

Runtime source and DSO snapshots, the interpreter and the cold dependency closure
are hash-bound separately from artifact bytes. The caller must deploy and admit
that runtime with exclusive trusted ownership; matching hashes do not defeat a
malicious same-UID writer or grant input rights. Runtime and invocation bytes are
charged in addition to the complete artifact. Samples stay private f64le stdout;
only a matching finite sample stream, redacted completion and successful exit are
accepted. Startup and execution errors are static refusal codes.

Tests use actual fresh processes for both controls, repeated seeds, source-file
removal, denied source/bundle reads after sealing, closed inherited descriptors,
and network/exec denials. They also retain actual missing-artifact, corrupt runtime,
symlink/FIFO/bomb, malformed output and startup redaction refusals. These fixture
proofs do not qualify a full comparator, E03 pilot, custody, E06 or release.
