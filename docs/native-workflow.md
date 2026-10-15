# Native raw-unit workflow

The Linux x86-64 default native installer ships `share/dope-kernel/LICENSE`
and `THIRD_PARTY_NOTICES` alongside the unchanged binary checksum receipt.
`installed.sha256` binds the binary, receipt and both legal files to fixed owned
paths. Uninstall validates the entire set before removing any member and preserves
unrelated files and old V1 environments. Historical two-file installs still uninstall
through their original binary receipt.

The notice inventory is frozen from the existing locked normal/build Cargo graph
and the pinned toolchain's local library legal inputs. `cargo run --locked
--offline --bin dope-native-notices -- --check` verifies regeneration from that
offline cache; plain `cargo run --locked --offline --bin dope-native-notices` is
the maintainer regeneration command after an admitted package-input change. Install needs neither Cargo nor Python.
It checks the shipped legal pins and the binary's embedded Cargo-lock and declared
toolchain-config digests before creating a destination. Default CPU packaging is
the supported profile; GPU/research adapters and separately provisioned OS runtime
require their own package notices. The declared toolchain digest binds build config,
not a compiler attestation or signed release. Added legal files are runtime/package
bytes, separately charged from a dataset's learned carrier bytes.

The native product is the Rust `dope-kernel` executable. Run its absolute path
or place its installation prefix first in `PATH`; an activated Python V1
environment must not choose the implementation. `dope-kernel doctor` reports
the actual executable SHA256, version, architecture, features and capabilities.
It opens no dataset and downloads no weights. Historical V1 artifacts remain
decodable; V1 Python remains a compatibility implementation.

Build the pinned CPU product with `cargo build --locked --release`. Install an
already built binary with `scripts/install-native.sh BINARY SHA256 PREFIX`.
The installer checks the checksum and native identity before copying and refuses
to overwrite an existing command. `--dry-run` verifies the pin and identity without
writing a destination. Copies are checked in a retained pending file and promoted
by a non-overwriting rename, so interruption cannot publish a partial binary.
This installer currently admits Linux x86-64;
other architectures and OS targets need actual platform qualification. The
installer uses standard Unix tools without installing a Python, research or GPU
environment. To remove a dedicated installation, use
`scripts/uninstall-native.sh PREFIX [--dry-run]`. It verifies the recorded
checksum and native identity, refuses a changed binary and removes only the
installed command and checksum receipt. Directories, unrelated files and the
V1 environment are preserved. Migrate using a new prefix and explicit native
path, keeping V1 available under its compatibility command.

`compile-raw --csv FIT.csv --schema PRIVATE_SCHEMA.json --out MODEL.raw`
fits the declared headered columns in fitting rows only. Use a separate query
file for evaluation; query labels never enter compilation. The default fixed
symbolic recipe is Chow–Liu, with seed 1729, a 60-second deadline and a 32,768-byte
complete-artifact cap. `--candidate`, `--seed`, `--deadline-seconds` and
`--max-complete-bytes` name changes explicitly. This raw profile is distinct
from historical rank-space paper and certification profiles.

An explicit residual comparison uses `--candidate development_symbolic` with
`--residual-profile no_noise_v1`, `student_t4_v1`, `gaussian_mixture2_v1`, or
`monotone_quantile17_v1`. This selects the named, unqualified
`raw-numeric-residual-v1` successor, with a 42-offer symbolic tournament and the
same complete-byte/deadline controls. Its `DOPERAW2` carrier contains DPK3.3
residual opcode4/version1; old `DOPERAW1` and DPK3.2 defaults keep their decoding
and behavior. A frozen backend cannot silently gain a residual profile.
`compile_raw_residual` exposes the same explicit library entry. All residual
parameters are encoded before candidate cap checks and included in inspection's
`native-raw-complete-residual-v1` charge.

Except for the explicit no-noise control, fitting uses at most256 permitted fit
rows in three fixed row folds and refuses unavailable OOF fitting. Student-t
uses fixed4 degrees of freedom with a second-moment scale; the compact mixture
uses lower/upper-half moments, without claiming an EM optimum. The17-knot
monotone quantile operator interpolates within its fitted OOF range. These are
development comparisons, not tail, calibration, privacy or transfer results.
Conditional masks and nominal columns remain explicitly unsupported here.

The current `raw-numeric-symbolic-v1` profile accepts fully observed continuous, integer, ordinal and Boolean
predictors with a declared continuous regression target. Nominal and missing-mask
capabilities refuse explicitly. Integer/ordinal samples round to the nearest
integer; Boolean samples use a fixed 0.5 threshold. The sampling receipt counts
these repairs. Any violation of a declared physical domain refuses without
clipping or retrying a draw. Source names, source rows and entity-key maps
are not retained. Fit-only affine transforms, units, domain declarations and any
approved semantic labels are private dataset-specific state, charged in full;
their retention is not a privacy guarantee. A constant column uses scale 1 and
its fitted offset. Public inspection does not reveal affine extrema.

`inspect-raw --artifact MODEL.raw` reports the canonical
`native-raw-complete-v1` charge: framing plus retained state plus native payload.
The inline transport file has that same actual byte total. Generic runtime and
full cold closure remain separately unmeasured until inventoried; they are not
free learned state and do not contribute a fabricated zero. This complete raw
charge does not replace historical `.dpk` tier accounting or certify a release.

`sample-raw --artifact MODEL.raw --rows 100 --seed 91 --out SAMPLES.csv` reloads
the complete artifact and outputs features followed by target in original units.
Feature marginal support remains bounded by the native fit-range sampler.
The successor regression target is sampled before affine inversion without the
historical `[0,1]` target clamp. This preserves an unbounded residual path but
does not qualify tail fidelity, conditional uncertainty or learner transfer.

For a fresh Linux x86-64 worker, run
`dope-kernel sample-raw-isolated --artifact MODEL.raw --rows 100 --seed 91`
with stdout and stderr redirected to write-only files or pipes. After reading
and validating artifact bytes and starting one fixed CPU thread, the process
closes stdin and every inherited descriptor above stderr, then installs a
thread-synchronized default-deny syscall filter. It permits memory, fixed-pool
synchronization, signals, time and output I/O; new file access, network, execution,
mount operations and borrowing another process's memory/descriptors are denied.
The worker inventories and hashes each actual file-backed runtime mapping once
before restriction, then reports its full file-byte cold closure beside the
primary artifact charge. Resident memory and a portable OS image are different
objects. Sampling then produces a deterministic sample digest. This irreversible mode
requires a dedicated fresh process and refuses unsupported platforms. Its
positive and negative fixture proof is local CPU evidence, not an arbitrary-code
isolation claim or protected evaluator deployment.

`inspect-development-pilot --plan PLAN.json
--expected-plan-digest SHA256 [--attempts OBSERVATIONS.json]` consumes the exact
development source-pilot v2 protocol, using the existing dope-finite-json-v1
numeric canonicalization. Each input is bounded to4MiB; jobs/observations to4096.
The digest must be supplied from the trusted published plan, rather than derived
from an edited cap. Its HMAC tokens and owner/registry commitments remain opaque
and unverified without custody's secret. The command reads metadata and emits
the v2 summary: missing/unavailable attempts count as non-success, failed+timeout
remain the separate execution-error numerator, and source counts are retained.
Caller observations are not cost receipts. All four admission flags remain false;
the reader performs no data fetch, pilot execution or scientific admission.

The native evidence library also exposes separate learner-role v2 and calibration
receipt v1 entries. Held-out learner families cannot win development selection.
A matching development-protocol digest authenticates its binding while remaining
unmeasured for N13b, qualified coverage, confirmation and release. Historical
behavior-contract-v1 validation stays separate and retains its overlap refusal.

The separate `FrozenNumericTransform::fit_csv` API fits numeric ranges and binary
label coding only on a development reference. `apply_csv` maps later columns to
that frozen schema, preserving missing-feature states and refusing unknown
binary labels. Its explicitly named min/max transform clamps outside-reference
values; this behavior does not establish tail or geometry calibration. Public
metadata contains a content digest and positional counts, never fitted extrema
or source headers. Historical `load_numeric_csv` and embedding-v1 decoding keep
their existing per-table behavior.

`DatasetEmbedder::embed_csv_frozen` returns a separate frozen-action wrapper with
router bundle/evidence, coordinate schema, candidate order, component offsets and
reference-transform identity bound into its basis digest. The capability records
the actual penultimate/action layer, concatenation, normalization, target/context
and width. A typed unavailable activation may coexist with an available declared
prediction interface. Neither capability metadata nor common coordinates qualifies
transfer, privacy or release.

The native `frozen_geometry_state_v1` entry recomputes the shared finite-JSON
digest of G1's `dope-frozen-geometry-state` body, including encoder, schema and
ordered column/level/range content. `compare_frozen_geometry_v1` validates both
original bodies. Caller labels and supplied digest-only claims are refused;
body commitments and official activation availability remain separate.
# Explicit mixed development profile

`compile-mixed --csv fit.csv --schema schema.json --out model.mixed --profile numeric_split_v1`
fits a separate `DOPMIX01` carrier. `independent_v1` is the matched baseline.
Only declared fitting rows enter the API fit; the CLI treats its entire input
as fitting data. Numeric features retain affine inverses; nominal dictionaries
are exact and column scoped, with at most 32 observed categories per column and
128 total numeric/one-hot coordinates. Missing, unknown category, structural
absence, padding and censored cells remain five distinct classes. Nominal inputs
outside an explicit domain become unknown-category masks, never dictionary
entries. Dictionaries learn only observed fit categories; no hash collisions or
category ordering become numeric distances.

The conditional profile fits one median split per variable categorical/mask
column, choosing among at most eight wholly observed numeric anchors in opaque-id
order by fit-only empirical entropy. Each branch requires four fit rows; absent
supported anchors return `mixed_conditional_anchor_unavailable`. This is a bounded
development model, not a general joint conditional-distribution fit. Fully masked
features, masked targets, nominal targets, text and datetime are unavailable.
Targets must be fully observed continuous regression. Masked feature completion
uses a retained numeric mean or modal category; it is applied before native
target and residual evaluation. Mask indicators are not separate target inputs.

`inspect-mixed --artifact model.mixed` reports complete accounting without private
categories, affine parameters or class probabilities. Framing, all private
dictionaries, transforms, completions, probabilities and splits are charged before
native candidate byte-cap acceptance. Preparation consumes the same declared
deadline as native fitting. Runtime/cold closure remains separate. Existing
DPK3.2, `DOPERAW1`, `DOPERAW2`, default Gaussian and raw CLI behavior stay fixed.

`sample-mixed --artifact model.mixed --rows 128 --seed 1729 --out samples.json`
writes tagged number/category/mask cells, preventing collisions between a nominal
string such as `missing` and a missing-state marker. A fresh Linux worker uses
`sample-mixed --artifact model.mixed --rows 128 --seed 1729 --isolated`; it emits a
digest and runtime accounting after the existing filesystem/network/exec seal.
Redirect its stdout/stderr to pipes or regular files. Samples use only the carrier
and native runtime. Integer/ordinal/boolean repairs are counted; declared physical
domain violations refuse rather than clamp or retry. Public errors remain static.
Privacy, tails, mixed transfer gains and calibration require separate measurements.
