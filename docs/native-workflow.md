# Native raw-unit workflow

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

The native evidence library also exposes separate learner-role v2 and calibration
receipt v1 entries. Held-out learner families cannot win development selection.
A matching development-protocol digest authenticates its binding while remaining
unmeasured for N13b, qualified coverage, confirmation and release. Historical
behavior-contract-v1 validation stays separate and retains its overlap refusal.
