# Jankurai audit and exceptions

The full v1.7.1 audit is a required PR gate: score at least 85,
`decision=pass`, zero applied caps, and zero high or critical findings. Its
JSON and Markdown reports are retained even when the gate fails. The audit
must cover tracked product source; ignored `runs/` output is not product code.
`agent/baselines/main.repo-score.json` preserves the original 1.6.11
read-only report as a ratchet comparison; the required PR decision is the
clean full v1.7.1 scan with the 85-point floor. The historical scan included
ignored generated data, so its fingerprints are triaged against the clean
tracked-source result instead of being treated as new product findings.

The Python V1 boundary is a dated advanced-data compatibility exception until
2027-09-29. It preserves the old CLI and 37 compatibility tests while Rust
owns V3 compilation, release decisions, and contract truth. The boundary
receipt includes current SHA-256 hashes and check results and is regenerated
before every audit. It cannot excuse missing tests or a failed builtin check.

The offline research benchmark has a separate dated advanced-data exception.
Its Python adapters run pinned author ML libraries for comparison and do not
serve product requests, own a database, or set product release truth. The
audited runtime boundary applies to its isolated adapter worker, which does not
spawn processes. Its proof checks all tracked benchmark Python source, runs the
benchmark contract tests, checks that final public-test admission fails closed,
and rejects product imports or routes. The dispatcher and other benchmark code
remain subject to the full audit; unsafe deserialization is never reclassified.

The private landmark preparation utility has a separate, dated advanced-data
exception under `research/landmark_cache/`, expiring 2027-03-01. It prepares
TRAIN-only inputs for the native Rust landmark executable and transports private,
hash-bound measurements; it has no product import, scorer, data-admission or
certification authority. Its source remains fully scanned. Offline contract tests
check TRAIN-only source access, cache integrity, resource limits and dashboard
redaction. This exception does not suppress unsafe code or waive required checks.

The audit job's strict security wrapper requires Cargo advisory and workflow
lint tools. The separate required CI security job runs the SHA-verified pinned
Gitleaks binary and SPDX SBOM action; those checks are independent of the wrapper's
local optional binaries. A release profile requires all four binaries.
`tools/security-lane.sh` is the canonical Jankurai wrapper and delegates to
the same scanner script used by local `just security` and the CI security job.

The frozen `legacy_*` candidate IDs are wire identifiers in historical
receipts. They are retained only for decode and comparator compatibility; no
new implementation may use their names as a substitute for measured evidence.
For each of the 83 baseline findings, `agent/emit-audit-receipts.sh` records
its fingerprint, source location, disposition, current rule fingerprints, and
rerun proof in `target/jankurai/repair-log.json`. The 54 original wording
findings were reviewed in context: launcher deletion and broad process killing
were removed; early stopping and prior coefficient variables were renamed;
atomic write paths are now named for staging; the V1 score probe was renamed.
The preserved `legacy_*` IDs are historical wire values. The remaining
literal terms describe checked lease states, explicit unsupported candidates,
or a compatibility deserializer, with locked behavior tests. Current shape,
build speed, and observability advisories remain visible with their own
fingerprints and proof commands. An advisory does not override the score gate.

## Native learner API identifier

The `dead_language.allow_terms = ["dummy"]` entry uses the pinned auditor's
platform API mechanism for scikit-learn's public `sklearn.dummy.DummyRegressor`.
The constant-null adapter calls that installed SDK with `strategy="mean"`; its
module origin remains hash-pinned. The second runtime occurrence requires the
same exact module in the registration pins. Renaming or disguising that module
would obscure the native dependency. This exception covers the external API word,
is owned by external-evidence, and expires on 2027-03-01; remove it when the pinned
auditor recognizes exact external SDK identifiers. It does not permit simulated
learner evidence or change any conformance, privacy, source, security, score,
required-check or protected-merge gate. The original failing PR291 audit remains
retained. The constant-null tests and native four-case qualification distinguish
fixture calls from actual native fit/predict execution.

Local `.agent/` tools are ignored development artifacts under the repository guide.
The full scan adds only that local prefix to its existing exclusions; native
`git ls-files -- .agent` must remain empty. No tracked source is excluded. The
first local audit included two ignored budget/launcher helpers and is retained as
a failed local observation. All three boundary checks passed, including1,577
benchmark tests, before that scanner failure. Unchanged boundary-source hashes
and original logs may support a later full scan; that is explicit equivalence,
not a claim that the old test command executed on a later commit.

The platform-word allowance is repository-wide at the auditor's word granularity.
Its intended use is the two external API references described above; review must
reject unrelated uses. Its recorded expiry is a maintenance obligation, not an
assertion that this auditor automatically enforces that field.
