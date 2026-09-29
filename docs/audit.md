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

The audit job's strict security wrapper requires Cargo advisory and workflow
lint tools. The separate required CI security job runs the pinned Gitleaks
action and SPDX SBOM action; those checks are independent of the wrapper's
local optional binaries. A release profile requires all four binaries.
`tools/security-lane.sh` is the canonical Jankurai wrapper and delegates to
the same scanner script used by local `just security` and the CI security job.

The frozen `legacy_*` candidate IDs are wire identifiers in historical
receipts. They are retained only for decode and comparator compatibility; no
new implementation may use their names as a substitute for measured evidence.
For each audit finding, record its fingerprint, source location, disposition,
and rerun proof in `target/jankurai/repair-log.json`. Advisory exceptions need
a specific technical reason and a passing compatibility fixture.
