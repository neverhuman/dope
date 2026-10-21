# Pinned audit remediation, 21 October 2026

The historical receiving-main audit is run `39661927770`, source
`36cf658a16b638c3a6a08140cbf65ea5eefccb73`, artifact `12377525854`
(`jankurai-v1.7.1`). Its downloaded ZIP matches the server SHA-256
`65f6f0caba7aac1df0050973d9fcc48bc7512aa63b2216053326559c721bb4f1`.
The enclosed repo-score JSON SHA-256 is
`a39be8618a00ff82c30792c3714722fc57c177d01315d63853f5c4fed07a3976`.
It reports 88/85, no applied caps, and three medium findings. These identities
are historical evidence only, not a result for the remediation source.

Retrieve with `gh run download 39661927770 -n jankurai-v1.7.1` and verify the
individual file hash. The server archive digest applies to the ZIP, not to a
repacked directory. Keep raw logs with `gh run view 39661927770 --log`.

## Changes and retained findings

The transfer-metric production primitives and their golden/invariant fixtures
now occupy separate modules. Public paths, arithmetic, test names and the
immutable golden input stay unchanged. `just focus transfer_metrics::tests`
executes those fixtures and refuses an empty or ignored-only result. Focused
proof retains commands, raw output and source hashes in `target/focus/` and
reuses the existing Cargo target/compiler wrapper. The redundant preliminary
`cargo check` is removed from this narrow lane; full `just check` remains required.

The engineering policy paragraph is now linked to the exact primary-human
receipt and describes the actual evidence required for each head. Its original
finding fingerprint is
`sha256:c811d15b2a74ef124155c091d2998156df4c4c56bbec14636677a8dd181815a3`.
The detector matched a prohibition on false attribution, not an assertion of
completed proof. The policy does not grant scientific independence or release.

Two pinned scoring limitations remain visible; neither justifies changing the
85-point gate, adding exclusions, or substituting a different auditor:

- Build-speed fingerprint
  `sha256:2f2531223d7f7036c20d44b58cd52e64aa53ffd6cb85e01e541c1feff0c09cb2`:
  the generic bonuses total at most 80 (20+20+10+10+10+10). The remaining
  bonuses require Jankurai-specific package/UX commands. DOPE cannot honestly
  invoke those absent packages. Proposed upstream correction: recognize actual
  package-scoped tests and measured cached proof without tool-repository names;
  add an adopter fixture with locked dependencies, cache and nonempty narrow tests.
- Shape fingerprint
  `sha256:e1035572a3e8a39ce5ed0de9221476f74c3f878b4e5feb8ab446f0bb75c8470c`:
  the diagnostic excludes exempt Python when reporting the largest product file,
  but the 350-line bonus uses the unfiltered file set. A single-file split cannot
  establish that repository-wide predicate. Proposed upstream correction: use
  the same product scoring population for both the evidence and bonus. Remaining
  sizeable Rust modules are not certified clean by this extraction.

The exact rules are in the v1.7.1 family lock: analyzers commit
`6cd55512e491a029397afb5e4e51531c9bffe203`, paths
`crates/jankurai-audit-analyzers/src/audit/analyzers/{speed,shape}.rs`;
review detector in kernel commit `cd94f46d3b4ba3b60780b996ed5f7d1f559dfaef`,
`crates/jankurai-audit-kernel/src/audit/scan.rs`.
The detector does literal substring matching without checking negation or linked
receipts. An upstream fixture should distinguish a policy prohibition from an
unsupported completed-review assertion. No local rule is disabled.

The baseline copy report contains 142 warning classes but serializes only its
first 50. A complete inventory requires `jankurai copy-code . --max-findings 10000`;
the default report alone cannot establish disposition of all classes. Historical
receipt producers, frozen adapter source closures and legacy wire algorithms
must not be deduplicated blindly: changing their dependency closure can invalidate
evidence. Per-class source excerpts, hashes and routing are retained with the
remediation handoff. Unresolved advisories remain open, not silently waived.

## README badge and complete advisory retention

The [per-class inventory](evidence/jankurai-copy-dispositions-20261021.json)
retains every class fingerprint, source locator/hash, specific disposition and
proof route. These are static dispositions, not claims that each mapped command
executed. Standard CI now retains the complete copy report rather than its first
50 classes. The native Rust witness tool's historical empty interface/implementation
hashes and zero file count are not evidence for this repository's custom Rust
layout; Cargo and the explicit source manifests carry that proof obligation.

`ops/merge_guard/audit_proof/badge.rs` validates the pinned version, full scope, exact checkout
head, two-hour report freshness, score, decision, caps and hard findings. Failed
reports produce red badges; invalid reports produce an unverified gray badge and
a failing command. Dirty local scans are gray. No input text is interpolated into
SVG labels. The source/report SHA binding is uploaded beside the SVG.

The README embeds a deterministic committed score/status snapshot. Each audit CI
run generates its own SVG from the actual report, checks it against that snapshot,
and uploads both even on failure. A changed score/status requires an ordinary PR
update; CI never commits, writes a Pages site, or receives publication credentials.
The badge links to the live audit workflow. An old committed image is not evidence
that a pending or failed new run passed. Reproduce generation with
`target/debug/audit-proof badge --report target/jankurai/repo-score.json --expected-head "$(git rev-parse HEAD)"`.

The first candidate scan correctly capped new Python repository tooling at 72.
Those proposed proof/badge helpers were migrated to the existing Rust tooling
package; their failed report and superseded source remain in the handoff. No
new Python tooling root or language-policy exception was introduced. Build the
CLI with `cargo build --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin audit-proof`.
