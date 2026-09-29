# Release procedure

The version source is `Cargo.toml`; ordinary binaries remain
`0.3.0-alpha.1`. This branch is a PR implementation, not a certified release.
Document release changes in `CHANGELOG.md` before a release candidate.

1. Run `just check`, `just security`, and `just score` on a clean tracked-source
   checkout. Keep JSON, Markdown, SBOM, and scan receipts.
2. Verify the v2 contract digest, v1 historical fixture, DPK3.2 decoding and
   byte-identical full-model encoding, DPK3.3 compact decoding, and exact byte
   ceilings of 10,240 for L3 and 32,768 for L2.
3. Freeze source, package, environment, corpus split, CUDA inventory, notices,
   and checkpoint manifest. The source commit and signed tag bind the release
   lineage. The v1 receipt path remains read-only.
4. Run all six frozen auditors and privacy attacks on the sealed holdout. PTF-v1,
   joint fidelity, applicable FI gates, membership/attribute/copy privacy, and
   file inventory must pass. Select the smallest exact artifact among certified
   candidates. External exact-library claims require separate hashed evidence.
5. Publish only the staged allowlisted bundle after required CPU CI and the
   scheduled GPU lane pass. Store manifests, hashes, certification, and the
   full Pareto vector with the release.

The launch checklist records: security scan and SBOM hashes; a backup of the
last signed passing manifest; monitoring locations for the ledger, audit, and
GPU report; the tested rollback manifest and restore command; and abuse-control
proof for privacy gates, input rejection, size caps, and inventory validation.
Treat any absent class as a failed launch gate. The checklist is tied to the
source commit and contract digest, not to a tier name alone.

## Rollback

If promotion fails, retain the candidate evidence directory for repair and
leave the public output path absent. For a published failure, withdraw the
candidate bundle, restore the last signed passing manifest and its exact
allowlisted files, then verify every restored hash against that manifest.
Record the withdrawn and restored digest in the incident log. An old receipt
keeps its original contract and tier lineage. Formal DP and HIPAA
de-identification are unsupported claims.
