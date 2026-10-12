# Release procedure

The version source is `Cargo.toml`; ordinary binaries remain
`0.3.0-alpha.1`. This branch is a PR implementation, not a certified release.
Document release changes in `CHANGELOG.md` before a release candidate.

1. Run `cargo test --locked --test vnext_custody`, `just check`, `just security`,
   and `just score` on a clean tracked-source
   checkout. Keep JSON, Markdown, SBOM, and scan receipts.
2. Verify the v2 contract digest, v1 historical fixture, DPK3.2 decoding and
   byte-identical full-model encoding, DPK3.3 compact decoding, and exact byte
   ceilings of 10,240 for L3 and 32,768 for L2.
3. Select once using development data and the compiler's declared train-only
   proxy. This is not decoded learner-transfer selection or certification.
   Freeze the selected artifact and all evaluation parameters before any final
   access; do not choose a runner-up if the final audit fails.
   Freeze source, package, environment, corpus split, CUDA inventory, notices,
   and checkpoint manifest. The source commit and signed tag bind the release
   lineage. The v1 receipt path remains read-only.
4. Run all six frozen auditors and privacy attacks on the sealed holdout. PTF-v1,
   joint fidelity, applicable FI gates, membership/attribute/copy privacy, and
   file inventory must pass for the frozen selection. The owner evaluates the
   exact predeclared batch in order, without ranking or calling the optimizer.
   External exact-library claims require separate hashed evidence.
5. Publish only the staged allowlisted bundle after required CPU CI and the
   scheduled GPU lane pass. Store manifests, hashes, certification, and the
   full Pareto vector with the release.

The launch checklist records: security scan and SBOM hashes; a backup of the
last signed passing manifest; monitoring locations for the ledger, audit, and
GPU report; the tested rollback manifest and restore command; and abuse-control
proof for privacy gates, input rejection, size caps, and inventory validation.
Treat any absent class as a failed launch gate. The checklist is tied to the
source commit and contract digest, not to a tier name alone.

## Protected evaluator deployment

`convert_release` reads only the development `train.csv` during selection. Its
`real_holdout_dir` argument now names the directory containing `evaluator.sock`,
rather than a directory of readable final CSVs. Set `DOPE_EVALUATOR_UID` to the
independently trusted owner UID. The producer must be non-root and use a different
UID. `EvaluationCapability` authenticates the live endpoint with `SO_PEERCRED`;
it cannot be reconstructed from a cache, array, report, or serialized role flag.
A same-principal endpoint, symlink, or absent endpoint fails closed.

Run the owner endpoint in a dedicated Linux x86-64 process using
`evaluation::serve_frozen_batch(&listener, &private_root)`. The operator binds a
Unix listener at the shared endpoint; only approved producer principals should
have socket access. The private root must belong to the owner, have mode `0700`,
and contain owner-only regular files (no symlinks or hardlink aliases):

- `train.csv` and `test.csv`, with the exact bytes committed by the declaration.
- `protocol.json`, a fully resolved, schema-valid frozen behavior contract.
  The tracked draft stays unfrozen and is refused for protected evaluation.
- `declaration.json`, an `EvaluationDeclaration` declaring the producer UID,
  confirmation ID, ordered batch hash, protocol/fit/query hashes, maximum attempts
  and query-open budget. Freeze this declaration before starting the endpoint.

Use `frozen_batch_hash` over the `FrozenSelection` objects produced by
`SelectionView::freeze` to declare the batch. The private frozen object binds
canonical artifact bytes, development receipt, fit bytes, compiler/evaluation
configuration and evaluator implementation/build identity. The producer retains
its private `candidate-evidence/frozen-selection.json` before sending a request.
The endpoint receives artifacts and configuration; protected tables stay in the
owner process. It accepts no executable models, input arrays or callbacks.

The evaluator installs a thread-synchronized seccomp filter that denies new
sockets, outbound connections and program execution. Deploy with no inherited
network connections or unneeded descriptors, a restricted mount inventory, and
an operator-enforced wall-time/resource limit. Give the producer no privilege
escalation route to the owner and no readable copies, archives or cached labels.
A byte commitment cannot undo prior label exposure; fresh-source eligibility
and deployment approval remain separately measured prerequisites.

The owner uses an exclusively locked, owner-only append ledger named by the
confirmation ID. Every attempted query open is charged and fsynced before label
access. Successful opens and terminal completion are actual events. Reports use
the measured successful-open total, including earlier interrupted attempts.
A completed batch is terminal even when certification fails. Crash/interruption
retries must preserve the exact batch and declaration and stay within both
attempt and query budgets; prior artifacts, reports and access charges remain.
A damaged or partial ledger blocks reuse. Retain this root independently of
producer caches and keep one authoritative ledger namespace per confirmation ID.

The custody test launches an owner subprocess with `sudo -n` using only synthetic
files under `target/custody/`. It checks UID separation, protected aliases/copies,
array/archive boundaries, callback rejection, network/exec denial, label
permutation, terminal failed audits and interruption budgets. Run it as a
non-root producer with the fixture launcher available; production does not invoke
sudo or launch an evaluator from the producer. Real owner deployment, signed
protocol custody, mounted sealed corpora and scheduled GPU qualification are
not established by these fixture tests. Missing frozen CPU/GPU auditor families
continue to block release.

## Rollback

If promotion fails, retain the candidate evidence directory for repair and
leave the public output path absent. For a published failure, withdraw the
candidate bundle, restore the last signed passing manifest and its exact
allowlisted files, then verify every restored hash against that manifest.
Record the withdrawn and restored digest in the incident log. An old receipt
keeps its original contract and tier lineage. Formal DP and HIPAA
de-identification are unsupported claims.
