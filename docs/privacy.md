# Privacy and release profiles

`schemas/release-profile-v1.schema.json` defines four independent axes. Rust
`release::profile` owns the state machine and release decision;
`research/transfer/privacy/release_profile.py` is its bounded research mirror.
Local research starts with every axis `unassessed` and no release authority.

| Axis | Positive state | Evidence required at the trusted boundary |
| --- | --- | --- |
| `utility_qualification_state` | `passed` | Exact qualification receipt for the declared utility scope |
| `empirical_privacy_state` | `screened` | Scoped threat-model receipt and qualified attack-power evidence, including controls, coverage, uncertainty and application margins |
| `formal_dp_state` | `verified` | Mechanism, accountant and proof identities, verified budget and complete composition scope |
| `legal_release_state` | `authorized` | Authorization from the organization or rights owner for the exact release scope |

Every axis also supports `unassessed` and `pending`. Utility and formal DP can
be `failed`; empirical privacy can be `failed` or `inconclusive`; legal release
can be `denied`. Utility, byte count, attack simulation, and certification cannot
advance any other axis. Empirical screening describes the measured attacks and
their power within a scope. It establishes no anonymity or de-identification.

## Transitions and evidence verification

An assessment explicitly begins at `pending`. Completing it requires a receipt
for that axis, bound to the current accounting digest, and a trusted verifier.
Rejection moves only that axis to its negative state. An empirical assessment
may instead finish `inconclusive`. Beginning a new assessment clears the earlier
receipt for that axis immediately, including any earlier authorization.
Changing the accounting creates a fresh profile with all axes `unassessed`.

`ReleaseEvidenceVerifier` is an integration boundary for independently trusted
evidence and rights-owner adapters. It must resolve the exact receipt, check its
authenticity and all supplied identities, and verify the scope and substantive
requirements in the table. In particular, empirical power needs independently
qualified controls and policy margins; formal DP needs the actual implemented
mechanism, a valid proof and an accountant covering the complete release history;
legal authorization needs the actual rights owner's decision. A digest, a JSON
state name or a caller-supplied flag establishes none of these facts.

The default verifier rejects every receipt. No product verifier or rights-owner
adapter ships in this task. Fixture verifiers recognize only synthetic test
identities and provide no scientific, legal or release qualification. Decoded
profiles are untrusted snapshots: every view, claim and export decision verifies
their receipts again. Unknown fields, duplicate keys, invalid types, missing
fields or mismatched state/receipt combinations fail closed. Use Rust
`ReleaseProfile::decode` for external bytes; its errors contain fixed codes.

## Shared presentation and export gate

CLI, service, UI and report callers consume `ReleaseProfile::view` (or the
Python mirror's `view`). Export uses that same method with the `export` surface,
which refuses an unauthorized profile. It requires utility `passed`, empirical
privacy `screened`, and legal release `authorized`. Formal DP is optional for an
explicit empirical route: it can remain `unassessed`, and must be described that
way. A pending or failed formal DP assessment blocks export. A request for formal
DP additionally requires `verified`. No state is inferred from a small artifact.

The existing `convert_release` entry has no trusted evidence ingress and now
uses the default profile at this gate before source access or staging. It
therefore refuses export, including an otherwise certified candidate. Wiring an
owner-approved adapter requires a separate integration change binding the exact
output inventory and channels; this implementation does not authorize one.
Conversion reports add `release_profile`; historical reports missing that field
decode to an unassessed profile. Their historical `certified` field describes
the old certification result and supplies no new release authorization.

Labels use an explicit allowlist: `local_research`, `utility_qualified`,
`empirical_screened`, `formal_dp_verified`, `organization_authorized`, and
`scoped_release`. Each affirmative label requires its own verified state.
Automatic `HIPAA`, `anonymous`, `de-identified`, and `safe-share` labels are
rejected even when all four axes are positive. An epsilon request requires a
verified formal receipt with the same epsilon, mechanism, accountant, proof and
delta. It does not create a mechanism, allocate a privacy budget or implement a
CLI `--epsilon` option. Unsupported labels and errors never reproduce the input.

## Observable channels and historical evidence

Accounting commits separately to the recipient-required artifact inventory,
release history, independently source-derived lengths, statuses and metrics,
and the total release count. Include failures, repeated artifacts, sidecars and
earlier releases. Even a channel with no observations needs an explicit empty
inventory commitment. These commitments identify the scope; independent review
must establish completeness and applicable attack/composition accounting.

`DisclosureThreatModel` adds optional commitments for the three observable
channels. Missing commitments produce `release_channels_unaccounted`. Repeated
releases continue to disable the existing independent single-release bounds;
this task supplies no new repeated-release inference method. Existing bounded
attacks remain `inconclusive`, formal/legal aliases remain `unverified`, and
`release_authorized` remains false. New calibrated measurements include the
aligned local inconclusive profile. `privacy-evidence-v1.schema.json` accepts
historical measurements without that optional field and constrains it when
present. It is an extension of the existing evidence format.

Public profile views contain fixed states, fixed labels, public budget parameters
and digest commitments. They contain no source values, column headers, extrema
or private paths. The implementation adds no protected-source access and leaves
the unresolved custody incidents unchanged. Fixture validation commands are
`cargo test --locked --lib release::` and
`PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_privacy -v`.
Neither these fixtures nor the schema certify a dataset or permit publication.
