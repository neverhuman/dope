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

## Formal DP profile and complete private-read ledger

`schemas/dp-profile-v1.schema.json` is a closed Draft 2020-12 design profile.
`research.transfer.privacy.accountant.account` evaluates its complete ledger.
The trusted caller supplies `expected_adjacency`: `unit` (`row`, `entity`, or
`user_group`), `relation` (`add_remove` or `replace_one`), and the pre-specified
`max_rows_per_unit`. Every private mechanism must prove its sensitivity for
that exact contract. Row privacy never silently becomes entity or group privacy;
a unit, relation, or contribution-cap mismatch forces `formal_dp_status=not_dp`.
Contribution caps and sensitivity bounds require proof, not observed row counts.

The profile pins clipping norm/bound, sensitivity, noise scale and implementation,
the accountant, total public `(epsilon, delta)` budget, complete release history,
cryptographic PRNG implementation, and floating-point discretization assumptions.
Secret DP noise randomness must use a cryptographic PRNG and remain secret; a
public deterministic sampling seed cannot supply it. No noise seed is accepted
in this closed profile or emitted in the accounting report.
Bounds are either strictly public and fixed before source access or obtained by
an accounted DP mechanism. The accountant never discovers source extrema or
quantiles. Supplied bounds, noise labels and cost numbers are proof obligations;
their schema validity does not establish the implemented mechanism's guarantee.

The ledger explicitly covers all seven decision stages, including unused stages
with evidence of a public pre-specified choice:

| Stage | Private dependency that must be charged |
| --- | --- |
| `domain_discovery` | Sample min/max, category discovery, missingness discovery |
| `binning` | Sample quantile edges or other data-dependent bins |
| `winsorization_clipping_bounds` | Sample extrema, quantiles or learned clip bounds |
| `transform_fitting` | Learned scaling, imputation, feature or coordinate transforms |
| `structure_selection` | Structure search or model/hyperparameter choices |
| `parameter_fitting` | Every private fit, refit or sufficient-statistic query |
| `candidate_selection` | Private scoring, validation or selection, including rejected candidates |

Four further mandatory entries account for observable channels and later reads:
`source_derived_lengths`, `source_derived_statuses`, `source_derived_metrics`,
and `post_summary_private_reads`. An unused channel needs an explicit pinned
public constant/empty specification. A source-dependent artifact length, failure
status or metric requires a charged mechanism; private access after a DP summary
does not inherit the summary's guarantee. Every such event is composed.

Each step declares `public_prespecified`, `accounted_dp`, or `unaccounted_private`.
Public choices need pinned provenance; a private mechanism needs pinned proof
evidence and a positive cost in the selected accountant. Public clipping bounds
only exempt the choice of bounds: private fitting still needs its mechanism cost.
An accounted private bound also needs a charged clipping-bound stage. A non-DP
min/max scan, quantile query, structure search or candidate selection is an
`unaccounted_private` read, even when its result is never separately published.
One such read immediately forces `not_dp`, overriding an audit flag or empirical
screen. Missing stage coverage, incompatible costs and excess budgets also refuse.

## Composition, audit signoff and integration boundary

All steps use the same adjacency and accountant. RDP adds per-order epsilon
vectors at common orders `alpha > 1`, then minimizes
`sum_rdp(alpha) + log(1/delta)/(alpha-1)` over the declared grid. See
[Mironov, Propositions 1 and 3](https://arxiv.org/html/1702.07476v3).
zCDP adds `rho`, converting with `rho + 2*sqrt(rho*log(1/delta))`; see
[Bun and Steinke, Proposition 1.3 and Lemma 2.3](https://arxiv.org/abs/1605.02065).
Both conversions require positive delta when private mechanisms are present.
The `pld` route accepts independently proved per-step privacy-loss envelopes
`(epsilon, delta)` and adds both coordinates conservatively by sequential
composition. It does not implement PLD convolution or claim tighter PLD bounds;
see the composition discussion in [Mironov](https://arxiv.org/html/1702.07476v3).
No route infers subsampling amplification, disjointness or independence. Sums
and conversion operations use upward floating-point margins; that precaution
does not replace an independent numerical/discretization proof audit.

Supply every adaptive query, failure, retry, rejected candidate and earlier
release in the ledger, bound to `release_history_sha256`. Costs must hold
conditionally for each adaptive step. Reassessing a truncated ledger cannot
reset a real privacy budget. This declarative module performs no source reads
and cannot detect accesses omitted or falsely marked public by external code.
The trusted producer and independent audit must establish complete coverage,
public provenance, contribution enforcement, cost derivation and history.

Even when `composition_passed=true`, verification requires all of:
`independent_cryptographic_audit=independently_audited`, the explicit trusted
`independent_audit_passed=True` argument, and a trusted `audit_verifier` returning
exactly `True` for an authenticated independent proof signoff. The verifier
receives the full profile digest, accountant and schema code digests, and composed
budget. The profile digest covers every step, history, bounds, implementation
and discretization pin. The default verifier denies; a JSON status or boolean
alone cannot pass. Signoff must check actual PRNG security, finite precision
noise, sensitivity and composition of the implemented mechanism, not fixtures.

Without signoff the result is `unverified_design`. `empirical_screened_only`
always leaves `formal_dp_verified=false`, including with a fixture audit adapter
(D15). Refusal reasons and exceptions are fixed codes. Reports expose only
states, public budget parameters and a profile digest; errors and reports never
copy source values, headers, extrema, private paths, callback errors or the input
profile (CR-04). No result grants legal authorization or sets
`release_authorized=true`.

This is a draft S02 interface for S00/D02 integration. S00's trusted
`ReleaseEvidenceVerifier` must independently bind a substantive formal receipt
to the release accounting, mechanism, accountant, proof and `(epsilon, delta)`;
this module does not populate or promote S00 profiles. D02's source-manifest
admission must supply the protected hierarchy, pre-specified bounds provenance
and source rights/custody context without copying its private mapping into this
profile. The integrator must also bind S00's complete observable-channel/history
accounting. No trusted product audit adapter or noise sampler ships here. Unit-test
audit callbacks recognize synthetic identities only. Deterministic validation:
`PYTHONPATH=. python3 -B -m unittest research.transfer.tests.test_dp_accountant research.transfer.tests.test_privacy -v`.
