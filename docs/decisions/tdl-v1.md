# TDL candidate protocol and integration decision (I00 / C00)

This is an engineering decision for the unified workplan. It does not freeze a
scientific protocol, certify a method, admit a native model, or authorize release.
The common base is main `2bfee82e05c264bdfb8ff26049a560a64d984817`, tree
`da7eeabcd709605d8eb1cc0f933b136309a09620`.

## One branch, one draft PR

Per human decision D01, `dope/unified-spec-workplan-v2` is the single integration
branch against main. Independently reviewable internal commits replace all source
proposals for stacked or feature PRs. The coordinator retains the full requirement
disposition and admits subsequent commits through exclusive checkout/path leases.

The order is I00 integration scope and C00 candidate identities/schemas, followed
by the admitted semantic, preparation, codec, worker, product and evidence tasks,
then final validation. This initial scope implements only I00 and C00. Dependency
readiness does not grant a write lease for later tasks. The existing integration
checkout remains under custody HOLD D18; no protected state is opened or moved.

Owner/test maps register the new paths. Rust owns candidate admission in the
existing `metrics` module; Python tests independently check JSON Schema and hash
goldens. This adds no parallel scorer, fitted-transform type or worker registry.

## Successor and historical boundary

The successor is
[`contracts/tdl-contract-v2.candidate.json`](../../contracts/tdl-contract-v2.candidate.json).
`contracts/tdl-task-1.json` is **withdrawn as a file path**. That name exists solely
as a historical/semantic alias for the candidate successor; no file or redirect is
created at the withdrawn path.

Historical `contracts/tdl-contract-v1.json` remains read-only, 4820 bytes, SHA-256
`281923c33da12e218d54c2226d138bbea2a22d86936bae5c3d7edfbfaf2c63ab`.
Historical schemas, generic `Margin` calculations, DPK3 decoding and receipt
readers retain their original semantics. A historical receipt carries no
certification into the successor. Candidate admission rejects a v1 receipt by
default. Explicit legacy replay uses the existing v1 component-case reader and
returns a separate `LegacyReplay` mode, never candidate qualification.

Candidate transfer adopts paired harm, at tau = 9/10:

```text
z = L_S - 0.10*L_0 - 0.90*L_R
```

Losses are paired on the same query rows in original target units (squared error
for regression; log loss for classification). Transfer qualification will require
a simultaneous one-sided upper bound at most zero. Raw transfer is not clipped.
An opposite-sign benefit receipt uses `-z` and a lower-bound rule. It cannot be
silently relabeled as harm. Migration requires explicit contract/orientation and
unit bindings plus re-evaluation of paired losses; a generic historical `Margin`
does not prove those bindings. C00 defines admission only; later semantic tasks
implement candidate inference, drivers and frontier reduction in the existing
metric core.

## Typed identities and canonical bytes

The protocol and result schemas are
[`tdl-protocol-v1`](../../schemas/tdl-protocol-v1.schema.json) and
[`tdl-result-v1`](../../schemas/tdl-result-v1.schema.json). Their **schema** version
does not turn a historical v1 **contract** into the candidate v2 contract.
Objects are closed and nested fields required. Identity kinds are `task`,
`method`, `panel`, `artifact`, `campaign`, and `release_profile`; each result entity
contains `kind`, `bindings` and `id`. Binding fields are fixed by
`identity_fields` in the protocol. Inputs use content digests instead of source
names, headers, values or private paths. Artifact identity includes both payload
and sampling-closure hashes, task, method and protocol. Campaign identity binds
protocol, panel, schedule and source roster. A release profile binds policy,
rights and privacy-profile hashes.

`dope-canonical-json-v1` is a versioned encoding, not RFC 8785:

1. Parse strict JSON: duplicate keys, nonfinite numbers and invalid syntax fail.
2. Canonical numbers are integers in `[-(2^53-1), 2^53-1]`. Floating-point tokens
   (including `1.0`) fail. Noninteger quantities use explicit
   `{numerator, denominator}` objects with positive denominators. There is no
   implicit fraction reduction; the declared representation is bound.
3. Sort object keys by Unicode scalar order, retain array order, encode compact
   JSON without optional whitespace, escape JSON controls/quotes/backslashes and
   emit other Unicode as UTF-8. Do not normalize Unicode.
4. Compute `kind + ":" + lowercase_hex(SHA256(prefix || canonical_payload))`,
   where prefix is UTF-8 `dope.identity.v1`, NUL, UTF-8 kind, NUL.

For `protocol_id`, the payload is the entire protocol except `protocol_id` itself.
It binds workload/source rosters, development/confirmation roles, eligible models,
loss definitions, transforms, seeds, byte grid, inference, multiplicity,
release-profile roster, applicability and comparison plan, including explicit
unresolved nulls. Resolving a null changes the identity. Task/method/panel base
identities avoid circular protocol references; artifacts and campaigns bind the
protocol explicitly. Cross-entity references and panel consistency are checked.

Shared fixtures in `tests/fixtures/tdl-v2/` include Unicode and escaping hash
goldens, the unfrozen admission example, counterexamples and an explicit legacy
replay wrapper. Synthetic fixture digests are not measurements. Rust uses a
closed local schema vocabulary with fixed source-free errors; Python's
Draft 2020-12 validator supplies an independent test oracle. Neither path is an
externally extensible schema engine or a scientific scoring implementation.

## Independent state axes

No axis implies another. A successful schema-admission result verifies shape,
identity and references; it does not establish the truth of declared states.
Every candidate result remains `citable=false` and `certified=false`.

| Axis | Meaning |
| --- | --- |
| `artifact_qualification_state` | measured artifact qualification or an explicit unavailable/failure disposition |
| `scheduled_cell_state` | planned, running, complete, failed, timed out, unsupported, over budget or missing execution |
| `campaign_completeness_state` | completeness of the scheduled campaign, independently of an artifact's result |
| `empirical_privacy_state` | measured empirical privacy evidence, independently of formal DP |
| `formal_dp_state` | separately evidenced formal differential privacy claim |
| `release_authorization_state` | separate legal/rights authorization |
| `claim_provenance_state` | provenance verification for a claim |

The protocol enumerates every allowed value; unknown values fail closed. Privacy
attack results never establish formal DP or HIPAA de-identification. A complete
campaign never establishes artifact qualification or release authorization.

## Unresolved decisions remain null

The candidate is strictly `status="candidate"`, `protocol_frozen=false`,
`certified=false`. It cannot be frozen or certified through this schema even if
someone fills a roster. An admitted future schema revision and evidence are
required before any scientific freeze.

Under D05 HOLD and CR-09, all three real-reference gate parameters (`absolute`,
`relative`, `lcb_required`) remain null. Driver stability, absolute importance and
synthetic noncollapse floors also remain null. These are unavailable parameters,
not zero floors or disabled gates. Required rosters, seeds, estimand, independent
unit, small-cluster abstention, calibration tolerances, release profiles,
applicability and comparison plan remain explicitly unresolved.

D19 is `DECISION_HOLD`: `multiplicity.procedure` and `multiplicity.family` remain
null. The fixed-artifact conjunction of all required constraints is recorded as
intersection-union; it does not select a procedure for simultaneous inference or
selection. The schema rejects a guessed procedure. The inference plan separates
at least 10,000 bootstrap draws from at least 2,000 independent Monte Carlo
calibration replications; neither count substitutes for the other.

No candidate result is currently measured or certified. Calibration, native
qualification, rights, privacy and protected-custody acceptance remain separate
gates after this engineering change.
