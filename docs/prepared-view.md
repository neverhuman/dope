# PreparedView v1

`PreparedView` is the canonical external name and `prepared-view-v1` is its
only wire format. `PreparedTable` is a deprecated Rust migration alias for
the same type; it does not introduce an alternate wire spelling or parser.
The private payload follows [the Draft 2020-12 schema](../schemas/prepared-view-v1.schema.json).
Every object is closed and every property is required, including nullable
bounds and rights evidence. Native semantic validation is also mandatory.

This contract describes a prepared fitting view for a declared task, horizon,
split, entity set, and frozen feature map. It does not promise compression of
arbitrary future tables. Source-name changes, input declaration changes, and
independently fitted synthetic-arm maps require new admission.

## Explicit boundaries

The Rust entry points live in `typed_data::prepared`:

1. `prepare(&TypedTable, fit_rows, PrepareSpec)` is the explicit raw operation.
   Only the ordered, nonempty, unique fitting row indices contribute to fitted
   offsets, scales, category dictionaries, the fit digest, and returned rows.
   No validation/query/holdout row participates in learning those parameters.
2. Retain `view.freeze()` as a trusted `PreparedContract`. When loading an
   external view, establish its manifest/provenance independently before
   freezing it. Parsing alone establishes shape and consistency, not trust.
3. `contract.apply(&TypedTable, rows)` applies that retained map to another arm
   or split without fitting a scaler, dictionary, or target mapping. It checks
   the raw declaration digest before transforming. Selected rows must be unique.
4. `validate_input(bytes, &contract)` parses prepared bytes and checks them
   against the frozen metadata. `PreparedView::validate()` alone checks local
   consistency; it cannot establish agreement with the experiment's frozen map.
5. `contract.compile_input(&view)` returns `PreparedCompileInput` only after
   full validation against that contract. It accepts only a `PreparedView`,
   performs no fitting or conversion, blocks rights state `blocked`, and blocks
   multiclass qualification. Consumers of this boundary take the admitted type
   and read `input.view()`; there is no raw-table conversion into that type.

D00 adds this library boundary. Existing legacy raw workflow/CLI paths are
unchanged; command routing and downstream compiler adoption must use these
entry points when integrated. Historical DPK3.2 and v1 receipts retain their
existing decoding. No new release, science, or custody gate is satisfied by
these engineering fixtures.

## Metadata and values

`task` is individually declared as `regression`, `binary`, or `multiclass`.
`horizon` contains positive `steps` and an explicit `rows` or `seconds` unit;
there is no inferred horizon. Column declarations list features followed by
exactly one target. Column IDs are opaque numeric integers `1..N` in that order.
`source_id` is the private opaque raw-schema ID, never a source name. Prepared
types are numeric: `continuous`, `integer`, `boolean`, and `category_code`.

`group_map` is an ordered partition of the feature IDs. Group IDs are positive,
unique opaque integers. Members are nonempty, cannot repeat or reference the
target, and cover every feature exactly once. Group and member order are part
of the frozen map; comparisons never sort them to hide a mismatch.

Rows contain finite numbers in exact column order. Features use the recorded
affine map `(x - offset) / scale`, identity map, or zero-based category index.
Numeric integer/ordinal source features become continuous affine coordinates;
booleans remain `0`/`1`. Categories use fitting-row first-occurrence order,
retained privately as an exact dictionary. All numeric bounds are in original
source units; null means no declared bound. Native validation checks bounds in
the corresponding affine coordinate to preserve representable endpoints.
Leaving the fitted range is allowed when declared support permits it.

Regression target rows stay in original units. The target column still records
its fitted affine offset and positive scale for workers that explicitly use
normalized targets. `regression_prediction(z)` returns `z * scale + offset`
without clipping, including extrapolated predictions. Do not inverse-transform
a prediction that is already in original units. Target units must equal the raw
declaration, with explicit `unspecified` when no units were declared.

Discrete target rows contain class IDs, not normalized scores. `class_order`
defines the exact probability/adapter coordinate order: exactly two distinct
`u32` IDs for binary, or 3–4096 for multiclass. The raw adapter accepts numeric
or boolean labels equal to those IDs. Nominal target strings require a separate
explicit label adapter; this version rejects them. Every target row must belong
to the frozen class list. No class sorting or task inference is performed.

Version 1 policy values are fixed: `missing=reject`,
`category=fit_dictionary`, `oov=reject`, `support=declared_bounds`, and
`saturation=none`. Raw masks, missing values, unseen categories, invalid numeric
types, and values outside declared support are rejected. No imputation,
category growth, saturation, or synthetic-arm refitting occurs implicitly.

## Provenance and independent gates

All five provenance fields are lowercase SHA-256 hex digests. `fit_digest`
hashes ordered fitting-row indices and typed original cells with length framing.
`column_digest` hashes visibility and the ordered raw column declarations,
including private names, IDs, roles, types, units, semantic labels,
bounds/domains, and mask tokens. Neither digest
appears in a public projection. `entity_digest` and `split_digest` are explicit
caller attestations and must come from the experiment's admitted manifests;
D00 does not independently recover entity identity or inspect sealed data.

`implementation_hash()` hashes length-framed source bytes for schema.rs,
typed_data.rs, the existing transform implementation, every PreparedView
implementation section, and the JSON schema. Applying/compiling against a frozen
map requires the current implementation hash. Altering any retained parameter,
declaration, digest, horizon, policy, rights, or projection state rejects.

Capabilities declare `schema`, `adapter`, and `qualified` independently for
each task. The initial lane declares regression and binary eligible
(`qualified: true`); multiclass parsing and adapter validation work with
`qualified: false`. Multiclass compilation remains blocked until an admitted
frozen panel supports qualification and the versioned policy changes. These
capability declarations describe scope, not measured certification.

Rights state is independently `research_unassessed` (default caller choice),
`approved` (requires an evidence digest), or `blocked`. It conveys neither
utility nor empirical disclosure nor formal DP. Approval evidence is an
attestation, not a legal determination. No DP, anonymity, HIPAA, or custody
clearance follows from this contract; D18 custody HOLD remains unchanged.

## Privacy, limits, and refusals

The prepared payload is **private**: rows, dictionaries, units, transform
parameters, support bounds, and digests must remain in private storage.
`public_projection()` emits only the canonical format/version, public state,
task, opaque numeric column types/roles/IDs, groups, capabilities, and rights
state. It excludes source names/IDs, values, extrema, dictionaries, units,
parameters, horizon, class order, and provenance. This summary cannot reload as
a PreparedView or enter the prepared compile boundary.

All supported entry points return fixed `SchemaError` codes without copying
parser diagnostics or input text. Debug output of views, contracts, and compile
inputs is redacted. Unknown, missing (including nullable), duplicate, or wrongly
typed fields produce `invalid_prepared_json`. Other principal refusals include:

| Code | Meaning |
| --- | --- |
| `prepared_input_required` | Incorrect prepared format or version |
| `invalid_prepared_contract` | Inconsistent roles, IDs, task metadata, or policies |
| `prepared_column_mismatch` | Raw declaration, ordered column map, or row width changed |
| `prepared_group_mismatch` | Invalid partition or changed ordered group map |
| `prepared_class_mismatch` | Invalid/changed classes or target mapping |
| `prepared_contract_mismatch` | Other frozen metadata changed |
| `prepared_implementation_mismatch` | Map was produced by a different implementation |
| `prepared_missing_rejected` / `prepared_oov_rejected` | Unsupported missing/category input |
| `prepared_support_mismatch` / `prepared_type_mismatch` | Invalid numeric support/type |
| `prepared_task_not_qualified` / `prepared_rights_blocked` | Independent admission gate closed |

Default limits are 16 MiB private payload, 2 MiB metadata, 100,000 rows,
1,000,000 cells, 4,096 total columns/classes/dictionary entries, and 4,096
UTF-8 bytes per category. The structural JSON schema cannot enforce every
cross-field, byte-limit, or frozen-order condition; the native parser does.

Positive regression/binary/multiclass examples and adversarial mutations live
under `tests/fixtures/prepared-view-v1/` and in the schema/typed-data unit tests.
Example provenance uses zero digests as test sentinels, not admitted evidence;
the adversarial harness supplies the current implementation hash explicitly.
Use `cargo test --locked --lib schema::`,
`cargo test --locked --lib typed_data::`, and
`cargo test --locked --test vnext_raw_schema` for focused validation. The
integrator must register all new paths in `agent/owner-map.json` and
`agent/test-map.json`; those files are outside D00's write lease.
