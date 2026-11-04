# DOPE SDK integration surface

`dope-sdk/v0` reexports existing native engine types. It duplicates no arithmetic, preparation, weights, checkpoint loading or sampler. Consumers depend on this package at a reviewed full Git revision and retain their separately pinned weights/recipe/preparation identities. The minimum compiler is Rust1.88; the engine workspace is separate from the root research kernel workspace. No Python or JopeDime dependency is required by this crate.

Use `dope_sdk::v0::prepare` for released-v3 preparation (`PreparedTable`, `FitOnlyTableTransform`, explicit row-ID preparation and conditioning builders), `v0::landmark` for native signatures/progressive bundles, and `v0::mlc_v3` for the existing model/routing types. These are the actual core types. Invalid input behavior and private/public boundaries remain owned by the engine. SDK version alone does not identify weights or qualify results.

`dope_sdk::nc1` exposes the separate `nc1-minmax-onehot-v1` successor: typed declarations, declared physical FIT rows, owner-private ScaleCard and public structural UnitSchema. Keep the card, real matrix and original row IDs owner-side. Pass only the admitted public graph to a recipient. Released-v3 weights are not automatically NC1-compatible; do not feed them a relabeled NC1 matrix through a legacy method.

The additive `mlc_v4` and `dvs` modules expose the closed six-gate DML4 sampler and prepared-evidence evaluator. Their recipe/hash contracts and qualification states are distinct from v0 legacy serving. No recipient operation requires the owner's source, card or Python encoder. These modules do not grant model publication, benchmark acceptance, privacy certification or live serving adoption.

From the repository root, use `cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-sdk` for direct-core/facade parity and refusals. The focused consumer handoff must retain exact source, weight, recipe, FIT/input-policy and conditioning/sampler identities; compare vectors/digests under the same qualified path before changing a consumer pin. Compilation alone does not establish parity or consumer qualification. See [the explicit workflows](../../docs/next-version-workflows.md) and [contract decisions](../../docs/adr/normalize-once-and-dvs-v1.md).

The additive native consumer view is available by importing
`dope_sdk::nc1::TargetLastViewExt` and calling
`unit_table.to_target_last_view()?`. It preserves physical row IDs, structural
schema and source/FIT/transform hashes, copies existing feature values in order,
and appends the declared target at `target_column()`. Allocation limits include
that extra target cell per row. Feature-only tables refuse with
`target_last_target_absent`. The view is owner-private, has redacted Debug and
no serialization implementation; it does not alter normalization, producer
identity, the existing x/y wire or released-v3 serving. Native boundary fixtures
must execute before this addition is called qualified.
