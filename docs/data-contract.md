# Mandatory target-last data contract

Every canonical table has its non-target columns in the declared stable order,
followed by exactly one declared target. For model matrices this is features,
then target. JSON object key order has no column semantics; separate `X` and `y`
interfaces retain their explicit roles.

The intake boundary resolves the target from the dataset declaration and its
schema. It never guesses the label from a position, spelling, values or model
outcomes. A disagreement between the expected target and the schema is an error.
An external table with a target in another position is normalized once into a
new CSV and schema. Downstream compilers, pilot input assembly and draw/evaluator
bindings validate the canonical layout and reject drift. They do not silently
select a different column as the label.

Use the shared intake command, with the actual declared target:

```sh
python3 -B -m research.transfer.target_last --csv SOURCE.csv --schema SOURCE.schema.json --target DECLARED_TARGET --out /absolute/new/canonical-directory
```

The output is `table.csv`, `schema.json` and `normalization.json`. Existing output
directories are refused. Original source files are never overwritten. The receipt
binds both source and normalized hashes, the exact column permutation, and row
count. Values remain strings during normalization: query labels are never parsed
numerically, used for selection, or projected into features. Row order and cell
text are unchanged. An already canonical input keeps its exact bytes.

A column permutation creates a new representation, not a new physical source.
Keep original physical row identities and bind the source-to-view mapping when
building membership, split and qualification receipts. Never transfer an old run
to the new CSV hash; preserve its actual source and add explicit array/view
correspondence where applicable.

Native Rust schema intake and public schema projection also place the declared
target last. `TypedTable::read_csv` is an intake boundary for programmatically
constructed schemas as well. It moves the single
explicitly declared target to the end of the internal schema and columns, while
retaining original input positions for provenance and error locations. Historical
receipt, stored-state and artifact decoders retain their existing formats.

The original adverse source is `y,x0,...,x5`. Its canonical view must be
`x0,...,x5,y`, with the schema still declaring `y` as the target. Relabeling `x5`
as the target is refused; including `y` in query features is invalid qualification.

Validation: `python3 -B -m unittest research.transfer.tests.test_target_last research.transfer.tests.test_pilot_worker_inputs -v`
and `cargo test --locked --lib typed_data::tests`.

The Python pilot/model CSV intake accepts feature and target roles only. Identifier,
timestamp, group and excluded-sensitive columns are refused rather than promoted
to features. The native mixed-table API retains its explicit non-feature roles and
excludes them through its typed projection; it still places its declared target last.
For a reordered CSV, the receipt identifies UTF-8, LF record separators and minimal
CSV quoting. Cell strings remain identical; CSV syntax bytes may differ. Already
canonical input preserves its exact CSV and schema bytes.

The Python pilot/model CSV intake accepts feature and target roles only. Identifier,
timestamp, group and excluded-sensitive columns are refused rather than promoted
to features. The native mixed-table API retains its explicit non-feature roles and
excludes them through its typed projection; it still places its declared target last.
For a reordered CSV, the receipt identifies UTF-8, LF record separators and minimal
CSV quoting. Cell strings remain identical; CSV syntax bytes may differ. Already
canonical input preserves its exact CSV and schema bytes.

Historical retained draw bytes may be decoded and reverified in their original
schema order so their receipts and row identities remain intact. This read-only
compatibility route grants no model use: creating a training pool or writing a
new draw handoff requires canonical target-last schema first. Original stored
proofs are never rewritten or relabeled as current-source execution.
