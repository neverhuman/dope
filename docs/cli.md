# Native command line

`dope` is the bounded, offline development facade. Its single argument registry
is `rust/main/facade/registry.rs`; `dope --help` and `dope COMMAND --help` are
generated from it. `dope --version` and `dope version [--json]` use Cargo's version.

`dope-kernel` retains its existing argument registry, certification behavior,
campaign commands, DPK3.2 decoding, and historical v1 readers. The facade does
not change those receipts or turn historical certification into TDL qualification.
Use `dope-kernel --help` for that full compatibility interface.

## Commands

| Command | Inputs and outputs | Behavior |
| --- | --- | --- |
| `prepare` | `--csv FILE --schema FILE --out FILE [--report FILE]` | Validate declared headered input and write an opaque intake receipt. Optional report receives the same receipt. |
| `validate-input` (alias `validate`) | `--csv FILE --schema FILE [--out FILE]` | Validate without fitting; print an intake receipt or write it to `--out`. |
| `compile` (alias `fit`) | `--csv FILE --schema FILE --out FILE [--report FILE]` | Compile fully observed numeric regression into a complete raw sampling artifact with retained original-unit transforms. |
| `inspect` | `--artifact FILE [--out FILE]` | Inspect retained raw/mixed state, native DPK, v1 JSON, bare TDL1, or a TDL charge manifest. Report measured byte scopes and explicit unmeasured qualification. |
| `sample` | `--artifact FILE --rows N --out FILE [--seed U64]` | Write headerless CSV in original units for raw artifacts or legacy native/v1 units. Mixed sampling requires a separate output contract and returns unsupported here. |
| `doctor` | `[--out FILE]` | Report static build capabilities and unavailable integrations without data access, downloads, imports, or network probes. |
| `evaluate` | `--artifact FILE --out FILE [--report FILE]` | Candidate evaluation is unavailable until the frozen evaluator/worker/panel interfaces are integrated; returns 3 before opening input. |
| `report` | `--report FILE [--out FILE]` | Read a historical certification receipt and emit an opaque display summary. A failed gate or missing rights yields the corresponding nonzero exit after writing the summary. |
| `demo` | `--out FILE [--report FILE] [--seed U64]` | Compile a built-in 16-row synthetic engineering fixture. The receipt says `fixture_only=true`, `qualified=false`, `certified=false`. |
| `version` | `[--json]` | Print the Cargo package version. |

## Table workflow (`dope-kernel`)

| Command | Inputs and outputs | Behavior |
| --- | --- | --- |
| `fit-table` | `--csv FILE --target NAME --out FILE [--drop COL]... [--task auto\|regression\|binary] [--seed U64] [--candidate ID] [--native-target-profile NAME\|none] [--min-level-count K]` | Infer each column's type from a headered CSV and store an invertible plan: numbers to the unit interval (log-scaled first when \|skewness\| > 2), categories to one code ordered by smoothed target mean (two or more rare levels pooled, missing markers kept as levels; numeric binary labels are compared by value), dates to day counts (calendar-validated; bounds kept to the second), missing numeric cells kept missing. Identifier-like and all-missing columns are dropped and reported. Compiles the plan's matrix with the native trainer and packed values; refuses an existing output before fitting. Prints a report with byte accounting and column kinds. |
| `sample-table` | `--model FILE --rows N --out FILE [--seed U64]` | Write a headered CSV in the source header order, category labels, integer and date formats. |
| `inspect-table` | `--model FILE` | Byte accounting (program, plan, dictionary, framing) and column kinds, without names, labels or extrema. |

The table artifact (`DOPETAB1`) contains the column names and category labels
needed to decode samples; treat it as private as the schema. A binary target
must have two levels; multiclass targets are refused.

`dope-kernel compile` accepts two development options: `--native-target-profile
NAME` trains the compact neural residual in Rust (profile names follow the GPU
research profiles, for example `features12_steps2048`), and `--packed-codec`
writes values as quantization-grid codes (DPK3 header flag bit 1). Neither
changes a default, so historical artifacts and receipts are reproduced as before.
Candidates can also be named as `symbolic_k{knots}_d{dependence}_t{target}_q{bits}`
(indices into the compiler's ladders); dependence family 7 is an exact dense
Gaussian copula of normal scores, available only by explicit request.

An intake receipt records input/schema hashes, row/column counts and
`state=input_validated`, `fit_transform_state=not_fitted`,
`prepared_view=unavailable`, `qualified=false`. `prepare` does not persist fitted
transforms or authorize reuse across partitions. Pass only the intended fitting
CSV to `compile`; held-out tables are never accepted by that command. Prepared
view replay awaits D01 integration. No intake, compiler, demo, or inspection
success constitutes scientific, privacy, formal-DP, or legal qualification.

The facade refuses nonconstant v1 tree sampling because its historical traversal
has no bound; `dope-kernel` retains that compatibility interface. V1 feature
counts, sparse term indices, quantile arrays, and copula work are checked before
sampling. The raw artifact retains private transforms required for sampling. Treat it and
synthetic CSVs as private data. Public JSON omits source names, labels, units,
domains, extrema, fitted transform parameters, and private paths. Inspection of
v1 artifacts also omits their arbitrary descriptive/program strings.

## Shared flags and bounds

Shared flags work before or after the subcommand.

| Flag | Default | Allowed range / meaning |
| --- | --- | --- |
| `--offline[=true]` | true | Mandatory. `--offline=false` returns 3. |
| `--network` | false | Any explicit network request returns 3 before file access. |
| `--panel NAME` | `development` | `development` is the engineering panel. `tdl-candidate` is recognized but unavailable and returns 3. Other names return 2. |
| `--panel-tag NAME` | none | Deprecated alias for `--panel`; prints a fixed migration notice. |
| `--max-input-bytes N` | 16,777,216 | 1–16,777,216 bytes per input; schema also has a 2,097,152-byte ceiling. DPK3 decoded sections share this cap. |
| `--max-output-bytes N` | 16,777,216 | 1–16,777,216 bytes per file or JSON response. |
| `--max-rows N` | 100,000 | 1–100,000 rows per table/sample. |
| `--max-cells N` | 1,000,000 | 1–1,000,000 cells including target cells. |
| `--max-field-bytes N` | 4,096 | 1–4,096 UTF-8 bytes per field and declared category. |
| `-h`, `--help` | none | Generated help, exit 0. |
| `-V`, `--version` | none | Package version, exit 0. |

There are at most 4,096 columns. All paths must be local and at most 4,096
bytes; URLs and network-style paths are refused. Input files must be regular
files; final-component symlinks and special devices are refused on Unix.
Output parents must exist. Outputs are exclusively created, never overwritten;
duplicate destinations are invalid. Unix output permissions are 0600.
Input reads consume at most the byte cap plus one sentinel byte. Sampling also
reserves 32 output bytes per requested cell before allocation. Existing output
files remain intact on refusal. A later I/O failure can leave an earlier output
or partial file for diagnosis; a nonzero exit never implies completion.

The facade invokes native local APIs and never opens a network transport or
downloads weights. It is an ordinary process, not a filesystem/network sandbox
for arbitrary mounted filesystems, inherited descriptors, or other principals.
The historical path-only inspection/decoding APIs consume sealed in-memory
snapshots on Linux; those operations return 3 on other platforms until an
equivalent bounded snapshot adapter is integrated. B01 source-free isolation is
not claimed by facade sampling.

## Compiler flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out FILE` | required | Private artifact destination. |
| `--report FILE` | absent | Optional opaque compiler summary, separate from artifact. |
| `--seed U64` / `--fit-seed U64` | native default | Fit seed; sampling uses its own seed. |
| `--candidate chow_liu` | `chow_liu` | Bounded symbolic backend. Other candidates are invalid in this facade. |
| `--deadline-seconds N` | 60 | Integer 1–600. Includes intake; overdue results are refused before writing. |
| `--max-complete-bytes N` | 32,768 | Integer 1–16,777,216; includes retained raw transforms. Native payload also respects the historical tier ceiling. `--max-artifact-bytes` is an alias. |
| `--task regression\|binary` | `regression` | Raw workflow supports regression; binary requires legacy numeric input. |
| `--tier l0\|l1\|l2\|l3` | `l3` | Historical compiler policy; never a new measured tier claim. |
| `--require-formal-dp` | false | Returns 8: no verified mechanism/accountant is integrated. |
| `--dataset-dir DIR` | absent | Explicit legacy input alternative: headerless numeric `train.csv`. Conflicts with `--csv`; prints a migration notice. |

`--csv` requires `--schema` and vice versa. Select exactly that pair or
`--dataset-dir`. The facade currently supports observed finite numeric legacy
input only; use `dope-kernel` for the broader historical interface. Deadline
checks use the compiler's bounded candidate work and refuse overdue output;
they are not an operating-system watchdog or external-worker budget.

## Legacy migration and report behavior

`--kernel` remains an alias for `--artifact` on inspect, sample and evaluate.
`fit` aliases compile. `certify` and `verify` alias evaluate.

For existing scripts, `evaluate --real-dir DIR --artifact FILE --out FILE`
explicitly replays historical certification against bounded numeric `train.csv`
and `test.csv`. Additional flags are `--seed U64` (1729), `--tier` (l3), and
`--report FILE` (optional opaque summary). This mode prints a migration notice.
Its historical exit-0 meaning is **receipt written**, including a receipt with
`certified=false`; inspect that field or run `dope report --report FILE` for
gate-sensitive exit codes. It does not perform candidate TDL evaluation.
`dope-kernel` retains the complete legacy certification interface and v1 receipt
commands with their existing status conventions.

`report` checks the historical envelope and boolean gate consistency. It does
not authenticate source, signature, protocol, or release evidence. Arbitrary
strings and nested data are excluded from its output. It always reports TDL
utility/empirical privacy unassessed, formal DP unverified, and legal release
pending. A locally edited `certified=true` receipt cannot authorize release.
Independent release profiles and worker qualification await S00/W00/C05
integration; no `--epsilon` or automatic de-identification claim is offered.

## Stable facade exit codes

| Code | Meaning |
| --- | --- |
| 0 | Command completed, help/version displayed, or explicit legacy receipt written as described above. |
| 2 | Invalid arguments, schema, input/artifact bytes, inconsistent report, nonlocal path, existing/duplicate output. |
| 3 | Unsupported capability, unavailable candidate panel/evaluator, or refused network request. |
| 4 | License/rights refusal: release authorization is absent even if historical certification claims success. |
| 5 | Input, field, dimension, sampling, artifact, output-byte, or compiler deadline budget exceeded. |
| 6 | Missing/unreadable input or incomplete local artifact/closure file availability. |
| 7 | A reported historical scientific gate failed. |
| 8 | A reported historical privacy gate failed, or formal-DP capability was requested without a verified mechanism. |
| 70 | Internal serialization, snapshot, or local output I/O failure. |

Flags outside their allowed ranges are usage errors (2); an admitted workload
exceeding its configured cap is a budget refusal (5). Unsupported evaluation
does not read a bundle and therefore reports 3 rather than inventing a closure
verdict. For a report with both scientific and privacy failures, 8 takes
precedence; otherwise an uncertified report returns 7, and a historically
certified report returns 4 because independent rights authorization is absent.

Failures emit fixed JSON on stderr: `error`, `exit_code`, and `docs`. Argument
parser diagnostics and underlying error strings are never printed because they
may contain private values, headers, extrema, or paths. Migration notices use
fixed text. Consult generated help to correct invalid flags.

## Example

```sh
dope validate-input --csv fit.csv --schema schema.json
dope prepare --csv fit.csv --schema schema.json --out intake.json
dope compile --csv fit.csv --schema schema.json --out candidate.dpk --report compile.json --seed 1729
dope inspect --artifact candidate.dpk --out inspection.json
dope sample --artifact candidate.dpk --rows 100 --seed 17 --out synthetic.csv
dope doctor
dope demo --out fixture.dpk --report fixture.json
```

These commands provide engineering evidence only. They never imply formal DP,
HIPAA de-identification, approved external checkpoints, or release authorization.


## Offline healthcare simulation

The [healthcare simulation example](../examples/healthcare-sim/) demonstrates
the actual native CPU CLI on deterministic, explicitly simulated patient rows.
It does not use real patient data or claim clinical validity, performance on
unseen real patients, HIPAA de-identification, formal differential privacy, or
observational confirmation.

From the repository root, prepare and check the fixture, compile only its
preprediction-time fit rows, sample from the compiled artifact, then evaluate
the generated rows against a patient-disjoint later-time simulated holdout:

```sh
python3 examples/healthcare-sim/demo.py prepare --out target/healthcare-sim
python3 examples/healthcare-sim/demo.py verify --out target/healthcare-sim
cargo build --locked --bin dope-kernel
target/debug/dope-kernel compile-csv --csv target/healthcare-sim/fit.csv \
  --target event_30d --candidate chow_liu --seed 1729 \
  --deadline-seconds 60 --out target/healthcare-sim/model.dpk
target/debug/dope-kernel sample-mixed --artifact target/healthcare-sim/model.dpk \
  --rows 640 --seed 1729 --out target/healthcare-sim/sampled.json
python3 examples/healthcare-sim/demo.py evaluate --out target/healthcare-sim \
  --sampled target/healthcare-sim/sampled.json
```

`verify` checks exact feature headers, disjoint simulated patient sets, the
date boundary, and a negative future-label leakage probe. The receipt and
manifest report actual fixture, artifact, and sample sizes and measured
outcomes. The small classical evaluator uses a standard-library ridge
linear-probability model on CPU. The optional modern-model panel is available
only with a separately admitted native adapter and qualified installed model;
the example never substitutes a fixture for native success. Runtime objectives
must be based on measurements from the intended hardware. Simulated results do
not count as observational evidence.

Reference-run measurements on Linux x86_64 with 1 visible CPU: prepare 0.21 s,
verify 0.05 s, compile 0.16 s, sample 0.08 s, evaluate 0.18 s, and debug CLI
build 109 s. These single-run measurements do not support adopting the proposed
five-minute CPU or fifteen-minute GPU objective. The run produced 9,665-byte fit
and 2,469-byte evaluation CSVs, a 2,697-byte artifact, and a 146,482-byte typed
sample. The 80-row simulated evaluation Brier score was 0.1173 after sampled
training and 0.1164 after fit-row training; respective Brier skills versus the
same null were 0.0109 and 0.0183. `target/healthcare-sim/receipt.json` records
these values after completing the steps.

## General compile and sample

For other headered CSV data, the core CLI supports `compile-csv`, `inspect`, and
`sample`. The target defaults to the last CSV column unless `--target` names a
column or gives its zero-based index. The compiled artifact does not preserve
training rows. A successful compile is not a release or privacy certification.
Keep protected or otherwise sensitive data within its authorized environment.
