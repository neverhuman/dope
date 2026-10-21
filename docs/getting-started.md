# From a real table to a sampling program

DOPE compiles a table into a small generator. Another process reloads that
generator, draws synthetic rows, and trains a learner on those draws. Evaluate
that learner on real rows that neither compilation nor model selection saw.
The artifact is a lossy, task-specific representation, not a lossless archive.

## Install and run the public example

The source-free demonstration supports Linux x86-64. Use Rust from
`rust-toolchain.toml`, its C build tools, and Python 3.12 with venv support.
Python is used for the optional example learner; the product compiler and
sampler are Rust. Run from the repository root:

```sh
git clone https://github.com/neverhuman/dope.git
cd dope
cargo build --locked --release --bin dope-kernel
python3 -m venv target/demo-env
target/demo-env/bin/python -m pip install -r examples/public-regression/requirements.txt
target/demo-env/bin/python -I -B examples/public-regression/run.py \
  --binary "$PWD/target/release/dope-kernel" \
  --out target/public-demo/airfoil-1
```

The command downloads the official UCI Airfoil Self-Noise source. Its license,
DOI, archive/file checksums and exact schema are documented in
[the example](../examples/public-regression/README.md). A verified local `.dat`
file can be supplied with `--source PATH`; it must match the same pinned bytes.
A new output directory is required for every attempt. Choose `airfoil-2` for a
second run, preserving the first receipt even if it failed.

The runner executes this sequence:

1. Check the source bytes, parse the declared target, and group entire physical
   configurations into disjoint fit/query arms. The target is the final column.
2. Give only fit rows to `compile-raw`. Fit affine numeric transforms and a
   fixed Chow-Liu symbolic program. Charge framing, schema, transforms and
   model bytes together, under a complete-artifact cap and a deadline.
3. Exit compilation, reload the artifact, and sample at the original fit-row
   count and four times that count. Reproduce each sample digest in a fresh
   native worker that cannot open the source table.
4. Fit the same StandardScaler/Ridge recipe on real fit rows and on each
   synthetic arm. Evaluate all predictions against the same untouched,
   original-unit query targets. Compare them with the fit-mean null.

Open `target/public-demo/airfoil-1/report.html`. The adjacent `receipt.json`
contains artifact bytes, original-unit losses, retention, sample/replay
digests, measured process costs and explicitly unavailable checks.
`attempts.jsonl` retains failed as well as successful step costs. A poor or
negative retention is an observed result; the example does not tune it away.

This is a single public development demonstration of the **raw symbolic
profile**, not a reproduction of the selected historical paper profile.
It does not measure TDL, a foundation-model panel or privacy certification.
For a dedicated checksum-verified CLI installation after building, follow
[the native installer](native-workflow.md).

## Bring your own table

Keep the original source. Declare a schema with exactly one target; normalize
to stable feature order followed by that target using the
[target-last intake](data-contract.md). Split entities, duplicate groups or
time periods **before** fitting any data-dependent transform. Query targets
are for the evaluator, never features, compilation inputs or selection inputs.

Choose an explicit supported route:

| Input | Route | Current boundary |
| --- | --- | --- |
| Observed numeric, integer, ordinal or Boolean features; continuous target | `compile-raw`, `inspect-raw`, `sample-raw` | Fit-only affine transforms; samples return to original units |
| Nominal strings or missingness | `compile-csv` / `sample-mixed`, or the explicit mixed schema API | Bounded one-hot nominal carrier; development capability, not the numeric profile |
| Already projected headerless `train.csv` | `dope compile`, `dope inspect`, `dope sample` | Historical benchmark interface; caller owns projection correctness and accounting |

The [CLI reference](cli.md) lists the exact flags. The
[simulated mixed example](../examples/healthcare-sim/README.md) runs offline
without Python packages and shows strings and masks. It is separate from the
real Airfoil example. Do not interpret category numbers as continuous values,
or treat the old projected interface as raw string ingestion.

## Interpret the result

For a fixed learner and real query set, let `L_null`, `L_real` and `L_syn` be
losses from a table-independent baseline, the real-trained learner and the
synthetic-trained learner. Retention is

```text
(L_null - L_syn) / (L_null - L_real)
```

Retention of one recovers the measured real-trained lift. Zero recovers none;
negative values are worse than the null; values above one are possible.
This is not accuracy, a correlation coefficient, or the fraction of raw data
that can be reconstructed. A denominator near zero is uninformative, not a
good compression result.

Compare complete **fitted-state bytes** with transfer, then report the generic
runtime and cold sampling closure separately. Dividing one median by another
is a descriptive size ratio, not a per-dataset compression guarantee. The
headline KPI, [Transfer Description Length](metric.md), requires additional
driver, budget-grid and hard-gate evidence that this walkthrough does not supply.
