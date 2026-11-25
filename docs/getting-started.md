# From a table to a sampling program

DOPE compiles a table into a small generator. Another process reloads that
generator, draws synthetic rows, and trains a learner on those draws. Evaluate
that learner on real rows that the fit did not see. The artifact is a lossy,
task-specific program. It contains fitted summaries and schema without any
complete fitting records.

The public path below is the signed engine `dope` from
[dope-engine-v3.0.0](https://github.com/neverhuman/dope/releases/tag/dope-engine-v3.0.0).
Agents that only need to fit a table should follow this page and
[AGENTS.md](../AGENTS.md). The Airfoil walkthrough further down uses the older
`dope-kernel` research binary.

## Install the signed engine

Use the [README quick start](../README.md#quick-start). It installs signed
`dope-engine-v3.0.0` under `$HOME/.local/opt/dope/3.0.0` and evaluates a
deterministic toy table. That toy checks the binary. The scores on the landing
page come from the development report, not from that table.

The commands below call `$HOME/.local/opt/dope/3.0.0/bin/dope`. The installer
does not add that directory to `PATH`.

The quick start uses `curl`, `jq`, and `sha256sum` and downloads a verified
temporary GitHub CLI if needed. For the [manual verification and installation
route](signed-engine-release.md#verify-before-install), install GitHub CLI
`2.102.0` or newer first. If glibc is older than `2.39`, the installer stops
before downloading the engine. To remove the standalone installation, remove
`$HOME/.local/opt/dope/3.0.0`. The `--uninstall` flag is for custom-directory
installations.

On macOS, Windows, or a Linux host older than `glibc 2.39`, build from source with Rust 1.88+ (`cargo install --locked --path engine/dope-cli`) or run the signed Linux binary in Docker. There is no macOS, Windows, or ARM signed release asset yet.

```bash
docker run --rm -it --platform linux/amd64 -v "$PWD":/work -w /work ubuntu:24.04 bash
```

Inside that container, run `apt-get update && apt-get install -y ca-certificates
curl jq`, then run the [README quick start](../README.md#quick-start). CI
executes that quick start inside `ubuntu:24.04` on a Linux x86-64 runner.

To run the complete lifecycle (`fit -> sample -> train models -> test on real holdout`) on a real, offline CC BY 4.0 dataset in under 10 seconds, see [examples/v3-quickstart](../examples/v3-quickstart/README.md).

## Fit a table

Declare the target by name. Keep the original file, and follow the
[target-last contract](data-contract.md) for intake and stable feature order.
Group duplicate physical rows before splitting and assign each group to one
split. Use only `train.csv` to fit and select the encoder. Hold back `test.csv`
for evaluation and keep its target out of the learner's features.

```bash
dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
"$dope" fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
wc -c < model.dope
"$dope" sample model.dope -n 1000 --seed 1 -o synthetic.csv
```

`--format auto-v8` fits Format 1 and analytic DSE8 separately and writes one
file of at most 2048 bytes. When both fit, a duplicate-grouped two-fold check
on the fitting rows can select DSE8. The command list is in the
[engine README](../engine/README.md).

### Evaluate the synthetic CSV

Choose the features, preprocessing, learner, hyperparameters and loss before
scoring. For example, use ridge regression with a fixed penalty, preprocessing
fitted on the real training rows, and mean squared error for a numeric target.
Fit one learner on `train.csv` and one on `synthetic.csv`. Score both on the
same real test rows. For the null loss, predict the real training mean on
each test row.

The v3.0.0 CLI can perform this comparison with its fixed linear evaluator
when the schema fits in a separate Format 1 artifact. The evaluator uses
that artifact for schema and encoding and trains the synthetic arm on the
CSV you pass to `--synth`, including samples drawn from DSE8.

```bash
"$dope" fit train.csv --target y --format 1 --budget 2048 -o audit-schema.dope
"$dope" eval -m audit-schema.dope --train train.csv --test test.csv \
  --synth synthetic.csv --auditors linear --seed 1 --json evaluation.json
```

The evaluation's artifact byte count refers to `audit-schema.dope`. Use the
earlier `wc -c` result for the size of `model.dope`. Direct native evaluation
of DSE8 artifacts is outside the v3.0.0 CLI. If Format 1 cannot fit your
schema, use the fixed-learner procedure directly on the two training CSVs.

Retention is `(null loss − synthetic loss) / (null loss − real loss)`.
One recovers the real-trained learner's measured lift over the null, zero
recovers none, and a negative value means the synthetic-trained learner
scored worse than the null. A denominator near zero makes retention
uninformative. The [v3 white paper](whitepaper-v3/dope-v3.pdf) reports the
fresh-split DEV-SELECT comparison and the scope of the privacy diagnostics.

## Historical kernel demonstration

The source-free Airfoil demonstration supports Linux x86-64. Use Rust from
`rust-toolchain.toml`, its C build tools, and Python 3.12 with venv support.
Python is used for the optional example learner. The compiler and sampler in
this section are the `dope-kernel` binary, which is separate from the signed
engine above. Run from the repository root:

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

## Historical kernel routes

These commands belong to `dope-kernel` and the alpha facade in
[cli.md](cli.md). A new table uses `dope fit --format auto-v8` in the signed
engine section above.

| Input | Route | Boundary |
| --- | --- | --- |
| Observed numeric, integer, ordinal or Boolean features; continuous target | `compile-raw`, `inspect-raw`, `sample-raw` | Fit-only affine transforms; samples return to original units |
| Nominal strings or missingness | `compile-csv` / `sample-mixed`, or the explicit mixed schema API | Bounded one-hot nominal carrier; development capability |
| Already projected headerless `train.csv` | `dope compile`, `dope inspect`, `dope sample` | Caller owns projection correctness and accounting |

The [simulated mixed example](../examples/healthcare-sim/README.md) runs offline
without Python packages. Do not interpret category numbers as continuous values.
[Transfer Description Length](metric.md) needs driver, budget-grid and hard-gate
evidence that this kernel walkthrough does not supply.
