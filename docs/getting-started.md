# From a table to a sampling program

DOPE compiles a table into a small generator. Another process reloads that
generator, draws synthetic rows, and trains a learner on those draws. Evaluate
that learner on real rows that the fit did not see. The artifact is a lossy,
task-specific program. It stores no source rows, and it is not a lossless archive.

The public path below is the signed engine `dope` from
[dope-engine-v3.0.0](https://github.com/neverhuman/dope/releases/tag/dope-engine-v3.0.0).
Agents that only need to fit a table should follow this page and
[AGENTS.md](../AGENTS.md). The Airfoil walkthrough further down uses the older
`dope-kernel` research binary.

## Install the signed engine

Linux x86-64, with `gh` 2.102.0 or newer. The installer checks `SHA256SUMS`
and the GitHub/Sigstore attestation before it copies the binary.
[Verify a package by hand](signed-engine-release.md#verify-before-install)
or [build from source](signed-engine-release.md#build-and-package) when you
are not using the one-line installer.

```bash
curl -fsSL https://raw.githubusercontent.com/neverhuman/dope/main/scripts/install.sh | bash -s -- --no-weights
dope doctor
```

`--no-weights` is the analytic release. The published tag does not ship
`dope-v8-weights.safetensors`, so leave installer `--v8-weights` off. The signed
tag users install is `dope-engine-v3.0.0`.

## Fit a table

Name the target. Keep the original file, and follow the
[target-last contract](data-contract.md): one declared target, stable feature
order, and duplicate physical rows in the same split. Query targets stay out
of the fit.

```bash
dope fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
dope inspect model.dope
dope sample model.dope -n 1000 --seed 1 -o synthetic.csv
dope eval -m model.dope --train train.csv --test test.csv
```

`--format 1` remains the CLI default. `--format auto-v8` fits Format 1 and the
weight-free DSE8 packet inside one file of at most 2048 bytes and keeps DSE8
when a duplicate-grouped two-fold check on the fitting rows prefers it.
`dope inspect` prints the format and the complete file size. The command list
for this binary is [engine/README.md](../engine/README.md).

`dope eval` trains linear and boosted-tree auditors on the synthetic rows and
scores them on the real test rows. For a fixed learner, retention is

```text
(L_null - L_syn) / (L_null - L_real)
```

One recovers the measured real-trained lift. Zero recovers none. A negative
value is worse than the null. A denominator near zero is uninformative. This
is not accuracy, and it is not the fraction of the table that can be rebuilt.

The landing-page scoreboard and the
[v3 white paper](whitepaper-v3/dope-v3.pdf) are the measured comparison.
That report is a fresh-split DEV-SELECT estimate. The paired difference versus
ARF includes zero. Storing no source rows is not differential privacy and not
HIPAA de-identification. Cross-model feature-driver preservation is not yet
measured.

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
