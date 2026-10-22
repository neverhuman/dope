# DOPE engine

The deployable DOPE implementation: a pure-Rust library (`dope-core`) and command-line
tool (`dope`). It compiles a CSV into a byte-budgeted generative program stored as a
safetensors file, samples from it on CPU, CUDA or Metal (CI tests the CPU paths only),
and measures retention with built-in auditors. Python is not needed to fit, sample or evaluate.

```bash
cargo build --release --manifest-path engine/Cargo.toml          # CPU
cargo build --release --manifest-path engine/Cargo.toml --features cuda    # NVIDIA GPU
cargo build --release --manifest-path engine/Cargo.toml --features metal   # Apple GPU
cargo test  --release --manifest-path engine/Cargo.toml          # unit, end-to-end and CLI tests
```

## Commands

| Command | What it does |
| --- | --- |
| `dope fit data.csv --target y [--budget 4k] [--seed 0] [--anonymize] [--device auto]` | Compile a program within a byte budget (whole file) and print its self-audit. |
| `dope fit data.csv -t y --size large --max-parameters 4000` | Apply the 16 KiB preset and a complete-generator scalar cap; explicit `--budget` overrides the preset. |
| `dope fit-tiers data.csv -t y --dataset-id ID --source URL --data-license SPDX --out DIR` | Train 2/4/16 KiB dataset-specific pretrained artifacts with checksums and a catalog for release review. |
| `dope sample model.dope -n N [--space raw\|ml] [--scaling unit\|quantile] [--seed S]` | Draw rows in original units and labels, or ML-ready (one-hot, `[0,1]`). |
| `dope predict model.dope data.csv` | Use the stored conditional model as a predictor (values or class probabilities). |
| `dope inspect model.dope [--json]` | Byte breakdown (stored, and per section before compression) and the program structure. |
| `dope encode data.csv [--model model.dope] [--scaling unit\|quantile]` | ML-ready encoding of real rows with the same schema as `sample --space ml`. |
| `dope eval -m model.dope --train t.csv --test h.csv` | Retention per auditor family (ridge/logistic, histogram GBDT, MLP). |
| `dope frontier --train t.csv --test h.csv -t y --budgets 1k,4k,16k` | Retention against bytes and the Transfer Description Length. |
| `dope throughput model.dope [-n 1000000] [--device cpu]` | Sampling speed in rows per second. |
| `dope doctor` | Backends compiled into this build. |
| `dope --threads 8 fit data.csv -t y --json --report fit.json` | Bound CPU parallelism and emit machine-readable fit output. |
| `dope --threads 4 doctor --json` | Runtime identity, backend, worker count and executable SHA-256. |

## Recipes

The deployed `dope` application is **Rust end to end**: fitting, sampling, prediction,
encoding and built-in evaluation need no Python interpreter or libtorch. Python research
training and benchmark harnesses stay outside the deployment package.

```bash
# Bound both deployment size and complete-generator state; cap CPU parallelism.
dope --threads 8 fit train.csv --target price --task regression \
  --size large --budget 12KiB --max-parameters 4000 --seed 42 \
  --device cpu --out prices.dope --report fit.json

# One JSON object on stdout for shell scripts, services and agents.
dope --threads 4 fit train.csv -t price --size medium --json -o prices.dope
# Diagnostics identify the Rust runtime, backends, worker count and executable hash.
dope --threads 4 doctor --json

# Generate original-unit rows, or one-hot features and numbers on [0,1].
dope --threads 8 sample prices.dope -n 100000 --seed 42 --device cpu -o synthetic.csv
dope sample prices.dope -n 100000 --space ml --scaling unit -o ml-ready.csv
# Inference CSVs may contain just the input features; no target values are required.
dope predict prices.dope unlabelled.csv -o predictions.csv
dope encode real.csv --model prices.dope --scaling quantile -o encoded.csv

# Evaluate on held-out real rows and save a machine-readable receipt.
dope eval -m prices.dope --train train.csv --test test.csv \
  --auditors linear,gbdt,mlp --factor 4 --seed 42 --json evaluation.json
dope frontier --train train.csv --test test.csv -t price \
  --budgets 2KiB,4KiB,8KiB,16KiB --tau 0.9 --json frontier.json
dope --threads 8 throughput prices.dope -n 1000000 --repeats 3 --device cpu
```

| Control | Meaning |
|---|---|
| `--target`, `--task` | Explicit target name/index; optional regression/classification override |
| `--size`, `--budget` | 2/4/16 KiB preset or exact whole-file cap; explicit budget wins |
| `--max-parameters` | Additional hard cap on the full generator's versioned scalar count |
| `--threads` | Global CPU worker limit, 1–1024; accepted before or after a subcommand |
| `--device`, `--seed` | Compute backend and reproducible random seed |
| `fit --json`, `fit --report` | JSON stdout and/or a saved report; `--quiet` suppresses human output |
| `sample --space`, `--scaling` | Original-unit or ML-ready rows; unit or quantile scaling |

Use `dope --help` and `dope fit --help` for all parameters. The signed CPU release
supports `--device cpu` and `auto`; GPU variants require separate builds and qualification.

## The program

All customer commands execute in Rust. The signed release is the `dope` CLI from
this workspace; Python training/research tools are not deployment dependencies.
See [the recipes](#recipes) and the
[signed binary release workflow](../docs/signed-engine-release.md).

| Part | Stored as | Fitted by |
| --- | --- | --- |
| Schema | column names, kinds, category levels (metadata JSON); affine maps (`F32`) | type inference; levels ordered by shrunken target association; rare levels pooled |
| Marginals | quantile knots on a uniform normal-score grid (`U8` codes + `F16` range), atoms, discrete cumulative probabilities (`U16`) | empirical quantiles |
| Dependence | full correlation, factor loadings, or Chow–Liu tree (`I8`) | rank (van der Waerden) scores with first-order polychoric correction |
| Regimes | weights, means, log-scales per regime (`U8`/`I8`) | diagonal Gaussian-mixture EM on scores; shared within-regime correlation |
| Conditional model | residual tanh network, bit-packed codes (`U8`) + scales and intercepts (`F32`) | AdamW with closed-form readout (variable projection), early stopping |
| Trees | oblivious trees in one `U8` blob: feature indices, threshold bytes, `f16` scales, 6-bit leaf codes | Newton boosting on the quantized network's loss; each tree quantized as it is fitted (error feedback); the prefix that fits the budget |
| Noise | heteroscedastic log-scale, standardized residual quantiles, or a softmax temperature (`F16`) | held-out residuals; temperature by held-out likelihood |

`fit` searches a small capacity ladder (knots, screened inputs, hidden units) and a
dependence ladder (family × regimes) and keeps the best held-out candidates that fit the
budget; the conditional model (network, then trees) is sized first (see the manuscript for
why). How much of the budget the dependence model may reserve is decided by a
**self-audit**: for a few reserve shares the compiler builds a complete program, samples
it, trains fast linear and boosted-tree auditors on the samples, scores them on its own
held-out training rows, and keeps the program with the best worst-family retention. The
report (`--report fit.json`, `self_audit`) and `dope fit` print that estimate. Every byte in
the file is counted, including the safetensors header and the schema.

## The artifact

`--size small|medium|large` selects a 2/4/16 KiB maximum; without a preset or
explicit budget the default stays 4 KiB. `KB`/`MB` are decimal and `KiB`/`MiB`
are binary; historical `k`/`m` stay binary. `--max-parameters N` is an optional
additional hard cap, applied to the full decoded generator, including its forest,
marginals and calibration. Fit reports and `inspect --json` expose a versioned
component count. Larger capacity does not guarantee larger lift. See
[the count definition, model-zoo builder and study protocol](../docs/model-sizes-and-elo.md).

`model.dope` is a standard safetensors file with one `U8` tensor, `dope`, and
`__metadata__ = {"dope": "1"}`. The tensor is a single brotli stream (pure-Rust
[`brotli`](https://crates.io/crates/brotli), quality 11) that inflates to a little-endian
`u32` length, that many bytes of metadata JSON (schema, structure, bit widths and the section
list `"sec": [[name, bytes], ...]`), then the sections in list order: `schema.affine`,
`marginal.*`, `discrete.*`, `copula.*`, `regime.*`, `net.codes`, `net.params`, `forest` and
`noise`. Compressing column names, category labels and codes together matters at this
scale: stored as plain header JSON they were most of a 4 KiB file. Decoding refuses truncated,
inconsistent or oversized (decompression-bomb) input, and the compiler measures the
compressed size of every candidate, so the budget holds exactly.

The decoder admits exactly one rank-one `U8` tensor, with every stored byte
accounted for. Stored payloads and inflation are capped at 64 MiB each; the
safetensors header and inflated JSON metadata are each capped at 1 MiB, with
at most 64 sections and 4096 features. Before materializing model arrays it
validates integer fields, schema/output agreement, 2–8-bit net/forest widths,
checked dimensions, exact section lengths, and acyclic in-range tree parents.
A separate 64 MiB conservative materialization charge covers metadata/schema
copies, section expansion, and coexisting numeric arrays and unpack temporaries;
this is not a whole-process RSS or sampling-output quota. Malformed or over-limit
artifacts return an artifact error before model allocation. These admission
limits do not change format version 1 or the bytes emitted by the encoder.

```python
import json, brotli                      # pip install brotli safetensors
from safetensors.numpy import load_file

blob = brotli.decompress(load_file("model.dope")["dope"].tobytes())
n = int.from_bytes(blob[:4], "little")
meta = json.loads(blob[4:4 + n])         # schema, structure, bit widths, section list
at, sections = 4 + n, {}
for name, size in meta["sec"]:
    sections[name], at = blob[at:at + size], at + size
```

Ablation switches: `--no-trees` (network only), `--no-audit` (fixed split), `--plan k,s,h`,
`--dependence independent|tree|factorN|full`, `--regimes K`, `--tolerance 0` (8-bit weights).

## Sampling on any device

A draw is a short tensor program — one GEMM for dependence, gathers for marginals, a
small network and one comparison per tree level plus one leaf gather per tree for the
target — written once against candle's `Tensor` API. The random
inputs come from a counter-based generator on the host, so a seed yields the same rows
on every device up to floating-point rounding. On CPU, `sample` evaluates the same draw
row by row in parallel (rayon) with no tensor temporaries; `dope throughput` measures it,
and a test keeps the native and tensor paths in agreement. Training uses a hand-vectorized f32
trainer on CPU and the candle trainer on CUDA/Metal; a test keeps the two in agreement.

## Library use

```rust
use dope_core::{decode, fit, sample, FitOptions, RawTable, SampleOptions};

let table = RawTable::read("train.csv".as_ref())?;
let fitted = fit(&table, table.column_index("price")?, &FitOptions { budget: Some(4096), ..Default::default() })?;
let program = decode(&fitted.bytes)?;                     // exactly what was evaluated
let rows = sample(&program, &SampleOptions { rows: 100_000, ..Default::default() })?;
```

## Layout

```
dope-core/src/
  table.rs schema.rs encode.rs output.rs      intake, ML-ready encoding, CSV output
  marginal.rs copula.rs regimes.rs screen.rs  marginals and dependence
  target.rs train.rs train_cpu.rs noise.rs    conditional model and noise
  forest.rs                                   boosted oblivious trees (byte-coded, error feedback)
  quant.rs artifact.rs artifact/               quantization, safetensors container (brotli), decoding
  compile.rs sample.rs device.rs              fit, sample, device selection
  eval/ (linear, gbdt, nn)                    retention auditors, the self-audit and the TDL frontier
  stats.rs rng.rs                             numerics and portable randomness
dope-cli/src/main.rs                          the `dope` command
```
