# D.O.P.E.: Downstream Objective-Preserving Encoding

Compile a sensitive table and its prediction target inside your firewall into a small **program** — train external AI models without moving complete records.

[![CI](https://github.com/neverhuman/dope/actions/workflows/ci.yml/badge.svg)](https://github.com/neverhuman/dope/actions/workflows/ci.yml)
[![Jankurai full-audit score snapshot](docs/readme/jankurai.svg)](https://github.com/neverhuman/dope/actions/workflows/jankurai.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CEO Brief PDF](https://img.shields.io/badge/CEO_Brief-PDF-D9480F.svg)](docs/briefs/dope-v3-ceo-brief.pdf)
[![CTO Brief PDF](https://img.shields.io/badge/CTO_Brief-PDF-1E3A5F.svg)](docs/briefs/dope-v3-cto-brief.pdf)
[![White Paper v3 PDF](https://img.shields.io/badge/White_Paper_v3-PDF-006D77.svg)](docs/whitepaper-v3/dope-v3.pdf)
[![arXiv Bundle](https://img.shields.io/badge/arXiv-Upload_Bundle-256ABF.svg)](docs/whitepaper-v3/ARXIV_UPLOAD.md)

| Executives & Leaders | CTOs & ML Architects | Full Technical Paper |
| :---: | :---: | :---: |
| **[CEO Brief (PDF)](docs/briefs/dope-v3-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md))<br>*Business case, scale proof & governance* | **[CTO Brief (PDF)](docs/briefs/dope-v3-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md))<br>*Rust codec, wire layout & Evidence Bank* | **[DOPE v3 White Paper (PDF)](docs/whitepaper-v3/dope-v3.pdf)**<br>*[arXiv Upload Bundle](docs/whitepaper-v3/ARXIV_UPLOAD.md) · [Receipts](bench/real-world/results/scoreboard/v3.0.0/README.md)* |

[Quick start](#quick-start) · [Real-data example](examples/v3-quickstart/README.md) · [Results](#results) · [Train a model](#train-the-neural-model) · [Roadmap & Co-Authorship](#roadmap--contributing) · [Docs](#documentation)

![Figure 1 from the DOPE v3 paper. The top panel shows how DOPE fits a file where the source table lives, then generates synthetic training rows elsewhere from the file and a seed. The bottom panel plots Transfer Score against whole-file size on the development report splits for DOPE, stored fitting rows, and four third-party generators. The paired interval against ARF includes zero.](docs/readme/dope-v3-figure-1.png)

## Quick start

Install [signed DOPE `v3.0.0`](https://github.com/neverhuman/dope/releases/tag/dope-engine-v3.0.0) on Linux x86-64 (`glibc 2.39+`, zero Python, zero GPUs, zero cloud calls). On Ubuntu, run `sudo apt-get install -y ca-certificates curl jq`. Binary installs to `$HOME/.local/opt/dope/3.0.0/bin/dope`.

```bash
(
  set -euo pipefail
  installer=$(mktemp)
  trap 'rm -f -- "$installer"' EXIT
  curl -q -fsSL --proto '=https' --tlsv1.2 -o "$installer" \
    https://raw.githubusercontent.com/neverhuman/dope/51df4554bc292c3ad59ec2099fccbbde19cffadc/scripts/install.sh
  printf '%s  %s\n' 808729d2382fc37fbdcf69520d808430536b858074387198920b250033baa213 "$installer" |
    sha256sum --check --status
  bash "$installer" --standalone-prefix "$HOME/.local/opt/dope/3.0.0" \
    --tag dope-engine-v3.0.0 --source-digest e15c415bad9554f7c419dd240aa670d1a79a27fe
)
```

### Run the toy example

Verifies the install on a toy table (test rows stay separate from fit).

```bash
(
  set -euo pipefail
  export LC_ALL=C
  dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
  cd "$(mktemp -d "${TMPDIR:-/tmp}/dope-quick-start.XXXXXX")"

  fixture='BEGIN {
    print "a,b,y"
    for (i=first; i<end; i++) {
      a=(17*i%997)/997; b=(31*i%991)/991
      print a "," b "," 2*a+b*b
    }
  }'
  awk -v first=0 -v end=160 "$fixture" > train.csv
  awk -v first=160 -v end=200 "$fixture" > test.csv

  "$dope" --threads 1 fit train.csv --target y --format 1 --budget 2048 --seed 42 -o model.dope
  bytes=$(wc -c < model.dope)
  test "$bytes" -le 2048
  printf 'Artifact size: %s bytes\n' "$bytes"
  "$dope" --threads 1 sample model.dope -n 640 --seed 7 -o synthetic.csv
  "$dope" --threads 1 eval -m model.dope --train train.csv --test test.csv \
    --synth synthetic.csv --auditors linear --seed 7 --json evaluation.json
  printf 'Workflow files: %s\n' "$PWD"
)
```

- **Real dataset (fast, offline):** Run `bash examples/v3-quickstart/run.sh` for an end-to-end UCI Airfoil lifecycle (`fit` → source-free `sample` → train/test on real holdout; [walkthrough](examples/v3-quickstart/README.md)).
- **macOS / ARM / older Linux / `PATH`:** Build offline via `cargo install --locked --path engine/dope-cli` (`Rust 1.88+`), or use Docker ([guide](docs/getting-started.md#install-the-signed-engine)).
- **Own CSV:** Follow [getting started](docs/getting-started.md#fit-a-table) (`--format auto-v8`, explicit `--target`, duplicate-grouped splits). [Paper appendix](docs/whitepaper-v3/dope-v3.pdf#nameddest=quick-start) · [Verify release](docs/signed-engine-release.md#verify-before-install) · [AGENTS.md](AGENTS.md)

## Results

Figure 1 plots Transfer Score against whole-file size on development report splits. Under the same byte cap, **DOPE transferred more predictive utility than storing and resampling fitting rows**; third-party generators ran uncapped with library defaults.

<!-- BEGIN DOPE1 SCOREBOARD -->
<!-- Generated by dope_kpi render-readme from dope.kpi-receipt/v2 receipts; do not edit. -->

**DOPE v3 ranks first at 87.9** (4× sampling); paired difference vs ARF is **+10.8 points** (95% CI -3.5 to +29.5; interval includes zero).

**Transfer Score = 100 × (1 + TRG).** A score of 100 matches a second real training sample. Higher is better.
On this report a copy of the real training rows scores **96.1**.

| Rank | Generator | Transfer Score ↑ | Median file |
|---:|---|---:|---:|
| 🟩 1 | DOPE v3 | 87.9 | 1,787 B |
| 🟦 2 | ARF | 77.2 † | 2.94 MB |
| ⬜ 3 | SDV Gaussian copula | 40.2 † | 141 KB |
| ⬜ 4 | SDV TVAE | -3.9 † | 475 KB |
| ⬜ 5 | SDV CTGAN | -69.2 † | 1.05 MB |
| — | *Copy of fitting rows* | *96.1* | *—* |
| — | *Bootstrap of fitting rows* | *91.2* | *—* |
| — | *Independent marginals* | *-22.2* | *—* |
| — | *Label shuffle* | *-16.4* | *—* |

† Every observed file exceeds 2048 B (0 files within 2048 B; Transfer Score shown is for those oversized files, whereas applying the 2048 B cap penalty without re-fitting scores 0.0).

A tie spans less than 1 point.

<!-- END DOPE1 SCOREBOARD -->

Across three development split sets, DOPE led Gaussian copula, TVAE, and CTGAN; the paired interval against ARF included zero on each set. Privacy checks are empirical diagnostics (no formal differential privacy mechanism and not HIPAA de-identification); `v3.0.0` is an uncertified `DEV-SELECT` release.

**[White paper (PDF)](docs/whitepaper-v3/dope-v3.pdf)** · **[CEO Brief (PDF)](docs/briefs/dope-v3-ceo-brief.pdf)** · **[CTO Brief (PDF)](docs/briefs/dope-v3-cto-brief.pdf)** · [Equal-budget comparison](bench/real-world/results/scoreboard/v3.0.0/evidence/addendum-k/) · [Full metrics & guards](bench/real-world/results/scoreboard/v3.0.0/README.md#full-benchmark-metrics-and-guard-results)

## Train the neural model

Release `v3.0.0` fits tables analytically with **no pretrained weights**. To train the optional `SingleEncV8` encoder-decoder (`research/levels/s3_v8_trainer.py`) and load your SafeTensors weights directly in pure Rust:

```bash
# 1. Build the Rust CLI, V8 packet emitter, and FFI library
cargo build --manifest-path engine/Cargo.toml --locked --release \
  -p dope-cli --bin dope -p dope-arena --bin dope_v8_emit
cargo build --manifest-path engine/Cargo.toml --locked --release -p dope-ffi

# 2. Run a 5-minute CPU smoke train on synthetic structural priors (needs PyTorch + safetensors)
python3 -m research.levels.s3_v8_trainer \
  --out-dir target/v8-quickstart --device cpu --steps 50 --episodes-per-step 2 \
  --d-model 64 --n-heads 4 --n-enc-layers 2 --n-dec-layers 2 \
  --use-rust-packet --budget-bytes 2048 --save-every 25 --seed 42

# 3. Load exported SafeTensors weights directly in pure Rust
engine/target/release/dope fit examples/v3-quickstart/data/airfoil_train.csv \
  --target sound_pressure_db --format v8 --budget 2048 \
  --v8-weights target/v8-quickstart/model_v8_ema.safetensors -o target/v8-quickstart/model.dope
```

See [docs/training.md](docs/training.md) for GPU arenas (`s3_v8_feed.py`), DiLoCo, and the Rust-vs-Python parity suite (`just v8-regression`).

## Roadmap & Contributing

The fastest way to push DOPE beyond `v3.0.0` is adding **data-specific features** to the FIT-only Evidence Bank (`contracts/evidence-layout-v1.json`) and running fast regression tests. External contributors whose merged work improves future versions (`v3.1+` / `v4.0`) are **invited as co-authors on future white papers, arXiv submissions, and `CITATION.cff`**. See [ROADMAP.md](ROADMAP.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

| Area | Quick Links |
| --- | --- |
| **Executive Briefs** | **[CEO Brief (PDF)](docs/briefs/dope-v3-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md)) · **[CTO Brief (PDF)](docs/briefs/dope-v3-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md)) |
| **White Paper & arXiv** | **[DOPE v3 PDF](docs/whitepaper-v3/dope-v3.pdf)** · [arXiv Upload Bundle](docs/whitepaper-v3/ARXIV_UPLOAD.md) · [LaTeX Sources](docs/whitepaper-v3/) |
| **Guides & Training** | [Real-dataset quickstart](examples/v3-quickstart/README.md) · [Getting started](docs/getting-started.md) · [Training](docs/training.md) · [Repo map](docs/REPO_MAP.md) |
| **Engine & Releases** | [Signed releases](docs/signed-engine-release.md) · [CLI](engine/dope-cli/) · [Rust SDK](engine/dope-sdk/) · [JopeDime pin](docs/jopedime-integration.md) |
| **Roadmap & Policy** | [ROADMAP.md](ROADMAP.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [Metrics](docs/metric.md) · [Release gates](docs/release.md) · [History](HISTORY.md) |

## Citation

```bibtex
@techreport{taylor2026dope,
  title       = {{D.O.P.E.}: Downstream Objective-Preserving Encoding},
  author      = {Taylor, Jepson and Butler, Ricky Ray and Morrow, Jordan and Alexander, Alton and Hildebrandt, Jordan and Jirongo, Besterlyne},
  institution = {NEVERHUMAN Research},
  year        = {2026},
  note        = {Release v3.0.0 white paper (docs/whitepaper-v3/dope-v3.pdf); source and receipts: https://github.com/neverhuman/dope}
}
```

Licensed under [MIT](LICENSE); see [CITATION.cff](CITATION.cff).

## Appendix: quick start and run examples

The first two blocks match [Quick start](#quick-start); the third follows [getting started](docs/getting-started.md#fit-a-table); the fourth shows the Ubuntu container.

```bash
(
  set -euo pipefail
  installer=$(mktemp)
  trap 'rm -f -- "$installer"' EXIT
  curl -q -fsSL --proto '=https' --tlsv1.2 -o "$installer" \
    https://raw.githubusercontent.com/neverhuman/dope/51df4554bc292c3ad59ec2099fccbbde19cffadc/scripts/install.sh
  printf '%s  %s\n' 808729d2382fc37fbdcf69520d808430536b858074387198920b250033baa213 "$installer" |
    sha256sum --check --status
  bash "$installer" --standalone-prefix "$HOME/.local/opt/dope/3.0.0" \
    --tag dope-engine-v3.0.0 --source-digest e15c415bad9554f7c419dd240aa670d1a79a27fe
)
```

```bash
(
  set -euo pipefail
  export LC_ALL=C
  dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
  cd "$(mktemp -d "${TMPDIR:-/tmp}/dope-quick-start.XXXXXX")"

  fixture='BEGIN {
    print "a,b,y"
    for (i=first; i<end; i++) {
      a=(17*i%997)/997; b=(31*i%991)/991
      print a "," b "," 2*a+b*b
    }
  }'
  awk -v first=0 -v end=160 "$fixture" > train.csv
  awk -v first=160 -v end=200 "$fixture" > test.csv

  "$dope" --threads 1 fit train.csv --target y --format 1 --budget 2048 --seed 42 -o model.dope
  bytes=$(wc -c < model.dope)
  test "$bytes" -le 2048
  printf 'Artifact size: %s bytes\n' "$bytes"
  "$dope" --threads 1 sample model.dope -n 640 --seed 7 -o synthetic.csv
  "$dope" --threads 1 eval -m model.dope --train train.csv --test test.csv \
    --synth synthetic.csv --auditors linear --seed 7 --json evaluation.json
  printf 'Workflow files: %s\n' "$PWD"
)
```

```bash
dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
"$dope" fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
wc -c < model.dope
"$dope" sample model.dope -n 1000 --seed 1 -o synthetic.csv

"$dope" fit train.csv --target y --format 1 --budget 2048 -o audit-schema.dope
"$dope" eval -m audit-schema.dope --train train.csv --test test.csv \
  --synth synthetic.csv --auditors linear --seed 1 --json evaluation.json
```

```bash
docker run --rm -it --platform linux/amd64 -v "$PWD":/work -w /work ubuntu:24.04 bash
```
