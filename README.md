# D.O.P.E. — Downstream Objective-Preserving Encoding

**Turn a real table into a small sampling program. Measure what models still learn.**

[![CI](https://github.com/neverhuman/dope/actions/workflows/ci.yml/badge.svg)](https://github.com/neverhuman/dope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DOPE learns a compact generator from your training table. The generator produces
synthetic rows; independent models learn from those rows and are tested on
untouched real data. The goal is simple: **preserve predictive lift using as
few fitted-state bytes as possible.**

![Historical selection-exposed validation: charged fitted-state bytes versus predictive lift for matched classical methods, including the untuned DOPE default; a separate foundation-model comparison. Driver preservation, TDL and cold closure remain unmeasured.](paper/generated/hero-frontier.png)

The figure shows the tradeoff: a small program can retain substantial measured
predictive lift. It includes the untuned default alongside the selected research
profile. The classical and foundation-model panels use separate cohorts; neither
establishes a sealed-test ranking or a TDL record. Pretraining overlap for the
foundation models is unknown. [Full evidence and limitations](#evidence).

**Start here:** [Real-data walkthrough](docs/getting-started.md) ·
[Preprocessing](docs/preprocessing.md) · [Metric](docs/metric.md) ·
[Active paper](paper/README.md) · [Publication review](docs/publication-upgrade.md)

## Run on real data

The public Airfoil example checks the official source, keeps physical groups
disjoint, compiles only fitting rows, and compares synthetic-trained and
real-trained learners on the same held-out measurements. The source-free worker
currently supports Linux x86-64. You need Rust, C build tools, and Python with
venv support for the optional example learner.

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

Open `target/public-demo/airfoil-1/report.html` for losses, predictive retention,
actual bytes and replay checks. Use a new output directory for each attempt.
The [walkthrough](docs/getting-started.md) explains the outputs, offline source
option, checksum installation and how to use your own table. Native compilation
and sampling require no Python or downloaded neural checkpoint. This demo uses
the raw symbolic profile; it does not reproduce the selected paper profile.

## How it works

| Stage | What DOPE does |
| --- | --- |
| Prepare | Resolve the declared target, place it last, split real rows, and fit numeric transforms and categorical vocabularies on fitting rows only |
| Compile | Encode marginals, dependence, a target model and residual uncertainty under an explicit complete-byte budget |
| Sample | Reload the artifact and generate rows using retained state; raw numeric samples return to original units |
| Evaluate | Train the same learner recipe on real or synthetic fitting rows, score on untouched real queries, and report transfer beside bytes |

Nominal strings use an explicit one-hot mixed-data carrier, with separate
missing and unknown states. Numeric affine scaling is the reproducible baseline.
The [preprocessing guide](docs/preprocessing.md) explains current limits and the
controlled alternatives worth testing. [The offline mixed example](examples/healthcare-sim/README.md)
shows categorical strings and missingness using simulated data.

## The KPI: Transfer Description Length

**How few bytes let different models recover what the real table teaches?**
[Transfer Description Length](docs/metric.md), or TDL, is the smallest **tested**
payload meeting `KiB@T90/D80`: at least nine-tenths of real-trained predictive
lift at the declared lower confidence bound (T90), and at least four-fifths
feature-driver agreement (D80). Copy, cleartext, positive-lift and byte-accounting
gates must also pass. Missing evidence stays unmeasured; an unreached budget
stays unreached.

This puts size in the objective and useful transfer in the constraints. A
lossless archive preserves rows; DOPE attempts to preserve a declared supervised
workload. It does not preserve every possible use of the original table.
Dataset-specific transforms, dictionaries, schema and sampling sidecars count.
The generic runtime and full cold sampling closure are reported separately.

## Evidence

The current figures report **historical, selection-exposed validation**. TDL,
cross-model feature-driver preservation and the full cold sampling closure are
unmeasured. The official test split is reserved for a frozen run. Empirical
privacy diagnostics do not establish formal differential privacy, and current
release scores remain null. The [evidence status](STATUS.md) and
[publication review](docs/publication-upgrade.md) identify the remaining work.

<details>
<summary>Historical benchmark tables, intervals and paired comparisons</summary>

<!-- BEGIN BENCHMARK -->
![Multi-model retention at synthetic size 4n against median charged artifact bytes on a log axis, and cross-architecture transfer across zero-shot tabular foundation models (NVIDIA Kumo-L, TabICLv2, Mitra-v2), classical learners, and historical feature-target correlation proxy (cross-model feature-driver preservation: not yet measured). DOPE's median artifact is 1.9 KB; the other generators' medians run from 2.5 KB to 325.8 MB.](docs/readme/pareto.png)

Validation panel: 100 PMLB regression tables across three task-trained model families (CatBoost, Linear, Neural MLP), synthetic size 4n. DOPE uses five fit seeds; intervals are 95% nested bootstrap intervals (simulated-family clusters, then lineages, then fit seeds). The difference column is DOPE minus the method on shared lineages (Hodges–Lehmann). The zero-shot tabular foundation-model block below is a separate 97-lineage matched historical cohort (fit seed 11; sample seeds 101, 211, 307; lineage-level bootstrap intervals).

#### 1. Multi-Model Task-Trained Transfer & Charged Artifact Bytes (Synthetic Size `4n`)

| Method | Lineages | CatBoost (4n) | Linear (4n) | Neural MLP (4n) | DOPE − method (CatBoost) | W/T/L | Median artifact | Within 10,240 B |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| **DOPE (ours)** | 99 | 0.898 [0.804, 0.972] | 0.991 [0.972, 0.997] | 1.133 [1.029, 1.246] | — | — | 1.9 KB | 98% |
| TabSyn (scaled schedule) | 99 | 0.924 [0.814, 0.965] | 0.952 [0.827, 0.987] | 1.013 [0.954, 1.142] | −0.003 [−0.053, 0.053] | 53/0/46 | 42.3 MB | 0% |
| ARF (author default) | 99 | 0.822 [0.689, 0.894] | 0.938 [0.846, 0.982] | 0.991 [0.886, 1.090] | +0.048 [−0.034, 0.149] | 60/0/39 | 1.4 MB | 0% |
| ARF (native selected) | 99 | 0.804 [0.647, 0.885] | 0.920 [0.828, 0.977] | 0.966 [0.843, 1.027] | +0.083 [0.006, 0.191] | 66/0/33 | 1.6 MB | 0% |
| Gaussian copula | 99 | 0.669 [0.369, 0.867] | 0.987 [0.963, 0.996] | 0.978 [0.892, 1.072] | +0.191 [0.089, 0.273] | 83/0/16 | 5.5 KB | 62% |
| Chow-Liu tree | 99 | 0.576 [0.294, 0.638] | 0.659 [0.282, 0.763] | 0.660 [0.347, 0.804] | +0.356 [0.242, 0.728] | 91/0/8 | 29.9 KB | 31% |
| Independent marginals | 99 | −0.012 [−0.040, −0.002] | 0.002 [−0.029, 0.019] | −0.049 [−0.161, −0.018] | +0.881 [0.790, 0.964] | 99/0/0 | 2.5 KB | 94% |
| TVAE (native selected) | 21 | 0.674 [0.452, 0.776] | 0.796 [0.581, 0.875] | 0.812 [0.644, 0.877] | +0.201 [0.097, 0.339] | 19/0/2 | 308.6 KB | 0% |
| CTGAN (native selected) | 21 | −0.080 [−0.282, 0.223] | −0.033 [−0.454, 0.124] | 0.033 [−0.338, 0.328] | +0.912 [0.731, 1.298] | 21/0/0 | 731.0 KB | 0% |
| Forest-Flow (historical) | 6 | 1.011 [0.905, 1.041] | 0.985 [0.977, 1.001] | 1.054 [0.929, 2.732] | −0.231 [−0.382, −0.005] | 0/0/6 | 325.8 MB | 0% |
| Real rows resampled (control) | 99 | 1.018 [1.005, 1.028] | 0.998 [0.991, 1.000] | 1.174 [1.051, 1.234] | −0.179 [−0.276, −0.074] | 10/0/89 | none | — |
| Predictor-only control (control) | 99 | 0.631 [0.375, 0.784] | 0.981 [0.955, 0.997] | 0.947 [0.883, 1.018] | +0.197 [0.112, 0.267] | 92/0/7 | none | — |

#### 2. Zero-Shot Tabular Foundation Models (TFM) & Historical Feature-Target Correlation Proxy

| Method | Lineages | NVIDIA Kumo Large (`kumo_tabular_l`) | TabICLv2 (`tabicl2`) | AutoGluon Mitra-v2 (`mitra_v2`) | Feature-Target Correlation Proxy (`driver_fidelity` proxy) | Matched median artifact |
| --- | ---: | --- | --- | --- | --- | ---: |
| **DOPE (ours)** | 97 | 0.863 [0.809, 0.880] | 0.871 [0.835, 0.893] | 0.858 [0.803, 0.887] | 0.905 (IQR [0.589, 1.000]) | 1.8 KB (1,844 B) |
| Gaussian copula | 97 | 0.747 [0.668, 0.778] | 0.754 [0.672, 0.787] | 0.713 [0.585, 0.771] | 0.975 (IQR [0.743, 1.000]) | 5.5 KB (5,464 B) |
| Chow-Liu tree | 97 | 0.562 [0.501, 0.629] | 0.587 [0.484, 0.650] | 0.583 [0.527, 0.643] | 0.775 (IQR [0.452, 0.971]) | 29.6 KB (29,635 B) |
| Independent marginals | 97 | −0.001 [−0.002, 0.000] | −0.004 [−0.012, 0.001] | −0.009 [−0.014, −0.004] | 0.304 (IQR [0.220, 0.400]) | 2.5 KB (2,538 B) |

On these matched historical TFM lineages (fit seed 11; sample seeds 101, 211, 307; synthetic size 4n; pretraining overlap is unknown), Gaussian copula is the strongest measured comparator on this historical block (not universally strongest), and cross-model feature-driver preservation is not yet measured (the proxy column reports absolute feature-target correlation agreement; Gaussian copula is 0.975 median of lineage medians vs. 0.923 median of lineage minima from historical assembler::_min; pooled-seed median is 0.973).

![DOPE minus each comparator, Hodges–Lehmann retention difference with 95% nested intervals, under the CatBoost, linear, and MLP auditors.](docs/readme/paired.png)

In the paired chart the axis runs from -0.3 to 0.7: an interval end beyond it is drawn as an arrow at the edge, and a point beyond it is drawn at the edge with its value printed.

The strongest full-panel comparator is TabSyn (scaled schedule). These are validation estimates on a training-derived cut; the official test split is reserved for one pre-registered run. Resampled real rows are a row-count diagnostic, neither a generator nor an upper bound. The release score (MFS-v3) is null for every method because its prerequisites and cleartext scan are unmeasured, and measured copy, membership, and attribute gates already fail on some cells. DOPE's median artifact is 1.9 KB, and 490 of 500 individual DOPE fits are within the 10,240-byte release cap; the paper names the lineages above it.
<!-- END BENCHMARK -->

</details>

<details>
<summary>Generated TDL evidence card and cohort qualifications</summary>

<!-- BEGIN TDL RESULTS -->
<!-- Generated by tools/claims/render.py; edit headline-results.json. -->
Historical validation; selection-exposed. TDL and cross-model feature-driver preservation (`D80`) are unmeasured.

#### Historical Foundation-Model Block (`mfs-v3-density-panel.json`, 97 matched mixed/simulated lineages, fit seed 11, sample seeds 101/211/307, 4n context)

| Method | Kumo Large [historical 95% interval] | TabICLv2 [historical 95% interval] | Mitra-v2 [historical 95% interval] | Matched charged bytes | Cross-model driver preservation | Target-correlation proxy (94/97 lineages) |
|---|---:|---:|---:|---:|---|---:|
| DOPE (ours) | 86.3% [80.9%, 88.0%] | 87.1% [83.5%, 89.3%] | 85.8% [80.3%, 88.7%] | 1,844 B | `unmeasured` | 0.905 [0.589, 1.000] (`absolute_target_correlation`) |
| Gaussian copula | 74.7% [66.8%, 77.8%] | 75.4% [67.2%, 78.7%] | 71.3% [58.5%, 77.1%] | 5,464 B | `unmeasured` | 0.975 [0.743, 1.000] (`absolute_target_correlation`) |
| Chow-Liu tree | 56.2% [50.1%, 62.9%] | 58.7% [48.4%, 65.0%] | 58.3% [52.7%, 64.3%] | 29,635 B | `unmeasured` | 0.775 [0.452, 0.971] (`absolute_target_correlation`) |
| Independent marginals | -0.1% [-0.2%, 0.0%] | -0.4% [-1.2%, 0.1%] | -0.9% [-1.4%, -0.4%] | 2,538 B | `unmeasured` | 0.304 [0.220, 0.400] (`absolute_target_correlation`) |

Gaussian copula is the strongest measured comparator on this 97-lineage historical foundation block, not universally strongest (TabSyn leads the 99-lineage classical CatBoost validation panel at 42,261,045.5 B). 0.975 is the 94-lineage median of per-lineage medians across three seeds; 0.923 (0.923106) is the 94-lineage median of per-lineage minimum seed values (matching historical assembler::_min, not a pooled-seed median, which is 0.973214). All are absolute feature-target association proxies from rust/certification/privacy_attacks.rs::driver_fidelity (absolute_target_correlation); cross-model Kumo/TabICL/Mitra feature-driver preservation remains unmeasured. Foundation-model auditor qualification is pending and pretraining overlap is unknown.

#### Historical Classical CatBoost Block (`v2-headline.json`, selection-exposed)

| Method | Retention [historical 95% interval] | Median charged artifact bytes | Observed lineages | Evidence status |
|---|---:|---:|---:|---|
| DOPE (ours) | 0.898 [0.804, 0.972] | 1895.0 | 99/100 | `selection_exposed_validation` |
| TabSyn (scaled schedule) | 0.924 [0.814, 0.965] | 42261045.5 | 99/100 | `selection_exposed_validation` |
| ARF (author default) | 0.822 [0.689, 0.894] | 1438304.5 | 99/100 | `selection_exposed_validation` |
| Gaussian copula | 0.669 [0.369, 0.867] | 5464.5 | 99/100 | `selection_exposed_validation` |
| Chow-Liu tree | 0.576 [0.294, 0.638] | 29865.0 | 99/100 | `selection_exposed_validation` |
| Independent marginals | -0.012 [-0.040, -0.002] | 2549.5 | 99/100 | `selection_exposed_validation` |
| CTGAN (native selected) | -0.080 [-0.282, 0.223] | 730999.0 | 21/100 | `selection_exposed_validation` |
| TVAE (native selected) | 0.674 [0.452, 0.776] | 308629.5 | 21/100 | `selection_exposed_validation` |
| Forest-Flow (historical) | 1.011 [0.905, 1.041] | 325848913.5 | 6/100 | `selection_exposed_validation` |
| Predictor-only control | 0.631 [0.375, 0.784] | unmeasured | 99/100 | `selection_exposed_validation` |
| Real rows resampled | 1.018 [1.005, 1.028] | unmeasured | 99/100 | `selection_exposed_validation` |

Bytes are historical charged artifacts; cold sampling closure and cross-model feature-driver preservation (`D80`) are unmeasured.
Smaller cohorts are separate comparisons. These points do not establish a TDL frontier.
<!-- END TDL RESULTS -->

</details>

<details>
<summary>Pinned engineering examples and byte accounting</summary>

<!-- BEGIN PINNED DEMOS -->

<!-- Generated historical engineering notes; not a TDL result. -->

Inspect of the artifact built for this page reports `artifact_bytes` 151, `rows_fitted` 400, three positional features, dependence `chow_liu`, and target `sparse_linear`. `compliant` is false: compile sees `train.csv` only and does not certify a release. The transcript is [examples/readme-regression/INSPECT.txt](examples/readme-regression/INSPECT.txt). This toy is not the benchmark above.

The installed public demo in [examples/public-regression](examples/public-regression/) fits the UCI Airfoil Self-Noise table (Brooks, Pope, and Marcolini; CC BY 4.0; DOI 10.24432/C5VW2C) on a grouped development split and writes one native raw-unit artifact, `model.raw` (SHA-256 `0950cbe79817…`), with profile `raw-numeric-symbolic-v1` under the byte profile `native-raw-complete-v1`. Compile, `inspect`, the demo's research receipt, and the file size all report 1,558 bytes: 56 bytes of framing, 1,080 bytes of fitted schema and transforms, and 422 bytes of sampling payload. The file is its own transport archive. The generic runtime is measured separately: the debug binary and its mapped libraries came to 72,049,152 bytes, so the cold sampling closure was 72,050,710 bytes. No portable bundle or release-build runtime was measured. The run used commit `59437b1`, and privacy, driver, and geometry checks are unmeasured for this artifact. It is an engineering conformance example on one development split, not a benchmark result, and a different object from both the toy above and the paper's `.dpk` plus `projection.json` charge. The pinned values are in [docs/whitepaper/receipts/native-byte-conformance-v1.json](docs/whitepaper/receipts/native-byte-conformance-v1.json), and the supplement repeats them.

<!-- END PINNED DEMOS -->

</details>

## Paper and reproduction

The active manuscript and supplement live in [paper/](paper/README.md).
The [historical validation paper](docs/whitepaper/README.md) remains available
with its original evidence qualifications. These are separate build targets.

```sh
python3 -B tools/claims/render.py --check  # Receipts, generated figures and cards
python3 -B tools/claims/check.py          # Publication claim checks
just paper-tdl                          # Active manuscript and supplement
just paper-tests                        # Historical paper pipeline tests
just paper-check                        # Historical paper rebuild and drift checks
just fast                               # Fast product and compatibility checks
just check                              # Full local pre-commit checks
```

See [testing](docs/testing.md) for dependencies and budgets, and the
[publication review](docs/publication-upgrade.md) for the proposed benchmark,
math ablations, submission gaps and staged compute request.

## Contribute

Start with [CONTRIBUTING.md](CONTRIBUTING.md), the
[architecture](docs/architecture.md), [CLI reference](docs/cli.md), and
[repository guide](AGENTS.md). Preserve historical decoding and attach measured
receipts to new performance claims. Bugs and poor-transfer cases are useful:
include the command, artifact profile and redacted receipt rather than source rows.

## Citation

```bibtex
@misc{taylor2026dope,
  title  = {{D.O.P.E.}: Downstream Objective-Preserving Encoding -- Generative Compression of Predictive Structure},
  author = {Taylor, Jepson and Alexander, Alton},
  year   = {2026},
  note   = {NEVERHUMAN Research. Source and receipts: https://github.com/neverhuman/dope}
}
```

See [CITATION.cff](CITATION.cff). Licensed under [MIT](LICENSE).

