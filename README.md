# D.O.P.E. — Downstream Objective-Preserving Encoding

**Turn a real table into a small program that generates useful training data.**

DOPE learns a compact sampling program from a declared fitting table. Another machine can generate new rows from that program, train its own models, and measure how much predictive gain survives on held-out real data. The program carries learned distributions, dependencies, target structure, and noise; it does not need the source table at sampling time.

[![CI](https://github.com/neverhuman/dope/actions/workflows/ci.yml/badge.svg)](https://github.com/neverhuman/dope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Historical selection-exposed validation: DOPE and Gaussian copula payload bytes beside predictive transfer across Kumo Large, TabICLv2 and Mitra-v2. Cross-model feature-driver preservation is not yet measured.](paper/generated/hero-frontier.png)

**What the evidence supports:** compact sampling programs with strong historical validation transfer. The figure compares the same mixed/simulated-lineage foundation-model cohort. TabSyn leads CatBoost retention in the separate classical panel; DOPE uses a much smaller artifact. A qualified TDL result and a universal compression ranking remain unmeasured. Synthetic artifacts still need a separate release assessment before leaving a sensitive environment.

[Paper](paper/README.md) · [Benchmark evidence](#benchmark-evidence) · [Metric](docs/metric.md) · [Rust deployment](deploy/README.md) · [Technical review and experiments](docs/technical-upgrade.md)

## Start here: native Rust, no Python runtime

Build from source with the pinned Rust toolchain. The supported native installation profile is Linux x86-64; see [installation](docs/native-workflow.md) for checksum installation and platform limits.

```sh
git clone https://github.com/neverhuman/dope.git
cd dope
cargo build --locked --release --bin dope --bin dope-kernel

# A small engineering fixture to learn the interface.
# Use fresh output names: commands refuse to overwrite prior results.
target/release/dope demo --out target/quickstart.dpk --report target/quickstart.json
target/release/dope inspect --artifact target/quickstart.dpk
target/release/dope sample --artifact target/quickstart.dpk \
  --rows 200 --seed 1729 --out target/quickstart.csv
```

The fixture is simulated and establishes no benchmark result. For your own numeric regression table, declare the schema and target explicitly, supply only fitting rows, then run:

```sh
target/release/dope compile --csv fit.csv --schema schema.json \
  --out target/model.raw --max-complete-bytes 32768
target/release/dope sample --artifact target/model.raw \
  --rows 1000 --seed 91 --out target/synthetic.csv
```

[CLI reference](docs/cli.md) explains the schema, bounds, errors, and supported profiles. Every new canonical model table has predictors followed by its explicitly declared target. Native intake resolves that declaration before placing the target last; an external source's final column is never assumed to be its label.

## Walk through a real open dataset

The [Airfoil example](examples/public-regression/) downloads and verifies UCI Airfoil Self-Noise (CC BY; see the source license), groups repeated physical configurations into disjoint fit/query partitions, compiles **fit rows only**, generates synthetic data at `n` and `4n`, and compares the same Ridge recipe on untouched real queries in the original sound-pressure units.

After the native build above:

```sh
python3 -m venv target/demo-env
target/demo-env/bin/python -m pip install -r validation/requirements-v1.txt
target/demo-env/bin/python -I -B examples/public-regression/run.py \
  --binary "$PWD/target/release/dope-kernel" \
  --out target/public-demo/airfoil-run-1
```

Open `target/public-demo/airfoil-run-1/report.html`. It reports actual artifact bytes, real/null/synthetic losses, retention, source-free replay, and measured costs. Python runs the optional external learner and report; the deployed compiler and sampler are Rust. A pinned local source can be supplied with `--source`. This is one development example, not independent confirmation or a privacy certificate.

## How the compiler works

1. **Prepare the fitting rows.** Validate types, declared target, feature roles, masks, and domains. Fit numeric transformations and categorical dictionaries on fitting rows only.
2. **Encode predictive structure.** Fit bounded marginals and dependence models, a target program, and residual noise. Select representations under an actual serialized-byte cap.
3. **Sample elsewhere.** Decode the charged artifact without opening fitting rows. The same seed reproduces the declared numerical profile.
4. **Test what survived.** Train or condition independent learners on synthetic rows, then score them on untouched real rows. Count the complete payload and report the cold runtime closure separately.

Nominal strings have no numeric ordering. The [mixed Rust profile](docs/native-workflow.md#explicit-mixed-development-profile) uses column-scoped dictionaries and one-hot coordinates; it supports bounded nominal/mask inputs through `dope-kernel compile-mixed` and emits typed cells. The simple `dope compile` facade currently accepts fully observed numeric regression. [Preprocessing guidance](docs/preprocessing.md) explains these boundaries and the proposed robust-transform ablations.

## Our KPI: useful transfer at the smallest tested byte budget

**Transfer Description Length (`TDL`), written `KiB@T90/D80`, asks how few dataset-specific bytes preserve what models learn.** The working operating point requires at least nine-tenths of real-trained predictive lift at the lower confidence bound (T90), and at least four-fifths of real-trained feature-driver agreement (D80), across the required learner families. Copy, cleartext, control-lift, and byte-accounting gates must also pass.

Bytes are minimized subject to those constraints. More synthetic rows alone do not improve the score. Missing prerequisites produce an explicit incomplete result; unsuccessful tested budgets remain visible. This is task-specific generative compression, with Shannon information, transfer to arbitrary targets, and formal differential privacy requiring separate evidence. See [the exact definitions](docs/metric.md) and [pending protocol decisions](docs/decisions.md).

## Benchmark evidence

The hero shows the matched historical foundation-model comparison. The classical panel covers different cohorts and recipes. Both are selection-exposed validation; official test confirmation remains reserved. Historical charged artifact bytes and native complete raw-artifact bytes are different profiles, and their runtime closures must be reported separately.

Frozen historical PDFs: [validation paper](docs/whitepaper/dope-mfs.pdf) and [supplement](docs/whitepaper/supplement.pdf). The [active manuscript](paper/README.md) develops the TDL protocol and compiler explanation.

<details>
<summary>Classical and foundation-model tables, paired comparisons, uncertainty, and coverage</summary>

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
<summary>Generated evidence card and TDL measurement status</summary>

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
<summary>Pinned native byte-accounting examples</summary>

<!-- BEGIN PINNED DEMOS -->

<!-- Generated historical engineering notes; not a TDL result. -->

Inspect of the artifact built for this page reports `artifact_bytes` 151, `rows_fitted` 400, three positional features, dependence `chow_liu`, and target `sparse_linear`. `compliant` is false: compile sees `train.csv` only and does not certify a release. The transcript is [examples/readme-regression/INSPECT.txt](examples/readme-regression/INSPECT.txt). This toy is not the benchmark above.

The installed public demo in [examples/public-regression](examples/public-regression/) fits the UCI Airfoil Self-Noise table (Brooks, Pope, and Marcolini; CC BY 4.0; DOI 10.24432/C5VW2C) on a grouped development split and writes one native raw-unit artifact, `model.raw` (SHA-256 `0950cbe79817…`), with profile `raw-numeric-symbolic-v1` under the byte profile `native-raw-complete-v1`. Compile, `inspect`, the demo's research receipt, and the file size all report 1,558 bytes: 56 bytes of framing, 1,080 bytes of fitted schema and transforms, and 422 bytes of sampling payload. The file is its own transport archive. The generic runtime is measured separately: the debug binary and its mapped libraries came to 72,049,152 bytes, so the cold sampling closure was 72,050,710 bytes. No portable bundle or release-build runtime was measured. The run used commit `59437b1`, and privacy, driver, and geometry checks are unmeasured for this artifact. It is an engineering conformance example on one development split, not a benchmark result, and a different object from both the toy above and the paper's `.dpk` plus `projection.json` charge. The pinned values are in [docs/whitepaper/receipts/native-byte-conformance-v1.json](docs/whitepaper/receipts/native-byte-conformance-v1.json), and the supplement repeats them.

<!-- END PINNED DEMOS -->

</details>

## Rust deployment and development

The existing native sampler needs no Python or libtorch in its default CPU build. [deploy/](deploy/) adds a separate, locked Rust runtime with SafeTensors transport and optional batched Candle CUDA inference for TVAE decoders. `--device auto` uses the available supported backend; `--device cuda` refuses unavailable or unsupported execution. Other neural architectures retain native CPU sampling. CUDA execution and corrected-activation transfer still need qualification on actual hardware.

SafeTensors stores tensors; it does not supply a model architecture, certify a generator, or make a payload smaller. The initial carrier retains native DPK plus typed decoder tensors, charges the whole file, and explicitly preserves legacy numerical behavior. Use native DPK for the compact symbolic path. [The deployment guide](deploy/README.md) documents exact capabilities and validation.

```sh
just fast           # Rust + compatibility + bounded research fixtures
just check          # required full local pre-commit checks
just paper-tdl      # active manuscript + supplement; checks Figure 1 on page 1
just paper-tests    # historical publication regression suite
just paper-check    # historical paper reproducibility and drift checks
```

[Contributing](CONTRIBUTING.md) · [Architecture](docs/architecture.md) · [Testing](docs/testing.md) · [Release](docs/release.md) · [Evidence status](STATUS.md) · [Security](SECURITY.md)

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

