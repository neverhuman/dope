# MFS history and moved README reference

**Archived 2026-10-15 (T-101). Historical: do not extend.** This page holds the root README
material that the 2026-10-15 rebuild moved out of the landing page. It is kept verbatim so
links, receipts and reviews that cite it still resolve. The active metric is TDL
([docs/metric.md](../metric.md)), which is specified and unmeasured. The active evidence is the
v2 validation panel ([docs/benchmark.md](../benchmark.md)). The claims rules are in
[docs/claims-policy.md](../claims-policy.md).

## Status of the retired scores

| Score | Contract | Status |
| --- | --- | --- |
| MFS-v1 / v2 | `production/kpi-contract-v1.json`, `production/kpi-contract.json` (v2 is the embedded production contract) | historical; v2 is the contract of the active v2 evidence, and its release score is null for every method |
| MFS-v3 | `production/kpi-contract-v3.json`, printed in the historical paper | historical; null for every method because its prerequisites and cleartext scan are unmeasured, and measured copy, membership and attribute gates already fail on some cells |
| MFS-v4 | `production/kpi-contract-v4.json` (`rust/fitness/v4.rs`, `research/benchmark/mfs_v3_score.py`) | historical draft; never measured; superseded by the TDL contract (T-003, T-004) |

The MFS-v4 contract names Kumo Tabular (`kumo_tabular_l`), Mitra-v2 and TabICLv2 as required
auditors, and uses Kumo as its representation encoder. **All three are planned and
unqualified**: their locks have not passed source, weight, environment, parity and resource
verification (T-061, T-065). A model named in a contract is not an evaluated model. The
`knowledge_compression` term below is a weighted compactness score. It is not a
description-length code, and it is retired from every headline (T-102). `encoded_numbers`
changes with packing and never ranks anything.

## Moved: preprocessing contract and MFS-v4 metric (README, 2026-10-15)

### Standard `[0, 1]` numeric preprocessing and MFS-v4 metric

#### Standard `[0, 1]` numeric and categorical preprocessing contract

Released `.dpk` artifacts and public sidecars omit source strings, column headers, and raw extrema. DOPE enforces a numeric-only `[0, 1]` contract ([rust/data.rs](../../rust/data.rs), [rust/error.rs](../../rust/error.rs), and [rust/representation.rs](../../rust/representation.rs) in Rust; `research.transfer.metrics.standardize_numeric_table_01` and [research/benchmark/representation.py](../../research/benchmark/representation.py) in Python):

- **Headerless numeric `[0, 1]` input**: L3 `compile`, `convert`, and `certify` consume headerless numeric CSV files with the target in the final column and every cell already in `[0, 1]`. Out-of-range values fail closed with normalization guidance.
- **Categorical bucket projection into `(0, 1)`**: Raw categorical tokens are mapped deterministically into `(0, 1)` before compilation using 65,536-bucket HMAC-SHA256 projection (`(bucket + 0.5) / 65536`), so `.dpk` artifacts store only numeric values and no cleartext strings. `rust/representation.rs` and `research/benchmark/representation.py` require a caller-supplied salt; in `standardize_numeric_table_01`, the fixed default salt (`dope-standard-01-v1`) provides deterministic pseudonymization rather than secrecy against dictionary guessing of low-cardinality vocabularies.
- **Numeric scaling to `[0, 1]` (fit on training split only)**: Continuous columns are standardized using training-split statistics only—either by dividing by the training standard deviation (`z_i = (x_i - μ_train) / σ_train`) and mapping smoothly onto `[0, 1]` via the standard normal CDF `Φ(z_i)` (`zscore_std_01`), by training min-max clamping (`minmax_01`), or by empirical mid-rank quantiles (`midrank_quantile_01`).

#### MFS-v4 quality metric: downstream tabular lift + feature-importance preservation

When tabular learners (CatBoost, LightGBM/tree ensembles, MLPs, linear models) train on synthetic rows and are evaluated on unseen real holdout rows, a synthetic generator should both beat trivial controls on predictive skill and preserve the key feature drivers. MFS-v4 ([production/kpi-contract-v4.json](../../production/kpi-contract-v4.json), verified against [schemas/metric-golden-v1.json](../../schemas/metric-golden-v1.json)) combines seven `[0, 1]` components with a gated geometric mean (`V4_WEIGHTS`):

| Component | Weight | What it measures on unseen real holdout data |
| --- | ---: | --- |
| `utility_transfer` | 0.30 | Null-normalized TSTR/TRTR retention blended with **control-contrastive lift** (`control_contrastive_lift`: `clamp_0_1(0.65 * clamp_0_1(retention) + 0.20 * clamp_0_1(max(0, retention - predictor_only) / 0.20) + 0.15 * clamp_0_1(max(0, retention - independent_marginals) / 0.25))`) |
| `driver_fidelity` | 0.20 | Holdout **feature-importance and attribution agreement** (`driver_attribution_agreement`) between synthetic-trained and real-trained models: `0.35 * NDCG@k + 0.25 * Top-k Jaccard + 0.25 * L1 attribution agreement + 0.15 * sign agreement`, with `k = max(1, min(features, active_real_features, clamp(ceil(0.10 * features), 5, 20)))`, `real_columns > len(real_importances)` (since `real_columns` includes the target column), a non-zero directional sign (`±1`) required on every active driver (`importance > 1e-12`), and only active features (`> 1e-12`) entering top-`k`, DCG, and main-effect sign agreement |
| `representation_closeness` | 0.15 | Encoder distance (`l2_unit_half` under `kumo_tabular_l`) against a matched null (`median_linear_mse_of_three`) on the shared `empirical_midrank_quantile_v1` map ([research/benchmark/representation.py](../../research/benchmark/representation.py)) |
| `knowledge_compression` | 0.15 | Composite MDL knowledge compression (`knowledge_compression_metrics`): `0.40 * bpc_score + 0.35 * parameter_sparsity_score + 0.25 * log_byte_score` (`1 / (1 + bits_per_original_cell)`, `1 / (1 + encoded_numbers / (real_columns * 8.0))`, and `clamp_0_1(1 - ln(max(1, artifact_bytes)) / ln(10240))`) when `retained_utility > 0` and `artifact_bytes <= 10240`, and `0.0` otherwise (so unverified compile-only reports with `retained_utility = 0.0` and L0/L1 artifacts over 10 KiB report `0.0`) |
| `distribution_fidelity` | 0.10 | Marginal 1-Wasserstein / Kolmogorov–Smirnov fidelity across features |
| `structure_fidelity` | 0.05 | Pairwise correlation, mutual-information, and target-association preservation |
| `coverage_realism` | 0.05 | Support precision and recall (`alpha_precision`, `beta_recall`, coverage balance) |

Every MFS-v4 release score fails closed (`null`) unless all six prerequisites (`real_vs_real_control_complete`, `required_size_cells_complete`, `artifact_sampling_verified`, `metric_implementations_locked`, `control_contrastive_baselines_complete`, `model_driver_attribution_measured`; MFS-v3 uses the first four) and all hard release gates (`exact_row_matches == exact_row_match_max (0)`, `near_copy` via `squared_l2_on_shared_map` against `nearest_other_distinct_fit_row` at the `0.01` quantile, `cleartext_absent == true`, `membership_auc <= membership_auc_max (0.55)`, `attribute_inference_advantage <= attribute_inference_advantage_max (0.05)`, `artifact_bytes <= byte_limit_l3 (10240)`, `encoded_numbers >= encoded_numbers_min (1)`, `predictor_only_lift > predictor_only_lift_min (0.0)`, `independent_marginals_lift > independent_marginals_lift_min (0.0)`, and positive top-`k` driver overlap) pass.

Across a three-seed lineage (`sample_seeds = [101, 211, 307]`, evaluated by `evaluate_v4_lineage` in [rust/fitness/v4.rs](../../rust/fitness/v4.rs) and `score_lineage_v4` in [research/benchmark/mfs_v3_assemble.py](../../research/benchmark/mfs_v3_assemble.py)), every seed must individually pass `evaluate_v4` (`all_seeds_eligible`) under identical dataset and protocol invariants (`lineage_schema_consistent`). Lineage utility uses a one-sided lower confidence bound (`retention_lcb <= retention_mean + 1e-12`, refusing optimistic bounds) evaluated against the worst-seed (`max`) `predictor_only_retention` and `independent_marginals_retention` controls; `driver_fidelity` and `representation_closeness` come from the worst-performing seeds (`min`), `knowledge_compression` uses the worst-case (`max`) `artifact_bytes` and `encoded_numbers`, and empirical privacy uses the worst-case (`max`) `membership_auc` and `attribute_inference_advantage`. Eligible lineages report `seed_variance` (`retention_mean`, `retention_sample_sd`, `retention_min`, `retention_max`, `retention_lcb`, `driver_fidelity_mean`, `driver_fidelity_sample_sd`, `driver_fidelity_min`, `driver_fidelity_max`, `seed_score_mean`, `seed_score_sample_sd`, `seed_score_min`, and `seed_score_max`).

## Moved: artifact format, GPU training and command reference (README, 2026-10-15)

### Artifact format

DPK3.2 uses canonical varints, fixed-point parameter ladders, deterministic static byte-rANS sections, a versioned header, and a BLAKE3 checksum. Sections 3–5 are symbolic; section 6 is neural and cannot fall back to a symbolic target. The decoder retains V1 JSON and V2.0/V2.1 plus V3.0/V3.1 binary compatibility. New compact neural shapes and direct-rank TabDDPM use DPK3.3; existing full-model DPK3.2 bytes remain unchanged. JSON reports, inspectable language dictionaries, and S-expression disassembly are sidecars only.

L3 `compile`, `convert`, and `certify` expect headerless numeric CSV files with the target in the final column and values already in `[0, 1]`. Out-of-range values fail with normalization guidance. L3 has a hard 10,240-byte artifact limit, L2 32,768 bytes, L1 no hard limit, and L0 is research use. A compile may return an uncertified artifact within its limit; release requires measured utility, structure, and empirical privacy gates. A tier or empirical privacy result does not establish formal DP or HIPAA de-identification.

The compiler fits sparse linear/logistic, GAM, GA²M, MARS, oblivious-tree, and compact neural-residual targets and places them in the same byte/quality beam. Candidate reports identify the selected target and dependence families; encoder fitting cost is not included in the artifact byte count. `.dpk` artifacts contain no source rows, column names, embedded reports, or opaque model states.

<details>
<summary>GPU training, research comparators, and V1 compatibility</summary>

Joint training is enabled with `--features gpu-training` and requires CUDA, libtorch 2.7, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and deterministic cuDNN settings. Seeded libtorch work is serialized process-wide, and joint trainers use deterministic bounded row minibatches. Micro-TVAEs use latent/hidden widths 4/16, 8/24, and 12/32; TinyMAT profiles use widths 16 and 24. Their per-channel int8 tensors, f32 scales and activations, normalization metadata, permutations, target marginal, and training/implementation hashes are encoded and charged to the artifact. Direct rank-space TabDDPM is a GPU research comparator eligible only for L1 certification, with no L3 size claim. Exported artifacts sample in Rust only. The historical deep names (`ctgan`, `taegan`, `tabddpm`, and `masked_diffusion_transformer`) remain explicit unavailable entries. Seven `legacy_*` IDs preserve the prior symbolic-plus-neural-residual comparators. The Python package under `src/dope_kernel` remains the V1 baseline; no V3 hot path calls Python.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 LIBTORCH_USE_PYTORCH=1 cargo build --release --features gpu-training
```

</details>

### Command reference

<details>
<summary>All <code>dope-kernel</code> commands</summary>

```bash
target/release/dope-kernel convert --dataset-dir data --real-holdout-dir data --task regression --out release
target/release/dope-kernel compile --dataset-dir data --task regression --out kernel.dpk
target/release/dope-kernel compile --dataset-dir data --task regression --language run/language.json --out kernel.dpk
target/release/dope-kernel inspect --kernel kernel.dpk
target/release/dope-kernel sample --kernel kernel.dpk --rows 1000 --out synth.csv
target/release/dope-kernel embed-dataset --csv train.csv --target-column outcome --task regression --router-bundle router.bundle.json --router-evidence router-evidence.json --out dataset-embedding.json --vector-out dataset-embedding.csv
target/release/dope-kernel compile --dataset-dir data --task regression --candidate tvae --neural-target-weight 2 --neural-structural-penalty 0.1 --out tvae.dpk
target/release/dope-kernel inventory-corpus --corpus corpus-a --corpus corpus-b --out inventory
target/release/dope-kernel pack-corpus --corpus corpus-a --corpus corpus-b --out packed
target/release/dope-kernel calibrate-language --corpus packed --out run
target/release/dope-kernel certify --real-dir data --kernel kernel.dpk --out certification.json
target/release/dope-kernel campaign freeze-deep-implementation --source-tree . --binary target/release/dope-kernel --out deep-freeze.json
target/release/dope-kernel campaign plan-deep-cohorts --training-gold training-gold.json --out deep-cohorts.json
target/release/dope-kernel campaign materialize-deep-cohort --plan deep-cohorts.json --stage confirmation --out confirmation.json
target/release/dope-kernel campaign build-deep-evidence --cohort confirmation.json --block campaign/blocks --out deep-outcomes.json
target/release/dope-kernel campaign select-deep-discovery --evidence discovery-outcomes.json --out discovery-freeze.json
target/release/dope-kernel campaign qualify-deep-confirmation --candidate tvae --evidence deep-outcomes.json --scheduled-cells 64908 --out tvae-qualification.json
target/release/dope-kernel campaign report-deep-campaign --evidence deep-outcomes.json --out deep-report.json
target/release/dope-kernel campaign select-deep-final --evidence validation-select-outcomes.json --qualification tvae-qualification.json --out winner.json
target/release/dope-kernel campaign build-deep-artifact-manifest --selection winner.json --artifact DATASET_ID=representative.dpk --representative-dataset-id DATASET_ID --out artifacts.json
target/release/dope-kernel campaign build-deep-model-card --selection winner.json --report deep-report.json --out MODEL_CARD.md

# V1 compatibility commands
dope-kernel fit --dataset-dir data --task regression --out kernel.dk.json
dope-kernel eval --real-dir data --kernel kernel.dk.json --out report.json
dope-kernel gp-search --dataset-dir data --budget-secs 10 --out best.dk.json
```

`convert` writes `kernel.dpk`, `conversion-report.json`, `synthetic-model.bundle`, and `tstr-certification.json`. Optional `--synthetic-seed` arguments also write reproducible synthetic CSVs. The generator opens only `train.csv`; `test.csv` is required and opened only by the certification evaluator.

`embed-dataset` accepts a numeric CSV with a mandatory header and exactly one supervised target selected by name or zero-based index. Regression CSVs may be target-only; their native sketch records zero features. It min-max normalizes features and regression targets from that training table alone, or maps two numeric binary labels deterministically to 0/1. The command fails closed unless the supplied router is trained, structurally valid, linked to finite valid evidence, has the recorded canonical bundle size, and derives the fixed 4,168 dimensions. Router promotion quality is reported separately in qualification version 2 and does not block embedding; it continues to block campaign/release promotion. Its version-1 vectors contain the standardized 856-value dataset sketch followed, in router candidate order, by each penultimate hidden state and ten standardized action predictions. Public JSON exports use positional feature labels and omit source headers and raw extrema. `--restricted-metadata` explicitly emits that research metadata; restricted exports are not eligible release sidecars. These embeddings have dimension `856 + candidates × (hidden_width + 10)` (4,168 for 24 candidates with width 128) and are action-set-specific rather than universal similarity embeddings. Compare them only when `schema_sha256`, `router_bundle_sha256`, and `router_evidence_sha256` all match.

Certification evaluates `n`, `2n`, `4n`, and `8n` with three generated tables and three auditor seeds. Gates are enforced at `n` and `4n` using unclamped null-normalized retention, explicit low-signal non-inferiority, fidelity, inference, and privacy checks. All six frozen auditor slots are present in the report. Missing pinned runtimes fail closed, as do unavailable empirical or formal-DP tournament entries; they are never represented as completed runs.

The deep campaign's 108,523 discovery rows, 409,686 confirmation rows, and 11,620,492 validation-select rows are accounting totals across independent per-dataset fits. They are not pooled into a global network. Evidence format 2 always writes one durable block per dataset matrix, including discovery matrices below 100 cells, and cannot be reconciled with superseded receipts.

`calibrate-language` consumes only the checksummed packed manifest and writes a restartable five-seed run directory. A language artifact is marked `release_eligible` only when every fixed generalization and auditor gate passes; missing pinned GPU auditor runtimes remain explicit failed gates.

Generated campaign output, datasets, trained router bundles, and local caches are not included in this repository. Campaign and cluster workflows require separately prepared inputs as described in [production/README.md](../../production/README.md).

</details>

## Moved: architecture, testing and conformance notes (README, 2026-10-15)

### Architecture, testing, and conformance

```bash
just setup
just fast     # format, cargo check, unit tests, Python tests
just check    # fast + budgets, contracts, clippy, CLI tests
```

The [architecture](../../docs/architecture.md), [testing](../../docs/testing.md), and [release](../../docs/release.md) guides cover product boundaries, proof commands, and promotion. Jankurai is this repository's conformance auditor; a passing audit means the build, tests, and boundary proofs cleared the release floor with no high finding. It is not a utility score, not an MFS value, and not a privacy certificate. The gate and what the auditor checks are in [docs/audit.md](../../docs/audit.md).

## Moved: full native-artifact paragraph (README, 2026-10-15)

The README now carries this paragraph as part of the generated `PINNED DEMOS` block (`paper/generated/readme-demo-notes.md`); this copy is the text as it stood on 2026-10-15.

The installed public demo in [examples/public-regression](../../examples/public-regression/) fits the UCI Airfoil Self-Noise table (Brooks, Pope, and Marcolini; CC BY 4.0; DOI 10.24432/C5VW2C) on a grouped development split and writes one native raw-unit artifact, `model.raw` (SHA-256 `0950cbe79817…`), with profile `raw-numeric-symbolic-v1` under the byte profile `native-raw-complete-v1`. Compile, `inspect`, the demo's research receipt, and the file size all report 1,558 bytes: 56 bytes of framing, 1,080 bytes of fitted schema and transforms, and 422 bytes of sampling payload. The file is its own transport archive. The generic runtime is measured separately: the debug binary and its mapped libraries came to 72,049,152 bytes, so the cold sampling closure was 72,050,710 bytes. On a release build (`target/release/dope-kernel`, 10,386,496 bytes, SHA-256 `91f6ff9da65d…` at commit `ee4a3d0`), the artifact is byte-identical (`0950cbe79817…`), the release binary and its mapped libraries come to 13,888,176 bytes (cold sampling closure 13,889,734 bytes), and the native steps run in 15.7 ms (`compile-raw`), 2.1 ms (`inspect-raw`), 8.5 ms / 27.0 ms (`sample-raw` / isolated replay at 1n, 1,190 rows), and 24.5 ms / 44.2 ms (`sample-raw` / isolated replay at 4n, 4,760 rows), with total end-to-end demo wall time 0.83 s; no portable bundle was measured. The run used commit `59437b1`, and privacy, driver, and geometry checks are unmeasured for this artifact. It is an engineering conformance example on one development split, not a benchmark result, and a different object from both the toy above and the paper's `.dpk` plus `projection.json` charge. The pinned values are in [docs/whitepaper/receipts/native-byte-conformance-v1.json](../../docs/whitepaper/receipts/native-byte-conformance-v1.json), and the supplement repeats them.
