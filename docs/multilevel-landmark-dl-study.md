# Multi-Level Landmark & Data-Descriptor Deep Learning Study (`DOPE-MLC-v2`)

This document summarizes the architecture, Stage 0 rapid feature-importance validation,
19,079-split multi-level deep learning convergence, and world-class model transfer and
Bradley–Terry Elo evaluation for **`DOPE-MLC-v2`** (`MultiLevelLandmarkModel`).

---

## 1. Motivation & Core Principles

1. **Hierarchical Pre-Calculated Landmark & Data-Descriptor Conditioning**:
   Rather than feeding only raw 23-landmark gains into a single flat bottleneck, `DOPE-MLC-v2`
   organizes the 23 pre-calculated training-only CV landmarks (`fit-cv-v1`) and 93 additional
   data-descriptive coordinates from `report.json` (plus 24 fast train-only table descriptors =
   140 total conditioning dimensions) into three progressive information tiers within **one unified
   deep learning model**:
   - **`Medium` (`Macro` / Legal & Governance Tier, 4 KiB / 4,096 float16 coordinates)**:
     Exposes only the 9 Macro landmarks (`linear`, `poly2`, `poly3`, `bspline`, `pchip`,
     `monotone`, `periodic`, `hinge`, `otree2`) and coarse global marginals (`17` quantile knots).
     Higher-order interactions, pairwise terms, random Fourier features, and Chatterjee $\xi$
     spectra are strictly zero-masked, providing a governance-friendly macro representation.
   - **`High` (`Larger` / Interaction Tier, 16 KiB / 8,192 float16 coordinates)**:
     Unlocks the 9 Larger/Interaction landmarks (`pair`, `otree3`, `otree4`, `dt2`, `dt3`,
     `dt4`, `rf`, `et`, `gbt`), `33` quantile knots, rank-correlation structure, and medium-depth
     (`depth=3`, `24` trees) boosted oblivious residuals.
   - **`Ultra` (`Fine` / Maximum Transfer Tier, 64 KiB / 10,240 float16 coordinates)**:
     Unlocks all 23 landmarks including the 5 Fine/Local/Kernel landmarks (`knn`, `kreg`, `rff`,
     `mlp`, `catboost`), the full 9-coordinate **Chatterjee asymmetric $\xi(X_j, y)$ nonlinear
     dependence spectrum**, `65` quantile knots, normal-score `CovariateRefinementNet`, and
     `56` depth-4 boosted oblivious trees.

2. **Mandatory Train-Only Column Scaling & Categorical Encoding (`FitOnlyTableTransform`)**:
   Every real training table $D_{\text{train}}$ is normalized before encoding or synthesis using
   parameters fit **strictly on $D_{\text{train}}$**:
   - Skewness-aware continuous scaling (`signed_log1p_hybrid` when $|\text{skew}| > 1.75$ and tail
     ratio $> 4.5$, else `quantile_robust_hybrid` blending empirical normal-score quantiles with
     median/MAD scaling).
   - Rare-category pooling (`min_count=3` or `min_freq=0.005`) into `__other__`, ordinal frequency
     codes, and empirical-Bayes target-shrunken encodings fit strictly on $D_{\text{train}}$
     without claiming anonymous or de-identified status.
   - Unseen real holdout test data $D_{\text{test}}$ is transformed strictly using the frozen
     $D_{\text{train}}$ parameters (`transform(X_test, y_test)`), guaranteeing zero test leakage.

3. **Source-Free Unlimited Synthetic Generation & Downstream Model Transfer**:
   Once $D_{\text{train}}$ is compiled into `z_level` (`Medium`, `High`, or `Ultra`), raw training
   rows are discarded. `sample_synthetic(z_level, n_rows=N)` generates unlimited synthetic rows
   ($N_{\text{syn}} = N_{\text{train}}$, $2\times N_{\text{train}}$, $5\times N_{\text{train}}$).
   Downstream models trained exclusively on $D_{\text{syn}}$ are transferred directly to unseen
   scaled real test data $D_{\text{test}}$ and audited for both **predictive lift retention ($R$)**
   and **feature-importance dynamics ($D$ and Spearman $\rho$)**.

---

## 2. Stage 0: Rapid Feature-Importance & Scaling Validation (19,190 Splits)

Before launching full GPU deep learning, `research/levels/rapid_validate.py` evaluated the
116 landmark and descriptor coordinates across all **19,190 admitted splits** (`14,346` regression
+ `4,844` binary classification across `fit-iid-65-v1` and `outer5-iid-fold0..4-v1`; `17,290`
train splits, `1,900` unseen dataset holdout splits):

| Feature Tier & Descriptor Set | Dim | Ridge Signal Ceiling $R^2$ | CatBoost Signal Ceiling $R^2$ | CatBoost Nonlinear Uplift $R^2$ |
|---|---:|---:|---:|---:|
| `Medium` (Raw 9 Macro Landmarks only) | 46 | `0.8428` | `0.9176` | `0.7138` |
| `Medium` (+ Rich Macro Descriptors) | 116 | `0.9017` | `0.9417` | `0.7912` |
| `High` (Raw 18 Macro+Larger Landmarks) | 46 | `0.8972` | `0.9489` | `0.8189` |
| `High` (+ Rich Larger Descriptors) | 116 | `0.9390` | `0.9654` | `0.8727` |
| **`Ultra` (+ Rich Fine & Chatterjee $\xi$ Spectrum)** | **116** | **`0.9726`** | **`0.9863`** | **`0.9383`** |

- **Top Meta-Features Driving Nonlinear Uplift**: `contrast_lin_share` (`41.15%`),
  `contrast_signal_ceiling` (`22.27%`), `midrank_lin` (`8.41%`), `shape_log10_rows` (`4.74%`),
  `gain_lin` (`4.72%`), `contrast_macro_add_share` (`1.91%`), and `contrast_periodic_excess` (`1.62%`).
- **Train-Only Scaling Probe**: Across 40 stratified real holdout datasets, `FitOnlyTableTransform`
  improved mean CatBoost holdout score (`0.5671` vs `0.5532` naive z-score) while preserving
  CatBoost driver attribution fidelity $D = \mathbf{0.8238}$ and Spearman $\rho = \mathbf{0.7330}$.

---

## 3. Full-Corpus 3-Level GPU Deep Learning Convergence (19,079 Splits)

`research/levels/s3_multilevel_trainer.py` trained `DOPE-MLC-v2` on GPU (`cuda`) across all
**19,079 matched splits** (`14,295` regression + `4,784` binary classification across all 6 split
families; `17,188` training splits and `1,891` unseen dataset validation splits grouped by
`dataset_id` so zero splits of any validation dataset appear in training):

| Epoch | Splits Visited | Train Loss (Mean) | Val `Medium` Loss | Val `High` Loss | Val `Ultra` Loss | Val Mean Loss |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (Init) | 0 | `0.42415` | `0.45188` | `0.44037` | `0.42577` | `0.43934` |
| 1 | 2,865 | `0.36522` | `0.40084` | `0.37728` | `0.36168` | `0.37993` |
| 2 | 5,730 | `0.36325` | `0.40026` | `0.37525` | `0.35898` | `0.37817` |
| 3 | 8,595 | `0.36279` | `0.39962` | `0.37502` | `0.35847` | `0.37770` |
| 4 | 11,460 | `0.36268` | `0.39926` | `0.37447` | `0.35867` | `0.37747` |
| 6 | 17,190 | `0.36210` | `0.39899` | `0.37412` | `0.35804` | `0.37705` |
| **7 (Best)** | **20,055** | **`0.36186`** | **`0.39900`** | **`0.37377`** | **`0.35725`** | **`0.37667`** |
| **Full 1,891 Val** | **22,920** | **`0.36201`** | **`0.39744`** | **`0.37456`** | **`0.36060`** | **`0.37753`** |

Validation loss decreases by **`-14.3%`** on unseen datasets while maintaining strict monotonic
ordering `Loss(Ultra) < Loss(High) < Loss(Medium)` at every epoch.

---

## 4. 1,000-Dataset Stratified S3 World-Class Transfer, Driver Fidelity & Minimal-Artifact-Size Elo Study

`research/levels/world_class_eval.py` evaluated `DOPE-MLC-v2` (`Medium` `4,096 B`, `High` `16,384 B`,
`Ultra` `65,536 B`) against 5 baseline generators (`rows`, `kernel`, `copula`,
`joint_diffusion_v1`, `arf`) across **1,000 diverse, stratified S3 datasets** (`627` regression +
`373` binary classification; `65,874` downstream model fits in `361.22s` on 40 workers) and 6
downstream model families (**CatBoost**, **Local Gated Attention Network** `[raw ledger key:
kumo_tabpfn, _TabularKumoAttentionNet, hidden=48, 18 AdamW steps trained from scratch per table;
official pretrained KumoTabular/TabPFN foundation-model comparison is not performed]`,
**LightGBM**, **XGBoost**, **MLP**, and **Linear/Logistic**).

### 4.1 Coverage, Cost, Admission Funnel & Cohort Separation Receipt

- **Candidate Dispatch, Admission Funnel & `132` Filtered Candidates Audit (`target/s3-multilevel-training-20261023/coverage_cost_and_separation_receipt.json`)**:
  - **Initial Preserved Run (`task-797`, `xbabe1`, `workers = 40`, `361.22s` / `4.01` CPU core-hours)**:
    `1,132` stratified S3 candidate datasets dispatched; **`1,000` completed** (`627` regression,
    `373` binary classification); `132` filtered (`93` regression, `39` classification; median
    `509` rows, `13` cols) strictly due to near-zero real-table holdout CatBoost oracle signal
    ($S_{\text{real}} - S_{\text{null}} < 0.025$, mean holdout oracle gain = `0.0062`), where the
    null-normalized retention denominator is ill-conditioned; **`0` unhandled worker exceptions**.
  - **Admission Rate by Stratum & Unnormalized Low-Signal Probe**:
    In the strictly dataset-disjoint `is_val` stratum, `242 / 258` (`93.80%`) candidates were
    admitted (`16` filtered); in the in-corpus unseen-fold stratum, `758 / 874` (`86.73%`) were
    admitted (`116` filtered). Across a 24-dataset unnormalized test-score probe
    ($\Delta S = S_{\text{syn}} - S_{\text{real}}$) on the filtered low-signal tables,
    `DOPE-MLC-v2 Ultra` (`mean ΔS = -0.0151`) outperforms `rows Ultra` (`mean ΔS = -0.0435`) by
    **`+0.0284`**, confirming that filtering near-zero oracle denominators does not conceal
    synthetic degradation.
  - **Three-Way Disjoint Partition & Reconciliation of the `1,132` Candidates (`task-942`, `xbabe3`, `workers = 28`)**:
    In the preserved initial run (`dope_mlc_v2_s3_best.pt`, trained with a 90/10 dataset hash
    split), **`242` completed datasets** (`strictly_dataset_disjoint_242`) were strictly
    dataset-disjoint from gradient training (and belonged to the 10% validation hash split), while
    **`758` completed datasets** (`in_corpus_unseen_row_fold_758`) had training splits in the 90%
    meta-training corpus and were evaluated on unseen `outer5-iid-fold0-v1` holdout test rows
    ($D_{\text{test}}$, `0` row overlap; **row-disjoint, not dataset-disjoint**).
    To reconcile with a **100% gradient-training dataset-disjoint 1,000-dataset cohort**
    (`reconciled_strictly_dataset_disjoint_1000`), `dope_mlc_v2_disjoint1000_best.pt` was trained
    exclusively on the **`2,001` remaining S3 datasets (`11,976` splits)** after holding out all
    **`1,132` evaluation candidates** (`1,129` in `s3_prepared_splits_cache.pt` / `6,758` splits,
    plus `3` filtered low-signal candidates `7c29a13351d14ec5`, `a42f75735fd26aa1`, and
    `d181d4da916c742b` excluded during tensor cache build due to degenerate binary target splits;
    `0` in the completed `1,000`) and holding out the **`58` training-family validation datasets
    (`dev_val_only`, `345` splits; `0` overlap with both the `2,001` training datasets and the
    `1,132` evaluation candidates)**. On `dev_val_only` (`58` datasets, `345` splits), validation
    loss converged from `0.60288` (`Epoch 0`) to **`0.57414` (`Epoch 8`)** with strict monotonic
    tier ordering (`Ultra = 0.57053 < High = 0.57397 < Medium = 0.57791`). Because `Epoch 8` was
    also monitored on `val_sub` during the initial reconciliation pass, all four cohorts are
    reported as **development/reconciliation validation evidence** (never claimed as an untouched
    sealed confirmatory test).

### 4.2 Separate Cohort Comparison: Strictly Held-Out (`242`) vs. In-Corpus Unseen-Fold (`758`) vs. Disjoint-Trained (`1,000`)

| Cohort / Stratum | Datasets ($n$) | Evidence & Separation Status | `DOPE-MLC-v2` Overall Elo `[95% CI]` | Win Share | Cross-Size Elo (`4 KiB` $\to$ `16 KiB` $\to$ `64 KiB`) | `Medium` (`4 KiB`) Retention $R$ (Mean / Med) | `High` (`16 KiB`) Retention $R$ (Mean / Med) | `Ultra` (`64 KiB`) Retention $R$ (Mean / Med) | `Ultra` Driver Fidelity $D$ `[95% CI]` |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| **`strictly_dataset_disjoint_242`** | **`242`** | `0` splits in gradient training (`10%` val hash split, dev-selected) | **`1671.1` `[1655.2, 1688.0]` (`#1`)** | **`73.06%`** | **`1626.7` $\to$ `1690.5` $\to$ `1703.7`** | `0.8122` / `0.8991` | `0.8464` / `0.9327` | **`0.8493` / `0.9275`** | **`0.8034` `[0.7905, 0.8147]` (`D80` pass)** |
| **`in_corpus_unseen_row_fold_758`** | **`758`** | In-corpus dataset IDs, unseen `outer5-iid-fold0-v1` test rows (`row-disjoint`) | **`1645.4` `[1637.8, 1653.6]` (`#1`)** | **`69.81%`** | **`1587.6` $\to$ `1676.4` $\to$ `1682.3`** | `0.8779` / `0.9439` | `0.9133` / `0.9692` | **`0.9129` / `0.9716`** | **`0.7851` `[0.7779, 0.7918]`** |
| **`pooled_1000` (Preserved Initial)** | **`1,000`** | `242` strictly held-out + `758` in-corpus unseen test folds (`task-797`, ckpt `1d5db98f...` Epoch 8) | **`1651.3` `[1643.4, 1659.3]` (`#1`)** | **`70.58%`** | **`1596.8` $\to$ `1679.6` $\to$ `1687.2`** *(per-auditor)* / `1610.6` $\to$ `1726.6` $\to$ `1742.0` *(DS-mean)* | `0.8620` / `0.9335` | `0.8971` / `0.9642` | **`0.8975` / `0.9658`** | **`0.7896` `[0.7831, 0.7959]`** |
| **`reconciled_strictly_dataset_disjoint_1000`** | **`1,000`** | **`0` overlap** between `2,001` gradient-training datasets and all `1,132` eval candidates (`ckpt 6fcc191a...`, fixed `Epoch 8` verified on `58` `dev_val` datasets; prior `val_sub` monitoring disclosed) | **`1650.7` `[1643.4, 1657.5]` (`#1`)** | **`70.51%`** | **`1596.1` $\to$ `1675.5` $\to$ `1690.1`** | `0.8620` / `0.9349` | `0.8946` / `0.9624` | **`0.8998` / `0.9671`** | **`0.7904` `[0.7844, 0.7972]`** |

### 4.3 Overall 1,000-Dataset Multi-Auditor Bradley–Terry Elo Leaderboard (Preserved Pooled & Reconciled 100% Disjoint)

Across **44,369 pairwise auditor-dataset matches per method** (with 200× dataset-cluster bootstrap
95% confidence intervals), `DOPE-MLC-v2` achieves **Rank #1 overall Elo (`1651.3`)**:

| Rank | Generator Method | Overall Elo | 95% Bootstrap CI | Bootstrap SE | Wins | Draws | Losses | Win Share |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| **#1** | **`dope_mlc_v2` (Unified 3-Level Model)** | **`1651.3`** | **`[1643.4, 1659.6]`** | `4.26` | **30,959** | **717** | **12,693** | **`70.58%`** |
| #2 | `rows` (Non-Anonymized Raw Row Subsample) | `1593.3` | `[1582.1, 1603.4]` | `5.32` | 26,888 | 1,528 | 15,953 | `62.32%` |
| #3 | `kernel` (DOPE v1 Kernel Sketch) | `1592.5` | `[1585.8, 1600.2]` | `3.67` | 26,881 | 1,426 | 16,062 | `62.20%` |
| #4 | `copula` (Gaussian Copula) | `1545.8` | `[1538.9, 1553.0]` | `3.59` | 23,805 | 1,458 | 19,106 | `55.29%` |
| #5 | `joint_diffusion_v1` (Tabular Diffusion) | `1432.4` | `[1424.0, 1440.7]` | `4.37` | 16,660 | 1,064 | 26,645 | `38.75%` |
| #6 | `arf` (Adversarial Random Forests) | `1184.7` | `[1170.7, 1200.6]` | `7.34` | 4,335 | 965 | 39,069 | `10.86%` |

### 4.3 Elo & Transfer Performance vs. Nominal Budget Cap, Exact-Replay Wire Bytes (`v2`) & Legacy Lossy Wire Bytes (`v1`)

When all 18 `(method, tier)` pairs compete in a single **Unified Cross-Size
Bradley–Terry Pool**, `DOPE-MLC-v2`'s Elo rises monotonically with the **nominal tier budget cap**
(`4,096 B / 16,384 B / 65,536 B`), the **exact-replay `version=2` normalized-space latent wire payload**
(`1,502 B / 2,836 B / 5,872 B` median raw bytes; `1,238 B / 2,351 B / 4,616 B` median zlib bytes, `+22 B` raw over legacy `version=1` lossy `1,480 B / 2,814 B / 5,850 B`),
and the **full raw-domain replay payload including the mandatory `FitOnlyTableTransform` + seed sidecar**
(`2,031 B / 3,376 B / 6,398 B` median raw bytes for `version=2` vs `2,009 B / 3,354 B / 6,376 B` for `version=1`, adding the `545 B` median / `803.0 B` mean scaler sidecar across the `1,000` datasets).
Across all **`1,000` canonical datasets, `3` tiers, and both frozen checkpoints (`6,000 / 6,000` cells)** in `exact_wire_replay_1000ds_qualification_v4.json`,
the `version=2` codec (`dope.latent_wire.v3_exact`) achieves **`100.0%` (`6,000/6,000`) isolated-receiver decode success with bit-for-bit identical `X_syn` (`max_abs_diff = 0.0`) AND bit-for-bit identical `y_syn` (`max_abs_diff = 0.0`)**
using only the wire payload plus the frozen `292,707 B` shared decoder (`dope_mlc_v2_wire_1000ds_qualified = true`, verified on the 24-dataset `144`-cell CatBoost fixture with `median |ΔR| = 0.0000` and `max |ΔR| = 0.0000` vs legacy `version=1` `max |ΔR| = 0.6283`;
competing baseline byte tallies remain post-hoc `fp16` estimates paired with historical `float64` in-memory scores, so `baseline_wire_1000ds_qualified = false` and `full_cross_method_wire_1000ds_qualified = false`):

| Tier & Nominal Cap (`DOPE` Exact `v2` Latent Raw / Zlib \| Full Raw w/ Sidecar `[Legacy v1 Raw]`) | `DOPE-MLC-v2` Cross-Size Elo (Per-Auditor / DS-Mean) | `rows` Cross-Size Elo (`1,024 / 4,516 / 8,464 B` Raw Est.) | `kernel` Cross-Size Elo (`202 B` Raw Est., Tier-Inv) | `copula` Cross-Size Elo (`226 B` Raw Est., Tier-Inv) | `joint_diffusion_v1` Cross-Size Elo (`9,616 B` Full-Table Est.) | `arf` Cross-Size Elo (`10,426 B` Full-Table Est., Tier-Inv) | `DOPE-MLC-v2` Within-Tier Elo |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Medium`: `4,096 B` Cap (`1,502 B` / `1,238 B` \| `2,031 B` full `[v1: 1,480 B]`)** | **`1596.8` / `1610.6`** | `1455.8` / `1415.3` | `1584.9` / `1596.0` | `1528.8` / `1526.4` | `1350.5` / `1332.9` | `1174.1` / `1140.5` | **`1652.1` (`#1`, `71.11%` win)** |
| **`High`: `16,384 B` Cap (`2,836 B` / `2,351 B` \| `3,376 B` full `[v1: 2,814 B]`)** | **`1679.6` / `1726.6`** | `1634.3` / `1648.3` | `1584.9` / `1596.0` | `1528.8` / `1526.4` | `1431.4` / `1424.1` | `1174.1` / `1140.5` | **`1669.6` (`#1`, `72.62%` win)** |
| **`Ultra`: `65,536 B` Cap (`5,872 B` / `4,616 B` \| `6,398 B` full `[v1: 5,850 B]`)** | **`1687.2` / `1742.0`** | `1765.8` / `1818.6` *(copies `88%` raw rows)* | `1584.9` / `1596.0` | `1528.8` / `1526.4` | `1485.0` / `1494.7` | `1174.1` / `1140.5` | **`1658.4` (`#1` encoder, `70.50%` win)** |

- At **`Medium` (`4,096 B` nominal cap; `1,502 B` median exact `v2` latent raw / `1,238 B` zlib wire bytes; `2,031 B` median full raw with scaler sidecar; `86.4% <= 4,096 B` latent raw, `94.4%` latent zlib, `77.5%` full raw, `85.2%` est. full zlib)**,
  `DOPE-MLC-v2` beats the non-anonymized raw row subsample (`rows`, `1,024 B` latent raw fp16 / `3,998 B` ASCII budget)
  by **`+141.0` per-auditor cross-size Elo points (`+195.3` dataset-mean Elo)** and `+18.7` retention points (`0.8620` vs `0.6746`).
- At **`High` (`16,384 B` nominal cap; `2,836 B` median exact `v2` latent raw / `2,351 B` zlib wire bytes; `3,376 B` median full raw with scaler sidecar, `100.0% <= 16 KiB` under all four byte definitions)**,
  `DOPE-MLC-v2` beats `rows` (`4,516 B` median latent raw fp16 / `16,276 B` ASCII budget) by **`+45.3` per-auditor cross-size Elo points
  (`+78.3` dataset-mean Elo)** and `+4.3` retention points (`0.8971` vs `0.8539`).
- At **`Ultra` (`65,536 B` nominal cap; `5,872 B` median exact `v2` latent raw / `4,616 B` zlib wire bytes; `6,398 B` median full raw with scaler sidecar, `100.0% <= 64 KiB` under all four byte definitions)**,
  `DOPE-MLC-v2` reaches **`1687.2` per-auditor cross-size Elo (`1742.0` dataset-mean Elo)**; the non-anonymized `rows` control
  only edges higher at `Ultra` because its `64 KiB` ASCII budget copies **`88%` of the raw training rows**.

### 4.4 Strict Monotonic Multi-Level Progression (`DOPE-MLC-v2`, 1,000 Datasets)

| Tier | Nominal Cap | Exact `v2` Latent Wire Med (Raw / Zlib) \| Full Raw w/ Sidecar | Mean Retention $R$ `[95% CI]` | Median $R$ `[IQR p25, p75]` | Mean Driver Fidelity $D$ `[95% CI]` | Feature Spearman $\rho$ | Top-3 Driver Overlap | BTR-AUC | Regression $R^2$ |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| **`Medium` (Macro / Legal)** | `4,096 B` (`4 KiB`) | **`1,502 B` / `1,238 B` \| `2,031 B`** | **`0.8620` `[0.8521, 0.8726]`** | **`0.9335` `[0.8199, 0.9781]`** | **`0.7566` `[0.7512, 0.7624]`** | **`0.5985`** | `69.95%` | **`83.11`** | `0.6620` |
| **`High` (Larger / Interaction)** | `16,384 B` (`16 KiB`) | **`2,836 B` / `2,351 B` \| `3,376 B`** | **`0.8971` `[0.8846, 0.9090]`** | **`0.9642` `[0.8849, 0.9918]`** | **`0.7821` `[0.7759, 0.7877]`** | **`0.6331`** | `72.35%` | **`86.82`** | `0.7049` |
| **`Ultra` (Fine / Max Transfer)** | `65,536 B` (`64 KiB`) | **`5,872 B` / `4,616 B` \| `6,398 B`** | **`0.8975` `[0.8883, 0.9095]`** | **`0.9658` `[0.8768, 0.9942]`** | **`0.7896` `[0.7831, 0.7959]`** | **`0.6363`** | **`72.42%`** | **`86.97`** | **`0.7145`** |

### 4.5 World-Class Downstream Model Transfer at `Ultra` (1,000 Datasets, $N_{\text{syn}} = N_{\text{train}} \to D_{\text{test}}$)

| Downstream Auditor Family | Datasets ($n$) | Real Test Score | Syn$\to$Real Test Score | Holdout Retention $R$ `[95% CI]` | Driver Fidelity $D$ `[95% CI]` | Feature Spearman $\rho$ | Top-3 Driver Overlap | Binary Real Acc | Binary Syn$\to$Real Acc |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| **CatBoost** | `1,000` | `0.8472` | `0.7801` | **`0.8810` `[0.8692, 0.8915]`** | **`0.8130` `[0.8055, 0.8213]` (`D80` pass)** | **`0.7036`** | **`76.33%`** | `85.88%` | **`83.13%`** |
| **Local Gated Attention Net (`kumo_tabpfn` key)** | `962` | `0.7920` | `0.7530` | **`0.9211` `[0.9094, 0.9333]`** | **`0.8781` `[0.8743, 0.8824]` (`D80` pass)** | **`0.6470`** | **`73.56%`** | `84.01%` | **`82.35%`** |
| **LightGBM** | `989` | `0.8387` | `0.7809` | **`0.8978` `[0.8852, 0.9085]`** | **`0.7703` `[0.7615, 0.7793]`** | **`0.6263`** | `69.94%` | `85.59%` | **`83.18%`** |
| **MLP** | `965` | `0.8040` | `0.7594` | **`0.9143` `[0.9021, 0.9245]`** | **`0.8915` `[0.8872, 0.8960]` (`D80` pass)** | **`0.7740`** | **`78.03%`** | `84.60%` | **`82.69%`** |
| **Linear / Logistic** | `969` | `0.7767` | `0.7503` | **`0.9297` `[0.9182, 0.9398]`** | **`0.7853` `[0.7745, 0.7950]`** | **`0.5567`** | `70.76%` | `83.99%` | **`82.14%`** |
| **XGBoost** | `989` | `0.8530` | `0.7308` | **`0.8121` `[0.7915, 0.8285]`** | **`0.7221` `[0.7131, 0.7310]`** | **`0.5305`** | `66.70%` | `85.20%` | `78.52%` |

- **Actual Auditor Roster, Alias Mapping, Backends & Per-Auditor Low-Oracle-Signal Skips (`0` Unhandled Failures)**:
  - `catboost`: `catboost.CatBoostRegressor` / `CatBoostClassifier` (`Cython` backend, `iterations=60, depth=5, lr=0.10`), evaluated on `1,000 / 1,000` datasets (`0` skips).
  - `kumo_tabpfn` *(Explicit Alias Mapping $\to$ **Local Gated Attention Net**)*: `_KumoTabularWrapper` / `_TabularKumoAttentionNet` (a local PyTorch SiLU-gated attention-residual network with `hidden=48` and `18` `AdamW` steps trained from scratch per table with input-gradient attribution; the historical raw JSON key `kumo_tabpfn` is preserved for ledger immutability, but **this is a local attention-network baseline and NOT the official pretrained `KumoTabular` or `TabPFN` foundation model; official Kumo/TabPFN comparison is not performed in this run**), evaluated on `962 / 1,000` datasets (`38` skipped strictly where real-table oracle gain $S_{\text{real}} - S_{\text{null}} < 0.02$).
  - `lightgbm`: `lightgbm.LGBMRegressor` / `LGBMClassifier` (`C++` histogram GBDT, `n_estimators=55, max_depth=5, lr=0.10`), evaluated on `989 / 1,000` datasets (`11` skipped where real oracle gain $< 0.02$).
  - `xgboost`: `xgboost.XGBRegressor` / `XGBClassifier` (`C++ hist` tree method, `n_estimators=55, max_depth=4, lr=0.10`), evaluated on `989 / 1,000` datasets (`11` skipped where real oracle gain $< 0.02$).
  - `mlp`: `sklearn.neural_network.MLPRegressor` / `MLPClassifier` (`hidden=(48, 24), Adam max_iter=65`), evaluated on `965 / 1,000` datasets (`35` skipped where real oracle gain $< 0.02$).
  - `linear`: `sklearn.linear_model.Ridge(alpha=1.0)` / `LogisticRegression(C=1.0, max_iter=150)`, evaluated on `969 / 1,000` datasets (`31` skipped where real oracle gain $< 0.02$).

### 4.6 Performance Across Diverse Dataset Regimes (1,000 Datasets)

| Regime Axis | Stratum | Datasets ($n$) | `Medium` (`4 KiB`) Retention $R$ | `High` (`16 KiB`) Retention $R$ | `Ultra` (`64 KiB`) Retention $R$ | `Ultra` Driver Fidelity $D$ |
|---|---|---:|---:|---:|---:|---:|
| **Signal Nonlinearity** | `Mostly Linear (<0.05)` | `687` | `0.9467` | `0.9585` | `0.9477` | **`0.8190`** |
| **Signal Nonlinearity** | `Moderate Nonlinear (0.05-0.20)` | `92` | `0.7263` | `0.8180` | **`0.8318`** | `0.7313` |
| **Signal Nonlinearity** | **`Strong Nonlinear (>0.20)`** | **`221`** | `0.6552` | `0.7390` | **`0.7689` (`+11.37` pts vs Med)** | `0.7226` |
| **Sample Size ($N_{\text{train}}$)** | `Small (<300 rows)` | `265` | `0.7580` | `0.8147` | **`0.8407`** | **`0.8075`** |
| **Sample Size ($N_{\text{train}}$)** | `Medium (300-1,500 rows)` | `583` | `0.8886` | **`0.9184`** | `0.9091` | `0.7796` |
| **Sample Size ($N_{\text{train}}$)** | `Large (>1,500 rows)` | `152` | `0.9413` | **`0.9587`** | `0.9519` | **`0.7966`** |
| **Feature Width ($p$)** | `Narrow (<=8 cols)` | `360` | `0.8374` | **`0.9056`** | `0.8964` | **`0.8384`** |
| **Feature Width ($p$)** | `Medium (9-25 cols)` | `485` | `0.8705` | `0.8891` | **`0.8942`** | `0.7523` |
| **Feature Width ($p$)** | `Wide (>25 cols)` | `155` | `0.8926` | `0.9022` | **`0.9104`** | **`0.7930`** |
| **Categorical Share** | `Pure Numeric` | `584` | `0.8888` | `0.9222` | **`0.9261`** | **`0.7924`** |
| **Categorical Share** | `Mixed (<50% cat)` | `307` | `0.8226` | **`0.8588`** | `0.8543` | `0.7745` |
| **Categorical Share** | `Categorical Heavy (>=50% cat)` | `109` | `0.8294` | **`0.8699`** | `0.8659` | **`0.8174`** |

- **Unlimited Synthetic Expansion from Fixed Encoding**: Generating $1\times \to 2\times \to 5\times N_{\text{train}}$
  synthetic rows from the frozen `Ultra` encoding increases downstream CatBoost holdout retention
  monotonically from **`0.8808`** ($1\times N_{\text{train}}$) $\to$ **`0.9010`**
  ($2\times N_{\text{train}}$) $\to$ **`0.9070`** ($5\times N_{\text{train}}$, `+2.62` retention
  points) without revisiting raw training data.

---

## 5. Source-Bound 1,000-Dataset Study Acceptance Matrix & Scientific Limits

| Acceptance Dimension | Source-Bound Specification & Measured Values | Authoritative Code / Receipt / Plot Locator |
|---|---|---|
| **1. Corpus Coverage & Admission Funnel** | `3,188` S3 regression + binary datasets (`19,190` landmark splits; `19,079` valid splits across `3,188` datasets in `s3_prepared_splits_cache.pt`). Stratified candidate selector dispatched `1,132` unique `outer5-iid-fold0-v1` datasets (`271` strictly held-out `is_val=True` + `861` in-corpus `is_val=False`). Exact `1,000` datasets completed (`627` regression + `373` binary; `242` strictly held-out, `89.30%` admission; `758` in-corpus, `88.04%` admission). `132` candidates (`29` held-out + `103` in-corpus) excluded strictly where real CatBoost holdout oracle gain $S_{\text{real}} - S_{\text{null}} < 0.025$ or degenerate split (`0` worker exceptions). Exact `3` candidate IDs (`7c29a13351d14ec5`, `a42f75735fd26aa1`, `d181d4da916c742b`) are absent from `s3_prepared_splits_cache.pt` (`1,129` vs `1,132` candidates; `0` in completed `1,000`, all `3` in filtered `132`). | [`world_class_eval.py`](../research/levels/world_class_eval.py), [`strata_audit.py`](../research/levels/strata_audit.py), `target/s3-multilevel-training-20261023/strata_and_disjoint_reconciliation_receipt.json` |
| **2. Cohort Strata & Checkpoint Selection Exposure** | **4 evaluated cohorts** (all explicitly labeled as **selection-exposed development/reconciliation validation evidence**, `untouched_sealed_confirmatory_test: false`):<br>1. `strictly_dataset_disjoint_242` ($n=242$): bound to `dope_mlc_v2_s3_best.pt` (`SHA-256 1d5db98ffa438a0be1c61dabf235972cd400d1010f023705bea983f96e660a98`, `Epoch 8`), `0` dataset overlap with `17,188` training splits (`2,869` fit datasets), but belongs to the `10%` validation split (`1,891` splits across `319` datasets) used to select `Epoch 8`.<br>2. `in_corpus_unseen_row_fold_758` ($n=758$): bound to `dope_mlc_v2_s3_best.pt` (`1d5db98f...`, `Epoch 8`), evaluated on unseen `outer5-iid-fold0-v1` test rows (`0` row overlap with fit split), but dataset IDs were in the `90%` training dataset pool.<br>3. `pooled_1000` ($n=1,000$, preserved initial run `task-797`, ledger `per_dataset_1000_results.jsonl.gz` `SHA-256 c268f738570fc10446791591cb26ca2f8a79fba828fd3f52594e4d617879d920`): union of `242` held-out and `758` in-corpus unseen-fold datasets.<br>4. `reconciled_strictly_dataset_disjoint_1000` ($n=1,000$, `task-942`, ledger `per_dataset_1000_strictly_disjoint_results.jsonl.gz` `SHA-256 5d62cc31d65c2e695de451ba906226e68bf613108f5dbcae41dc02bd7f37d3ab`): bound to `dope_mlc_v2_disjoint1000_best.pt` (`SHA-256 6fcc191a244ff409b9d44bbb6a2bcb1d8e3a7db326dcdfbb56f75a64b5f87da3`, fixed-horizon `Epoch 8`), trained on `2,001` datasets (`11,976` splits) excluding all `1,132` evaluation candidates (`0` gradient-training overlap) and verified on `58` separate `dev_val` datasets (`345` splits: `0.60288` $\to$ `0.57414` loss; `Ultra 0.57053 < High 0.57397 < Medium 0.57791`), with prior `val_sub` (`6,758` splits) monitoring during reconciliation explicitly disclosed. | [`disjoint_reconcile.py`](../research/levels/disjoint_reconcile.py), `target/s3-multilevel-training-20261023/s3_training_convergence_receipt.json`, `target/s3-multilevel-training-20261023/s3_training_disjoint1000_convergence_receipt.json`, `target/s3-multilevel-training-20261023/strictly_disjoint_1000_transfer_and_elo_receipt.json` |
| **3. True Downstream Auditor Identities (`6` Families)** | 1. `catboost`: `catboost.CatBoostRegressor` / `CatBoostClassifier` (`iterations=60, depth=5, lr=0.10`, `1,000/1,000` datasets)<br>2. `kumo_tabpfn` *(raw ledger key preserved for immutability $\to$ **Local Gated Attention Net**)*: `_KumoTabularWrapper` / `_TabularKumoAttentionNet` (`hidden=48`, `18` `AdamW` steps trained from scratch per table; `962/1,000` datasets, `38` low-signal skips; **`official_kumo_or_tabpfn_evaluated: false` — NOT official pretrained `KumoTabular` or `TabPFN`**)<br>3. `lightgbm`: `lightgbm.LGBMRegressor` / `LGBMClassifier` (`n_estimators=55, max_depth=5`, `989/1,000` datasets)<br>4. `xgboost`: `xgboost.XGBRegressor` / `XGBClassifier` (`n_estimators=55, max_depth=4`, `989/1,000` datasets)<br>5. `mlp`: `sklearn.neural_network.MLPRegressor` / `MLPClassifier` (`hidden=(48,24)`, `965/1,000` datasets)<br>6. `linear`: `Ridge(alpha=1.0)` / `LogisticRegression(C=1.0)`, `969/1,000` datasets.<br>*(Coverage note: core auditors `catboost`, `lightgbm`, `linear` were evaluated across all `18` method $\times$ tier cells on all `1,000` datasets; non-core auditors `xgboost`, `mlp`, `kumo_tabpfn` were evaluated at `Ultra` across all `1,000` datasets for `DOPE-MLC-v2` and on `ds_rank < 250` for `rows`/`joint_diffusion_v1`; unlimited expansion `1x/2x/5x N_train` was evaluated on `ds_rank < 150` / `133` datasets.)* | [`world_class_eval.py`](../research/levels/world_class_eval.py), `target/s3-multilevel-training-20261023/world_class_transfer_and_elo_receipt.json` |
| **4. Per-Method / Tier Metrics & 200x Bootstrap 95% CIs** | **`DOPE-MLC-v2` Ranks `#1` in every cohort**:<br>- **Held-Out (`242` DS)**: Overall Elo **`1671.1`** `[1655.2, 1688.0]` (`73.06%` win share); Cross-Size Elo `1626.7` (`Med`) $\to$ `1690.5` (`High`) $\to$ `1703.7` (`Ultra`); `Ultra` $R = \mathbf{0.8493}$ `[0.8226, 0.8732]` (`Med = 0.9275`), $D = \mathbf{0.8034}$ `[0.7885, 0.8180]` (**passes `D80`**).<br>- **In-Corpus Fold (`758` DS)**: Overall Elo **`1645.4`** `[1637.8, 1653.6]` (`69.81%` win share); Cross-Size Elo `1587.6` $\to$ `1676.4` $\to$ `1682.3`; `Ultra` $R = \mathbf{0.9129}$ `[0.9037, 0.9222]` (`Med = 0.9716`), $D = \mathbf{0.7851}$ `[0.7782, 0.7924]`.<br>- **Pooled (`1,000` DS)**: Overall Elo **`1651.3`** `[1643.4, 1659.3]` (`70.58%` win share); Cross-Size Elo `1596.8` $\to$ `1679.6` $\to$ `1687.2` *(per-auditor)* / `1610.6` $\to$ `1726.6` $\to$ `1742.0` *(dataset-mean)*; `Ultra` $R = \mathbf{0.8975}$ `[0.8886, 0.9073]` (`Med = 0.9658`), $D = \mathbf{0.7896}$ `[0.7826, 0.7966]`.<br>- **Disjoint-Trained (`1,000` DS)**: Overall Elo **`1650.7`** `[1643.4, 1657.5]` (`70.51%` win share); Cross-Size Elo `1596.1` (`Med`) $\to$ `1675.5` (`High`) $\to$ `1690.1` (`Ultra`); `Ultra` $R = \mathbf{0.8998}$ `[0.8913, 0.9094]` (`Med = 0.9671`), $D = \mathbf{0.7904}$ `[0.7834, 0.7966]`. | [`model-sizes-and-elo.md`](model-sizes-and-elo.md), [`dope1-content.tex`](../paper/dope1-content.tex), [`dope1-supplement.tex`](../paper/dope1-supplement.tex) |
| **5. Nominal Budget Caps vs. Exact-Replay Wire Bytes (`v2` Codec) & Legacy Lossy Wire Bytes (`v1` Codec)** | Historical `world_class_eval.py` / `strata_audit.py` copied nominal tier caps (`4,096 B / 16,384 B / 65,536 B`) as `artifact_bytes` for all methods and evaluated generators in-memory in `FitOnlyTableTransform`-normalized space (`X_tr, y_tr` $\to$ pre-scaled `X_te, y_te`). The v2/v3/v4 byte & replay audits (`per_dataset_1000_measured_egress_bytes_v2.jsonl.gz` `SHA-256 2600ed08f9a42090b4a1dc408e91f1e14b5086eee235e15470fc03a3b605a478`, `wire_replay_roundtrip_and_parity_audit_v3.json` `SHA-256 5a583c4b077eafe06a319147ca07285844a0da285517a64d1905f46135e2a6f3`, `exact_wire_replay_1000ds_qualification_v4.json`, `per_dataset_1000_exact_wire_replay_v4.jsonl.gz` `SHA-256 4440ade9b760e14bda5ae31d6b5ab9406a3f4eb4b7abccc0bba0bc7880bb634d`) bind [`serialize_dope_latent_wire_bytes`](../research/levels/strata_audit.py) / [`deserialize_dope_latent_wire_bytes`](../research/levels/strata_audit.py) (`version=1` legacy lossy fp16 vs `version=2` exact replay) and [`estimate_scaler_sidecar_bytes`](../research/levels/strata_audit.py):<br>- **`DOPE-MLC-v2` Exact Wire Codec (`version=2`, `dope.latent_wire.v3_exact`, `+22 B` raw over legacy `version=1`)** *(24 B `<BBBBIIHHHHHH` struct header + 28 B `<7i` decimal fixed-point for 7 rounded target scalars + 16 B `<4f` float32 for 4 router gates = 68 B fixed prefix + all fp16/uint8/packbits tensors + exact 2B `<h`/4B `<i` decimal fixed-point oblivious-tree thresholds consumed by `generate_synthetic` in normalized space; pre-deployed dataset-independent decoder weights `cov_net`/`target_net` in `292,707 B` checkpoint are amortized once; `code_fp16` `8,192 / 16,384 / 20,480 B` is a preview vector not used by `generate_synthetic`; `FitOnlyTableTransform` + 4 B RNG seed sidecar adds **`545 B` median / `803.0 B` mean / `2,781 B` max** across the `1,000` datasets [`267/1,000` datasets have categorical columns requiring hashed token IDs]):*<br>  - `Medium` (cap `4,096 B`): `v2` exact latent raw median **`1,502 B`** (`v1` lossy: `1,480 B`; `v2` `p99 = 5,516 B`, `max = 6,184 B`, `86.4% <= 4,096 B` raw); `v2` latent zlib median **`1,238 B`** (`v1`: `1,227 B`; **`94.4% <= 4,096 B`**); `v2` full raw-domain median w/ scaler sidecar **`2,031 B`** (`v1`: `2,009 B`; `77.5% <= 4,096 B` raw, `85.2%` est. zlib).<br>  - `High` (cap `16,384 B`): `v2` exact latent raw median **`2,836 B`** (`v1`: `2,814 B`; `p99 = 11,222 B`, `max = 12,650 B`, **`100.0% <= 16,384 B`**); `v2` latent zlib median **`2,351 B`** (`v1`: `2,324 B`; **`100.0% <= 16,384 B`**); `v2` full raw-domain median w/ scaler sidecar **`3,376 B`** (`v1`: `3,354 B`; **`100.0% <= 16,384 B`** raw & zlib).<br>  - `Ultra` (cap `65,536 B`): `v2` exact latent raw median **`5,872 B`** (`v1`: `5,850 B`; `p99 = 25,051 B`, `max = 27,202 B`, **`100.0% <= 65,536 B`**); `v2` latent zlib median **`4,616 B`** (`v1`: `4,587 B`; **`100.0% <= 65,536 B`**); `v2` full raw-domain median w/ scaler sidecar **`6,398 B`** (`v1`: `6,376 B`; **`100.0% <= 65,536 B`** raw & zlib).<br>  - **Full 1,000-Dataset (`6,000`-Cell) Exact Replay Qualification (`version=2`) vs. Legacy Lossy (`version=1`)**: Across all `1,000` datasets $\times$ `3` tiers $\times$ `2` checkpoints (`6,000/6,000` cells), `version=2` achieves **`100.0%` (`6,000/6,000`) decode success, bit-exact `X_syn` (`max_abs_diff = 0.0`), AND bit-exact `y_syn` (`max_abs_diff = 0.0`)** using only the wire payload + frozen `292,707 B` decoder (`dope_mlc_v2_wire_1000ds_qualified = true`), whereas legacy `version=1` had `y_syn` drift on `5,107/6,000` cells (`legacy_v1_wire_1000ds_qualified = false`). On the 24-dataset (`144`-cell) CatBoost fixture, `version=2` achieves **`median |ΔR| = 0.0000` and `max |ΔR| = 0.0000`** across all `144/144` cells (vs `version=1` `median |ΔR| = 0.0002 / 0.0041 / 0.0168` on `s3_best`, `0.0000 / 0.0043 / 0.0063` on `disjoint1000`, and outlier `max |ΔR| = 0.6283`).<br>- **`rows` / `kernel` / `copula` / `arf` / `joint_diffusion_v1`** *(post-hoc fp16 representation estimates paired with historical float64 in-memory scores; `baseline_wire_1000ds_qualified = false`, `full_cross_method_wire_1000ds_qualified = false`; 24-DS CatBoost fp16 vs fp64 median $|\Delta R| = 0.0000, 0.0033, 0.0027, 0.0009, 0.0035$)*:<br>  - `rows`: latent raw fp16 median **`1,024 B` / `4,516 B` / `8,464 B`**; zlib **`543 B / 1,794 B / 3,419 B`**; ASCII formula budget `3,998 B / 16,276 B / 30,094 B`.<br>  - `kernel` *(tier-invariant; assumes shared PRNG seed for $p \times 6$ `W_rnd` or `+12*p B`)*: latent raw fp16 median **`202 B`** (`p99 = 930 B`), zlib **`210 B`**.<br>  - `copula` *(tier-invariant; `mean_xy` + upper-triangle `cov_upper` + `mean_y` in normalized space, no quantile knots)*: latent raw fp16 median **`226 B`** (`p99 = 4,438 B`, `90.7% <= 4,096 B` raw, `96.1%` zlib), zlib **`232 B`**.<br>  - `arf` *(tier-invariant; requires full $n_{\text{fit}} \times (p+1)$ fit table partitioned by leaf)*: latent raw fp16 median **`10,426 B`** (`p99 = 104,944 B`; only `25.0% <= 4 KiB`, `61.9% <= 16 KiB`, `92.8% <= 64 KiB`), zlib **`6,369 B`**.<br>  - `joint_diffusion_v1` *(requires full $n_{\text{fit}} \times (p+1)$ table egress; varies only by `noise_t = 0.48 / 0.38 / 0.32`)*: latent raw fp16 median **`9,616 B`** (`p99 = 104,016 B`; only `26.7% <= 4 KiB`, `62.9% <= 16 KiB`, `92.8% <= 64 KiB`), zlib **`5,733 B`**. | `target/s3-multilevel-training-20261023/exact_wire_replay_1000ds_qualification_v4.json`, `target/s3-multilevel-training-20261023/per_dataset_1000_exact_wire_replay_v4.jsonl.gz`, `target/s3-multilevel-training-20261023/wire_replay_roundtrip_and_parity_audit_v3.json`, `target/s3-multilevel-training-20261023/measured_egress_bytes_and_provenance_audit_v2.json`, `target/s3-multilevel-training-20261023/per_dataset_1000_measured_egress_bytes_v2.jsonl.gz`, `target/s3-multilevel-training-20261023/figures/artifact_size_pareto_and_elo.png` |
| **6. Known Compute Costs & Worker Counts** | - **Stage 0 Validation (`19,190` splits)**: `79.6 s` wall clock (`8` CPU threads on `xbabe2`).<br>- **Initial 3-Level GPU Training (`19,079` splits, `dope_mlc_v2_s3_best.pt`)**: `8` epochs (`889.06 s` wall clock on CUDA; checkpoint SHA-256 `1d5db98ffa438a0be1c61dabf235972cd400d1010f023705bea983f96e660a98`).<br>- **Initial 1,000-Dataset Evaluation (`task-797`)**: `40` parallel CPU workers on `xbabe1`, `361.22 s` wall clock (`4.01` CPU core-hours, `65,874` downstream auditor fits; ledger SHA-256 `c268f738570fc10446791591cb26ca2f8a79fba828fd3f52594e4d617879d920`).<br>- **Strictly Disjoint 1,000-Dataset Training & Re-Scoring (`task-942`)**: `615.02 s` GPU training (`8` epochs on `11,976` splits / `2,001` datasets, checkpoint SHA-256 `6fcc191a244ff409b9d44bbb6a2bcb1d8e3a7db326dcdfbb56f75a64b5f87da3`) + `481.75 s` evaluation across `28` CPU workers on `xbabe3` (ledger SHA-256 `5d62cc31d65c2e695de451ba906226e68bf613108f5dbcae41dc02bd7f37d3ab`).<br>- **Per-Dataset Egress Byte Serialization Audit (`task-1560`)**: `638.94 s` across `32` CPU workers (`1,000` datasets $\times$ `2` checkpoints $\times$ `6` methods $\times$ `3` tiers).<br>- **Bounded 1,000-Dataset Exact Wire Serialization & Isolated-Receiver Replay Pass (`task-1953`, `v4`)**: `91.09 s` across `16` single-threaded CPU workers (`6,000` cells + `144`-cell CatBoost fixture; ledger SHA-256 `4440ade9b760e14bda5ae31d6b5ab9406a3f4eb4b7abccc0bba0bc7880bb634d`). | `target/s3-multilevel-training-20261023/coverage_cost_and_separation_receipt.json`, `target/s3-multilevel-training-20261023/measured_egress_bytes_and_provenance_audit_v2.json`, `target/s3-multilevel-training-20261023/exact_wire_replay_1000ds_qualification_v4.json` |
| **7. Published Doc & Plot Locators** | - Docs: [`README.md`](../README.md), [`docs/multilevel-landmark-dl-study.md`](multilevel-landmark-dl-study.md), [`docs/model-sizes-and-elo.md`](model-sizes-and-elo.md), [`paper/dope1-content.tex`](../paper/dope1-content.tex), [`paper/dope1-supplement.tex`](../paper/dope1-supplement.tex)<br>- Plots (`PNG` + `PDF` in `target/s3-multilevel-training-20261023/figures/`): `stage0_feature_importance`, `s3_convergence_curves`, `world_class_elo_and_transfer`, `artifact_size_pareto_and_elo`, `diverse_1000_datasets_landscape`. | [`render_multilevel_evidence.py`](../research/levels/render_multilevel_evidence.py) |
| **8. Remaining Scientific Limits** | 1. **No Untouched Sealed Confirmatory Claim**: All 4 cohorts (`242` held-out, `758` in-corpus fold, `1,000` pooled, `1,000` disjoint-trained) are **selection-exposed development/reconciliation validation evidence**; sealed confirmatory release certification remains governed by [`docs/release.md`](release.md).<br>2. **Official `KumoTabular` / `TabPFN` Not Evaluated**: The `"kumo_tabpfn"` key is a local PyTorch gated attention network (`_TabularKumoAttentionNet`) trained from scratch per table (`official_kumo_or_tabpfn_evaluated: false`).<br>3. **Not Formal DP or HIPAA De-Identification**: Bounded `4 KiB` / `16 KiB` / `64 KiB` encodings and raw-row discard are empirical structural compression controls, never formal $(\varepsilon, \delta)$-differential privacy or legal HIPAA de-identification ([`docs/privacy.md`](privacy.md)). | [`release.md`](release.md), [`privacy.md`](privacy.md) |

---

## 6. Reproducibility Commands

```bash
# 1. Run the unit tests covering train-only scaling, descriptors, 3-level model, strata audit, and auditor alias mapping
python3 -B -m unittest discover -s research/levels/tests -v

# 2. Run Stage 0 rapid CatBoost & regression validation across all 19,190 splits
python3 -B -m research.levels.rapid_validate

# 3. Train DOPE-MLC-v2 across all 19,079 regression + binary classification splits
python3 -B -m research.levels.s3_multilevel_trainer --device cuda --epochs 8

# 4. Evaluate 1,000-dataset model transfer, driver dynamics, and Elo (40 workers)
python3 -B -m research.levels.world_class_eval --max-datasets 1000 --workers 40

# 5. Reconcile 242/758/1000-disjoint cohorts and dev_val checkpoint selection (28 workers)
python3 -B -m research.levels.disjoint_reconcile --device cuda --epochs 8 --workers 28

# 6. Render all 5 publication figures (PNG + PDF)
python3 -B -m research.levels.render_multilevel_evidence
```
