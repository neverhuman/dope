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
  `contrast_signal_ceiling` (`22.27%`), `midrank_lin` (`8.41%`), `shape_log10_rows` (`4.80%`),
  `gain_lin` (`4.79%`), `contrast_macro_add_share` (`1.96%`), and `contrast_periodic_excess` (`1.63%`).
- **Train-Only Scaling Probe (`stage_0b`)**: Across `36` stratified real holdout datasets, `FitOnlyTableTransform`
  improved mean CatBoost holdout score (`0.5671` vs `0.5532` naive z-score) while preserving
  CatBoost driver attribution fidelity $D = \mathbf{0.8238}$ and Spearman $\rho = \mathbf{0.6125}$ (`lightgbm` $\rho = 0.4555$, `linear` $\rho = 0.6079$).

---

## 3. Full-Corpus 3-Level GPU Deep Learning Convergence (19,079 Splits; 600-Split Per-Epoch Monitors + Full 1,891-Split Final Evaluation)

`research/levels/s3_multilevel_trainer.py` trained `DOPE-MLC-v2` on GPU (`cuda`) across
**19,079 matched splits** (`3,188` unique datasets; `14,295` regression + `4,784` binary
classification across all 6 split families; **`17,188` training splits** partitioned into 6
disjoint shards of `~2,865` splits for Epochs `1..6` plus 2 refinement shards for Epochs `7..8`
[`22,918` cumulative splits visited] and **`1,891` unseen-dataset validation splits** grouped by
`dataset_id` so zero splits of any validation dataset appear in training). Per
`research/levels/s3_multilevel_trainer.py:319-321,396-397` and canonical receipt
`target/s3-multilevel-training-20261023/s3_training_convergence_receipt.json`
(`SHA-256: e026225c7a0ee496d788888f152a81b9c4a0cbba7775d46cc8639dacd04083ab`, tracked snapshot:
[`bench/real-world/results/multilevel-convergence-snapshot.json`](../bench/real-world/results/multilevel-convergence-snapshot.json)),
each epoch `0..8` is monitored on deterministic **`600`-split subsets** (`train_eval_subset` of
`600 / 17,188` training splits and `val_eval_subset` of `600 / 1,891` unseen-dataset validation
splits), followed by a separate post-training evaluation (`full_1891_val_final`) of the reloaded
best checkpoint across **all `1,891` unseen-dataset validation splits**:

| Checkpoint / Evaluation Scope | Cumulative Splits Visited | Monitored Sample (`train / val`) | Train Loss (`600`-Split Mean) | Val `Medium` (`4 KiB`) Loss | Val `High` (`16 KiB`) Loss | Val `Ultra` (`64 KiB`) Loss | Val Mean Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Epoch `0` (Init) | `0` | `600 / 600` | `0.42415` | `0.45188` | `0.44037` | `0.42577` | `0.43934` |
| Epoch `1` | `2,865` | `600 / 600` | `0.36515` | `0.40085` | `0.37730` | `0.36162` | `0.37992` |
| Epoch `2` | `5,730` | `600 / 600` | `0.36322` | `0.40030` | `0.37515` | `0.35890` | `0.37812` |
| Epoch `3` | `8,595` | `600 / 600` | `0.36279` | `0.39961` | `0.37489` | `0.35850` | `0.37767` |
| Epoch `4` | `11,460` | `600 / 600` | `0.36272` | `0.39919` | `0.37430` | `0.35865` | `0.37738` |
| Epoch `5` | `14,324` | `600 / 600` | `0.36273` | `0.39927` | `0.37469` | `0.35917` | `0.37771` |
| Epoch `6` (`100%` of `17,188` train splits visited) | `17,188` | `600 / 600` | `0.36217` | `0.39899` | `0.37401` | `0.35791` | `0.37697` |
| **Epoch `7` (Best Monitored `val_eval_subset`, Saved `best_state`)** | **`20,053`** | **`600 / 600`** | **`0.36193`** | **`0.39900`** | **`0.37377`** | **`0.35725`** | **`0.37667`** |
| Epoch `8` (Final Epoch; Receipt `final_best_val` Field) | `22,918` | `600 / 600` | `0.36201` | `0.39893` | `0.37394` | `0.35770` | `0.37686` |
| **Post-Train Full `1,891` Unseen Val (`full_1891_val_final`, Reloaded Ep `7` `best_state`)** | **`22,918`** | **`— / 1,891`** | **`—`** | **`0.39744`** | **`0.37456`** | **`0.36060`** | **`0.37753`** |

- **Monitored Endpoint Reduction (`-14.3%`) vs. Full `1,891`-Split Final Evaluation**:
  On the matching `600`-split unseen-dataset validation monitor (`val_eval_subset`), mean validation
  loss decreases by **`-14.3%` (`-14.27%`, `0.43934 -> 0.37667` at Epoch `7`, or `-14.22%`,
  `0.43934 -> 0.37686` at Epoch `8`)** while maintaining strict monotonic tier ordering
  `Loss(Ultra) < Loss(High) < Loss(Medium)` at every single epoch `0..8`. After Epoch `8`,
  reloading `best_state` (Epoch `7`) and evaluating across all `1,891` unseen-dataset validation
  splits (`full_1891_val_final`, a distinct `1,891`-split sample for which no Epoch `0` full-panel
  baseline was recorded) yields **`0.37753`** (`Ultra = 0.36060 < High = 0.37456 < Medium = 0.39744`).
- **Checkpoint Selection & Receipt-Field Binding (`dope_mlc_v2_s3_best.pt`)**:
  In `s3_multilevel_trainer.py:409-416`, `best_state` is updated only when
  `val_ep["mean_total_loss"] <= best_val_loss`; thus `dope_mlc_v2_s3_best.pt`
  (`SHA-256: 1d5db98ffa438a0be1c61dabf235972cd400d1010f023705bea983f96e660a98`) stores the
  **Epoch `7`** weights (`0.37667` on the `600`-split monitor) and `full_1891_val_final` (`0.37753`)
  is evaluated on those reloaded Epoch `7` weights, whereas line `431` wrote
  `"final_best_val": history[-1]["val"]` (recording the Epoch `8` `600`-split monitor entry `0.37686`).

### 3.1 Maximized-Training (`~96.4%–97.2%` Train, `< 5%` Validation) Full-Validation-Per-Round Convergence & Dataset-Grouped Bootstrap SEs (`task-4221`, `maxtrain97_val_lt5pct`)

To maximize training coverage while enforcing **`< 5%` validation share on both datasets and splits**
and eliminating subsampled validation-monitor noise, the `task-4221` runner (`research/levels/maxtrain_val_lt5pct.py`,
`608` lines, `36,509` bytes, `SHA-256: 701d9cca26c2f8101802448974febbe12fcc21892734da208123a3b2fa4023fb`;
preserved untouched at `target/s3-multilevel-training-20261023/source_maxtrain_val_lt5pct_at_task4221.py`
and embedded byte-for-byte with SHA-256 verification under `executed_source_custody` in
[`bench/real-world/results/multilevel-convergence-snapshot.json`](../bench/real-world/results/multilevel-convergence-snapshot.json);
tracked custody verifier and byte-for-byte restorer:
[`research/levels/disjoint_reconcile.py`](../research/levels/disjoint_reconcile.py); pre-launch manifest
`target/s3-multilevel-training-20261023/maxtrain97_val_lt5pct_split_and_dispatch_manifest.json`
`SHA-256: 6ad04935...`; round/pass accounting addendum
`target/s3-multilevel-training-20261023/maxtrain97_training_round_and_pass_accounting_addendum.json`)
evaluates **`100%` of the validation split population at every training round `R0..R8`** and computes
`200x` **dataset-grouped bootstrap standard errors (`SE`) and 95% CIs** (`_dataset_grouped_bootstrap`,
aggregating splits to dataset means before resampling datasets with replacement), while preserving
all historical checkpoints (`dope_mlc_v2_s3_best.pt`, `dope_mlc_v2_disjoint1000_best.pt`) and
receipts untouched:

- **Training-Round vs. Full-Population Pass Accounting (`R0..R8`)**:
  In the executed `task-4221` script (`maxtrain_val_lt5pct.py:363-402`, restorable via
  [`disjoint_reconcile.py`](../research/levels/disjoint_reconcile.py)), the loop index `ep = 1..8` iterates over 6 disjoint training
  shards (`R1..R6`, completing **`1.000000×` full-population pass at `R6`**) followed by 2 additional
  refinement shard rounds (`R7..R8`, reaching **`1.333351×`** effective full-population passes in
  Track 1 and **`1.333333×`** in Track 2) and logs raw JSON receipt keys `"epoch": 0..8` and
  `"splits_visited_cumulative"`. Accordingly, `R0..R8` are reported as **Training Rounds (`R0..R8`)**
  alongside **Attempted Cumulative Split Visits**, **Effective Full-Population Passes**, and
  **Scheduled Optimizer Updates** (`batch_accum = 32`; actual finite-gradient accumulation counts
  were not separately logged in the raw receipts).
- **Track 1 — Full-Corpus `DOPE-MLC-v2` (`v2_full_corpus_maxtrain97_val3pct`, `s3_training_maxtrain97_val3pct_convergence_receipt.json` `SHA-256: 3025e025...`, Checkpoint `dope_mlc_v2_s3_maxtrain97_best.pt` `SHA-256: 11e64c63...`)**:
  - **Dataset & Split Partition (`int(sha256(dataset_id)[:8], 16) % 32 == 0`)**: **`3,073 / 3,188` training datasets (`96.3927%`)** and **`18,395 / 19,079` training splits (`96.4149%`, `+1,207` splits vs. historical `17,188`)** vs. **`115 / 3,188` unseen validation datasets (`3.6073% < 5%`)** and **`684 / 19,079` validation splits (`3.5851% < 5%`: `87` regression DS / `519` splits, `28` binary DS / `165` splits; `0` dataset overlap with training)**. Because `295` former `10%`-validation datasets from `s3_training_convergence_receipt.json` are reassigned into the `3,073`-dataset training pool, Track 1 is explicitly classified as `in_corpus_full_s3_track_with_295_reused_v2_dev_val_datasets_not_disjoint_from_1000_eval` (`unqualified_for_historical_1000_v2_or_1069_v3_benchmark_claims_until_dedicated_full_panel_evaluation`).
  - **Trajectory Summary (`100%` Full `684`-Split Validation Every Round)**: Across `R0..R8` (`24,527` cumulative split visits = `1.333351×` passes, `768` scheduled updates), `600`-split training monitor loss decreases from `0.41216 ± 0.00866` (`R0`) to **`0.35175 ± 0.01022`** (`R8*`), and **full `684/684`-split unseen validation loss** decreases by **`-16.52%`** from `0.40988 ± 0.01767` (`R0`, 95% CI `[0.38301, 0.45101]`) to **`0.34216 ± 0.01621`** (`R8*`, selected best & final round, `reevaluated_best_state_val_loss = 0.34216`, 95% CI `[0.31104, 0.36990]`), maintaining strict monotonic tier ordering `Loss(Ultra) < Loss(High) < Loss(Medium)` at every round `R0..R8`:

| Training Round (`raw epoch`) | Cumulative Splits Visited | Effective Full-Pop Passes (`18,395` Train Splits) | Scheduled Updates (`batch=32`) | Train Monitor (`600` Splits) Mean $\pm$ SE | Full Val (`684/684` Splits, `115` DS) Mean $\pm$ SE `[95% CI]` | Val `Medium` (`4 KiB`) $\pm$ SE | Val `High` (`16 KiB`) $\pm$ SE | Val `Ultra` (`64 KiB`) $\pm$ SE | Reg (`87` DS) / Bin (`28` DS) Mean $\pm$ SE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R0` (`epoch=0`) | `0` | `0.000000×` | `0` | `0.41216 ± 0.00866` | **`0.40988 ± 0.01767`** `[0.38301, 0.45101]` | `0.42542 ± 0.01727` | `0.40850 ± 0.01779` | **`0.39572 ± 0.01807`** | `0.40301 ± 0.02227` / `0.43147 ± 0.02139` |
| `R1` (`epoch=1`) | `3,066` | `0.166676×` | `96` | `0.35967 ± 0.00960` | **`0.34878 ± 0.01894`** `[0.32026, 0.39003]` | `0.37269 ± 0.01876` | `0.34477 ± 0.01910` | **`0.32889 ± 0.01936`** | `0.33254 ± 0.02337` / `0.39986 ± 0.02419` |
| `R2` (`epoch=2`) | `6,132` | `0.333351×` | `192` | `0.35359 ± 0.00960` | **`0.34308 ± 0.01846`** `[0.31038, 0.38084]` | `0.36488 ± 0.01819` | `0.33930 ± 0.01858` | **`0.32506 ± 0.01906`** | `0.32709 ± 0.02469` / `0.39338 ± 0.02437` |
| `R3` (`epoch=3`) | `9,198` | `0.500027×` | `288` | `0.35331 ± 0.01000` | **`0.34256 ± 0.01995`** `[0.30593, 0.38080]` | `0.36464 ± 0.01947` | `0.33854 ± 0.02015` | **`0.32450 ± 0.02067`** | `0.32642 ± 0.02406` / `0.39333 ± 0.02311` |
| `R4` (`epoch=4`) | `12,264` | `0.666703×` | `384` | `0.35324 ± 0.00999` | **`0.34379 ± 0.01804`** `[0.31376, 0.37975]` | `0.36459 ± 0.01772` | `0.34017 ± 0.01812` | **`0.32661 ± 0.01867`** | `0.32800 ± 0.02454` / `0.39345 ± 0.02331` |
| `R5` (`epoch=5`) | `15,330` | `0.833379×` | `480` | `0.35250 ± 0.00946` | **`0.34230 ± 0.01946`** `[0.30710, 0.37750]` | `0.36442 ± 0.01940` | `0.33843 ± 0.01959` | **`0.32404 ± 0.01982`** | `0.32600 ± 0.02202` / `0.39356 ± 0.02422` |
| `R6` (`epoch=6`, `1.00×` Pass) | `18,395` | `1.000000×` | `576` | `0.35238 ± 0.01092` | **`0.34264 ± 0.01976`** `[0.30282, 0.37724]` | `0.36432 ± 0.01962` | `0.33912 ± 0.02002` | **`0.32448 ± 0.02001`** | `0.32661 ± 0.02547` / `0.39305 ± 0.02202` |
| `R7` (`epoch=7`) | `21,461` | `1.166676×` | `672` | `0.35208 ± 0.00981` | **`0.34216 ± 0.01779`** `[0.30806, 0.37548]` | `0.36417 ± 0.01737` | `0.33839 ± 0.01801` | **`0.32391 ± 0.01849`** | `0.32609 ± 0.02143` / `0.39269 ± 0.02452` |
| **`R8*` (`epoch=8`, Selected Best & Final)** | `24,527` | `1.333351×` | `768` | `0.35175 ± 0.01022` | **`0.34216 ± 0.01621`** `[0.31104, 0.36990]` | `0.36407 ± 0.01608` | `0.33844 ± 0.01635` | **`0.32396 ± 0.01672`** | `0.32614 ± 0.02142` / `0.39255 ± 0.02317` |

- **Track 2 — Strictly Disjoint-1,000 `DOPE-MLC-v2` (`v2_disjoint1000_maxtrain97_devval345`, `s3_training_disjoint1000_devval345_convergence_receipt.json` `SHA-256: e510a630...`, Checkpoint `dope_mlc_v2_disjoint1000_maxtrain97_best.pt` `SHA-256: 3886676a...`)**:
  - **Dataset & Split Partition (`0` Overlap with `1,132` Evaluation Candidates)**: **`2,001 / 2,059` training datasets (`97.1831%`)** and **`11,976 / 12,321` training splits (`97.1999%`)** vs. **`58 / 2,059` `dev_val` datasets (`2.8169% < 5%`)** and **`345 / 12,321` `dev_val` splits (`2.8001% < 5%`: `41` regression DS / `243` splits, `17` binary DS / `102` splits)**, eliminating the historical `val_sub` evaluation-candidate monitor during training.
  - **Trajectory Summary (`100%` Full `345`-Split `dev_val` Every Round)**: Across `R0..R8` (`15,968` cumulative split visits = `1.333333×` passes, `504` scheduled updates), `600`-split training monitor loss decreases from `0.47854 ± 0.00904` (`R0`) to **`0.41492 ± 0.00937`** (`R8`), and **full `345/345`-split `dev_val` loss** decreases by **`-5.48%`** from `0.60691 ± 0.01915` (`R0`) to **`0.57363 ± 0.02131`** at selected best round **`R5*`** (`0.833333×` passes, `reevaluated_best_state_val_loss = 0.57363`, 95% CI `[0.52561, 0.61162]`; `Ultra 0.56909 ± 0.02190 < High 0.57383 ± 0.02155 < Medium 0.57798 ± 0.02078`) and **`0.57371 ± 0.02074`** at final round `R8` (`1.333333×` passes; `Ultra 0.56935 ± 0.02174 < High 0.57394 ± 0.02098 < Medium 0.57784 ± 0.01980`):

| Training Round (`raw epoch`) | Cumulative Splits Visited | Effective Full-Pop Passes (`11,976` Train Splits) | Scheduled Updates (`batch=32`) | Train Monitor (`600` Splits) Mean $\pm$ SE | Full `dev_val` (`345/345` Splits, `58` DS) Mean $\pm$ SE `[95% CI]` | `dev_val` `Medium` $\pm$ SE | `dev_val` `High` $\pm$ SE | `dev_val` `Ultra` $\pm$ SE | Reg (`41` DS) / Bin (`17` DS) Mean $\pm$ SE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `R0` (`epoch=0`) | `0` | `0.000000×` | `0` | `0.47854 ± 0.00904` | **`0.60691 ± 0.01915`** `[0.57115, 0.64344]` | `0.61345 ± 0.01834` | `0.60194 ± 0.01932` | **`0.60535 ± 0.01990`** | `0.63276 ± 0.02228` / `0.54533 ± 0.02556` |
| `R1` (`epoch=1`) | `1,996` | `0.166667×` | `63` | `0.42499 ± 0.00864` | **`0.58960 ± 0.02298`** `[0.54024, 0.63502]` | `0.59437 ± 0.02193` | `0.58906 ± 0.02347` | **`0.58536 ± 0.02392`** | `0.61660 ± 0.03024` / `0.52527 ± 0.02387` |
| `R2` (`epoch=2`) | `3,992` | `0.333333×` | `126` | `0.41636 ± 0.00953` | **`0.57558 ± 0.02163`** `[0.53243, 0.61414]` | `0.57933 ± 0.02115` | `0.57586 ± 0.02199` | **`0.57154 ± 0.02211`** | `0.60058 ± 0.02545` / `0.51602 ± 0.02636` |
| `R3` (`epoch=3`) | `5,988` | `0.500000×` | `189` | `0.41625 ± 0.00967` | **`0.57433 ± 0.02170`** `[0.52461, 0.61121]` | `0.57829 ± 0.02114` | `0.57432 ± 0.02195` | **`0.57037 ± 0.02231`** | `0.59927 ± 0.02549` / `0.51490 ± 0.02468` |
| `R4` (`epoch=4`) | `7,984` | `0.666667×` | `252` | `0.41548 ± 0.00925` | **`0.57652 ± 0.02030`** `[0.52820, 0.61198]` | `0.57867 ± 0.01976` | `0.57719 ± 0.02076` | **`0.57370 ± 0.02078`** | `0.60250 ± 0.02715` / `0.51463 ± 0.02272` |
| **`R5*` (`epoch=5`, Selected Best `best_state`)** | `9,980` | `0.833333×` | `315` | `0.41521 ± 0.00932` | **`0.57363 ± 0.02131`** `[0.52561, 0.61162]` | `0.57798 ± 0.02078` | `0.57383 ± 0.02155` | **`0.56909 ± 0.02190`** | `0.59855 ± 0.02632` / `0.51428 ± 0.02442` |
| `R6` (`epoch=6`, `1.00×` Pass) | `11,976` | `1.000000×` | `378` | `0.41516 ± 0.00942` | **`0.57493 ± 0.02208`** `[0.52872, 0.61125]` | `0.57829 ± 0.02157` | `0.57536 ± 0.02237` | **`0.57113 ± 0.02268`** | `0.60052 ± 0.02879` / `0.51396 ± 0.02258` |
| `R7` (`epoch=7`) | `13,972` | `1.166667×` | `441` | `0.41506 ± 0.00906` | **`0.57402 ± 0.01972`** `[0.52692, 0.61484]` | `0.57796 ± 0.01927` | `0.57429 ± 0.01999` | **`0.56982 ± 0.02027`** | `0.59934 ± 0.02630` / `0.51371 ± 0.02432` |
| `R8` (`epoch=8`, Final Round) | `15,968` | `1.333333×` | `504` | `0.41492 ± 0.00937` | **`0.57371 ± 0.02074`** `[0.53669, 0.61258]` | `0.57784 ± 0.01980` | `0.57394 ± 0.02098` | **`0.56935 ± 0.02174`** | `0.59890 ± 0.02657` / `0.51370 ± 0.02233` |

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
    `1,132` evaluation candidates)**. On `dev_val_only` (`58` datasets, `345` splits; `dev_val_only_convergence` in `s3_training_disjoint1000_convergence_receipt.json`, `SHA-256: 60e8081bc23c210294558881eb3fd7eabb04e0a0bd5a574b0c47288cf1a7078a`), validation
    loss converged from `0.60288` (`Epoch 0`) to **`0.57414` (`Epoch 8`)** with strict monotonic
    tier ordering (`Ultra = 0.57053 < High = 0.57397 < Medium = 0.57791`), while the receipt's per-epoch `convergence_history` (`Epochs 0..8`, `15,968` cumulative splits) recorded the `600`-split training monitor (`0.45380 -> 0.39686`) and the prior `600`-split `val_sub` monitor drawn from the `6,758` held-out evaluation splits in cache (`0.32574 -> 0.24904` at `Epochs 5` and `8`, `selected_epoch = 8`). Because `Epoch 8` was
    also monitored on `val_sub` during the initial reconciliation pass, all four cohorts are
    reported as **development/reconciliation validation evidence** (never claimed as an untouched
    sealed confirmatory test).

### 4.2 Separate Cohort Comparison: Strictly Held-Out (`242`) vs. In-Corpus Unseen-Fold (`758`) vs. Disjoint-Trained (`1,000`)

| Cohort / Stratum | Datasets ($n$) | Evidence & Separation Status | `DOPE-MLC-v2` Overall Elo `[95% CI]` | Win Share | Cross-Size Elo (`4 KiB` $\to$ `16 KiB` $\to$ `64 KiB`) | `Medium` (`4 KiB`) Retention $R$ (Mean / Med) | `High` (`16 KiB`) Retention $R$ (Mean / Med) | `Ultra` (`64 KiB`) Retention $R$ (Mean / Med) | `Ultra` Driver Fidelity $D$ `[95% CI]` |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| **`strictly_dataset_disjoint_242`** | **`242`** | `0` splits in gradient training (`10%` val hash split, dev-selected) | **`1671.1` `[1655.2, 1688.0]` (`#1`)** | **`73.06%`** | **`1626.7` $\to$ `1690.5` $\to$ `1703.7`** | `0.8122` / `0.8991` | `0.8464` / `0.9327` | **`0.8493` / `0.9275`** | **`0.8034` `[0.7905, 0.8147]` (`D80` pass)** |
| **`in_corpus_unseen_row_fold_758`** | **`758`** | In-corpus dataset IDs, unseen `outer5-iid-fold0-v1` test rows (`row-disjoint`) | **`1645.4` `[1637.8, 1653.6]` (`#1`)** | **`69.81%`** | **`1587.6` $\to$ `1676.4` $\to$ `1682.3`** | `0.8779` / `0.9439` | `0.9133` / `0.9692` | **`0.9129` / `0.9716`** | **`0.7851` `[0.7779, 0.7918]`** |
| **`pooled_1000` (Preserved Initial)** | **`1,000`** | `242` strictly held-out + `758` in-corpus unseen test folds (`task-797`, ckpt `1d5db98f...` Epoch 7 `best_state`; receipt `final_best_val` field = Epoch 8) | **`1651.3` `[1643.4, 1659.3]` (`#1`)** | **`70.58%`** | **`1596.8` $\to$ `1679.6` $\to$ `1687.2`** *(per-auditor)* / `1610.6` $\to$ `1726.6` $\to$ `1742.0` *(DS-mean)* | `0.8620` / `0.9335` | `0.8971` / `0.9642` | **`0.8975` / `0.9658`** | **`0.7896` `[0.7831, 0.7959]`** |
| **`reconciled_strictly_dataset_disjoint_1000`** | **`1,000`** | **`0` overlap** between `2,001` gradient-training datasets and all `1,132` eval candidates (`ckpt 6fcc191a...`, fixed `Epoch 8` verified on `58` `dev_val` datasets; prior `val_sub` monitoring disclosed) | **`1650.7` `[1643.4, 1657.5]` (`#1`)** | **`70.51%`** | **`1596.1` $\to$ `1675.5` $\to$ `1690.1`** | `0.8620` / `0.9349` | `0.8946` / `0.9624` | **`0.8998` / `0.9671`** | **`0.7904` `[0.7844, 0.7972]`** |

### 4.3 Overall 1,000-Dataset Multi-Auditor Bradley–Terry Elo Leaderboard (Preserved Pooled & Reconciled 100% Disjoint)

Across **44,369 pairwise auditor-dataset matches per method** (with 200× dataset-cluster bootstrap
95% confidence intervals), `DOPE-MLC-v2` achieves **Rank #1 overall Elo (`1651.3`)**:

| Rank | Generator Method | Overall Elo | 95% Bootstrap CI | Bootstrap SE | Wins | Draws | Losses | Win Share |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| **#1** | **`dope_mlc_v2` (Unified 3-Level Model)** | **`1651.3`** | **`[1643.4, 1659.6]`** | `4.26` | **30,959** | **717** | **12,693** | **`70.58%`** |
| #2 | `rows` (Non-Anonymized Raw Row Subsample) | `1593.3` | `[1584.4, 1602.7]` | `4.40` | 27,150 | 993 | 16,222 | `62.32%` |
| #3 | `kernel` (DOPE v1 Kernel Sketch) | `1592.5` | `[1583.0, 1603.0]` | `4.86` | 27,108 | 977 | 16,284 | `62.20%` |
| #4 | `copula` (Gaussian Copula) | `1545.8` | `[1535.7, 1553.5]` | `4.42` | 23,905 | 1,255 | 19,209 | `55.29%` |
| #5 | `joint_diffusion_v1` (Tabular Diffusion) | `1432.4` | `[1423.8, 1440.8]` | `4.58` | 16,822 | 746 | 26,801 | `38.75%` |
| #6 | `arf` (Adversarial Random Forests) | `1184.7` | `[1167.3, 1202.3]` | `9.37` | 4,370 | 894 | 39,105 | `10.86%` |

*(Match count note: across the `8,874` `elo_cells` in `per_dataset_1000_results.jsonl.gz`, `8,873` cells contain all `6` methods while `1` cell [`17cd766e7ec82f1e` at `medium|catboost`] contains `5` methods with `rows` absent, yielding `8,873 × 5 + 4 = 44,369` pairwise matches for `dope_mlc_v2`, `kernel`, `copula`, `joint_diffusion_v1`, and `arf`, and `8,873 × 5 = 44,365` pairwise matches for `rows`.)*

### 4.3 Elo & Transfer Performance vs. Nominal Budget Cap, Exact-Replay Wire Bytes (`v2`) & Legacy Lossy Wire Bytes (`v1`)

When all 18 `(method, tier)` pairs compete in a single **Unified Cross-Size
Bradley–Terry Pool**, `DOPE-MLC-v2`'s Elo rises monotonically with the **nominal tier budget cap**
(`4,096 B / 16,384 B / 65,536 B`), the **exact-replay `version=2` normalized-space latent wire payload**
(`1,502 B / 2,836 B / 5,872 B` median raw bytes; `1,238 B / 2,351 B / 4,616 B` median zlib bytes, `+22 B` raw over legacy `version=1` lossy `1,480 B / 2,814 B / 5,850 B`),
and the **full raw-domain replay payload including the mandatory `FitOnlyTableTransform` + seed sidecar**
(`2,031 B / 3,376 B / 6,398 B` median raw bytes for `version=2` vs `2,009 B / 3,354 B / 6,376 B` for `version=1`, adding the `545 B` median / `876.3 B` mean scaler sidecar across the `1,000` datasets).
Across all **`1,000` canonical datasets, `3` tiers, and both frozen checkpoints (`6,000 / 6,000` cells)** in `exact_wire_replay_1000ds_qualification_v4.json`,
the `version=2` codec (`dope.latent_wire.v3_exact`) achieves **`100.0%` (`6,000/6,000`) isolated-receiver decode success with bit-for-bit identical `X_syn` (`max_abs_diff = 0.0`) AND bit-for-bit identical `y_syn` (`max_abs_diff = 0.0`)**
using only the wire payload plus the frozen `292,707 B` shared decoder (`dope_mlc_v2_wire_1000ds_qualified = true`, verified on the 24-dataset `144`-cell CatBoost fixture with `median |ΔR| = 0.0000` and `max |ΔR| = 0.0000` vs legacy `version=1` `max |ΔR| = 0.6283`;
competing baseline byte tallies remain post-hoc `fp16` estimates paired with historical `float64` in-memory scores, so `baseline_wire_1000ds_qualified = false` and `full_cross_method_wire_1000ds_qualified = false`):

| Tier & Nominal Cap (`DOPE` Exact `v2` Latent Raw / Zlib \| Full Raw w/ Sidecar `[Legacy v1 Raw]`) | `DOPE-MLC-v2` Cross-Size Elo (Per-Auditor / DS-Mean) | `rows` Cross-Size Elo (`1,024 / 4,516 / 8,464 B` Raw Est.) | `kernel` Cross-Size Elo (`202 B` Raw Est., Tier-Inv) | `copula` Cross-Size Elo (`226 B` Raw Est., Tier-Inv) | `joint_diffusion_v1` Cross-Size Elo (`9,616 B` Full-Table Est.) | `arf` Cross-Size Elo (`10,426 B` Full-Table Est., Tier-Inv) | `DOPE-MLC-v2` Within-Tier Elo |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Medium`: `4,096 B` Cap (`1,502 B` / `1,238 B` \| `2,031 B` full `[v1: 1,480 B]`)** | **`1596.8` / `1610.6`** | `1424.5` / `1415.3` | `1594.4` / `1596.0` | `1546.0` / `1526.4` | `1358.5` / `1332.0` | `1175.9` / `1140.5` | **`1652.1` (`#1`, `71.11%` win)** |
| **`High`: `16,384 B` Cap (`2,836 B` / `2,351 B` \| `3,376 B` full `[v1: 2,814 B]`)** | **`1679.6` / `1726.6`** | `1622.2` / `1648.3` | `1594.4` / `1596.0` | `1546.0` / `1526.4` | `1435.5` / `1423.5` | `1175.9` / `1140.5` | **`1669.6` (`#1`, `72.62%` win)** |
| **`Ultra`: `65,536 B` Cap (`5,872 B` / `4,616 B` \| `6,398 B` full `[v1: 5,850 B]`)** | **`1687.2` / `1742.0`** | `1753.0` / `1818.5` *(copies `88%` raw rows)* | `1594.4` / `1596.0` | `1546.0` / `1526.4` | `1493.6` / `1494.7` | `1175.9` / `1140.5` | **`1640.1` (`#1` encoder, `68.02%` win)** |

- At **`Medium` (`4,096 B` nominal cap; `1,502 B` median exact `v2` latent raw / `1,238 B` zlib wire bytes; `2,031 B` median full raw with scaler sidecar; `86.4% <= 4,096 B` latent raw, `94.4%` latent zlib, `77.5%` full raw, `85.2%` est. full zlib)**,
  `DOPE-MLC-v2` beats the non-anonymized raw row subsample (`rows`, `1,024 B` latent raw fp16 / `3,998 B` ASCII budget)
  by **`+172.3` per-auditor cross-size Elo points (`+195.3` dataset-mean Elo)** and `+18.77` retention points (`0.8620` vs `0.6743`).
- At **`High` (`16,384 B` nominal cap; `2,836 B` median exact `v2` latent raw / `2,351 B` zlib wire bytes; `3,376 B` median full raw with scaler sidecar, `100.0% <= 16 KiB` under all four byte definitions)**,
  `DOPE-MLC-v2` beats `rows` (`4,516 B` median latent raw fp16 / `16,276 B` ASCII budget) by **`+57.4` per-auditor cross-size Elo points
  (`+78.3` dataset-mean Elo)** and `+4.45` retention points (`0.8971` vs `0.8526`).
- At **`Ultra` (`65,536 B` nominal cap; `5,872 B` median exact `v2` latent raw / `4,616 B` zlib wire bytes; `6,398 B` median full raw with scaler sidecar, `100.0% <= 64 KiB` under all four byte definitions)**,
  `DOPE-MLC-v2` reaches **`1687.2` per-auditor cross-size Elo (`1742.0` dataset-mean Elo, `#1` encoder within-tier Elo `1640.1`)**; the non-anonymized `rows` control (`1714.7` within-tier Elo)
  only edges higher at `Ultra` because its `64 KiB` ASCII budget copies **`88%` of the raw training rows**.

### 4.4 Strict Monotonic Multi-Level Progression (`DOPE-MLC-v2`, 1,000 Datasets)

| Tier | Nominal Cap | Exact `v2` Latent Wire Med (Raw / Zlib) \| Full Raw w/ Sidecar | Mean Retention $R$ `[95% CI]` | Median $R$ `[IQR p25, p75]` | Mean Driver Fidelity $D$ `[95% CI]` | Feature Spearman $\rho$ | Top-3 Driver Overlap | BTR-AUC | Regression $R^2$ |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| **`Medium` (Macro / Legal)** | `4,096 B` (`4 KiB`) | **`1,502 B` / `1,238 B` \| `2,031 B`** | **`0.8620` `[0.8503, 0.8722]`** | **`0.9335` `[0.8190, 0.9784]`** | **`0.7566` `[0.7518, 0.7631]`** | **`0.5985`** | `69.95%` | **`83.11`** | `0.6620` |
| **`High` (Larger / Interaction)** | `16,384 B` (`16 KiB`) | **`2,836 B` / `2,351 B` \| `3,376 B`** | **`0.8971` `[0.8848, 0.9080]`** | **`0.9642` `[0.8839, 0.9919]`** | **`0.7821` `[0.7759, 0.7878]`** | **`0.6331`** | `72.35%` | **`86.82`** | `0.7049` |
| **`Ultra` (Fine / Max Transfer)** | `65,536 B` (`64 KiB`) | **`5,872 B` / `4,616 B` \| `6,398 B`** | **`0.8975` `[0.8883, 0.9094]`** | **`0.9658` `[0.8770, 0.9942]`** | **`0.7896` `[0.7831, 0.7959]`** | **`0.6363`** | **`72.42%`** | **`86.97`** | **`0.7145`** |

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
| **Signal Nonlinearity** | `Mostly Linear (<=0.04)` | `581` | `0.9401` | `0.9579` | `0.9459` | **`0.7836`** |
| **Signal Nonlinearity** | `Moderate Nonlinear (0.04-0.14)` | `198` | `0.8637` | `0.8953` | **`0.8990`** | **`0.7953`** |
| **Signal Nonlinearity** | **`Strong Nonlinear (>0.14)`** | **`221`** | `0.6552` | `0.7390` | **`0.7689` (`+11.37` pts vs Med)** | **`0.8001`** |
| **Sample Size ($N_{\text{train}}$)** | `Small (<=250 rows)` | `313` | `0.8590` | **`0.8806`** | `0.8788` | `0.7728` |
| **Sample Size ($N_{\text{train}}$)** | `Medium (251-750 rows)` | `304` | `0.8480` | **`0.8929`** | `0.8917` | **`0.7978`** |
| **Sample Size ($N_{\text{train}}$)** | `Large (>750 rows)` | `383` | `0.8757` | `0.9139` | **`0.9174`** | **`0.7967`** |
| **Feature Width ($p$)** | `Narrow (<=8 cols)` | `358` | `0.8467` | `0.8764` | **`0.8810`** | **`0.8585`** |
| **Feature Width ($p$)** | `Medium (9-20 cols)` | `332` | `0.8806` | **`0.9213`** | `0.9186` | `0.7964` |
| **Feature Width ($p$)** | `Wide (>20 cols)` | `310` | `0.8598` | **`0.8952`** | `0.8940` | `0.7026` |
| **Task Family** | `Regression` | `627` | `0.8335` | `0.8842` | **`0.8933`** | **`0.8023`** |
| **Task Family** | `Binary Classification` | `373` | `0.9100` | **`0.9189`** | `0.9045` | `0.7682` |
| **Categorical Share** | `Pure Numeric (cat_fraction = 0.0)` | `1,000` | `0.8620` | `0.8971` | **`0.8975`** | **`0.7896`** |
| **Categorical Share** | `Non-Zero Categorical (>0% cat)` | `0` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |

- **Unlimited Synthetic Expansion from Fixed Encoding (`ds_rank < 150`, `n = 150` Datasets)**: Generating $1\times \to 2\times \to 5\times N_{\text{train}}$
  synthetic rows from the frozen `Ultra` encoding increases downstream CatBoost holdout retention
  monotonically from **`0.8808`** ($1\times N_{\text{train}}$) $\to$ **`0.9010`**
  ($2\times N_{\text{train}}$) $\to$ **`0.9070`** ($5\times N_{\text{train}}$, `+2.62` retention
  points) without revisiting raw training data (`850` remaining datasets: `UNKNOWN`).

---

## 5. Source-Bound 1,000-Dataset Study Acceptance Matrix & Scientific Limits

| Acceptance Dimension | Source-Bound Specification & Measured Values | Authoritative Code / Receipt / Plot Locator |
|---|---|---|
| **1. Corpus Coverage & Admission Funnel** | `3,188` S3 regression + binary datasets (`19,190` landmark splits; `19,079` valid splits across `3,188` datasets in `s3_prepared_splits_cache.pt`). Stratified candidate selector dispatched `1,132` unique `outer5-iid-fold0-v1` datasets (`258` strictly held-out `is_val_dataset=True` + `874` in-corpus `is_val_dataset=False`). Exact `1,000` datasets completed (`627` regression + `373` binary, sorted ID `SHA-256 6e37082867ae0147f3ddb4beffc4a3bc908c804d524fd1318ea5768fb7ee1be2`; `242` strictly held-out [`188` reg + `54` bin, `SHA-256 a785bf9c86aa56af59e479895e774d1a34bd963904398f3a54258eb4de064c89`], `93.80%` admission; `758` in-corpus [`439` reg + `319` bin, `SHA-256 0988a39b6be2d600ae757d0c80b6599b9fd9076db36b40457ebdf653ded37470`], `86.73%` admission; `0` overlap between `242` and `758`, exact union = `1,000`). `132` candidates (`16` held-out + `116` in-corpus) excluded strictly where real CatBoost holdout oracle gain $S_{\text{real}} - S_{\text{null}} < 0.025$ or degenerate split (`0` worker exceptions). Exact `3` candidate IDs (`7c29a13351d14ec5`, `a42f75735fd26aa1`, `d181d4da916c742b`) are absent from `s3_prepared_splits_cache.pt` (`1,129` vs `1,132` candidates; `0` in completed `1,000`, all `3` in filtered `132`). | [`world_class_eval.py`](../research/levels/world_class_eval.py), [`strata_audit.py`](../research/levels/strata_audit.py), `target/s3-multilevel-training-20261023/strata_and_disjoint_reconciliation_receipt.json`, `target/s3-multilevel-training-20261023/study_1000ds_and_1069ds_reconciled_canonical_index_v6.json` |
| **2. Cohort Strata & Checkpoint Selection Exposure** | **4 evaluated cohorts** (all explicitly labeled as **selection-exposed development/reconciliation validation evidence**, `untouched_sealed_confirmatory_test: false`):<br>1. `strictly_dataset_disjoint_242` ($n=242$): bound to `dope_mlc_v2_s3_best.pt` (`SHA-256 1d5db98ffa438a0be1c61dabf235972cd400d1010f023705bea983f96e660a98`, reloaded `Epoch 7` `best_state` after `Epoch 8`), `0` dataset overlap with `17,188` training splits (`2,869` fit datasets), but belongs to the `10%` validation split (`1,891` splits across `319` datasets) used to select `best_state`.<br>2. `in_corpus_unseen_row_fold_758` ($n=758$): bound to `dope_mlc_v2_s3_best.pt` (`1d5db98f...`, `Epoch 7` `best_state`), evaluated on unseen `outer5-iid-fold0-v1` test rows (`0` row overlap with fit split), but dataset IDs were in the `90%` training dataset pool.<br>3. `pooled_1000` ($n=1,000$, preserved initial run `task-797`, ledger `per_dataset_1000_results.jsonl.gz` `SHA-256 c268f738570fc10446791591cb26ca2f8a79fba828fd3f52594e4d617879d920`): union of `242` held-out and `758` in-corpus unseen-fold datasets.<br>4. `reconciled_strictly_dataset_disjoint_1000` ($n=1,000$, `task-942`, ledger `per_dataset_1000_strictly_disjoint_results.jsonl.gz` `SHA-256 5d62cc31d65c2e695de451ba906226e68bf613108f5dbcae41dc02bd7f37d3ab`): bound to `dope_mlc_v2_disjoint1000_best.pt` (`SHA-256 6fcc191a244ff409b9d44bbb6a2bcb1d8e3a7db326dcdfbb56f75a64b5f87da3`, fixed-horizon `Epoch 8`), trained on `2,001` datasets (`11,976` splits) excluding all `1,132` evaluation candidates (`0` gradient-training overlap) and verified on `58` separate `dev_val` datasets (`345` splits: `0.60288` $\to$ `0.57414` loss; `Ultra 0.57053 < High 0.57397 < Medium 0.57791`), with prior `val_sub` (`6,758` splits) monitoring during reconciliation explicitly disclosed. | [`disjoint_reconcile.py`](../research/levels/disjoint_reconcile.py), `target/s3-multilevel-training-20261023/s3_training_convergence_receipt.json`, `target/s3-multilevel-training-20261023/s3_training_disjoint1000_convergence_receipt.json`, `target/s3-multilevel-training-20261023/strictly_disjoint_1000_transfer_and_elo_receipt.json` |
| **3. True Downstream Auditor Identities (`6` Families)** | 1. `catboost`: `catboost.CatBoostRegressor` / `CatBoostClassifier` (`iterations=60, depth=5, lr=0.10`, `1,000/1,000` datasets)<br>2. `kumo_tabpfn` *(raw ledger key preserved for immutability $\to$ **Local Gated Attention Net**)*: `_KumoTabularWrapper` / `_TabularKumoAttentionNet` (`hidden=48`, `18` `AdamW` steps trained from scratch per table; `962/1,000` datasets, `38` low-signal skips; **`official_kumo_or_tabpfn_evaluated: false` — NOT official pretrained `KumoTabular` or `TabPFN`**)<br>3. `lightgbm`: `lightgbm.LGBMRegressor` / `LGBMClassifier` (`n_estimators=55, max_depth=5`, `989/1,000` datasets)<br>4. `xgboost`: `xgboost.XGBRegressor` / `XGBClassifier` (`n_estimators=55, max_depth=4`, `989/1,000` datasets)<br>5. `mlp`: `sklearn.neural_network.MLPRegressor` / `MLPClassifier` (`hidden=(48,24)`, `965/1,000` datasets)<br>6. `linear`: `Ridge(alpha=1.0)` / `LogisticRegression(C=1.0)`, `969/1,000` datasets.<br>*(Coverage note: core auditors `catboost`, `lightgbm`, `linear` were evaluated and stored across all `18` method $\times$ tier cells on all `1,000` datasets [`8,874` `elo_cells`]; non-core auditors `xgboost`, `mlp`, `kumo_tabpfn` were evaluated and stored at `Ultra` across all `1,000` datasets for `DOPE-MLC-v2` [and are `UNKNOWN` in the `v2` ledger at `Medium`/`High` and for baselines]; unlimited expansion `1x/2x/5x N_train` was evaluated on `ds_rank < 150` / `150` datasets [`850` remaining datasets: `UNKNOWN`].)* | [`world_class_eval.py`](../research/levels/world_class_eval.py), `target/s3-multilevel-training-20261023/world_class_transfer_and_elo_receipt.json` |
| **4. Per-Method / Tier Metrics & 200x Bootstrap 95% CIs** | **`DOPE-MLC-v2` Ranks `#1` in every cohort**:<br>- **Held-Out (`242` DS)**: Overall Elo **`1671.1`** `[1655.2, 1688.0]` (`73.06%` win share); Cross-Size Elo `1626.7` (`Med`) $\to$ `1690.5` (`High`) $\to$ `1703.7` (`Ultra`); `Ultra` $R = \mathbf{0.8493}$ `[0.8226, 0.8732]` (`Med = 0.9275`), $D = \mathbf{0.8034}$ `[0.7885, 0.8180]` (**passes `D80`**).<br>- **In-Corpus Fold (`758` DS)**: Overall Elo **`1645.4`** `[1637.8, 1653.6]` (`69.81%` win share); Cross-Size Elo `1587.6` $\to$ `1676.4` $\to$ `1682.3`; `Ultra` $R = \mathbf{0.9129}$ `[0.9037, 0.9222]` (`Med = 0.9716`), $D = \mathbf{0.7851}$ `[0.7782, 0.7924]`.<br>- **Pooled (`1,000` DS)**: Overall Elo **`1651.3`** `[1643.4, 1659.3]` (`70.58%` win share); Cross-Size Elo `1596.8` $\to$ `1679.6` $\to$ `1687.2` *(per-auditor)* / `1610.6` $\to$ `1726.6` $\to$ `1742.0` *(dataset-mean)*; `Ultra` $R = \mathbf{0.8975}$ `[0.8883, 0.9094]` (`Med = 0.9658`), $D = \mathbf{0.7896}$ `[0.7831, 0.7959]`.<br>- **Disjoint-Trained (`1,000` DS)**: Overall Elo **`1650.7`** `[1643.4, 1657.5]` (`70.51%` win share); Cross-Size Elo `1596.1` (`Med`) $\to$ `1675.5` (`High`) $\to$ `1690.1` (`Ultra`); `Ultra` $R = \mathbf{0.8998}$ `[0.8913, 0.9094]` (`Med = 0.9671`), $D = \mathbf{0.7904}$ `[0.7834, 0.7966]`. | [`model-sizes-and-elo.md`](model-sizes-and-elo.md), [`dope1-content.tex`](../paper/dope1-content.tex), [`dope1-supplement.tex`](../paper/dope1-supplement.tex) |
| **5. Nominal Budget Caps vs. Exact-Replay Wire Bytes (`v2` Codec) & Legacy Lossy Wire Bytes (`v1` Codec)** | Historical `world_class_eval.py` / `strata_audit.py` copied nominal tier caps (`4,096 B / 16,384 B / 65,536 B`) as `artifact_bytes` for all methods and evaluated generators in-memory in `FitOnlyTableTransform`-normalized space (`X_tr, y_tr` $\to$ pre-scaled `X_te, y_te`). The v2/v3/v4 byte & replay audits (`per_dataset_1000_measured_egress_bytes_v2.jsonl.gz` `SHA-256 2600ed08f9a42090b4a1dc408e91f1e14b5086eee235e15470fc03a3b605a478`, `wire_replay_roundtrip_and_parity_audit_v3.json` `SHA-256 5a583c4b077eafe06a319147ca07285844a0da285517a64d1905f46135e2a6f3`, `exact_wire_replay_1000ds_qualification_v4.json`, `per_dataset_1000_exact_wire_replay_v4.jsonl.gz` `SHA-256 4440ade9b760e14bda5ae31d6b5ab9406a3f4eb4b7abccc0bba0bc7880bb634d`) bind [`serialize_dope_latent_wire_bytes`](../research/levels/strata_audit.py) / [`deserialize_dope_latent_wire_bytes`](../research/levels/strata_audit.py) (`version=1` legacy lossy fp16 vs `version=2` exact replay) and [`estimate_scaler_sidecar_bytes`](../research/levels/strata_audit.py):<br>- **`DOPE-MLC-v2` Exact Wire Codec (`version=2`, `dope.latent_wire.v3_exact`, `+22 B` raw over legacy `version=1`)** *(24 B `<BBBBIIHHHHHH` struct header + 28 B `<7i` decimal fixed-point for 7 rounded target scalars + 16 B `<4f` float32 for 4 router gates = 68 B fixed prefix + all fp16/uint8/packbits tensors + exact 2B `<h`/4B `<i` decimal fixed-point oblivious-tree thresholds consumed by `generate_synthetic` in normalized space; pre-deployed dataset-independent decoder weights `cov_net`/`target_net` in `292,707 B` checkpoint are amortized once; `code_fp16` `8,192 / 16,384 / 20,480 B` is a preview vector not used by `generate_synthetic`; `FitOnlyTableTransform` + 4 B RNG seed sidecar adds **`545 B` median / `876.3 B` mean / `2,781 B` max** across the `1,000` datasets [`0/1,000` datasets have categorical columns requiring hashed token IDs]):*<br>  - `Medium` (cap `4,096 B`): `v2` exact latent raw median **`1,502 B`** (`v1` lossy: `1,480 B`; `v2` `p99 = 5,516 B`, `max = 6,184 B`, `86.4% <= 4,096 B` raw); `v2` latent zlib median **`1,238 B`** (`v1`: `1,227 B`; **`94.4% <= 4,096 B`**); `v2` full raw-domain median w/ scaler sidecar **`2,031 B`** (`v1`: `2,009 B`; `77.5% <= 4,096 B` raw, `85.2%` est. zlib).<br>  - `High` (cap `16,384 B`): `v2` exact latent raw median **`2,836 B`** (`v1`: `2,814 B`; `p99 = 11,222 B`, `max = 12,650 B`, **`100.0% <= 16,384 B`**); `v2` latent zlib median **`2,351 B`** (`v1`: `2,324 B`; **`100.0% <= 16,384 B`**); `v2` full raw-domain median w/ scaler sidecar **`3,376 B`** (`v1`: `3,354 B`; **`100.0% <= 16,384 B`** raw & zlib).<br>  - `Ultra` (cap `65,536 B`): `v2` exact latent raw median **`5,872 B`** (`v1`: `5,850 B`; `p99 = 25,051 B`, `max = 27,202 B`, **`100.0% <= 65,536 B`**); `v2` latent zlib median **`4,616 B`** (`v1`: `4,587 B`; **`100.0% <= 65,536 B`**); `v2` full raw-domain median w/ scaler sidecar **`6,398 B`** (`v1`: `6,376 B`; **`100.0% <= 65,536 B`** raw & zlib).<br>  - **Full 1,000-Dataset (`6,000`-Cell) Exact Replay Qualification (`version=2`) vs. Legacy Lossy (`version=1`)**: Across all `1,000` datasets $\times$ `3` tiers $\times$ `2` checkpoints (`6,000/6,000` cells), `version=2` achieves **`100.0%` (`6,000/6,000`) decode success, bit-exact `X_syn` (`max_abs_diff = 0.0`), AND bit-exact `y_syn` (`max_abs_diff = 0.0`)** using only the wire payload + frozen `292,707 B` decoder (`dope_mlc_v2_wire_1000ds_qualified = true`), whereas legacy `version=1` had `y_syn` drift on `5,107/6,000` cells (`legacy_v1_wire_1000ds_qualified = false`). On the 24-dataset (`144`-cell) CatBoost fixture, `version=2` achieves **`median |ΔR| = 0.0000` and `max |ΔR| = 0.0000`** across all `144/144` cells (vs `version=1` `median |ΔR| = 0.0002 / 0.0041 / 0.0168` on `s3_best`, `0.0000 / 0.0043 / 0.0063` on `disjoint1000`, and outlier `max |ΔR| = 0.6283`).<br>- **`rows` / `kernel` / `copula` / `arf` / `joint_diffusion_v1`** *(post-hoc fp16 representation estimates paired with historical float64 in-memory scores; `baseline_wire_1000ds_qualified = false`, `full_cross_method_wire_1000ds_qualified = false`; 24-DS CatBoost fp16 vs fp64 median $|\Delta R| = 0.0000, 0.0033, 0.0027, 0.0009, 0.0035$)*:<br>  - `rows`: latent raw fp16 median **`1,024 B` / `4,516 B` / `8,464 B`**; zlib **`543 B / 1,794 B / 3,419 B`**; ASCII formula budget `3,998 B / 16,276 B / 30,094 B`.<br>  - `kernel` *(tier-invariant; assumes shared PRNG seed for $p \times 6$ `W_rnd` or `+12*p B`)*: latent raw fp16 median **`202 B`** (`p99 = 930 B`), zlib **`210 B`**.<br>  - `copula` *(tier-invariant; `mean_xy` + upper-triangle `cov_upper` + `mean_y` in normalized space, no quantile knots)*: latent raw fp16 median **`226 B`** (`p99 = 4,438 B`, `90.7% <= 4,096 B` raw, `96.1%` zlib), zlib **`232 B`**.<br>  - `arf` *(tier-invariant; requires full $n_{\text{fit}} \times (p+1)$ fit table partitioned by leaf)*: latent raw fp16 median **`10,426 B`** (`p99 = 104,944 B`; only `25.0% <= 4 KiB`, `61.9% <= 16 KiB`, `92.8% <= 64 KiB`), zlib **`6,369 B`**.<br>  - `joint_diffusion_v1` *(requires full $n_{\text{fit}} \times (p+1)$ table egress; varies only by `noise_t = 0.48 / 0.38 / 0.32`)*: latent raw fp16 median **`9,616 B`** (`p99 = 104,016 B`; only `26.7% <= 4 KiB`, `62.9% <= 16 KiB`, `92.8% <= 64 KiB`), zlib **`5,733 B`**. | `target/s3-multilevel-training-20261023/exact_wire_replay_1000ds_qualification_v4.json`, `target/s3-multilevel-training-20261023/per_dataset_1000_exact_wire_replay_v4.jsonl.gz`, `target/s3-multilevel-training-20261023/wire_replay_roundtrip_and_parity_audit_v3.json`, `target/s3-multilevel-training-20261023/measured_egress_bytes_and_provenance_audit_v2.json`, `target/s3-multilevel-training-20261023/per_dataset_1000_measured_egress_bytes_v2.jsonl.gz`, `target/s3-multilevel-training-20261023/figures/artifact_size_pareto_and_elo.png` |
| **6. Known Compute Costs & Worker Counts** | - **Stage 0 Validation (`19,190` splits)**: `79.6 s` wall clock (`8` CPU threads on `xbabe2`).<br>- **Initial 3-Level GPU Training (`19,079` splits, `dope_mlc_v2_s3_best.pt`)**: `8` epochs (`889.06 s` wall clock on CUDA; checkpoint SHA-256 `1d5db98ffa438a0be1c61dabf235972cd400d1010f023705bea983f96e660a98`).<br>- **Initial 1,000-Dataset Evaluation (`task-797`)**: `40` parallel CPU workers on `xbabe1`, `361.22 s` wall clock (`4.01` CPU core-hours, `65,874` downstream auditor fits; ledger SHA-256 `c268f738570fc10446791591cb26ca2f8a79fba828fd3f52594e4d617879d920`; per-method isolated fit/sample wall-clock breakdown: `UNKNOWN`).<br>- **Strictly Disjoint 1,000-Dataset Training & Re-Scoring (`task-942`)**: `615.02 s` GPU training (`8` epochs on `11,976` splits / `2,001` datasets, checkpoint SHA-256 `6fcc191a244ff409b9d44bbb6a2bcb1d8e3a7db326dcdfbb56f75a64b5f87da3`) + `481.75 s` evaluation across `28` CPU workers on `xbabe3` (ledger SHA-256 `5d62cc31d65c2e695de451ba906226e68bf613108f5dbcae41dc02bd7f37d3ab`).<br>- **Per-Dataset Egress Byte Serialization Audit (`task-1560`)**: `638.94 s` across `32` CPU workers (`1,000` datasets $\times$ `2` checkpoints $\times$ `6` methods $\times$ `3` tiers).<br>- **Bounded 1,000-Dataset Exact Wire Serialization & Isolated-Receiver Replay Pass (`task-1953`, `v4`)**: `91.09 s` across `16` single-threaded CPU workers (`6,000` cells + `144`-cell CatBoost fixture; ledger SHA-256 `4440ade9b760e14bda5ae31d6b5ab9406a3f4eb4b7abccc0bba0bc7880bb634d`). | `target/s3-multilevel-training-20261023/coverage_cost_and_separation_receipt.json`, `target/s3-multilevel-training-20261023/measured_egress_bytes_and_provenance_audit_v2.json`, `target/s3-multilevel-training-20261023/exact_wire_replay_1000ds_qualification_v4.json` |
| **7. Published Doc & Plot Locators** | - Docs: [`README.md`](../README.md), [`docs/multilevel-landmark-dl-study.md`](multilevel-landmark-dl-study.md), [`docs/model-sizes-and-elo.md`](model-sizes-and-elo.md), [`paper/dope1-content.tex`](../paper/dope1-content.tex), [`paper/dope1-supplement.tex`](../paper/dope1-supplement.tex)<br>- Plots (`PNG` + `PDF` in `target/s3-multilevel-training-20261023/figures/`): `stage0_feature_importance`, `s3_convergence_curves`, `world_class_elo_and_transfer`, `artifact_size_pareto_and_elo`, `diverse_1000_datasets_landscape`. | [`render_multilevel_evidence.py`](../research/levels/render_multilevel_evidence.py) |
| **8. Remaining Scientific Limits & Explicit `UNKNOWN` Matrix (`v6` Canonical Index)** | 1. **No Untouched Sealed Confirmatory Claim**: All `v2` cohorts (`242` held-out, `758` in-corpus fold, `1,000` pooled, `1,000` disjoint-trained) and `v3` cohorts (`1,069` datasets / `4,276` cells) are **selection-exposed development/reconciliation validation evidence** (`untouched_sealed_confirmatory_test = false`); sealed confirmatory release certification remains governed by [`docs/release.md`](release.md).<br>2. **Official `KumoTabular` / `TabPFN` / `TabICL` / `Mitra` Not Evaluated (`UNKNOWN`)**: The `"kumo_tabpfn"` / `"local_gated_attention"` keys are local PyTorch gated attention networks (`_TabularKumoAttentionNet` / `_LocalGatedAttentionNet`) trained from scratch per table (`official_kumo_or_tabpfn_evaluated: false`).<br>3. **Unmeasured Subset Cells Marked `UNKNOWN`**: Non-core auditors (`xgboost`, `mlp`, `kumo_tabpfn`) for `DOPE-MLC-v2` at `Medium`/`High` and for baselines across all tiers in the `v2` ledger; non-core auditors (`xgboost`, `mlp`, `local_gated_attention`) and `isolated_receiver_csc1_raw_domain` for equal-wire comparators (`rows_wire`, `copula_wire`, `dope_mlc_v2_wire`) in `v3`; full-panel (`1,000`/`1,069`-dataset) downstream auditor campaigns for the `<5%`-validation `maxtrain97` checkpoints; and standalone Rust `--seed` statistical sampling campaigns are explicitly marked **`"UNKNOWN"`** in the additive canonical reconciliation receipt `target/s3-multilevel-training-20261023/study_1000ds_and_1069ds_reconciled_canonical_index_v6.json` (`SHA-256: 3cca5ce808081f122f9b5f28bbb4aa012d2684c9caf4352f999565cff9a45e37`, superseding preserved `v5` `6a7a156bffa8dc7a16526301a6496f603047676a16947f7b5609a60e83522405`).<br>4. **Not Formal DP or HIPAA De-Identification**: Bounded `2 / 4 / 16 / 64 KiB` encodings and raw-row discard are empirical structural compression controls, never formal $(\varepsilon, \delta)$-differential privacy or legal HIPAA de-identification ([`docs/privacy.md`](privacy.md)). | [`release.md`](release.md), [`privacy.md`](privacy.md), `target/s3-multilevel-training-20261023/study_1000ds_and_1069ds_reconciled_canonical_index_v6.json` |

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

# 7. Run DOPE-MLC-v3 split/exposure manifest + A0..A3 ablation (initial task-2354 source)
python3 -B -m research.levels.v3_ablation_and_eval

# 8. Run bounded frozen-A3 per-dataset/per-tier/per-auditor ledger & targeted recovery pass
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 nice -n 10 python3 -B -m research.levels.v3_ledger_replay

# 9. Run bounded canonical 6-auditor roster & equal-wire comparator evaluation (<= 840s wall cap per shard)
python3 -B -m research.levels.v3_canonical_roster_eval --run-id canonical_v3_bounded_run_v2 --max-v2-slice 80 --workers 4 --wall-budget-s 840
python3 -B -m research.levels.v3_canonical_roster_eval --shard-id canonical_v3_v2panel_shard_01_of_04 --workers 4 --wall-budget-s 840
python3 -B -m research.levels.v3_canonical_roster_eval --shard-id canonical_v3_v2panel_shard_02_of_04 --workers 4 --wall-budget-s 840
python3 -B -m research.levels.v3_canonical_roster_eval --shard-id canonical_v3_v2panel_shard_03_of_04 --workers 4 --wall-budget-s 840
python3 -B -m research.levels.v3_canonical_roster_eval --shard-id canonical_v3_v2panel_shard_04_of_04 --workers 4 --wall-budget-s 840

# 10. Aggregate all 5 disjoint canonical runs into the unified 1,069-dataset (4,276-cell) receipt
python3 -B -m research.levels.v3_canonical_roster_eval --run-id canonical_v3_full_1069_unified_v1 --aggregate-runs canonical_v3_bounded_run_v2,canonical_v3_v2panel_shard_01_of_04,canonical_v3_v2panel_shard_02_of_04,canonical_v3_v2panel_shard_03_of_04,canonical_v3_v2panel_shard_04_of_04

# 11. Build dope CLI and run fail-closed trained-checkpoint Rust SafeTensors + DML3/CSC1 parity verifier
cargo build --manifest-path engine/Cargo.toml --bin dope
python3 -B -m research.levels.rust_v3_parity
DOPE_REQUIRE_RUST_PARITY=1 cargo test --manifest-path engine/Cargo.toml -p dope-core --test mlc_v3_safetensors
DOPE_REQUIRE_RUST_PARITY=1 python3 -B -m unittest research.levels.tests.test_dope_mlc_v3

# 12. Verify/restore exact task-4221 executed source (SHA-256: 701d9cca...) and run <5% validation reproduction
python3 -B -m research.levels.disjoint_reconcile --verify-executed-source
python3 -B -m research.levels.disjoint_reconcile --restore-executed-source target/s3-multilevel-training-20261023/source_maxtrain_val_lt5pct_at_task4221.py
PYTHONPATH=. python3 -B target/s3-multilevel-training-20261023/source_maxtrain_val_lt5pct_at_task4221.py --workers 16
```

---

## 7. `DOPE-MLC-v3` Building-Block Landmarks, Inter-Landmark Summary Statistics, 4-Tier Complete Envelope & Multi-Task Ledger (`task-2354` & `task-2502`)

### 7.1 Architectural & Wire-Envelope Additions
- **Building-Block Landmark Probes (`16` Global + `7` Per-Column + `8` Heteroscedastic/Tail + `20` Multiclass Simplex Probes)**: Implemented in [`research/levels/building_block_landmarks.py`](../research/levels/building_block_landmarks.py) (`compute_row_summary_features`, `compute_building_block_landmarks`) using first-principles mathematical building blocks (`asinh`, `atan`, `erf`, `cbrt`, `log1p`, invariant row summaries `row_mean/std/l1/l2/min/max/skew/missing_frac`, `tanh`, `expm1`, `clamp`, `step`, `hinge`, `sin/cos`, `fract`, `hypot`, `atan2`, and normalized difference ratio).
- **Inter-Landmark Summary Statistics (`48` Features *Between* Landmarks + `InterLandmarkAttentionBlock`)**: Implemented in [`building_block_landmarks.py`](../research/levels/building_block_landmarks.py) (`compute_inter_landmark_summary_stats`) and [`multiclass_head.py`](../research/levels/multiclass_head.py) (`InterLandmarkAttentionBlock`), capturing functional-family moments, cross-landmark ratios/uplifts/velocities, precision-weighted `gain/SE` $t$-statistics, and 2D column $\times$ landmark spectrum alignment correlations (`vec256 = 140 + 48 + 68`).
- **Active Per-Column Gate Head (`col_gate_head`)**: Fixed the previously inactive `col_gate_head` in [`multilevel_model.py`](../research/levels/multilevel_model.py) (`nn.Linear(latent_dim + 24, 1)`), wiring it into `forward_conditioning` (`loss`), `encode_table`, and `generate_synthetic` (`col_gate_head_mean_grad_norm = 0.000721 > 0`, `col_gate_head_mean_column_std = 0.003894 > 0` in `A3`).
- **Four Complete-Envelope Byte Tiers (`Small` `2,048 B`, `Medium` `4,096 B`, `High` `16,384 B`, `Ultra` `65,536 B`) & Multiclass (`LR-VLOT`)**: Implemented in [`budget_planner.py`](../research/levels/budget_planner.py) (`DML3` envelope + `CSC1` compact scaler sidecar) and [`multiclass_head.py`](../research/levels/multiclass_head.py) (low-rank simplex logit basis + shared-split vector-leaf oblivious trees with minimum-quota class support preservation).

### 7.2 Reconciled `3,859`-Dataset Catalog Split & Prior-Exposure Table (`v3_corpus_exposure_and_split_manifest.json`, `SHA-256: 66d2a63c...`)

| Assigned `v3` Role | Prior `v2` Exposure Classification | Catalog Task Breakdown | Manifest Count | Evaluated in `task-2354` (`ce58d6b8...`) | Evaluated in New Frozen-`A3` Replay (`task-2502`, `8574f2d7...`) | Notes & Exact Skip / Subsample Disclosures |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`train_gradient`** | `v2_train_2001_exposed` (`2,001`) + `unexposed_in_v2_multiclass_train` (`198`) | `1,623` regression, `378` binary, `198` `multi_label` (`multiclass`) | **`2,199`** | **`677`** (`480` reg/bin + `197` mc) | `1` historical skip record (`2f7cb79fa3035f03`) | `task-2354` subsampled `480/2,001` reg/bin `fit-iid-65-v1` cells for 4-epoch CPU turnaround; `2f7cb79fa3035f03` (`1,000` rows, class counts `[2,996,1,1]`) skipped by unstratified 384-row fit subsample (`sampled_fit_classes_1_lt_3_unstratified`). |
| **`dev_val_ablation`** | `v2_dev_val_58_exposed` (`58`) + `unexposed_in_v2_multiclass_dev_val` (`54`) | `41` regression, `17` binary, `54` `multi_label` (`multiclass`) | **`112`** | **`110`** (`40` reg, `17` bin, `53` mc) | **`112` DS (`448` cells)** + `2` skip records | `task-2502` is a **new frozen-checkpoint replay** (not recovered `task-2354` raw measurements); `4f118539d1a4c08c` (`reg`, lacked `fit-iid-65-v1`) and `df23b8acc038703b` (`mc`, rare class `0.0`) were newly evaluated under fallback split/class-seeding rules. |
| **`eval_gradient_disjoint_v2_panel`** | `v2_eval_1000_selection_and_error_analysis_exposed` (**`untouched_sealed_confirmatory_test = false`**) | `628` regression, `372` binary | **`1,000`** | Not re-run in `task-2354` | `OPEN_unmeasured` in `task-2502`; **`1,000/1,000` (`4,000` cells) completed in `canonical_v3_full_1069_unified_v1` (`Section 7.5`)** | Evaluated under `v2` in `strata_1000_audit_receipt_v4_exact_wire_replay.json` and under `v3` in `canonical_v3_full_1069_unified_v1` (`80` DS in `canonical_v3_bounded_run_v2` + `920` DS across 4 disjoint `230`-DS shards). |
| **`eval_gradient_disjoint_multiclass_panel`** | `unexposed_in_v2_multiclass_heldout_eval` (`0` prior `v2` exposure) | `69` `multi_label` (`multiclass`, $3 \le K \le 32$) | **`69`** | **`68`** (preliminary 2-auditor probe, no per-DS JSONL saved) | **`69` DS (`276` cells)** + `1` skip record | `task-2502` newly replays all `68` `task-2354` datasets + newly evaluates `ad55f22e74c051c2` (`19` train rows, skipped in `task-2354` by `arr_tr.shape[0] < 20`) with `min_rows=16`. |
| **Explicit Exclusions** | `132` `v2` low-oracle + `261` unindexed/degenerate + `83` `timeseries` + `3` missing local blobs | `298` regression, `95` binary, `1` `multi_label`, `85` `timeseries` | **`479`** | `0` | `0` | Documented in `v3_corpus_exposure_and_split_manifest.json`. |
| **Total Catalog** | **All 6 active cross-role `train_sha256` overlaps = `0`** *(proves exact file/table SHA disjointness, not upstream benchmark-family disjointness)* | `2,589` reg, `863` bin, `322` `multi_label`, `85` `timeseries` | **`3,859`** | — | **`724` measured cells (`181` DS) + `4` skip records = `728` rows** | Hash-bound scope & audit addendum: `target/s3-multilevel-training-20261023/v3/v3_task2354_task2502_scope_and_audit_addendum.json`. |

### 7.3 Matched `A0 -> A1 -> A2 -> A3` Ablation Progression (`task-2354`, `v3_dev_val_ablation_and_multitask_receipt.json`, `SHA-256: ce58d6b8...`)
- Trained for `4` epochs over the `677`-cell training pool (`480` reg/bin + `197` multiclass) warm-started from `dope_mlc_v2_disjoint1000_best.pt`, evaluated on the `110` `dev_val_ablation` cells (`40` regression, `17` binary, `53` multiclass; frozen `A3` checkpoint archived at `target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.pt`, `SHA-256: a36510b9...`). Note that `A0..A3` are **architectural ablation variants** evaluated post-training (not temporal training epochs), and per-epoch `DOPE-MLC-v3` training/validation loss histories were not recorded in `v3_dev_val_ablation_and_multitask_receipt.json`:

| Ablation Variant | Mean `dev_val` Total Loss (`110` Cells) | `cond_y` Loss | `driver` Loss | By-Tier Total Loss (`Small / Med / High / Ultra`) | By-Task Total Loss (`Reg / Bin / Multiclass`) | `col_gate_head` Grad Norm / Column Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0_v2_baseline_140`** | `0.83084` | `0.81326` | `0.04205` | `0.84642 / 0.84010 / 0.82212 / 0.81471` | `0.60803 / 0.51266 / 1.10105` | `0.000000 / 0.000000` |
| **`A1_plus_16_building_blocks`** | `0.82908` | `0.81201` | `0.03969` | `0.84613 / 0.83950 / 0.82013 / 0.81056` | `0.61014 / 0.50964 / 1.09678` | `0.000000 / 0.000000` |
| **`A2_plus_48_inter_landmark_stats`** | `0.82757` | `0.81048` | `0.03963` | `0.84582 / 0.83902 / 0.81792 / 0.80753` | `0.60769 / 0.50927 / 1.09562` | `0.000000 / 0.000000` |
| **`A3_full_v3_with_col_gate_head`** | **`0.82748`** | **`0.81041`** | **`0.03954`** | **`0.84579 / 0.83900 / 0.81781 / 0.80732`** | **`0.60746 / 0.50920 / 1.09562`** | **`0.000721 / 0.003894`** |

#### 7.3.1 Maximized-Training (`96.74%` Loaded Train, `< 5%` Validation) Disjoint-1,069 `DOPE-MLC-v3` Multi-Task Convergence & Matched-Budget `A0..A3` Ablation (`task-4221`, `v3_maxtrain97_val3pct`)
- Bound to `target/s3-multilevel-training-20261023/v3/v3_maxtrain97_val3pct_convergence_receipt.json` (`SHA-256: 9e21851914fbaf6133fe4c4c87cc9eea2a802154d8b806a5cdf4958abe858b3f`), `target/s3-multilevel-training-20261023/v3/v3_maxtrain97_corpus_exposure_and_split_manifest.json` (`SHA-256: 1c75ff09...`), and checkpoint `target/s3-multilevel-training-20261023/v3/dope_mlc_v3_maxtrain97_best.pt` (`SHA-256: 77ffb94f50060bfe147b6c5bb6d3b91f9e2099e0a5cfa8a1a7e34dbecb8e1dc0`; preserved `dope_mlc_v3_multitask_best.pt` remains the canonical checkpoint for Sections 7.4–9.6):
- **Loaded 1-Episode-Cell-Per-Dataset Pool (`2,304` Cells) vs. Pre-Load Catalog Manifest (`2,311` Datasets) & 6-Split Catalog Reference**:
  - In the pre-load catalog manifest (`v3_maxtrain97_corpus_exposure_and_split_manifest.json`), `2,235 / 2,311` non-evaluation datasets (`96.7114%`: `2,001` reg/bin + `234` multiclass, including `36` former `v3` `dev_val` multiclass datasets flagged `v3_mc_dev_val_54_development_exposed`) are assigned to `train_gradient`, and `76 / 2,311` datasets (`3.2886% < 5%`: `58` reg/bin `dev_val_58` + `18` multiclass `71 <= sha256(canonical_sha256) % 100 < 75`) are assigned to `dev_val_ablation`, with **`0` overlap** against the `1,000` `v2` evaluation datasets and `69` held-out multiclass evaluation datasets (`1,069` evaluation datasets).
  - The `v3` training loader uses **1 episode cell per dataset** (`fit-iid-65-v1` for regression/binary and 1 inner train-blob split for multiclass), yielding **`2,229 / 2,304` loaded training episode cells (`96.7448%`: `1,997` reg/bin + `232` multiclass, `+1,552` unique cells vs. `677` in `task-2354`)** and **`75 / 2,304` loaded `dev_val` episode cells (`3.2552% < 5%`: `40` regression, `17` binary, `18` multiclass across `k=3..10` classes)** (`7` datasets lack an admitted `fit-iid-65-v1` or non-degenerate 3-class inner split). The raw receipt field `val_split_pct_across_all_6_reg_bin_splits_plus_mc = 2.8876%` (`363 / (12,208 + 363)`) is a 6-split catalog reference denominator, not the loaded 1-cell-per-dataset `v3_maxtrain97` episode pool.
- **Training-Round Schedule (`R0..R8`) & `A3_full_v3_with_col_gate_head` Full-Validation Trajectory**:
  - Each variant (`A0..A3`, warm-started from `dope_mlc_v2_disjoint1000_maxtrain97_best.pt`) is trained for **`8` quarter-pool shard rounds (`R1..R8`)**, visiting `4,457` cumulative episode cells across `2,229` unique training cells (**`1.999551×` effective full-population passes**, completing `1.000000×` pass at `R4` and `2.00×` at `R8`, with `280` scheduled optimizer updates at `batch_accum = 16`).
  - Evaluating **`100%` of the `75` loaded `dev_val` episode cells across all 4 tiers (`Small`, `Medium`, `High`, `Ultra`) at every round `R0..R8`** reduces `A3` full `dev_val` loss by **`-1.77%`** from `0.69727 ± 0.03888` (`R0`, 95% CI `[0.62909, 0.76794]`) to **`0.68492 ± 0.03877`** (`R8*`, selected best & final round, 95% CI `[0.60885, 0.76099]`) and `600`-cell training monitor loss from `0.47454 ± 0.01122` (`R0`) to **`0.46132 ± 0.01064`** (`R8*`), with strict monotonic 4-tier hierarchy `Loss(Ultra) < Loss(High) < Loss(Medium) < Loss(Small)` at `R0` and `R4..R8`:

| Training Round (`raw epoch`) | Cumulative Cells Visited | Effective Full-Pop Passes (`2,229` Loaded Cells) | Scheduled Updates (`batch=16`) | Train Monitor (`600` Cells) Mean $\pm$ SE | Full `dev_val` (`75/75` Cells) Mean $\pm$ SE `[95% CI]` | `Small` / `Med` / `High` / `Ultra` Mean Loss | Reg (`40`) / Bin (`17`) / MC (`18`) Mean $\pm$ SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R0` (`epoch=0`) | `0` | `0.000000×` | `0` | `0.47454 ± 0.01122` | **`0.69727 ± 0.03888`** `[0.62909, 0.76794]` | `0.70427 / 0.70162 / 0.69407 / 0.68914` | `0.61072±0.03144` / `0.52099±0.02344` / `1.05611±0.10857` |
| `R1` (`epoch=1`) | `558` | `0.250336×` | `35` | `0.46372 ± 0.01182` | **`0.68797 ± 0.03728`** `[0.61932, 0.76691]` | `0.69614 / 0.69264 / 0.68287 / 0.68022` | `0.60274±0.03265` / `0.51215±0.02634` / `1.04342±0.09998` |
| `R2` (`epoch=2`) | `1,115` | `0.500224×` | `70` | `0.46344 ± 0.01130` | **`0.68909 ± 0.04013`** `[0.61742, 0.75956]` | `0.69600 / 0.69252 / 0.68368 / 0.68415` | `0.60409±0.02954` / `0.51263±0.02510` / `1.04462±0.10873` |
| `R3` (`epoch=3`) | `1,672` | `0.750112×` | `105` | `0.46390 ± 0.01148` | **`0.69066 ± 0.03868`** `[0.61668, 0.76062]` | `0.69583 / 0.69262 / 0.68649 / 0.68770` | `0.60644±0.03563` / `0.51271±0.02268` / `1.04588±0.10235` |
| `R4` (`epoch=4`, `1.00×` Pass) | `2,229` | `1.000000×` | `140` | `0.46225 ± 0.01191` | **`0.68548 ± 0.03923`** `[0.60666, 0.76152]` | `0.69526 / 0.69169 / 0.68020 / 0.67475` | `0.60078±0.03067` / `0.50922±0.02428` / `1.04015±0.10534` |
| `R5` (`epoch=5`) | `2,786` | `1.249888×` | `175` | `0.46198 ± 0.01211` | **`0.68542 ± 0.04251`** `[0.59961, 0.76024]` | `0.69491 / 0.69146 / 0.67993 / 0.67537` | `0.60163±0.03530` / `0.50943±0.02264` / `1.03782±0.09778` |
| `R6` (`epoch=6`) | `3,343` | `1.499776×` | `210` | `0.46155 ± 0.00991` | **`0.68583 ± 0.04117`** `[0.60148, 0.76176]` | `0.69479 / 0.69143 / 0.68024 / 0.67686` | `0.60171±0.03245` / `0.50938±0.02433` / `1.03941±0.09458` |
| `R7` (`epoch=7`) | `3,900` | `1.749664×` | `245` | `0.46140 ± 0.01067` | **`0.68493 ± 0.03435`** `[0.62129, 0.75603]` | `0.69478 / 0.69128 / 0.67956 / 0.67411` | `0.60088±0.03063` / `0.50879±0.02202` / `1.03806±0.10444` |
| **`R8*` (`epoch=8`, Selected Best & Final)** | `4,457` | `1.999551×` | `280` | `0.46132 ± 0.01064` | **`0.68492 ± 0.03877`** `[0.60885, 0.76099]` | `0.69473 / 0.69128 / 0.67958 / 0.67408` | `0.60091±0.03066` / `0.50884±0.02333` / `1.03791±0.10093` |

- **Matched-Budget `A0 -> A1 -> A2 -> A3` Ablation Under `< 5%` Validation (`8` Rounds / `4,457` Visits = `1.999551×` Passes Each)**:

| Ablation Variant (`8` Rounds, `4,457` Visits = `1.999551×` Passes) | Best Round (`raw epoch`) | Best Full `dev_val` (`75/75` Cells) Mean $\pm$ SE `[95% CI]` | Final `R8` Full `dev_val` Mean $\pm$ SE | `R8` `cond_y` / `driver` Loss | `R8` By-Tier Loss (`Small / Med / High / Ultra`) | `R8` By-Task Loss (`Reg / Bin / MC`) | `col_gate_head` Grad Norm / Col Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0_v2_baseline_140`** | `R5` (`5`) | **`0.68700 ± 0.04276`** `[0.59970, 0.76184]` | `0.68709 ± 0.03910` | `0.66769 / 0.04572` | `0.69578 / 0.69233 / 0.68186 / 0.67837` | `0.60132 / 0.51169 / 1.04331` | `0.000000 / 0.000000` |
| **`A1_plus_16_building_blocks`** | `R7` (`7`) | **`0.68531 ± 0.03433`** `[0.62170, 0.75654]` | `0.68547 ± 0.03877` | `0.66684 / 0.04168` | `0.69480 / 0.69146 / 0.68003 / 0.67560` | `0.60170 / 0.50893 / 1.03837` | `0.000000 / 0.000000` |
| **`A2_plus_48_inter_landmark_stats`** | `R8` (`8`) | **`0.68489 ± 0.03877`** `[0.60873, 0.76098]` | `0.68489 ± 0.03877` | `0.66630 / 0.04160` | `0.69474 / 0.69126 / 0.67960 / 0.67395` | `0.60087 / 0.50896 / 1.03773` | `0.000000 / 0.000000` |
| **`A3_full_v3_with_col_gate_head`** | `R8` (`8`) | **`0.68492 ± 0.03877`** `[0.60885, 0.76099]` | `0.68492 ± 0.03877` | `0.66636 / 0.04143` | `0.69473 / 0.69128 / 0.67958 / 0.67408` | `0.60091 / 0.50884 / 1.03791` | `0.001014 / 0.005489` |

### 7.4 Held-Out Multiclass Panel (`68` Initial & `69` Total Datasets) — Complete `DML3` Wire Bytes, Preliminary 2-Auditor vs. 6-Auditor Probes & Isolated-Receiver `CSC1` Raw-Domain Qualification
- Bound to `target/s3-multilevel-training-20261023/v3/v3_per_dataset_multitask_ledger.jsonl.gz` (`SHA-256: 8574f2d71afc2526f5134192c3d76dcd3e5b1a13e023469208529019065ad09a`, `724` measured dataset-tier cells across `181` datasets + `4` historical skip records = `728` rows, `measurement_provenance = "frozen_checkpoint_replay"` — a **new frozen-checkpoint replay**, not recovered `task-2354` raw measurements), `target/s3-multilevel-training-20261023/v3/v3_multitask_ledger_receipt.json` (`SHA-256: 44b22686...`), and `target/s3-multilevel-training-20261023/v3/v3_task2354_task2502_scope_and_audit_addendum.json`:
- **Explicit Auditor Roster & Raw-Receiver Scope Notes**:
  1. The `task-2502` 6-auditor probe (`RF40_d8, LogisticL2_iter200, HistGBR_MC, ExtraTrees32_d8, KNN5_dist, RidgeClassifier_L2` for multiclass; `Ridge_L2, HistGBR_Reg, RF32_d6, ExtraTrees32_d6, MLP32, KNN5_dist` for regression; `LogisticL2, HistGBR_Bin, RF32_d6, ExtraTrees32_d6, MLP32, KNN5_dist` for binary) uses fast scikit-learn estimators and is **not** the original `v2` canonical `CatBoost / LightGBM / XGBoost / MLP / Linear / LocalGatedAttention` roster (`_KumoTabularWrapper` / `_TabularKumoAttentionNet`, a local gated-attention neural network trained from scratch per table, **never** official pretrained KumoTabular or TabPFN).
  2. In `task-2502`, isolated-receiver `CSC1` raw-domain evaluation (`isolated_receiver_csc1_raw_domain`) covers **multiclass cells only** (`69` held-out multiclass datasets $\times$ `4` tiers = `276` cells + `54` `dev_val` multiclass datasets $\times$ `4` tiers = `216` cells = `492` multiclass cells; regression and binary `dev_val` cells were evaluated in `sender_scaled_test_domain` only). Furthermore, `CSC1` (`float16` `location`, `scale`, `power_lambda`, `winsor_low`, `winsor_high`) is a compact approximate wire inverse-scaler (`max_abs_diff = 1.8005` on test features at `High`/`Ultra`), **not** bit-exact `float64` `FitOnlyTableTransform` scaler parity.

| Tier (`Cap`) | Complete `DML3` Wire Bytes (`68` DS Mean / `CSC1` Sidecar / Max / `% <= Cap` on `69` DS) | `task-2354` Preliminary 2-Auditor (`RF40+LogReg`) Mean $R$ (`Clipped / Unclipped / Floor=0.03 Count / Clip>1.25 Count` on `68` DS) | `sklearn` 6-Auditor Probe (`Sender-Scaled Test`) Mean $R$ (`68 DS Clipped / Unclipped` \| `All 69 DS Clipped`) | `Isolated Receiver CSC1 Raw-Domain Inverse` 6-Auditor Mean $R$ (`68 DS Clipped / Unclipped` \| `Max CSC1 Test Diff`) | Mean Costs (`Fit+Prep / Encode / Ser+Deser / Gen` ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Small` (`2,048 B`)** | **`1,300.8 B`** (`944.3 B` sidecar + `346.5 B` payload, max `2,042 B`, **`100.0%`**) | `0.7966 / 0.8078` (`0` floor, `2` clip) | `0.7940 / 0.7940` \| `0.8007` | **`0.7838 / 0.7875`** \| `8.1437` | `10.4 / 23.9 / 1.59 / 9.9` ms |
| **`Medium` (`4,096 B`)** | **`2,404.2 B`** (`944.3 B` sidecar + `1,449.9 B` payload, max `4,067 B`, **`100.0%`**) | `0.8393 / 0.8393` (`0` floor, `0` clip) | `0.8370 / 0.8376` \| `0.8429` | **`0.8355 / 0.8355`** \| `2.6884` | `10.4 / 32.5 / 1.35 / 10.0` ms |
| **`High` (`16,384 B`)** | **`6,293.9 B`** (`944.3 B` sidecar + `5,339.6 B` payload, max `14,909 B`, **`100.0%`**) | `0.9120 / 0.9139` (`0` floor, `1` clip) | `0.8982 / 0.8982` \| `0.9033` | **`0.8888 / 0.8903`** \| `1.8005` | `10.4 / 90.5 / 1.92 / 11.1` ms |
| **`Ultra` (`65,536 B`)** | **`13,146.8 B`** (`944.3 B` sidecar + `12,192.5 B` payload, max `39,629 B`, **`100.0%`**) | `0.9202 / 0.9422` (`0` floor, `2` clip) | **`0.9156 / 0.9185`** \| **`0.9205`** | **`0.9071 / 0.9148`** \| `1.8005` | `10.4 / 154.8 / 1.67 / 12.0` ms |

- **Per-Auditor Held-Out Multiclass Breakdown at `Ultra` (`68` Initial Datasets, Clipped / Unclipped $R$ in `Sender-Scaled` \| `Receiver CSC1 Raw-Domain`)**:
  - `RandomForest40_d8`: `0.9050 / 0.9114` \| `0.9060 / 0.9161` (`0` floor)
  - `LogisticL2_iter200`: `0.9324 / 0.9780` \| `0.9011 / 0.9267` (`0` floor)
  - `HistGBR_MC`: `0.8730 / 0.8797` \| `0.8639 / 0.8682` (`2` floor)
  - `ExtraTrees32_d8`: `0.8925 / 0.8940` \| `0.8854 / 0.8925` (`0` floor)
  - `KNN5_dist`: `0.9143 / 0.9905` \| `0.9272 / 0.9675` (`1` floor)
  - `RidgeClassifier_L2`: `0.9007 / 0.9034` \| `0.9191 / 0.9284` (`0` floor)
- **`dev_val_ablation` (`110` Initial + `2` Newly Evaluated Previously-Skipped = `112` Datasets) 6-Auditor Ledger Summary at `Ultra` (`Sender-Scaled` for Reg/Bin, Both Domains for MC)**:
  - `multiclass` (`53` initial / `54` all, `100.0% <= cap`): mean wire `11,964.9 B`, `2/54` floor applied, all-dataset 6-auditor `clipped_R = 0.8998` (`unclipped_R = 0.9167`), non-floor (`52/54`) `clipped_R = 0.9021`.
  - `regression` (`40` initial / `41` all, `100.0% <= cap`): mean wire `3,632.5 B`; because `world_class_eval.py` selected high-signal tables for the `1,000`-dataset panel (`cb_sig >= 0.025`), `30/41` residual `dev_val` regression cells have near-zero oracle signal (`real_score - chance < 0.05` floor; mean `real_score = 0.1000`, `syn_score = 0.0747`), yielding all-dataset `clipped_R = 0.2296` (`unclipped_R = 0.2296`) and non-floor (`11/41`) `clipped_R = 0.7241` (`0.7669` at `High`).
  - `classification` (`17/17`, `100.0% <= cap`): mean wire `5,301.9 B`; `13/17` residual `dev_val` binary cells have `real_score - 0.5 < 0.03` floor (mean `real_score = 0.5132`, `syn_score = 0.5119`), yielding all-dataset `clipped_R = 0.5271` (`unclipped_R = 0.1782`) and non-floor (`4/17`) `clipped_R = 0.7250`.
- **Visibly Open vs. Closed Obligations After `task-2502` & Unified `1,069`-Dataset Canonical 6-Auditor Evaluation (`Section 7.5`)**:
  1. `v2_canonical_gbdt_mlp_attention_6aud_roster_on_v3`: **EXECUTED UNDER BOUNDED CPU PROTOCOL (`Section 7.5`)** — `canonical_v3_full_1069_unified_v1` evaluates all `69/69` held-out multiclass datasets and all `1,000/1,000` `eval_gradient_disjoint_v2_panel` datasets (`1,069` unique datasets, `4,276` dataset-tier cells) across all 6 canonical model families (`CatBoost`, `LightGBM`, `XGBoost`, `MLP`, `Linear/Logistic`, and `LocalGatedAttentionAuditor` [`_LocalGatedAttentionNet`, local gated attention trained from scratch per table, **never** official Kumo/TabPFN]) under `CANONICAL_V3_AUDITOR_PROTOCOL_SPEC`.
  2. `v3_full_1000_panel_transfer_elo_and_driver_fidelity`: **CLOSED (`1,000/1,000` datasets completed in `canonical_v3_full_1069_unified_v1`)** — `80/1,000` datasets were evaluated in `canonical_v3_bounded_run_v2` and the remaining `920/1,000` datasets (`v3_remaining_920_v2panel_inventory.json`) were evaluated across 4 disjoint `230`-dataset shards (`canonical_v3_v2panel_shard_01_of_04..04_of_04`) with `0` duplicate `(dataset_id, level)` cells and `0` failed cells.
  3. `source_family_disjointness_audit`: **OPEN (unverified beyond exact SHA-256)** — `0` cross-role `train_sha256` and `canonical_sha256` overlaps are verified, which does not by itself prove upstream benchmark-family disjointness.
  4. `untouched_sealed_confirmatory_generalization`: **OPEN** — all `v2` and `v3` panels are selection/development/reconciliation validation cohorts (`untouched_sealed_confirmatory_test = false`).

---

### 7.5 Unified `1,069`-Dataset (`1,000` `v2`-Panel + `69` Multiclass) Canonical 6-Auditor Evaluation, Authentic-Label Integrity & Equal-Wire Comparator Tournament (`canonical_v3_full_1069_unified_v1`)

- **Immutable Evidence Custody & Run Provenance**:
  - **Unified Canonical Receipt (`canonical_v3_full_1069_unified_v1`, `task-3148`)**: Stored in `target/s3-multilevel-training-20261023/v3/runs/canonical_v3_full_1069_unified_v1/` (`ledger.jsonl` `SHA-256: 38a4094067b7118575af181825631b0c1404cadb3d6020e3bfb123d93aa3c7b0`, `ledger.jsonl.gz` `SHA-256: ce017e21abed60095999ab5079efe9032cb11b84d2312a9b6370b4776a56875e`, `receipt.json`, `total_source_wall_seconds = 1,877.0s`, `completed_dataset_tier_cells = 4,276/4,276` across `69/69` multiclass + `1,000/1,000` `v2`-panel datasets, `duplicate_logical_dataset_level_cells = 0`, `failed_or_skipped_dataset_records = 0`, `unexecuted_or_interrupted_records = 0`), formed by the disjoint union of 5 bounded runs:
    1. `canonical_v3_bounded_run_v2` (`task-2954`, `281.3s <= 840s`, `69` multiclass + `80` `v2`-panel = `149` DS, `596/596` cells, `ledger.jsonl` `SHA-256: e60e1954090b2ea94596272811f810bd0be0ac4b7d9df8806ef1a555cc89b5f1`, `ledger.jsonl.gz` `SHA-256: f9cca53e8b49a98b8722a3b6c87144346358198aa5ca72b7683d3b9107c513a5`), plus `source_binding_and_nonbehavioral_diff_addendum.json` (`SHA-256: 633cd3aa34b5dfc2956a20a5b3688cc2e59d5d0f38be359ffcbff5f5f58df071`) archiving the exact executed source files (`source_v3_canonical_roster_eval_at_run.py` `SHA-256: b55b8e5b0b082ae36ebb5a8960653eff306e1b334c3cbcf844a67ff2fb47e6ac` and `source_v3_canonical_runner_at_run.py` `SHA-256: fdd792deb5bcddc7ef64c3ab1ebae6bd557d7ea4a0141f4987581102ade36e7c`).
    2. `canonical_v3_v2panel_shard_01_of_04` (`task-3132`, `359.16s <= 840s`, `230` `v2`-panel DS, `920/920` cells, `ledger.jsonl` `SHA-256: d9b7dbeb2f29db45c1344d1836c59c3e00c3eeafa384767c149c4bb85290ef4b`, `ledger.jsonl.gz` `SHA-256: b3c7b446ebf1b3633f9f41c11e2fcf327682fb4b54c392bcae23c1ef4b235e3f`).
    3. `canonical_v3_v2panel_shard_02_of_04` (`task-3136`, `462.99s <= 840s`, `230` `v2`-panel DS, `920/920` cells, `ledger.jsonl` `SHA-256: c916f4c837adf55b2c6aada3180d983d63efe4dcee093461063876f482414eac`, `ledger.jsonl.gz` `SHA-256: 82ce7af8afe2400053c855be4d7d7c59914cd65a2bf02310a2bb36ff1fae1901`).
    4. `canonical_v3_v2panel_shard_03_of_04` (`task-3140`, `400.61s <= 840s`, `230` `v2`-panel DS, `920/920` cells, `ledger.jsonl` `SHA-256: 8309979bdefb5d6d6d6597434937f3ad89435effa7adad50814e9e939764b5ae`, `ledger.jsonl.gz` `SHA-256: caba147b3b93b5bd0f20896cc73a8707792ed1bb95ec6518366e0343c513d2af`).
    5. `canonical_v3_v2panel_shard_04_of_04` (`task-3144`, `372.94s <= 840s`, `230` `v2`-panel DS, `920/920` cells, `ledger.jsonl` `SHA-256: 4122a3efd46f802cf1279aba80e2678436df62ce72ddc04d0a5d87dc16fa3400`, `ledger.jsonl.gz` `SHA-256: da6205c55cf6eb00631ed83b2c14dad9e035eb1b786898d064c78d19f7a054fa`).
    - Disjoint 920-ID shard manifest: `target/s3-multilevel-training-20261023/v3/v3_remaining_920_v2panel_inventory.json` (`SHA-256: be6fce8ce4bc0f486fdfaec8509188eb8b7e479f1ec92e57930525010aff936d`, `577` regression + `343` binary = `920` remaining IDs, `0` ID or `train_sha256` overlap with `canonical_v3_bounded_run_v2` or any `train_gradient`/`dev_val`/`multiclass` cohort).
  - **Preserved Earlier Runs & Snapshots (Never Overwritten)**:
    - `target/s3-multilevel-training-20261023/v3/runs/task2808_smoke_20261024T163116Z/`: Preserved 25s smoke test (`v3_canonical_6aud_and_v2panel_slice_ledger.jsonl.gz` `SHA-256: 6ccd1412...`, decompressed `recovered_from_gzip_v3_canonical_6aud_and_v2panel_slice_ledger.jsonl` `SHA-256: 8299b25b...`, `77` lines; `exact_source_bytes_captured_at_163043Z = false`, `exact_source_sha256 = null`).
    - `target/s3-multilevel-training-20261023/v3/runs/monitor_observed_20261024T163552Z/`: Preserved monitor-captured source snapshot (`20261024T163552Z-observed-v3_canonical_roster_eval.py` `SHA-256: c59b1d45...`, `20261024T163552Z-observed-v3_canonical_runner.py` `SHA-256: 4ecce5b6...`).
    - `target/s3-multilevel-training-20261023/v3/runs/task2842_canceled_partial_20261024T163834Z/`: Preserved canceled partial run (`v3_canonical_6aud_and_v2panel_slice_ledger.jsonl` `SHA-256: e1dcecb8...`, `84` lines) prior to label-integrity fixes.
    - `target/s3-multilevel-training-20261023/v3/runs/canonical_v3_bounded_run_v1/`: Preserved initial full run (`ledger.jsonl.gz` `SHA-256: 23c0f21e...`, `596` cells) prior to removing the redundant `y_tr_ro * (k_cls - 1)` multiclass index multiplier on already-indexed `FitOnlyTableTransform` targets.
- **Authentic-Label Integrity, Protocol & Comparator Scope Disclosure (`CANONICAL_V3_AUDITOR_PROTOCOL_SPEC`)**:
  - `fit_and_score_canonical_auditor` enforces `y_tr_ro.flags.writeable = False` and `y_te_ro.flags.writeable = False` with pre/post SHA-256 verification (`label_sha256_verified = True` across all `25,656` auditor evaluations per domain across the `4,276` dataset-tier cells).
  - Degenerate constant regression (`ptp < 1e-12`) or single-class binary/multiclass (`len(present) < 2`) targets use authentic-label constant predictors; partial multiclass support (`2 <= len(present) < k_cls`, occurring in `rows_wire` auditor cells where row subsampling at tight caps missed a rare class; `0` cells in `DOPE-MLC-v3`) uses an invertible contiguous bijection (`np.searchsorted(present, y_tr_c)` and inverse index mapping `present[pred_contig]`), never perturbing, flipping, or inventing labels.
  - All auditor constructors are driven by `CANONICAL_V3_AUDITOR_PROTOCOL_SPEC` (`24-25` tree estimators, `32x16` `MLP`, `12`-step `hidden_dim=32` `LocalGatedAttentionAuditor`, max `340` train / `160` test rows / `36` features), which differs from the historical `v2` `world_class_eval.py` wrapper (`is_identical_to_historical_v2_world_class_eval_wrapper = false`). Historical `v2` receipt scores are not compared directly against this bounded protocol; all comparator numbers below are re-evaluated side-by-side on identical splits, auditors, and byte caps inside `canonical_v3_full_1069_unified_v1`.
  - **Explicit Subset Disclosure (`elo_and_comparator_scope_disclosure`)**: `DOPE-MLC-v3` retention and driver-fidelity tables (`Section 7.5.1`) evaluate all `CANONICAL_6_AUDITORS` (`catboost`, `lightgbm`, `xgboost`, `mlp`, `linear`, `local_gated_attention`) in both `sender_scaled_test_domain` and `isolated_receiver_csc1_raw_domain`. Cross-method Elo and paired retention differences (`Section 7.5.2`) are computed on sender-scaled `CORE_COMPARATOR_AUDITORS` (`catboost`, `lightgbm`, `linear`), and `dope_mlc_v2_wire` is evaluated on `regression` and `classification` tasks at `Medium`, `High`, and `Ultra` tiers (`9,000` cells across the `1,000` `v2`-panel datasets) because historical `DOPE-MLC-v2` had no `Small` tier or native `multiclass` head.
  - Host CPU guard (`82%` `/proc/stat` check + `os.nice(10)` + single-threaded BLAS/OpenMP/Torch workers) operates as a dispatch admission gate before queueing new dataset tasks (`cpu_guard_semantics = "dispatch_admission_gate_only_82pct_proc_stat_plus_nice10_single_thread_workers_does_not_throttle_already_active_in_flight_workers"`).

#### 7.5.1 Held-Out Multiclass Panel (`69/69` Datasets, `276` Cells) & Complete Disjoint `v2` Evaluation Panel (`1,000/1,000` Datasets: `628` Reg + `372` Bin, `4,000` Cells; `0` Failed Auditor Cells Across All `4,276` Cells)

| Cohort & Tier (`Cap`) | Completed / Planned DS (`Missing Rate`) | Mean / Max `DML3` Wire Bytes (`% <= Cap`) | `Sender-Scaled` 6-Aud Clipped / Unclipped $R$ (`Zero-Imputed Sensitivity`) | `Sender-Scaled` Driver Fidelity $D$ | `Isolated Receiver CSC1 Raw` (Deployed Wire) Clipped / Unclipped $R$ (`D`) | `Diagnostic Exact fp64 Inverse Control` (Uncharged Sidecar) Clipped $R$ (`Mean \|Delta vs CSC1\|`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Multiclass Small` (`2,048 B`)** | `69 / 69` (`0.0%`) | `1,418.0 B / 2,047 B` (`100.0%`) | **`0.7271 / 0.7098`** (`0.7271`) | `0.6462` | **`0.7018 / 0.6939`** (`0.6484`) | `0.7248` (`0.03205`) |
| **`Multiclass Medium` (`4,096 B`)** | `69 / 69` (`0.0%`) | `2,510.7 B / 3,812 B` (`100.0%`) | **`0.7767 / 0.7712`** (`0.7767`) | `0.6715` | **`0.7789 / 0.7554`** (`0.6734`) | `0.7812` (`0.00510`) |
| **`Multiclass High` (`16,384 B`)** | `69 / 69` (`0.0%`) | `6,435.3 B / 14,048 B` (`100.0%`) | **`0.8402 / 0.8355`** (`0.8402`) | `0.7157` | **`0.8451 / 0.8423`** (`0.7114`) | `0.8470` (`0.00801`) |
| **`Multiclass Ultra` (`65,536 B`)** | `69 / 69` (`0.0%`) | `9,877.5 B / 24,195 B` (`100.0%`) | **`0.8362 / 0.8415`** (`0.8362`) | `0.7158` | **`0.8404 / 0.8340`** (`0.7147`) | `0.8440` (`0.00917`) |
| **`Full v2-Panel Small` (`2,048 B`)** | `1,000 / 1,000` (`0.0%`) | `960.6 B / 1,472 B` (`100.0%`) | **`0.8080 / 0.7119`** (`0.8080`) | `0.7536` | **`0.8055 / 0.6912`** (`0.7531`) | `0.8151` (`0.01780`) |
| **`Full v2-Panel Medium` (`4,096 B`)** | `1,000 / 1,000` (`0.0%`) | `1,550.2 B / 2,459 B` (`100.0%`) | **`0.8286 / 0.7325`** (`0.8286`) | `0.7619` | **`0.8363 / 0.7371`** (`0.7619`) | `0.8363` (`0.00348`) |
| **`Full v2-Panel High` (`16,384 B`)** | `1,000 / 1,000` (`0.0%`) | `3,429.2 B / 7,203 B` (`100.0%`) | **`0.8633 / 0.7502`** (`0.8633`) | `0.7793` | **`0.8706 / 0.7529`** (`0.7791`) | `0.8699` (`0.00566`) |
| **`Full v2-Panel Ultra` (`65,536 B`)** | `1,000 / 1,000` (`0.0%`) | `6,144.6 B / 13,010 B` (`100.0%`) | **`0.8517 / 0.7283`** (`0.8517`) | `0.7764` | **`0.8581 / 0.7244`** (`0.7763`) | `0.8581` (`0.00655`) |
| *(Preserved `80`-DS `v2`-Slice in `run_v2`)* | `80 / 80` (`0.0%`) | `974.4 / 1,616.0 / 3,420.5 / 6,079.5 B` | `0.8297 / 0.8544 / 0.8708 / 0.8723` | `0.7654..0.7797` | `0.8379 / 0.8626 / 0.8768 / 0.8774` | `0.8406 / 0.8640 / 0.8801 / 0.8809` |

- **Per-Auditor Canonical Breakdown Across All 4 Tiers (`Sender-Scaled Clipped R / CSC1 Raw Clipped R / Driver Fidelity D`, `0` Failed Cells & `0` Partial/Degenerate Label Cells Across All Auditors)**:
  - **Held-Out Multiclass (`69/69` DS)**:
    - `catboost`: `Small` `0.6975 / 0.6812 / 0.6300`, `Med` `0.7420 / 0.7393 / 0.6741`, `High` `0.8350 / 0.8339 / 0.7058`, `Ultra` `0.8301 / 0.8372 / 0.7124`
    - `lightgbm`: `Small` `0.7288 / 0.7299 / 0.6364`, `Med` `0.7833 / 0.8000 / 0.6649`, `High` `0.8279 / 0.8305 / 0.7318`, `Ultra` `0.8234 / 0.8423 / 0.7277`
    - `xgboost`: `Small` `0.6923 / 0.6793 / 0.5506`, `Med` `0.7430 / 0.7373 / 0.5996`, `High` `0.8138 / 0.8147 / 0.6463`, `Ultra` `0.8034 / 0.7908 / 0.6125`
    - `mlp`: `Small` `0.7073 / 0.6739 / 0.8544`, `Med` `0.7600 / 0.7553 / 0.8522`, `High` `0.7834 / 0.7973 / 0.8684`, `Ultra` `0.8200 / 0.8332 / 0.8795`
    - `linear`: `Small` `0.7863 / 0.7213 / 0.6438`, `Med` `0.8353 / 0.8468 / 0.6579`, `High` `0.9110 / 0.9115 / 0.6683`, `Ultra` `0.8910 / 0.8909 / 0.6745`
    - `local_gated_attention` (`LocalGatedAttentionAuditor`, **not** official Kumo/TabPFN): `Small` `0.7505 / 0.7251 / 0.5623`, `Med` `0.7963 / 0.7949 / 0.5806`, `High` `0.8698 / 0.8825 / 0.6733`, `Ultra` `0.8493 / 0.8480 / 0.6880`
  - **Complete Disjoint `v2` Evaluation Panel (`1,000/1,000` DS: `628` Regression + `372` Binary)**:
    - `catboost`: `Small` `0.7827 / 0.7841 / 0.7647`, `Med` `0.8072 / 0.8098 / 0.7754`, `High` `0.8654 / 0.8670 / 0.7993`, `Ultra` `0.8699 / 0.8702 / 0.7936`
    - `lightgbm`: `Small` `0.8029 / 0.8016 / 0.7025`, `Med` `0.8225 / 0.8242 / 0.7069`, `High` `0.8791 / 0.8787 / 0.7507`, `Ultra` `0.8783 / 0.8805 / 0.7553`
    - `xgboost`: `Small` `0.7391 / 0.7738 / 0.6730`, `Med` `0.7626 / 0.8000 / 0.6858`, `High` `0.8060 / 0.8513 / 0.6977`, `Ultra` `0.8035 / 0.8425 / 0.6950`
    - `mlp`: `Small` `0.7218 / 0.7031 / 0.8980`, `Med` `0.7649 / 0.7689 / 0.9026`, `High` `0.8028 / 0.8024 / 0.9085`, `Ultra` `0.7730 / 0.7739 / 0.9108`
    - `linear`: `Small` `0.9281 / 0.9103 / 0.7619`, `Med` `0.9280 / 0.9289 / 0.7632`, `High` `0.9213 / 0.9207 / 0.7526`, `Ultra` `0.8993 / 0.8974 / 0.7388`
    - `local_gated_attention` (`LocalGatedAttentionAuditor`, **not** official Kumo/TabPFN): `Small` `0.8736 / 0.8598 / 0.7213`, `Med` `0.8865 / 0.8859 / 0.7373`, `High` `0.9053 / 0.9036 / 0.7670`, `Ultra` `0.8862 / 0.8843 / 0.7648`

#### 7.5.2 Equal-Wire Comparator Tournament & Paired Uncertainty (`200x` Dataset-Cluster Bootstrap on Sender-Scaled `CORE_COMPARATOR_AUDITORS` = `CatBoost`, `LightGBM`, `Linear/Logistic`)

| Cohort & Wire Method | Supported Scope in Tournament | Paired Elo (`200x` Bootstrap `95% CI`, `SE`) | Tournament Win Share (`W / D / L`) | Paired `DOPE-MLC-v3 minus Comparator` Mean $\Delta R$ (`95% CI`) / Median $\Delta R$ | Paired Cells (`Wins / Ties <=0.10 [<=0.01] / Losses / Missing`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Combined `1,069` DS: `rows_wire` (`RWW1`)** | All `1,069` DS $\times$ `4` tiers $\times$ `3` core auditors (`12,828` cells) | **`1597.3`** (`[1589.7, 1606.1]`, `SE=4.24`) | `68.48%` (`22,664 / 2,139 / 9,853`) | `-0.0250` (`[-0.0297, -0.0201]`) / `-0.0255` | `12,828` (`1,919 / 7,481 [1,755] / 3,428 / 0`) |
| **Combined `1,069` DS: `dope_mlc_v2_wire`** | `1,000` reg/bin DS $\times$ `3` tiers (`Med/High/Ultra` only; no `Small` or `multiclass`, `9,000` cells) | **`1538.0`** (`[1531.8, 1543.2]`, `SE=3.05`) | `56.97%` (`14,334 / 2,094 / 10,572`) | `-0.0084` (`[-0.0104, -0.0063]`) / `-0.0093` | `9,000` (`392 / 8,136 [2,843] / 472 / 0`) |
| **Combined `1,069` DS: `dope_mlc_v3_wire` (`DML3`)** | All `1,069` DS $\times$ `4` tiers $\times$ `3` core auditors (`Small/Med/High/Ultra`, `12,828` cells) | **`1457.6`** (`[1449.5, 1464.3]`, `SE=3.73`) | `42.66%` (`13,548 / 2,472 / 18,636`) | Reference (`0.0000`) | — |
| **Combined `1,069` DS: `copula_wire` (`CPW1`)** | All `1,069` DS $\times$ `4` tiers $\times$ `3` core auditors (`12,828` cells) | **`1407.1`** (`[1399.8, 1416.3]`, `SE=4.32`) | `33.43%` (`10,580 / 2,011 / 22,065`) | **`+0.0941`** (`[+0.0897, +0.0997]`) / **`+0.0151`** | `12,828` (`3,980 / 7,441 [1,911] / 1,407 / 0`) |
| **Multiclass `69` DS Only (`828` cells)** | `rows_wire` `1643.0` (`[1615.8, 1693.0]`), `dope_mlc_v3_wire` **`1540.9`** (`[1521.4, 1567.3]`), `copula_wire` `1313.8` (`[1263.9, 1345.2]`) | — | `v3` win share `57.22%` | `v3 - rows`: `-0.0723` (`[-0.0908, -0.0552]`); `v3 - copula`: **`+0.2526`** (`[+0.2309, +0.2709]`) | `v3 vs copula`: `828` (`493 / 283 [127] / 52 / 0`) |
| **Full `v2`-Panel `1,000` DS Only (`12,000` cells)** | `rows_wire` `1596.0` (`[1588.0, 1605.6]`), `dope_mlc_v2_wire` `1537.9` (`[1532.8, 1543.7]`), `dope_mlc_v3_wire` `1454.1` (`[1446.2, 1462.3]`), `copula_wire` `1412.0` (`[1403.6, 1419.0]`) | — | `v3` win share `41.93%` | `v3 - rows`: `-0.0218`; `v3 - v2`: `-0.0084` (`9,000` cells); `v3 - copula`: **`+0.0832`** (`[+0.0782, +0.0888]`) | `v3 vs copula`: `12,000` (`3,487 / 7,158 [1,784] / 1,355 / 0`) |

### 9.6 100% Pure-Rust `SafeTensors` Inference & `DML3`/`CSC1` Receiver Sampling (`dope mlc-v3`)

- **Weight Export & Source Binding:** `export_v3_checkpoint_to_safetensors` exports `target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.pt` (`SHA-256: a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952`) to `target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.safetensors` (`1,781,800` bytes, `56` `F32` tensors, `444,168` parameters, `SHA-256: d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`; recorded in `target/s3-multilevel-training-20261023/v3/dope_mlc_v3_safetensors_export_receipt.json`).
- **Native Rust Execution (`engine/dope-core/src/mlc_v3/` & `engine/dope-cli/src/mlc_v3.rs`):** `MlcV3Model::load_safetensors` verifies all `56` `F32` tensors and `444,168` finite parameters without Python or LibTorch, executes `LandmarkHyperRouter::forward` (`8`-token `InterLandmarkAttentionBlock` + bilinear mixer + `14` level-scaled gates + `128`-d `task_latent`/`dist_latent` + `landmark_pred` + `bb_pred`), `ConditionalHyperTargetNet::forward` (including `compute_column_gates` in both `col_landmarks` and receiver `X_pad` 17-d summary modes), and `CovariateRefinementNet::forward` (`dope mlc-v3 {inspect-weights,inspect-wire,route,sample}`), and decodes/samples CRC32-verified `DML3` (`zlib`/`brotli`/uncompressed) + `CSC1` compact scaler sidecar + `LR-VLOT` multiclass packets alongside legacy `DML2` v1 & v2 backward compatibility and fail-closed tamper/malformed/oversized (`> 262,144 B`) rejection.
- **Fail-Closed Trained-Checkpoint Parity Receipt (`research/levels/rust_v3_parity.py` & `dope_mlc_v3_rust_parity_receipt.json`):** `python3 -B -m research.levels.rust_v3_parity` (plus `DOPE_REQUIRE_RUST_PARITY=1` gates in `engine/dope-core/tests/mlc_v3_safetensors.rs` and `research/levels/tests/test_dope_mlc_v3.py`) fails closed if the built `dope` binary, exported `.safetensors` checkpoint, or reference fixtures are absent. Across all `4` tiers (`Small`, `Medium`, `High`, `Ultra`) on the trained checkpoint, max absolute tensor difference is **`1.3255e-6 < 1e-5`** across router `gates` (`<= 1.47e-7`), `task_latent` (`<= 1.33e-6`), `dist_latent` (`<= 1.31e-6`), `landmark_pred` (`<= 8.84e-7`), `bb_pred` (`<= 8.19e-7`), `col_gates` (`<= 1.67e-7`), `target_net` (`<= 2.70e-7`), and `cov_net` (`<= 1.66e-7`). Across all `12` `(tier, task)` cells (`Small`/`Medium`/`High`/`Ultra` $\times$ `regression`/`classification`/`multiclass`) on compressed Python-generated `DML3` + `CSC1` packets with common fixed intermediate draws (`--draws-json`), max absolute difference across `x_scaled`, `y_scaled`, `x_raw`, and `y_raw` is **`1.1111e-6 < 1e-5`**, accompanied by separate standalone `--seed` statistical sampling checks (finite values, schema `f_000..f_007,target`, `CSC1` raw-domain restoration, and multiclass class-support quotas), `DML2` v1/v2 compatibility, and 8 tamper/oversized refusal checks.
- **Explicit Qualification Boundary (Python Study Benchmarks vs. Rust Sampler):** All `1,000`-dataset `DOPE-MLC-v2` and `1,069`-dataset (`4,276`-cell) `DOPE-MLC-v3` downstream auditor retention, driver-fidelity, and Elo benchmarks reported in Sections 4–7 were measured in the Python reference pipeline (`research/levels/`). While `dope mlc-v3` is qualified for `< 1e-5` trained-checkpoint forward parity and `< 1e-5` deterministic fixed-draw `DML3`/`CSC1` decoding and sampling parity, prior Python study quality claims are not transferred to standalone Rust `--seed` statistical sampling prior to dedicated end-to-end campaign measurement.
