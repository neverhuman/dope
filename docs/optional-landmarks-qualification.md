# Optional and progressive landmarks: qualification record

Measured on 27–28 October 2026 on the host xbabe2: AMD Threadripper PRO 3995WX (64 cores), RTX 3090, CUDA 12.8,
Rust 1.88.0, numpy 1.26.4. All numbers below come from the evidence files in
[`evidence/optional-landmarks/`](evidence/optional-landmarks/) and the post-`#349` (`29ba3db72df79b2a799cbfb13d55485eb46bb2d5`)
qualification runs. The harness that produced them is
`research/levels/tools/optional_landmarks_qa/`. Checkpoint: released multitask
`d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`. This page records measurements only. It
authorizes no default change, and N=300 stays the progressive default.

**Recipe identifier.** The full-report conditioning recipe was renamed from `legacy-full-v1` to `full-report-v1`. Inputs and the 629-D features are unchanged; full-mode effective-conditioning digests change. Evidence below that predates the rename keeps the old label; every old-to-new golden digest is listed in [`recipe-transition.json`](evidence/optional-landmarks/recipe-transition.json).

**Status:**
- **Sections 1, 2, 3, and 4:** Baseline measurements on the released checkpoint (`d0b69603…`) and Section 2a measurements on the landmark-dropout-trained candidate (`ecded022…`, evaluated on the 102 width-valid `dev_val_ablation` tables; not registered) are recorded below.
- **Section 5 (Latency):** Updated on 28 October 2026 after merging PR `#349` (`29ba3db72df79b2a799cbfb13d55485eb46bb2d5`, column-parallel preparation and table descriptors). Per owner decision (`2026-10-28T21:45Z` / `22:05Z`), the `>= 5x` `--landmark-mode none` vs `--landmark-mode full` gate is evaluated on the **resident/session regime** (`40 / 40` real tables `>= 5x`), with **strict-cold per-process CLI numbers preserved alongside as failing-regime context** and resident/session release test bounds (`<= 5 ms` resident `none`-mode overhead beyond read + prepare; `<= 30 ms` resident `none`-mode `p50` on real tables with `n <= 1,024`) scoped strictly to the resident/session regime.

## 1. Full-path byte identity

`dope landmarks real --json` ran over 285 tables. All runs used `work_budget` 2e12; every other option was left at its default.

| Source | Tables | Contents |
|---|---|---|
| Precalc `fit.csv` cells | 160 | 88 classification; 40 regression with more than 64 columns |
| JopeDime `datasets/test` | 10 | |
| Synthetic | 27 | regression, binary and 5-class; n 1k/10k/50k; p 10/64/200; 2% missing values and one categorical column |

Each candidate was compared with main `245d4f18` at one thread
([`identity-sweeps.json`](evidence/optional-landmarks/identity-sweeps.json)):

| candidate | runs | report bytes |
|---|---|---|
| ridge/ξ perf (PR #339, `ededfc15`), RAYON 1/2/8/64 | 882 | identical |
| `signature_requested` refactor (`3a0c9bb4`) | 285 | identical |
| the same plus PR #339, 1 thread | 285 | identical |
| optional-landmarks head `82cea8c4` | 285 | identical |
| column-parallel prepare & descriptors (PR #349, `29ba3db7`), `none` + `full` at 1 and 8 threads (285 real) plus 1/2/8 threads (27 synthetic) | 1,221 | identical (`0 / 1,221` diffs) |

On every Rust change the effective-conditioning digest and `features_629` output were also unchanged on real and synthetic tables. This covers `7a5cffa9` (prepare HPC), `82cea8c4` (single-pass parsing), and `29ba3db7` (PR `#349`: `prepare/scaling_par.rs`, `prepare/descriptors_par.rs`, and zero-clone cell moves in `dope-cli/src/mlc_v3_encode.rs`, where sorting categorical distinct values by deterministic first-occurrence row index resolved the earlier `26 / 285` multithreaded digest drift).

## 2. Retention (`retention-dev112.json`)

**Roster:** the 112 `dev_val_ablation` tables of the pinned role manifest (`66d2a63c…`, read directly; no mixed-role cache is opened), which excludes MLC-v3 training tables.
- Roster hash: sha256 of the sorted IDs, `c3cc1dc2500c3baa7eb99ee339948501cf7d8d83561d0a72ea051581b6e6353a`.
- Data: catalog train/test splits, capped at 4,000 fit rows and 2,000 test rows.
- Native evidence per arm: from the integrated CLI with one thread at level High.

**Served semantics.** Every arm is conditioned the way `dope mlc-v3 encode` conditions it:
- Regression arms get raw-target moments.
- Sub-N arms build their building blocks on exactly the rows the native report measured (the stratified selection), while table descriptors stay on the full fit table.
- Each table's zero-arm conditioning digest is checked against the served none-mode encoding of the same `fit.csv`. It matches on all 102 measured tables.

An earlier version of this page did not do the first two, so its regression and N numbers measured a conditioning the product never serves. Those runs are superseded.

**Metric:** the six-auditor `v3_ledger_replay` clipped R, computed as the paired mean over generation seeds 20261024 and 20261025, then averaged with equal weight across tables. The ratio is a ratio of means against full, with a paired bootstrap 95% CI. The evidence file also carries a four-seed supplement.

**Coverage (102 of 112 tables valid):**
- 10 multiclass tables (p 76–795) are wider than the generator's 64 output columns. Their synthetic rows cannot be scored on the full-width test split, so they are protocol-unavailable.
- Stratified subsampling refuses classes with fewer than two rows at that N. Those arms are recorded as unavailable rather than dropped; N=300 is measured on 99 tables.
- No auditor failed on any valid table.

| arm | valid | ratio of mean R | 95% CI | RF importance Spearman (arm / full) |
|---|---|---|---|---|
| zero (`none`) | 102 | 0.985 | 0.941–1.030 | 0.602 / 0.655 |
| macro | 102 | 1.003 | 0.977–1.029 | 0.641 / 0.655 |
| macro + larger | 102 | 1.001 | 0.976–1.028 | 0.648 / 0.655 |
| full at the former default 2e9 budget | 102 | 1.007 | 0.989–1.028 | 0.653 / 0.655 |

**Paired transitions:**

| transition | ratio | 95% CI |
|---|---|---|
| zero → macro | 1.018 | 0.980–1.062 |
| macro → larger | 0.998 | 0.978–1.016 |
| larger → full | 0.999 | 0.973–1.025 |

11 tables have full R ≤ 0. Excluding them, zero comes to 0.984 (0.942–1.029) on 91 tables.

**By task:**

| task | tables | zero vs full | 95% CI | macro vs full | N=300 vs full |
|---|---|---|---|---|---|
| regression | 41 | 0.920 | 0.808–1.026 | 0.998 | 1.011 (0.983–1.051) |
| binary | 17 | 0.923 | 0.797–1.048 | 1.048 | 0.924 (0.836–1.008) |
| multiclass | 44 | 1.025 | 0.986–1.079 | 0.995 | 1.002 (0.997–1.011) |

**Random partial masks.** These keep each observed High cell with probability 0.25 (sparse) or 0.75 (dense), seeded per table so a candidate sees the same masks:

| task | random sparse | random dense |
|---|---|---|
| regression | 0.938 (0.842–1.025) | 0.953 (0.860–1.028) |
| binary | 0.982 (0.892–1.085) | 1.027 (0.934–1.132) |
| multiclass | 1.014 (0.973–1.063) | 1.012 (0.967–1.065) |

**Subsample size (served sampled conditioning):**

| N | valid | ratio | 95% CI | truly subsampled tables | ratio | 95% CI |
|---|---|---|---|---|---|---|
| 100 | 99 | 1.009 | 0.989–1.032 | 80 | 1.012 | 0.985–1.045 |
| 200 | 99 | 1.010 | 0.994–1.027 | 70 | 1.015 | 0.990–1.043 |
| 300 | 99 | 0.994 | 0.978–1.010 | 66 | 0.989 | 0.959–1.017 |
| 500 | 100 | 0.999 | 0.980–1.017 | 61 | 0.997 | 0.959–1.035 |
| 1000 | 101 | 1.000 | 0.999–1.000 | 6 | 0.996 | 0.993–0.999 |

**Against the targets (released pin):**
- Pooled, zero-landmark loss is 1.5% at the point estimate, and the CI does not exclude a loss above 5%.
- On regression the loss is 8.0%, and on binary 7.7%.
- The MACRO group alone restores both to full.
- N=300 keeps at least 95% pooled (0.994) and on regression, multiclass; not on binary: 0.924 (0.836–1.008).

**What these numbers do and don't say.** High-tier retention here is only weakly sensitive to landmark conditioning. The representation moves more: the Encoder629 cosine to full is 0.77 for zero, 0.92 for macro and 0.99 for N=300. JopeDime consumes that representation, and no policy has been evaluated on it.

## 2a. Mask-trained candidate `ecded022` (dev roster; not registered)

Addendum item 1 asks for quality at each mask level after landmark-dropout training.
- **Identity:** `ecded0225fca7c6e13c413b0e39b1040bf95f7694940b2809f1ce2ce2abf7bd9`, warm-started from the released pin and trained on `train_gradient` tables only.
- **Evaluation:** the same harness, served semantics, roster, masks, native reports and seeds as section 2 ([`retention-candidate.json`](evidence/optional-landmarks/retention-candidate.json)), on the same 102 tables.
- **Custody:** its training-source custody has been independently verified.
- **Status:** it is not a registered checkpoint; registration and redistribution need an owner decision. Nothing below is a JopeDime policy-quality claim.

| arm | candidate / candidate full | candidate / released, same arm | candidate / released full |
|---|---|---|---|
| zero (`none`) | 0.994 (0.962–1.028) | 1.024 (0.994–1.059) | 1.008 (0.970–1.047) |
| random sparse | 0.996 (0.970–1.024) | 1.021 (0.990–1.051) | 1.010 (0.979–1.045) |
| macro | 0.998 (0.973–1.022) | 1.009 (0.993–1.026) | 1.012 (0.984–1.043) |
| random dense | 0.994 (0.968–1.020) | 1.010 (0.980–1.042) | 1.008 (0.973–1.049) |
| macro + larger | 0.999 (0.979–1.020) | 1.012 (0.982–1.049) | 1.013 (0.986–1.045) |
| N=300 | 1.008 (0.989–1.028) | 1.029 (1.002–1.061) | 1.022 (0.996–1.053) |
| full | 1.000 | 1.014 (0.987–1.049) | 1.014 (0.987–1.049) |

| task | tables | zero / full (candidate) | zero / full (released) | candidate zero / released zero | N=300 / full (candidate) |
|---|---|---|---|---|---|
| regression | 41 | 0.961 (0.878–1.028) | 0.920 | 1.040 (0.981–1.110) | 1.014 (0.996–1.039) |
| binary | 17 | 1.053 (0.915–1.264) | 0.923 | 1.192 (1.046–1.429) | 1.033 (0.917–1.187) |
| multiclass | 44 | 0.995 (0.963–1.021) | 1.025 | 0.984 (0.957–1.008) | 0.999 (0.990–1.005) |

- Dropout training narrows the zero-landmark gap on regression (from 8.0% to 3.9% at the point estimate) and on binary, without making full mode worse (1.014 vs released).
- With served sampling, N=300 keeps at least 95% of full: 1.008 pooled and 1.014 on the 66 truly subsampled tables. It is measured on 99 of the 102 valid tables: the native subsampler refuses 3 multiclass tables whose classes are too small for stratification at that N. The lower CI bound is under 0.95 on binary. That is a point-estimate reading; no lower-bound rule is applied.
- Multiclass zero vs the released pin: 0.984 (0.957–1.008).
- The Encoder629 cosine of zero to full rises from 0.77 to 0.89.
- The four-seed supplement agrees in sign on every row: zero 0.998, N=300 1.011.

## 3. Subsample landmark fidelity (`subsample-ablation.json`)

There are 341 catalog train splits with at least 1,000 rows. Squashed High landmark values were compared against the unsampled bundle. As a control, the full-row panel was rerun with CV seed 1 instead of 0. Seed noise floor: Spearman 0.946, top-1 agreement 65%, max|Δ| 0.017.

| N | median Spearman | top-1 High landmark agreement | median max abs Δ |
|---|---|---|---|
| 100 | 0.40 | 16% | 0.22 |
| 300 | 0.67 | 28% | 0.085 |
| 1000 | 0.85 | 40% | 0.036 |

Subsampling changes the landmark ranking well beyond CV noise, even where downstream retention is unchanged.

## 4. Level-High dependency

The full-report High builder was run on 24 native reports, raising each cell by 0.05 or dropping it. The FINE cells (gbdt, knn1, knn16, net, gbdt_quarter) changed no coordinate of `vec256` or of the 64×17 column input. Every MACRO and LARGER cell, and the spectrum, did change them.

FINE costs 43–78% of single-thread native time on 10k×64 tables. It sits on the critical path of the final High encoding only because the spectrum is computed after the whole panel.

## 5. Latency

### 5a. Post-`#349` qualification at `29ba3db7` (28 October 2026)

Following PR `#349` (`29ba3db72df79b2a799cbfb13d55485eb46bb2d5`, column-parallel preparation and table descriptors with zero-clone cell moves), `--landmark-mode none` versus `--landmark-mode full` latency was measured at `--threads 8` (pinned to cores 64–71 under the shared qualification lock, `work_budget` 2e12) across:
- **Headline 40-real-table cohort:** 30 stratified `registry-real-v1` benchmark tables (`n = 32..1,024, p = 1..202`) plus 10 `jopedime-test` fixtures (`n <= 400`).
- **Supporting 27-cell synthetic matrix:** `{binary, multiclass, regression} × n ∈ {1,000, 10,000, 50,000} × p ∈ {11, 65, 201}` (10, 64, and 200 features plus target).

#### Headline 40-real-table cohort by regime (`29ba3db7` vs `42e5b4f5`)

| Regime | Cohort | Tables `>= 5x` | Geomean ratio (`29ba3db7`) | Median ratio | Min–Max ratio | `42e5b4f5` geomean | Fast (`none`) p50 range | Full p50 range | Gate status |
|---|---|---:|---:|---:|---|---:|---|---|---|
| **Resident/session CUDA** (`landmark_mode_bench`, 5 warm reps after checkpoint load) | `registry-real-v1` (30) | **30 / 30** | **33.9×** | 34.8× | 13.3×–62.9× | 29.5× | 1.4–27.9 ms | 53–967 ms | **PASS (accepted evaluation regime)** |
| **Resident/session CUDA** (`landmark_mode_bench`, 5 warm reps after checkpoint load) | `jopedime-test` (10) | **10 / 10** | **41.7×** | 39.4× | 27.0×–61.5× | 37.5× | 1.4–3.2 ms | 55–183 ms | **PASS (accepted evaluation regime)** |
| **Strict-cold per-process CUDA** (`dope mlc-v3 encode --device cuda:0`, 15 interleaved A/B reps) | `registry-real-v1` (30) | **0 / 30** | **1.22×** | 1.16× | 0.97×–2.18× | 1.19× | 573–612 ms | 589–1,303 ms | **FAIL (cold-CLI regime context)** |
| **Strict-cold per-process CUDA** (`dope mlc-v3 encode --device cuda:0`, 15 interleaved A/B reps) | `jopedime-test` (10) | **0 / 10** | **1.25×** | 1.24× | 1.06×–1.66× | 1.26× | 575–601 ms | 620–979 ms | **FAIL (cold-CLI regime context)** |
| **Strict-cold per-process CPU** (`dope mlc-v3 encode --device cpu --threads 8`, 15 interleaved A/B reps on 15-table subset; 3–15 reps on full 30) | `registry-real-v1` (30) | **3 / 30** | **2.31×** | 2.13× | 1.09×–5.76× | 2.27× | 110–190 ms | 120–1,050 ms | **FAIL (cold-CLI regime context)** |
| **Strict-cold per-process CPU** (`dope mlc-v3 encode --device cpu --threads 8`) | `jopedime-test` (10) | **0 / 10** | **1.44×** | 1.40× | 1.19×–1.86× | 1.42× | 108–125 ms | 132–220 ms | **FAIL (cold-CLI regime context)** |

- **Accepted regime & release test bounds (`2026-10-28T21:45Z` / `22:05Z`):**
  - The `>= 5x` qualification gate is evaluated on the **resident/session regime**, where all **40 / 40** real tables pass (`30 / 30` on `registry-real-v1` at `33.9×` geomean; `10 / 10` on `jopedime-test` at `41.7×` geomean).
  - Two resident/session release test bounds are accepted for the resident/session regime only (not as a fresh-process CLI SLA):
    1. Resident `none`-mode overhead beyond CSV read + table preparation is **`<= 5 ms`**.
    2. Resident `none`-mode `p50` is **`<= 30 ms`** on real tables with `n <= 1,024`.
  - Strict-cold per-process CLI numbers (`0 / 40` on Cold CUDA due to the fixed `~550–650 ms` CUDA-context + cuBLAS + checkpoint weight-upload startup floor; `3 / 40` on Cold CPU due to the `~100 ms` fixed process/model-load floor on `n <= 1,024`) are retained alongside as separate product-regime context. On the 15-rep interleaved Cold CPU A/B (`29ba3db7` vs `42e5b4f5`), `#349` is `+3.3%` faster on `none` (geomean `0.967×`) and `+1.2%` faster on `full` (geomean `0.988×`). A persistent CLI worker or daemon is deferred to a future serving-contract item.

#### Supporting 27-cell synthetic matrix (`{binary, multiclass, regression} × n ∈ {1k, 10k, 50k} × p ∈ {11, 65, 201}`, `--threads 8`)

| Regime | Cells `>= 5x` (`42e5b4f5` → `29ba3db7`) | Geomean ratio (`42e5b4f5` → `29ba3db7`) | Median ratio (`29ba3db7`) | Min–Max ratio (`29ba3db7`) | Notable `29ba3db7` cell behavior |
|---|---|---|---|---|---|
| **Cold CPU** (`dope mlc-v3 encode --device cpu --threads 8`, paired A/B) | **17 / 27 → 25 / 27** | **5.83× → 9.32×** | 8.68× | 3.84×–25.73× | All **9 / 9** `n = 50,000` cells pass (`5.93×–9.59×`, geomean `7.27×`, up from `0 / 9` before `#349`); only 2 tiny cells (`n = 1,000, p = 11`: binary `4.53×`, regression `3.84×`) miss due to the `~100 ms` process floor |
| **Cold CUDA** (`dope mlc-v3 encode --device cuda:0 --threads 8`) | **7 / 27 → 8 / 27** | **3.32× → 4.01×** | 3.95× | 1.98×–6.21× | `regression n = 50,000, p = 201` crosses `5×` (`2.24× → 5.31×`, fast path `13.6 s → 3.5 s`); `n = 1,000` cells are bounded by the `~550–650 ms` CUDA startup floor |
| **Resident CUDA** (`landmark_mode_bench`, 13-cell subset) | **9 / 13 → 10 / 13** | **6.88× → 9.54×** | 7.64× | 3.52×–26.95× | All `n <= 10,000` cells pass (`5.14×–26.95×`); the 3 `n = 50,000` misses (`3.52×–4.59×`) are bounded by single-threaded CSV text parsing (`0.5–3.0 s` in `RawTable::from_csv_path`), not preparation or GPU forward |

### 5b. Historical pre-`#349` cold baseline at `82cea8c4` (27 October 2026)

**Cold public calls** ([`latency-public-cold.json`](evidence/optional-landmarks/latency-public-cold.json)): fresh
`dope mlc-v3 encode` processes at Dope head `82cea8c4` (prior to PR `#349`), with:

- `--threads 8`, pinned to cores 64–71, under the shared qualification lock;
- work_budget 2e12;
- 15 repetitions, or 5 where a cold full call takes over 3 s.

Values are p50 wall times in seconds, fast (`none`) / full.

| cell | CUDA fast / full | ratio | CPU fast / full | ratio |
|---|---|---|---|---|
| real fixtures (≤1k fit rows) | 0.55–0.60 / 0.62–0.77 | 1.1–1.3× | 0.21–0.42 / 0.25–0.60 | 1.2–2.0× |
| regression 1k × 10/64/200 | not measured | — | 0.27–0.74 / 0.67–2.09 | 2.5–2.9× |
| regression 10k × 10/64 | not measured | — | 0.37 / 2.06 and 1.03 / 6.25 | 5.6×, 6.1× |
| regression 10k × 200 | 2.89 / 12.0 | 4.2× | 2.67 / 12.4 | 4.6× |
| regression 50k × 10/64/200 | 1.19 / 3.58 to 11.9 / 28.5 | 2.4–3.3× | 0.85 / 2.68 to 11.8 / 30.4 | 2.6–3.1× |
| binary 1k / 10k / 50k × 64 | 0.70 / 1.42, 1.39 / 6.73, 5.59 / 12.5 | 2.0×, 4.9×, 2.2× | 0.32 / 1.10, 0.99 / 5.90, 5.21 / 11.1 | 3.4×, 6.0×, 2.1× |
| multiclass 1k / 10k / 50k × 64 | not measured | — | 0.27 / 1.65, 0.87 / 8.91, 5.40 / 16.0 | 6.0×, 10.3×, 3.0× |

**Why the pre-`#349` cold baseline missed on `n = 50,000` and why cold CUDA has a fixed floor:**
- A cold CUDA call has a floor of about 0.55–0.60 s: process and CUDA library start (about 0.26 s) plus context and first forward.
- On large tables at `82cea8c4`, fast mode was dominated by single-threaded `prepare_table_cells` (about 4 s at 50k × 64 on `82cea8c4`). PR `#349` (`29ba3db7`) resolved the `26 / 285` multithreaded categorical-digest drift and parallelized column scaling and descriptor extraction across columns (`prepare/scaling_par.rs`, `prepare/descriptors_par.rs`), raising 8-thread Cold CPU on the 27-cell synthetic matrix from `17 / 27` (`5.83×` geomean) to `25 / 27` (`9.32×` geomean) and `9 / 9` on `n = 50,000` (`5.93×–9.59×`).
- Genuine full evidence at 8 threads is bounded by the native row caps, so it grows slowly with n.

**Device study** (`examples/mlc_v3_device_bench.rs`, warm, p50):

| path | latency |
|---|---|
| CUDA forward, B=1 | 0.68–0.73 ms |
| CUDA forward, B=1024 | 2.4 µs per table |
| CPU forward, one thread | 84 µs (host API) |
| CPU forward, default candle pool on the loaded host | 55 ms |

**Cold costs:**

| step | time |
|---|---|
| CUDA context | 0.40 s |
| first CUDA forward | 0.27 s |
| first CPU op (any size; candle one-time setup) | 0.085 s |
| CUDA-build process start (shared CUDA libraries) | 0.12 s |
| checkpoint SHA-256 | 0.9 ms |

**Stage-timed resident encode** (`examples/landmark_mode_bench.rs`, one thread at `82cea8c4`; 8 threads at `29ba3db7`):
- At 10k×64 (one thread, `82cea8c4`), `prepare_table_cells` takes 0.5–0.7 s and conditioning plus forward under 1 ms; genuine full landmarks at 2e12 take 25 s (`38×`), and at 1k×19 the ratio is `123×`.
- At 8 threads on `29ba3db7` (`landmark_mode_bench`, Resident CUDA), warm `--landmark-mode none` across the 40 real tables (`registry-real-v1` + `jopedime-test`) takes `1.4–27.9 ms` (`40 / 40 >= 5x`, geomean `33.9×` on `registry-real-v1` and `41.7×` on `jopedime-test`), with resident `none`-mode overhead beyond CSV read + table preparation under `1 ms` (within the `<= 5 ms` resident overhead bound and `<= 30 ms` resident `p50` bound for `n <= 1,024`).
