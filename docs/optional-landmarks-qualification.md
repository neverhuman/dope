# Optional and progressive landmarks: qualification record

Measured on 27 October 2026 on the host xbabe2: AMD Threadripper PRO 3995WX (64 cores), RTX 3090, CUDA 12.8,
Rust 1.88.0, numpy 1.26.4. All numbers below come from the evidence files in
[`evidence/optional-landmarks/`](evidence/optional-landmarks/). The harness that produced them is
`research/levels/tools/optional_landmarks_qa/`. Checkpoint: released multitask
`d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`. This page records measurements only. It
authorizes no default change, and N=300 stays the progressive default.

**Recipe identifier.** The full-report conditioning recipe was renamed from `legacy-full-v1` to `full-report-v1`. Inputs and the 629-D features are unchanged; full-mode effective-conditioning digests change. Evidence below that predates the rename keeps the old label; every old-to-new golden digest is listed in [`recipe-transition.json`](evidence/optional-landmarks/recipe-transition.json).

**Status: pre-training baseline.** The binding mission addendum requires a landmark-dropout-trained encoder.
Everything below measures the released pin as it is, and is the baseline that a mask-trained candidate will be
compared against with the same harness and the same masks.

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

On every Rust change the effective-conditioning digest was also unchanged on real and synthetic tables. This covers `7a5cffa9` (prepare HPC) and `82cea8c4` (single-pass parsing).

## 2. Retention (`retention-dev112.json`)

**Roster:** the 112 `dev_val_ablation` tables of the pinned role manifest (`66d2a63c…`, read directly; no mixed-role cache is opened), which excludes MLC-v3 training tables.
- Roster hash: sha256 of the sorted IDs, `c3cc1dc2500c3baa7eb99ee339948501cf7d8d83561d0a72ea051581b6e6353a`.
- Data: catalog train/test splits, capped at 4,000 fit rows and 2,000 test rows.
- Native evidence per arm: from the integrated CLI with one thread at level High.

**Served semantics.** Every arm is conditioned the way `dope mlc-v3 encode` conditions it:
- Regression arms get raw-target moments.
- Sub-N arms build their building blocks on exactly the rows the native report measured (the stratified selection), while table descriptors stay on the full fit table.
- Each table's zero-arm conditioning digest is checked against the served none-mode encoding of the same `fit.csv`.

An earlier version of this page did not do the first two, so its regression and N numbers measured a conditioning the product never serves. Those runs are superseded.

**Metric:** the six-auditor `v3_ledger_replay` clipped R, computed as the paired mean over generation seeds 20261024 and 20261025, then averaged with equal weight across tables. The ratio is a ratio of means against full, with a paired bootstrap 95% CI. The evidence file also carries a four-seed supplement.

**Coverage (101 of 112 tables valid):**
- Ten multiclass tables (p 76–795) are wider than the generator's 64 output columns. Their synthetic rows cannot be scored on the full-width test split, so they are protocol-unavailable.
- One multiclass table (ten fractional class labels) is protocol-unavailable: one column's Spearman descriptor differs between the Python evaluator and the served path (0.020782 vs 0.020763, a tie-ranking difference), so its zero-arm digest does not match the served one.
- Stratified subsampling refuses classes with fewer than two rows at that N. Those arms are recorded as unavailable rather than dropped.
- No auditor failed on any valid table.

| arm | valid | ratio of mean R | 95% CI | RF importance Spearman (arm / full) |
|---|---|---|---|---|
| zero (`none`) | 101 | 0.985 | 0.942–1.030 | 0.600 / 0.655 |
| macro | 101 | 1.003 | 0.978–1.030 | 0.641 / 0.655 |
| macro + larger | 101 | 1.001 | 0.977–1.029 | 0.647 / 0.655 |
| full at the former default 2e9 budget | 101 | 1.007 | 0.988–1.029 | 0.653 / 0.655 |

**Paired transitions:**

| transition | ratio | 95% CI |
|---|---|---|
| zero → macro | 1.019 | 0.980–1.064 |
| macro → larger | 0.998 | 0.976–1.017 |
| larger → full | 0.999 | 0.972–1.024 |

Eleven tables have full R ≤ 0. Excluding them, zero comes to 0.984 (0.938–1.028) on 90 tables.

**By task:**

| task | tables | zero vs full | 95% CI | macro vs full | N=300 vs full |
|---|---|---|---|---|---|
| regression | 41 | 0.920 | 0.808–1.026 | 0.998 | 1.011 (0.983–1.051) |
| binary | 17 | 0.923 | 0.797–1.048 | 1.048 | 0.924 (0.836–1.008) |
| multiclass | 43 | 1.026 | 0.986–1.085 | 0.995 | 1.002 (0.997–1.011) |

**Random partial masks.** These keep each observed High cell with probability 0.25 (sparse) or 0.75 (dense), seeded per table so a candidate sees the same masks:

| task | random sparse | random dense |
|---|---|---|
| regression | 0.938 (0.842–1.025) | 0.953 (0.860–1.028) |
| binary | 0.982 (0.892–1.085) | 1.027 (0.934–1.132) |
| multiclass | 1.014 (0.972–1.067) | 1.013 (0.966–1.071) |

**Subsample size (served sampled conditioning):**

| N | valid | ratio | 95% CI | truly subsampled tables | ratio | 95% CI |
|---|---|---|---|---|---|---|
| 100 | 99 | 1.009 | 0.989–1.032 | 80 | 1.012 | 0.985–1.045 |
| 200 | 99 | 1.010 | 0.994–1.027 | 70 | 1.015 | 0.990–1.043 |
| 300 | 99 | 0.994 | 0.978–1.010 | 66 | 0.989 | 0.959–1.017 |
| 500 | 100 | 0.999 | 0.980–1.017 | 61 | 0.997 | 0.959–1.035 |
| 1000 | 100 | 1.000 | 0.999–1.000 | 6 | 0.996 | 0.993–0.999 |

**Against the targets (released pin):**
- Pooled, zero-landmark loss is 1.5% at the point estimate, and the CI does not exclude a loss above 5%.
- On regression the loss is 8.0%, and on binary 7.7%. Both are outside the 5% target.
- The MACRO group alone restores both to full.
- N=300 keeps at least 95% pooled and on regression and multiclass. On binary it does not (0.924, CI 0.836–1.008).

**What these numbers do and don't say.** High-tier retention here is only weakly sensitive to landmark conditioning. The representation moves more: the Encoder629 cosine to full is 0.76 for zero, 0.92 for macro and 0.99 for N=300. JopeDime consumes that representation, and no policy has been evaluated on it.

## 2a. Mask-trained candidate `ecded022` (dev roster; not registered)

Addendum item 1 asks for quality at each mask level after landmark-dropout training.
- **Identity:** `ecded0225fca7c6e13c413b0e39b1040bf95f7694940b2809f1ce2ce2abf7bd9`, warm-started from the released pin and trained on `train_gradient` tables only.
- **Evaluation:** the same harness, served semantics, roster, masks, native reports and seeds as section 2 ([`retention-candidate.json`](evidence/optional-landmarks/retention-candidate.json)), on the same 101 tables.
- **Custody:** its training-source custody has been independently verified.
- **Status:** its preparation lineage (lossless class vocabulary for K ≥ 33) is still open, so it is not a registered checkpoint. Nothing below is a JopeDime policy-quality claim.

| arm | candidate / candidate full | candidate / released, same arm | candidate / released full |
|---|---|---|---|
| zero (`none`) | 0.994 (0.961–1.028) | 1.024 (0.993–1.060) | 1.008 (0.971–1.048) |
| random sparse | 0.996 (0.969–1.023) | 1.021 (0.990–1.051) | 1.010 (0.980–1.046) |
| macro | 0.999 (0.973–1.022) | 1.009 (0.993–1.028) | 1.013 (0.984–1.043) |
| random dense | 0.994 (0.968–1.021) | 1.009 (0.981–1.043) | 1.008 (0.974–1.049) |
| macro + larger | 1.000 (0.979–1.021) | 1.012 (0.981–1.050) | 1.014 (0.986–1.046) |
| N=300 | 1.008 (0.989–1.028) | 1.029 (1.002–1.061) | 1.022 (0.996–1.053) |
| full | 1.000 | 1.014 (0.987–1.049) | 1.014 (0.987–1.049) |

| task | tables | zero / full (candidate) | zero / full (released) | candidate zero / released zero | N=300 / full (candidate) |
|---|---|---|---|---|---|
| regression | 41 | 0.961 (0.878–1.028) | 0.920 | 1.040 (0.981–1.110) | 1.014 (0.996–1.039) |
| binary | 17 | 1.053 (0.915–1.264) | 0.923 | 1.192 (1.046–1.429) | 1.033 (0.917–1.187) |
| multiclass | 43 | 0.995 (0.964–1.023) | 1.026 | 0.984 (0.955–1.009) | 0.999 (0.990–1.005) |

- Dropout training narrows the zero-landmark gap on regression (from 8.0% to 3.9% at the point estimate) and closes it on binary, without making full mode worse.
- With served sampling, N=300 keeps at least 95% of full on every task: 1.008 pooled and 1.014 on the 66 truly subsampled tables. The lower CI bound is under 0.95 only on binary (0.917, 17 tables).
- Multiclass zero is slightly below the released pin (0.984, CI crossing 1).
- The Encoder629 cosine of zero to full rises from 0.76 to 0.89.
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

**Cold public calls** ([`latency-public-cold.json`](evidence/optional-landmarks/latency-public-cold.json)): fresh
`dope mlc-v3 encode` processes at Dope head `82cea8c4`, with:

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

**Against the target.** Every measured synthetic cold CUDA cell fails the per-cell ≥5× fast-versus-full target (7 of 7; descriptive geomean 3.0×). The quarantined and not-yet-run cells are listed in the evidence file.

**Why it fails:**
- A cold CUDA call has a floor of about 0.55–0.60 s: process and CUDA library start (about 0.26 s) plus context and first forward.
- On large tables, fast mode is dominated by `prepare_table_cells` (about 4 s at 50k × 64 on this head).
- Genuine full evidence at 8 threads is bounded by the native row caps, so it grows slowly with n.

The H1 phase-2 preparation work targets the second cause. It is not included: its port changed full-mode conditioning digests on 26 of 285 corpus tables, so it does not yet meet the exactness bar.

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

**Stage-timed resident encode** (`examples/landmark_mode_bench.rs`, one thread):
- At 10k×64, `prepare_table_cells` takes 0.5–0.7 s and conditioning plus forward under 1 ms.
- Genuine full landmarks at 2e12 take 25 s, so fast versus full is 38×.
- At 1k×19 the ratio is 123×.
