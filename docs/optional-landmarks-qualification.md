# Optional and progressive landmarks: qualification record

Measured on 27 October 2026 on the host xbabe2: AMD Threadripper PRO 3995WX (64 cores), RTX 3090, CUDA 12.8,
Rust 1.88.0, numpy 1.26.4. All numbers below come from the evidence files in
[`evidence/optional-landmarks/`](evidence/optional-landmarks/). The harness that produced them is
`research/levels/tools/optional_landmarks_qa/`. Checkpoint: released multitask
`d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`. This page records measurements only. It
authorizes no default change, and N=300 stays the progressive default.

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

**Roster:** the 112 `dev_val_ablation` tables of the reconciled v3 manifest, which excludes MLC-v3 training tables.
- Roster hash: sha256 of the sorted IDs, `c3cc1dc2500c3baa7eb99ee339948501cf7d8d83561d0a72ea051581b6e6353a`.
- Data: catalog train/test splits, capped at 4,000 fit rows and 2,000 test rows.
- Native evidence per arm: from the integrated CLI with one thread at level High.

**Metric:** the six-auditor `v3_ledger_replay` clipped R, computed as the paired mean over generation seeds 20261024 and 20261025, then averaged with equal weight across tables. The ratio is a ratio of means against full, with a paired bootstrap 95% CI.

**Coverage:**
- Two tables wider than 512 encoded columns have no landmark arms.
- Stratified subsampling refuses classes with fewer than two rows at that N. Those arms are recorded as unavailable rather than dropped.
- Restricting a full report to MACRO cells matched a native `--ids macro` run on 12 of 12 tables.

| arm | valid | ratio of mean R | 95% CI | RF importance Spearman (arm / full) |
|---|---|---|---|---|
| zero (`none`) | 110 | 0.983 | 0.940–1.028 | 0.608 / 0.655 |
| macro | 110 | 1.005 | 0.980–1.031 | 0.640 / 0.655 |
| macro + larger | 110 | 1.001 | 0.973–1.031 | 0.653 / 0.655 |
| full at the legacy 2e9 budget | 109 | 1.007 | 0.989–1.029 | 0.653 / 0.655 |
| N=100 | 106 | 1.004 | 0.985–1.024 | 0.668 / 0.651 |
| N=300 | 107 | 1.002 | 0.989–1.016 | 0.653 / 0.651 |
| N=1000 | 109 | 1.000 | 0.999–1.000 | 0.655 / 0.655 |

**Truly subsampled tables only** (fewer rows than full):

| N | tables | ratio | 95% CI |
|---|---|---|---|
| 100 | 86 | 1.005 | 0.980–1.034 |
| 300 | 72 | 1.004 | 0.980–1.028 |
| 500 | 67 | 1.006 | 0.979–1.044 |

**Paired transitions:**

| transition | ratio | 95% CI |
|---|---|---|
| zero → macro | 1.023 | 0.984–1.066 |
| macro → larger | 0.996 | 0.972–1.017 |
| larger → full | 0.999 | 0.970–1.027 |

Nineteen tables have full R ≤ 0. Excluding them, zero comes to 0.983 (0.938–1.027) on 91 tables.

**Against the targets:**
- Zero-landmark loss is 1.7% at the point estimate. That is within the 8% and 5% targets, but the CI does not exclude a loss above 5%.
- N=300 keeps at least 95% of full retention.
- N=100 is the smallest tested size that meets 95%.

**What these numbers do and don't say.** High-tier retention here is only weakly sensitive to landmark conditioning. The representation moves more: the Encoder629 cosine to full is 0.78 for zero, 0.92 for macro and 0.99 for N=300. JopeDime consumes that representation, and no policy has been evaluated on it.

## 3. Subsample landmark fidelity (`subsample-ablation.json`)

There are 341 catalog train splits with at least 1,000 rows. Squashed High landmark values were compared against the unsampled bundle. As a control, the full-row panel was rerun with CV seed 1 instead of 0. Seed noise floor: Spearman 0.946, top-1 agreement 65%, max|Δ| 0.017.

| N | median Spearman | top-1 High landmark agreement | median max abs Δ |
|---|---|---|---|
| 100 | 0.40 | 16% | 0.22 |
| 300 | 0.67 | 28% | 0.085 |
| 1000 | 0.85 | 40% | 0.036 |

Subsampling changes the landmark ranking well beyond CV noise, even where downstream retention is unchanged.

## 4. Level-High dependency

The legacy High builder was run on 24 native reports, raising each cell by 0.05 or dropping it. The FINE cells (gbdt, knn1, knn16, net, gbdt_quarter) changed no coordinate of `vec256` or of the 64×17 column input. Every MACRO and LARGER cell, and the spectrum, did change them.

FINE costs 43–78% of single-thread native time on 10k×64 tables. It sits on the critical path of the final High encoding only because the spectrum is computed after the whole panel.

## 5. Latency

These measurements are provisional; the matched cold public-call matrix is pending.

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
| first CPU forward | 0.11 s |
| checkpoint SHA-256 | 0.9 ms |

**Stage-timed resident encode** (`examples/landmark_mode_bench.rs`, one thread):
- At 10k×64, `prepare_table_cells` takes 0.5–0.7 s and conditioning plus forward under 1 ms.
- Genuine full landmarks at 2e12 take 25 s, so fast versus full is 38×.
- At 1k×19 the ratio is 123×.

The matched public-call matrix (real fixtures plus synthetic 1k/10k/50k; CPU and CUDA; p50/p95; reserved window) will be added to this file.
