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

**Roster:** the 112 `dev_val_ablation` tables of the reconciled v3 manifest, which excludes MLC-v3 training tables.
- Roster hash: sha256 of the sorted IDs, `c3cc1dc2500c3baa7eb99ee339948501cf7d8d83561d0a72ea051581b6e6353a`.
- Data: catalog train/test splits, capped at 4,000 fit rows and 2,000 test rows.
- Native evidence per arm: from the integrated CLI with one thread at level High.

**Metric:** the six-auditor `v3_ledger_replay` clipped R, computed as the paired mean over generation seeds 20261024 and 20261025, then averaged with equal weight across tables. The ratio is a ratio of means against full, with a paired bootstrap 95% CI.

**Coverage:**
- Ten multiclass tables (p 76–795) are wider than the generator's 64 output columns. Their synthetic rows cannot be scored on the full-width real test split, so they are **protocol-unavailable** and excluded from every comparison; two of them (wider than 512 encoded columns) also have no landmark arms. An earlier version of this page counted eight of them as valid tables scoring R = 0 in every arm. That left every ratio unchanged but overstated the valid counts.
- Stratified subsampling refuses classes with fewer than two rows at that N. Those arms are recorded as unavailable rather than dropped.
- Restricting a full report to MACRO cells matched a native `--ids macro` run on 12 of 12 tables.

| arm | valid | ratio of mean R | 95% CI | RF importance Spearman (arm / full) |
|---|---|---|---|---|
| zero (`none`) | 102 | 0.983 | 0.939–1.028 | 0.608 / 0.655 |
| macro | 102 | 1.005 | 0.980–1.030 | 0.640 / 0.655 |
| macro + larger | 102 | 1.001 | 0.973–1.030 | 0.653 / 0.655 |
| full at the former default 2e9 budget | 102 | 1.007 | 0.989–1.028 | 0.653 / 0.655 |

**Paired transitions:**

| transition | ratio | 95% CI |
|---|---|---|
| zero → macro | 1.023 | 0.984–1.066 |
| macro → larger | 0.996 | 0.972–1.017 |
| larger → full | 0.999 | 0.971–1.028 |

Eleven tables have full R ≤ 0. Excluding them, zero comes to 0.983 (0.938–1.027) on 91 tables.

**By task.** The pooled zero-landmark ratio is carried by the high-R multiclass tables:

| task | tables | zero vs full | 95% CI | macro vs full |
|---|---|---|---|---|
| regression | 41 | 0.913 | 0.793–1.017 | 1.007 |
| binary | 17 | 0.923 | 0.797–1.048 | 1.048 |
| multiclass | 44 | 1.026 | 0.987–1.079 | 0.995 |

**Random partial masks.** These keep each observed High cell with probability 0.25 (sparse) or 0.75 (dense), seeded per table so a candidate sees the same masks. They are worse than the structured MACRO group on regression:

| task | random sparse | random dense |
|---|---|---|
| regression | 0.941 (0.839–1.030) | 0.946 (0.854–1.024) |
| binary | 0.982 (0.892–1.085) | 1.027 (0.934–1.132) |
| multiclass | 1.014 (0.973–1.063) | 1.012 (0.968–1.065) |

**Against the targets:**
- Pooled, zero-landmark loss is 1.7% at the point estimate, and the CI does not exclude a loss above 5%.
- On regression the loss is 8.7% (binary 7.7%). That is outside the 5% target on both, and outside the 8% target on regression.
- The MACRO group alone restores both to full.
- No subsample-size retention claim is made yet; see below.

**What these numbers do and don't say.** High-tier retention here is only weakly sensitive to landmark conditioning. The representation moves more: the Encoder629 cosine to full is 0.78 for zero, 0.91 for macro and 0.99 for N=300 (evaluator partition; see below). JopeDime consumes that representation, and no policy has been evaluated on it.

**Subsample arms (not the served path).** The N arms pair genuinely sampled native gains with building blocks that the Python evaluator computed on all fit rows. The served Rust path computes the building blocks, and the inter-landmark statistics that depend on them, on the native selection. These arms therefore do not certify served N=300 conditioning, and they are reported here only as measured. A rerun that passes the native selection to the evaluator is required before any subsample retention claim.

| N | valid | ratio | 95% CI | truly subsampled tables | ratio | 95% CI |
|---|---|---|---|---|---|---|
| 100 | 99 | 1.004 | 0.985–1.024 | 80 | 1.005 | 0.980–1.033 |
| 200 | 99 | 1.002 | 0.986–1.022 | 70 | 1.004 | 0.976–1.036 |
| 300 | 99 | 1.002 | 0.989–1.015 | 66 | 1.004 | 0.980–1.028 |
| 500 | 100 | 1.003 | 0.989–1.023 | 61 | 1.006 | 0.979–1.045 |
| 1000 | 101 | 1.000 | 0.999–1.000 | 6 | 0.997 | 0.994–1.000 |

## 2a. Mask-trained candidate `ecded022` (dev roster; not registered)

Addendum item 1 asks for quality at each mask level after landmark-dropout training. The candidate
`ecded0225fca7c6e13c413b0e39b1040bf95f7694940b2809f1ce2ce2abf7bd9` was warm-started from the released pin and trained on
`train_gradient` tables only. It was evaluated with the same harness, roster, masks, native reports and seeds as section 2
([`retention-candidate.json`](evidence/optional-landmarks/retention-candidate.json)), on the 102 protocol-valid tables. Its
training-source custody has been independently verified. Its preparation lineage (lossless class vocabulary for K ≥ 33)
is still open, so it is not a registered checkpoint, and nothing below is a JopeDime policy-quality claim.

| arm | candidate / candidate full | candidate / released, same arm | candidate / released full |
|---|---|---|---|
| zero (`none`) | 0.995 (0.962–1.028) | 1.027 (0.997–1.061) | 1.009 (0.970–1.048) |
| random sparse | 0.997 (0.970–1.024) | 1.020 (0.989–1.051) | 1.011 (0.979–1.046) |
| macro | 0.995 (0.970–1.018) | 1.004 (0.988–1.020) | 1.009 (0.980–1.040) |
| random dense | 0.996 (0.970–1.022) | 1.013 (0.983–1.047) | 1.010 (0.975–1.050) |
| macro + larger | 1.002 (0.982–1.023) | 1.015 (0.985–1.051) | 1.016 (0.989–1.047) |
| full | 1.000 | 1.014 (0.987–1.049) | 1.014 (0.987–1.049) |

| task | tables | zero / full (candidate) | zero / full (released) | candidate zero / released zero |
|---|---|---|---|---|
| regression | 41 | 0.964 (0.883–1.034) | 0.913 | 1.052 (0.997–1.153) |
| binary | 17 | 1.053 (0.915–1.264) | 0.923 | 1.192 (1.046–1.429) |
| multiclass | 44 | 0.995 (0.963–1.021) | 1.026 | 0.984 (0.957–1.008) |

- Dropout training narrows the zero-landmark gap on regression (from 8.7% to 3.6% at the point estimate) and on binary, without making full mode worse.
- Multiclass zero is slightly below the released pin (0.984, CI crossing 1).
- The Encoder629 cosine of zero to full rises from 0.78 to 0.89.
- The four-seed supplement in the evidence file agrees in sign on every row.

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
