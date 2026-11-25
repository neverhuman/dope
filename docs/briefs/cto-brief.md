# D.O.P.E. v3.0.0 — Technical & Architecture Brief

> **Download formatted 2-page PDF:** [**`dope-v3-cto-brief.pdf`**](dope-v3-cto-brief.pdf) *(publication layout with trust-boundary wire diagram, task/budget charts, and full benchmark table)*

**Pure-Rust Tabular Generative Compression Under a Strict 2,048-Byte Per-Table Cap**

## 1. Executive Technical Summary

**D.O.P.E. v3.0.0** (`dope-engine-v3.0.0`, MIT) is a standalone, pure-Rust CPU binary (`dope`, Linux x86-64 GNU, `glibc 2.39+`) with **zero Python, zero LibTorch, and zero pretrained weights** in the release encoder. Given a headered CSV and explicit `--target`, `dope fit --format auto-v8 --budget 2048` compiles a table into a CRC-checked, byte-capped (**≤ 2,048 B** whole per-table file; sampling uses the shared `dope` binary) generative program.

`--format auto-v8` emits a `DPA1` container (`13 B` header: magic, encoder tag, all-zero weights fingerprint) selecting between two analytic codecs fitted on the training split:

- **Format 1** *(CLI default)*: single-tensor safetensors holding a Brotli-11 stream (quantile marginals, Gaussian-copula regimes, quantized residual tanh net, boosted oblivious trees, noise calibration).
- **Analytic `DSE8`**: TLV-framed binary packet with closed-form mandatory byte budgeting and utility-ordered optional residual sections.

When both fit, `auto-v8` runs a duplicate-grouped 2-fold check (≥ 80 groups, class-balanced, 2× sampling, linear + GBDT auditors) and picks `DSE8` iff mean fold retention leads Format 1 by `δ = 0.02` with fold veto `ν = 0.05`.

## 2. System Architecture & Data Boundary

```text
INSIDE FIREWALL (owner CPU)                         OUTSIDE (recipient / any ML stack)
================================================    =======================================
Real table (train CSV + declared --target)
  │
  ▼
dope fit --format auto-v8 --budget 2048
  • FIT-only (no query rows; k=10 support floor)
  • Format 1 vs. analytic DSE8 (dup-grouped 2-fold)
  │
  ▼
[ model.dope <= 2,048 B | DPA1 + CRC-32 ] ────────▶ dope sample model.dope -n N --seed S
(fitted summaries & schema; no complete records)     • Row-local, counter-seeded RNG (Rayon)
                                                     • Zero source-table access
                                                     │
                                                     ▼
                                                   Any downstream ML stack (linear/GBDT/NN)
                                                     │
Held-out real test split (never leaves) ◀────────────┘ (trained models return)
  │
  ▼
Owner scores returned models on held-out real targets
```

Sampling is row-local via counter-based RNG keyed by `(seed, row_index, stream, coordinate)` — deterministic and Rayon-parallel (`--threads 1..1024`) with zero source access.

## 3. Inside the 2 KiB File & FIT-Only Evidence Bank

- **`DSE8` wire layout (`13 B DPA1` + schema + `24 B header` + `3 B TLV` frames + `4 B CRC-32`):**
  - *Mandatory (closed-form sized before fit):* scale card; typed marginals (continuous, lattice, atoms, zero-inflated, constant) quantile-clipped at support floor **`k = 10`**; low-rank factor copula (`r ≤ 12`); calibrated ridge-IRLS linear head; jointly decoded nominal one-hot groups (≤ 1 active level/row).
  - *Optional (utility-ordered prefix truncated at cap without refitting):* regimes (`≤ 8`), gated 8-bin main effects (`12 B`, `≤ 24`), pair product/XOR interactions (`5 B`, `≤ 24`), depth-3 oblivious trees (`14 B`, `≤ 40`), regression residual model. Post-pack synthetic probing recalibrates intercepts.
  - *Parser hardening:* `DSE8` rejects unknown magic/version/flags/tags, duplicate/out-of-order tags, truncation, trailing bytes, missing mandatory sections, and CRC-32 errors. Format 1 caps payload/inflation/materialization at 64 MiB, headers/JSON at 1 MiB, sections at 64, features at 4,096, and checks acyclic trees against decompression bombs.
- **FIT-Only Evidence Bank (`evidence-layout-v1`):** Pure, single-threaded, deterministic, uncached Rust with no query-row field in its input type. Emits row-count-independent masked tensors — **128** col / **16** pair / **160** table slots (+ 16/class, 4/class-col, 8/group) across **4 tiers**: `T0` (1 sort/col: support at `k = 10`, 17-knot quantile residuals, L-moments/tails, missingness, task/size), `T1` (rank-12 factor/precision, non-Gaussian/tail/directional dependence, 16 pair channels incl. product/XOR target interactions, 3-fold OOF IRLS residuals, spectrum/proximity), `T2` (2-fold depth-3 GBDT witness, Walsh–Hadamard functional ANOVA, calibration, 1-NN noise floor), and `T3` (uncached closed-loop packet AUC, correlation gap, DCR share — masked unless a packet is passed). Enforces canonical row order and zero raw-value, header, or extrema echo.

## 4. Supply Chain, Security & Deployment Footprint

- **Signed releases (`dope-engine-v3.0.0`):** Built on Linux x86-64 (`--no-default-features`, LTO, stripped, abort-on-panic) and signed via commit-pinned `actions/attest` with GitHub OIDC and short-lived Sigstore credentials (no long-lived keys). Ships a fixed 6-member weight-free inventory (`dope-linux-x86_64`, `BUILDINFO.json`, `Cargo.lock`, `DEPENDENCIES.txt`, `LICENSE`, `THIRD_PARTY_NOTICES`) + `SHA256SUMS` + `dope.sigstore.json`.
- **Verification & hygiene:** SHA-256-pinned `scripts/install.sh` verifies Sigstore provenance accountlessly (manual check: `scripts/verify-engine-release.sh`, `gh ≥ 2.102.0`). CI blocks Python, LibTorch, CUDA, and MKL linkage. *(Attestation verifies build identity only, not quality or privacy.)*
- **Source & portability:** `cargo install --locked --path engine/dope-cli` (optional `--features cuda`/`metal`, unqualified in CPU CI). For macOS, Windows, ARM, or glibc < 2.39: `docker run --rm -it --platform linux/amd64 -v "$PWD":/work -w /work ubuntu:24.04 bash`.
- **Runtime & contract:** Zero outbound network calls during `fit`/`sample`/`eval`; no Python or GPU required. Enforces mandatory declared target-last intake, duplicate-group splitting, and zero source value/header/extrema echo in errors or sidecars.

## 5. Benchmark Numbers, Guards & Technical Limits

*DEV-SELECT fresh splits (`dope-kpi/v2` candidate contract; 44 informative cells, 11 families: 4 regression, 3 binary, 4 multiclass; median float32 fit matrix 25.7× artifact bytes). Uncertified release. Transfer Score = `100 × (1 + TRG)` (100 = second real sample; 0 = null/failure floor); weakest of linear and GBDT auditors counts.*

| Generator | 4× [95% CI] | 1× | Regr / Bin / Multi | Held-out MLP | File | Realism / Privacy | Copies mean (max) | C2ST | Failed |
| --- | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |
| **DOPE v3** (`auto-v8`, ≤2 KiB) | **87.9** [81.1, 95.1] | **69.1** | 86.3 / 86.4 / 91.1 | 65.1 | **1,787 B** | 0.194 / 0.688 | 0.000 (0.000) | 0.967 | **0/44** |
| ARF (`arfpy`)† | 77.2 [58.4, 90.0] | 63.0 | 92.9 / 58.8 / 79.7 | **77.0** | 2.94 MB | 0.278 / 0.031 | 0.000 (0.004) | 0.941 | 0/44 |
| SDV Gaussian copula† | 40.2 [27.9, 51.8] | 17.2 | 83.2 / 38.8 / -1.5 | 18.9 | 141 KB | 0.111 / 0.601 | 0.005 (0.200) | 0.992 | 0/44 |
| SDV TVAE† | -3.9 [-26.2, 17.8] | -16.1 | 30.0 / -43.2 / 1.4 | -4.5 | 475 KB | 0.000 / 0.326 | 0.000 (0.000) | 0.977 | 5/44 |
| SDV CTGAN† | -69.2 [-82.7, -55.2] | -69.4 | -50.2 / -88.8 / -68.5 | -65.6 | 1.05 MB | 0.000 / 0.743 | 0.006 (0.271) | 0.996 | 0/44 |
| Stored rows (≤2 KiB) | 2.6 [-30.9, 34.8] | -0.9 | — | — | 2,016 B | — | — | — | 0/44 |
| *Copy / Bootstrap* | *96.1 / 91.2* | *96.1 / 78.6* | *—* | *98.6 / 85.5* | *—* | *—* | *—* | *0.530 / 0.572* | *0/44* |

*† Uncapped defaults; 0 files ≤ 2,048 B (0.0 under cap penalty). Ablation: Format 1 alone 73.3 (7/44 failed); `DSE8` alone 65.8 (0/44 failed).*

- **Paired differences (4×):** vs. **ARF**: **+10.8 (95% CI -3.5 to +29.5, Holm *p* = 0.114)** → **HOLD** (missed tasks, MLP, realism); the 95% paired interval includes zero across all three DEV split sets, so DOPE is not shown to transfer better than ARF on this look. vs. **SDV**: **PROMOTE** vs. TVAE (+91.9 [+71.9, +112.3]); **HOLD** vs. Gaussian copula (+47.8 [+33.7, +62.6], tasks) and CTGAN (+157.1 [+141.8, +173.6], privacy).
- **Why `SingleEncV8` did not ship:** A neural refinement trained on real, target-swap, and synthetic-prior episodes (3 runs × 8,000 updates, RTX 4090) lowered training loss and screening error, but scored **-0.37** (95% CI **[-2.71, +1.72]**) on fresh DEV splits and missed task/MLP guards; v3.0.0 ships zero learned V8 weights.
- **Hard limits:** DEV-SELECT only (no sealed panel; uncertified release). Empirical privacy gate only (DCR ≤ 0.55, MIA AUC ≤ 0.55, TPR@1%FPR ≤ 0.02, copy excess ≤ 0.005, `k ≥ 10`); no formal differential privacy mechanism and **not HIPAA de-identification**. Boosted-tree C2ST separates rows easily (AUC **0.967**); cross-model feature-driver preservation is unmeasured; foundation-model auditors are unqualified. On the published `v3.0.0` binary, `fit` and raw `sample` support `auto-v8`, while `eval`/`inspect`/`predict`/`encode`/`throughput` read Format 1 files (source builds on `main` add direct `DPA1` evaluation to `dope eval`).

## 6. 5-Minute Technical Evaluation & Contribution Path

```bash
dope="$HOME/.local/opt/dope/3.0.0/bin/dope"
"$dope" fit train.csv --target y --format auto-v8 --budget 2048 -o model.dope
wc -c < model.dope                                            # <= 2048 bytes
"$dope" sample model.dope -n 1000 --seed 1 -o synthetic.csv
# Published v3.0.0 eval uses a Format 1 schema carrier (source builds on main also accept -m model.dope directly):
"$dope" fit train.csv --target y --format 1 --budget 2048 -o audit-schema.dope
"$dope" eval -m audit-schema.dope --train train.csv --test test.csv \
  --synth synthetic.csv --auditors linear --seed 1 --json evaluation.json
```

**Pointers:** [CEO Brief](ceo-brief.md) · [Real-Data Walkthrough](../../examples/v3-quickstart/README.md) · [Training Guide](../training.md) · [Roadmap](../../ROADMAP.md) *(Evidence Bank feature engineering, formal differential privacy, sealed-panel certification, foundation-model auditors, cross-platform builds)* · [Signed Engine Releases](../signed-engine-release.md) · [Engine CLI](../../engine/README.md) · [White Paper (PDF)](../whitepaper-v3/dope-v3.pdf)
