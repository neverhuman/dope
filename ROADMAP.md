# D.O.P.E. Roadmap & Open Call for Contributors

> **Co-Authorship Invitation:** DOPE is an open research and engineering project. Contributors whose merged work adds a validated Evidence Bank feature family, a neural architecture or training improvement, a formal differential privacy mechanism, new downstream auditor benchmarks, or major engine capabilities that ship in a future release (`v3.1+` / `v4.0`) will be **credited as co-authors on the corresponding technical report / arXiv white paper, [`CITATION.cff`](CITATION.cff), and release notes**.

---

## 1. Where DOPE v3.0.0 Stands Today

Release `v3.0.0` establishes a clean, weight-free pure-Rust baseline:

- **Strengths:** Transfer Score **87.9** (4× sampling) from a **1,787-byte** median whole file (strictly `<= 2,048 B`, `0/44` failed cells), outperforming stored real rows under the same 2 KiB cap (**2.6**) and leading SDV Gaussian copula (**40.2**), TVAE (**-3.9**), and CTGAN (**-69.2**) across all three development split sets from files 79× to 590× smaller.
- **Concrete gaps we want to close next:**
  1. **Make the learned `SingleEncV8` tier beat analytic `DSE8` decisively on fresh splits:** In the v3.0.0 pilot, neural refinement lowered training loss but landed within noise on fresh splits (`-0.37`, 95% CI `[-2.71, +1.72]`).
  2. **Close the regression & held-out MLP gap vs. ARF:** On the report splits, DOPE's paired difference vs. ARF is `+10.8 [-3.5, +29.5]` (includes zero), while ARF scores higher on regression (`92.9` vs. `86.3`) and on the held-out MLP auditor (`77.0` vs. `65.1`).
  3. **Improve structural realism without memorization:** Boosted-tree C2ST AUC is `0.967`, so synthetic rows remain easy to distinguish from real rows.
  4. **Expand guarantees and evaluation:** Measure cross-model feature-driver preservation, add formal $(\varepsilon, \delta)$ differential privacy, evaluate tabular foundation models (TabPFN / TabICL / Mitra) as auditors, and open a sealed confirmation panel.

---

## 2. Highest-Leverage Path: Data-Specific Features for the Deep Learning Network

Why did the first `SingleEncV8` pilot stall near the analytic baseline? Because the owner-side encoder **never sees raw rows** — it only sees the fixed-size **FIT-only Evidence Bank** (`contracts/evidence-layout-v1.json`: `128` column slots, `16` pairwise edge channels, `16` class slots, and `160` global table slots across tiers `T0`–`T3`).

A deep network can only outperform closed-form analytic estimators if the Evidence Bank exposes **richer, data-specific structural signals** capturing the exact failure modes where analytic `DSE8` loses transfer or realism — and if every new feature hypothesis is tested rapidly with fast local regression tests before committing GPU hours.

Note that `contracts/evidence-layout-v1.json` already contains **unallocated reserved slots** (e.g., `W_CLOSED_LOOP_COL` slot `127`, `class_channels` slots `14–15`, and unallocated `glob_slots` ranges `57–63`, `70–87`, `95–103`, `112–139`, `145–159`), making incremental feature packs clean to prototype before cutting `evidence-layout-v2`.

### Brainstorm: High-Value Data-Specific Feature Families to Prototype

#### A. Nonlinearity, Smoothness & Oblique Geometry *(targets held-out MLP & regression lift)*
| Feature Idea | Scope | Signal It Gives `SingleEncV8` |
| --- | --- | --- |
| **1D Spline / RBF Smoothness Ratio** | Per-column | Compares smooth cubic-spline or RBF-ridge OOF gain against linear and 1D-stump gain — detects smooth curved effects where MLPs outperform axis-aligned oblivious trees. |
| **Oblique (Rotated) Depth-2 Split Gain** | Per-pair / Global | Gain of a cheap 2-feature linear-combination split minus axis-aligned tree gain — identifies diagonal decision boundaries ($a x_i + b x_j > c$) that oblivious trees miss. |
| **3-Way Interaction & Higher-Order Parity Energy** | Global / Top-triplets | Residual energy from fast Walsh–Hadamard / random 3-way product probes beyond pairwise `s10_prod` and `xor_quadrant` channels. |
| **Periodic / Cyclical Spectral Peak Ratio** | Per-column | Normalized top DFT / autocorrelation peak on sorted or lattice-indexed numeric columns (angles, hour-of-day, day-of-year, frequency sweeps). |
| **Target-Decile Conditional Spread & Skew** | Per-column / Global | Conditional IQR and L-skew of residuals across target quantiles — richer heteroscedasticity and heavy-tail guidance for regression decoders. |

#### B. Support Grammar, Constraints & Manifold Realism *(targets C2ST reduction without leakage)*
| Feature Idea | Scope | Signal It Gives `SingleEncV8` |
| --- | --- | --- |
| **Pairwise Order / Inequality Prevalence** | Per-pair | Share of fitting rows satisfying $x_i \le x_j$ (e.g., start/end timestamps, min/max measurements, principal vs. total balance), thresholded at support floor $k \ge 10$. |
| **Simplex & Constant-Sum Group Flags** | Per-group / Global | Detects compositional or accounting constraints ($\sum_{j \in G} x_j \approx c$) and deterministic linear ratios/differences that tree discriminators instantly exploit. |
| **Conditional Zero-Inflation Coupling** | Per-pair | Asymmetric co-zero probability $P(x_j = 0 \mid x_i = 0)$ vs. marginal $P(x_j = 0)$ — captures structural zeros (e.g., no claim $\implies$ zero payout). |
| **Multi-Scale Heaping & Censoring Walls** | Per-column | Mass concentrated at human-rounded multiples (5, 10, 100) and sharp left/right truncation walls backed by $\ge 10$ rows. |
| **Asymmetric Rank-Copula Quadrant Residuals** | Per-pair | Empirical upper-left / lower-right quadrant asymmetry and Clayton / Gumbel / Student-$t$ residual excess beyond symmetric Gaussian factors. |
| **Cluster Anisotropy & GMM BIC Elbow** | Global | Optimal regime count ($K \in \{1..8\}$) from a fast BIC sweep and eigenvalue anisotropy per cluster, guiding `DSE8` regime allocation. |

#### C. Missingness Mechanism & Categorical Semantics
| Feature Idea | Scope | Signal It Gives `SingleEncV8` |
| --- | --- | --- |
| **MCAR / MAR / MNAR Fingerprints** | Per-column | OOF AUC of predicting `is_missing(x_j)` from other features $X_{-j}$ (MAR strength) vs. from $y$ (informative missingness), plus co-missingness block structure. |
| **High-Cardinality Dispersion & Hierarchy** | Per-group / Per-pair | Within-level target variance/skew, Zipf rare-tail exponent, and conditional entropy $H(\text{cat}_A \mid \text{cat}_B)$ detecting hierarchical parent-child categories (e.g., city $\to$ state). |

#### D. Downstream Learner Sensitivity & Closed-Loop T3 Upgrades
| Feature Idea | Scope | Signal It Gives `SingleEncV8` |
| --- | --- | --- |
| **Cross-Learner Contrast Landmarks** | Global | Cheap FIT-only OOF gain gaps between Ridge, RBF-kernel Ridge, Shallow MLP, and Oblivious GBDT — explicitly tells the encoder what signal an MLP extracts that a tree does not. |
| **Feature-Driver Concentration & Agreement** | Global / Per-column | Gini/entropy of feature importances and Spearman rank agreement between linear vs. GBDT feature attributions. |
| **Closed-Loop T3 Discrepancy Profiles** | Per-column / Per-pair | When refining an analytic Step-0 packet, feed per-column Wasserstein-1 gap, worst-pair correlation error, and random-tree leaf-occupancy KL divergence directly into `W_CLOSED_LOOP_COL` so the decoder repairs exact Step-0 weaknesses. |

---

## 3. Fast Contributor Loop: Idea -> Feature -> Quick Regression Test

Every feature proposal follows a fast, self-contained engineering loop you can run on a laptop in **under 10 minutes**:

1. **Respect the 5 Evidence Bank Invariants** (enforced by tests):
   - **FIT-only by construction:** Zero access to query/test rows; modifying query rows must produce byte-identical evidence tensors.
   - **Support floor $k \ge 10$:** Never emit statistics backed by fewer than 10 fitting rows.
   - **Zero raw-value echo:** Never emit raw cell values, headers, or exact min/max extrema in features or error messages.
   - **Permutation & order symmetry:** Strictly invariant to row order and equivariant to column permutation.
   - **Bounded & masked:** Squash unbounded quantities to finite ranges and clear the validity bitmask bit whenever a statistic is undefined.
2. **Implement in Rust + Python parity:**
   - Add the Rust computation under `engine/dope-core/src/evidence/` (keeping every `.rs` file `< 500` lines).
   - Mirror the slot calculation in `research/levels/v8_evidence.py` and document the slot in `contracts/evidence-layout-v1.json` (or draft `v2`).
3. **Run fast local regression & parity checks (`just v8-regression`):**
   ```bash
   # Rust evidence + DSE8 codec unit tests
   cargo test --manifest-path engine/Cargo.toml --locked -p dope-core evidence single_encoding::v8 auto::v8

   # 12-fixture Rust-vs-Python parity suite (max |py - rs| <= 1e-4, <= 2048 B, 0 illegal one-hot rows)
   cargo build --manifest-path engine/Cargo.toml --locked --release -p dope-arena --bin dope_v8_emit
   pytest research/levels/tests/test_v8_rust_parity.py research/levels/tests/test_v8_losses.py -v
   ```
4. **Run a quick synthetic/real ablation (`docs/training.md`):**
   - Run a short CPU or single-GPU training comparison (`baseline` vs. `baseline + your feature`) using `research/levels/s3_v8_trainer.py` and compare `s3_v8_eval.py` (`trg_lin`, `trg_gbdt`, `trg_mlp_heldout`, `c2st_xy`, `ece15`, `dcr_share`, `packet_bytes`).
5. **Open a Pull Request:**
   - Include your feature definition, unit/parity test results, and quick regression delta.
   - GitHub Actions automatically runs `public-smoke`, `public-hygiene`, and `v8-feature-regression` on fork PRs. For promising feature PRs, maintainers run the full multi-split `dope_kpi` DEV/CONFIRM benchmark and publish receipts.

---

## 4. Release Milestones (`v3.1` -> `v4.0`)

| Milestone | Key Deliverables |
| --- | --- |
| **`v3.1` — Feature Pack & Portability** | - First community-contributed Evidence Bank feature pack (`evidence-layout-v2`)<br>- Learned `SingleEncV8` weights that clear the acceptance guards over analytic `DSE8` on fresh splits<br>- Native `DPA1` support in `dope eval` & `dope inspect` (shipping in source builds now)<br>- Signed release binaries for macOS (Apple Silicon), Linux ARM64, and older-glibc/musl Linux x86-64 |
| **`v3.2` — Realism, Drivers & Foundation Auditors** | - Cross-model feature-driver preservation benchmark (measuring attribution fidelity across learners)<br>- Tree-visible joint realism & constraint-aware decoding to reduce boosted-tree C2ST without increasing exact copies<br>- Tabular foundation model downstream auditor evaluations (TabPFN, TabICL, Mitra) with explicit pretraining-overlap disclosure |
| **`v4.0` — Formal Privacy & Certification** | - **Formal $(\varepsilon, \delta)$ `DP-DSE8`:** Calibrated noise injection on bounded packet sufficient statistics with formal $(\varepsilon, \delta)$ privacy-budget accounting (v3.0.0 has no formal DP)<br>- Sealed confirmation panel evaluation (`docs/release.md` certification gates)<br>- End-to-end vendor-selection regret study quantifying real-test utility lost when choosing models via DOPE artifacts vs. real tables<br>- Completion of the Rust-only nontraining migration (`docs/plans/next-version/dope/language-policy/README.md`) |

---

## 5. How to Get Involved & Co-Authorship Criteria

We welcome ML researchers, statisticians, tabular-data practitioners, and Rust systems engineers:

- **Start a Feature RFC:** Open a GitHub Issue describing the tabular failure mode, the proposed FIT-only feature(s), and how you plan to test them on the 12-fixture / synthetic-prior regression suite.
- **Good First Contributions:**
  - Prototype one of the Evidence Bank features in Section 2 (with Rust + Python parity tests).
  - Add new tabular stress fixtures (e.g., survival/censored data, skewed pricing tables, hierarchical categories) to `research/levels/tests/test_v8_rust_parity.py`.
  - Improve `dope inspect` / `dope eval` ergonomics for `DPA1`/`DSE8` artifacts in `engine/dope-cli/`.
  - Help port remaining nontraining Python utilities to Rust (`docs/plans/next-version/dope/language-policy/README.md`).
- **Authorship on Future Versions:** Any contributor whose merged pull request(s) contribute a substantive, verified improvement to a future release (such as a retained Evidence Bank feature family, a promoted neural architecture/loss upgrade, an $(\varepsilon, \delta)$ differential-privacy mechanism, a new auditor/driver benchmark suite, or a major engine subsystem) will be invited to be listed as a **co-author on the corresponding DOPE technical report / arXiv submission, `CITATION.cff`, and release notes**. Authorship inclusion is confirmed collaboratively before each release by the project steward (`GOVERNANCE.md`).

**Helpful links:** [Contributing Guide](CONTRIBUTING.md) · [Training Guide](docs/training.md) · [Repository Map](docs/REPO_MAP.md) · [Real-Dataset Quick Start](examples/v3-quickstart/README.md) · [CEO Brief](docs/briefs/ceo-brief.md) · [CTO Brief](docs/briefs/cto-brief.md)
