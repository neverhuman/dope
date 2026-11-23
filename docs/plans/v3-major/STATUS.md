# DOPE v3 DEV decision ledger

This is the canonical `tips/v3/*.md` decision table. It includes the pinned DEV report and the paired DEV-lite screens for `v8-s7w` w1 k1500 and `v8-r2` k3000. The two Dataset Features tips are distinct. `Catalog` supplies `F001`–`F144` and `RichStats` supplies the bounded layout proposal. See the [six-tip crosswalk](verification/evidence-crosswalk.md). This closeout uses DEV evidence only.

## Executive Summary & Status Totals

| Scope | `KEPT` | `REJECTED` | `TESTING` | `NOT STARTED` | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-Level Engineering & Policy Dispositions | 6 | 9 | 5 | 3 | 23 |
| Proposal Groups (`V3-01`–`V3-22`) | 0 | 2 | 18 | 2 | 22 |
| Six-Tip Source Register (all 6 `tips/v3/*.md` files) | 8 | 2 | 111 | 2541 | 2662 |

## How to read the ledger

- **`KEPT`**: Verified by a cited measured run (run ID / commit SHA, receipt/test name + SHA-256 prefix, metric, and 95% CI or exact pass/fail measurement). Every `KEPT` row cites a measured run.
- **`REJECTED`**: Disproved or superseded by a cited measured run (run ID / commit SHA, receipt/test name + SHA-256 prefix, metric, and 95% CI or exact failure/interference measurement). Every `REJECTED` row cites a measured run.
- **`TESTING`**: Implemented and under DEV screening, or awaiting promotion evidence. `V3-22` `auto::v8` passed release-route selection at `+0.060 [+0.009, +0.114]` over v2.0.0 `auto` on the fresh DEV report (Holm `p = 0.0091`). Its scientific promotion decision is **`HOLD`**. Jepson authorized publishing the analytic package after release verification, with the four DEV guard misses disclosed. The [release closeout](RELEASE_CLOSEOUT.md) records the final candidate decision process.
- **`NOT STARTED`**: Not yet executed on DEV (including policy-excluded sealed CONFIRM/protected-panel items and superseded draft ABIs that were never run).
- Source locators use physical lines in the six SHA-pinned `tips/v3/*.md` files listed in [Source register](#source-register). Empirical privacy diagnostics reported in DEV receipts are neither formal differential privacy nor HIPAA de-identification.

## Current dispositions with evidence

| Proposal | Owner | State | Measured run, metric, 95% CI and DEV test |
| --- | --- | --- | --- |
| Retain native v2 TRG at 4× and 1× as the score of record | Claude | KEPT | Run `public-lite-replay` (`7c1f4598`, binary `506fdfae`, `primary.json` `59854869`, `compare.json` `f07f2647`): 125 informative 4× cells; `dope-2.0.0-small` −0.232 vs SDV GC −0.686 (Δ +0.454 [+0.342, +0.553]) & TVAE −1.070 (Δ +0.838 [+0.662, +1.042]); native & independent reducers agree on 2,022 entries/arm. [Baseline replay](#baseline-replay) |
| Preserve and report v2 alongside a failure-floor sensitivity | Codex | KEPT | Run `public-lite-replay` (`7c1f4598`, `compare.json` `f07f2647`): paired leads +0.454 [+0.342, +0.553] (vs SDV GC) and +0.838 [+0.662, +1.042] (vs SDV TVAE) survive unit failure-floor penalty and removal of clipping. [Failure-floor analysis](#failure-floor-analysis) |
| Change the score of record to the unit floor | Owner | TESTING | Run `public-lite-replay` (`7c1f4598`, `compare.json` `f07f2647`) provides sensitivity diagnostic only; successor freeze remains open. [Failure-floor analysis](#failure-floor-analysis) |
| Use the exact legal 2,048 B cap for produced headline artifacts | Antigravity | KEPT | Run `public-lite-replay` (`7c1f4598`, 310/310 DOPE files ≤ 2,048 B) + Run `v3-report-dev-s5-8` (`3fb802fd`, `primary.json` `55974867`, `compare.json` `5ac5a9bd`): `dope-v3b-auto-v8` 108/108 packets ≤ 2,048 B (1,787 B med, 2,048 B max, 0/44 failed, 4× TRG −0.121 [−0.189, −0.049], bytes guard PASS). [Packet bytes and layout](#packet-bytes-and-layout) |
| Candidate's 64-slot normalized admission guard for the approved 48-slot route | Antigravity | REJECTED | Run `codex-candidate-replay@8f4b4b92`: 49-slot, 8-class fixture produced 3,199 B at 4,096 B cap instead of refusing; superseded by `7968ea7f` admission refusal (`single_encoding_v8` 5/5 passed). [Packet bytes and layout](#packet-bytes-and-layout) |
| Candidate's pre-fix normalized layout choice at 48 slots and 8 classes | Antigravity | REJECTED | Run `codex-candidate-replay@8f4b4b92`: direct 2,048 B fit returned `over_budget` (2,119 B mandatory base); replaced by `7968ea7f` adaptive planner (`v3-report-dev-s5-8` `55974867`: 0/44 failed, max 2,048 B). [Packet bytes and layout](#packet-bytes-and-layout) |
| Candidate's positional tiebreak on identical column-permutation fingerprints | Antigravity | REJECTED | Run `codex-candidate-replay@8f4b4b92`: transposing 256 paired rows altered evidence bytes; replaced by `4c7b16a0` (`evidence::extract_with_rows` + `dope-ffi` ABI 4 `RowIdentity`, 12/12 tests passed). [Extraction identity](#extraction-identity) |
| Pre-fix uncalibrated DSE8 analytic target/copula fitter (`3b653a66`) | Antigravity | REJECTED | Run `v3-dev-wide` (`primary-dse8-p0.json` `d79f5cbf`, 46 informative cells): 4× TRG −0.38216 [−0.62519, −0.17441] vs `dope-2.0.0-auto` −0.20726; DEV-lite −0.55546 [−0.92725, −0.24889]. Replaced by calibrated `7968ea7f` (`dse8-p0-v3b`: DEV-lite −0.27758, Wide DEV −0.18968). [DEV receipts](#dev-select-receipts) |
| Pre-fix single 75/25 `auto::v8` acceptance guard (`3b653a66`) | Antigravity | REJECTED | Run `v3-dev-wide` (`compare-auto-v8-vs-2.0.0-auto.json` `24a09277`, 46 informative cells / 11 families): 4× ΔTRG −0.02667 [−0.06788, +0.00750], Holm p = 1.0, HOLD. Replaced by `7968ea7f` target-stratified 2-fold grouped guard (+0.03678 [+0.00839, +0.06234] Wide DEV; +0.060 [+0.009, +0.114] Fresh DEV Report). [DEV receipts](#dev-select-receipts) |
| `v8-r1` neural checkpoints (`k6000`, `k10000`) on pre-fix DSE8 packets as release candidate | Claude | REJECTED | Run `v3-dev-wide` (`compare-v8r1-vs-dse8-p0.json` `baca818c`, 46 informative cells / 11 families): `k10000` −0.39357 (ΔTRG −0.01142 [−0.06422, +0.02764], HOLD), `k6000` −0.41223 (ΔTRG −0.03008 [−0.07049, +0.00002], HOLD) vs `dse8-p0` −0.38216; archived on S3 as training record only. [DEV receipts](#dev-select-receipts) |
| Explicit stable physical row and group IDs for extraction and fold assignment | Claude | TESTING | Commit `4c7b16a0` (`dope-ffi` ABI 4 + `evidence::extract_with_rows`): 12/12 `dope-ffi` tied-pair permutation, duplicate-ID refusal, grouped-fold, and query-poison tests pass; DSE8 packet-fit row order (`single_encoding/v8/mod.rs:281`) queued post-release. [Extraction identity](#extraction-identity) |
| Strict little-endian DSE8 wire v1 framing and stable analytic decoder identity | Antigravity | TESTING | Run `single_encoding_v8@7968ea7f` (5/5 passed, 12 malformed vectors rejected) + `v3-report-dev-s5-8@3fb802fd` (`55974867`: 0/44 failed, 868/2,048 B, 4× TRG −0.342 [−0.652, −0.103]). [Packet bytes and layout](#packet-bytes-and-layout) |
| Strict hard training-reference parity on the pinned candidate | Codex | KEPT | Independent replay `codex-hard-parity@8f4b4b92` tightened a generated test copy and measured 12/12 within 1e-5 with exact labels. The committed suite allows 1e-4 and ≥95% neural classification agreement. [Hard forward](#hard-forward) |
| Source-free narrow binary sampling and per-row seed prefix | Codex | KEPT | Run `codex-recipient-replay@7c1f4598` (797 B fixture reproduced fit & row 0 after input removal; 339/339 root Rust tests pass) + Run `v3-report-dev-s5-8@3fb802fd` (`58314f0a`: 108/108 release packets SHA-identical). [Recipient behavior](#recipient-behavior) |
| Complete DSE8 recipient scope, including range/chunk and restricted package | Codex | TESTING | Native row-local & `auto_v8_cli` tests pass (`ops/ci/engine.sh@3fb802fd`: 655/655 passed); restricted bubblewrap package verification pending suitable runner. [Recipient behavior](#recipient-behavior) |
| Fresh group-aware episode splits, FIT-only production and parallel feed | Claude | TESTING | Run `c6-factory-bench@075b6752` (97/167/256 eps/s at 1/2/4 producers, 0/1,440 failures, 842 B med) + `4c7b16a0`/`7bdd58f4` (`dope-ffi` 12/12, factory/feed pytest 23/23); native V8 matched exposure/resume audit continues via `v8-r2`. [Episode stream](#episode-stream) |
| Background or corpus-wide caches of row-derived FIT evidence | Claude | REJECTED | Run `c6-factory-bench@075b6752` (256 eps/s at 4 producers, 10–12 ms/ep, 0/1,440 failures) + Run `dope-ffi@4c7b16a0` (12/12 query-poison & permutation tests: 0 byte diff on poisoned query), proving uncached FIT-only extraction meets throughput without cross-split leakage (09:50Z binding owner rule). [Extraction identity](#extraction-identity) |
| Release a trained V7 candidate on the basis of H2's selected gain | Claude | REJECTED | Run `h2-round108-screen` (`codex-v3` replay, 12 DEV cells): 4× ΔTRG −0.004245 [−0.045155, +0.039275] vs `v72-step0` (no lift over Step-0). [Repaired V7](#repaired-v7) |
| Use authentic SDV GaussianCopula and TVAE at the native budget | Codex | KEPT | Run `public-lite-replay` (`7c1f4598`, `primary.json` `59854869`, `compare.json` `f07f2647`, 125 informative cells): SDV GC −0.686 (105,848 B med, 0 failed) & TVAE −1.070 (404,722 B med, 2 failed) vs `dope-2.0.0-small` −0.232 (Δ +0.454 [+0.342, +0.553] & +0.838 [+0.662, +1.042]). [Published generators](#published-generators) |
| Claim superiority to authentic CTGAN, ARF, TabSyn, TabDDPM or TabDiff on the v3 common roster | Codex | NOT STARTED | Historical runs and 3-cell CTGAN smoke are narrower; pinned common DEV comparison not yet run. [Published generators](#published-generators) |
| Publish joint T90/D80 at a measured complete-byte frontier | Claude | NOT STARTED | Independent T90/D80 DEV run and gate receipts not yet executed. [T90 and D80](#t90-and-d80) |
| Run a protected qualification campaign (CONFIRM/HOLDOUT/SEALED) in this handoff | Owner | NOT STARTED | Policy-blocked by active DEV-only pause — never run. [Execution scope](#execution-scope) |
| Use a JSR GPU or add DOPE CPU load on JSR H53_S2 ranks (`xbabe5/7/8/9`) | Owner | REJECTED | Run `jsr-gpu-census@20261123T1025Z` (`~/dope/.agent/reports/v3-train/baseline/`, 1 Hz × 600 s): xbabe3 3090 40% util / 18.6 GB VRAM, xbabe1 4090 45% / 18.5 GB, xbabe6 A10 80% util; plus Run `a1-h53-interference@20261122-23`: spare-CPU load on H53 ranks slowed synchronous JSR by −3.6% (×2) and −4.5% (×1), recovering +3.9% on pause. [Execution scope](#execution-scope) |

## Proposal groups from the six tips

`V3-01`–`V3-22` cover all unique architectural, evidence, wire, training, evaluation, and boundary proposals synthesized from the six `tips/v3/*.md` specs (`Catalog`, `RichStats`, `DFP1`, `Evidence`, `Runbook`, `Compiler`). The [Source register](#source-register) indexes all 2,662 individual source clauses.

The frozen evidence ABI is `contracts/evidence-layout-v1.json` (`COL=128`, `EDGE=16`, `CLASS=16`, `CLASS_COL=4`, `GROUP=8`, `GLOB=160`, SHA-256 `e99647eb75cd078e0951ea93e12f48980475cab17f9fae028e05e4eb0d38999c`). The episode factory (`engine/dope-ffi` ABI 4) generates fresh duplicate-grouped random splits and extracts FIT-only evidence in parallel with zero per-table caching. The `DSE8` v1 wire uses nested ascending TLVs, CRC32, and `TAG_WEIGHTS_FP` inside `DPA1` tag 8; legacy Format 1, `DPK3.2`/v1, and `DSE7` (tag 7) remain bit-identical. Native `dope_kpi` and `contracts/dope-kpi.json` (`200ea0ed1ba2...`) govern DEV-SELECT scoring.

| ID | Tips | Proposal | Owner | State | Measured run, metric, 95% CI and DEV test |
| --- | --- | --- | --- | --- | --- |
| V3-01 | RichStats §3, Catalog §3, Evidence §1.2 | FP-0 lattice-safe jitter, per-episode masked marginal/correlation/SWD without 26-row truncation, duplicate-group splits and unseen multiclass refusal | Antigravity | TESTING | Focused `test_fp0_loop.py` 2/2 passed on `b8a0037c` (`fp0-v8-losses.log` `42eea45c`, 7 cases combined with the V8 loss suite). Matched-exposure repaired-vs-prior V7 DEV ablation open. [Repaired V7](#repaired-v7) |
| V3-02 | Evidence §3, DFP1 §3, RichStats §5 | FIT-only `EvidenceInput` with no query fields, stable `RowIdentity`, grouped folds and zero per-table cache | Claude | TESTING | `engine/dope-core/src/evidence/` + `engine/dope-ffi` ABI 4 (`4c7b16a0`): 12/12 `dope-ffi` query-poison, tied-pair permutation, and grouped-fold tests pass; DSE8 packet-fit row order queued post-release; matched native V8 evidence ablation open. [Extraction identity](#extraction-identity) |
| V3-03 | Evidence §4.1, Catalog §4.3, RichStats §8.3 | Value/mask pairs, finite-sample support and uncertainty with `k >= 10` floor | Antigravity | TESTING | `contracts/evidence-layout-v1.json` (`e99647eb`) & `evidence_v8` (5/5 passed at `7968ea7f`); mask-only vs full-evidence DEV ablation open. [Evidence bundles](#evidence-bundles) |
| V3-04 | DFP1 §4.3, Evidence §4.2, Compiler §2.1 | FIT-only typed metadata in `Nc2ColumnMetaV2`, `Dse8ColumnMeta` and `fit_unit_meta` | Claude | TESTING | DEV-lite E2 (`primary-dse8-p0.json` vs `primary-dse8-p0-nometa.json`): −0.55546 vs −0.55687 (+0.00142, both 1,496 B med); Wide DEV pre-fix (`primary-dse8-p0.json` `d79f5cbf` vs `primary-dse8-p0-nometa.json` `9aff70e8`): −0.38216 vs −0.38024 (neutral at Step-0); wired identically into `auto/v8.rs` & `auto/v8_dpa.rs` at `7968ea7f`. [DEV receipts](#dev-select-receipts) |
| V3-05 | Evidence §4.2, DFP1 §4.3, Catalog §6.2 | T0 marginal evidence in `COL[0..38]` (lattice, heaping, atoms, zeros, modes, tails, L-moments, skew, kurtosis, missingness) | Antigravity | TESTING | `support_marginal.rs`, `rank_stats.rs`, `evidence_v8` (5/5 passed at `7968ea7f`); matched-compute T0 block DEV ablation open. [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| V3-06 | Evidence §4.2, Compiler §3.1, Catalog §6.4 | T1 dependence in `COL[38..68]` & `EDGE[0..16]` (rank ties, Chatterjee ξ, dCor, Clayton/Gumbel tails, collinearity, signed XOR/XNOR) | Antigravity | TESTING | `dependence_pairs.rs`, `target_interactions.rs`, `evidence_v8` (5/5 passed incl. XOR vs XNOR sign resolution); paired DEV block ablation open. [Dependence and regimes](#dependence-and-regimes) |
| V3-07 | Catalog §6.5, RichStats §8.1–8.2, Evidence §4.2 | T1 OOF IRLS, stump gain, quadratic interaction, residual variance, class-conditional moments and group summaries (`COL[68..112]`, `CLASS`, `CLASS_COL`, `GROUP`) | Antigravity | TESTING | `irls.rs`, `target_helpers.rs`, `target_interactions.rs` (`evidence_v8` 5/5, `4c7b16a0` grouped folds); leave-one-block-out DEV ablation open. [Evidence bundles](#evidence-bundles) |
| V3-08 | Evidence §4.2 & §4.5, DFP1 §4.6, Catalog §6.8 | T2 depth-3 GBDT split/permutation witnesses, pairwise Walsh-Hadamard fANOVA, tree-vs-linear lift and learner spread | Antigravity | TESTING | `witness.rs` (`evidence_v8` 5/5 planted mechanisms + `4c7b16a0` grouped folds); T2-over-T0/T1 DEV ablation open. [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| V3-09 | Evidence §4.5, Catalog §6.9, Compiler §3.3 | T2 PCA decay and effective rank, local intrinsic dimension, kNN margins, duplicate ratio, and FIT-half DCR | Antigravity | TESTING | `geometry_privacy.rs` (`evidence_v8` 5/5); matched-byte T2 realism/privacy DEV ablation open. [Realism and privacy](#realism-and-privacy) |
| V3-10 | Evidence §4.6, DFP1 §4.2 | T3 closed-loop packet PIT and discrepancy features gated by ephemeral owner-side `PacketProbe` | Antigravity | TESTING | `closed_loop.rs` masks absent probe coordinates (`evidence_v8` 5/5); matched-compute T3-vs-T0–T2 DEV ablation open. [Closed-loop evidence](#closed-loop-evidence) |
| V3-11 | Evidence §5, Runbook §4, Compiler §5 | DSE8 v1 nested-prefix raw and normalized wire, ascending TLV, CRC32, `TAG_WEIGHTS_FP`, dimension refusals and adaptive budget planner (512–65,536 B) | Antigravity | TESTING | `single_encoding_v8@7968ea7f` (5/5 passed, 12 malformed vectors refused) + `v3-report-dev-s5-8@3fb802fd` (`primary.json` `55974867`): `dse8-p0-v3b` 0/44 failed, 868/2,048 B; `dope-v3b-auto-v8` 0/44 failed, 1,787/2,048 B, 108/108 SHA-identical packets. Multi-rung (512/1024/4096 B) frontier ablation open. [Packet bytes and layout](#packet-bytes-and-layout) |
| V3-12 | Evidence §5.1, DFP1 §2, Runbook §4 | Typed marginal decoding and exclusive nominal one-hot groups with `TAG_NOMINAL` | Antigravity | TESTING | `marginals.rs`, `nominal.rs` (`single_encoding_v8` 5/5: 0 illegal multi-hot/all-zero rows across 5,000 draws); equal-byte typed-X-only ablation open. [Typed rows and marginals](#typed-rows-and-marginals) |
| V3-13 | Evidence §5.1, Compiler §4.3–4.4, Runbook §4 | Calibrated analytic Step-0 (`calibrate.rs`, `effects_v8.rs`, `target_v8.rs`, `trees_v8.rs`, `copula_regimes.rs`, `irls.rs`): tie-interval binning, OOF-CV IRLS, deflated sequential boosting, 512-row synthetic-X intercept calibration, unit-variance regimes | Antigravity | TESTING | Calibrated `dse8-p0-v3b` (`7968ea7f`): DEV-lite `7e01e049` −0.27758 [−0.46536, −0.12647] (+0.27788 over pre-fix −0.55546); Wide DEV `c63b2fd2` −0.18968 [−0.36193, −0.04685] (+0.19248 over pre-fix −0.38216; beats `2.0.0-auto` −0.20726 on point); Fresh DEV Report `55974867` −0.342 [−0.652, −0.103] (868/2,048 B, 0/44 failed). Section-level ablation (`no_trees`/`no_regimes`) and v3.1 Ledoit-Wolf + sparse-tail MLP fixes open. [DEV receipts](#dev-select-receipts) |
| V3-14 | Runbook §5.4–5.5, Compiler §4.3 | Optional skew-t copula and target-tail noise (`FLAG_HAS_SKEW_T`) | Antigravity | TESTING | `Dse8FitOptions::enable_skew_t` defaults off (`false`); equal-byte opt-in DEV ablation open. [Dependence and regimes](#dependence-and-regimes) |
| V3-15 | Evidence §6, DFP1 §5, Runbook §5, Compiler §4 | Pre-bottleneck `SingleEncV8`, split-QKV `PairSplitAttnBlock`, nested 4-bit FSQ, row-local RealNVP, entmax Soft-NODE, rank-4 bilinear and trust gate | Antigravity | TESTING | Parity: `codex-hard-parity@8f4b4b92` (12/12 ≤ 1e-5) & `test_v8_rust_parity@7968ea7f` (13/13 ≤ 1e-4). `v8-r1` rejected on Wide DEV (`baca818c`: −0.01142 / −0.03008 vs `dse8-p0`). **`v8-r2-k001000`** (`w0 EMA` sha `99210221`, emitted via `dope_v8_emit-4b68275e`, `primary-v8-r2-k001000.json` `27074a87`, 12-cell DEV-lite screen only): 4× TRG **−0.25367 [−0.42798, −0.11669]** (+0.02391 4× & +0.12250 1× over `dse8-p0-v3b` −0.27758, MLP −0.224, C2ST 0.960, 834 B med, 0 failed); decision requires paired Wide-DEV (`≥ +0.010` over `dse8-p0-v3b` with 2.5% bound `> 0`, Addendum B). [DEV receipts](#dev-select-receipts) · [Hard forward](#hard-forward) |
| V3-16 | Evidence §7, Compiler §6, Runbook §6 | Twin-referenced smooth tanh-saturated linear/tree/RFF retention gap, direct conditional NLL, anchor, PIT, copula, tree, driver, DCR privacy hinge, Gumbel-Rao K=4 STE, `--grad-accum-steps`, width curriculum, optional PCGrad/ASAM | Claude | TESTING | Focused `test_v8_losses.py` 5/5 parametrized cases passed on `b8a0037c` (`fp0-v8-losses.log` `42eea45c`). The 8-episode overfit control reduced `l_ret` from 0.87 to 0.51. `v8-r2` k1000 scored −0.25367 on DEV-lite (`27074a87`). Full objective ablations and wide-DEV pairing remain open. [Hard-transfer objective](#hard-transfer-objective) · [Gradient methods](#gradient-methods) |
| V3-17 | Evidence §6.4, RichStats §7.12 | Owner-side per-table test-time gradient code refinement (TT-Opt at `fit`) | Claude | NOT STARTED | Deferred post-v3.0 until amortized `SingleEncV8` clears paired Wide-DEV lift; not run. [Closed-loop evidence](#closed-loop-evidence) |
| V3-18 | DFP1 §4.6, Catalog §6.10 | Expanding core v3 to temporal/panel autocorrelation shifts and permutation recipes F133–F144 | Owner | NOT STARTED | Out of v3 scope (standard `dope-kpi/v2` tables lack universal time keys; permutation p-values exceed <25 ms/ep budget) — never run. [Catalog](#catalog) |
| V3-19 | Earlier caching proposals in the tips | Per-table caches for landmarks, evidence, column metadata or teacher outputs | Owner | REJECTED | Run `c6-factory-bench@075b6752` (256 eps/s, 10–12 ms/ep, 0/1,440 failures) + Run `dope-ffi@4c7b16a0` (12/12 query-poison & permutation tests passed with 0 byte diff), proving uncached FIT-only extraction satisfies throughput without split leakage (09:50Z binding rule). [Extraction identity](#extraction-identity) |
| V3-20 | Earlier product and objective proposals | Python, ONNX or LibTorch in native product serving, or training against the held-out MLP auditor | Owner | REJECTED | Run `ops/ci/engine.sh@3fb802fd` (655/655 passed) + static Rust binary `58314f0a` (108/108 packets, zero Python linkage) + Run `compare-v3b-guards.json` (`782376d3`: independent held-out MLP caught Δ = −0.211 on `0a64bb31` missed by linear/GBDT, proving holdout diagnostic necessity). [Recipient behavior](#recipient-behavior) · [Hard-transfer objective](#hard-transfer-objective) |
| V3-21 | Evidence §2.2, C1 | Native E0 marginal, copula and conditional attribution oracles (`oracle_*` arms) | Claude | TESTING | #400 (`a3e9a986`) added 8 `oracle_*` arms + factorial reducer tests; full native DEV factorial sweep open. [E0 attribution](#e0-attribution) |
| V3-22 | Compiler §5, `auto::v8` guard | Target-stratified 2-fold grouped 3-way `auto::v8` guard, `DPA1` tag 8, and `--format auto-v8` / `v8-analytic` / `v8` CLI routes | Antigravity | TESTING | **Pre-reg DEV PASS, contract HOLD (analytic fallback approved with HOLD disclosed):** DEV-lite `fda33820` −0.11872 [−0.18782, −0.05313] (1,651 B, 0 fail); Wide DEV `72a5347d`/`782376d3` −0.17048 vs `2.0.0-auto` −0.20726 (Δ **+0.03678 [+0.00839, +0.06234]**, Holm p = 0.0047, contract HOLD: reg CI −0.034, MLP −0.211, privacy −0.066); Fresh DEV Report (`primary.json` `55974867`, `compare.json` `5ac5a9bd`, seeds 5–8, 44 informative headline cells / 11 families): **−0.121 [−0.189, −0.049]** vs `2.0.0-auto` −0.181 [−0.280, −0.085] (Δ **+0.060 [+0.009, +0.114]**, Holm p = 0.0091, 1,787/2,048 B, 0/44 failed, 108/108 SHA-identical; contract HOLD: task −0.053, MLP −0.007, realism −0.111, privacy −0.031). [DEV receipts](#dev-select-receipts) |

### DEV-SELECT receipts

#### 1. Fresh DEV Report Look (`bench/real-world/results/scoreboard/v3.0.0/`, seeds 5–8, scored once)

Unseen data seeds 5–8 on the 22 DEV tables (108 total scheduled cells; 64 headline scheduled / **44 informative headline cells** across **11 informative families**; scored once after selection froze under Addendum C; no CONFIRM/HOLDOUT/SEALED data). Static `dope 3.0.0` (`3fb802fd`, binary SHA-256 `58314f0a...`) reproduced **108/108 packets with identical SHA-256**.

- **Receipts:** `primary.json` (SHA-256 `55974867a938ff03c9c12108c9bc054c0dd40a6e2c6691e32a8b2c074c8eefba`), `compare.json` (SHA-256 `5ac5a9bd0d49cef9454da192ef7adee969d0788b2d01fa76000532f6c3c36c49`).

| Arm | Artifact class | Bytes (Med / Max) | Fresh DEV 4× TRG [95% CI] | Reg / Bin / Mc | Held-Out MLP | C2ST | Failed / cells |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| **`dope-v3b-auto-v8`** (`dope 3.0.0 --format auto-v8`) | source_free | 1,787 / 2,048 | **−0.121 [−0.189, −0.049]** | −0.137 / −0.136 / −0.089 | −0.349 | 0.967 | 0 / 44 |
| `dope-2.0.0-auto` | source_free | 1,817.5 / 2,048 | −0.181 [−0.280, −0.085] | −0.118 / −0.278 / −0.146 | −0.341 | 0.958 | 0 / 44 |
| `dope-2.0.0-small` | source_free | 1,508 / 2,043 | −0.267 [−0.402, −0.133] | −0.250 / −0.438 / −0.112 | −0.415 | 0.925 | 7 / 44 |
| `dope-1.3.0-small` (signed) | source_free | 1,508 / 2,043 | −0.266 [−0.399, −0.133] | −0.249 / −0.438 / −0.109 | −0.415 | 0.925 | 7 / 44 |
| `dse8-p0-v3b` (analytic alone) | source_free | 867.5 / 2,048 | −0.342 [−0.652, −0.103] | −0.094 / −0.276 / −0.655 | −0.491 | 0.977 | 0 / 44 |
| `pr390-v7-step0` | source_free | 782.5 / 2,026 | −0.432 [−0.585, −0.273] | −0.102 / −0.545 / −0.648 | −0.344 | 0.986 | 0 / 44 |
| `copy_a` / `bootstrap_a` (controls) | control | — | −0.039 [−0.069, −0.014] / −0.088 [−0.147, −0.036] | — | −0.014 / −0.145 | 0.530 / 0.572 | 0 / 44 |
| `indep_marginals` / `label_shuffle` (nulls) | control | — | −1.222 [−1.387, −1.095] / −1.164 [−1.269, −1.063] | — | −1.406 / — | 0.967 / 0.687 | 0 / 44 |

- **Paired comparison (`compare.json` `5ac5a9bd`, `dope-v3b-auto-v8` vs `dope-2.0.0-auto`):**
  - **4× ΔTRG** **+0.060 [+0.009, +0.114]**, Holm `p = 0.0091` (m=1). **1× matched ΔTRG** +0.024 [−0.020, +0.071]. In the sign-flip calibration, 0/2,000 passed the full guarded rule and 260/2,000 (13%) passed the statistical conditions alone, above the 5% tolerance.
  - **Per-task ΔTRG:** binary `+0.141 [+0.000, +0.287]`, multiclass `+0.057 [+0.000, +0.154]`, regression `−0.019 [−0.053, +0.000]`.
  - **Contract outcome: `HOLD`** (disclosed in the authorized analytic fallback): passed `ci_low`, `holm_p`, `min_delta`, `exact_copy_excess = 0.0000` (≤ 0.005), `bytes = 2,048 B` (≤ 2,048 B); failed `task_ci_low = −0.053` (> −0.02), `mlp_delta = −0.007` (> 0.0), `realism_coverage_delta = −0.111` (> −0.03), `privacy_coverage_delta = −0.031` (> −0.03).

#### 2. Wide DEV Selection Runs (`xbabe6:~/dope-data/kpi-runs/v3-dev-wide/`, seeds 1–4, 46 informative / 11 families)

| Run / Arm | Median bytes | Wide DEV 4× TRG [95% CI] | Paired ΔTRG vs Baseline [95% CI] / Holm p | Outcome & Receipt (SHA-256 prefix) |
| --- | ---: | --- | --- | --- |
| `dope-2.0.0-auto` (baseline) | 1,885 | −0.20726 | — | `primary-dope-2.0.0-auto.json` |
| `dope-v3-auto-v8` (pre-fix 75/25) | 1,885 | −0.23393 | −0.02667 [−0.06788, +0.00750], p=1.0 vs `2.0.0-auto` | **REJECTED** (`compare-auto-v8-vs-2.0.0-auto.json` `24a09277`) |
| **`dope-v3b-auto-v8`** (`7968ea7f` 2-fold) | 1,814 | **−0.17048 [−0.26531, −0.08435]** | **+0.03678 [+0.00839, +0.06234]**, p=0.0047/0.0094 vs `2.0.0-auto` | Pre-reg PASS / contract HOLD (reg CI −0.034, MLP −0.211, priv −0.066; `compare-v3b-vs-2.0.0-auto.json` `72a5347d`, `compare-v3b-guards.json` `782376d3`, `primary-dope-v3b-auto-v8.json` `8375116c`) |
| `dse8-p0` (pre-fix analytic) | 1,511.5 | −0.38216 [−0.62519, −0.17441] | −0.17490 point vs `2.0.0-auto` | **REJECTED** (`primary-dse8-p0.json` `d79f5cbf`; nometa −0.38024 `9aff70e8`) |
| **`dse8-p0-v3b`** (`7968ea7f` calibrated) | 866 | **−0.18968 [−0.36193, −0.04685]** | +0.01758 point vs `2.0.0-auto` (+0.19248 over pre-fix) | TESTING (`primary-dse8-p0-v3b.json` `c63b2fd2`) |
| `v8-r1` step 10,000 (pre-fix packets) | 1,557.5 | −0.39357 | −0.01142 [−0.06422, +0.02764] vs `dse8-p0` | **REJECTED** (`compare-v8r1-vs-dse8-p0.json` `baca818c`) |
| `v8-r1` step 6,000 (pre-fix packets) | 1,557.5 | −0.41223 | −0.03008 [−0.07049, +0.00002] vs `dse8-p0` | **REJECTED** (`compare-v8r1-vs-dse8-p0.json` `baca818c`) |

#### 3. DEV-Lite Screening Runs (`xbabe6:~/dope-data/kpi-runs/v3-dev-lite/`, 16 scheduled / 12 informative / 10 families)

| Arm | Artifact class | Median bytes | DEV-lite 4× TRG [95% CI] | Reported 1× TRG | Failures at 4× | Receipt (SHA-256 prefix) |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `dope-2.0.0-auto` | source_free | 1,810.5 | −0.22458 [−0.42249, −0.08737] | −0.379 | 0 | `primary-dope-2.0.0-auto.json` |
| `dope-2.0.0-small` | source_free | 1,573.5 | −0.23863 [−0.44260, −0.07269] | −0.359 | 1 | `primary-dope-2.0.0-small.json` |
| `pr390-v7-step0` | source_free | 782.5 | −0.41078 [−0.58096, −0.24801] | −0.593 | 0 | `primary-pr390-v7-step0.json` |
| `dse8-p0` pre-fix | source_free | 1,496 | −0.55546 [−0.92725, −0.24889] | −0.641 | 0 | `primary-dse8-p0.json` |
| `dse8-p0-nometa` pre-fix | source_free | 1,496 | −0.55687 [−0.92866, −0.25189] | — | 0 | `primary-dse8-p0-nometa.json` |
| `dope-v3-auto-v8` pre-fix | source_free | 1,810.5 | −0.18514 [−0.30488, −0.08013] | −0.271 | 0 | `primary-dope-v3-auto-v8.json` |
| `v8-r1` steps 1k / 2k / 4k / 6k / 8k / 10k / 12k | source_free | 1,542 | −0.5398 / −0.6036 / −0.5478 / −0.5398 / −0.5445 / −0.5288 / −0.5457 | — | 0 | `primary-v8-r1-k001000..k012000.json` |
| **`dse8-p0-v3b`** (`7968ea7f` calibrated) | source_free | 788 | **−0.27758 [−0.46536, −0.12647]** | −0.53577 | 0 | `primary-dse8-p0-v3b.json` (`7e01e049`) |
| **`dope-v3b-auto-v8`** (`7968ea7f` 2-fold) | source_free | 1,651 | **−0.11872 [−0.18782, −0.05313]** | −0.30870 | 0 | `primary-dope-v3b-auto-v8.json` (`fda33820`) |
| **`v8-r2-k001000`** (`w0 EMA` `99210221`, `dope_v8_emit-4b68275e`, MLP −0.224, C2ST 0.960) | source_free | 834 | **−0.25367 [−0.42798, −0.11669]** | **−0.41327** | 0 | `primary-v8-r2-k001000.json` (`27074a87`, 12-cell screen only; paired Δ +0.02392 [−0.00738, +0.05805] HOLD) |
| **`v8-r2-k001500`** (`w0 EMA` `0b4f1889`, Codex paired DEV-lite vs `dse8-p0-v3b`) | source_free | 834 | **−0.25065** (paired Δ **+0.02694 [+0.00949, +0.04570]**, Holm p=0.00010, HOLD on task −0.031 / MLP −0.028) | — | 0 | `daa73595` / paired `ced92a36` (12-cell screen only) |
| `v8-r2-k002000` (`w0 EMA`, Claude DEV-lite screen) | source_free | 834 | −0.285 [−0.446, −0.155] (regressed below `dse8-p0-v3b` −0.278 under gate collapse + dead `bilin`) | — | 0 | Claude `16:50Z` DEV-lite screen |
| `v8-r2-k003000` (`w0 EMA` `b2020dfe`, Codex) | source_free | 834 | −0.27431 [−0.43568, −0.14776] | −0.53288 | 0 | `0ec3d846` / paired `64041c43`, **HOLD**. Δ +0.00327 [−0.04380, +0.03293] vs `dse8-p0-v3b`, Holm p=0.35286, task bound −0.15630 and MLP Δ −0.02109. |
| `v8-s7w-k001500` (`w1 EMA` `c8718c46`, Codex) | source_free | 834 | −0.45945 [−0.70110, −0.25290] | −0.63114 | 0 | `f5d1b5c9` / paired `8fdcd6ea6`, **REJECT**. Δ −0.18186 [−0.39764, −0.01582] vs `dse8-p0-v3b`, Holm p=0.99900, task bound −1.04380 and MLP Δ −0.30980. The declared surgery finalist rule uses w0. |

The k3000 and w1 k1500 comparisons share the frozen DEV-lite schedule, real
reference and analytic baseline from the r2 k1500 comparison. Both passed the
2,048 B maximum. The paired screen leaves realism, empirical privacy and exact-copy
guards unevaluated. Full receipt digests are recorded in `agent_chat.md` at 19:08 UTC.

## Measured DEV tests and outstanding comparisons

### Baseline replay

Use the same declared target, canonical view, row roles, physical families, rows, draws, seeds and audited ledger keys for all arms. Historical public-validation replay covers shipped DOPE 2.0.0 small, SDV GaussianCopula and SDV TVAE at 4× and 1×. At 4× the paired leads were +0.454 and +0.838 TRG on 125 common informative cells. Historical comparator claims stay within this selection-exposed cohort. Re-run the released V7.1 weights, analytic V7 step zero, Format 2, digest-verified H2 challenger if available, and DSE8 with a common DEV declaration before any new claim. [Published replay, inputs and digests](verification/published-method-benchmark.md)

### E0 attribution

Pair packet X / packet Y, real FIT X / packet Y, packet X / FIT small-tree Y, and real FIT X / FIT small-tree Y. Report both main effects and their interaction by task and at 1× and 4×. Include exact-marginal rank-12 and full-rank arms, real twin, GAM and a stronger FIT teacher on the same rows and seeds. The 12-cell native DEV-lite screen used schedule `9207d157` and four oracle arms on the same splits. The 4× TRGs were −0.40436 for packet X/packet Y, −0.48115 for real FIT X/packet Y, −0.28845 for packet X/tree Y and −0.13637 for real FIT X/tree Y. Independent factorial audit `593624da54974f8b5ecc2e7dac12d4a2b200d23560b45a05aa213448837dfe69` measured X main +0.03764, Y main +0.23035 and interaction +0.22887 at 4×. The corresponding 1× effects are +0.15219, +0.28025 and +0.06390. Real FIT X supplied to the sampler is an oracle input. These joint effects motivate comparison of X and conditional representations under the complete-byte cap. The full wide-DEV matrix stays TESTING. [E0 run and oracle identities](verification/README.md#native-receipt-audit)

### Failure-floor analysis

Retain failed entries and compare v2’s constant against `clip_low − R_B` at the same keys. Record unclipped and clipped outcomes, then test null, twin, deliberate failure and single-class controls. Historical public replay preserves the direction of the two SDV comparisons under both changes. Measure a v3 DEV promotion sensitivity with the paired bootstrap and its guards before considering a successor. [Historical sensitivity](verification/published-method-benchmark.md#verified-public-validation-replay)

### Repaired V7

Pair V7 step zero, released V7.1, prior versus repaired old-objective training, and the separately identified H2 challengers. Compare native pack→parse, round-trip row identity, lattice-safe augmentation, per-episode row and column masks, and full-query vocabulary refusal. Measure proxy-to-hard gap, actual row/exposure counts, per-term gradients and native ΔTRG before changing objectives. The retained round-108 screen reconstructed −0.004245 4× ΔTRG [−0.045155, +0.039275] on 12 cells against `v72-step0`. The prospective repaired V7 DEV comparison is open. [Pinned H2 replay](verification/README.md)

### Extraction identity

Use declared target-last intake, stable physical row/group IDs, FIT-only transforms, and fits on fold-training complements for supervised summaries. Poison query X and labels separately and poison every other non-FIT role. Test duplicate groups, unsupported vocabularies, equal fingerprints, column and sign permutations, class relabeling, rank ties, constants, all-missing inputs, and measured zero versus unavailable evidence. The V7 factory passed query-poison fixtures. DSE8 candidate `8f4b4b92` changed evidence under a 256-row physical permutation. Claude commit `4c7b16a0` (ABI 4 `dope_ffi_v8_episode` + `evidence::extract_with_rows`) landed stable `RowIdentity` tie-breaking and grouped T1/T2 folds, passing 12/12 `dope-ffi` tied-pair permutation, duplicate-ID refusal, and query-poison tests; DSE8 packet-fit row order (`single_encoding/v8/mod.rs:281`) is queued post-release. [Candidate probe](verification/README.md#candidate-replay-at-8f4b4b92)

### Packet bytes and layout

Measure complete wire, framing, decoder identity, mandatory typed/group sections, optional sections, unused bytes, refusals, checksum and sidecars at 512, 1,024, 2,048 and 4,096 B. Exercise 9/10 support, 48/49 slots, eight/nine classes, maliciously valid CRC, exact bytes and truncation from one legal base. Compare fixed order against FIT-estimated utility per serialized byte at equal complete bytes. Candidate `8f4b4b92` passed 11 malformed probes, emitted 3,157 B at the 4 KiB wide-eight-class point, had a 2,119 B legal base, refused 2 KiB on that fixture and emitted 3,199 B for 49 slots where the shape should refuse. 16/64 KiB are DEV representation probes. Antigravity commit `7968ea7f` corrected headline-budget planning (`single_encoding_v8` 5/5 passed; `v3-report-dev-s5-8@3fb802fd` `55974867`: 108/108 `auto-v8` and `dse8-p0-v3b` packets ≤ 2,048 B with 0/44 failures). The exact-main `b8a0037c` research emitter fit the 48-slot, eight-class fixture at 2,046 B. It still accepted 49 slots at a 4,096 B cap, and truncation of the 4 KiB fixture to 2 KiB refused because its legal base took 2,119 B. [Closeout evidence](RELEASE_CLOSEOUT.md#evidence-currently-staged) [Exact byte evidence](verification/README.md#candidate-replay-at-8f4b4b92)

#### DSE8 fit ablations on DEV-lite

Antigravity evaluated the variants on the 12 informative cells in `schedule-lite-dev.jsonl` (`9207d157`). The analytic `dse8-p0-v3b` control scored −0.27758 at 4× with 788 B median. The decisions apply to the proposed default changes. Receipt values are paired against that control.

| Proposed change | Owner | State | ΔTRG [95% CI] | Held-out MLP Δ | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| Remove trees | Antigravity | REJECTED | −0.03603 [−0.08239, +0.00349] | −0.05224 | `compare-dse8-no-trees.json` `0b1467a0`, HOLD; 724 B median |
| Remove residual quantiles | Antigravity | REJECTED | −0.00197 [−0.00569, −0.00001] | −0.00546 | `compare-dse8-no-resid.json` `c4b2e788`, REJECT; 779.5 B median |
| Remove FIT column metadata | Antigravity | REJECTED | −0.00074 [−0.00231, +0.00010] | −0.01773 | `compare-dse8-no-meta.json` `96fdc3fc`, HOLD; same 788 B median |
| Remove fitted effects | Antigravity | TESTING | −0.00053 [−0.01579, +0.01389] | −0.00316 | `compare-dse8-no-effects.json` `a8cfb533`, HOLD; 856.5 B median |
| Remove regimes | Antigravity | REJECTED | +0.01716 [−0.00067, +0.03692] | −0.06831 | `compare-dse8-no-regimes.json` `7e8286ea`, HOLD; 762 B median |
| Use skew-t by default | Antigravity | REJECTED | +0.00219 [−0.00707, +0.01128] | −0.02004 | `compare-dse8-skew-t.json` `13842a30`, HOLD; Holm p = 0.31757 |

The measured frontier scored −0.48726 at a 512 B cap (`2feb3149`), −0.31158 at 1,024 B (`9df05828`), −0.27758 at the 2,048 B control and −0.29114 at 4,096 B (`33c80378`). The three alternative cap comparisons were HOLD. These are DEV-lite frontier points.

### Typed rows and marginals

Pair typed-X and analytic V7 at equal complete bytes with the same conditional. Check lattice, ordinal/count bounds, heaping, atoms, zero inflation, legal one-hot groups, aligned and informative missingness, rare levels, all-missing and constant columns. Separate typed semantics, marginal precision, tail and modes, support masking and fair effect size. The pinned V8 core fixture passed nominal exclusivity and support canaries. Exact marginals in the tip’s preliminary public study moved feature C2ST 0.871 → 0.699 at the same dependence rank. Native DEV ablations and mixed legal-row coverage remain open.

### Dependence and regimes

Hold typed marginals and conditional fixed. Compare rank zero/low/full, exact-marginal rank-12/full-rank, rank ties with and without disattenuation, class means, shared-covariance regimes and any skew/tail/coupling variant individually. Require calibrated marginals after latent changes. Measure joint feature C2ST, class-conditioned structure, tail and missingness cells, transfer and serialized bytes. Full rank in the earlier preliminary study moved feature C2ST 0.699 → 0.679. No guarded native estimate for an optional dependence family is available.

### Conditional sections

Pair analytic ridge, logistic and multinomial conditionals with calibrated supported main effects, signed/exploratory pairs, depth-2/3 trees, class-temperature and residual scale/quantile variants. Use XOR/XNOR, U-shaped response, substitutes, signed pair negation, rare classes, heteroscedastic and heavy-tail cases, then measure native ΔTRG, calibration, minority loss and bytes by section. Test each addition at equal complete bytes. Planted fixtures and E0 code exist. Release DEV comparison remains open.

### Evidence bundles

Use the [stable recipe crosswalk](verification/evidence-crosswalk.md) and the approved 128/16/16/8/160 blocks plus a separate class-column block. Compare equal-capacity packet-only, FIT T0–T1, then T2/T3 one at a time with leave-one-family-out, mask-only, within-task shuffle, equal-width noise and matched owner-compute controls. Report rank/null calibration, support, presence, uncertainty validity, consumed coordinates, wall time and hard native gain. The draft layout passes dimension/mask fixtures. It still needs complete per-value support and uncertainty semantics. Individual recipe DEV admission remains open.

#### Input-feature screen F0

F0 `da813445` used the training producer to draw 2,000 distinct TRAIN-split FIT views across 1,834 families. Every episode recomputed its features from its own FIT rows. Replacing QRY features and labels with noise left all computed blocks bit-identical. The diagnostic is family-grouped five-fold out-of-fold ΔR² for predicting eight analytic step-zero loss components beyond the existing decoder inputs. Its 95% intervals use 1,000 family resamples of fixed out-of-fold predictions. The 48-feature noise control measured −0.01218 [−0.02041, −0.00205]. This measures TRAIN-split loss prediction. Feature admission for the release requires a separate native DEV transfer result.

The preregistered screen rejects a block if the CI upper bound is below 0.01, absolute redundancy with a baseline coordinate reaches 0.95, CPU time exceeds 2 ms per episode with ΔR² below 0.03, or the poison check fails. The baseline is retained. Of the added blocks, one advances to conditional testing, 21 are rejected and three await an extractor. The retained `f0_blocks.jsonl` is SHA-256 `36fd2535114bb36a9dea4b79bb438691888fdbce496325897deb00510d9e4946` and `f0_report.json` is SHA-256 `7507be70dc63ae6e906080c537982775923319b1bcb20eec6c47a194f29fe11e`. See the Claude F0 handoff in `agent_chat.md` for receipt locations.

| Feature block | Owner | State | ΔR² [95% CI] | CPU ms per episode | Max abs. ρ | F0 reason or dependency |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `BASELINE_DSE8_DECODER_INPUTS` | Claude | KEPT | +0.31391 [+0.28655, +0.35334] | 97.05 | 1.000 | Existing decoder inputs. This is absolute out-of-fold R² against an intercept. |
| `EV_COL.S_SUPPORT` | Claude | REJECTED | +0.00297 [-0.00424, +0.01045] | 3.49 | 0.954 | baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.M_QF_RESID` | Claude | REJECTED | -0.00743 [-0.01521, +0.00140] | 3.49 | 0.948 | CI upper bound +0.00140 < 0.01, cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.M_SHAPE_TAIL` | Claude | REJECTED | +0.00987 [-0.00001, +0.02178] | 3.49 | 0.848 | cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.M_MISS_UNC` | Claude | REJECTED | -0.00035 [-0.00788, +0.00796] | 3.49 | 1.000 | CI upper bound +0.00796 < 0.01, baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.M_PACKET_PIT` | Antigravity | NOT STARTED | — | — | — | The native extractor needs the T3 `packet_probe` input. Measurement is pending. |
| `EV_COL.D_DEP_PREC` | Claude | REJECTED | +0.00024 [-0.00815, +0.00998] | 65.96 | 0.975 | CI upper bound +0.00998 < 0.01, baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.R_UNIV_TARGET` | Claude | REJECTED | +0.03458 [+0.02521, +0.04621] | 65.96 | 0.992 | baseline redundancy ≥ 0.95 |
| `EV_COL.R_OOF_RESID_T03` | Claude | REJECTED | +0.00602 [-0.00255, +0.01587] | 65.96 | 1.000 | baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.W_GBDT_WITNESS` | Claude | REJECTED | +0.00039 [-0.00651, +0.00776] | 72.04 | 0.821 | CI upper bound +0.00776 < 0.01, cost > 2 ms with ΔR² < 0.03 |
| `EV_COL.W_CLOSED_LOOP_COL` | Antigravity | NOT STARTED | — | — | — | The native extractor needs the T3 `packet_probe` input. Measurement is pending. |
| `EV_EDGE.T0T1` | Claude | REJECTED | +0.00385 [-0.00478, +0.01300] | 65.96 | 0.833 | cost > 2 ms with ΔR² < 0.03 |
| `EV_EDGE.T2` | Claude | REJECTED | -0.00399 [-0.01011, +0.00254] | 72.04 | 0.588 | CI upper bound +0.00254 < 0.01, cost > 2 ms with ΔR² < 0.03 |
| `EV_GLOB.G_META_TASK` | Claude | REJECTED | -0.00808 [-0.01398, -0.00140] | 3.49 | 1.000 | CI upper bound -0.00140 < 0.01, baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_GLOB.G_SPEC_MP` | Claude | REJECTED | -0.00524 [-0.01207, +0.00289] | 65.96 | 0.909 | CI upper bound +0.00289 < 0.01, cost > 2 ms with ΔR² < 0.03 |
| `EV_GLOB.G_GEOM_PRIV` | Claude | REJECTED | +0.00744 [-0.00222, +0.01895] | 65.96 | 1.000 | baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_GLOB.G_TARGET_RESID_T03` | Claude | REJECTED | +0.02763 [+0.01876, +0.03882] | 65.96 | 0.805 | cost > 2 ms with ΔR² < 0.03. The owner encoder already consumes the T2 inputs. |
| `EV_GLOB.G_ALIGN_TWIN_STAB` | Claude | REJECTED | +0.01141 [+0.00328, +0.01985] | 65.96 | 0.877 | cost > 2 ms with ΔR² < 0.03. The owner encoder already consumes the T2 inputs. |
| `EV_GLOB.G_WITNESS_FANOVA_CAL_LC` | Claude | REJECTED | +0.02323 [+0.01509, +0.03139] | 72.04 | 0.906 | cost > 2 ms with ΔR² < 0.03. The owner encoder already consumes the T2 inputs. |
| `EV_GLOB.G_CLOSED_LOOP` | Antigravity | NOT STARTED | — | — | — | The native extractor needs the T3 `packet_probe` input. Measurement is pending. |
| `EV_CLASS_TOK` | Claude | REJECTED | +0.00063 [-0.00492, +0.00650] | 65.96 | 0.983 | CI upper bound +0.00650 < 0.01, baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_CLASS_COL` | Claude | REJECTED | +0.00001 [-0.00572, +0.00635] | 65.96 | 0.981 | CI upper bound +0.00635 < 0.01, baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `EV_GROUP_TOK` | Claude | REJECTED | -0.00283 [-0.00846, +0.00245] | 3.49 | 1.000 | CI upper bound +0.00245 < 0.01, baseline redundancy ≥ 0.95, cost > 2 ms with ΔR² < 0.03 |
| `NC2_COLUMN_META_V2` | Claude | REJECTED | -0.00497 [-0.01380, +0.00317] | 1.23 | 1.000 | CI upper bound +0.00317 < 0.01, baseline redundancy ≥ 0.95 |
| `LANDMARK_NATIVE23` | Claude | REJECTED | +0.02230 [+0.01450, +0.03256] | 248.67 | 0.733 | cost > 2 ms with ΔR² < 0.03 |
| `LANDMARK_BUILDING_BLOCKS` | Claude | TESTING | +0.05045 [+0.03963, +0.06534] | 8.19 | 0.874 | Passed all F0 rules. F1 requires an eligible neural arm and a matched three-seed native DEV ablation. |

`EV_COL.R_UNIV_TARGET` missed the block redundancy rule despite a positive interval. A coordinate-level redundancy screen remains a future experiment. `LANDMARK_BUILDING_BLOCKS` stays **TESTING** until the F1 transfer evidence exists.

### Shape and nonlinear statistics

Use matched low-order moment pairs, fitted-vs-exact quantile and support fixtures, signed/nonmonotone target functions, equal-means variance, mixture/tail and missingness contrast pairs. Compare each curve or cheap functional summary with and without extra scalars at fixed extract time, output capacity and bytes. Several native mechanisms have fixture coverage. No descriptor can inherit DEV lift from a different statistic or estimator.

### Tree witnesses and driver fidelity

Use cross-fitted linear, GAM and shallow-tree/boosted witnesses on FIT-only folds. Measure grouped permutation mass/rank, supported centered ALE, interaction and conditional effect checks with real–real normalization and correlated-substitute controls. DFP and evidence proposals for fANOVA, OOF residuals/disagreement, landmarks, learning curves, local geometry and finer feature groups need individual ablations. Preliminary public tree-witness agreement was 0.64 against 0.67 real-twin. Native DEV paired results are open.

### Closed-loop evidence

Generate fixed-seed analytical samples on the owner side of the current episode, bind the analytic packet identity, and use only FIT comparison rows. Compare low-cost T3 discrepancies against T0–T2 at equal bytes and owner compute. Probe joint and factorized discrepancies, retention, calibration, privacy, under/overspent bytes and uncertainty. No measured DEV result for T3 or owner refinement is available.

### Owner compiler

Require zero-init packed bytes and native sampled rows to match analytic step zero. Compare matched-size packet-only encoder, FIT evidence, random/noise/masked controls, bounded sender parameter corrections and only then a transmitted code. Compare FSQ/VQ/code/refinement with the same bytes spent on explicit statistics, trees, regimes or precision. Check source-free receiver and exact shared dependencies. V8 native fixture verified zero-init identity. No incremental DEV transfer win is established.

### Advanced decoder

Admit tree refiners, Soft-NODE, residual flows, shared-latent skew/tail models, FiLM/receiver weights and wider architectural variants only as individual matched-compute and matched-byte comparisons after analytic base admission. Verify class and lattice legality and conditional calibration before transfer. Verify latent PIT for changes that alter its marginals. There is no accepted DEV gain for these variants.

### Hard forward

Verify Rust pack→parse authority, SafeTensors F32 rank ≤2 inventory, parameter names/shapes, quantization fences, ordered knots and exact flags, masks and classes. Require float tolerance at most `1e-5`, row-local counter streams and product/native forward gap logging. The pinned candidate passed twelve tightened synthetic training references. Exact integration and hardware/runtime-specific qualification remain open. [Strict replay](verification/README.md#candidate-replay-at-8f4b4b92)

### Recipient behavior

Replay sample at a fixed seed across single, batch, chunk, and resumed range with detached owner inputs. Bind product CLI, decoder identity, artifact and declared runtime inventory. Run the packaged CLI under recipient isolation and verify redaction. A 797 B narrow DSE8 fixture passed deterministic repeat and first-row parity, and existing V7/DSE1 and DPK3.2/v1 compatibility checks passed. Full DSE8 range/package closure remains open.

### Hard-transfer objective

At fixed model and exposure compare original versus smooth worst-auditor hard-transfer proxy, proper conditional scoring, calibration, driver/ALE, gradient matching, likelihood and rate terms individually. Use linear/logistic and fixed-region tree proxies with untouched-query losses and real-A/null/twin references. Keep independent held-out MLP and CatBoost outside optimization. Compare actual native ΔTRG and calibration today, with proxy/float mismatch and failures retained. No prospective objective DEV matrix is available.

### Gradient methods

Begin with common random numbers, fixed projection banks, per-episode normalization, a fixed 256-episode probe, effective batch and split-noise measurement. Compare separate antithetic/stratified/low-discrepancy draws, Gumbel–Rao, PCGrad/conflict, GradNorm/EMA, SAM/ASAM and alternate optimizer or step schedules only after their diagnosed failure mode. Report interaction cosines, clipping, p99/median norms, loss variance, wall time and hard native effect at matched compute. No method-specific guarded DEV result is available.

### Episode stream

Use family-balanced fresh group-safe episodes, declared FIT/query and real-twin sizes, no row-derived table cache, explicit exposure counters, semantic augmentation and prior curriculum arms. Check zero-repeat versus bounded replay, actual real/prior shares, masks and label support, CPU producer latency/wait and fixed per-family sample counts. The #400 V7 factory handoff measured 97/167/256 episodes per second with one/two/four producers, zero failures over 1,440 episodes and a median 842 B packet. A V7 preset report records 75% real views per step over 640 slots. Integrated native V8 throughput and real-exposure measurements remain open. [Factory handoff](README.md#5-claudes-work-packages)

### Training operations

Use per-step atomic weights/EMA/optimizer/RNG/cursor/exposure/distributed checkpoint and exact interruption replay. Require a real-view 20-step smoke, 200-step interruption/resume comparison, hard export and three independent seeds. Record prespecified budget, phase fractions, gradient behavior, early stopping and seed selection. Candidate handoff reports a prior-only launch smoke and native parity. The approved real-view and interruption criteria still need their own receipts.

### Realism and privacy

Use marginal-only, feature-joint and joint-with-label boosted-tree C2ST plus twin reference, typed illegal-row checks, per-auditor calibration and class/minority losses. Track exact copy excess, DCR, membership/attribute attacks and empirical coverage with baseline controls and no raw sidecar values. Report whether lowered C2ST harms conditional/dependence fidelity. The historical public validation and synthetic fixtures are context. New matched DEV guard coverage is open.

#### Reused wide DEV tests for v3c

The ten guard comparisons for the four changes and `v3c-cand1` independently re-reduced from the retained score and guard ledgers. The shared schedule (`97439e8a`) has 46 informative 4× cells across 11 families. State reflects the component decision. A selected component still needs end-to-end release evidence.

| Proposal | Owner | State | ΔTRG versus v3b [95% CI] | ΔTRG versus v2.0.0 auto | Measured guard evidence |
| --- | --- | --- | ---: | ---: | --- |
| `fix1` OAS covariance shrinkage, tie damping and EB regimes | Antigravity | KEPT | +0.03856 [−0.00305, +0.11909] | +0.07534 [+0.00896, +0.16355] | `06d5bcb6` / `563bea8d`; all guards pass against v3b. Against v2, task −0.03356, MLP −0.18818 and privacy −0.05556 fail. Both comparisons are HOLD. |
| `fix2` sparse, zero-inflated and lattice marginal corrections | Antigravity | KEPT | +0.00011 [−0.00123, +0.00152] | +0.03689 [+0.00891, +0.06207] | `33892e65` / `4f596b0d`; held-out MLP improves +0.16554 versus v3b. Its delta against v2 is −0.04564. Both comparisons are HOLD. |
| `fix3` refit temperature on a narrow validation fold | Antigravity | REJECTED | −0.00654 [−0.01619, −0.00057] | +0.03024 [+0.00228, +0.05616] | `fcf884ec` / `ce5d67fe`; REJECT against v3b, with task lower bound −0.04286 and MLP +0.03233. |
| `fix4` increase the regression-only selection margin | Antigravity | REJECTED | −0.00123 [−0.00368, 0.00000] | +0.03555 [+0.00640, +0.06191] | `72de31cc` / `46ce6563`; HOLD. Task lower bound versus v2 is −0.04461 and the change misses incremental lift over v3b. |
| `cand1` combine fixes 1 and 2 with fit-size support scaling | Antigravity | TESTING | +0.03766 [−0.01232, +0.12228] | +0.07444 [+0.00835, +0.16766] | `37d347e9` / `25a45025`; HOLD against both. Versus v3b the task lower bound is −0.04147. Versus v2 the task, MLP and privacy guards fail at −0.03356, −0.17726 and −0.07639. |

The cand1 executable SHA-256 is `b6965c13a05cc7ac773d5798ada71f633d10829da326bc14f08053a91c044ee3`. Its artifact audit checked 108 complete packets per arm for cand1 and v3b, all at or below 2,048 B, with cand1 at 1,751.5 B median on the headline cells. The audit receipt is `064a5d18de3303f701b0b9b2266c3f88acaa26bd0a92b232bc430968b92fdf77`. Exact clean source and the fresh DEV decision are pending for a replacement RC.

### Published generators

Use authentic implementations for SDV/CTGAN, ARF, TabSyn, TabDDPM and TabDiff, with native-budget and valid byte-matched points, verified environment/recipe and full saved-state accounting. Preserve unsupported/timeouts in the ledger. Apply the same FIT teacher to qualified competitor features. The historical SDV native point measured GaussianCopula 105,848 B median and TVAE 404,722 B median. Three CTGAN smoke cells and a historical TabSyn comparison are narrower. The rest of the common DEV roster is open. [Source and recipe inventory](verification/published-method-benchmark.md#next-public-dev-roster)

### Power and promotion

Use physical family units within task, native failures, 10,000 paired family draws, Holm adjustment, 2,000 or more sign flips, task/MLP/privacy/realism/legality/byte guards and seed agreement. Confirm family-level noise and task power with a DEV simulation before selecting a recipe. The staged wide DEV schedule has 108 cells and 22 physical families overall, with 46 informative headline cells across 3 binary, 4 multiclass and 4 regression task-family units. At least 80% simulated power is the target. No current V8 keep or promotion receipt is available.

| Release gate | Owner | State | Evidence and remaining work |
| --- | --- | --- | --- |
| G10 formal protocol freeze | Jepson and custodian | NOT STARTED | Reserved for certification. The seeds 9–12 DEV look will bind the embedded candidate contract SHA-256 `200ea0ed1ba23a25be89de2dac2a4101656119514e0d69b4ff86a98fa4857623` in `PREREG_LOG` before scoring, with the schedule, DEV table set and scorer digest. The final-look receipt check requires that exact contract hash and `source: embedded`. The final look is pending. |

### T90 and D80

Freeze the loss, real-vs-real eligibility and driver components independently of TRG. For each measured DEV rung use raw held-out losses with the one-sided T90 harm bound, grouped importance and supported ALE/conditional drivers with real-twin D80 normalization. Keep both gate states, measured complete bytes and the passing bracket or unresolved state. The logged historical transfer threshold supplies no joint result. [Metric specification](../../metric.md)

### JopeDime and embedding

Bind version/layout/width and target-free masks for exported descriptors. Compare at equal reveal and owner cost, same candidate pool and history-kernel baseline. Test correct/shuffled task context, released DOPE, descriptors, measured program oracle and equal-capacity consumer. Require positive paired held-out-family program-quality gain before any new embedding training. Evaluate 100k versus compute-matched old-objective continuation, then gate 1M/3M/10M by incremental value and cost. No JopeDime DEV acceptance receipt is available.

### Execution scope

This ledger covers DEV only. The protected-campaign and JSR GPU suggestions are out of scope under the watcher instruction. Engineering verification runs on local bounded CPU. Fitted candidates and the final product decision remain with the existing owners and single integration branch. Source, module tests, public claim generation and release packaging remain separately auditable from a measured promotion decision.

## Source register

Each row records the tip locator, its proposal, responsible owner, current state and the applicable test above. For compact technical inventories a row lists every coordinate/recipe named in that source entry. Code/configuration blocks retain their source span and named keys so their exact settings can be recovered. Boundary decisions override conflicting draft proposals.

| Source alias | File under `tips/v3` | SHA-256 | Indexed suggestions |
| --- | --- | --- | ---: |
| Catalog | `DOPE_Dataset_Features_and_Transfer_Engineering_Spec.md` | `a8075975264260c90b24345e2938b87152d038f64d8c6cc131c7f0f3ae082daf` | 721 |
| RichStats | `DOPE_Dataset_Features_and_Transfer_Engineering_Spec (1).md` | `cb0a2c2e6e4c674a48bd3884a247fa7d7dba64edaf139d87688247dc744be69c` | 778 |
| DFP1 | `DOPE_FEATURE_PACK_V1_SPEC.md` | `b3fdfdb36bbc57c6c945b42cd9316f381813fad516644e7cdac2a4d3c33e7081` | 306 |
| Evidence | `DOPE_V8_EVIDENCE_SPEC.md` | `b3e984efefb96dc7a45ed90d5d83d11d3c1f122aab9caa37321abfc644be4c53` | 278 |
| Runbook | `dope_v8_engineering_agent_execution_spec.md` | `7cda5da0bbb2e8a71291c95919f921f23479261873d6bad85da5c68e41861826` | 386 |
| Compiler | `dope_v8_engineering_specification.md` | `cb92fc541e82f0d0a016cd409f192a7f1cdccaec9c5b3982b57f63ad1b132bcb` | 193 |

### Catalog

<details>
<summary>Source proposals and DEV test routing</summary>

| Source | Proposal | Owner | State | DEV test |
| --- | --- | --- | --- | --- |
|  | **1. The recommendation** |  |  |  |
| L13 | Build a **typed, uncertainty-aware statistical encoder of the normalized FIT dataset**, with explicit representations of columns, column pairs, conditional target functions, landmark errors, and stable subpopulations. … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L15 | The objective is useful information per byte and per second of preparation. A large flat list of test statistics is not enough. Every major new descriptor should answer a question the generator can act on: which … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L17 | The highest-priority work is: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L19 | **Make training consume the actual Rust-produced packet and normalization contract.** There are material source-level differences between the current Python extractor and native path. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L20 | **Add signed conditional target curves and explicit interaction surfaces.** Pure interactions are unidentifiable from the current V7 summaries in important cases. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L21 | **Reuse landmark out-of-fold predictions, not just aggregate gains.** Their residuals, disagreement, local failures, and response curves contain substantially more task information. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L22 | **Model missingness and nominal groups explicitly.** Arbitrary category integers, imputed values, and independent one-hot channels are inadequate representations of mixed data. | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L23 | **Train and select on hard-sampled, quantized, synthetic-to-real transfer across different learner families.** Direct target fit is an auxiliary objective; it is not the final product KPI. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L24 | **Preserve driver functions and correlated driver groups.** A single ranking of absolute linear coefficients cannot establish general feature-importance fidelity. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L25 | **Allocate bytes according to estimated downstream distortion.** Do not spend the entire budget on marginal knots while losing a crucial interaction or category. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L27 | The complete catalog in Section 6 contains **144 feature recipes**. Several expand into curves, matrices, classes, or multiple scales. This is a candidate registry for staged ablation, not a recommendation to enable … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **2.1 Main objective and important distinctions** |  |  |  |
| L35 | There are multiple implemented generations. Their results and conditioning features must not be mixed: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L39 | Original compiler / Format 1 — Explicit marginals, dependence, regimes, target models, trees, noise, quantization, and bounded fitting · Reuse established generator primitives where useful; an entirely new generator is unnecessary | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **3.1 A concrete impossibility example** |  |  |  |
| L99 | Take a balanced binary dataset with two binary features. Repeat each of the four patterns equally, with sufficient repetitions for the current support thresholds: | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L103 | −1 — −1 · 0 · 1 | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L104 | −1 — +1 · 1 · 0 | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L105 | +1 — −1 · 1 · 0 | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L106 | +1 — +1 · 0 · 1 | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L108 | Both datasets have the same feature marginals, covariance, class balance, and zero single-column target association. Their optimal additive logistic base has zero feature coefficients and intercept zero. The quartile … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L110 | The signed joint moment `E[x1*x2*(2*y−1)]` is **−1 for XOR and +1 for XNOR**. A 2×2 class-probability table also distinguishes them. This is a mathematical construction, not a new benchmark result. It shows why **signed … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **3.2 Findings from the current source** |  |  |  |
| L114 | These are static observations. Their contribution to measured model quality must be isolated experimentally. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L118 | Different discreteness rules — Python uses rounded values from up to 256 rows and `&lt;=16` unique values; native packet fitting uses the full scaled column and `&lt;=12` · One Rust implementation of type/support semantics; fixtures with 12, 13, 16, 17 categories and rare later rows | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L119 | Different knot quantization — Python rounds low and high endpoints to f16; Rust rounds low and span · Training must consume dequantized native bytes, not an approximate numerical mirror | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L120 | Different normal scores — Python uses midpoint ranks for discrete atoms and then standardizes; Rust uses its conditional normal-score procedure with tail bounds · Verify ties, atoms, covariance, and packet outputs; use native-produced tensors | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L121 | Normalization naming hides different transforms — Python `_fit_only_scale` calls its robust asinh/clipped transform “NC2-compatible”; canonical `nc2-bulk-tails-v1` uses a different bulk/tail mapping · Bind exact transform identity; do not interchange these coordinate systems | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L122 | Packet accounting differs — Python estimates packet bytes; Rust chooses layouts with column maps, dropped-column state, class tables, CRC and scale information · Byte caps and training layout decisions must come from actual serialization | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L123 | Column selection differs — Python extractor truncates to the first 48 after episode selection; native scale-card fitting can rank columns by absolute target correlation under width pressure · Train on the same selector; replace label-number correlation for nominal targets; protect pure interactions | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L124 | Category information is flattened — V7 raw intake converts nominal levels to numerical indices; packet inference operates on numerical columns · Use categorical group/type tokens and categorical sampling, not Euclidean meaning assigned to arbitrary IDs | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L125 | Missingness is lost on V7's compact path — Median imputation enters the packet; `SingleEncV7Packet` has no explicit missing-mask generator fields · Preserve observed/missing state and conditional missingness in a new format | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L126 | Class overflow can change the target — Native scale card keeps a bounded class list; packet mapping uses a fallback index for absent classes · Explicitly reject unsupported classes or use a declared, evaluated pooling policy; never silently relabel them as class 0 | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L127 | Small duplicate-group count permits row splitting — Episode builder falls back to random row splitting when unique duplicate groups are below 16 · Keep groups intact or mark episode ineligible; few groups are not permission to leak duplicates | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L128 | Unseen query classes can be remapped — Episode multiclass mapping uses `mp.get(..., 0)` · Preserve declared labels, use an explicit unseen-class policy, and record denominator consequences | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L129 | Training jitter ignores declared type in this path — Gaussian jitter is added to the selected feature matrix before packet extraction · Jitter only eligible continuous features; keep nominal, binary, integer, missing, and constraint semantics intact | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L130 | Input-size regime is narrow — Defaults crop FIT to roughly 32–384 rows and query to at most 128 · Add larger-FIT episodes and sample-size curves; do not infer large-data performance from small episodes | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L131 | Classification TSTR surrogate is mismatched — Ridge on soft class targets with probability clipping, plus optional hard-sample straight-through variant · Add proper differentiable logistic/softmax learners and exact hard-sample native evaluation | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L132 | Driver telemetry is narrow — Fast DEV evaluator ranks absolute ridge/logistic coefficients · Retain as a diagnostic, add grouped nonlinear importance and effect-curve fidelity | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L133 | Extra “tree-visible” loss is incomplete — Optional loss uses fixed column shifts and only class 0 for multiclass conditional summaries · Use selected plus exploratory pairs, complete class vectors, masks, and native validation | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L134 | Eigenvector coordinates have arbitrary gauge — Numerical eigensolvers can differ in sign and within repeated-eigenvalue subspaces · Prefer covariance/Gram-invariant conditioning; test sign/rotation invariance where factor inputs remain | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L136 | Do not silently rewrite old artifacts or receipts. Introduce versioned successor recipes. Preserve the old paths as baselines until replacements pass conformance and efficacy checks. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **3.3 One authoritative extractor** |  |  |  |
| L140 | Rust should own normalization, statistical extraction, feature selection, packet layout, quantization, and runtime decoding. Python should own genuine neural model/loss/autograd/optimizer work and necessary training … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L142 | For V7 repair, expose a Rust-produced record containing actual bytes plus tensors obtained by native unpacking. For the new encoder, expose a typed private feature record. Shared schemas bind both. Do not independently … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **4.1 Separate the levels of holdout** |  |  |  |
| L148 | Use four different concepts explicitly: | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L150 | **Meta-TRAIN families:** datasets used to learn shared encoder/decoder weights. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L151 | **Episode support A:** rows used to fit transforms, statistics, landmarks, and the per-table generator. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L152 | **Episode query Q:** real rows used as loss supervision during meta-training, never encoder inputs or statistic-selection evidence. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L153 | **Meta-DEV and sealed meta-CONFIRM families:** independent dataset families for shared-model selection and final confirmation. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L155 | Inside A, produce out-of-fold predictions with group- or time-aware folds. Hyperparameter selection for those probes stays inside each fold's training side, or uses a fixed preregistered probe configuration. A probe … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L157 | For an actual deployment evaluation, fit the generator on real A and downstream models on generated S. Evaluate on untouched real T. The real reference learner also fits A, uses an equivalent allowed tuning budget, and … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **4.2 Normalize once per declared fit object** |  |  |  |
| L161 | The user requires normalized real data before it enters the neural model. Implement this with explicit custody: | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L163 | Resolve the target and task from declarations; canonicalize target-last while retaining physical row IDs. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L164 | Split before fitting data-dependent transformations. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L165 | Fit a versioned transform `T_A` only on A. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L166 | Produce normalized A and apply the same immutable map to Q/T and to the recipient's real inference data. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L167 | The new generator samples directly in that normalized coordinate system. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L168 | Any raw-unit reconstruction is an owner-local optional operation using the private card. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L170 | Inner landmark CV may fit `T_AminusFold` for its own fit object. That is a distinct fold-local artifact, not a second fit of the deployment transform. Use it to prevent fold-validation rows from defining learned … | Claude | TESTING | [Extraction identity](#extraction-identity) |
| L172 | Preserve NC1 and existing V7 readers. For the new normalized-only path, use the repository's **canonical NC2 recipe as the first candidate**, with NC1 as an explicit comparison. Do not change the behavior of an existing … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L176 | Continuous — Frozen FIT-only canonical numeric transform; keep out-of-FIT tail values representable | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L177 | Regression target — Strictly affine FIT-only transformation; never clip real query/test labels for headline scoring | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L178 | Binary / nominal — Typed levels or grouped one-hot view; numeric code magnitude has no meaning | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L179 | Ordered discrete — Declared ordinal order plus support; do not infer ordinal meaning from a category ID | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L180 | Missing feature — Separate mask/state; observed-value statistics exclude missing cells | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L181 | Constant feature — Constant flag and valid support; undefined correlations are masked | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L182 | All-missing feature — Distinct state from a constant observed feature | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L183 | Unknown category — Explicit unknown state, separate from missing unless the declared recipe says otherwise | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L184 | One-hot group — Original parent group and mutual-exclusion constraint retained throughout extraction, sampling, and importance scoring | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L186 | Compute the proposed statistical features **from normalized A, its masks, and its structural schema**. Private raw extrema, names, and category strings are unnecessary neural inputs. Some facts lost by … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L188 | A marginal normality test after rank-Gaussianization mostly tests the transformation. Use canonical normalized-value and rank/copula views for their different purposes; state which view each descriptor uses. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **4.3 Every feature carries reliability and provenance** |  |  |  |
| L192 | Each scalar/vector estimate must carry: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L194–210 | feature_id, scope, value, valid, stderr, stderr_valid, n_observed, n_effective, status, recipe_digest, fit_rows_digest, coordinate_digest | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L212 | Effective sample size must reflect duplicate/entity groups or declared weights when relevant. Bootstrap at the independent sampling unit. Adjacent rows or repeated measurements are not independent observations. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L214 | Use finite effect sizes, support, uncertainty, and stability as primary conditioning. P-values may be optional diagnostics, but an underpowered `p > .05` is not evidence of independence and a tiny p-value on a huge … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **5. Ranked additions with the highest expected value** |  |  |  |
| L218 | This ranking is engineering judgment based on the audited bottlenecks. It is not a measured ranking of DOPE ablations. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L222 | 1 — Cross-fitted signed target curves and class-conditional curves · Shape, direction, thresholds, saturation, nonmonotonicity · Allocate spline/step/atom capacity to the right columns | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L223 | 2 — Sparse pair-target surfaces plus exploratory interaction probes · XOR, products, conditional reversals, category combinations · Emit explicit interactions or interaction-conditioned latent code | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L224 | 3 — Landmark OOF residual/disagreement tensor · Which learner families fail on which regions · Train for complementary downstream learners and preserve difficult regions | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L225 | 4 — Conditional noise and residual distribution · Heteroskedasticity, asymmetry, heavy tails, noisy boundaries · Sample appropriate target uncertainty instead of globally scaled … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L226 | 5 — Missingness / categorical dependency graph · Informative absence and legal mixed-data structure · Generate masks and categorical groups jointly with the data | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L227 | 6 — Nonlinear dependency and finite-threshold tail graph · Relationships not summarized by Gaussian covariance · Add limited mixture or nonlinear dependence capacity | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L228 | 7 — Grouped OOF permutation importance and ALE curves · Redundant drivers, stable main effects, interaction dependence · Preserve driver groups and response functions | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L229 | 8 — Bootstrap reliability and learning curves · Which apparent effects survive resampling and more data · Shrink weak estimates, defer expensive probes, route capacity | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L230 | 9 — Boundary / local geometry descriptors · Narrow decision regions and local class overlap · Improve local coverage and difficult synthetic examples | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L231 | 10 — Gradient and curvature sketches · What training signals different learners receive · Match selected learner updates and sensitivities | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L232 | 11 — Slice/environment stability · Whether a relationship is stable across real subpopulations · Preserve subgroup transfer and flag unsupported extrapolation | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L233 | 12 — Rich marginal shape and constraint diagnostics · Atoms, modes, truncation, count structure, invalid combinations · Choose appropriate marginal families and guarantee valid outputs | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L235 | Implement ranks 1–5 first after P0. Ranks 6–9 are the next development batch. Gradient/curvature sketches and broad automated testing should be admitted only if their incremental benefit survives equal-cost ablation. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **6. Candidate feature registry: 144 recipes** |  |  |  |
| L239 | Notation: **G** global, **C** per-column, **E** pair/edge, **Y** class-conditioned, **L** landmark/probe, **S** slice. **P1** first implementation batch, **P2** next, **P3** exploratory. **Low** usually linear or sorting … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **6.1 Structure and usable information: F001–F012** |  |  |  |
| L245 | F001 — Sample-size profile · G / P1 / Low · `log1p(n)` for rows, unique rows, independent entities, valid target rows; uncertainty and capacity routing | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L246 | F002 — Dimensionality profile · G / P1 / Low · Logical columns, expanded channels, numeric/nominal/ordinal proportions, `p/n_eff`; do not confuse one-hot width with … | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L247 | F003 — Constant and unavailable fractions · G,C / P1 / Low · Constant, nearly constant, all-missing, unsupported, and clipped-at-bound fractions as separate states | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L248 | F004 — Duplicate-group concentration · G / P1 / Low · Group-size quantiles, largest-group share, Simpson concentration; safe splitting and effective sample size | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L249 | F005 — Missing-pattern diversity · G / P1 / Low · Number/entropy of bounded frequent missingness patterns and pooled remainder mass | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L250 | F006 — Target balance · G,Y / P1 / Low · Class proportions, entropy, minimum supported class count; regression quantiles and robust scale | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L251 | F007 — Covariance effective rank · G / P2 / Med · Participation ratio and entropy rank of a shrinkage covariance matrix; controls dependence capacity | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L252 | F008 — Spectral energy curve · G / P2 / Med · Leading eigenvalue shares and rank needed for 50/90/95% variance; masks for degenerate matrices | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L253 | F009 — Regularized conditioning · G,C / P2 / Med · Condition number at fixed shrinkage and diagonal leverage summaries; identifies redundant coordinates | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L254 | F010 — Constraint inventory · G / P1 / Low · Counts of nominal groups, bounded/count/ordinal variables, declared sum/order/exclusivity constraints | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L255 | F011 — Supported target gain · G,L / P1 / Med · OOF loss reduction over a fold-trained null, with uncertainty; distinguishes learnable signal from noise | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L256 | F012 — Preparation evidence coverage · G / P1 / Low · Fraction measured/undefined/budget-skipped by feature family, sampled rows and elapsed cost; not a dataset/source … | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **6.2 Marginal distribution and shape: F013–F028** |  |  |  |
| L262 | F013 — Multiresolution quantiles · C / P1 / Low · Shared 5/9/17/33-point quantile sketches with tail support counts; reuse a single sorted view | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L263 | F014 — Robust scale ratios · C / P1 / Low · MAD, IQR, q95−q05, standard deviation, and their stable ratios in normalized units | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L264 | F015 — Quantile asymmetry · C / P2 / Low · Bowley skewness and upper/lower quantile arm ratios; stable alternatives to third moments | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L265 | F016 — L-moment shape · C / P2 / Low · L-skewness and L-kurtosis with bootstrap uncertainty; robust shape comparison | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L266 | F017 — Clipped-moment shape · C / P2 / Low · Skewness/kurtosis on a declared robust view plus untruncated tail mass; distinguish a robust statistic from original … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L267 | F018 — Modal/atom concentration · C / P1 / Low · Largest atom masses, atom count and remaining mass; repeated values must remain discrete | Antigravity | TESTING | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L268 | F019 — Entropy / effective support · C / P2 / Low · Discrete entropy, exp(entropy), Simpson support; fixed-bin entropy for continuous variables has explicit bin … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L269 | F020 — Tail mass curve · C / P2 / Low · Fractions beyond FIT median ± fixed robust-scale multiples, with uncertainty | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L270 | F021 — Tail quantile spacing · C / P2 / Low · Normalized upper/lower quantile increments and asymmetry; allocate tail knots | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L271 | F022 — Multimodality statistic · C / P2 / Med · Dip statistic or bounded KDE mode persistence across bandwidths; report estimator and sample cap | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L272 | F023 — Shape-model distances · C / P3 / Med · Quantile/Wasserstein distance to fitted Gaussian, Laplace, Student-t and simple bounded families, using held-out fit … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L273 | F024 — EDF goodness-of-fit profile · C / P3 / Med · KS/Cramér–von Mises/Anderson–Darling statistics with fit-parameter-aware calibration; no uncorrected classical … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L274 | F025 — Discreteness / spacing profile · C / P1 / Low · Unique ratio, tie fraction, lattice-gap stability; use declared type as authority | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L275 | F026 — Hurdle and zero-inflation profile · C / P1 / Low · Declared zero/atom mass plus conditional positive-part shape; normalization may move literal zero | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L276 | F027 — Boundary pile-up / censoring evidence · C / P2 / Low · Mass near declared boundaries and repeated endpoints; distinguish declared censoring from a speculative statistical … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L277 | F028 — Quantile-estimation stability · C / P2 / Med · Subsample/bootstrap variability at each retained knot; route precision to reliable structure | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **6.3 Missingness and categorical structure: F029–F044** |  |  |  |
| L283 | F029 — Observed/missing support · C / P1 / Low · Missing rate, observed count, group-adjusted uncertainty; never mix imputed values into observed moments | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L284 | F030 — Co-missingness dependence · E / P1 / Med · Phi/MI/Jaccard of missing indicators with support and bias control | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L285 | F031 — Missingness–target effect · C,Y / P1 / Med · OOF target differences or loss gain from a mask-only probe | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L286 | F032 — Predictable missingness · C / P2 / Med · OOF predictability of `M_j` from other observed columns; a pattern descriptor, not proof of MAR/MNAR | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L287 | F033 — Missingness–value association · E / P1 / Med · Dependence of `M_j` on observed values of `X_k`, respecting pairwise support | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L288 | F034 — Frequent missingness regimes · G,S / P2 / Med · Pooled pattern probabilities and target differences within supported patterns | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L289 | F035 — Categorical support spectrum · C / P1 / Low · Cardinality, top masses, singleton/rare mass, cumulative top-k coverage; private label strings excluded | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L290 | F036 — Category uncertainty · C / P1 / Low · Dirichlet-smoothed probabilities and concentration/support; avoid zero-probability collapse | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L291 | F037 — Categorical pair association · E / P1 / Med · Bias-corrected Cramér's V and shrinkage contingency residuals; distinguish signed cell structure from a scalar … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L292 | F038 — Category–numeric association · E / P1 / Med · Correlation ratio and distribution differences across supported levels; opaque IDs never enter Pearson correlation | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L293 | F039 — Class–category target table · C,Y / P1 / Med · Smoothed class probabilities conditional on level and their support; cross-fitted scoring for reliability | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L294 | F040 — Regression–category target table · C / P1 / Med · Shrunk target mean, scale and quantile shifts by category; include pooled rare-level state | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L295 | F041 — Category combination support · E / P1 / Med · Frequent joint levels and structurally forbidden combinations, with absence distinguished from impossibility | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L296 | F042 — One-hot validity / parent map · G,E / P1 / Low · Parent identity, exactly-one/at-most-one rule, missing/unknown channels; support joint generation | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L297 | F043 — Unknown-level stress profile · C / P2 / Med · Inner-fold unseen-category rate and associated OOF loss penalty; fit vocabularies inside folds | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L298 | F044 — Categorical stability · C,E / P2 / Med · Frequency/association changes across support resamples or declared environments | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **6.4 Feature dependence: F045–F060** |  |  |  |
| L304 | F045 — Signed Pearson / rank correlation · E / P1 / Med · Signed Pearson, Spearman with midranks, and optional Kendall on supported numeric/ordinal pairs | Antigravity | TESTING | [Dependence and regimes](#dependence-and-regimes) |
| L305 | F046 — Chatterjee dependence, both directions · E,C / P2 / Med · Tie-aware ξ in each direction; retain estimator uncertainty; scalar association is not causation [R2] | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L306 | F047 — Distance correlation · E / P2 / High · Bounded-sample dependence that can detect nonmonotone relationships; record biased/debiased estimator [R3] | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L307 | F048 — Kernel dependence · E / P2 / Med · Multi-bandwidth HSIC or fixed random-feature approximation, plus permutation null offset [R4] | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L308 | F049 — Mutual-information estimate · E / P3 / High · Type-appropriate discrete/shrinkage or bounded kNN estimator; no universal cross-estimator scale | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L309 | F050 — Shrinkage partial correlations · E / P2 / Med · Precision-matrix conditional linear associations; not generic nonlinear conditional independence | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L310 | F051 — Copula checkerboard · E / P1 / Med · Signed residuals of 2×2/4×4 rank-bin joint mass against independent marginals | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L311 | F052 — Finite-threshold tail association · E / P2 / Med · Joint upper/lower exceedance probabilities at supported quantiles, minus independence reference | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L312 | F053 — Asymmetric quadrant dependence · E / P2 / Med · Upper–lower, lower–upper, upper–upper and lower–lower excess mass | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L313 | F054 — Dependence by regime · E,S / P2 / Med · Pair associations within target-free support clusters or declared groups; weighted contrasts | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L314 | F055 — Nonlinear reconstruction gain · C / P2 / High · OOF gain predicting one feature from the others, relative to a linear predictor; informative redundancy | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L315 | F056 — Dependence graph summaries · G,C / P2 / Low · Weighted degree, connected-component sizes and edge concentration, from a fixed admitted graph | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L316 | F057 — Copula eigenstructure · G / P2 / Med · Spectrum of rank/copula covariance; keep separate from value-space covariance | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L317 | F058 — Conditional mutual-information proxy · E / P3 / High · Incremental cross-fitted predictive gain after a conditioning set; label as predictive proxy, not exact CMI | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L318 | F059 — Empirical characteristic-function sketch · G / P3 / Med · Fixed multi-scale random projections, mean cos/sin and uncertainty; multivariate shape beyond moments | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L319 | F060 — Higher-order dependence sketch · G,E / P3 / Med · Bounded random triples and tensor sketches, with permutation controls; exploration beyond pairwise structure | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **6.5 Target relationships and driver functions: F061–F080** |  |  |  |
| L325 | F061 — Signed numeric target association · C / P1 / Low · Pearson/rank association for regression/binary targets; keep sign rather than only magnitude | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L326 | F062 — Class-conditional feature profile · C,Y / P1 / Med · Numeric quantiles/means/scales and categorical masses within each supported class | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L327 | F063 — Univariate OOF response curve · C,Y / P1 / Med · 8/16-bin or spline conditional target function with out-of-fold quality and support | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L328 | F064 — Response curvature · C / P1 / Low · Finite differences of supported response curves, scale-normalized; smoothness and nonlinearity | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L329 | F065 — Monotonicity consistency · C / P1 / Low · Weighted share of positive/negative derivatives, violations and bootstrap sign stability | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L330 | F066 — Threshold / discontinuity profile · C / P1 / Med · Cross-fitted stump/step gain and threshold-location distribution | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L331 | F067 — Saturation and plateau profile · C / P2 / Low · Supported flat regions and endpoint slope changes from response curves | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L332 | F068 — Periodic target evidence · C / P2 / Med · CV gain of bounded sin/cos bases over additive baseline; wavelength and phase only when stable | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L333 | F069 — Nonmonotone association gap · C / P2 / Med · Nonlinear probe gain minus linear/monotone gain; retain uncertainty and matched folds | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L334 | F070 — OOF drop-column gain · C / P2 / High · Refit learner without a logical column/group; distinct from permutation importance | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L335 | F071 — Grouped permutation importance · C,G / P1 / Med · OOF loss increase after joint permutation of parent/correlated group; negative estimates remain visible | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L336 | F072 — Conditional permutation importance · C / P2 / High · Approximate conditional resampling with support diagnostics; compare to marginal importance [R6] | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L337 | F073 — Importance stability · C / P1 / Med · Rank, sign, top-k inclusion and mass variance across folds/seeds/learner families | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L338 | F074 — ALE main-effect curves · C / P2 / Med · Support-weighted local effect curves, centered on a shared FIT reference; reduces extrapolation issues [R5] | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L339 | F075 — Redundant driver groups · G,C / P1 / Med · Group importance, interchangeable-member evidence and group size; prevent false individual ranking claims | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L340 | F076 — Target conditional quantiles · C,Y / P1 / Med · q10/q50/q90 or a proper quantile grid in supported feature bins; preserves more than conditional means | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L341 | F077 — Class separation profile · C,Y / P2 / Med · Class-pair distribution distance, Fisher-style between/within variation, with regularization | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L342 | F078 — Calibration profile · L,Y / P1 / Med · OOF Brier/log loss, calibration intercept/slope, reliability bins with uncertainty | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L343 | F079 — Target–feature nonlinear dependence · C / P2 / Med · Type-appropriate HSIC/distance correlation/ξ; class relabeling invariant aggregation | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L344 | F080 — Signal concentration · G / P1 / Low · Grouped importance entropy, top-k mass and number of supported drivers; controls target capacity allocation | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **6.6 Interactions: F081–F092** |  |  |  |
| L350 | F081 — Signed product-target moments · E,Y / P1 / Med · Centered numeric/binary products with target or per-class residual; catches XOR direction | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L351 | F082 — Two-way target checkerboard · E,Y / P1 / Med · 2×2/4×4 conditional target means/probabilities, shrinkage and support | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L352 | F083 — Interaction beyond additive effects · E / P1 / Med · Cross-fitted improvement of a pair model over the same two main effects; signed function and scalar gain | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L353 | F084 — Depth-two tree gain · E / P1 / Med · Held-out gain from small pair trees versus single-feature stumps; mixed-type friendly | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L354 | F085 — Pair residual-basis coefficients · E,Y / P1 / Med · Ridge coefficients on products of centered spline/step bases fitted to OOF baseline residuals | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L355 | F086 — Conditional sign reversal · E / P1 / Med · Main-effect direction changes across supported bins of a second variable | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L356 | F087 — Category–category interaction · E,Y / P1 / Med · Joint target table minus fitted additive categorical effects; pooled sparse cells | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L357 | F088 — Category–numeric interaction · E,Y / P1 / Med · Group-dependent numeric response curves and shrinkage toward a common curve | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L358 | F089 — Missingness interaction · E,Y / P1 / Med · Pair mask/value or mask/mask target effects beyond main effects | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L359 | F090 — Interaction graph stability · G,E / P2 / Med · Edge selection frequency, effect variance and top-edge concentration under resampling | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L360 | F091 — Triple/parity probe · E,Y / P3 / High · Seeded exploratory triples or shallow depth-three structures, with matched null controls | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L361 | F092 — Low-rank interaction spectrum · G,E / P2 / Med · Singular values/energy of the supported response-coefficient tensor; choose sparse edges versus shared factors | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **6.7 Residual and noise diagnostics: F093–F108** |  |  |  |
| L367 | F093 — OOF residual distribution · L / P1 / Low · Mean, robust scale, skew, tail quantiles and asymmetry; probabilities/residual vectors for classification | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L368 | F094 — Residual dependence on columns · C,L / P1 / Med · Residual mean curve and nonlinear dependence after a baseline learner | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L369 | F095 — Residual scale curve · C,L / P1 / Med · Shrunk within-bin mean squared residual, absolute residual and residual quantiles; heteroskedastic decoder input | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L370 | F096 — Heteroskedasticity score · L / P2 / Med · Held-out gain predicting residual magnitude; auxiliary Breusch–Pagan/White-style statistic with assumptions explicit | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L371 | F097 — Landmark residual covariance · L,G / P1 / Med · Error covariance/correlation among probes on identical OOF rows; basis for complementary learner conditioning | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L372 | F098 — Landmark disagreement · L,G / P1 / Low · Prediction variance and class-probability disagreement; class-label aligned | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L373 | F099 — Error overlap by region · L,S / P1 / Med · Shared versus complementary errors in fixed support slices, especially hard tails/classes | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L374 | F100 — Bias–variance proxy · L / P2 / High · Error and prediction dispersion across bootstrap fits; explicitly a finite-fit proxy, not identified noise | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L375 | F101 — Local noise proxy · C,S / P2 / Med · Nearby-target dispersion after accounting for supported mean variation; distinguish unresolved function from noise | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L376 | F102 — Standardized residual tail shape · L / P1 / Low · Quantiles/atom mass after a positive conditional scale model; select residual distribution family | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L377 | F103 — Residual class-probability vector · L,Y / P1 / Med · Every `1[y=c]−p_c`, not only true-class surprise or class 0; preserves class-specific correction direction | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L378 | F104 — Class confusion structure · L,Y / P1 / Med · OOF normalized confusion and probabilistic confusion matrices with rare-class uncertainty | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L379 | F105 — Label-conflict evidence · G,S / P2 / Med · Different targets among duplicate/near-duplicate features; diagnostic only, not automatic relabeling | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L380 | F106 — Residual temporal structure · L / P3 / Med · Lag correlations and block residual diagnostics only when time/order is declared | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L381 | F107 — Residual environment structure · L,S / P2 / Med · Groupwise bias/scale/calibration shifts for declared environments | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L382 | F108 — Analytic-base correction reliability · G,C / P1 / Med · Out-of-fold benefit and uncertainty of each proposed correction over step zero; controls shrinkage and fallback | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
|  | **6.8 Learning behavior and learner sensitivity: F109–F120** |  |  |  |
| L388 | F109 — Extended landmark gain profile · L / P1 / Med · Existing 23 probes plus only justified additions; same folds, loss units and failure accounting | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L389 | F110 — Sample-size learning curves · L / P1 / High · Gains at 1/4, 1/2, full support and selected larger regimes; shared validation and nested samples | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L390 | F111 — Learning-curve slopes · L / P2 / Low · Local slope and saturation uncertainty; avoid confident extrapolated asymptotes from three points | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L391 | F112 — Regularization response · L / P2 / Med · Fixed-grid gain changes for ridge/logit/tree capacity; detect fragile high-capacity fits | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L392 | F113 — Depth / interaction response · L / P1 / Med · Matched depth-1/2/3 tree and additive-versus-interaction gains | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L393 | F114 — Neighborhood-scale response · L / P2 / Med · OOF loss for a short kNN/bandwidth grid; local complexity and smoothing needs | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L394 | F115 — Learner rank stability · G,L / P2 / Med · Kendall/pairwise ordering agreement across folds, sizes and seeds; ties/ineligible comparisons explicit | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L395 | F116 — Optimism gap · L / P1 / Low · In-sample minus OOF gain; highlights overfitting and unstable descriptors | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L396 | F117 — Gradient mean sketch · L / P3 / Med · Fixed-projection mean gradient at pinned learner states, normalized per parameter block | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L397 | F118 — Gradient second-moment sketch · L / P3 / High · Diagonal/Fisher-like gradient variance summaries and top projected directions | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L398 | F119 — Curvature / effective-dimension proxy · L / P3 / High · Regularized Gram spectrum or Hessian trace estimates; distinguish exact linear quantities from neural approximations | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L399 | F120 — Influence concentration · G,L / P3 / High · Aggregate leverage/influence quantiles and tail mass; never export individual high-influence records | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **6.9 Local geometry and boundaries: F121–F132** |  |  |  |
| L405 | F121 — Neighbor-distance distribution · G,S / P2 / Med · Quantiles of mixed-type nearest-neighbor distances on a bounded sample | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L406 | F122 — Distance concentration · G / P2 / Med · Near/far distance ratios across a small radius grid; detects unhelpful high-dimensional distance geometry | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L407 | F123 — Intrinsic-dimension estimate · G,S / P3 / Med · Bounded-neighbor dimension estimates with resampling sensitivity; not ground-truth manifold dimension | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L408 | F124 — Local label disagreement · S / P2 / Med · Same/different-class neighbor proportions or regression local response dispersion | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L409 | F125 — Class-boundary fraction · G,Y / P2 / Med · Neighbor/MST boundary proxy, using training-only labels and valid metric | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L410 | F126 — Margin distribution · L,S / P2 / Med · OOF probability margins and wrong-side mass; not training margins | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L411 | F127 — Neighborhood class entropy · S,Y / P2 / Med · Smoothed local entropy over several k values; respects rare-class support | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L412 | F128 — Class overlap · G,Y / P2 / Med · Regularized Fisher/nearest-class overlap measures; label-invariant class aggregation | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L413 | F129 — Local linearity · S / P2 / High · OOF local-linear versus global-linear fit and residual variation | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L414 | F130 — Clusterability / regime evidence · G,S / P2 / Med · Stable target-free clusters across subsamples, with no fixed cluster count assumed correct | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L415 | F131 — Coverage and density imbalance · G,S / P2 / Med · Mass by stable support region, low-density tail share and density contrast | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L416 | F132 — Constraint-respecting local connectivity · G,S / P3 / Med · Connectivity after respecting categorical groups and declared constraints; avoids invalid Euclidean shortcuts | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **6.10 Stability, environments, and practical value: F133–F144** |  |  |  |
| L422 | F133 — Feature-statistic bootstrap stability · All / P1 / Med · Estimator variance and sign/selection agreement at independent-group level | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L423 | F134 — Split sensitivity · G,L / P1 / Med · Change under multiple valid support partitions; quantifies artifact instability | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L424 | F135 — Sample-size sensitivity of descriptors · All / P2 / Med · Drift of each descriptor from small to larger support samples | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L425 | F136 — Subgroup predictive structure · S,Y / P1 / Med · OOF gain, error, calibration and support in declared or support-discovered slices | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L426 | F137 — Subgroup driver structure · S,C / P2 / High · Group-level importance and curve changes by supported slice | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L427 | F138 — Environment distribution shift · S / P2 / Med · MMD/energy/contingency effect sizes among support environments; no TEST-derived conditioning | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L428 | F139 — Time trend and seasonality · S,C / P3 / Med · Declared-time trend/periodic strength, block stability and lag descriptors; absent for IID tables | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L429 | F140 — Effective entity balance · S,G / P1 / Low · Entity-count distribution, repeated-measure concentration and class coverage by group | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L430 | F141 — Normalization sensitivity · G,C / P2 / Med · OOF loss and descriptor change across preregistered normalization candidates, selected within support | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L431 | F142 — Quantization sensitivity · G,C,E / P1 / Med · Change in diagnostic learner loss/response after candidate bit allocations, using support-internal validation | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L432 | F143 — Synthetic-size response · G,L / P2 / High · Development-only transfer curves at fixed synthetic size multipliers; future inference use requires … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L433 | F144 — Marginal value of an extra probe · G,L / P3 / High · Learned expected improvement conditional on measured evidence and cost, trained on meta-TRAIN only | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L435 | F143 is initially an evaluation/allocator-training target, not a free deployment input. F144 requires logged probe outcomes and an exploration policy. Neither may use DEV/CONFIRM outcomes as inference features. This … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **7.1 Cross-fitted target-function fingerprints** |  |  |  |
| L441 | For each numeric column, form quantile bins or a short spline basis using only the inner training rows. Fit a regularized univariate predictor there and predict the held-out fold. For categorical columns, fit pooled, … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L443 | Return two clearly distinguished objects: | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L445 | A **final support-fitted response function**, in canonical A coordinates, used as generator conditioning. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L446 | **Cross-fitted evidence of its usefulness and stability**, estimated without fitting on the rows being predicted. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L448 | For a regression bin b, a simple shrinkage estimator is | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L450 | `mu_b = (n_b * mean(y_b) + lambda * mu_parent) / (n_b + lambda)`. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L452 | For a class c, | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L454 | `p_bc = (n_bc + lambda * pi_c) / (n_b + lambda)`. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L456 | Choose lambda from a short meta-TRAIN/development grid or support-internal CV. It is not a universal constant. Include `n_b`, effective independent support, standard error, pooled/unpooled status, and a bin-validity … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L458 | A compact fingerprint can contain 8 centered curve values, 8 uncertainty/support coordinates, monotonicity, curvature, threshold strength, and cross-fitted gain. Keep the richer curve privately even if the production … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L460 | This is higher value than stacking many transforms whose aggregate correlations are almost identical. Existing log/sqrt/hinge/saturation probes remain useful as low-cost summaries and initialization choices. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **7.2 Cross-fitted residual surfaces and interaction discovery** |  |  |  |
| L464 | Start from a fixed additive baseline trained inside A. Obtain OOF residuals: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L466 | Regression: `e_i = y_i − f_-fold(i)(x_i)`. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L467 | Classification: `e_ic = 1[y_i=c] − p_-fold(i),c(x_i)`. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L469 | Construct centered univariate bases `phi_j(x_j)`, then pair bases `phi_j ⊗ phi_k`. Fit regularized pair coefficients, or a 4×4 smoothed residual table, without using outer query Q. Evaluate incremental gain using an … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L471 | Subtracting two marginal means from a 2D table is not a valid functional ANOVA decomposition under arbitrary dependence. For correlated features, fit the additive reference on the observed support, then fit residual … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L473 | Pair selection must combine: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L475 | Known structural pairs: parent groups, declared constraints, co-missingness. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L476 | High dependence pairs, including nonlinear and categorical dependence. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L477 | Pairs involving strong single-column target effects. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L478 | A **target-independent exploration quota** that can discover XOR-like pairs with zero marginal signal. | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L480 | For the initial p≤48 experiment, scan **all 1,128 possible pairs** with cheap 2×2 target and occupancy statistics on a capped sample, then refine at most 64 edges. This is often a better starting point than a complicated … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L482 | Track multiple-testing optimism. A maximum over 1,128 noisy interactions is biased upward even if every interaction is zero. Use shrinkage, held-out scoring, stability selection and label-permutation controls. If … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **7.3 A landmark error tensor** |  |  |  |
| L486 | The existing 23-probe panel is a valuable measurement system. Extend the private report to retain OOF predictions and fold membership, or compute the following summaries while predictions are available: | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L488 | Per-probe loss, null-relative gain, uncertainty, train/OOF gap and fitting cost. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L489 | Pairwise residual covariance/correlation on identical valid rows. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L490 | Region-conditioned residual means, scales and difficult-class errors. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L491 | Pairwise prediction disagreement and corrected ensemble gain. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L492 | Per-column response, grouped importance and pair interaction summaries. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L494 | For regression, let `E[i,l] = y_i − pred[i,l]`. Normalize by a fold-trained robust target scale. For classification, use class-aligned residual vectors or proper per-row scores; correlation of class IDs is meaningless. … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L496 | An ensemble gain must be evaluated honestly: ensemble weights are fixed in advance or learned inside additional training folds, then scored on untouched internal rows. Otherwise, it mainly measures overfitting to the OOF … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L498 | Model input should include a small probe token for each method and a low-rank summary of the residual covariance matrix. Use a fixed probe-identity embedding, not a dataset identity. This allows the encoder to … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L500 | The raw OOF matrix is private intermediate data and does not go into the distributable packet. Cache its summary under the exact support-row, fold, transform, learner, seed, and binary identities. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **7.4 Missingness is a joint distribution** |  |  |  |
| L504 | Represent `P(M, X_observed, Y)`, where M is the missingness mask. A single missing-rate value for each column cannot reproduce correlated or informative missingness. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L506 | Start with a low-rank mask model conditioned on the same target-free latent regime as the covariates. Add a few stable mask–value or mask–mask edges when supported. If target-conditioned mask generation is used, model a … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L508 | Generate a logical categorical variable once, then derive its one-hot channels. Never independently sample one-hot indicators and repair them by arbitrary argmax without measuring the distributional distortion. Add … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L510 | This does not identify a missing-data mechanism. Observational data alone generally cannot prove that missingness is MCAR, MAR, or MNAR. These features are predictive structure descriptors. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **7.5 Nonlinear dependence and tails** |  |  |  |
| L514 | Use different tools for different effects: | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L516 | Pearson/rank correlation: linear or monotone relationships. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L517 | Checkerboard rank masses: local copula shape and asymmetry. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L518 | HSIC/distance correlation: nonlinear association screening. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L519 | Conditional feature prediction: redundancy not explained by pairwise correlation. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L520 | Supported finite-threshold co-exceedances: tail behavior. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L522 | At a quantile u, record `P(U_j>u, U_k>u)` and its difference from `(1−u)^2`, with binomial/group-bootstrap uncertainty. Call this **finite-threshold tail association**, not an estimate of asymptotic tail dependence … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L524 | For Chatterjee ξ, use the published tie-aware estimator and test it on categorical and repeated data. Do not reuse an arbitrary double-argsort rank jump as if it were the same statistic. Keep the existing recipe intact … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **7.6 Geometry and type-aware distances** |  |  |  |
| L528 | A scaled Euclidean metric is reasonable for continuous numeric channels; it is not a general mixed-data metric. Use a documented distance with numeric robust scaling, categorical mismatch, and explicit missingness … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L530 | Compute bounded-sample neighborhood features at k∈{4,16,64}, subject to n. Mask a scale when there are too few independent neighbors. Use identical neighbor definitions for real/synthetic comparisons and include … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **7.7 Gradient and curvature fingerprints** |  |  |  |
| L534 | This is an advanced experiment after conditional structure works. At fixed learner states w drawn from a meta-TRAIN-only probe bank, compute | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L536 | `g_w(A) = mean_i grad_w loss(f_w(x_i), y_i)`. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L538 | Store a small fixed random projection of each parameter block's gradient, its norm, and projected second moments. Use the same task-compatible architecture and state definitions for all tables. No original examples or … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L540 | Why it may help: two datasets with similar scalar statistics can induce different optimization directions. Gradient matching has precedent in dataset condensation [R9]. Its transfer to this compact mixed-tabular setting … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L542 | Use multiple learner states and architectures to avoid preserving only one random network's peculiarities. Empirical Fisher diagonal values are squared-gradient summaries; do not relabel them as exact Hessian diagonals. … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **7.8 Statistical tests to down-rank or make optional** |  |  |  |
| L546 | Keep a broad registry, but prioritize effect estimators over tests that add cost without actionable information: | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L550 | Shapiro–Wilk / normality p-values — Small, eligible continuous samples; explicit diagnostic · Transformation-dependent, sample-size-sensitive, not a direct predictor of transfer | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L551 | Anderson–Darling / KS / CvM — Shape distance and calibrated goodness-of-fit · Fitted parameters and ties change null distributions | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L552 | Levene / Brown–Forsythe / BP / White — Supported variance differences or residual heteroskedasticity · Redundant with a richer conditional scale curve; assumptions and selection matter | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L553 | ANOVA / Kruskal–Wallis — Effect-size profile across groups · A p-value alone discards shape, direction and support | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L554 | Chi-square / Fisher exact — Categorical association with valid cell support · Sparse high-cardinality tables require pooling or exact/simulation calibration | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L555 | ADF / KPSS / Ljung–Box — Declared time-series panels with valid ordering · Meaningless on arbitrarily ordered IID rows | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L556 | Mardia / multivariate normality — Small continuous blocks as diagnostic · Does not say which dependence primitive improves the generator | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L557 | Topological persistence — Special geometric datasets, exploratory budget · High cost and unclear incremental value for the first tabular transfer milestone | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L558 | Automated causal discovery — Separate study with explicit assumptions · Not a source of ground-truth causal drivers from arbitrary observational tables | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L560 | Use primary statistical definitions and reference documentation for applicability checks [R21]. Implement native estimators under the repository policy; the reference implementation's language is not a deployment … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **8.1 Progressive extraction tiers** |  |  |  |
| L566 | Do not bind extraction effort to the old output-tier names. A 2 KiB artifact may deserve an expensive owner-side fit; a fast preview may use minimal descriptors. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L570 | `core` — Types, masks, quantiles, class profiles, linear association, cheap all-pair 2×2 scans · Full streaming summaries; pair scan n≤2,048 initially · First implementation and cheap inference | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L571 | `structure` — Cross-fitted target curves, 4×4 selected pairs, residual summaries, categorical/missing graph, small learner panel · n≤4,096 per expensive fit initially · Default quality experiment | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L572 | `extended` — Existing full landmark panel, learning curves, HSIC/dCor, ALE/group importance, slice and stability measurements · Method-specific caps, usually 512–4,096 · High-value deployment and research | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L573 | `research` — Gradient/curvature, triples, expensive conditional importance, alternative distributions · Explicit job budget · Discovery; no production-default promise | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L575 | These are initial computational caps, not statistical guarantees. Support counts always describe the actual computation. Full-table quantities and sampled estimates must be distinguishable. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L577 | Cost controls: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L579 | Sort each numeric column once and reuse quantiles, ranks, atoms and tail thresholds. | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L580 | Stream counts and low-order moments; use a reproducible row-ID-based sample for expensive methods. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L581 | p≤48 all-pair 2×2 scans cost `O(n*p²)` but require no quadratic row-distance matrix. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L582 | Dense dCor/HSIC on m rows is `O(m²)` per selected comparison; cap m, reuse buffers, and prefer admitted random-feature approximations where useful. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L583 | A feature RFF Gram requires `O(n*d)` for fixed d; pair interactions of projected features need separate accounting. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L584 | OOF landmarks cost approximately number_of_folds × learner_fit_cost. Reuse predictions rather than refitting for each descriptor. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L585 | Bootstrap cheap sufficient accumulators when statistically valid; full model refitting is a separately budgeted operation. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L586 | Record wall time, CPU time, peak RSS, rows examined, pairs examined, model fits and timeout/failure counts by family. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L588 | Target overhead is a development constraint to measure, not an invented throughput number. Compare a richer feature set against spending the same owner CPU budget on better base fitting or a larger packet. Also compare … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **8.2 Feature registry and stable identities** |  |  |  |
| L592 | Each feature definition needs: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L595 | feature_id: F082 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L596 | name: target_checkerboard | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L597 | version: 1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L598 | scope: pair_class | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L599 | input_view: normalized_numeric_or_typed_nominal | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L600 | requires_target: true | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L601 | allowed_rows: fit_only | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L602 | grid: [2, 4] | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L603 | estimator: dirichlet_or_ridge_shrunk_cell_response | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L604 | uncertainty: independent_group_bootstrap | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L605 | undefined: explicit_mask | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L606 | sample_cap: 2048 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L607 | output_layout: generated_from_schema | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L608 | cost_family: pair_scan | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L611 | Generate tensor layouts and Rust/Python training bindings from a versioned schema. Do not manually append fields to an unnamed vector. The manifest must distinguish: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L613 | Value slots, validity masks, support, uncertainty, and feature status. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L614 | Symmetric versus directed edge features. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L615 | Class-dependent outputs and class masks. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L616 | Numeric, nominal, ordinal, parent-group, and missingness roles. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L617 | Inner-fold estimate versus final support-fit function. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L619 | The reproducible meta-feature extraction literature motivates this registry discipline [R1]. The particular feature bundles and packet consumers proposed here are DOPE-specific design choices. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L621 | For neural preprocessing, fit robust feature centering/scaling **across meta-TRAIN datasets only**. Use signed-asinh or bounded signed transforms where appropriate, `log1p` for counts, and bounded probability transforms … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **8.3 Cache identity and invalidation** |  |  |  |
| L625 | Key feature caches by source version, physical support rows, target declaration, task/class schema, normalization card digest, split/fold specification, feature recipe, extractor binary, seed, sample-selection recipe, … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L627 | For pure, unsupervised streaming accumulators, support exact subsetting only if the accumulator algebra and row custody actually permit it. Target-dependent and cross-fitted features generally require recomputation for … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **8.4 Statistical invariances** |  |  |  |
| L631 | Require row-order invariance within numerical tolerance and deterministic sampling under stable physical IDs. Require column-permutation equivariance and category/class relabeling equivariance. Apply permutations to … | Antigravity | TESTING | [Extraction identity](#extraction-identity) |
| L633 | For covariance factors, the representation `L` and `LQ` for orthogonal Q may describe the same covariance. Conditioning directly on signed factor coordinates gives the neural model an arbitrary coordinate problem. Prefer … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **9.1 Separate two experiments** |  |  |  |
| L639 | **Track A — repaired packet-conditioned V7.** Keep the existing packet and 42/46 input semantics. Train from native extraction, repair episodes and transfer objectives, and measure the incremental effect. New features … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L641 | **Track B — richer owner-side encoder with a new versioned packet.** The encoder sees the richer normalized-FIT evidence. It emits compact task-specific codes/parameters. The recipient uses only the packet, declared … | Antigravity | TESTING | [Owner compiler](#owner-compiler) |
| L643 | This separation is essential. Current V7 applies its neural refinement to the compact packet at sampling time. Feeding private rich features only during training would create train/serve skew. Using them at encode time … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **9.2 Typed graph/set encoder** |  |  |  |
| L647 | Recommended initial Track B architecture: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L649 | Shared column-family adapters: project marginal, target, missingness and categorical feature blocks separately. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L650 | Column tokens: start at width **160**, 3 residual blocks, 4 heads to stay near the released model's scale. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L651 | Pair edges: selected dependence/interaction tokens with endpoint references and reliability gates. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L652 | Probe tokens: existing landmark identities plus OOF residual and learning-curve summaries. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L653 | Class tokens: variable count, full class alignment, and shared operations so arbitrary class relabeling is supported. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L654 | Global tokens: shape, signal strength, preparation coverage, capacity budget and target task. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L655 | Optional slice tokens: supported target-free clusters or declared environments, capped initially at 8. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L657 | Use edge-conditioned attention or sparse message passing. Column updates must preserve column identity; pooled global vectors alone should not carry the whole dataset. Pool a small set of global slots only after … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L659 | Feature reliability should gate messages, for example: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L661 | `h_feature = adapter(value, valid, log_support, uncertainty, status)` | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L663 | `gate = sigmoid(g(h_feature)) * valid`. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L665 | Do not hard-code inverse-variance weighting without floors: a spuriously tiny estimated SE can dominate. Standard errors are themselves noisy, especially for adaptive features. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L667 | Use feature-family dropout during training: remove whole evidence families according to measured deployment availability, retain availability masks, and distinguish missing observations from observed zeros. Create a real … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L669 | Parameter count, inference MACs, native latency, weights size, and peak memory must be reported. New richer inputs do not justify quietly moving from a sub-million-parameter model to tens of millions without … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **9.3 Separate distribution and task representations** |  |  |  |
| L673 | Use two information paths: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L675 | **Distribution representation:** normalized X, masks, types, constraints and target-free dependence. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L676 | **Task representation:** y-related curves, interactions, landmark errors, class tokens and conditional noise. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L678 | A narrowly defined task-aware byte allocator may prioritize covariate precision in regions important for prediction. Preserve a distribution-fidelity constraint so it cannot manufacture unrealistic X values to exploit … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L680 | This structure is especially useful for future multiple-target or unsupervised uses. A supervised packet that preserves one Y need not preserve every other possible target. Evaluate any broader claim separately. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **9.4 Generator upgrades with direct feature consumers** |  |  |  |
| L686 | Signed response curves — Additive conditional functions · Piecewise-linear/spline or step terms on selected columns, with quantized coefficients | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L687 | Pair residual surfaces — Sparse nonlinear target interactions · Small 2×2/4×4 cell tables or low-rank spline products, selected under byte budget | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L688 | Conditional residual scale — Heteroskedastic target noise · Positive scale `softplus(a + sum_j h_j(x_j)) + floor`, strongly regularized | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L689 | Residual skew/tails — Non-Gaussian residual distribution · Few residual quantiles or a small supported mixture; standardized and centered | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L690 | Non-Gaussian copula evidence — Richer covariate dependence · Small mixture of low-rank copulas or selected nonlinear conditional edges | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L691 | Missingness graph — Joint mask generator · Shared latent mask factors plus sparse stable dependencies | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L692 | Nominal conditional tables — Typed categorical sampler · Group probabilities with shared latent dependence and legal-state generation | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L693 | Multimodal marginals / atoms — Hybrid marginal model · Explicit atoms plus continuous quantile spline | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L694 | Stable groups / constraints — Valid joint output · Generate constrained primitives directly; measure any repair fallback | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L696 | Prefer a bounded expansion over one huge generic decoder. An arbitrary mixed joint distribution cannot always fit in 2 KiB. Larger artifact tiers should be admitted when necessary, and inadequacy of the small tier should … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L698 | For regression, use a structured target mean: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L700 | `mu(x) = b + linear(x) + sum_j spline_j(x_j) + sum_(j,k in E) interaction_jk(x_j,x_k)`. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L702 | A small shared neural residual is an optional complement. For classification, emit class-aligned logits and sample actual class IDs. For class count beyond a supported cap, reject or use a separately declared format; … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L704 | For continuous targets, constrain residual mean to zero so noise does not silently move the conditional mean. Preserve conditional variance and tail mass, but do not add synthetic noise merely to lower a realism … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **9.5 Source-independent sampling and row locality** |  |  |  |
| L708 | The sampler must remain a function of `(packet, shared_decoder_digest, row_counter, seed)`. It must not need FIT rows, OOF predictions, private descriptor caches, or the original feature extractor. Sampling 1,024 rows in … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L710 | Whole-table operations are allowed when compiling the packet. Cross-row batch normalization, rescaling based on the requested synthetic batch, or recipient-time fitting against real data is not allowed in the standard … | Codex | TESTING | [Recipient behavior](#recipient-behavior) |
|  | **9.6 A proposed DSE8 format, not a silent V7 extension** |  |  |  |
| L714 | `DSE8` is a working proposal name. Final naming belongs to repository maintainers. Preserve all DSE7 and historical DML readers. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L716 | Required sections: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L720 | Header — Format version, task, logical width, class count, budget, feature/schema recipe, decoder digest and checksums | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L721 | Public normalized schema — Typed groups, ordering, nominal support/unknown/missing state, constraints; no private original labels or extrema | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L722 | Marginal state — Atoms, quantile/codebook state, per-column masks and precision | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L723 | Dependence state — Low-rank factors or selected conditional edges/regimes | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L724 | Task state — Main effects, interactions, class alignment and conditional noise | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L725 | Learned code — Optional global/per-column/per-edge quantized latent codes consumed by the shared decoder | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L726 | Accounting — Exact bytes and semantic parameter counts for all required per-dataset sections | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L728 | Keep raw-unit scale cards owner-private on the normalized-only path. If a user explicitly needs a portable raw-unit artifact, count every required transform/schema byte in a separate declared product profile. Do not … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L730 | The initial output-cap grid should retain the current KPI grid **512, 1,024, 2,048 and 4,096 bytes**. Keep 16/64 KiB as separate development frontiers where supported. Use actual whole-artifact lengths, not an estimated … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L732 | An illustrative 4×4 pair-target table uses 16 int8 coefficients plus endpoint indices, scale and flags—roughly two dozen bytes before alignment/schema overhead. A set of 16 such edges can consume several hundred bytes. … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **9.7 Value-aware byte allocation** |  |  |  |
| L736 | Estimate the supported marginal benefit of each candidate component under quantization: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L738 | `utility(component) = estimated reduction in downstream distortion / incremental serialized bytes`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L740 | Start with a deterministic greedy allocator using support-internal validation and reliability penalties. Recompute true marginal benefits for interacting components; independent gains do not add exactly. Compare against … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L742 | The allocator may trade 17 knots on an almost-uniform irrelevant feature for a pair interaction or a rare supported category. It must not erase columns silently: dropped/pooled structure has explicit semantics and counts … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L744 | Report packet bytes, shared decoder bytes, owner-only encoder bytes, preparation CPU, recipient latency and recipient RSS separately. For M tables sharing a decoder, show amortized bytes `packet + decoder/M` alongside … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **10.1 Formal target** |  |  |  |
| L750 | Let A be normalized support data, `C_theta(A,B)` the byte-budgeted compiled packet, and `S_theta` hard samples obtained after actual packing/unpacking. For downstream learner family a with training randomness omega, | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L752 | `w_a(S_theta,omega) = Train_a(S_theta; omega)` | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L754 | `L_transfer = E_(dataset,split,omega,sample_seed)[ loss_Q(w_a(S_theta,omega)) ]`. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L756 | The generator is useful when training on S produces good predictions on independent normalized real Q/T. A low direct loss for the generator's own target head is helpful but does not establish this. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L758 | For nonparametric infinite data, exact conditional distributions are powerful. In finite and misspecified learner families, **P(X) matters as well as P(Y&#124;X)** because it changes which regions and functions the learner … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L760 | For a bounded loss, a standard total-variation bound controls a fixed predictor's real/synthetic risk difference. It offers no guarantee that a tiny arbitrary packet preserves every learner, target, rare subgroup, or … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **10.2 Learner ensemble inside training** |  |  |  |
| L764 | Use a small rotating panel of differentiable learners: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L768 | Linear — Ridge for regression; damped logistic/softmax for classification · Stable analytic anchor and coefficient-sensitive transfer | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L769 | Kernel-like — Random Fourier / spline features with ridge or proper classification fit · Smooth nonlinear and local relationships | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L770 | Axis-aligned — Soft shallow trees / differentiable leaf-statistic objective · Rectangular interactions and category/missing structure | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L771 | Neural — Small MLP, 3–10 differentiable inner updates, sampled initialization · Learner dynamics beyond a closed-form fit | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L773 | Use at least two families per batch when affordable; rotate the others at a fixed schedule. All means are first normalized per dataset/family, then balanced across dataset families and tasks. A table with more columns, … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L775 | Classification should fit proper logistic/softmax objectives. Ridge on one-hot targets is a useful control but should not be the only differentiable classification auditor. Clamping a ridge output into `[0,1]` creates … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L777 | Exact CatBoost, LightGBM, XGBoost, Kumo/other tabular foundation models, and strong MLPs belong in an external validation panel where admitted and available. They need not be differentiable. Validate whether proxy … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **10.3 Normalize losses without hiding failures** |  |  |  |
| L781 | For learner a, form a training-scale-normalized excess loss: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L783 | `e_a = (L_a(S→Q) − stop_gradient(L_a(A→Q))) / max(scale_a, epsilon_a)`. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L785 | Use a scale derived from the corresponding real null loss or a fixed meta-TRAIN reference distribution, with a predeclared floor. Classification and regression use separate scales. Do not divide every episode by an … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L787 | Train with a mean plus a smooth difficult-family term, for example: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L789 | `L_multi = mean_a(e_a) + gamma * tau * log(mean_a(exp(e_a/tau)))`. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L791 | Compute log-sum-exp stably. Start with gamma=0, then test a small fixed grid; a smooth maximum can overfocus on a noisy weak learner. Report the full vector, not only the aggregate. The public KPI still uses its defined … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L793 | Never clip final scientific retention values to make plots or tables favorable. If the existing selection statistic clips by contract, keep both that statistic and the unclipped underlying losses/retention, explicitly … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **10.4 Auxiliary objectives** |  |  |  |
| L797 | Use a constrained or carefully normalized multi-objective formulation: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L799 | `L = L_multi + lambda_joint*L_joint + lambda_driver*L_driver + lambda_cond*L_cond + lambda_anchor*L_anchor + lambda_code*L_code`. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L801 | The byte budget and hard output validity should be enforced by the compiler/sampler, not merely a weak penalty. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L803 | **Joint distribution.** Match marginal quantile/atom distributions, selected feature-pair masses, mask/category structure and class-conditioned feature distributions. Include a genuine multi-bandwidth MMD or … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L805 | **Conditional distribution.** Use real-query proper scoring losses: regression likelihood/CRPS or pinball loss with valid distribution parameterization; classification cross-entropy/Brier [R20]. Report NLL and … | Antigravity | TESTING | [Hard-transfer objective](#hard-transfer-objective) |
| L807 | **Driver structure.** During training, match group-sensitive effects and supported response curves on independent episode query points. Use differentiable permutations or gradient/effect proxies as auxiliaries, then … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L809 | **Anchoring.** Prefer prediction/distribution anchoring to the analytic baseline in addition to parameter norms. Parameter L2 is not invariant to equivalent parameterizations. For classification, use a support-weighted … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L811 | **Code regularization.** Quantization robustness, active component count and latent smoothness can help packet efficiency. Do not penalize a needed interaction merely because its code is less smooth than a marginal … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L813 | Begin by making `L_multi` the dominant task objective. Select a small preregistered auxiliary-weight grid on DEV. No coefficient in this report should be treated as a scientifically optimized constant. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **10.5 Hard labels, exact quantization and forward/backward mismatch** |  |  |  |
| L817 | Forward behavior must match what recipients get: | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L819 | Select a feasible layout and serialize all dataset-specific state. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L820 | Unpack it using the native definitions. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L821 | Sample actual categorical states and class IDs. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L822 | Train the downstream learner on those samples. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L824 | During autograd training, use exact quantized/dequantized forward values with an explicitly approximate straight-through derivative where necessary. Categorical straight-through or Gumbel relaxations are biased gradient … | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L826 | Report two gaps: | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L828 | **Soft-to-hard gap:** differentiable soft-label proxy versus hard-sample transfer. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L829 | **Training-to-native gap:** Torch training forward versus native serialized sampling and scoring. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L831 | The first is a model/objective gap; the second is an implementation or coordinate gap. Neither should be hidden by reporting only the better path. Native hard-sample results decide model promotion. | Codex | TESTING | [Hard forward](#hard-forward) |
| L833 | Avoid “quantization-aware” training that quantizes only an approximate intermediate while the native codec changes column selection, rank, support or class mapping. Train layout choices and endpoint behavior too. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **10.6 Gradient matching as an optional transfer auxiliary** |  |  |  |
| L837 | For sampled probe states w, compare gradients induced by S and A using a block-normalized mixture of magnitude and direction: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L839 | `L_grad = mean_blocks [ &#124;&#124;g_S−g_A&#124;&#124;²/(&#124;&#124;g_A&#124;&#124;²+epsilon) + eta*(1−cos(g_S,g_A)) ]`. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L841 | Mask near-zero gradients for cosine terms and cap unstable normalization. Use multiple architectures/states and include nonlinear interaction probes. Gradient matching and kernel meta-learning offer useful precedents … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L843 | Matching a single gradient at one initialization can be gamed. It should complement hard TSTR evaluation, not become the headline metric. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **11.1 First reduce avoidable gradient noise** |  |  |  |
| L849 | **Common random numbers:** compare a candidate and analytic baseline using identical latent noise, class-sampling uniforms, support/query splits and learner initializations. Rotate the noise bank during training so the … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L850 | **Antithetic continuous noise:** pair z with −z when valid for the base noise distribution; measure variance reduction. Do not assume it works for every nonlinear decoder or categorical draw. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L851 | **Low-discrepancy noise experiment:** scrambled Sobol streams with independent scrambles across episodes can reduce sampling variance; verify row-counter semantics and high-dimensional behavior before adoption. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L852 | **Stable inner solves:** use damped Cholesky or a documented stable solve; log condition estimates and damping. Avoid explicit matrix inverses. Keep critical solves/reductions in fp32 or fp64, even if neural layers later … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L853 | **Normalize by actual masks/counts:** no padded rows in quantiles, losses or covariance. Variable query lengths should not force every dataset to the smallest query subset without accounting. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L854 | **Family-balanced accumulation:** accumulate across different tables/families, not only many resamples of the same source. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L856 | These address the variance of the estimator being optimized. Arbitrarily increasing batch size or momentum will not fix an unidentifiable packet or the wrong objective. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **11.2 Stage residual learning** |  |  |  |
| L860 | Use a fixed initial sequence, compared with joint training: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L862 | **Stage 0:** analytic decoder, zero learned changes; establish exact-wire baseline. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L863 | **Stage 1:** learn target main-effect corrections with covariate distribution frozen and strong baseline anchoring. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L864 | **Stage 2:** add sparse interactions and conditional noise, while measuring actual hard transfer. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L865 | **Stage 3:** admit small covariate/mask/dependence changes under joint-distribution constraints. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L866 | **Stage 4:** jointly fine-tune the admitted components at lower learning rate. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L868 | Stages advance by predeclared development evidence, not by training loss alone. Also run the 2×2 ablation of X adaptation on/off and Y adaptation on/off; this directly addresses the release note's finding. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L870 | Keep a baseline mixture gate with an explicit near-baseline initialization. Train its decision from support evidence and meta-TRAIN outcomes. A support-internal validation fallback is an optional bounded fitting profile … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L872 | Zero-initialized output heads can initially block gradients into earlier layers. This is expected for a residual baseline but should be measured. Log per-head and backbone gradient norms; do not mistakenly interpret a … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **11.3 Handle gradient conflict only when measured** |  |  |  |
| L876 | Log pairwise gradient cosine similarities among transfer, joint fidelity, driver and anchor objectives on shared parameters. If sustained conflict is present, test **one intervention at a time**: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L878 | GradNorm-style adaptive balancing [R12]. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L879 | PCGrad-style projection for conflicting objectives [R13]. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L880 | Alternating a transfer update and a distribution update. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L881 | Partial separation of distribution/task parameters. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L883 | Do not automatically apply gradient surgery to every batch. It adds compute and changes the optimization target. An apparent training improvement may reduce the actual transfer KPI or realism. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L885 | The existing global clip and EMA are useful baselines. Add per-head norm/saturation diagnostics before changing thresholds. Track the fraction of clipped steps and the proportion of features or logits at clamps. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **11.4 Smoothness methods to test, not assume** |  |  |  |
| L891 | Measurement-noise augmentation — Perturb summary estimates using their estimated sampling uncertainty, preserving bounds/masks · Better held-out transfer versus equal-cost dropout; no corruption of categorical types | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L892 | Smooth histogram continuation — Train with soft bin assignments, gradually sharpen, then evaluate hard native bins · Hard-forward fidelity and gradient bias gap | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L893 | Warmup + cosine / WSD — Retain current alternatives and fix schedule before each comparison · Same update/compute budget and exact resume | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L894 | EMA — Retain current implementation as a baseline · Raw versus EMA native KPI at predetermined checkpoints | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L895 | SWA / checkpoint averaging — Average compatible nearby checkpoints only [R15] · Native evaluation; never assume averaging arbitrary basins improves quality | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L896 | SAM — Small explicit ablation after pipeline repair [R14] · Roughly additional gradient work accounted for; paired noise during perturbation | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L897 | Jacobian regularization — Optional small penalty on smooth numeric regions · Must not erase genuine thresholds, atoms or narrow class boundaries | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L899 | For hard thresholds, piecewise structure is useful. A globally smooth generator can fail to preserve tree-relevant decision boundaries. Regularize according to feature type and evidence. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **11.5 Training diagnostics** |  |  |  |
| L903 | Persist at least: all loss components; per-task/per-family loss; real and synthetic learner loss; hard/soft gap; native/Torch gap; per-head/backbone gradient norms; inter-objective cosine matrix; clipping/saturation … | Claude | NOT STARTED | [Training operations](#training-operations) |
| L905 | Use dataset-family aggregates for progress plots. Keep raw training scalar loss separate from the native DEV KPI. Checkpoint selection must read the latter under a fixed selection policy. | Claude | NOT STARTED | [Training operations](#training-operations) |
|  | **12.1 Family-level independence** |  |  |  |
| L911 | Audit datasets into families before train/DEV/CONFIRM assignment. Derivatives, target variants, duplicate uploads, row subsamples, and near-duplicate public benchmark versions belong to the same family. A new content … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L913 | All descriptors, feature-normalization statistics, probe states, codebooks and learned acquisition policies must be fit using meta-TRAIN only. DEV selects among declared methods. CONFIRM is used only for a frozen final … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **12.2 Episode distribution** |  |  |  |
| L917 | Retain balanced regression/binary/multiclass exposure, but report the actual achieved task mix after transformations and failures. Use a diversity schedule over: | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L919 | FIT row counts: small, medium, and selected large episodes, for example 32–128, 256–1,024, and 2,048–8,192. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L920 | Logical width: small and wide; keep categorical groups intact during column cropping. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L921 | Signal strength: null, weak, moderate, strong. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L922 | Functional structure: linear, monotone, thresholds, smooth nonlinear, sparse interactions, pure parity/XOR, high-order interactions. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L923 | Covariates: independent, correlated, multimodal, nonlinear manifolds, discrete/continuous mixtures. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L924 | Noise: homoskedastic, heteroskedastic, asymmetric, heavy-tailed, label noise. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L925 | Missingness, rare categories/classes, constraints, and declared group/time structure. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L927 | These size bins are proposed pilot settings. The packet's own n-dependent quantization and shrinkage must see the actual support size. Do not train exclusively on 384-row estimates and expect it to treat million-row … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L929 | The existing source starts with a 20% synthetic-prior probability, but its no-repeat queue can fall back to priors when real tables run out. Log this explicitly. Either reset a declared epoch with independent supports, … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **12.3 Augmentations with semantics** |  |  |  |
| L935 | Column permutation — Values, edges, schema and decoder output permuted together · Leaving pair endpoints or target coefficients in old order | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L936 | Row permutation — Physical groups and order contracts preserved · Treating temporal rows as exchangeable | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L937 | Class relabeling — Every class-conditioned tensor/output remapped · Numeric-ID statistics or class-0-only losses | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L938 | Category relabeling — Typed level dictionaries and probabilities remapped · Assuming label magnitude represents distance | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L939 | Monotone numeric warp — Rebuild normalization and affected descriptors; evaluate as a new episode · Reusing stale curves or quantized knots | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L940 | Continuous jitter — Only appropriate continuous channels, with scale defined · Destroying binary, categorical, integer or constraint structure | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L941 | Target swap — New explicit target declaration, source-family identity and all features rebuilt · Reusing target-dependent caches or leaking the old target | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L942 | Missingness perturbation — A declared simulation regime with masks retained · Claiming simulated missingness identifies the real mechanism | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L943 | Feature cropping — Whole parent groups and pair references retained · Partial one-hot groups or hiding dropped-feature penalties | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L945 | Give each independently randomized purpose a named RNG stream. Checkpoint resume must restore dataset queues, optimizer, EMA, RNG streams, feature dropout, layout choice and accumulation state. Noise pairing should … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **12.4 A useful curriculum** |  |  |  |
| L949 | First train on analytic synthetic families with known ground-truth functions and controlled feature collisions. Establish that richer descriptors actually allow recovery of the intended response and interaction. Then mix … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L951 | Synthetic priors are a way to teach structural coverage. They are not evidence of real-data lift. Maintain a real-only evaluation and a real-only training ablation. Include null-label datasets so the encoder learns to … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **13.1 Define “lift” precisely** |  |  |  |
| L957 | For lower-is-better loss, let L0 be the FIT-trained null predictor's test loss, LA the real-A-trained learner's test loss, and LS the synthetic-trained learner's test loss: | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L959 | `gain_real = L0 − LA` | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L961 | `gain_synthetic = L0 − LS` | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L963 | `retention R_S = (L0 − LS)/(L0 − LA)` when the real gain is informative. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L965 | Also report `LS−LA`, null-relative gain, absolute RMSE/MAE/log loss, and comparison against the previous DOPE artifact at identical bytes. Retention above one is possible through regularization and finite-sample effects. … | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L967 | Classify zero/negative/uncertain real gain explicitly using real-reference evidence only. Do not make a cell eligible because the new generator performs well. Do not average unstable ratios around a zero denominator as … | Claude | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L969–975 | Retain native v2 TRG, including the two selection auditors, independent MLP guard, headline 4× and matched 1×. | Claude | KEPT | Run `public-lite-replay` (`7c1f4598`, `primary.json` `59854869`, `compare.json` `f07f2647`): 125 informative 4× cells, `dope-2.0.0-small` −0.232 vs SDV GC −0.686 (Δ +0.454 [+0.342, +0.553]) & TVAE −1.070 (Δ +0.838 [+0.662, +1.042]); native & independent reducers match on 2,022 entries/arm. [Baseline replay](#baseline-replay) |
|  | **13.2 Respect the current candidate KPI contract** |  |  |  |
| L971 | The reviewed `contracts/dope-kpi.json` is **candidate**, `protocol_frozen: false`. Its selection statistic is the **twin-referenced retention gap**: | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
| L973 | `entry_TRG = clip(R_S, −1, 1.25) − R_B` for scored entries, with contract-defined treatment of failures. | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
| L975 | It averages sampling seeds per cell/auditor, takes the minimum of its two selection auditors (`linear-seeded-v1`, `gbdt`), then aggregates families within tasks and tasks. `mlp-seeded-v1` is a held-out auditor. Headline … | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
| L977 | The contract currently specifies 10,000 stratified family-bootstrap draws, Holm-adjusted comparisons, at least 3 training seeds, a 0.01 minimum delta, task/MLP/realism/privacy constraints, and an exact-copy constraint. … | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
| L979 | The KPI documentation itself notes limitations with few families and null-calibration behavior. Before a new confirmatory campaign, freeze a successor or this contract, calibrate the complete promotion rule under null … | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
|  | **13.3 Evaluation matrix** |  |  |  |
| L985 | Tasks — Regression, binary, multiclass; multilabel is separately declared and unsupported until implemented | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L986 | Byte caps — 512/1,024/2,048/4,096; optional larger-tier frontier | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L987 | Sample counts — 1× and 4× headline; development learning curve at 0.25×/1×/4×/16× | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L988 | Learners — Native linear/logistic, native GBDT, held-out MLP; exact external vendors when admitted | Claude | NOT STARTED | [Baseline replay](#baseline-replay) |
| L989 | Comparators — Analytic step zero, released V7, repaired V7, richer encoder, original compiler/MLC at actual comparable bytes | Codex | TESTING | [Baseline replay](#baseline-replay) |
| L990 | Controls — Real A, independent real twin, equal-byte raw subsample, bootstrap/copy controls, independent marginals, shuffled target, Gaussian copula | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L991 | Generative baselines — ARF/CTGAN/TabSyn/TabDiff or available pinned equivalents, with full per-table model bytes and tuning costs counted [R18,R19] | Codex | TESTING | [Published generators](#published-generators) |
| L992 | Splits — IID only where valid; group/time/OOD panels separately reported | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L993 | Seeds — Multiple training seeds and independent synthetic/learner seeds; paired noise/splits across arms | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L994 | Cohorts — n, p, task, class imbalance, rare levels, missingness, interaction strength, dependency, domain and support size | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L996 | The repository has exact external-process interfaces, but the current language policy disallows new nontraining Python. Use only admitted evaluator routes compatible with that policy. If an exact vendor training … | Codex | NOT STARTED | [Published generators](#published-generators) |
| L998 | Account for shared weights and access requirements of tabular foundation models. Report cold and warm runs and context limits. Pin versions rather than relying on a model name. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **13.4 Driver fidelity** |  |  |  |
| L1002 | Evaluate **the downstream models trained on synthetic data**, not only the generator's own importance proxies. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1004 | Train real-A, real-twin-B, and synthetic-S learners with matched protocols. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1005 | Define importance groups using A and its structural schema; never derive them from test labels. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1006 | Evaluate both real and synthetic learners on the same untouched T reference. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1007 | Compute grouped permutation importance using common permutations, and conditional importance as a separately labeled diagnostic. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1008 | Compare support-weighted ALE/response curves at common A-defined grid points, not at each model's unrelated synthetic quantiles. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1010 | Report: | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1012 | Spearman/Kendall or pairwise rank agreement among informative groups. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1013 | Top-k precision/recall/Jaccard, with a preregistered k policy. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1014 | Normalized importance-mass agreement; retain negative raw permutation estimates as diagnostics. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1015 | Main-effect curve error and supported derivative/sign agreement. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1016 | Interaction-edge recovery and interaction-strength agreement. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1017 | Subgroup driver agreement and stability. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1018 | Real-A versus real-B agreement as a reliability reference, with confidence intervals. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1020 | For strongly correlated predictors, several individual importance assignments can be valid. Report group-level preservation and within-group uncertainty instead of penalizing every swap as an error. ALE is a fitted-model … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1022 | Driver conditioning and driver evaluation must be distinct: training may use OOF support explanations; final scoring must use untouched T and independently fitted auditors. Do not grade the generator solely on the same … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **13.5 Distribution, calibration, support and efficiency** |  |  |  |
| L1028 | Point prediction — RMSE/MAE and task-specific errors; classification accuracy/balanced accuracy only as complements to proper losses | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1029 | Ranking — ROC-AUC, PR-AUC, top-decile lift for declared binary tasks; ranking improvements can coexist with poor calibration | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1030 | Probabilities — Log loss, Brier score, calibration intercept/slope, adaptive/fixed-bin reliability with counts | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1031 | Regression uncertainty — Pinball/CRPS, interval coverage and width, tail errors; actual synthetic-trained uncertainty estimator identified | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1032 | Marginals — Quantile/Wasserstein distance, atom/category mass error, rare-category support, missing-rate error | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1033 | Joint structure — Pair copula/contingency errors, nonlinear dependence, target-conditional structure, joint C2ST with held-out discriminator splits | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1034 | Local structure — Coverage/nearest-neighbor profiles and class-boundary preservation using the same type-aware metric | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1035 | Constraints — Illegal row rate, group validity, integer/ordinal/bound constraints, supported combinations | Claude | TESTING | [Realism and privacy](#realism-and-privacy) |
| L1036 | Subgroups — Worst supported-group predictive gap, calibration and driver errors; support and uncertainty for every group | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1037 | Reliability — Failure/over-budget/unsupported shares, estimator coverage, run-to-run variation and simultaneous intervals | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1038 | Costs — Exact artifact and shared weights bytes; encode CPU/wall time, p50/p95 latency, sample throughput, peak RSS and energy where measured | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1040 | A C2ST AUC near 0.5 can mean similar distributions or an underpowered discriminator. Compare real-real twins, report uncertainty and include more than one reasonable discriminator. C2ST does not substitute for transfer … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1042 | If a temperature or other calibration correction is fitted, fit it only on the declared training/calibration partition, then freeze it before T. Calibration methods established elsewhere still need DOPE-specific … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1044 | Retain the repository's empirical privacy and support checks. Richer statistics and more detailed conditional distributions may change memorization behavior even if no source rows are stored. A formal DP claim would need … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **13.6 Selection utility** |  |  |  |
| L1048 | Beyond TSTR, preserve the ability to choose useful learners/hyperparameters from synthetic data: | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L1050 | **Selection regret:** real-test loss of the method selected using synthetic validation minus real-test loss of the corresponding best allowed method. | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L1051 | **Learner-order agreement:** rank/pairwise agreement of learner performance after transfer, with all-tied cases undefined. | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L1052 | **Driver stability under model choice:** whether important feature groups persist across plausible selected learners. | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L1054 | Use the existing DVS selection surfaces where compatible. Selecting a model on synthetic rows sampled from the same generator can favor its biases; the real-test evaluation is the decisive measurement. | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
|  | **14.1 Stage experiments instead of one large combined change** |  |  |  |
| L1060 | Use matched initialization, support/query schedules, sample noise, learner seeds, training compute and evaluation cells wherever the comparison permits it. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1064 | A0 — Native analytic step zero · What does the compressed analytic program already achieve? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1065 | A1 — Released V7.1 checkpoint, unchanged · What does the actual shipped model achieve under this protocol? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1066 | A2 — Train a 42-input model with exact native extraction, corrected episodes and existing objective · How much was implementation/episode mismatch costing? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1067 | A3 — A2 with proper multi-family hard-transfer-oriented objective · Does objective alignment improve actual transfer? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1068 | A4 — A3 with 46-input native Q_nl, otherwise matched · What is the incremental value of existing quartile summaries? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1069 | B0 — Rich-encoder architecture, baseline evidence only, matched output capacity · Does architecture alone help? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1070 | B1 — B0 + target curves and class-conditional fingerprints · Does function shape add useful information? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1071 | B2 — B1 + interaction surfaces and exploration · Does explicit joint target information help? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1072 | B3 — B2 + OOF residual/disagreement and noise · Does learner-error structure add value? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1073 | B4 — B3 + categorical/missingness/dependence graph · Does better joint structure improve hard transfer and validity? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1074 | B5 — Best admitted B arm + value-aware allocation · Is it a better use of equal bytes? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1075 | C1 — Best admitted B arm + grouped driver/effect objective · Does driver fidelity improve without predictive harm? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1076 | C2 — Best admitted B arm + gradient/curvature sketch or gradient-matching loss, separately · Is the expensive learner-sensitivity signal worthwhile? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1078 | A4 is not compared to A2 as if only features changed when its objective also changes. B experiments must report any change in decoder expressivity separately. At least one pair of B arms should use identical decoder … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1080 | Run the X-adaptation/Y-adaptation 2×2 control on the strongest A or B configuration. This gives a direct answer to whether covariate refinement, conditional refinement, or their interaction drives gains or failures. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **14.2 Essential negative and capacity controls** |  |  |  |
| L1084 | Same richer model with optional feature values masked and availability preserved. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1085 | Same model with optional feature values shuffled across **meta-TRAIN** datasets within task/size strata. This is an ablation, never a deployment feature. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1086 | Equal-dimensional random-noise inputs. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1087 | Equal-parameter wider baseline encoder. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1088 | Same richer features with the existing decoder only. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1089 | Richer decoder with baseline features only. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1090 | Extra owner-compute spent on conventional fitting instead of descriptors. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1091 | Equal-byte larger-precision baseline where feasible. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1093 | These distinguish useful information from extra parameters, extra compute, task/size leakage, or decoder changes. Dataset-family IDs, source URLs, filenames and benchmark roles must not enter the model. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **14.3 Progressive feature selection** |  |  |  |
| L1097 | Do not launch 144 single-feature experiments and pick the lucky winner. Use a hierarchy: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1099 | Test the major bundles in the table above on development families. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1100 | Within an admitted bundle, use leave-family-out feature importance/ablation and redundancy analysis. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1101 | Test only a small preregistered shortlist of removals/additions. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1102 | Refit the chosen compact feature manifest with multiple seeds. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1103 | Freeze the model, feature manifest, codebook, normalizer, decoder, evaluator and selection rule before confirmation. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1105 | Drop highly redundant features when their removal does not materially reduce transfer/driver fidelity. Feature coverage is not a vanity metric. The final deployed manifest may contain far fewer recipes than the research … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **14.4 Statistical power and uncertainty** |  |  |  |
| L1109 | Datasets from the same family are correlated; so are splits, seeds and learners on one dataset. Report family-level paired differences and use the repository's family-aware procedure. Millions of episodes are not … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1111 | Estimate the standard deviation of paired family-level KPI changes on development data, then simulate power for the planned panel and full promotion rule. As a rough planning calculation only, a conventional paired … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1113 | `N ≈ ((1.96 + 0.84) * sigma_delta / delta_target)^2` | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1115 | independent units for 80% power at a two-sided 5% level. With `sigma_delta=0.10` and target difference `0.01`, that is about **784 independent units** before accounting for the actual stratified/multiple-comparison … | Claude | TESTING | [Power and promotion](#power-and-promotion) |
| L1117 | When the panel is small, show broad uncertainty, use null-control calibration, and avoid promoting on a lucky seed. Do not repeatedly inspect CONFIRM and keep training until it passes. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **14.5 Stop / continue rules** |  |  |  |
| L1121 | Continue a feature bundle only if it adds information beyond uncertainty/noise controls and improves at least one intended KPI without unacceptable harm to the others at matched cost. Stop or redesign when: | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1123 | Torch proxy loss improves but native hard TSTR does not. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1124 | Benefit disappears after native packet replay. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1125 | Improvement comes entirely from a few related dataset families. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1126 | Driver agreement increases only because all importance scores collapse toward uniformity or zero. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1127 | Lower C2ST is accompanied by worse conditional target structure. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1128 | Extra packet bytes, shared weights or hidden preparation dominate the claimed gain. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1129 | Unsupported, failed or dropped-class cases increase enough to explain a better average on the remaining cells. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1131 | An honest outcome may be that a repaired analytic-plus-sparse-interaction compiler is better than a larger neural encoder at 2 KiB. Keep that arm. The product objective does not require neural changes to win every … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **15.1 Proposed code ownership and seams** |  |  |  |
| L1137 | Paths marked **new** are proposed additions, not claims about existing files. Keep individual Rust source files within the repository's current source-size policy; split by statistical responsibility, not arbitrary line … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1141 | WP0: replay and audit — `single_encoding/{packet,scale_card,math,mod}.rs`, `s3_v7_batch_ops.py` · Native packet-to-training-record API; recipe/digest binding; explicit class/width refusal; typed episode repairs · Packet/tensor … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1142 | WP1: canonical normalized intake — `dope-norm/src/{nc2,fit,apply,card}.rs`, SDK intake · Frozen normalized-view type, parent groups/missing masks, fold-local transform identities · No query influence, transform parity, … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1143 | WP2: feature schema/runtime — `dope-core/src/prepare/` · **New** `prepare/meta_features/{mod,registry,types,marginal,missing,categorical,dependence}.rs`; **new** … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1144 | WP3: task structure — Landmark/report and prepare surfaces · **New** `prepare/meta_features/{target_curves,interactions,residuals,stability}.rs`; OOF summary extension · XOR/XNOR, … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1145 | WP4: neural encoder/objectives — `research/levels/` neural model/training boundary · **New** `v8_stat_encoder.py`, `v8_transfer_objectives.py`, `v8_train.py`; no standalone Python extractor/evaluator · Real training … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1146 | WP5: output format/runtime — `single_encoding/`, artifact/runtime modules · **New** versioned `single_encoding_v8/{wire,code,refine,sample,budget}.rs`; shared-decoder loader · Exact caps, no-source reload, … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1147 | WP6: allocation — Existing `budget.rs`, quantization and compiler paths · Support-only distortion measurement; deterministic baseline allocator; later learned allocator · Equal-byte attribution and packing … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1148 | WP7: drivers/evaluation — `dvs/drivers.rs`, `dvs/v3/`, `dope-arena/src/kpi/` · Grouped nonlinear importance/effect diagnostics, complete ledger, native arm adapters · Real-real reference, null/failure tests, … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1149 | WP8: benchmark/qualification — KPI docs/contracts and test-map · New experiment manifests, frozen feature schema, claims ledger, model card · All acceptance gates and disclosed unsupported cases | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1151 | Update `agent/owner-map.json` and `agent/test-map.json` for new source areas. Keep historic source, models, fixtures and receipts. No mass removal of retained Python is required to implement this spec; new nontraining … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **15.2 Proposed APIs** |  |  |  |
| L1155–1182 | extract_evidence, fit, config, landmarks, encode_normalized, evidence, base, weights, budget, unpack_for_training, bytes, expected_schema, sample_counter_range, packet, decoder, seed, start, count | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1184 | `NormalizedFitView` must be impossible to construct from an arbitrary matrix without the fit/coordinate/schema identities. The training adapter receives Q separately; the encoder function has no Q argument. The final … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1186 | The output codec should make the normalized-only / raw-portable profile explicit. A decoder missing a required shared weight digest must fail, not fall back to a different model. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **15.3 Proposed development configuration** |  |  |  |
| L1190 | This is an illustrative new configuration, not an existing runnable CLI or claim of tuned hyperparameters: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1193 | schema: dope.stat-transfer-experiment/v1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1194 | baseline_source: e3be91de02ac3cddc1bd22611bcc47d9195fae1a | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1195 | coordinate_recipe: nc2-bulk-tails-v1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1196 | source_scope: fit_only | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1197 | output_profile: normalized_only | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1198 | output_caps_bytes: [512, 1024, 2048, 4096] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1200 | evidence: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1201 | profile: structure | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1202 | recipe_manifest: core_structure_v1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1203 | covariance_shrinkage: development_fixed | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1204 | inner_folds: 5 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1205 | group_policy: preserve_or_refuse | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1206 | coarse_pair_grid: 2 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1207 | refined_pair_grid: 4 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1208 | coarse_pair_sample_cap: 2048 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1209 | refined_pair_cap: 64 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1210 | independent_pair_exploration_fraction: 0.25 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1211 | response_curve_bins: 8 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1212 | missing_semantics: explicit_state | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1213 | category_semantics: typed_group | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1214 | uncertainty: independent_group_resampling | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1216 | encoder: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1217 | width: 160 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1218 | blocks: 3 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1219 | heads: 4 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1220 | edge_conditioning: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1221 | class_tokens: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1222 | feature_family_dropout: development_fixed | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1224 | training: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1225 | seeds: [101, 202, 303] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1226 | real_family_sampling: balanced | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1227 | synthetic_prior_fraction_target: 0.20 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1228 | stop_or_reset_on_real_queue_exhaustion: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1229 | train_from_native_roundtrip: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1230 | hard_forward_labels: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1231 | common_random_numbers: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1232 | gradient_clip_norm: 1.0 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1233 | ema: baseline_recipe | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1234 | objective: multi_learner_transfer_with_fidelity_constraints | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1235 | optional_gradient_surgery: disabled | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1236 | optional_sam: disabled | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1238 | evaluation: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1239 | native_exact_packet: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1240 | matched_rows_multiplier: 1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1241 | primary_rows_multiplier: 4 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1242 | selection_auditors: [linear-seeded-v1, gbdt] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1243 | held_out_auditor: mlp-seeded-v1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1244 | external_auditors: admitted_exact_versions_only | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1245 | current_contract: contracts/dope-kpi.json | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1246 | final_protocol_status: must_be_frozen_before_confirm | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1249 | Fill all `development_fixed` fields from the declared DEV search, then serialize their exact numerical values before final training/confirmation. A config with unresolved fields is not runnable campaign admission. Small … | Claude | NOT STARTED | [Training operations](#training-operations) |
|  | **15.4 End-to-end training pseudocode** |  |  |  |
| L1253–1279 | T_A, A_norm, Q_norm, OOF, evidence, base, code, forward_state, S, losses, optimizer_step_with_declared_gradient_approximations, export_shared_weights, score_native_packets_on_DEV_with_full_ledger, … | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1281 | The exact native roundtrip is the forward-value authority. The Torch autograd graph uses the declared derivative approximation through quantization/discrete decisions; this pseudocode does not imply that Rust packet … | Claude | NOT STARTED | [Training operations](#training-operations) |
|  | **16.1 Deterministic engineering tests** |  |  |  |
| L1289 | Target/row custody — Explicit target-last declaration; shuffled query targets cannot change input features or packet; group overlap rejected | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1290 | Normalization — FIT-only affine target, out-of-range feature tails, constant/all-missing columns, unknown categories, repeat-apply identity | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1291 | Native training parity — Compare actual Rust packed/unpacked tensors with training inputs across real fixture shapes and type patterns | Codex | TESTING | [Hard forward](#hard-forward) |
| L1292 | Type boundaries — 12/13/16/17 unique levels, rare atoms after row 256, binary/int columns under augmentation, missing-versus-zero | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1293 | Packet boundaries — Exactly cap−1/cap/cap+1 bytes, changing rank/knot/column layouts, schema overhead, CRC, invalid scale/nonfinite fields | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1294 | Class overflow — More classes than supported, class absent from FIT or Q, reordered labels; no silent class-0 mapping | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1295 | Width pressure — p=1, p=48, p>48, wide one-hot groups and constrained pairs; explicit complete-schema or refusal policy | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1296 | Permutations — Row order invariance, column equivariance, category/class relabeling, edges and output schema permuted consistently | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1297 | Eigen-gauge — Factor sign flips and repeated-eigenvalue rotations do not change intended distributional semantics | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1298 | Statistical edge cases — Ties, tiny support, constant target, all-same class, perfectly dependent columns, singular Gram matrices | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1299 | Reliability — Undefined estimate versus observed zero; unavailable SE versus zero variance; group-bootstrap reproducibility | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1300 | Cache invalidation — Changed FIT rows/target/transform/fold/recipe/seed/binary cannot reuse stale evidence | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1301 | Decoder isolation — Hide/remove access to private rows and descriptor cache; recipient still samples from packet plus pinned runtime | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1302 | Row locality — Same seed/counter range across batch sizes, worker counts and call order under the declared numeric contract | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1303 | Gradient behavior — Tiny-fixture finite differences where smooth; quantized/hard-forward checks; ST bias documented; no NaN under singular/rare cases | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1304 | Resume — Interrupted versus uninterrupted updates/checkpoint state, including queue, noise, EMA, dropout and accumulation | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1305 | Failure accounting — Failed/unsupported/over-budget/timeouts remain scheduled; no empty-test or partial-ledger success | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1306 | Historical compatibility — Existing DSE7/DML/DPK fixtures still decode under old recipes; new files do not change old expected bytes | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L1306 | Retain historical packet and receipt readers within their existing versioned scope. | Codex | KEPT | Run `codex-root-rust@7c1f4598` (339/339 root Rust tests pass incl. DPK3.2 & v1 receipts) + `ops/ci/engine.sh@3fb802fd` (655/655 passed) + `v3-report-dev-s5-8@3fb802fd` (108/108 packets SHA-identical). [Recipient behavior](#recipient-behavior) |
|  | **16.1 Deterministic engineering tests** |  |  |  |
| L1308 | Use small synthetic fixtures with known answers. New nontraining tests are Rust. Python tests are limited to genuine neural training/autograd/resume responsibilities under the repository policy. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **16.2 Scientific diagnostic suite** |  |  |  |
| L1314 | XOR versus XNOR — Collision of current marginal/additive summaries · New signed pair features distinguish them and recover opposite conditional effects | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1315 | Product plus distractors — Sparse interactions at zero/weak marginal signal · Exploration finds the informative pair without marginal-only selection | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1316 | Same marginals, different copulas — Tests multivariate structure · Dependence evidence separates distributions; joint sampler improves pair fidelity | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1317 | Mixture with correlated classes — Tests multimodal and target-conditioned structure · Class-conditioned covariates and calibration are preserved | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1318 | Heteroskedastic regression — Tests noise beyond a single SD · Conditional scale/quantiles improve without biasing the mean | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1319 | Skewed/heavy-tailed residuals — Tests symmetric Gaussian limitation · Tail/interval statistics improve at matched point-prediction quality | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1320 | Informative co-missingness — Tests missingness as a signal · Mask relationships and mask-conditioned prediction survive | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1321 | High-cardinality nominal — Tests arbitrary integer encoding · Relabeling categories leaves results unchanged within tolerance | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1322 | Rare class / rare valid category — Tests support and overflow policy · Correct class accounting; no silent disappearance or category reassignment | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1323 | Strongly correlated interchangeable drivers — Tests importance ambiguity · Group fidelity remains high; individual ranks carry uncertainty | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1324 | Null or permuted target — Tests feature mining optimism · No confident fabricated target structure; null-control behavior calibrated | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1325 | Shifted time/group split — Tests leakage and IID assumptions · Valid splits, honest degradation, no use of future/test targets | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1327 | Some expected behaviors are implementation requirements; improvements in retention are empirical hypotheses. Do not turn qualitative expectations into prefilled pass results. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **16.3 Commands and test routing** |  |  |  |
| L1331 | These existing targets were identified in the repository. They are commands for the implementation team, **not tests executed for this report**: | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1333–1338 | cargo test --manifest-path engine/Cargo.toml --locked -p dope-norm --test nc2 cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test single_encoding_v7 cargo test --manifest-path engine/Cargo.toml … | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1340 | Add focused targets for the new feature/codec/training seams, verify nonzero test counts, and follow current repository `just fast`, `just check`, and engine/full-check routing before a proposed implementation commit. … | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **16.4 Acceptance gates** |  |  |  |
| L1344 | **Gate 1 — trustworthy inputs.** Native extraction/normalization authority, immutable split custody, all critical type/label/byte cases covered, no query-derived features, no silent class/column corruption. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1346 | **Gate 2 — informative new evidence.** Collision fixtures distinguish target structures; feature bundles add value beyond masked/random/equal-capacity controls on development families. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1348 | **Gate 3 — train what ships.** Actual native bytes, hard samples, pinned shared weights, full schema/cost accounting, and documented training/native numerical parity. Tolerances are set per operation before checking … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1350 | **Gate 4 — better transfer.** Frozen native KPI comparison against analytic step zero and released V7, at equal caps, with multiple training seeds and paired family uncertainty. Apply the frozen promotion rule; no claim … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1352 | **Gate 5 — useful drivers and calibrated distributions.** Improve or preserve declared group/curve/interaction fidelity, conditional calibration, validity and subgroup performance. Predefine noninferiority margins from … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1354 | **Gate 6 — operational value.** Preparation, weights, recipient time/RSS, and failure coverage fit the selected deployment profile. Public model card lists supported widths/classes, normalization, optional evidence, data … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1356 | A result can pass engineering gates and fail scientific efficacy. Keep those verdicts separate. The immediate success criterion is a trustworthy comparison; the desired product result is a statistically supported … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **17. Recommended delivery order** |  |  |  |
| L1362 | M0 — Pinned native replay and discrepancy fixtures · Current snapshot · Determine whether losses/metrics are comparing the same pipeline | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1363 | M1 — Core typed feature runtime and conditional fingerprints · M0 · Demonstrate new information on collision/structure fixtures | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1364 | M2 — Repaired V7 transfer baseline · M0 · Establish effect of parity and objective changes before richer architecture | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1365 | M3 — Rich encoder + bounded interactions/noise packet · M1, M2 · Measure feature value and decoder value separately | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1366 | M4 — Categorical/missing/dependence and driver extensions · M3 · Improve mixed-data validity and explanations without transfer harm | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1367 | M5 — Cost-aware feature/byte allocation · M3, M4 · Retain the smallest useful evidence manifest and efficient packet frontier | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1368 | M6 — Frozen external validation and confirmation · Admitted M2–M5 arms · Establish exactly which claims are supported | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1370 | Do not estimate calendar time before profiling the native feature extractor and exact evaluator availability. The critical path is parity → conditional/interaction evidence → a decoder that can use it → hard native TSTR … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1372 | The most promising deliverable is a compact generator that preserves the **functions, uncertainties, interactions and support regions that downstream learners need**. Extra descriptors are valuable when they make those … | Owner | NOT STARTED | [Execution scope](#execution-scope) |

</details>

### RichStats

<details>
<summary>Source proposals and DEV test routing</summary>

| Source | Proposal | Owner | State | DEV test |
| --- | --- | --- | --- | --- |
|  | **1. Decisions to implement first** |  |  |  |
| L19 | 1 — Unify normalized coordinates and make training consume the native codec's reconstructed inputs · Current normalization and feature extraction have several distinct semantics. Extra features cannot repair a train/deploy mismatch. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L20 | 2 — Add signed, class-aware interaction summaries, with exploration beyond individually strong columns · Marginals, correlations, linear coefficients, and univariate residual means cannot identify pure interactions such as XOR. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L21 | 3 — Replace four in-sample residual means with supported, cross-fitted conditional mean, variance, and class-probability summaries · The generator needs the shape of the target relationship and its uncertainty, not just a table-level accuracy score. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L22 | 4 — Preserve missingness, nominal groups, discrete atoms, and class identities explicitly · Imputation, continuous treatment of category codes, and unsupported-class remapping can destroy both fidelity and useful drivers. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L23 | 5 — Turn the existing landmark panel into structured learner-response inputs · Keep per-feature/per-region responses, residual disagreement, learning curves, and uncertainty instead of reducing everything to a … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L24 | 6 — Add a private rich encoder and a versioned compact residual code · V7's recipient model sees packet-reconstructible statistics. Rich features must affect serialized state; they cannot remain hidden … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L25 | 7 — Match deployed hard-row semantics and add a genuinely diverse training-learner portfolio · The current fast loss is principally one linear-plus-random-feature ridge system; it is not broad downstream learner coverage. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L26 | 8 — Align driver preservation and joint fidelity with real-vs-real reference variability · Good marginal appearance or linear coefficient rank alone does not establish transferable explanations. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L27 | 9 — Allocate bytes by measured reduction in downstream loss · Fixed priorities between knots, rank, and target information leave potentially valuable information out. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L28 | 10 — Run staged, equal-budget ablations before expanding training compute · The latest committed V7 selection note reports no group-level improvement over the analytic baseline beyond noise on the small DEV … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L30 | The first substantial experiment should compare **current packet inputs**, **richer conditional inputs**, and **richer conditional plus interaction inputs**, all at the same measured packet budget and with identical … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **2.1 The objective** |  |  |  |
| L52 | The normalized real test table must never be used to fit \(N_A\), extract input descriptors, choose packet contents, tune downstream models, or choose a checkpoint. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **2.2 Distinct source paths: do not mix their claims** |  |  |  |
| L65 | V7.2/V7.3 training source — 46 column values: the original 42 plus four nonlinear residual bins; default width 128, 2 layers, 4 heads, target hidden width 24 · New experiments must explicitly identify checkpoint architecture and feature schema. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L68 | Training DEV monitor — Python research diagnostics, median retention and coefficient correlations; some skipped cases · It is different from the native scoreboard and must not decide scientific claims by itself. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **2.3 Existing features worth preserving** |  |  |  |
| L81 | **Action:** Reuse existing native measurement machinery and retained reports. Add new versioned outputs and improve estimators; do not rebuild the same information under different names. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **2.5 The current V7 information path** |  |  |  |
| L120 | These are source defaults, not a claim that every reported checkpoint was trained with them. A run manifest must identify the values actually used. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **3. Concrete issues to fix before attributing lift to new features** |  |  |  |
| L124 | These findings come from source inspection. Their effect sizes require controlled measurement. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L128 | FIX-01 — `_fit_only_scale` calls itself NC2-compatible, but it uses median/IQR-or-SD scaling, asinh beyond magnitude 3, and clipping to ±6. `dope-norm::nc2` implements a different bulk/tail map into unit coordinates, nominal groups, missing slots, and affine target scaling. · Give every map an unambiguous recipe ID. Choose one normalized product coordinate system; train and evaluate with its exact native … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L129 | FIX-02 — Python packet extraction and Rust extraction differ in details: discrete detection uses ≤16 uniques in up to 256 rounded rows versus ≤12 uniques over native scaled values; FP16 endpoint/span representation differs; native scaling quantizes scales to BF16; wide-column selection and layout selection differ. · Use native pack→parse as the input authority. Keep a Python differentiable decoder mirror, with explicit parity fixtures. Do not … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L130 | FIX-03 — Four Q_nl bins use residuals of an analytic model fitted on the same rows. For multiclass, residuals are scalar centered true-class error. · Add OOF residual summaries; retain class-specific signed information. Keep in-sample/full-fit summaries separately identified if … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L131 | FIX-04 — V7 learns \(P(X)\) through a single factor-Gaussian dependence model plus marginal transforms. · Rich non-Gaussian statistics require a decoder with capacity to represent them. Add bounded mixture/conditional-dependence … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L132 | FIX-05 — Native scaling imputes nonfinite values; V7 packet inputs lack a general missingness process. · Carry observation masks through extraction and generate valid missingness states. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L133 | FIX-06 — Wide native column selection uses Pearson correlation with scalar target values; nominal class IDs make this score label-order dependent. · Replace with class-invariant, interaction-aware selection or explicitly report unsupported width. Retain a uniform exploration … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L134 | FIX-07 — Low-level native class fitting can retain only a limited class set and map an unmatched class to index zero; training uses class crops, and query remapping also has a zero fallback. The auto wrapper has its own eligibility logic. · Test every public and low-level entry point. Refuse unsupported class vocabularies or implement an explicit class-token extension. … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L135 | FIX-08 — A small duplicate-group count triggers a row-random split fallback in the episode builder. · Preserve duplicate/entity groups even when inconvenient. Return “insufficient independent groups” or use a qualified smaller split. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L136 | FIX-09 — Gaussian jitter is appended to all fit columns after scaling. · Apply augmentations by semantic type; do not turn categorical, binary, missing-indicator, or ordinal states into continuous noise. | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L137 | FIX-10 — The optional tree-visible loss uses a few positional column shifts and a class-zero scalar, rather than all-class conditional pair structure. · Select informative plus exploratory pair edges and use class-aware joint targets. Respect padded class masks. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L138 | FIX-11 — Scalar loss clamps create zero-gradient regions; some probability proxies are obtained by clipping ridge outputs. · Use proper classification solvers, smooth robust regression losses, stable solves, and gradient diagnostics. Keep reporting metrics … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L139 | FIX-12 — Losses pool differently across tables, columns, and tasks; large-width tables can dominate some terms. · Reduce within table first, then family/task. Log actual realized episode counts and optimizer weights. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L140 | FIX-13 — Skipped/invalid episodes and partial microbatches can change requested quotas; throughput counts use nominal microbatch sizes in parts of the trainer. · Count completed examples, failures, and independent physical tables separately. Synchronize rank-level skip decisions. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L141 | FIX-14 — A strict one-pass real-data queue eventually switches remaining training to procedural priors. · Treat this as an explicit curriculum arm; compare to bounded family-balanced replay. “Zero repeat” is not inherently the most … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L142 | FIX-15 — The current fixed byte-selection order is a heuristic, and training's estimated packet size omits some deployment-dependent distinctions. · Measure actual bytes and choose payload components using held-out fit-internal utility estimates or a validated learned allocator. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L143 | FIX-16 — Train-loss improvement, shallow-monitor retention, coefficient rank, and native TRG are different quantities. · Name every metric precisely and select with the native, decoded-artifact scoreboard. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L145 | Additional references: [normalization](https://github.com/neverhuman/dope/blob/e3be91de02ac3cddc1bd22611bcc47d9195fae1a/engine/dope-norm/src/nc2.rs), [scale card and layout](https://github.com/neverhuman/dope/blob/e3be91 … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **4. Why input information matters more than another layer** |  |  |  |
| L149 | Consider balanced, independent \(X_1,X_2\in\{-1,+1\}\) and two tasks: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L151–154 | \[ Y_+=\mathbf1[X_1X_2>0],\qquad Y_-=\mathbf1[X_1X_2&lt;0]. \] | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L156 | Repeat each of the four feature combinations equally often. Both tasks have: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L158 | Identical \(X\) marginals and dependence. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L159 | Identical 50/50 class priors. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L160 | Zero population linear feature-target slopes. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L161 | Zero univariate target-residual mean in each supported bin of either feature. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L163 | Consequently, V7-style marginals, linear coefficients, priors, and per-column residual means cannot distinguish the two relationships in this construction. A neural network receiving only those summaries has no … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L165–168 | A signed interaction statistic can distinguish them: \[ \mathbb E[(2Y-1)X_1X_2]=+1\quad\text{or}\quad-1. \] | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L170 | This is an analytical identifiability example, not a claim of measured benchmark performance. Add an exact balanced-grid codec fixture to verify the collision and its resolution in the implementation. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L172 | Three consequences: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L174 | **Dependence strength is insufficient.** Mutual information, HSIC, distance correlation, and an accuracy score can report strong relationships without specifying their sign or functional shape. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L175 | **Univariate screening is insufficient.** Every single column can look uninformative even when a pair is decisive. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L176 | **The decoder must support the represented relationship.** Information about heteroscedasticity is wasted if the generator has only a global noise scale. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L178 | The design below therefore includes effect-shape summaries, interaction locations and signs, uncertainty, and a matching generative representation. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **5.1 The required workflow** |  |  |  |
| L199 | The normalizer may inspect only the declared owner fit partition. The recipient learns in the resulting normalized coordinates. The owner applies that same frozen map before using the returned learner on real rows. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **5.2 Distinguish four objects** |  |  |  |
| L205 | Private normalization card — Owner · Raw-unit mappings and private dictionaries, with recipe/schema identity | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L206 | Normalized typed table — Owner · Values, observation masks, semantic feature-group identities, explicit target | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L207 | Rich descriptor cache — Owner · Statistical estimates, uncertainty, private measurement lineage | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L208 | Public/recipient packet — Recipient · Only required schema, quantized generative state, dependency identity, and bounded metadata | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L210 | A normalized-only successor should not silently embed raw medians, raw category strings, column names, or raw-unit inverse scales merely because legacy DSE7 does. Raw-unit synthesis remains a separate legacy/explicit … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L212 | If the owner holds the inverse map, synthetic-to-real use does not require sending it. If a deployment explicitly needs raw-unit generation, charge and assess the complete required mapping. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **5.3 Preserve type and missingness semantics** |  |  |  |
| L216 | Numeric values: use the selected FIT-frozen normalization recipe. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L217 | Regression target: use a strictly affine map when the goal includes preserving means, residual variance, and MSE interpretation. A nonlinear target transform requires a separately defined prediction/inverse-estimation … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L218 | Nominals: opaque categories with declared vocabulary, unknown, and missing states. Never interpret code magnitude as distance. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L219 | Ordinals: preserve the declared order and distinguish it from nominal IDs. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L220 | One-hot groups: retain group membership and sample one valid state, rather than independent dummy columns. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L221 | Missingness: keep the observation mask independently of imputed numeric values. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L222 | Constants/all-missing columns: explicit types/statuses; missing estimates are not measured zeros. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L223 | Declared identifiers, entity/time columns, and excluded columns: use their allowed roles. Do not silently promote them into predictors. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L225 | A fixed derivation such as rank scores or spline bases may be computed **inside the private analysis view** using \(A\). It does not authorize fitting another public normalization map on synthetic rows or test rows. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **5.4 Fold-local preprocessing for supervised feature extraction** |  |  |  |
| L229 | Outer normalization and inner landmark fitting have distinct responsibilities. For an honest inner-CV learner score, fit learned preprocessing and hyperparameters on that fold's training rows, then transform its held-out … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L231 | For descriptors displayed in a common coordinate grid, map each fold's estimated response back into the owner \(A\)-coordinate grid, or compute the grid from fold-training rows and retain its identity. Do not use global … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L233 | An already-normalized upload whose transform was fitted on the entire dataset cannot be retroactively called leakage-free. Mark provenance unknown or obtain a fit-only version before making a generalization claim. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **5.5 What is allowed to influence an episode** |  |  |  |
| L237 | During meta-training, its support \(A\) supplies encoder inputs; a disjoint query \(Q\) supplies supervised loss. Both belong to meta-training sources. Query labels are loss targets, never descriptors. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L239 | During meta-development, the same support/query structure is used without optimizer updates. Final confirmation families and final test labels enter only the final frozen evaluation. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L241 | If a table is cropped, its target changes, or its fit rows change, its supervised descriptors must be recomputed for that episode or retrieved from an exactly matching cache. A whole-table descriptor cache cannot be … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **6. Ranked feature families** |  |  |  |
| L247 | The ranking below is an engineering priority for DOPE, not an empirically established ordering of lift. “New to V7” may already exist partially in another repository path. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L251 | 1 — Signed supervised pair/interaction summaries · Which columns interact, where, and in which direction · P1; bounded pair screen plus selected pair grids | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L252 | 2 — Cross-fitted residual response and conditional noise · What the analytic target model misses; uncertainty as a function of features · P1; reuse OOF landmark predictions | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L253 | 3 — Class-conditional distributions and residual vectors · Minority-class structure and multiclass geometry without arbitrary ID ordering · P1; count/quantile accumulation | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L254 | 4 — Missingness and nominal/discrete structure · Informative absence, atom masses, category interactions, valid one-hot states · P0/P1; mostly streaming | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L255 | 5 — Learner-response and regularization-path summaries · What different learning biases see in this table · P1; reuse/extend landmarks | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L256 | 6 — Nonlinear dependence and conditional association · Relationships invisible to Pearson/Spearman alone · P1; capped dCor/HSIC and rank estimators | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L257 | 7 — Feature-group driver stability and response curves · Which explanatory signals are reproducible, including correlated groups · P1/P2; OOF perturbation cost | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L258 | 8 — Joint geometry, effective rank, and redundancy · Whether a low-rank copula is adequate; redundant and degenerate directions · P1; shrinkage covariance and bounded sketches | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L259 | 9 — Local neighborhoods and boundary structure · Thin boundaries, rare islands, local noise, minority overlap · P2; bounded neighbor sample | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L260 | 10 — Tail co-occurrence, atoms, and multimodality · Rare regimes and non-Gaussian joint behavior · P1/P2; support-gated | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L261 | 11 — Gradient and sufficient-statistic sketches · How training on the data changes several small learners · P2; high-potential experimental family | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L262 | 12 — Sampling uncertainty, drift, and quantization sensitivity · Which estimates and packet components deserve confidence and bytes · P1; cheap bootstrap subsets / validation | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L263 | 13 — Group/time/environment structure · Dependence and heterogeneity that IID summaries hide · P2; only for explicitly declared roles | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L264 | 14 — Higher-order information and topology · Residual structure beyond pairwise information · P3; expensive, hypothesis-driven | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L266 | **Default implementation:** ranks 1–8 plus inexpensive uncertainty and quantization features. Add ranks 9–14 when an ablation shows useful incremental information. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **7. Feature catalog and estimator requirements** |  |  |  |
| L270 | Notation: | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L272 | \(n\): rows in the owner fit partition; \(p\): semantic feature groups; \(K\): target classes. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L273 | \(M_j\): observation/missingness indicator, with convention specified in the schema. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L274 | \(U_j\): FIT empirical-rank view; ties use mid-distribution ranks unless a method specifies otherwise. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L275 | \(\hat f_{-k}\): learner fitted without fold \(k\). | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L276 | \(r_i=y_i-\hat f_{-k(i)}(x_i)\) for regression; \(r_{ic}=\mathbf1[y_i=c]-\hat p_{-k(i),c}(x_i)\) for classification. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L277 | \(n_s\): bounded row sketch; \(E\): selected pair edges. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L278 | “P1” is the first feature release; “P2” is the next experiment; “P3” is optional research. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L279 | Every estimate carries valid/status/support information. Unsupported quantities are masked, never assigned a favorable score. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L281 | The 120 entries below are **feature families**, frequently yielding multiple coordinates. The bounded first-release tensor contract is in section 9; do not concatenate every possible coordinate from this catalog. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **7.1 Size, support, data quality, and schema** |  |  |  |
| L287 | A01 — log row count, log group count, log \(p/n\), task, target cardinality · Calibrate shrinkage, complexity, and sampling uncertainty · P1; global | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L288 | A02 — Per-group observed count, missing fraction, all-missing flag · Prevent imputed values from becoming apparent evidence · P1; stream | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L289 | A03 — Unique-value count/ratio and effective discrete support \(\exp H\) · Separate continuous-looking atoms from true continuous values · P1; declare estimator/sketch precision | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L290 | A04 — Largest atom mass, total atom mass, support below \(k\) · Represent zero inflation, spikes, and fragile categories · P1; observed values only | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L291 | A05 — Constant and near-constant status, robust spread · Prevent meaningless correlations and wasted capacity · P1; tolerance depends on recipe | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L292 | A06 — Duplicate and near-duplicate group-size summaries · Estimate effective independent evidence and prevent split leakage · P1; physical lineage remains private | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L293 | A07 — Semantic kind, group width, ordinal status, unknown/missing states · Ensure valid categorical and integer synthesis · P0/P1; schema-derived | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L294 | A08 — Normalization clipping/tail-extension mass and collision rate · Detect structure destroyed by preprocessing · P1; distinguish true atom from clipping | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L295 | A09 — Missing/unknown target support and unseen-class policy · Preserve task identity and label mapping · P0; invalid targets do not become class zero | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L296 | A10 — Feature-selection coverage and discarded-group mass/utility · Reveal the cost of width reduction · P1; never call selected-column fidelity full-table fidelity | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **7.2 Univariate shape after normalization** |  |  |  |
| L302 | B01 — Quantiles at a fixed 17-point grid, with supported tails · Preserve shape more directly than a normality decision · P1; distinguish observed vs imputed values | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L303 | B02 — Median, trimmed mean, IQR, MAD, SD/IQR ratio · Robust center, bulk spread, tail contamination · P1; omit redundant affine invariants if provably constant | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L304 | B03 — Bowley skewness and a quantile-based tail-weight ratio · Stable asymmetry and tail shape for small samples · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L305 | B04 — Winsorized standardized third/fourth moments · Complement robust quantiles · P1; retain winsorization identity | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L306 | B05 — L-moment ratios / quantile-integral shape summaries · Stable shape information with less extreme-value sensitivity · P2 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L307 | B06 — Histogram entropy on FIT-frozen bins, bias correction, support · Discrete/continuous concentration; complexity allocation · P1; bin count enters recipe | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L308 | B07 — Tail exceedance rates and supported tail conditional means · Preserve rare but consequential rows · P1; thresholds fixed from fit, not test | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L309 | B08 — Atom locations and masses in normalized coordinates · Continuous interpolation cannot preserve point masses correctly · P1; bounded atoms, pooled remainder | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L310 | B09 — Dip/multimodality statistic and 1-vs-2 component held-out gain · Identify mixtures and separated regimes · P2; do not interpret nonrejection as unimodality | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L311 | B10 — KS/Cramér–von Mises/Anderson–Darling distance to fitted reference families · Identify failures of a Gaussian-like marginal assumption · P2; calibrated null if parameters are estimated | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L312 | B11 — Empirical characteristic-function coefficients at fixed frequencies · Compact multiscale shape sketch · P2; real/imaginary parts, shared frequencies | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L313 | B12 — Local quantile-spacing ratios · Reveal gaps, saturation, integer grids, tail steepness · P1; mask zero denominators | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L315 | If normalization is a full marginal rank transform, many marginal shape features become almost constants. Measure them in the normalized product coordinates and/or retain a separately named, owner-private pre-transform … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **7.3 Single-feature target relationships** |  |  |  |
| L321 | C01 — Signed Pearson and Spearman; tie-aware Kendall where affordable · Linear and monotone direction · P1; not scalar-coded multiclass | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L322 | C02 — Directional Chatterjee \(\xi(X_j,Y)\) and reverse direction · Nonmonotone functional dependence · P1/P2; proper tie handling and applicability | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L323 | C03 — Distance correlation · General nonlinear association · P1; cap \(n_s\), subtract/record permutation-null bias | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L324 | C04 — Normalized HSIC at several fixed/kernel-scale settings · Nonlinear association for numeric and nominal kernels · P1; bounded low-rank/RFF approximation | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L325 | C05 — Mutual information on discrete or rank-binned views · Redundant/nonlinear signal · P2; smoothing and permutation baseline; avoid one estimator for mixed types | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L326 | C06 — CV univariate spline, monotone, and stump gain · Actual out-of-sample usefulness of simple effect shapes · P1; extend existing probes | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L327 | C07 — Coefficients of a fixed orthogonal basis against target/residual · Preserve sign and shape that association strength discards · P1; Legendre/Hermite/hinge/Fourier choices versioned | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L328 | C08 — FIT quantile-bin target means and class probabilities · Coarse response shape suitable for tiny packets · P1; shrink by support | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L329 | C09 — Missingness-to-target effect · Whether absence itself is predictive · P1; learned inside fit only | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L330 | C10 — Monotonicity score and supported sign changes of the response curve · Choose monotone versus nonmonotone components · P2; empirical predictive relation, not causal direction | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L332 | Use equality-kernel HSIC, one-vs-rest effects, or probability-vector responses for nominal targets. A correlation with arbitrary class codes is not a valid class-invariant target descriptor. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **7.4 Residual shape and heteroscedasticity** |  |  |  |
| L338 | D01 — OOF binned residual mean \(E[r\mid X_j\in b]\) · Where the analytic target base is biased · P1; signed and support-shrunk | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L339 | D02 — OOF binned residual second moment/variance · Which regions need more or less target noise · P1; separate mean bias from variance | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L340 | D03 — OOF binned residual absolute deviation and quantiles · Non-Gaussian/asymmetric noise · P1/P2 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L341 | D04 — Association of \(r^2\) and \(\lvert r\rvert\) with each feature · Cheap heteroscedasticity screen · P1; label it model residual structure | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L342 | D05 — White/Breusch–Pagan-style auxiliary-regression effect and robust uncertainty · Structured variance misspecification · P2; predictive effect first, p-value secondary | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L343 | D06 — Residual-vs-prediction calibration curve · Under/overconfident or biased conditional models · P1 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L344 | D07 — Residual local autocorrelation over a feature-space neighbor graph · Missing local structure · P2; distinguish from temporal autocorrelation | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L345 | D08 — Residual tail frequency and positive/negative tail imbalance · Heavy-tailed conditional output · P1/P2; supported cells only | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L346 | D09 — OOF learner residual covariance/disagreement · Different learners see different missing structures · P1; preserve learner IDs | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L347 | D10 — Noise estimates from close-neighbor outcome differences · Approximate local irreducible variation · P2; not a proven Bayes-noise estimate | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L349 | For D02, compute \(E[r^2]-E[r]^2\) with stable nonnegative shrinkage. A large residual variance can reflect omitted predictors or model bias; it is not automatically irreducible noise. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **7.5 Class-conditional structure and imbalance** |  |  |  |
| L355 | E01 — Smoothed class priors, entropy, effective class count, minority support · Generate appropriate class coverage · P1; retain actual class set | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L356 | E02 — Per-class feature means, quantiles, missingness and atom mass · Explain which values characterize which classes · P1; class tokens, not pooled maxima only | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L357 | E03 — Between-class/within-class scatter and regularized separation · Compact multivariate class geometry · P1/P2; singular covariance handled | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L358 | E04 — Per-class OOF recall/log loss/Brier loss · Detect minority failure hidden by average accuracy · P1; split-aware support | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L359 | E05 — OOF confusion structure or its low-rank sketch · Which classes are interchangeable/difficult · P2; class permutation equivariant | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L360 | E06 — Signed per-class residual basis coefficients · Which class probability needs correction in which region · P1; sums constrained across classes | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L361 | E07 — Margin and entropy distributions conditioned on class · Boundary thickness, ambiguity, calibration · P1/P2 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L362 | E08 — Class-conditional pair dependence · Same overall copula can hide very different class structure · P1/P2 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L363 | E09 — Minority/majority neighbor overlap · Rare islands versus broadly overlapping classes · P2 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L364 | E10 — Multilabel prevalence and label-dependence graph · Future multilabel support · P3; a separate task contract, not multiclass aliasing | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L366 | Do not force class balance in a faithful generator. A deliberately rebalanced training set is an explicit utility mode with documented class weights or prior correction, measured separately. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **7.6 Pairwise and higher-order interactions** |  |  |  |
| L372 | F01 — Signed residual product moment \(E[r\,\phi_j(X_j)\phi_k(X_k)]\) · Detect pure interactions and their direction · P1; centered basis; include nonscreened exploratory pairs | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L373 | F02 — Supported 4×4 pair-bin joint counts · Non-Gaussian \(X_j,X_k\) structure · P1; sparse/shrunk representation | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L374 | F03 — Pair-bin target/residual means or class probabilities · Target relation changes across combinations · P1; class-aware | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L375 | F04 — Pair learner improvement over additive learner on identical OOF rows · Empirical interaction usefulness · P1/P2; paired uncertainty, nested selection | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L376 | F05 — Conditional association after residualizing against a small feature set · Redundancy and incremental predictive structure · P2; descriptive, not a generic conditional-independence test | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L377 | F06 — Pair dependence in residual variance · Interactions controlling uncertainty, not just mean · P2 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L378 | F07 — Pairwise tail co-exceedance at several quantiles · Extreme events coupled across columns · P1/P2; finite-threshold dependence, not asymptotic tail-index certainty | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L379 | F08 — Nominal-nominal smoothed contingency, corrected Cramér's V, log-odds · Valid categorical interactions · P1 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L380 | F09 — Numeric-nominal conditional quantiles and missingness · Mixed-type interactions · P1 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L381 | F10 — Triple moments or small parity probes on selected triples · Higher-order pure interactions · P2/P3; explicit budget | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L382 | F11 — Random projected joint-feature/target characteristic functions · Broad higher-order sketch beyond hand-selected pairs · P2 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L383 | F12 — Pair/triple selection stability across fit resamples · Suppress spurious discovered interactions · P1/P2 | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L385 | For F04, an improvement over an additive model is evidence for extra predictive capacity under that learner recipe, not a causal interaction or a unique functional ANOVA decomposition. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **7.7 Geometry, redundancy, and local complexity** |  |  |  |
| L391 | G01 — Shrinkage covariance/correlation eigenvalue shares · Choose copula rank and detect redundancy · P1; common observed-row policy | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L392 | G02 — Effective rank \(\exp[-\sum q_i\log q_i]\), stable rank, condition summary · Estimate concentration of information · P1; definitions versioned | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L393 | G03 — Per-column redundancy-group membership and residual unexplained variance · Avoid spending equal bytes on duplicates · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L394 | G04 — Regularized partial-correlation edges · Candidate conditional linear dependencies · P2; not causal graph discovery | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L395 | G05 — Local intrinsic-dimension estimates at multiple neighborhood sizes · Curved/thin manifold structure · P2; unstable in small/high-dimensional samples | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L396 | G06 — Neighbor-distance quantiles, hubness, density heterogeneity · Detect sparse pockets and distance concentration · P2 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L397 | G07 — Cross-validated clusterability and cluster-conditional moments · Decide whether multiple regimes are useful · P2; stable clusters only | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L398 | G08 — Neighborhood target variation / class disagreement · Local learning difficulty and boundaries · P1/P2 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L399 | G09 — Low-order conservation, sum, ratio, and inequality residuals · Preserve learned numerical constraints · P2; declared hard constraints take precedence | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L400 | G10 — Dependency-component sizes and edge-spectrum summaries · Global structure of the selected graph · P2; retain edge identities in model input | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L402 | Pairwise-deletion covariance need not be positive semidefinite. Use one declared missing-data estimator and a PSD-preserving regularization policy. Record repair size if a projection is required. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **7.8 Richer landmark outputs** |  |  |  |
| L408 | H01 — Existing gains, uncertainty, ranks, regularization choices · Preserve useful current conditioning · P1; versioned conversion, no invented missing values | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L409 | H02 — Per-feature OOF gain / leave-group-out decrement · Bind learner responses to feature groups · P1/P2 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L410 | H03 — Ridge/logistic coefficient paths across a fixed \(\lambda\) grid · Collinearity, signal stability, regularization need · P1; per-fold fit | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L411 | H04 — Learning curves at fractions 1/8, 1/4, 1/2, 1 · Sample efficiency and expected benefit of synthetic row count · P1/P2; minimum support | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L412 | H05 — Tree split thresholds, supported leaf masses, leaf target distributions · Direct information for tree-learnable structure · P1/P2; bounded and serialized only after compression | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L413 | H06 — Residual disagreement and prediction covariance across learners · Complementary inductive biases · P1 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L414 | H07 — Calibration curves/slopes for classification probes · Prevent confident but wrong synthetic labels · P1 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L415 | H08 — Sensitivity to irrelevant-column injection and feature removal · Fragility and useful denoising opportunities · P2; fit-local experiment | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L416 | H09 — Hyperparameter response surface summary · Which algorithm settings this dataset supports · P2; very small fixed grid | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L417 | H10 — Per-regime and minority-support landmark scores · Aggregate gain can hide failure on valuable subpopulations · P1/P2 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L419 | A tree's training impurity importance is a weak driver target: it can favor high-cardinality columns and reflect correlated-feature substitution. Prefer OOF perturbation or grouped removal. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **7.9 Driver and effect-shape preservation** |  |  |  |
| L425 | I01 — Grouped permutation importance on OOF rows · Direct relationship to prediction loss · P1/P2 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L426 | I02 — Conditional-permutation importance with a qualified conditional sampler · Reduce off-manifold perturbation artifacts · P2; report sampler quality and assumptions | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L427 | I03 — Importance mean, dispersion, sign/stability, selection frequency · Distinguish stable drivers from noise · P1/P2 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L428 | I04 — Real-vs-real driver variability across independent fits · Sets an attainable fidelity tolerance · P1 evaluation; optional fit-internal input | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L429 | I05 — ALE curves on supported normalized bins · Preserve effect shape without unrealistic marginal sweeps · P2; predictive effects only | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L430 | I06 — Pair ALE / held-out interaction response sketches · Preserve explanatory interactions · P2/P3 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L431 | I07 — Agreement across linear, tree, and neural driver rankings · Exposes model-dependent explanations · P2 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L432 | I08 — Group-level importance for correlated/one-hot features · Avoid penalizing harmless within-group attribution swaps · P1 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L434 | Feature-importance agreement alone does not establish causal driver transfer. Preserve stable predictive explanations and call them that. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **7.10 Gradient, information, and uncertainty summaries** |  |  |  |
| L440 | J01 — \(\Phi^\top\Phi/n\), \(\Phi^\top y/n\), target moments for several fixed bases · Directly describes linear/ridge learning in each feature space · P2; sketch big matrices | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L441 | J02 — Mean gradients at a small fixed bank of probe parameters · Captures what data would teach a learner · P2; local computation, compressed inputs | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L442 | J03 — Gradient covariance / diagonal Fisher-like summaries · Where signal/noise and curvature concentrate · P2; not automatically a calibrated Fisher matrix | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L443 | J04 — Score-function residual summaries for logistic/multiclass probes · Match class-aware training dynamics · P2 | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L444 | J05 — Bootstrap/subsample dispersion and effective support per feature family · Gate unreliable estimates · P1; grouped resampling when required | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L445 | J06 — Null-calibrated effect \(t-\operatorname{mean}(t_{\rm perm})\) and null scale · Differentiate estimator bias from association · P1/P2; preserve raw estimate too | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L446 | J07 — Descriptor sensitivity to row/column subsampling · Generalization across support sizes · P1/P2 | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L447 | J08 — Quantization error and objective sensitivity per candidate block · Allocate bytes to useful, fragile signals · P1/P2 | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L448 | J09 — Analytic-generator residual discrepancies in marginals, joint cells, learner responses · Tell the encoder exactly what its baseline fails to preserve · P2; use fit-internal held-out rows | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L449 | J10 — Conditional uncertainty about the missing generator structure · Encourage conservative residual corrections · P2; ensemble/posterior approximation, not certainty | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **7.11 Time/group data and optional expensive tests** |  |  |  |
| L455 | K01 — Within/between-group variance and intraclass correlation · Entity/group roles are explicitly declared; identifiers stay private | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L456 | K02 — Group-specific effect heterogeneity and leave-group-out stability · Deployment requires new-group generalization | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L457 | K03 — Lag dependence, autocorrelation, partial autocorrelation · Chronology and sampling interval are valid, with forward splits | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L458 | K04 — Change-point/seasonality summaries · A temporal synthesis contract exists; arbitrary CSV order is not time | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L459 | K05 — Feature/target shift across declared fit environments · Environments lie wholly inside allowed fit data | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L460 | K06 — Persistent-homology or sophisticated topology sketches · Cheaper local/mixture features demonstrably leave useful gaps | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L461 | K07 — Copula-family goodness-of-fit, higher-order cumulants · Supported sample size and a decoder that can use the result | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L462 | K08 — Learned density-ratio / two-sample discrepancy against the analytic base · A separate fit-internal calibration split prevents discriminator overfit | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L464 | No finite collection of observational statistical tests identifies every causal relation or missing-data mechanism. A rejection of MCAR does not prove MAR; failure to reject does not establish MCAR. Do not infer a causal … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **7.12 Influence, conditioning, and probe sensitivity** |  |  |  |
| L468 | These optional descriptors diagnose whether a small subset of observations or fragile decision regions controls learning. They should follow the core conditional/interactions release, not delay it. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L472 | L01 — Ridge leverage-score quantiles and maximum, using a regularized shared basis · Detect learning dominated by unusual feature directions · P2; aggregate values only; never export row identities | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L473 | L02 — Approximate leave-one-group-out parameter/prediction influence distribution · Identify fragility to a few entities or observations · P2; qualify approximation against actual deletions on small fixtures | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L474 | L03 — Fit-internal held-out covariance shrinkage response · How strongly a small sample supports a dependence estimate · P2; use a fixed small shrinkage grid and proper covariance score | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L475 | L04 — Complete/quasi-separation diagnostics and coefficient-norm growth in logistic probes · Detect unstable probability fits despite apparently easy classification · P2; distinguish separability from robust out-of-sample … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L476 | L05 — Local input-gradient norm/anisotropy of shared differentiable probes · Summarize predictive smoothness and thin decision directions · P2; normalized numeric coordinates; masks/categories handled by … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L477 | L06 — Input-Hessian diagonal/mixed-derivative sketches of smooth probes · Describe curvature and local interactions missed by a global linear view · P3; model-dependent estimate, not a property uniquely … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L478 | L07 — Cross-fitted quantile-regression pinball-loss and coverage profiles · Distinguish target distributions requiring asymmetric/quantile-specific predictions · P2; several fixed quantiles, supported regions … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L479 | L08 — Feature measurement-noise response curves · Show which declared numeric measurement errors destroy predictive information · P2; only plausible, type-valid perturbations; … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L480 | L09 — Prequential target-code-length curve under fixed learner/ordering recipes · Summarize how quickly a probe learns the relationship from growing prefixes · P3; train each prefix before scoring new rows; average … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L481 | L10 — Controlled target-corruption sensitivity of small probes · Characterize dependence on label quality and robustness of learned structure · P3; perturb fit labels in private diagnostics only; … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L483 | For L01, a basic numeric-basis score is \(h_i=\phi_i^\top(\Phi^\top\Phi+\lambda I)^{-1}\phi_i\), with an explicitly declared intercept convention. Summarize its distribution, not the high-leverage records themselves. For … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **8.1 Cross-fitted residual profiles** |  |  |  |
| L489 | Use three folds in the initial CPU recipe, optionally five for larger tables. Keep duplicate/entity groups intact; time tasks use qualified forward folds. Reuse OOF predictions from the existing native learners. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L491 | For regression and a fixed bin \(b\) of feature \(j\): | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L493–498 | \[ \hat m_{jb}=\frac{\sum_{i\in b}w_i r_i+\alpha m_0}{\sum_{i\in b}w_i+\alpha}, \qquad \hat v_{jb}=\frac{\sum_{i\in b}w_i(r_i-\hat m_{jb})^2+\alpha v_0} {\sum_{i\in b}w_i+\alpha}. \] | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L500 | Use separate count, mean, second moment, and uncertainty estimates. The displayed variance formula is a regularized descriptive estimator, not an unbiased variance formula. Validate it under small samples and … | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L502–508 | For classification, retain the vector: \[ \hat m_{jbc} =\frac{\sum_{i\in b}w_i[\mathbf1(y_i=c)-\hat p_{ic}]+\alpha\,0} {\sum_{i\in b}w_i+\alpha}. \] Center across classes after any approximation so that residual … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L510 | Initial settings: 4/8 bins as an ablation, pseudo-count \(\alpha=8\) as a starting value, minimum effective support 16 for full-resolution estimates. At smaller support merge bins and expose the resulting resolution; do … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **8.2 Signed interaction summaries and pair discovery** |  |  |  |
| L514–518 | Use centered, bounded basis functions on normalized numeric/rank views: \[ b_j(x)=\{z_j,\ z_j^2-E[z_j^2],\ \tanh z_j,\ \mathbf1[U_j>1/2]-1/2,\ \text{two hinge terms}\}. \] | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L520–525 | For an edge \(j,k\), calculate a small selection of: \[ C_{jk,ab,c} =\frac1n\sum_i b_{ja}(x_{ij})b_{kb}(x_{ik})r_{ic}. \] Regression has one residual output. Store support, variance, and sign; a maximum absolute value … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L527 | **Candidate policy:** | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L529 | At \(p\le48\), all simple pair products are feasible on a capped row sketch: at most 1,128 unordered pairs. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L530 | At larger \(p\), combine existing dependence edges, promising residual edges, random-projection candidates, and a deterministic uniform pair sample. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L531 | Reserve at least 25% of the pair budget for exploration independent of univariate target scores. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L532 | Select expensive pair grids using one inner split; estimate their effects on another, then aggregate cross-fitted results. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L533 | Retain at most \(E=4p\) directed edges for the first encoder, with an undirected-pair identity. Use a smaller wire-edge subset chosen by the byte allocator. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L535 | No bounded search guarantees finding all high-order interactions in a very wide table. Record the candidate coverage and test adversarial XOR/rare-interaction cases. | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
|  | **8.3 Uncertainty is an input, not a footnote** |  |  |  |
| L539 | Each estimator returns: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L541–551 | value, valid, uncertainty, effective_support, status | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L553 | A measured zero, a negative association estimate, an uninformative model gain, and a failed calculation are different states. New model input normalization is fitted only on meta-training families. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L555 | Use bounded signed transforms such as \(x/(s+&#124;x&#124;)\), log1p for positive counts/scales, and appropriate probability transforms. The scale \(s\) is a pinned feature-family parameter or a meta-training-only statistic. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L557–561 | For noisy estimates, a shrinkage gate can use \[ \tilde t=\frac{\tau^2}{\tau^2+\widehat{\operatorname{Var}}(t)}t, \] where \(\tau\) is fitted from meta-training data. Supply both estimate and reliability; avoid hard … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **8.4 P-values and statistical tests** |  |  |  |
| L565 | Feed **effect size, sign, sample size, support, and uncertainty** first. P-values depend heavily on \(n\), selected hypotheses, fitted reference distributions, ties, and multiple comparisons. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L567 | If p-values are retained: | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L569 | Name the null hypothesis and exact calibration method. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L570 | Use restricted permutations for groups/time where exchangeability permits them. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L571 | For \(R\) permutations use \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\); never zero. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L572 | Log-transform with a finite floor tied to permutation resolution. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L573 | Use FDR or familywise correction for declared discovery decisions, with appropriate dependence assumptions. Raw p-values can remain diagnostic. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L574 | Do not backpropagate a hard “reject/not reject” decision. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L575 | Calibrate KS/AD tests when distribution parameters were fitted from the same sample. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L576 | Do not interpret a nonsignificant C2ST as proof that two distributions match. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L578 | The estimator references [R1–R5] support the statistical building blocks, not a claim that adding every test improves DOPE. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **8.5 Basis sufficient statistics and gradient sketches** |  |  |  |
| L582–586 | For ridge on a fixed feature map \(\Phi\): \[ \hat\beta=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top y. \] Matching only marginals does not match this learning problem. Matching the Gram matrix, target cross-moment, target … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L588 | Use a small bank: linear, spline/additive, RFF, and selected interaction maps. Do not claim their moments are sufficient for arbitrary learners. An \(h\times h\) Gram matrix costs \(O(h^2)\); keep large versions private … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L590 | For neural/logistic probes, compute mean gradients at **initial and partially trained** probe parameters. At an exact empirical optimum the mean training gradient can be nearly zero, making it a poor dataset descriptor. … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L592 | Never send individual-example gradients or fitted per-table probe parameters as uncharged recipient dependencies. Gradient matching has a research precedent [R7]; its benefit under DOPE's tabular, byte-limited, … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **9. Bounded feature-tensor contract: RichStats V1** |  |  |  |
| L597 | The catalog contains more information than should be enabled in one experiment. Implement a compact core schema, with optional token groups for deeper measurements. The private descriptor is not constrained to 2 KiB; the … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **Conflicting release layout** |  |  |  |
| L599 | RichStats-specific distribution of block widths as the final ABI (superseded by frozen `contracts/evidence-layout-v1.json` 128/16/16/4/8/160; not run). | Antigravity | NOT STARTED | Superseded draft ABI — never run. [Evidence bundles](#evidence-bundles) |
|  | **9.1 Containers and dimensions** |  |  |  |
| L601 | All dimensions below are **proposed**, not current repository APIs. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L605 | Global estimates — 96 values · Support, target, geometry, fit reliability, cost/codec summaries | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L606 | Column-group estimates — `p × 128` · Core per-group values defined below | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L607 | Pair-edge estimates — `E × 32` · Endpoint IDs plus 32 edge measurements; IDs are separate | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L608 | Class tokens — `K × 16` · Prior, support, OOF difficulty, uncertainty; no semantic interpretation of class ID | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L609 | Class-column tokens — `K × p × 24` · Class-conditioned location/shape/residual summaries; sparse implementation permitted | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L610 | Landmark tokens — `L × 16` · Probe identity, gain, uncertainty, learning-curve and calibration summaries | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L611 | Optional sketch tokens — `q × 64` · Shared-basis moments, gradients, characteristic-function or geometry sketches | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L612 | Structural metadata — Variable · Group/type definitions, estimator recipes, masks, support, cost tier, normalization identity | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L614 | Start with `E ≤ 4p` directed edges, `L=23` existing landmark identities, and no optional sketch tokens. Missing measurements consume masks, not invented values. For wide tables use sparse/class-conditioned attention … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L616 | For each numeric estimate carry separate arrays for: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L618 | Value, finite FP32 on extraction. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L619 | Status enum: measured, uninformative, insufficient support, unsupported, failed, budget skipped, constant input. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L620 | Effective support. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L621 | Standard error or uncertainty surrogate, plus a validity flag identifying whether it exists. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L623 | A measured value can have unavailable uncertainty. Model projections embed statuses and transform support/uncertainty; do not assign `SE=0` to an unmeasured estimate. Private caches may quantize these arrays only after a … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **9.2 Exact 128-coordinate first-release column layout** |  |  |  |
| L629 | Support/type — 12 · Observed and missing fractions, unique and effective support, maximum atom mass, total tracked atom mass, constant indicator, … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L630 | Marginal quantiles — 17 · Fixed normalized-coordinate quantiles; unsupported endpoints masked | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L631 | Robust shape — 15 · Median, trimmed mean, IQR, MAD, SD, SD/IQR, Bowley skew, tail-weight ratio, two winsorized moments, two tail masses, two tail means, … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L632 | Univariate target — 20 · Signed Pearson/Spearman where applicable; two nonlinear effect estimates; six signed basis responses; four-bin target response; … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L633 | Residual response — 32 · Eight-bin OOF residual mean, variance, absolute deviation, and support; classification pooled values supplemented by class-column … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L634 | Local/redundancy — 8 · Mean/max absolute dependence, redundancy degree, residual variance after redundancy fit, two local-distance summaries, neighbor … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L635 | Driver/learner response — 16 · Four regularization-path responses, four OOF group-decrement summaries, four response-curve coefficients, four stability/reliability … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L636 | Codec sensitivity — 8 · Estimated distortion at selected quantizations, discrete snap sensitivity, support loss, marginal/rank/target residual errors, … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L637 | **Total** — **128** · Structural semantic type remains separate metadata | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L639 | Several entries are task-specific. “Four-bin target response” is a signed scalar for regression or binary tasks and masked/pooled for multiclass; class-column tokens retain the full class directions. Do not compute … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L641 | The 20-coordinate target block sums to `2+2+6+4+4+1+1=20`. The local and codec blocks need named coordinates in the versioned JSON schema; placeholders must not reach a production manifest. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L643 | Global 96-coordinate allocation: 16 support/schema, 16 target/difficulty, 16 dependence-spectrum/geometry, 24 aggregate learner-response, 12 estimation reliability, 12 codec/cost. Class and landmark tokens carry … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **9.3 Edge and class definitions** |  |  |  |
| L647 | A core 32-coordinate edge token contains: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L649 | 4 dependence measures: Pearson, Spearman, normalized HSIC, and distance correlation. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L650 | 8 signed residual-basis products. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L651 | 8 pair-grid coefficients in a fixed orthogonal basis. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L652 | 4 finite-threshold tail co-exceedances. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L653 | 4 missingness/atom/nominal interaction summaries. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L654 | 4 support, stability, and candidate-coverage summaries. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L656 | The eight pair-grid coefficients may represent target-free joint counts or supervised responses depending on a **typed edge-channel ID**. Do not overwrite one with the other. Where both are required, use two typed tokens … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L658 | A 24-coordinate class-column token starts with 8 conditional quantiles, 4 class residual-bin means, 4 bin supports, and 8 conditional location/spread/missingness/atom/reliability values. These are first-release … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L660 | Endpoint and class IDs are routing indices, not learned semantic embeddings tied to an arbitrary order. Permuting columns or class labels must produce the corresponding permutation of encoded outputs and generated … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **9.4 Identity, caching, and reproducibility** |  |  |  |
| L664 | A descriptor-cache key must include source version, exact fit-row identity digest, semantic schema, target identity, normalization recipe/state digest, feature recipe, candidate-pair seed, CV split digest, estimator … | Claude | TESTING | [Extraction identity](#extraction-identity) |
| L666 | Keep row-ID digests and data-derived cache keys private. Do not publish a hash of private rows as a substitute for a privacy analysis. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L668 | A native cache should contain typed numeric arrays and bounded metadata, not arbitrary Python objects. Validate array dimensions, finite values, enum values, and checksum before training. Schema changes create a new … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **10.1 Three tiers** |  |  |  |
| L676 | Fast — Support/type, moments, quantiles, missingness, cheap signed associations, shrinkage spectrum, existing cached landmarks · Streaming full fit rows for cheap summaries; capped 2,048-row joint sketch · Every eligible episode/table | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L677 | Standard — OOF residual summaries, class structure, signed pair screen, selected pair grids, small regularization paths · Three folds; ≤512 rows for quadratic kernels; ≤4p directed retained edges · First main training experiment | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L678 | Deep — Neighbor geometry, driver perturbations, richer learning curves, gradient sketches, resampling, selected triples · 4–8 fit resamples initially; fixed per-table work cap · Ablations and owner-side high-quality mode | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L680 | These are starting resource settings, not measured runtimes or statistically universal thresholds. Small datasets use all available independent rows. Samples must preserve declared group/time restrictions; … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L682 | A two-stage allocation is preferable: compute cheap uncertainty/difficulty information, then allocate additional measurement effort where it is likely to change the packet. An expensive estimate should be skipped when … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **10.2 Complexity and controls** |  |  |  |
| L686 | Streaming summaries: approximately `O(np)` plus quantile-sketch cost. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L687 | Full numeric covariance: `O(np²)`; switch to blocked/sketched operations for wide inputs. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L688 | Exact distance correlation/HSIC kernels: potentially `O(n_s²)` time and memory per comparison; use a bounded sketch or explicit approximation. Never enable this for every pair at full dataset size. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L689 | Full pair screens: roughly `O(n_s p²)` for a small fixed basis. Larger tables need candidate generation and bounded exploration. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L690 | OOF probe work: count **fitted probe instances**, not just distinct probe names. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L691 | Driver perturbations: count predictions and any refits separately. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L692 | Pair/triple selection must have its own budget; “top interactions” without a bounded search specification is not an implementation plan. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L694 | Report p50/p95 owner encode time, peak resident memory, core-seconds, descriptor-cache size, decoder time per 1,000 rows, and output size. Report cold and warm cache paths separately. No runtime target should be claimed … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L696 | Batch vectorized extraction by type and reuse OOF predictions, rank views, bins, kernel maps, and fold assignments. Avoid materializing repeated `n×p×p` tensors. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **10.3 Avoid biased progressive stopping** |  |  |  |
| L700 | A feature computed only on hard tables can accidentally become a table-family identifier. Include tier/missingness indicators, random feature-block dropout, and training examples at every supported tier. Evaluate … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L702 | Use predeclared limits for convergence and timeout. A failed or budget-skipped estimate is retained as that status. Do not remove the entire dataset from evaluation because the richer extractor was inconvenient. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **11.1 Separate the encoder information problem from the decoder capacity problem** |  |  |  |
| L708 | Two useful experiments serve different purposes: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L710 | **Compatibility control:** keep the current DSE7 decoder and packet layout, train a rich private encoder to predict the existing quantized knots/loadings/target fields. This asks whether better estimation alone helps. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L711 | **Recommended successor:** allow a new packet residual code and a typed decoder that can represent interactions, conditional noise, missingness, and discrete structure. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L713 | The first control can reveal a better estimator but cannot solve a representation collision if the decoder's output family cannot express the desired law. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **11.2 Private set/graph encoder** |  |  |  |
| L717 | Proposed first model: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L719 | Separate projections for global, column, edge, class, and landmark tokens. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L720 | Width 192, 4 attention heads, 3 residual blocks as an initial experiment; compare with a width/parameter-matched current model. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L721 | Sparse edge message passing between semantic feature groups. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L722 | Class-to-column cross-attention with shared class-token weights. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L723 | Attention pooling with multiple learned summary slots; retain per-column and selected edge outputs. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L724 | Masks/status/support embedded explicitly. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L725 | Separate task and byte-budget conditioning. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L726 | Output heads for analytic-base corrections, packet component allocation, and quantized residual codes. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L728 | Set-based attention has an established architectural precedent [R6]. Its use here is a proposed design choice, not evidence of improved DOPE transfer. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L730 | Keep a target-free distribution stream for support/marginal/joint `X` statistics and a supervised stream for `Y&#124;X` statistics. They may exchange information through an explicit ablated gate; if supervised information … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L732 | For very wide data, use group blocks and shared global summary slots so model compute does not grow quadratically with all expanded one-hot slots. Chunking must retain cross-block edges; a purely independent block … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **11.3 Symmetries and latent-factor gauge** |  |  |  |
| L736 | Preserve: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L738 | Row permutation invariance within an IID fit set. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L739 | Column-group permutation equivariance. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L740 | Nominal category relabeling equivariance. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L741 | Target-class relabeling equivariance. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L742 | Declared time/group semantics when exchangeability is intentionally absent. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L744 | Raw factor loadings `L` are not uniquely identified: sign flips, column rotations, and nearly repeated eigenvalues can represent the same covariance. A network that consumes raw loading coordinates may learn arbitrary … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L746 | Prefer invariant edge quantities from `LLᵀ`, spectral summaries, and a deterministic canonicalization where needed. Include factor-sign/rotation augmentation in compatibility experiments. Matching covariance or output … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **11.4 Decoder: small explicit components before a large generative model** |  |  |  |
| L750 | Use an analytic base plus compact learned corrections, with a measured no-regression fallback. Suggested component order: | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L754 | Numeric marginals — Monotone quantile spline plus explicit supported atoms · Quantiles, atom masses, tail statistics | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L755 | Feature dependence — Low-rank base plus a small mixture, or selected conditional residual edges · Joint geometry, non-Gaussian pair structure | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L756 | Missingness — Shared low-dimensional mask process plus per-group conditional probabilities · Mask associations and class/feature-conditioned missingness | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L757 | Nominal groups — Categorical probabilities and selected conditional logits · Contingency/class structure; valid group states | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L758 | Regression mean — Analytic base + additive response bases + selected pair terms + small shared nonlinear residual · Signed OOF shape and interaction inputs | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L759 | Regression noise — Positive feature-dependent scale, optional bounded residual mixture · Conditional variance/quantiles/tails | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L760 | Classification — Class-masked logits with additive/pair corrections and calibrated temperature · Full class-residual vectors and class tokens | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L762 | Do not add all components at once. First add signed pair terms and feature-dependent regression scale, because they directly correspond to identified missing information. Add multimodal `X` generation only after a … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L764–771 | A useful regression form is: \[ Y=\mu_{\rm base}(X)+\sum_j a_j(X_j) +\sum_{(j,k)\in E_{\rm wire}}h_{jk}(X_j,X_k) +u_\psi(X,z) +\sigma(X,z)\epsilon. \] Use centered bases and a fixed convention to reduce redundancy … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L773–778 | For classification: \[ P(Y=c\mid X)=\operatorname{softmax}_{c\in\mathcal C} \left[\ell_{{\rm base},c}(X)+a_c(X)+h_c(X)+u_{\psi,c}(X,z)\right]. \] Apply the class mask before normalization. Generate actual labels from … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L780 | A mixture model introduces component label symmetry; use permutation-invariant component matching for losses or a pinned canonical component ordering. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
|  | **11.5 Source-free generation** |  |  |  |
| L784 | The recipient must generate rows using only: | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L786 | The complete serialized packet. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L787 | A pinned shared decoder identified by a full cryptographic digest. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L788 | The public recipe/schema definitions. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L789 | An explicit seed and row count. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L791 | No training-time descriptor cache, private probe, normalization card, owner-side row, hidden per-table weight file, or retrieval service may be required. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L793 | A larger private encoder does not force a larger recipient decoder. Keep those sizes and latencies separate. Measure whether a decoder comparable to the existing few-megabyte checkpoint suffices before increasing it … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
|  | **11.6 Width and class limits are product decisions** |  |  |  |
| L797 | Do not advertise unrestricted tables if only 48 numeric slots and 8 classes are actually supported. Choose and document one of: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L799 | Exact support within the current envelope, with explicit rejection outside it. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L800 | Versioned dynamic group/class tokens, bounded by tested resource limits. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L801 | An explicitly partial selected-feature mode, with coverage and lost-utility metrics. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L803 | For the successor, prioritize full semantic-group coverage at 2 KiB on the measured supported envelope. Offer a larger budget or a declared lower-fidelity tier when the complete schema plus model cannot fit. Never … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **12.1 Treat the wire artifact as the learning bottleneck** |  |  |  |
| L809 | Every forward evaluation used for promotion must include: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L811–814 | rich descriptors → encoder outputs → quantize → serialize → parse → pinned native decoder → hard synthetic rows → downstream fit → real evaluation | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L816 | A differentiable training mirror can substitute smooth gradients, but it cannot substitute a different forward distribution. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L818 | Keep legacy DSE7/DPA1 readers and fixtures. A normalized-only format with different fields requires a new magic/version and an explicit normalizer-coordinate identity. Historical bytes and scoreboards remain unchanged. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **12.2 Illustrative 2,048-byte allocation** |  |  |  |
| L822 | This is an **engineering budget example**, not an implemented codec or a claim that every table fits it. It assumes 24 simple numeric groups, a binary target, no raw inverse map, short schema metadata, and a shared … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L826 | Header, version, dimensions, full shared-model digest, recipe IDs — 64 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L827 | Normalized semantic schema and group map — 96 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L828 | Marginal knots/atoms and group flags — 432 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L829 | Rank-four dependence representation — 96 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L830 | Base target coefficients and intercept/scale — 64 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L831 | Per-column residual codes: 24 × 16 int8 values — 384 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L832 | 16 selected pair components including endpoint IDs and coefficients — 192 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L833 | Global residual code — 128 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L834 | Missingness/conditional-noise fields — 192 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L835 | Quantization scales, component directory, integrity fields, slack — 400 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L836 | **Total complete artifact** — **2,048** | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L838 | The schema and component definitions must prove the assumed capacity. Large category vocabularies, high class count, additional columns, and raw-unit mapping change the allocation. If an outer envelope is required, its … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L840 | Shared weights are a separately disclosed reusable dependency, not “free model size.” Report first-use bytes and amortized bytes over a declared number of datasets. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **12.3 Rate-distortion allocation** |  |  |  |
| L844–849 | Let `c` identify a candidate packet component or precision increment. Estimate: \[ u_c=\frac{\widehat{\Delta L_{\rm transfer,c}}+ \eta\,\widehat{\Delta L_{\rm fidelity,c}}}{\Delta B_c}. \] Here positive improvement is a … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L851 | Use fit-internal cross-validation for allocation targets. An encoder can learn predicted utility per byte from meta-training episodes, then allocate without an expensive downstream search at deployment. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L853 | Recommended progression: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L855 | Deterministic greedy allocator over a small component menu, with measured marginal utility. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L856 | Compare with current fixed-priority allocation at identical file sizes. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L857 | Train an allocator to imitate successful choices. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L858 | Consider relaxed gates/entropy models only if their complexity is justified. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L860 | Interactions between components make greedy utility imperfect. Evaluate a few joint alternatives for rank, marginal precision, and supervised residual capacity. Reserve minimum schema/support and marginal quality before … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **12.4 Quantization-aware training** |  |  |  |
| L864 | Train with the exact supported rounding/saturation conventions. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L865 | Use explicit per-block quantization scales; their bytes count. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L866 | Compare quantization noise and straight-through rounding gradients. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L867 | Apply exact quantization in validation and all final inference. | Codex | TESTING | [Hard forward](#hard-forward) |
| L868 | Parameterize monotone knots through positive increments; check their serialized reconstruction remains ordered. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L869 | Parameterize variance/scale positivity before quantization and validate after parsing. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L870 | Preserve categorical probability normalization and minimum valid support. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L871 | Measure utility sensitivity to each component's bit depth. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L873 | A bit-rate surrogate is insufficient for a fixed-size release claim. Measure the actual encoded file, including headers, class vocabulary, group maps, checksums, decoder identity, and required envelopes. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **12.5 Progressive budgets** |  |  |  |
| L877 | Evaluate 512, 1,024, 2,048, and 4,096 bytes. A common component ordering can support progressive truncation, but only call it a valid prefix format if every allowed prefix is independently parseable and contains all … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L879 | The headline should remain the existing 2,048-byte cell until a versioned decision changes it. A 4,096-byte winner does not establish a 2,048-byte improvement. | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **13.1 Primary downstream objective** |  |  |  |
| L886–892 | For episode support `A`, meta-training query `Q`, decoded synthetic rows `S`, and training learner `a`, define: \[ g_a= \frac{L_Q(\operatorname{Train}_a(S)) -\operatorname{stopgrad}[L_Q(\operatorname{Train}_a(A))]} … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L894 | The denominator is a positive, fit-derived task scale with a predeclared floor. For regression use a robust target-variation/null-loss scale; for classification use a bounded null cross-entropy scale. Do not divide … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L896–902 | Use mean and smooth worst-learner loss: \[ L_{\rm transfer} =(1-\rho)\frac1M\sum_{a=1}^{M}g_a+ \rho\,\tau\log\left(\frac1M\sum_{a=1}^{M}e^{g_a/\tau}\right). \] Evaluate this with log-sum-exp stabilization. Initial … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L904 | Use a randomly rotated subset of training learners per update, with periodic complete-portfolio updates. Log selection probabilities and actual weights. Keep at least one meaningful learner family entirely outside … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **13.2 Training portfolio and independent auditors** |  |  |  |
| L910 | Differentiable transfer — Ridge / properly fitted logistic-softmax · Stable baseline; direct gradient through a small solve | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L911 | Differentiable transfer — Spline/additive and RFF kernel-ridge/logistic variants · Nonlinear but smooth inductive biases | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L912 | Differentiable transfer — Small soft decision trees or differentiable oblivious-tree ensemble · Piecewise thresholds and interactions | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L913 | Independent native audit — Existing seeded linear and GBDT recipes · Preserve the declared native scoreboard | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L914 | Held-out generalization guard — Existing seeded MLP recipe · Remains outside training in the first successor campaign | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L915 | External audit — Named production learners such as CatBoost and selected tabular foundation-model implementations · Measures actual intended downstream use, with pinned versions and legitimate runtime access | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L917 | The current native `gbdt` auditor is not automatically a proxy for every CatBoost configuration. Name exact implementations. A successful soft-tree loss does not establish tree-ensemble transfer. | Codex | NOT STARTED | [Published generators](#published-generators) |
| L919 | If a later experiment trains through MLP probes, the MLP can no longer be called an unseen learner family for that campaign. Freeze a revised held-out-family contract before selection. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L921 | Use small fixed hyperparameter grids or randomized training-only settings. Downstream audit hyperparameters must be selected without test labels. Report both a controlled fixed-recipe comparison and a realistic … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L923 | Production evaluators, including classical fitting and metric calculation, must follow the Rust policy. Use available official native interfaces or an explicit supported adapter. Do not silently add a new Python … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **13.3 Auxiliary objectives** |  |  |  |
| L927–936 | Use: \[ L=L_{\rm transfer} +\lambda_cL_{\rm conditional} +\lambda_sL_{\rm statistics} +\lambda_gL_{\rm learning\ geometry} +\lambda_dL_{\rm drivers} +\lambda_rL_{\rm residual} +\lambda_bL_{\rm rate}. \] | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L938 | Every term is reduced to a per-table scalar before task/family balancing. Normalize loss scales from a fixed meta-training calibration window, then freeze those scales for a campaign. Otherwise the meaning of a weight … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L942 | Conditional — Real-query Gaussian/mixture NLL and robust mean loss, or proper class cross-entropy/Brier · A richer density does not by itself preserve downstream learning | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L943 | Statistics — Marginal quantiles/atoms, missingness, class priors, selected pair/class grids, actual multiscale MMD or sliced distances · Must include joint and conditional properties, not only pooled marginals | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L944 | Learning geometry — Shared-basis Gram/cross-moments and/or matched gradients · Sufficient only for the specified feature map/learner | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L945 | Drivers — Stable group-level response and loss-decrement agreement · Avoid treating unstable real-model importance as ground truth | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L946 | Residual trust region — Penalty on deviation from analytic base, scaled by evidence/reliability · Bounded corrections do not prove nonnegative downstream lift | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L947 | Rate — Estimated component bytes during training; hard measured-byte guard for selection · Surrogate rate is not the encoded-file length | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L949 | Begin with transfer, conditional, selected statistics, and a small residual penalty. Introduce learning-geometry and driver terms separately, with a loss-only ablation before combining them. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **13.4 Sufficient-statistic and gradient alignment** |  |  |  |
| L953–959 | For a shared feature map with an intercept and a fixed regularization convention, match: \[ \frac{\Phi_S^\top\Phi_S}{&#124;S&#124;},\quad \frac{\Phi_S^\top y_S}{&#124;S&#124;},\quad E_S[y],\quad E_S[y^2]. \] Use the same normalized-loss … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L961–970 | For differentiable probes at parameters `ω_t`: \[ L_{\rm grad}=\sum_{a,t} \left&#124; \frac{\nabla_\omega L_{S,a}(\omega_t) -\nabla_\omega L_{A,a}(\omega_t)} {&#124;\nabla_\omega L_{A,a}(\omega_t)&#124;_2+\epsilon_a} \right&#124;_2^2. … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L972 | Dataset condensation and kernel-ridge meta-learning motivate these targets [R7, R8]. Their results do not show that DOPE's fixed 2 KiB representation will preserve arbitrary learners. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **13.5 Driver preservation** |  |  |  |
| L976 | Train synthetic and real reference learners, and compare their behavior on a **common normalized real reference distribution**. During meta-training that reference comes from allowed query rows; during final evaluation … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L978 | First targets: | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L980 | Grouped permutation loss increments, with repeated permutations. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L981 | Centered ALE/response-curve coefficients in supported regions. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L982 | Signed linear/logistic effects where identifiable and stable. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L983 | Redundant-feature group importance, not arbitrary allocation within a collinear group. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L984 | Pair interaction response on the selected edges. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L986 | For training, use differentiable perturbation losses and smooth response summaries. For final reports, use the actual auditors and native metric implementations. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L988 | Weight agreement by real-vs-real reproducibility. A useful reliability factor is estimated from two independently fitted real reference learners or fit resamples. If real references disagree, report uncertainty and avoid … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L990 | ALE is preferable to unrestricted partial dependence for many correlated-input checks because it uses local conditional regions, but it still requires adequate support [R10]. Feature importance is model- and … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **13.6 Faithful synthesis versus utility-optimized training sets** |  |  |  |
| L994 | The default generator should preserve the normalized data-generating relationships, including noise and missingness. A utility optimizer can otherwise improve a downstream score by producing overly deterministic labels, … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L996 | Maintain two explicitly labeled modes if desired: | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L998 | **Faithful synthetic data:** predictive utility plus distribution/conditional/driver constraints. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L999 | **Training-set distillation:** intentionally altered rows/weights optimized for specified learners; fidelity and general-purpose reuse are separate claims. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1001 | Do not merge these results in one headline. A distilled set may be valuable, but it should not pass as a faithful substitute solely because a selected learner performs well. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **13.7 Constrained selection, not one opaque aggregate** |  |  |  |
| L1005 | Training needs a differentiable scalar; release selection needs interpretable gates. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1007 | Choose on the native primary transfer statistic, subject to complete-schema coverage, byte limits, realism, driver reliability, runtime/resource limits, and empirical disclosure diagnostics. Report the Pareto frontier … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1009 | An optional dual update for constraints can be tested after fixed weights are understood. No loss-balancing method substitutes for final independent measurement. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **14.1 Improvements to try first** |  |  |  |
| L1015 | **Common random numbers:** use the same generation noise, probe initialization, feature-map draw, and query partition for paired candidate/base losses within an update. Refresh randomness across updates; do not train … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1016 | **Antithetic continuous noise:** test paired `ε,−ε` draws where valid. Do not assume categorical antithetics are unbiased without defining the coupling. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1017 | **Proper solver gradients:** use Cholesky/QR or stable solves instead of explicit matrix inversion; keep solve operations at adequate precision. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1018 | **Smooth robust regression:** replace hard-clipped error with pseudo-Huber/Huber for training where appropriate. Keep unclipped MSE/RMSE for the declared audit. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1019 | **Stable logits:** train classification with proper log-softmax/cross-entropy, active-class masks, and fitted logistic/softmax probes. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1020 | **Reliable weighting:** reduce the influence of low-support noisy auxiliary targets without removing their uncertainty/status information. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1021 | **Per-table reduction:** prevent large `p`, `K`, or synthetic row count from changing the intended table weight. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1023 | EMA and gradient clipping already exist in the current trainer. Preserve and tune them as controls; do not describe them as newly invented improvements. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **14.2 Differentiation through fitting** |  |  |  |
| L1027–1033 | For ridge: \[ A=\Phi^\top W\Phi/n+\lambda D,\qquad b=\Phi^\top Wy/n,\qquad \beta=\operatorname{solve}(A,b). \] `D` should implement the declared intercept rule. Start with an unpenalized intercept and positive feature … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1035 | For logistic/softmax, use a fixed small number of damped Newton/IRLS or gradient steps with checked convergence. A short unrolled optimization is an approximation to the converged learner; quantify that approximation on … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1037 | Log condition estimates, jitter, solver failures, gradient norm through the fit, and residual norm of each solve. A “successful” finite loss with an unusable solver gradient should not count as a healthy training step. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **14.3 Hard sampling and quantization** |  |  |  |
| L1041 | Straight-through estimators have biased gradients. Their value must be established empirically: | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1043 | Forward-pass labels, atoms, categories, missingness, and packet values follow deployment semantics. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1044 | Backward pass may use a soft relaxation with an explicit temperature. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1045 | Compare temperature schedules and relaxation choices on the same native decoded-artifact metric. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1046 | Measure the gap between soft-row training evaluation and hard-row evaluation. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1047 | Use an exact-enumeration or Rao–Blackwellized control for small categorical supports when it is cheaper and well-defined. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1048 | For features expected to be differentiable, compare autograd with finite differences in tiny deterministic fixtures. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1050 | Do not eliminate label noise merely to make gradients convenient. A low-variance biased gradient and an unbiased high-variance gradient have different failure modes; report the actual hard-output result. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **14.4 Multiobjective gradient conflicts** |  |  |  |
| L1054 | Record per-loss gradient norms and pairwise cosine similarity on a small shared parameter subset. Distinguish three cases: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1056 | A loss is too large because of units/reduction. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1057 | Two losses genuinely trade off utility and fidelity. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1058 | A loss is noisy because the measurement is unreliable. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1060 | Fix scale/reduction first. Then compare one method at a time: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1064 | Fixed normalized weights — Baseline · Most interpretable | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1065 | GradNorm — Balance learned gradient magnitudes [R12] · Adds adaptive behavior; can overweight noisy tasks | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1066 | PCGrad — Project conflicting auxiliary gradients [R13] · Extra gradient work; does not guarantee the desired Pareto point | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1067 | Small auxiliary-gradient norm cap — Prevent fidelity/driver terms from dominating transfer · Simple heuristic; report cap and frequency | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1068 | SAM — Late optional robustness experiment [R14] · Extra passes; not a first response to missing input information | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1070 | Do not combine GradNorm, PCGrad, SAM, curriculum changes, and a new decoder in a single “improved model” experiment. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **14.5 Safe optimization and synchronized failures** |  |  |  |
| L1074 | Retain AdamW as the initial optimizer. Compare current defaults against tuned learning rate, warmup expressed in completed episodes, weight decay excluding biases/norms, and EMA decay with an interpretable effective … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L1076 | In multiworker training: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1078 | All workers agree on a step-validity flag before entering gradient collectives. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1079 | A nonfinite local loss/gradient triggers a synchronized action. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1080 | Count successful support/query examples and actual batch sizes. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1081 | Preserve RNG, sampler/queue state, scheduler, optimizer, EMA, model, and feature-schema identity on resume. | Claude | TESTING | [Training operations](#training-operations) |
| L1082 | Do not silently roll back only one worker. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1083 | Separate data failures, solver failures, model nonfinites, and collective failures. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1085 | First remove the known clipping and preprocessing issues. Larger optimizers will not recover information that never reaches the packet. | Claude | NOT STARTED | [Training operations](#training-operations) |
|  | **15.1 Split at the unit that can leak** |  |  |  |
| L1091 | Maintain distinct meta-training, meta-development, and untouched confirmation **source families**. A stem-based filename split is insufficient when renamed copies, alternate versions, task variants, or shared entities … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1093 | Build lineage from explicit source metadata plus normalized duplicate/near-duplicate evidence. Preserve original physical row identities privately. Use entity, time, and related-task grouping where declared. Overlapping … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1095 | Within each table: | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1097 | Owner support `A` fits the normalizer and descriptors. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1098 | Inner folds inside `A` build OOF features and allocate packet information. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1099 | Query `Q` supplies episode loss/evaluation only. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1100 | An optional real twin `B` supports the native baseline comparison under its declared protocol. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1101 | The final test partition is never an optimization/query partition. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1103 | Class-stratified sampling must preserve groups. If support is insufficient, record that condition; do not split duplicate groups or map an unseen class to a known one. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **15.2 Sampling and replay** |  |  |  |
| L1107 | Treat the existing task quotas and procedural mixture as a baseline, then test: | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1109 | Family-balanced sampling with inverse-frequency caps. | Claude | TESTING | [Episode stream](#episode-stream) |
| L1110 | Hardness strata defined from **fit-only/reference** features. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1111 | Controlled replay of real tables with fresh valid support/query splits. | Claude | TESTING | [Episode stream](#episode-stream) |
| L1112 | A bounded per-family replay cap and effective-unique-data logging. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1113 | Explicit proportions of IID, grouped, time-split, low-row, wide, imbalanced, and mixed-type episodes. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1115 | Compare zero-repeat and bounded replay at equal optimizer steps **and** equal CPU-hours. They answer different efficiency questions. Procedural samples are not new independent real datasets. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1117 | Do not let adaptation overfit only failures on a repeatedly inspected development panel. Keep a fixed audit subset and a separate training curriculum signal. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **15.3 Mechanism curriculum** |  |  |  |
| L1121 | Add targeted generators with known ground truth: | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1125 | Linear/additive smooth response — Existing analytic base and response-bin accuracy | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1126 | Balanced XOR/XNOR; k-way parity — Signed interaction information and univariate-screening failure | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1127 | Rare localized interaction — Pair discovery and minority support | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1128 | Heteroscedastic and asymmetric noise — Conditional variance/tail heads | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1129 | Multimodal feature regimes — Single-copula limitations | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1130 | Informative missingness — Observation-mask model | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1131 | Nominal permutations and one-hot groups — Type invariance and valid sampling | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1132 | Collinear substitute drivers — Group-level importance versus unstable individual importance | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1133 | Class-prior/label permutations — Class equivariance and calibration | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1134 | Pure noise / no-signal targets — Shrinkage, false structure, and safe fallback | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1135 | Train-only outliers and rank ties — Robust normalization and atom treatment | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1136 | Explicit group/time drift — Appropriate splits and optional environment descriptors | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1138 | Generate continuous difficulty parameters and held-out mechanism combinations. Avoid making each mechanism recognizable from an irrelevant signature such as a unique row count, column order, or fixed seed pattern. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **15.4 Training stages** |  |  |  |
| L1142 | **Stage A — Exact contract and analytic controls.** Native normalization, extraction, codec, and sampler fixtures pass. Establish step-zero decoded-artifact results before learned changes. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1144 | **Stage B — Rich descriptors with current output capacity.** Freeze the decoder architecture; train the private encoder to improve existing packet fields. Include an oracle-statistics control to estimate whether input … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1146 | **Stage C — Minimal additional capacity.** Add signed pair residuals and conditional regression scale. Train on mechanism fixtures plus family-balanced real episodes. | Claude | TESTING | [Episode stream](#episode-stream) |
| L1148 | **Stage D — Learner portfolio and driver constraints.** Add soft-tree transfer, then driver/learning-geometry terms separately. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1150 | **Stage E — Rate allocation and quantization.** Train/validate across the byte grid, hard sampling, and complete serialized files. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1152 | **Stage F — Frozen comparison.** Freeze code, normalizer, feature recipe, model, allocator, runtime, corpus manifest, downstream recipes, seeds, and selection rule before final confirmation. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1154 | A stage advances on held-out evidence and acceptance tests, not on an arbitrary number of training steps. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **15.5 Descriptor dropout and generalization** |  |  |  |
| L1158 | Use feature-block dropout and cost-tier dropout so the encoder can function when expensive or unreliable estimates are unavailable. Drop an entire estimate and its related uncertainty coherently; do not leave a private … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1160 | Include irrelevant/noise columns and redundant groups in augmentation, but apply type-safe transformations. Column permutations and class permutations should not change the semantic task. Regression-target sign/affine … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L1162 | Cache unsupervised streaming summaries only when their fit-row scope matches the episode. Recomputing OOF supervised descriptors is often expensive; optimize that path rather than borrowing whole-table information. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L1167–1177 | Retain the native v2 real-only informative mask and seed, cell, family and task reduction. | Claude | KEPT | Run `public-lite-replay` (`7c1f4598`, `compare.json` `f07f2647`): 125 informative cells, Δ +0.454 [+0.342, +0.553] vs SDV GC & +0.838 [+0.662, +1.042] vs TVAE; replicated in `v3-report-dev-s5-8` (`55974867`, 44 informative headline cells / 11 families). [Baseline replay](#baseline-replay) |
|  | **16.1 Preserve the current native primary statistic** |  |  |  |
| L1169–1173 | The current candidate contract defines: \[ R_x=\frac{L_0-L_x}{L_0-L_A},\qquad v=\operatorname{clip}(R_S,-1,1.25)-R_B. \] | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1175 | Here `A` is the real training reference, `S` is synthetic training data, `B` is the real twin, and the losses share the declared evaluation partition and learner recipe. Informative status is determined from the real … | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1177 | Current reduction: average sample seeds for each cell/auditor, take the minimum over seeded linear and GBDT auditors, average cells within family, families within task, then tasks. The held-out MLP remains a guard. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1179 | Record both the clipped primary and unclipped retention/loss. A model beyond the upper clip can still improve raw utility even when the primary statistic saturates. Elo is a useful secondary ranking, but depends on the … | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1181 | When no tested budget reaches the declared TDL threshold, report TDL as greater than the largest tested budget or unqualified, according to the frozen rule. Do not interpolate an untested size or describe the first … | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1183 | The source [contract](https://github.com/neverhuman/dope/blob/e3be91de02ac3cddc1bd22611bcc47d9195fae1a/contracts/dope-kpi.json) currently specifies: | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1187 | Primary improvement — Paired CI lower bound >0, Holm-adjusted p &lt;0.05, minimum delta 0.01 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1188 | Task protection — Each applicable task CI lower bound >−0.02 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1189 | Held-out MLP — Delta ≥0 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1190 | Realism coverage — Delta >−0.03 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1191 | Privacy coverage — Delta >−0.03 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1192 | Exact-copy excess — ≤0.005 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1193 | Training seeds — At least 3 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1194 | Headline bytes — 2,048; full grid 512/1,024/2,048/4,096 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1195 | Primary synthetic size — 4×; matched-size result at 1× | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1196 | Bootstrap — 10,000 stratified family resamples | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1197 | Null calibration — At least 2,000 family sign-flip replicates; false-promotion rate ≤0.05 | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L1199 | These are existing candidate settings, not new measured qualifications. Freeze them, or a transparently versioned successor, before final selection. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **16.2 Failure-value issue requiring a versioned fix** |  |  |  |
| L1203 | The current failed-entry value is `−1`. A valid entry at the synthetic retention floor is `−1−R_B`, which can be lower than `−1` when `R_B>0`. Thus a failure can receive a better entry value than a very poor valid model. | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
| L1205 | For a successor contract, use a failure value no better than the valid lower bound, such as `−1−R_B` when the real reference exists. If the reference itself is invalid, use the existing separately defined … | Claude | TESTING | [Failure-floor analysis](#failure-floor-analysis) |
| L1207 | Retain historical v2 scores exactly. Add a fixture proving that intentionally failing a bad prediction cannot improve the successor score. | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L1207 | Keep historical v2 values reproducible alongside failure-floor sensitivity. The proposed successor remains open. | Codex | KEPT | Run `public-lite-replay` (`7c1f4598`, `compare.json` `f07f2647`): SDV GC (+0.454 [+0.342, +0.553]) and TVAE (+0.838 [+0.662, +1.042]) directions both hold under unit failure-floor and unclipped sensitivity diagnostics. [Failure-floor analysis](#failure-floor-analysis) |
|  | **16.2 Failure-value issue requiring a versioned fix** |  |  |  |
| L1209 | Common-cohort rules must not allow a candidate to improve its headline by declaring difficult tables unsupported. Report coverage on the predeclared full task/schema universe, and distinguish permitted not-applicable … | Claude | NOT STARTED | [Failure-floor analysis](#failure-floor-analysis) |
|  | **16.3 Dashboard requirements** |  |  |  |
| L1215 | Regression transfer — MSE/RMSE, MAE, R², retained gain, candidate-vs-analytic delta · Real-trained learner, real twin, null | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1216 | Classification transfer — Log loss, Brier, accuracy, macro-F1, balanced accuracy, AUROC/AUPRC where defined · Same rows/classes; prevalence and undefined states | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1217 | Calibration — Reliability curve, calibration intercept/slope, chosen ECE definition · Real-vs-real sampling variability | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1218 | Minority/regime utility — Per-class and predeclared subgroup losses, worst supported-group loss · Group counts and uncertainty | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1219 | Driver preservation — Grouped permutation importance, stable rank/overlap, sign, ALE response, interactions · Independent real-trained references | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1220 | Marginal fidelity — Quantile/Wasserstein distances, atom/nominal mass, missingness, support · Real twin variability | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1221 | Joint fidelity — Multivariate C2ST, selected pair/class grids, MMD/energy/sliced distances · Held-out discriminators and real-vs-real null | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1222 | Structural validity — Type/group validity, constraints, impossible combinations, class/schema coverage · Declared schema | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1223 | Compression — Complete artifact bytes; shared dependency bytes; raw/normalized input size ratio; TDL at declared utility/guard thresholds · Exact file inventory and byte grid | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1224 | Resources — Encode core-seconds, peak RSS, generation throughput, p50/p95 latency · Named hardware, threads, cold/warm cache | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1225 | Reliability — Failed, timed out, unsupported, skipped and unmeasured cells · Full intended denominator | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1226 | Empirical disclosure — Copy excess, nearest-neighbor proximity, membership/attribute inference diagnostics · Real-vs-real controls and declared threat model | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1228 | Classification accuracy can hide poor rare-class or probability performance. AUPRC depends on prevalence; compare it only under the same evaluation distribution. Calibration bins and uncertainty handling must be pinned. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1230 | The existing shallow marginal AUC diagnostic based on the raw column as a score can miss symmetric variance/shape differences. Replace it with proper univariate two-sample distances or a held-out classifier able to model … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **16.4 Driver score details** |  |  |  |
| L1234 | Report several complementary quantities rather than a single Spearman coefficient: | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1236 | Signed effect agreement for stable linear directions. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1237 | Grouped importance rank and magnitude. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1238 | Top-k overlap at declared `k` values, with chance/reference overlap. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1239 | ALE curve distance on supported bins, weighted by real probability mass. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1240 | Pair interaction preservation on predeclared/generated selected pairs. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1241 | Stability of the synthetic-trained explanation under training seed/resampling. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1242 | Real-vs-real ceiling for each quantity. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1244 | Avoid dividing by a near-zero real-real correlation to create a “normalized driver score.” Report the candidate's distance and the reference distribution separately. Individual-feature rank is undefined or unstable for … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L1246 | Negative permutation importance is possible because of sampling noise or model behavior. Do not discard it to make agreement look better. Distinguish unconditional shuffling from conditional/group shuffling; they answer … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **16.5 Distribution and disclosure evidence** |  |  |  |
| L1250 | Use a genuinely held-out C2ST split and multiple discriminator families. An in-sample discriminator score measures fitting capacity, not a two-sample generalization result. Report its uncertainty and the real-vs-real … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1252 | Rich owner-side descriptors and a small packet are not formal privacy mechanisms. A compressed artifact can still retain sensitive information. Preserve the repository's empirical disclosure checks, and separately assess … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L1254 | Preserve rare but useful structure only at supported resolution. Avoid both memorizing rare rows and destroying every minority signal through indiscriminate smoothing. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **17.1 Sequence the ablations** |  |  |  |
| L1262 | A0 — Existing native analytic/step-zero, embedded V7.1, current trainable baseline · What are the reproducible starting points? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1263 | A1 — Current preprocessing vs unified native normalization/codec; no new feature families · How much gain/loss is contract alignment? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1264 | A2 — Current inputs vs richer univariate/support inputs; same decoder and parameter control · Do easy extra summaries help? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1265 | A3 — Add OOF residual mean/variance/class vectors · Does conditional shape improve transfer? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1266 | A4 — Add signed pair information and exploration · Does interaction information resolve identifiable failures? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1267 | A5 — Current decoder vs minimal pair/conditional-noise decoder, crossed with old/new inputs · Is the bottleneck input information or output capacity? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1268 | A6 — Add nonlinear dependence/geometry after A3–A5 · Does target-free joint structure add incremental value? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1269 | A7 — Scalar landmarks vs structured learner-response tokens · Is the probe output being used effectively? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1270 | A8 — Add driver constraint only · Does explanation preservation improve without utility loss? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1271 | A9 — Add basis/gradient matching only · Does learning-geometry alignment improve unseen learners? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1272 | A10 — Fixed layout vs utility-per-byte allocation · Does byte allocation improve the frontier? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1273 | A11 — Fast vs standard vs deep extraction · What is the benefit per owner CPU-second? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1274 | A12 — Current corpus curriculum vs bounded family-balanced replay · Does training efficiency improve? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1275 | A13 — One learner family held out at a time · Does transfer generalize across inductive biases? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1276 | A14 — 1×, 4×, 16× synthetic rows with fixed packet · Does additional sampling help or just change regularization? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1277 | A15 — Exact hard decoder vs soft research proxy · How much apparent improvement is a deployment mismatch? | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1279 | A5 should be a 2×2 design: | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1283 | Current inputs — Baseline · Capacity alone | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1284 | Rich inputs — Information alone · Information + capacity | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1286 | Also include a large **private oracle/ceiling** model using fit-only rich information without the 2 KiB bottleneck. It estimates whether sufficient signal exists in the extracted information. It is not an eligible … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1288 | A richer target predictor fitted directly to `A` can serve as a diagnostic ceiling. Its performance does not prove a generative model can achieve the same result, and its parameters cannot be smuggled into the recipient … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **17.2 Fair comparisons** |  |  |  |
| L1292 | Match: | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1294 | Physical table families, support/query/test rows, class set, and semantic schema. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1295 | Normalization recipe, downstream algorithms, tuning budget, and evaluator versions. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1296 | Complete packet bytes and disclosed shared-decoder size. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1297 | Training seeds and paired generation/probe seeds. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1298 | Total optimization steps and, separately, total wall/core/GPU compute. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1299 | Synthetic sample count and downstream regularization convention. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1300 | Eligible cells, failure policy, and fidelity/disclosure guards. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1302 | For a feature-only comparison, parameter-match the projection/control network or include a dummy/noise-feature control. More input dimensions may also add more trainable parameters. A shuffled-target descriptor control … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1304 | Report per-family paired deltas and distributions, not only an average or the best seed. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **17.3 Statistical power** |  |  |  |
| L1308–1312 | A rough paired normal approximation for 80% power and a two-sided 5% test is: \[ N_{\rm independent}\approx \frac{(1.96+0.84)^2\sigma_\Delta^2}{\delta^2}. \] | Claude | TESTING | [Power and promotion](#power-and-promotion) |
| L1314 | If paired family-level SD is 0.10, detecting a 0.02 improvement requires about 196 independent families; a 0.01 improvement requires about 784. These are illustrative calculations, not estimates of the repository's … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1316 | Measure `σ_Δ` on development families and simulate the actual reducer/bootstrap decision before confirmation. Thousands of row crops or training seeds from the same family do not provide thousands of independent dataset … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1318 | The current 12 informative DEV headline cells are useful for debugging, not a reliable basis for a broad breakthrough claim. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **17.4 Stop rules and interpretation** |  |  |  |
| L1322 | If rich inputs do not improve a high-capacity ceiling, revisit estimator quality and relevance before growing the codec. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1323 | If the ceiling improves but 2 KiB does not, investigate rate allocation and representational capacity. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1324 | If the differentiable monitor improves but native transfer does not, inspect hard sampling, codec parity, and learner mismatch. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1325 | If mean utility improves but minority/driver/fidelity gates fail, report the tradeoff; do not average it away. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1326 | If a single seed wins and the seed group does not, treat the result as unconfirmed. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1327 | If added features cost substantially more CPU time without improving the equal-cost frontier, keep them in an optional deep tier. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1329 | A negative ablation is useful: it removes unnecessary computation and narrows the research problem. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **18.1 Language and compatibility boundary** |  |  |  |
| L1335 | The root [AGENTS.md](https://github.com/neverhuman/dope/blob/e3be91de02ac3cddc1bd22611bcc47d9195fae1a/AGENTS.md) and [migration plan](https://github.com/neverhuman/dope/blob/e3be91de02ac3cddc1bd22611bcc47d9195fae1a/docs/ … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1337 | This specification does not authorize deleting old implementations. Preserve historical readers, fixtures, models, receipts, and benchmark meanings until native replacement coverage exists. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **18.2 Work packages** |  |  |  |
| L1341 | Paths labeled **new** are proposed additions, not present APIs. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1345 | WP0: normalization authority — `engine/dope-norm/src/nc2.rs`; `engine/dope-core/src/single_encoding/scale_card.rs` · One explicit normalized-coordinate path; fit-only state and typed masks · Frozen-normalizer and train/deploy parity fixtures | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1346 | WP1: RichStats schema — `engine/dope-core/src/prepare/descriptors*.rs`; **new** `prepare/rich_stats/` · Versioned arrays, statuses, support, feature normalization, cache identity · Schema validation and deterministic replay | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1347 | WP2: OOF responses — `engine/dope-core/src/landmark/panel.rs`, `folds.rs`, `learners.rs` · Shared OOF prediction buffer; residual/class/learning-curve outputs · No row leakage; supported-class fixtures | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1348 | WP3: interactions — `engine/dope-core/src/prepare/inter_landmark.rs`; **new** `rich_stats/interactions.rs` · Signed pair discovery, exploratory quota, pair/class tokens · XOR/XNOR and rare-interaction tests | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1349 | WP4: Rust feature feed — `engine/dope-cli/src/landmarks.rs`; `engine/dope-probes/src/landmarks.rs` · Bounded cache/export feed for training · No nontraining Python extractor | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1350 | WP5: training model/loss — `research/levels/v7_single_encoding.py`; `s3_v7_batch_ops.py`; `s3_v7_fast_trainer.py` · Versioned successor training modules with typed tokens and portfolio losses · Training-boundary tests; stable gradient/forward … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1351 | WP6: normalized successor codec — `engine/dope-core/src/single_encoding/packet.rs`; **new** successor module · Rich encoder output → bounded residual code; complete byte count · Golden serialization, rejection and legacy-read tests | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1352 | WP7: native decoder — `engine/dope-core/src/single_encoding/mod.rs`; `engine/dope-core/src/auto/v7.rs` · Source-free typed sampler and pinned weight export · Tensor/native parity plus full pipeline replay | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1353 | WP8: KPI/driver evaluation — `engine/dope-arena/src/kpi/`; `contracts/dope-kpi.json` · Additional metrics, successor failure semantics, complete coverage · Reducer/null/failure/driver controls | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1354 | WP9: evidence and claims — `models/single-enc-v7/`; `bench/real-world/results/` · New successor manifest and frozen comparison receipts · Exact artifact/code/corpus identity and reproducible scoreboard | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L1356 | Resolve the actual local ownership rules before editing. Update `agent/owner-map.json` and `agent/test-map.json` when adding source areas, as required by the repository. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **18.3 Native interface sketch** |  |  |  |
| L1360 | The following interfaces express responsibilities; names and crate placement are proposed. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1362–1398 | fit, schema, folds, recipe, budget, seed, identity, global, columns, edges, classes, landmarks, cost, coverage, extract_rich_stats, encode_normalized, stats, model, budget_bytes, sample_packet, packet, decoder, rows | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1400 | The encoder request deliberately has no query-label field. The sampler deliberately has no source-table or descriptor-cache field. Enforce those boundaries structurally, then test process-level source-free use. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **18.4 Experiment configuration sketch** |  |  |  |
| L1404 | This is a **proposed configuration**, not a command that already runs. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1407 | schema: dope.richstats-experiment/v1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1408 | normalization: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1409 | authority: rust-fit-only | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1410 | product_coordinates: normalized | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1411 | recipe_digest_required: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1412 | retain_raw_inverse_in_packet: false | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1414 | features: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1415 | recipe: richstats-core/v1 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1416 | cost_tier: standard | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1417 | global_values: 96 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1418 | column_values: 128 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1419 | edge_values: 32 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1420 | class_column_values: 24 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1421 | cv_folds: 3 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1422 | pair_edges_per_group: 4 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1423 | pair_exploration_fraction: 0.25 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1424 | joint_row_sketch_max: 2048 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1425 | quadratic_kernel_rows_max: 512 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1426 | status_and_support_required: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1428 | model: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1429 | private_encoder_width: 192 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1430 | private_encoder_layers: 3 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1431 | attention_heads: 4 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1432 | preserve_group_and_class_equivariance: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1433 | decoder_components: [analytic_base, signed_pairs, conditional_scale] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1434 | legacy_reader_compatibility: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1436 | training: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1437 | family_balanced: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1438 | typed_augmentation: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1439 | learner_families: [linear, additive_rff, soft_tree] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1440 | held_out_learner_family: mlp | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1441 | exact_hard_forward: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1442 | quantization_aware: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1443 | synchronized_nonfinite_policy: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1445 | evaluation: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1446 | complete_byte_grid: [512, 1024, 2048, 4096] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1447 | headline_bytes: 2048 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1448 | synthetic_multipliers: [1, 4, 16] | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1449 | minimum_training_seeds: 3 | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1450 | source_family_split_required: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1451 | native_decoded_artifacts_only: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1452 | unresolved_or_unmeasured_guards_block_promotion: true | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1455 | Pin the exact downstream recipes, objective weights, loss-normalization constants, seeds, and code/weight digests in the final manifest. These values cannot remain implicit defaults. | Claude | NOT STARTED | [Training operations](#training-operations) |
|  | **18.5 Run record** |  |  |  |
| L1459 | Each training/evaluation receipt should contain: | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1461 | Code commit; dirty-tree status and source digest if applicable. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1462 | Feature schema/normalizer/codec versions and hashes. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1463 | Dataset/family manifest and exact split identity. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1464 | Model architecture, trainable/frozen parameter counts, encoder/decoder bytes. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1465 | All objective weights, reduction conventions, optimizer and curriculum settings. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1466 | Completed episodes, independent sources/families, actual rows, failures and skipped measurements. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1467 | Training seeds, downstream seeds, sampler seeds, and resume lineage. | Claude | TESTING | [Training operations](#training-operations) |
| L1468 | Full generated artifact inventory and actual lengths. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1469 | Runtime hardware, threads, peak memory, wall time and core/GPU time. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1470 | Native metric rows, reducer version, applicability/coverage, guard status and confidence intervals. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L1472 | A training log row or a nominal episode counter is not a substitute for independent dataset coverage. | Claude | NOT STARTED | [Training operations](#training-operations) |
|  | **19. Acceptance tests** |  |  |  |
| L1477 | These tests are required for the proposed implementation. They were **not** run as part of this source-review/specification task. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1481 | T01: fit-only normalization — Change query/test values and labels while holding fit rows fixed · Normalizer, descriptors, packet and encoder output remain unchanged | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1482 | T02: query-label boundary — Replace query labels with unrelated values · Only training loss/evaluation changes; encoder input does not | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1483 | T03: exact codec feed — Extract, serialize, parse and reconstruct inputs · Training input matches native reconstructed values within declared precision | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1484 | T04: source-free generation — Load packet/decoder in a process without source/caches · Valid rows generated with no hidden dependency | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1485 | T05: row invariance — Permute IID fit rows · Estimates agree within floating-point/sketch tolerance; deterministic recipes reproduce | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1486 | T06: column equivariance — Permute semantic groups and all corresponding metadata · Generated/evaluated law maps back to the same result | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1487 | T07: class equivariance — Relabel binary/multiclass classes · Probabilities, predictions and score map back correctly | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1488 | T08: category equivariance — Permute opaque nominal codes · No ordering-induced change in task semantics | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1489 | T09: XOR collision/resolution — Balanced XOR and XNOR grids · Baseline summary collision demonstrated; signed interactions differ and the extended decoder can represent both | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1490 | T10: hidden interaction — Relevant columns have zero marginal target association · Exploration discovers signal at the declared bounded-search rate | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1491 | T11: conditional variance — Same mean, different feature-dependent variance · Variance descriptors and decoder distinguish the regimes | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1492 | T12: multiclass direction — Same overall difficulty, different confused-class pairs · Class-conditioned inputs retain the difference | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1493 | T13: missingness — Same imputed values, different mask-target relationship · Inputs and generated masks preserve the distinction | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1494 | T14: discrete validity — Binary, ordinal, one-hot, nominal, atoms · No impossible group states or out-of-vocabulary values | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1495 | T15: support edge cases — Constant, all-missing, one-row, single-class, tied columns · Explicit valid/insufficient/unsupported result; no fabricated signal | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1496 | T16: duplicate split — Few duplicate/entity groups · No row-random leakage fallback | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1497 | T17: PSD and solve — Singular covariance, collinear basis, extreme imbalance · Stable declared repair or explicit failure; finite checked gradients | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1498 | T18: quantization — Boundary values and every supported bit/layout setting · Correct rounding, ordered knots, valid scales/probabilities, exact bytes | Codex | TESTING | [Hard forward](#hard-forward) |
| L1499 | T19: width/classes — Test every public entry point at/above limits · Full coverage or explicit refusal; no class-zero fallback | Codex | TESTING | [Hard forward](#hard-forward) |
| L1500 | T20: packet accounting — Include all required schema/envelopes/dependencies · Measured full bytes and separately disclosed shared dependency | Codex | TESTING | [Hard forward](#hard-forward) |
| L1501 | T21: hard/soft gap — Evaluate identical checkpoints through both paths · Gap reported; selection uses native hard rows | Codex | TESTING | [Hard forward](#hard-forward) |
| L1502 | T22: finite differences — Tiny smooth decoder/learner fixture · Autograd agrees where derivatives exist; STE components identified separately | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1503 | T23: worker failure — Inject one-worker nonfinite/partial batch · Synchronized behavior, no hang, accurate counts | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1504 | T24: resume — Resume model/EMA/optimizer/queue/RNG state · Same continuation under declared deterministic runtime, or documented bounded nondeterminism | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1505 | T25: no-signal control — Independent target · No systematic manufactured signal; proper shrinkage/coverage | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1506 | T26: correlated drivers — Swappable substitute features · Group-level preservation succeeds; individual ambiguity reported | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1507 | T27: metric control — Perfect twin, null, deliberately poor model, failed run · Reducer ordering and successor failure floor correct | Codex | TESTING | [Hard forward](#hard-forward) |
| L1508 | T28: family leakage — Renamed/derived copies across proposed partitions · All variants assigned to one family or explicitly excluded | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1509 | T29: cache identity — Change fit rows, target, recipe, fold or pair seed · Cache invalidates correctly | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1510 | T30: sampling invariance — Different request batch sizes for same seed/row stream · Declared sampler replay behavior holds; no cross-row batch-normalization dependence | Codex | TESTING | [Hard forward](#hard-forward) |
| L1511 | T31: adversarial C2ST — Equal means, different variances or nonlinear dependence · Appropriate diagnostic detects the difference at validated power | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1512 | T32: historical compatibility — Existing DPK3.2/v1/DSE7 fixtures · Old readers/results retain their versioned behavior | Codex | TESTING | [Hard forward](#hard-forward) |
| L1514 | T30 is especially important when reusing older research samplers: batchwise centering/rescaling changes the distribution of a row depending on how many other rows were requested. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **19.1 Focused repository commands** |  |  |  |
| L1518 | Existing relevant suites include: | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1520–1525 | cargo test --manifest-path engine/Cargo.toml --locked -p dope-norm --test nc2 cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test single_encoding_v7 cargo test --manifest-path engine/Cargo.toml … | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1527 | Add successor tests under the appropriate Rust crate; run nonempty focused filters during implementation. Follow the repository's `just fast`, `just check` and standalone engine checks before integrating code. The … | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L1529 | Passing deterministic fixtures proves specific contracts, not improved generalization. Only the frozen benchmark establishes lift. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **20. Delivery milestones and promotion gates** |  |  |  |
| L1535 | M0: reproducible baseline — Exact-head native baseline, current checkpoint identity, corpus/family inventory · Reproducible decoded-artifact score; measured failure/coverage/bytes | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1536 | M1: coordinate and leakage repair — Unified normalized path, typed augmentation, class handling, duplicate-group split · T01–T08 and relevant boundary tests pass | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1537 | M2: core rich features — Rust RichStats, OOF responses, class tokens, signed pair explorer · T09–T17; measured extraction cost; feature-schema frozen | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1538 | M3: rich-input compatibility control — Same output family with richer private encoder · Paired feature-only ablation completed; no unsupported claim | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1539 | M4: minimal successor decoder — Pair corrections, conditional scale, normalized wire format · Source-free hard decoding, byte/parity tests pass | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1540 | M5: portfolio objective — Linear/additive/soft-tree transfer; then separate driver/sketch arms · Native improvement survives hard sampling and held-out learner guard | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1541 | M6: rate and compute frontier — Complete byte grid and fast/standard/deep measurements · Best eligible configuration selected on development only | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1542 | M7: confirmation — Frozen multi-seed comparison and full receipts · Existing or explicitly versioned successor promotion rule passes | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1544 | Do not schedule a large model search before M0–M2. The current sources expose issues that can invalidate attribution even if training loss looks excellent. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1546 | Before M7, write the exact claim in advance: supported schema/task envelope, byte limit, synthetic row multiplier, learner set, baseline, improvement threshold, and guard conditions. A broad “best synthetic generator” … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **21.1 Learn which statistics matter, with explicit controls** |  |  |  |
| L1552 | Train a reliability-aware feature gate and measure leave-one-family-out descriptor ablations. A learned attention weight is not evidence of causal feature usefulness; the actual held-out ablation is. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1554 | Measure marginal improvement per extraction second and per packet byte separately. Some statistics are expensive to extract but cheap to compress; others are cheap to calculate but require many packet bits to retain. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L1556 | A promising experimental objective is to maximize downstream information per byte: predict which descriptor changes would alter the preferred learner response, then preferentially encode those changes. Evaluate this … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **21.2 Contrastive dataset pairs with known missing distinctions** |  |  |  |
| L1560 | Construct fit-only training pairs that preserve selected low-order statistics but change an important relationship: | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L1562 | XOR versus XNOR. | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L1563 | Same conditional mean with different conditional variance. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L1564 | Same marginals/correlation with different mixtures or tail dependence. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L1565 | Same class priors with swapped class-conditional regions. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L1566 | Same imputed matrix with different informative missingness. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L1568 | Require the encoder/packet to distinguish the pairs in the coordinates needed for decoding, and require the generated data to exhibit the correct difference. Merely separating latent embeddings is insufficient. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L1570 | This turns “more statistical features” into a testable information objective. It is a proposed research direction, not an established result on DOPE. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **21.3 Compare two equivalent factorizations** |  |  |  |
| L1574–1577 | For classification, compare: \[ P(X)P(Y\mid X)\qquad\text{and}\qquad P(Y)P(X\mid Y). \] | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L1579 | The latter can make rare class-conditional structure easier to parameterize; the former may better share feature geometry and scale with many classes. Both describe a joint law in principle, but finite packet/decoder … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L1581 | Measure class-specific dependence and fidelity, not only accuracy. Do not claim the factorization alone solves class imbalance. A hybrid must remain a valid coherent joint sampler with one clearly defined forward law. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **21.4 Conditional noise as a controlled regularizer** |  |  |  |
| L1585 | A feature-dependent noise model may help preserve calibration and avoid synthetic labels becoming too easy. Test three arms: empirical conditional noise, calibrated shrinkage toward a global noise model, and deliberately … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L1587 | Evaluate raw downstream utility, calibration, residual distribution, and driver behavior. If denoising helps only the utility mode, label that distinction rather than forcing one objective to satisfy incompatible uses. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **21.5 Shared basis bank and local adaptation** |  |  |  |
| L1591 | Random Fourier features approximate kernel structure [R9]. Expand the existing fixed RFF learner into a small shared bank of bandwidths, spline resolutions, and interaction maps. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L1593 | A compact per-table selector can choose useful bases without sending large per-table weights. Its selectors and coefficients still count in the packet. Avoid tuning the bank to confirmation families. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L1595 | Compare fixed features, learned shared bases, and a basis bank selected by landmark responses. Retain an unseen learner/bandwidth test to detect over-specialization. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **21.6 Strong generator baselines as ceilings** |  |  |  |
| L1599 | Use TabDDPM [R15], TabSyn [R16], and TabDiff [R17] as research comparators where they can be run under an admitted implementation boundary. They offer concrete alternatives for mixed-type and joint distribution modeling. | Codex | TESTING | [Published generators](#published-generators) |
| L1601 | They should have two separate comparison roles: | Codex | NOT STARTED | [Published generators](#published-generators) |
| L1603 | **Unconstrained or larger-budget quality ceiling:** clarifies what richer generative capacity can preserve. | Codex | NOT STARTED | [Published generators](#published-generators) |
| L1604 | **Matched-budget deployable competitor:** only if all table-specific weights/state and required metadata actually fit the declared budget. | Codex | NOT STARTED | [Published generators](#published-generators) |
| L1606 | Their published results do not establish superiority under DOPE's normalization, byte budget, source-family split, or downstream audit. Do not replace the small direct sampler with iterative diffusion merely because … | Codex | NOT STARTED | [Published generators](#published-generators) |
|  | **21.7 Hypothesis register** |  |  |  |
| L1612 | Signed interactions outperform additional scalar landmark scores — They resolve an explicit information collision · Pair search misses rare/wide interactions · A4/A5 and held-out interaction families | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1613 | OOF conditional noise improves calibration and utility — Current global scale omits heteroscedasticity · Estimation noise exceeds useful structure · A3 plus noise fixtures and real families | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1614 | Rich private inputs help within the current packet — Better estimation can correct noisy analytic summaries · Output family is the limiting factor · A3/A5 compatibility arm | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1615 | Typed/class-aware tokens improve coverage — Current limits and scalar summaries lose task identity · Schema bytes dominate · Full-schema mixed/class benchmark | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1616 | Gradient/Gram matching transfers beyond its training probes — Matches learning dynamics more directly · Surrogate overfitting and extra autograd cost · A9/A13 with unseen families | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1617 | Utility-aware byte allocation beats fixed priorities — Packet components have unequal value · Noisy allocation targets overfit · A10 at all four measured byte budgets | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1618 | Group-level driver constraints preserve useful explanations — Real effects can be stable at group level · Imposes unstable reference behavior · A8 with real-vs-real stability | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1619 | Adaptive extraction improves quality per CPU-second — Hard tables may need different statistics · Missingness/tier pattern becomes a shortcut · A11 at equal total owner cost | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L1621 | The best first investment is still the combination of **exact normalized coordinates, signed interaction information, cross-fitted conditional summaries, and a decoder that can represent them**. Larger feature catalogs … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |

</details>

### DFP1

<details>
<summary>Source proposals and DEV test routing</summary>

| Source | Proposal | Owner | State | DEV test |
| --- | --- | --- | --- | --- |
|  | **0. Summary** |  |  |  |
| L11 | **What is specified.** DFP1 is a versioned set of dataset descriptors computed on the normalized FIT table: 188 features per column, 19 per column pair, 24 per class and 503 per table, in four compute tiers. 256 of the … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **Conflicting release layout** |  |  |  |
| L11 | `188/19/24/503` as the release ABI (superseded by frozen `contracts/evidence-layout-v1.json` 128/16/16/4/8/160; named recipes kept as separate candidates; not run). | Antigravity | NOT STARTED | Superseded draft ABI — never run. [Evidence bundles](#evidence-bundles) |
|  | **0. Summary** |  |  |  |
| L13 | **The finding that decides where the features go.** More features through today's input path will not move the KPI. The repo's own evidence says so twice: | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L15 | MLC-v3 with every landmark zeroed keeps 0.985 of the retention of full conditioning (95% CI 0.941 to 1.030) [read: `docs/optional-landmarks-qualification.md:63`]. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L16 | No trained V7 group beats the untrained step-0 packet decode on DEV by more than noise; the trained target heads cost retention and realism does not move [read: `models/single-enc-v7/V7_RELEASE_CHECKPOINT.md`]. | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L18 | The reason is structural. The V7 network runs on the receiver and sees only the ≤ 2 KiB packet, which is the same information the analytic decoder already uses [read: `engine/dope-core/src/single_encoding/mod.rs:273`]. … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L20 | **Two defects to fix before any feature ablation is trusted** (both in the V7 training loop): | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L22 | *Train/serve skew on discrete columns.* Training adds Gaussian jitter to every FIT cell, which makes the packet's `is_disc` flag false for binary, ordinal and count columns in training, while evaluation (no jitter) sets … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L23 | *The realism losses use about 26 real rows.* The marginal, correlation and sliced-Wasserstein terms truncate every episode to the smallest query set in the micro-batch. They carry 3.1 of the loss weight against 1.0 for … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L25 | **Unused budget.** For tables narrower than about 40 columns the DSE7 packet leaves hundreds to more than 1,800 of its 2,048 bytes unused, and it has no section that can express an interaction [calc: Appendix C.3]. … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L27 | **Order of work** (section 9): FP-0 fix the loop and re-measure V7 against step 0 → FP-1 DFP1 tiers 0 and 1 in Rust → FP-2 optional packet sections with statistical gates (no network) → FP-3 twin-referenced objective … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **2. Findings that shape the design** |  |  |  |
| L73 | **F1. The receiver-side network has no new information.** `refine_packet` takes `col_feats`, `q_nl`, `glob_feats` and nothing else [read: `single_encoding/mod.rs:273`]. Whatever it outputs is a function of the packet, so … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L75 | **F2. In MLC-v3 the features have low leverage by construction.** The 256-D vector drives 14 gates and two bounded latents (`gate_head` 10 + `v3_gate_head` 4; the covariate refinement is mixed in at 6%) [read: … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L77 | **F3. Train/serve skew on `is_disc`.** `Xa = clip(Xa + N(0, 0.015))` is applied to every FIT cell after scaling [read: `s3_v7_batch_ops.py:250`]; `is_disc` is `unique(round(x, 4)) ≤ 16` on the first 256 rows [read: … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L79 | **F4. The dominant loss terms are the noisiest.** `n_q_use = max(16, min over the batch of the query count)` [read: `s3_v7_batch_ops.py:414`] with query counts log-uniform on [24, 128] gives a median of 26 rows at … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L81 | **F5. Training sizes sit below the headline stratum.** `n_fit` is log-uniform on [32, cap] with the cap ramping 256 → 512 [read: `s3_v7_batch_ops.py:160`, `s3_v7_fast_trainer.py:306,320`]. At the final cap 34% of … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L83 | **F6. The packet cannot express an interaction and leaves bytes unused.** DSE7 has linear weights and four main-effect bins per column; the only interaction capacity is the hypernetwork MLP, whose weights are inferred … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L85 | **F7. Driver fidelity is not in the V7 objective.** The loss has no importance term; the dev monitor reports a Spearman of absolute ridge coefficients only [read: `research/levels/s3_v7_eval.py:59`]. V6 and MLC-v3 had … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L87 | **F8. Effect sizes arrive without uncertainty.** Existing features are magnitudes (correlations, gains, ξ). A correlation of 0.2 is noise at n = 40 and strong at n = 4,000, and the network is given no standard error to … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L89 | **F9. Column selection for wide tables uses Pearson correlation with the class index.** For P > 48 the scale card keeps the top columns by `abs(corr(x_j, y))` for every task, including multiclass where `y` is an … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L91 | **F10. The training arena carries no column kinds or groups.** `dope.flat-corpus/v1` is a float matrix with no schema [read: `docs/plans/v2-gap-closure/formats.md:6`], while the product path has them (`UnitSchema.groups` … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **3. Rules every feature obeys** |  |  |  |
| L97 | **FIT rows only.** Every statistic, fold, threshold, bandwidth and subsample comes from the FIT role. Each block ships a query-poison test: replacing every non-FIT row with garbage leaves the block bit-identical. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L98 | **After normalization, and scale-free.** Features are computed on the unit-space table `U` produced by the normalize-once step (the product path's scale card, whether NC1, NC2 or the packet's own `ScaleCardV7`; the … | Claude | TESTING | [Extraction identity](#extraction-identity) |
| L99 | **Effect and evidence.** Every association or test statistic is emitted as a pair where both are meaningful: the effect size, and a z-score against its null (`Z`). `Z` uses an analytic null where one exists and a fixed … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L100 | **Masked, never synthesized.** Each feature has a mask bit. A feature whose tier was not computed, whose `n_min` is not met, or that is undefined for the task is masked and embedded as a learned "absent" vector, as … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L101 | **Equivariance is a tested property.** Row order: invariant. Column order: column and edge blocks permute, global blocks are invariant. Class relabeling: class tokens permute. Sign flip of a column: each signed feature … | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L102 | **No extrema, support floor 10.** No minimum or maximum enters a feature; end knots are smoothed order statistics at clipped probabilities (DSE1 §7.3). Nothing is computed from a level, class, bin or leaf with fewer than … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L103 | **Functional first, scalar second.** Where a curve or small histogram contains a family of scalar tests (a 33-point quantile function contains skew, kurtosis, tails, modes and atoms), the curve is the primary input and … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L104 | **Versioned layout.** `engine/dope-features/layout-dfp1.json` fixes id, block, index, tier, `n_min`, calibration and equivariance tag. Its SHA-256 is bound into checkpoints and receipts, as the landmark panel identity … | Antigravity | NOT STARTED | [Extraction identity](#extraction-identity) |
| L105 | **Earn the place.** A block stays only if it passes the ablation rule of section 10. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **4.1 Notation** |  |  |  |
| L113 | `n` FIT rows, `P` columns, `K` classes. `u_j` is unit-space column `j`; `z_j` its normal scores, with conditional-mean scores for tied blocks (as `normal_scores_conditional`); `R` the correlation matrix of the `z_j`. `r` … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **4.2 Tiers** |  |  |  |
| L119 | T0 — Per-column marginal blocks; packet-codec residual · one sort per column, O(nP log n) · every episode | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L120 | T1 — Pairwise dependence, table geometry, univariate target tests and profiles, class tokens · O(mP²) plus O(m²P) for distance statistics · every episode | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L121 | T2 — Cross-validated landmark panel (existing 23 plus 10), cross-fitted teacher, drivers, fANOVA, complexity, split-half · tens of learner fits · a fixed share of episodes (default 50%, reduced folds and rounds) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L122 | T3 — Closed loop: sample the candidate packet, compare with the real rows · one decode plus auditor fits · a fixed share (default 25%) | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L124 | Tier dropout in training (independent masks per tier) gives one model that serves `none &#124; fast &#124; progressive &#124; full`, matching the existing landmark modes. Wall-clock per tier is [est] until FP-1 measures it on xbabe0; … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **4.3 Column token (188 per column)** |  |  |  |
| L128 | **C.TYPE (10), T0.** Type and support. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L132 | 1–4 — `is_const`, `is_binary`, `is_disc16`, `is_lattice` · DSE1 §7.2 rules: one level; two levels; ≤ 16 levels after merging levels under 10 rows; values on a regular grid … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L133 | 5 — `lattice_step` · grid step in unit space, squashed | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L134 | 6 — `distinct_ratio` · distinct values / n | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L135 | 7–8 — `top1_mass`, `top2_mass` · mass of the two most frequent values | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L136 | 9 — `floor_loss` · share of rows in levels with fewer than 10 rows (information the support floor removes) | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L137 | 10 — `level_entropy` · entropy of level masses / ln(levels); masked for continuous columns | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L139 | **C.BOUND (7), T0.** Legality: what a depth-3 tree in a C2ST finds first. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L143 | 1–2 — `lo_atom`, `hi_atom` · mass of the lowest / highest level when it holds ≥ 10 rows (floor, cap, censoring), else 0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L144 | 3–4 — `inner_atom`, `inner_atom_loc` · largest atom strictly inside an otherwise continuous column (zero inflation), and its normal-score position / 3 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L145 | 5 — `heaping` · excess mass on coarse multiples (5× and 10× the lattice step) over the mean of neighbouring grid points | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L146 | 6–7 — `end_ratio_lo`, `end_ratio_hi` · ratio of the last two knot spacings at each end (a narrowing end marks bounded support; DSE1 §8.1) | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L148 | **C.QF (33), T0, functional.** The smoothed FIT quantile function at 33 normal-score knots `z_i = z_hi·(i − 16)/16`, emitted as `Q_j(z_i) − z_i` (zero for a Gaussian column in unit space). Window smoothing and `z_hi` … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L150 | **C.SHAPE (12), T0.** | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L154 | 1–2 — L-skewness `τ3`, L-kurtosis `τ4 − 0.1226` · L-moment ratios; robust to outliers | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L155 | 3–4 — Bowley skewness, Moors kurtosis − 1.233 · quantile-based | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L156 | 5 — medcouple · robust skewness, on a 256-point quantile subsample | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L157 | 6 — negentropy · Gaussian entropy minus the Vasicek spacing estimate; continuous columns only | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L158 | 7 — Anderson–Darling `A*²` vs normal · squashed | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L159 | 8 — bimodality coefficient − 5/9 · `(g1² + 1)/(g2 + 3(n−1)²/((n−2)(n−3)))` | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L160 | 9 — Hartigan dip `Z` · dip·√n against a shipped null table | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L161 | 10–11 — `gap_ratio`, `gap_loc` · largest spacing in the central 90%, times n / ln n; and its probability position | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L162 | 12 — `kde_modes` · modes of a Silverman-bandwidth KDE on 64 quantile points, capped at 4, / 4 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L164 | **C.TAIL (8), T0.** | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L168 | 1–2 — Hill tail index, lower / upper · on exceedances beyond the `k = clamp(√n, 10, n/10)`-th order statistic, in the pre-compression coordinate when the scaler exposes it | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L169 | 3–4 — tail weight, lower / upper · `(q50 − q01)/(q50 − q25)` and its mirror, squashed | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L170 | 5–6 — outlier share, lower / upper · cells beyond 3.5 robust scales | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L171 | 7–8 — `frac_compressed`, `frac_clipped` · share beyond 3 units (asinh region) and at ±6; tagged `scaler` | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L173 | **C.MISS (5), T0/T1.** `miss_rate`; `Z` of missingness against the target (t-test or χ²); largest `abs(corr(miss_j, z_k))`; largest co-missingness correlation; `has_miss`. Requires NaN to reach the extractor before … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L175 | **C.CODEC (6), T0, closed loop.** How well the packet's own marginal reproduces this column: maximum absolute gap between the empirical quantile function and the 5-, 9- and 17-knot reconstructions on the 33-point grid (3 … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L177 | **C.UNC (4), T0.** `log2(n_j)/12` (non-missing rows); standard error of the median knot; mean standard error of the two outer knots (`√(u(1−u)/n)` over the local density from knot spacing); `1/√n`. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L179 | **C.DEP (12), T1.** Summary of the column's row of the edge tensor: largest, second largest and mean `abs(ρ)`; `R²_j` from the inverse of `R + εI` (variance inflation); communality at ranks 4, 8, 12; largest residual … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L181 | **C.RULE (4), T1.** Count of implications involving the column / P; smallest order-violation share with any partner (`min(P(a&lt;b), P(a>b))` near 0 marks an ordering constraint); one-hot group size / 16; membership in a … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L183 | **C.Y (27), T1, supervised.** | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L187 | 1–2 — Pearson `r` (signed), `Z` · on normal scores; t-statistic | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L188 | 3 — Spearman `ρ` (signed) ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L189 | 4–5 — Chatterjee `ξ`, `Z` · `Z = ξ·√(5n/2)` for a continuous target; permutation null otherwise | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L190 | 6–7 — distance correlation, `Z` · bias-corrected, t-approximation; on `m ≤ 512` rows | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L191 | 8–9 — binned MI (Miller–Madow, bits), `Z` · G-test with Wilson–Hilferty | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L192 | 10 — `abs(r) − abs(partial r)` · redundancy / suppression | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L193 | 11 — partial t (signed) · ridge coefficient over its standard error | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L194 | 12 — LOCO share · `β_j² / (G⁻¹)_jj` over the model sum of squares | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L195 | 13 — monotonicity ratio · R² of the better isotonic fit over R² of the 16-bin means | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L196 | 14 — slope sign changes / 3 · of the 8-bin profile, counting only changes larger than 2 SE | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L197 | 15–16 — stump location `u*`, stump gain `Z` · best single threshold as a probability position, and its OOF gain | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L198 | 17–19 — effect terciles · share of binned-effect variance in the lower, middle, upper third of the column | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L199 | 20 — quadratic `Z` · `corr(z_j² − 1, r)` | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L200 | 21–22 — heteroscedasticity `Z` (linear, U-shaped) · `corr(z_j, abs(r))`, `corr(z_j², abs(r))` (Breusch–Pagan and White style, per column, on OOF residuals) | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L201 | 23 — ANOVA `η²` · classification | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L202 | 24 — Kruskal–Wallis `Z` · classification | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L203 | 25 — best one-vs-rest AUC, `2·(AUC − 0.5)` · classification | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L204 | 26 — class-conditional KS `Z` · largest two-sample KS between a class and the rest (shape, not just mean) | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L205 | 27 — Brown–Forsythe `Z` · class-conditional spread (the "equal mean, different variance" case) | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L207 | **C.YF (24), T1, functional.** Over the column's 8 bins, each empirical-Bayes shrunk with `count + 8`: the mean of the standardized target or class surprise (8); the mean of the base-model residual `r` (8, the 8-bin … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L209 | **C.SPEC (16), T2, existing.** The native panel's 9-slot per-feature spectrum and the 7 building-block probes, unchanged [read: `research/levels/descriptors.py`, `research/levels/building_block_landmarks.py`]. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L211 | **C.DRV (12), T2.** Real-reference drivers from a cross-fitted teacher (the scoreboard's own `linear-seeded-v1` and `gbdt` recipes at reduced size). These are inputs **and** the targets of the driver loss (section 6.3). | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L215 | 1–2 — linear permutation importance, fold SE · grouped, on OOF rows (`dvs::permutation::grouped_permutation_importance`), normalised to sum 1 | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L216 | 3–4 — GBDT permutation importance, fold SE · same | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L217 | 5 — mean absolute SHAP share · TreeSHAP on the oblivious trees | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L218 | 6 — split-gain share ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L219 | 7 — mean first-split depth / max depth ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L220 | 8–9 — fANOVA main-effect share, interaction share · Walsh–Hadamard decomposition of the oblivious-tree leaf tables (A.9) | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L221 | 10–11 — partial-effect sign, sign stability across folds ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L222 | 12 — GBDT importance rank / P ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L224 | **C.RES (8), T3, closed loop.** Real rows against rows sampled from the candidate packet, for this column: two-sample KS; Wasserstein-1; split-importance share in a cross-fitted boosted-tree C2ST; linear and GBDT driver … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
|  | **4.4 Edge channels (19 per ordered column pair)** |  |  |  |
| L228 | Used as additive attention bias (section 5.2). Symmetric unless marked. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L232 | 1 — T1 · `rho` · normal-score correlation, tie-corrected (DSE1 §7.5) | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L233 | 2 — T1 · `rho_s − rho` · Spearman minus normal-score correlation | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L234 | 3 — T1 · `xi(j→k)` · Chatterjee ξ, asymmetric; the transpose carries `xi(k→j)` | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L235 | 4 — T1 · MI excess `Z` · binned MI minus the MI a Gaussian copula with the same `rho` gives under the same binning (lookup table) | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L236 | 5–6 — T1 · upper / lower tail-dependence excess `Z` · `P(U_j > q, U_k > q)/(1 − q)` at `q = 0.9`, minus the Gaussian-copula value | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L237 | 7 — T1 · copula-cell `χ²` `Z` · 4 × 4 quartile-cell counts against the Gaussian-copula cells, 8 degrees of freedom | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L238 | 8 — T1 · partial correlation · from `(R + εI)⁻¹` | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L239 | 9 — T1 · residual correlation · after the rank-12 factor fit | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L240 | 10 — T1 · functional dependency (asymmetric) · `1 − H(k given j)/H(k)` for discrete pairs; ξ ≥ 0.95 for continuous | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L241 | 11 — T1 · order constraint · `1 − 2·min(P(u_j < u_k), P(u_j > u_k))` on raw-comparable pairs; masked otherwise | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L242 | 12 — T1 · one-hot sibling · 1 if the pair is mutually exclusive binaries from one group (schema, or inferred: never both 1, union coverage ≥ … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L243 | 13 — T1 · co-missingness · correlation of the two missing indicators | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L244 | 14 — T1 · interaction `Z` · two-way interaction of the 4 × 4 binned cell means of `r` after removing both main effects (A.8) | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L245 | 15 — T1 · product `Z` · `corr(z_j·z_k − ρ, r)` | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L246 | 16 — T2 · fANOVA pair energy · Walsh–Hadamard pair energy of the teacher trees (A.9), share of total | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L247 | 17 — T3 · correlation residual · real minus sampled-packet correlation | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L248 | 18 — T3 · copula-cell residual `Z` · 4 × 4 cells, real vs sampled packet | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L249 | 19 — T3 · interaction gap · channel 14 on real rows minus the same on sampled rows | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **4.5 Class tokens (24 per class; one target token for regression)** |  |  |  |
| L253 | Prior and log prior; `min(n_c, 100)/100` and a below-floor flag; centroid norm in normal-score space; centroid projections on the first 8 factor axes; within-class scale ratio `tr(S_c)/tr(S)`; `‖R_c − R‖_F / P` … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L255 | Plus the tensor **YC (K × P × 4)**: per class and column, the shrunken deviation of the class rate from its prior in the column's four quartile bins. It enters as a bias on class-to-column attention. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L257 | For regression the single target token carries G.Y. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **4.6 Global token (503)** |  |  |  |
| L261 | **G.META (12), T0.** `log2 n / 12`, `log2 P / 8`, `log10(P/n) / 4`, task one-hot (3), `K/8`, events per variable (`min class count / P`, squashed), shares of continuous / discrete / constant columns (3), number of … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L263 | **G.Y (28), T1.** Regression (13): `τ3`, `τ4`, Hill lower/upper, lattice flag, largest atom, lower/upper atom, Box–Cox `λ̂/2` (profile likelihood on a 9-point grid), bimodality, dip `Z`, gap ratio, **noise floor** … | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L265 | **G.SPEC (24), T1.** Top-12 eigenvalue shares of `R`; effective rank / P; participation ratio / P; number of eigenvalues above the Marchenko–Pastur edge / P; Ledoit–Wolf shrinkage intensity; residual RMS correlation … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L267 | **G.GEOM (16), T1.** TwoNN intrinsic dimension / P; components for 90% variance / P; Hopkins statistic; coefficient of variation of nearest-neighbour distance; relative contrast; exact-duplicate rate; near-duplicate … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L269 | **G.PRIV (8), T1.** Singleton rate over combinations of the four highest-entropy discrete columns; share of rows in two-way discrete cells under 10 rows; share of rows whose nearest-neighbour distance exceeds 5 times the … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L271 | **G.ALIGN (8), T1.** Where the predictive direction lies relative to the covariance spectrum: share of `Var(x'β)` carried by the top 1, 2, 4, 8, 12 eigenvectors of `R` (5); effective predictive rank / P; share in the … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L273 | **G.LM (256), T2, existing.** `vec256` of `build_v3_conditioning_bundle` and its Rust twin in `dope_core::mlc_v3`, unchanged: 23 gains, masks, midranks, 16 contrasts, target and shape, spectrum summary, group summary, … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L275 | **G.LM2 (34), T2.** Eleven further landmarks, each gain + fold SE + mask (33), same protocol as the native panel, plus the lasso's share of non-zero coefficients (1): | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L279 | L24 `gam8` — additive model on 8-bin indicators per column · additive non-linearity without any basis assumption | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L280 | L25 `otree2`, L26 `otree4` — one oblivious tree of depth 2 and 4 (with the existing depth 3) · interaction order | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L281 | L27 `lin@½`, L28 `gbdt@½` — half-data fits (with the existing quarter fits) · learning-curve shape | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L282 | L29 `gnb` — Gaussian naive Bayes (classification) · value of ignoring dependence | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L283 | L30 `qda−lda` — regularised QDA gain minus LDA gain (classification) · class-conditional covariance | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L284 | L31 `lasso` — L1 path at the 1-SE rule · sparsity of the signal | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L285 | L32 `pcr8` — ridge on the top-8 principal components · signal in the leading directions (pairs with G.ALIGN) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L286 | L33 `core_lin`, L34 `core_gbdt` — the scoreboard's `linear-seeded-v1` and `gbdt` recipes verbatim · the auditors themselves, as the landmark plan §4.2 asked | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L288 | **G.LC (6), T2.** Three-point learning-curve fit (n/4, n/2, n) for the linear and GBDT landmarks: exponent, asymptote, and the extrapolated gain at 4n (A.10). The evaluator trains on 4n synthetic rows; this says whether … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L290 | **G.CPLX (16), T2.** Data-complexity measures. Classification: Fisher ratio F1, directional F1v, overlap volume F2, feature efficiencies F3 and F4, borderline fraction N1 (minimum spanning tree), intra/extra-class … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L292 | **G.TWIN (8), T2.** Two random halves of FIT against each other: total variation between their linear and GBDT driver vectors (2: the real-vs-real driver ceiling); RMS difference of their correlation matrices; absolute … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L294 | **G.RESID (16), T2.** OOF residuals of the GBDT teacher (regression): `τ3`, `τ4 − 0.1226`, Hill lower/upper, global Breusch–Pagan `Z`, residual spread by quartile of the prediction (4), 7-knot residual quantile function … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L296 | **G.CAL (8), T2.** Reliability slope and intercept of the OOF linear and GBDT predictions (4), expected calibration error (2), resolution share of the Brier score (2). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L298 | **G.LIFT (6), T2.** AUC, lift in the top decile and KS statistic of the OOF linear and GBDT scores (binary; macro one-vs-rest for multiclass; masked for regression). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L300 | **G.WH (4), T2.** fANOVA shares of the teacher: main effects, pairs, order ≥ 3, largest single pair. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L302 | **G.RES (53), T3, closed loop.** For the candidate packet on an inner split: transfer retention `R_m` per landmark (23) and masks (23); cross-fitted boosted-tree C2ST AUC; median marginal AUC; DCR share; membership-AUC … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L305–309 | Treat the 2,048 B headline as a complete artifact cap. A lower rung below a legal base can refuse. | Antigravity | KEPT | Run `public-lite-replay` (`7c1f4598`, 310/310 DOPE files ≤ 2,048 B) + `v3-report-dev-s5-8` (`3fb802fd`, `primary.json` `55974867`): `dope-v3b-auto-v8` 108/108 packets ≤ 2,048 B (1,787 B med, 2,048 B max, 0/44 failed, 4× TRG −0.121 [−0.189, −0.049]). [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **4.7 Target-free variant and levels** |  |  |  |
| L306 | Two axes, kept separate: | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L308 | **Compute tier** (what the sender computed): `low` = T0, `med` = T0–T1, `high` = T0–T2, `ultra` = T0–T3. This is the level name for embeddings exported to downstream models. | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L309 | **Byte level** (what crosses the wire): 512 / 1,024 / 2,048 / 4,096 B. | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
| L311 | With `target_free = 1` every supervised block (C.Y, C.YF, C.SPEC, C.DRV, class tokens, G.Y, G.ALIGN, the supervised part of G.LM, G.LM2, G.LC, G.CPLX, G.RESID, G.CAL, G.LIFT, G.WH, edge channels 14–16 and 19) is masked. … | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
|  | **4.8 Relation to names already in the plan** |  |  |  |
| L317 | M5.5 class tokens — 4.5 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L318 | M5.5 dependence tokens / attention edge bias — 4.4, 5.2 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L319 | M5.5 type, lattice and functional-dependency descriptors — C.TYPE, C.BOUND, C.RULE, channels 10–12 | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L320 | M5.5 evaluator-matched signed teacher coefficients — C.DRV 10–11, C.Y 11, L33 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L321 | M5.5 relative learner-risk landmarkers — G.LM, G.LM2, G.RES | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L322 | M5.5 tree-region witnesses; `teacher:T21` tree regions — C.DRV 8–9, channel 16, section 5.4 forest section, 6.2 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L323 | `teacher:T21` reliability bins, dependence parents — G.CAL, C.DEP, channel 8 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L324 | `neural:N17` E03 scalar vs functional landmarks — rule 7, experiment E-5 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L325 | `landmark-signatures.md` T03 heteroscedasticity (always masked today) — C.Y 21–22, G.RESID | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **4.9 Planted-mechanism acceptance (Rust tests, one fixture each)** |  |  |  |
| L331 | Independent Gaussians, target independent — none · every `Z` within ±3 in ≥ 99% of draws; no eigenvalue above the MP edge | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L332 | `y = x1² + noise` — C.Y ξ `Z`, quadratic `Z`; low monotonicity ratio · Pearson `Z` | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L333 | XOR of two balanced binaries — channel 14, fANOVA pair share · every C.Y feature of both columns | Antigravity | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L334 | `y = 1[x1 > q80]` — stump location ≈ 0.8, step over hinge ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L335 | Heavy-tailed, heteroscedastic regression — Hill, C.Y 21, G.RESID ·  | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L336 | Classes with equal means, different variances — Brown–Forsythe, class KS, L30 · ANOVA `η²`, Pearson | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L337 | Clayton-copula pair — lower tail excess, copula-cell `χ²`, MI excess · Gaussian pair of equal `ρ` | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L338 | `x3 = x1 + x2` — `R²_j`, sum-constraint flag ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L339 | One-hot triple — sibling channel, group size ·  | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L340 | Zero-inflated column — inner atom, its location ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L341 | Integer lattice with heaping at multiples of 5 — lattice, heaping · a continuous column | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L342 | Two separated modes — dip, gap ratio and location · a unimodal column | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L343 | Missingness driven by `x1` — C.MISS 3 · MCAR control | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L344 | Duplicated correlated feature — redundancy cluster, C.Y 10 ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L345 | Binary target under 1% — events per variable, floor flags ·  | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L346 | Label-shuffled table — none · every supervised block | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L347 | Row-memorising generator (T3) — DCR share, membership proxy · a fresh real sample | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L348 | Monotone transform of a column; row and column permutation; column sign flip —  · rank-based features unchanged; equivariance exact | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **5.1 Tokens and embeddings** |  |  |  |
| L356 | **Tokens:** `P` column tokens, `K` class tokens (or one target token), one global token. With a schema, one-hot siblings also get a group token that pools them, and driver vectors are formed per group, as the metric … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L357 | **Scalar features:** piecewise-linear encoding on 8 fixed breakpoints per feature, then a per-block linear layer and LayerNorm. A single linear layer over raw scalars (today's `col_in`) makes thresholds hard to learn. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L358 | **Functional features:** C.QF, C.YF and the residual quantile function go through a small shared 1-D convolution stack (two layers, kernel 5) before the block projection. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L359 | **Masks:** a learned absent vector per block replaces the block embedding when masked. Block dropout 0.1 on top of tier dropout. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L360 | **Packet P0:** the dequantized packet fields that V7 reads today stay as one more column block (46 values), so the encoder sees what it is correcting. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **5.2 Encoder** |  |  |  |
| L364 | Pre-LN transformer over all tokens, `d = 192`, 4 layers, 6 heads [hyp: sizes to be set by E-9]. Attention logits between column tokens get an additive bias `Σ_c w[h, c]·E[j, k, c]` (one learned scalar per head and … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L366 | Rust inference follows `single_encoding/mod.rs`: linear layers, LayerNorm, attention and an added bias; all tensors stay rank ≤ 2 for the existing SafeTensors loader. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **5.3 Outputs** |  |  |  |
| L370 | **(a) Shrinkage of the stored statistics, bounded in standard-error units.** For each packet field `θ̂` with standard error `se` (from C.UNC, the ridge covariance, or the bin count): | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L372–376 | θ = θ_prior + κ·(θ̂ − θ_prior) + δ κ = 1 + 0.5·tanh(h_κ) identity at initialisation δ = min(cap, 2·se)·tanh(h_δ) zero at initialisation | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L378 | `θ_prior` is the Gaussian reference for knots and zero for loadings, target weights, `Q_nl` bins, tree leaves and class factor means. This replaces V7's fixed bounds (0.06, 0.10, 0.50) with a trust region in statistical … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L380 | **(b) Byte allocation.** For each optional section (5.4) a head predicts the retention gained per byte. A deterministic greedy fill admits sections in that order while the packet stays under the cap and the predicted … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L382 | **(c) A learned code `z`.** Finite-scalar quantisation, 5 levels per dimension, straight-through: 48 global dimensions (14 bytes) and 3 per column (1 byte per column). Nested dropout over the global dimensions makes … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **5.4 Optional packet sections** |  |  |  |
| L386 | All are explicit statistics with an analytic fit and an analytic decode, each admitted by a statistical gate first (as DSE1 does for S8 and S10) and then by the allocator. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L390 | `Q_nl` bitmap — `Q_nl` only for gated columns · `⌈P/8⌉ + 4·m` instead of `4·P` · DSE1 §7.8 χ² gate · bins ≥ 10 rows | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L391 | Forest — residual oblivious trees, depth 2–3; per tree `d` (column index, threshold as an index into that column's own decoded quantile function) and `2^d` companded leaf values per output · 2 + per tree `2d + O·2^d` (14 B at depth 3, one output) · fitted on the residual of linear + `Q_nl`, leaves shrunk with `count + 8`; … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L392 | Noise shape — 7-knot quantile function of the standardized OOF residual; spread by quartile of the prediction · 7 + 4 · kept if G.RESID rejects Gaussian, homoscedastic noise at the family-wise 5% level · | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L393 | Class regimes — class factor means with mean carriers (port of DSE1 §5.4, §7.5) · `4·K` per rank block · kept if between-class share of latent variance exceeds its permutation null · classes ≥ 10 rows | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L394 | Target support — lattice step or atom table of a discrete regression target · 2–10 · when G.Y flags a lattice or atoms · atoms ≥ 10 rows | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L395 | Missingness — rate per column with missing cells · `⌈P/8⌉ + m` · when any `miss_rate` is non-zero; decode needs an agreed missing representation with the evaluator (open decision 4) · | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L396 | Code `z` — 5.3 (c) · 14 + P · E-8 · canary tests | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L398 | At depth 3 a forest of 40 trees costs 562 bytes, which fits beside a full-rank 24-column packet (C.3). Thresholds are stored as quantile indices, so no raw value is added to the packet. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L400 | **Column selection for P > 48** uses the largest of the C.Y `Z` scores (Pearson, ξ, MI, class KS) instead of Pearson with the class index (F9), with ties broken by C.DEP redundancy so near-duplicates do not crowd out … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **5.5 Receiver** |  |  |  |
| L404 | Unchanged contract: row `i` depends only on (packet, seed, `i`); no batch statistic; labels are draws. Decode order: features from the copula and marginals (with class regimes when present) → `η = b + W·x + Q_nl +` … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L406 | **`stat` kind (default):** no network at the receiver. The cold closure drops by the 2.76 MB of V7 weights; the packet is plain statistics that can be inspected. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L407 | **`hybrid` kind (only if E-8 passes):** the receiver additionally runs the V7-style hyper head on (packet, `z`). | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
|  | **5.6 Encode-time acceptance against step 0** |  |  |  |
| L411 | `dope fit --format auto` already scores two candidates on a grouped 75/25 split of the input rows and keeps the challenger only if it wins by more than 0.02 [read: `engine/dope-core/src/auto/mod.rs:43`, `auto/check.rs`]. … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L413 | This makes the trained model's contribution non-negative by construction up to selection noise, measured on real auditors rather than a proxy (`neural:N12`). Report "encoder accepted on x% of tables" beside every result. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
|  | **5.7 Embedding export** |  |  |  |
| L417 | The pooled global token before quantisation (192-D) and the mean of the column tokens (192-D) form `dope.dfp-embedding/v1` at the compute tier that was run (4.7), with a target-free twin. This is separate from the pinned … | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
|  | **6. Objective** |  |  |  |
| L423 | Episode roles: **A** (FIT: packet and all features), **B** (a disjoint real twin of the same size), **Q** (disjoint query rows, ≥ 256 when the table allows). `S` is the decoded synthetic sample, differentiable in the … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.1 Twin-referenced retention deficit** |  |  |  |
| L427 | For each surrogate family `f`, fit on `S` (with gradient), on `A` and on `B` (no gradient), score all on `Q`: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L429–433 | gain_f, d_f | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L435 | This is the contract's statistic turned into a loss: same normalisation, same twin subtraction, worst family. The twin term is a control variate (it shares `Q` and the table's difficulty with the synthetic term) and the … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L437 | Surrogate families, none of them the held-out MLP: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L441 | linear — ridge / logistic / multinomial IRLS on standardized inputs, 8 Newton steps, the evaluator's penalties | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L442 | kernel — the existing RFF ridge, widened to 128 features at three bandwidths from a fixed bank (7.3) | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L443 | tree-region — 6.2 | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L444 | local — Nadaraya–Watson on `S` with a bandwidth set on `A` | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.2 Tree-region surrogate** |  |  |  |
| L448 | Fit the teacher forest on `A` (not differentiable; it is the C.DRV teacher). Freeze its split columns and thresholds. On `S`, compute soft leaf memberships (`sigmoid((x − t)/τ)` per split, `τ` annealed 0.5 → 0.1) and … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.3 Driver term** |  |  |  |
| L452 | For linear the grouped permutation importance has a closed form under squared loss, `2·β_j²·Var_Q(x_j)` summed over the group; for classification the same quantity on the logit scale is the proxy. For the tree-region … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L454–456 | ℓ_drv = Σ_f relu( TV(p_{S,f}, p_{A,f}) − TV(p_{B,f}, p_{A,f}) ) + pairwise rank loss on the top-8 real drivers | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L458 | The first term is the metric's mass agreement measured against its own real-vs-real ceiling. The rank term reuses `pairwise_logistic` from `transfer_losses.py`, with pairs restricted to gaps larger than the twin's. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **6.4 Realism** |  |  |  |
| L462 | All terms are per episode with their own masks, against reference rows `B ∪ Q` (features only), each minus the same statistic between `A` and the reference (detached), floored at zero. The baseline subtraction means the … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L464 | **Tree-kernel discrepancy:** on the 64 random trees of 6.2, L1 distance between leaf-occupancy vectors of `S` (soft) and of the reference, plus the occupancy-weighted distance between leaf target means. This is what a … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L465 | **Marginals:** Wasserstein-1 per continuous column from sorted samples; total variation of atom masses per discrete column. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L466 | **Dependence:** squared difference of normal-score correlations, each pair weighted by `n_ref/(n_ref + 64)`. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L467 | **MMD:** unbiased multi-bandwidth RBF estimate. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L468 | **Legality:** mass of `S` off the stored lattice, atoms or bounds. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **6.5 Conditional likelihood, privacy, anchor, rate** |  |  |  |
| L472 | `ℓ_cond`: the existing proper-scoring term on `Q`, extended with a pinball loss at 0.1 and 0.9 so the noise shape is trained, not only its scale. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L473 | `ℓ_priv = relu(share_S − 0.55)`: soft share of synthetic rows closer to `A` than to `B` (the DCR gate as a hinge). Required once `z` exists. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L474 | `ℓ_anchor`: L2 on `h_κ`, `h_δ`. With bounds in SE units this can be small. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L475 | `ℓ_rate = λ·bytes/2048`, with `λ` drawn log-uniformly per episode and given to the encoder as an input, so one model covers the byte grid. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.6 Total** |  |  |  |
| L479 | Each term is divided by an exponential moving average of its own magnitude (detached) before weighting, so the weights mean what they say: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L481–483 | L | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L485 | These weights are starting values [hyp], to be set by E-6. Auxiliary heads (reconstruct masked DFP1 blocks from the global token; agreement of the encodings of two FIT halves) get 0.1 each and are dropped if they do not … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **7.1 Data path (FP-0)** |  |  |  |
| L493 | **Jitter:** build episodes with `s3_view_builder` / `s3_view_scale.raw_augment` (lattice columns untouched, jitter on 30% of episodes), or port that rule into `build_unique_v7_episode`. Log `disc_col_share` per step in … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L494 | **Sizes:** draw `n_fit` with the sampler's eval-matched law (`s3_counter_sampler.draw_sizes`; plan target ≥ 40% of episodes at `n_fit ≥ 256`), up to 1,024. | Claude | TESTING | [Episode stream](#episode-stream) |
| L495 | **Roles:** add twin `B`; raise the query cap to 512. Rows are free in the memory-mapped arena. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L496 | **Kinds:** add `kinds.u8` and `group.u16` per column to the arena (`dope.flat-corpus/v2`), filled from the corpus builder's schema where known and by C.TYPE / channel 12 inference otherwise. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L497 | **Missing cells:** pass NaN through to the feature extractor; impute after. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **7.2 Per-episode statistics** |  |  |  |
| L501 | Replace the batch-minimum truncation with per-episode masks in every realism term. No statistic may be computed across episodes. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **7.3 Variance reduction** |  |  |  |
| L507 | Common random numbers — generator noise from the counter RNG keyed by (run seed, split hash); the same episode always gets the same noise | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L508 | Antithetic pairs — second half of the synthetic rows uses the negated normals | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L509 | Synthetic rows — `n_syn = min(4·n_fit, 1024)` (the evaluator uses 4n; 192 rows make the surrogate fit itself noisy) | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L510 | Fixed projection bank — 256 RFF frequencies and 128 projections drawn once per run; each step uses a deterministic rotating subset | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L511 | Per-episode normalisation and Huber — 6.1; replaces the raw clamps at 25 and 10 | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L512 | Term normalisation — 6.6 | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L513 | Clipping — clip at the running 80th percentile of the unclipped norm over the last 200 steps; skip a step above 10× the median (`spike_rule` in `s3_grad_farm.py`) | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L514 | Batch — set from the measured gradient-noise scale; start at an effective 64 episodes (accumulate 4) | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L515 | EMA — keep three (0.99, 0.999, 0.9998) and choose on DEV-select; 0.992 is a horizon of 125 steps in a 12,000-step run | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L516 | Annealing — tree `τ` 0.5 → 0.1; FSQ with additive uniform noise for the first 30% of steps, then straight-through | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L518 | Instrument before tuning, as plan M6 says: per-term gradient norm and pairwise cosine, clipped-step share, gradient-noise scale, loss variance on a frozen 256-episode probe. The plan's targets stand: clipped share under … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **7.4 Feature supply** |  |  |  |
| L522 | Prefetch workers call the Rust extractor through a C ABI (Appendix B) and pad to `MAX_COLS`. T2 and T3 run on their fixed shares with reduced settings (3 folds, 32 rounds). The extractor is single-threaded per call; … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **8. Monitor and reporting** |  |  |  |
| L528 | The scoreboard stays the only authority. The training-time monitor should stop disagreeing with it: | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L530 | Compute the monitor's headline as TRG on a DEV-lite schedule with `dope_kpi` (linear and GBDT, twin subtraction, 4n rows, sampled labels). Today's monitor is a median over ridge and a 40-round HGB, uses the twin only for … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L531 | Add grouped permutation-importance agreement (mass and rank, over the twin ceiling) for both auditors. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L532 | Always print beside TRG: Δ against step 0 with its interval, the share of tables on which the encoder was accepted (5.6), bytes used (median and max), and TRG at each byte level. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L533 | Diagnostics, never gates or composites: agreement of partial-dependence curves of the top-5 drivers; calibration gap; decile-lift curve gap; per-class recall gap; rank agreement of the top interactions. These answer "do … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **9. Work packages** |  |  |  |
| L539 | Each ends in a measured result. Rust files stay under 500 lines; new areas are added to `agent/owner-map.json` and `agent/test-map.json`. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L543 | **FP-0** — Loop repairs 7.1 (items 1–3, 5), 7.2, 7.3, objective unchanged; then 6.1 alone · `s3_v7_batch_ops.py`, `s3_v7_fast_trainer.py`, tests under `research/levels/tests/` · `disc_col_share` matches DEV; no cross-episode … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L544 | **FP-1** — DFP1 T0 + T1 extractor, layout file, C ABI; arena kinds and groups (7.1 item 4) · new crate `engine/dope-features/` (`column/`, `pair/`, `global/`, `target/`, `null/`, `abi.rs`, `layout.rs`); … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L545 | **FP-2** — Optional sections of 5.4 with gates, analytic decode, Python decode mirror · `engine/dope-core/src/single_encoding/sections/`, `alloc.rs`; `research/levels/v8_packet_torch.py` · round trip; byte table; … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L546 | **FP-3** — Objective 6.1–6.5 · `research/levels/v8_losses.py`, `v8_tree_surrogate.py` · relaxed-vs-hard gap logged; on a frozen probe the surrogate ranking of 20 … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L547 | **FP-4** — Sender encoder, trainer, Rust inference · `research/levels/v8_sender_encoder.py`, `v8_tokens.py`, `s3_v8_trainer.py`; `engine/dope-core/src/single_encoding/sender.rs` · … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L548 | **FP-5** — DFP1 T2 + T3 · `engine/dope-features/src/learners/`, `residual/` · fixtures; reuse of `landmark/`, `teacher/`, `dvs/permutation.rs` verified by … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L549 | **FP-6** — Three-way acceptance 5.6 · `engine/dope-core/src/auto/check.rs`, `auto/mod.rs` · never selects a candidate below P0 by more than the margin on the check split … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L550 | **FP-7** — Embedding export 5.7 · `engine/dope-features/src/embed.rs` · split-half cosine above the twin reference; permutation stable | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L552 | FP-1 and FP-2 are independent and can run in parallel after FP-0. FP-2 needs no network and no feature pack beyond the statistics its gates use. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **10. Experiments and decision rule** |  |  |  |
| L558 | Decision rule, unchanged from plan M5.6: fixed budget, ≥ 3 training seeds, same sampler plan for arm and control, scored by `dope_kpi` on DEV-select. KEEP if the lower bound of ΔTRG is above 0; DROP if the upper bound is … | Claude | TESTING | [Power and promotion](#power-and-promotion) |
| L562 | E-0 — Which component carries the gap? · plan M5.1 attribution: real X + packet Y, packet X + real Y, n vs 4n. Run first; it orders everything below | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L563 | E-1 — Did the loop defects hide a working model? · V7.2 as released vs FP-0 repairs, both against step 0 | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L564 | E-2 — What do explicit sections buy without a network? · P0 vs + forest vs + noise shape vs + class regimes, at equal bytes | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L565 | E-3 — Does sender-side information matter? · sender encoder with packet-only input vs + T0/T1 vs + T2 | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L566 | E-4 — Which blocks pay? · leave-one-block-out and add-one-block-in over {QF, SHAPE+TAIL, TYPE+BOUND+RULE, edges 1–13, C.Y, C.YF, class tokens, DRV, G.LM, … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L567 | E-5 — Scalar tests or curves? · functional blocks only vs scalar only vs both | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L568 | E-6 — Which objective terms pay? · raw V7 loss vs 6.1 vs + tree-region vs + driver term; realism terms one at a time | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L569 | E-7 — Closed loop · T3 off vs on; acceptance margin 0, 0.02, 0.05 | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L570 | E-8 — Learned code · global `z` prefix of 0, 16, 32, 48 dimensions, with and without the per-column byte, against the same bytes spent on forest trees | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L571 | E-9 — Capacity · `d` 128/192/256, layers 2/4/6, with the scaling evidence `neural:N07` requires | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L572 | E-10 — Calibration of evidence · features with and without the `Z` channels and C.UNC | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L574 | My expectations, to be confirmed or refuted [hyp]: E-1 moves the trained model from below step 0 to at least equal; E-2's forest section is the largest single gain on tables with tree-visible structure; in E-4 the … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **11. Risks and open decisions** |  |  |  |
| L580 | **Risks** | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L582 | Confirm surviving ablations once on DEV-confirm / protected split (blocked by active DEV-only pause; never run). | Owner | NOT STARTED | Policy-blocked by DEV-only pause — never run. [Execution scope](#execution-scope) |
| L583 | *Small n.* Most features are noise at n = 32. Masks by `n_min`, the `Z` channels and tier dropout are the mitigation; E-10 measures whether they work. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L584 | *Leakage through `z`.* A learned code can memorise. Canary tests, the DCR hinge and the disclosure gates apply before any `hybrid` packet is scored. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L585 | *Optimism of the closed loop.* T3 features and the acceptance check use FIT-internal splits; at small n the split is noisy, hence the margin and the reported acceptance share. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L586 | *Compute.* T2 and T3 at encode time are the cost of "accurate" mode. Report encode seconds per tier; `fast` mode must remain T0 + T1. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L587 | *Sequencing.* The gap matrix (`neural:N01`) puts the learned lane after the scoreboard gates. FP-0, FP-1 and FP-2 do not depend on that ruling; FP-4 does. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L589 | **Decisions for the owner** (each has a default so work is not blocked) | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L591 | *Base packet.* DSE7 v2 (dope-core, embedded scale card, product path) or DSE1 (dope-arena, levels A–E, gated S8/S10, class carriers). Default: DSE7 container, port DSE1's gates and class carriers into it. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L592 | *Receiver weights.* May the receiver need a shared network at all? Default: no (`stat` kind) unless E-8 passes. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L593 | *Encode-time budget.* Seconds allowed for `full` mode per table. Default: report, do not cap, until measured. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L594 | *Missing cells.* How the evaluator represents them decides whether the missingness section can be scored. Default: off. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L595 | *Arena schema.* Adding kinds and groups changes `dope.flat-corpus` to v2. Default: new version beside v1. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **Appendix A. Formulas** |  |  |  |
| L601 | **A.1 L-moments.** With order statistics `x_(1) ≤ … ≤ x_(n)` and `b_r = n⁻¹ Σ_i [(i−1)(i−2)…(i−r)] / [(n−1)(n−2)…(n−r)] · x_(i)`: `l2 = 2b1 − b0`, `l3 = 6b2 − 6b1 + b0`, `l4 = 20b3 − 30b2 + 12b1 − b0`, `τ3 = l3/l2`, `τ4` … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L603 | **A.2 Quantile shape.** Bowley `(q75 + q25 − 2·q50)/(q75 − q25)`. Moors `((o7 − o5) + (o3 − o1))/(o6 − o2)` with octiles `o_k`; Gaussian value 1.233. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L605 | **A.3 Hill.** Upper tail with threshold `t = x_(n−k)`: `γ̂ = k⁻¹ Σ_{i=1..k} ln((x_(n−i+1) − m)/(t − m))`, `m` the median, `k = clamp(round(√n), 10, n/10)`. Lower tail by symmetry. Emit `squash(γ̂)`. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L607 | **A.4 Chatterjee ξ.** Sort by `x`, let `r_i` be the rank of `y` in that order: `ξ = 1 − 3·Σ abs(r_{i+1} − r_i)/(n² − 1)`; with ties use the tie-corrected form already in `dope_core::stats::chatterjee_xi`. Null for … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L609 | **A.5 Binned mutual information and Gaussian excess.** Equal-frequency bins, `B = clamp(round(n^{1/3}), 2, 8)` per axis. `G = 2n·MI` is `χ²` with `(B−1)²` degrees of freedom under independence; Miller–Madow subtracts … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L611 | **A.6 Noise floor.** Regression: `σ̂² = (2n)⁻¹ Σ_i (y_i − y_{NN(i)})²`, nearest neighbour in the standardized top-8 columns by ξ. Classification, with 1-NN leave-one-out error `e1` and `K` classes: Bayes error `e* ≥` … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L613 | **A.7 Spectrum.** Marchenko–Pastur upper edge `(1 + √(P/n))²` for a correlation matrix of independent columns. Ledoit–Wolf intensity toward the identity on normal scores. TwoNN: `d̂ = n' / Σ ln(r2_i/r1_i)` over the 90% … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L615 | **A.8 Binned interaction.** Cells from the two columns' quartiles; cell means `m_ab` of `r` with counts `n_ab ≥ 10`. Fit `m_ab ≈ μ + α_a + β_b` by weighted least squares; `SS = Σ n_ab·(m_ab − μ − α_a − β_b)²`; `F =` … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L617 | **A.9 fANOVA of oblivious trees.** A depth-`d` oblivious tree is a table `f(b_1…b_d)` over split indicators. Its Walsh–Hadamard coefficients `c_S = 2^{−d} Σ_b f(b)·Π_{i∈S}(2b_i − 1)`, weighted by leaf occupancy, give … | Antigravity | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L619 | **A.10 Learning curve.** With gains `g1, g2, g3` at n/4, n/2, n: `b = clamp(log2((g2 − g1)/(g3 − g2)), 0.25, 1.5)`, `g∞ = g3 + (g3 − g2)/(2^b − 1)`, `g(4n) = g∞ − (g∞ − g3)·4^{−b}`; masked when the gains are not … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L621 | **A.11 Signal–spectrum alignment.** With eigenpairs `(λ_k, v_k)` of `R` and ridge `β` on normal scores: `A_r = Σ_{k≤r} λ_k (v_k·β)² / Σ_k λ_k (v_k·β)²`; for classification sum over the `K − 1` discriminant directions. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L623 | **A.12 Finite-scalar quantisation.** `q = round(2·tanh(h))`, five levels, straight-through gradient; 48 dimensions pack into `⌈48·log2(5)/8⌉ = 14` bytes by mixed radix; three dimensions (125 codes) pack into one byte. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **Appendix B. Extractor interface** |  |  |  |
| L627–654 | /* engine/dope-features, cdylib. No panic crosses the boundary; errors are a fixed enum and never echo a cell, header or extremum. */ typedef struct { uint32_t abi; /* 1 */ | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L656 | The Python side is a `ctypes` binding and a collate function; it computes nothing. The product path calls the same Rust functions directly. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |

</details>

### Evidence

<details>
<summary>Source proposals and DEV test routing</summary>

| Source | Proposal | Owner | State | DEV test |
| --- | --- | --- | --- | --- |
|  | **0. Bottom line** |  |  |  |
| L10 | **Today no feature of the real table can reach the model unless it is already in the packet.** `SingleEncV7.refine_packet(col_feats, glob_feats, col_mask)` runs on the receiver from the ≤ 2 KiB `DSE7` bytes alone … | Claude | NOT STARTED | [E0 attribution](#e0-attribution) |
| L11 | **So "more features" needs a place to live.** This spec adds four channels: (A) an owner-side *evidence encoder* that reads a large statistic bank of the normalised FIT rows and writes a small quantised code into the … | Claude | NOT STARTED | [E0 attribution](#e0-attribution) |
| L12 | **Measured where the gap is** (81 public tables, your packet code, §2): the step-0 packet scores −0.31 on a TRG-like statistic (your scoreboard: −0.39). Fixing the feature distribution alone gives +0.05; fixing the … | Claude | NOT STARTED | [E0 attribution](#e0-attribution) |
| L13 | **Realism is mostly a marginal-representation problem, and it is cheap to fix.** Boosted-tree C2ST: packet 0.87, exact marginals with the same rank-12 dependence 0.70, twin 0.51. 45% of lattice-valued columns are … | Claude | TESTING | [E0 attribution](#e0-attribution) |
| L14 | **Univariate statistical tests are not the big lever for drivers.** The packet's linear weights already track real GBDT permutation importance at Spearman 0.51; mutual information, correlation ratio, KS and best-split … | Claude | TESTING | [E0 attribution](#e0-attribution) |
| L15 | **Order of work** (each step has a measured gate, §9): typed marginals and group-exclusive decode → regime dependence → analytic main effects and distilled oblivious trees → evidence encoder with a quantised code → … | Claude | NOT STARTED | [E0 attribution](#e0-attribution) |
|  | **1.3 What we optimise for** |  |  |  |
| L68 | In priority order, as the contracts state it: (1) TRG at ≤ 2,048 B, which needs the *worst* of the linear and boosted-tree auditors to match a second real sample; (2) realism coverage (C2ST ≤ 0.70, max marginal AUC ≤ … | Claude | NOT STARTED | [T90 and D80](#t90-and-d80) |
|  | **2.3 Which FIT-only statistics know the drivers a tree model will use** |  |  |  |
| L168 | The witness has to be a properly boosted, cross-fitted model. A random-candidate oblivious forest (32 trees, 8 random splits tried per level) scored 0.47 in the same test, below the stored weights. | Antigravity | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L169 | So the catalogue of §4 keeps the cheap tests mainly for their noise channel (`z`), and puts weight on model-based witnesses, on support detection, and on what the packet gets wrong. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **3. Design rules** |  |  |  |
| L194 | **R1. Every feature names its channel.** A feature that cannot name one of these has no effect on a sampled row and is not built. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L198 | **A. Owner evidence → code** — Statistics of the FIT rows read by an owner-side encoder that writes a quantised code into the packet · Yes (2–8 B per column, 16–64 B global) · Yes (`CODE` section) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L199 | **B. Packet-derived** — Functions of the dequantised packet, recomputed by the receiver · No · No | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L200 | **C. Typed statistic section** — A stored statistic with an analytic decode rule (lattice step, atom table, group map, regime means, main effects, trees) · Yes, per section · Yes | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L201 | **D. Training-only** — Loss terms, auxiliary prediction targets, teacher soft targets, sampler policy · No · No | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L203 | **R2. Analytic first, learned residual on top.** Every new section decodes without weights, and the network's contribution to it is zero-initialised, so step 0 of training equals the analytic packet on the scoreboard. … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L205 | **R3. Evidence comes as estimate, noise and presence.** Each dependence statistic is supplied as `(value, z, mask)`: the estimate, its z-score under the no-association null at this `n_fit`, and a presence bit. Training … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L207 | **R4. FIT rows only, with a poison test.** Every evidence function takes `(X_fit, y_fit, schema)` and nothing else. Test: replacing QRY/T labels and rows with garbage leaves every evidence byte and every packet byte … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L209 | **R5. Support floor everywhere.** No stored or encoded statistic may depend on fewer than `k = 10` rows (atoms, classes, regimes, tree leaves, bins), as `single-encoding-v1.md` §10 already requires for the stat packet. … | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L211 | **R6. One implementation for serving, one mirror for training.** Evidence extraction is Rust (`AGENTS.md` language policy). A batched torch mirror exists only inside the training step, pinned to the Rust output by parity … | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L213 | **R7. A feature family is kept only on the scoreboard.** KEEP if the lower bound of ΔTRG is above 0 at equal measured bytes, DROP if the upper bound is below +0.01, otherwise not adopted (plan M5.6). No proxy-loss … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **4.1 Conventions** |  |  |  |
| L221 | Inputs: `X` = FIT rows in model unit space (`n × p`, after the normalise-once step and the FIT-only scale card), `y`, the NC1 public `UnitSchema`, and the raw FIT values where a row says "raw". Nothing else. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L222 | `s(x) = x / (1 + &#124;x&#124;)`. All outputs are finite, in [−1, 1] or [0, 1], float32. A statistic that cannot be computed (constant column, too few rows) is 0 with mask 0. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L223 | `(v, z)` = value and null z-score, both squashed; null formulas are in Appendix B. Statistics without a closed-form null share `Q = 8` permutations of `y` across all columns. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L224 | Multiclass: per-class statistics are reduced by root-mean-square over classes unless the row says otherwise. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L225 | Priority: **P0** has measured headroom in §2; **P1** is expected to help and is cheap; **P2** is exploratory and goes in last. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L226 | Ch = channel (A code, B packet-derived, C typed section, D training-only). "A" features feed the encoder; a "C" next to it means the same detection also drives a stored section. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L228 | Shared primitives, each computed once per table and reused by every row below: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L232 | P1 — one stable sort per column · sorted values, order, mid-ranks, gaps, normal scores `Z` · `O(n p log n)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L233 | P2 — 16 value-consistent quantile bins per column (equal values share a bin) · bin index `n × p` (u8) · `O(n p)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L234 | P3 — normal-score correlation · `p × p` · `O(n p²)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L235 | P4 — ridge linear probe of the centred response on standardised `X` (IRLS for classes), with hat-matrix diagonal · coefficients, residual `R`, leave-one-out predictions · `O(n p² + p³)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L236 | P5 — per-bin sums of `Y`, `R`, `&#124;R&#124;` and counts · `p × 16 × C` · `O(n p C)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L237 | P6 — pairwise distances on a stratified subsample of `m ≤ 256` rows in normal-score space · `m × m` · `O(m² p)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L238 | P7 — 2-fold cross-fitted boosted oblivious forest on the P2 bins (40 trees, depth 3, shrinkage 0.2, exhaustive search over `p × 15` edges; the engine's existing oblivious booster) · out-of-fold predictions, per-column gain, held-out permutation importance, co-split matrix · `O(T d n p)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L239 | P8 — analytic packet decode: per-column CDF/quantile function, implied correlation, implied `η` · closed forms + 256 counter-seeded rows · `O(n p)` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **4.2 Column evidence (128 per column)** |  |  |  |
| L243 | **S — support and type (28).** Feeds the encoder and decides the typed marginal section. This block is the realism fix of §2.2. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L247 | S01 — kind · one-hot {numeric, integer, boolean, ordinal, nominal level, missing indicator} from `UnitSchema.groups` · 6 · B, C · P0 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L248 | S02 — group geometry · `log2(width)/4`; position/width; has_missing; FIT mass of this level · 4 · B, C · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L249 | S03 — distinctness · `ln d / ln n`; `1 − d/n` (`d` distinct values) · 2 · A · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L250 | S04 — atoms · values with count ≥ 10 and mass ≥ 3× the smooth expectation, at most 32: `log2(1+a)/5`; total atom mass; top-3 masses; mass entropy … | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L251 | S05 — boundary and zero mass · mass at the lowest atom, at the highest atom, at raw 0 · 3 · A, C · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L252 | S06 — decimal lattice (raw values) · coarsest `q0 ∈ {1, 2, 2.5, 5}·10^d` with ≥ 99.5% of values on it (gap-matrix rule M01): has-lattice; `log10(range/q0)/6`; share on … | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L253 | S07 — heaping · excess share on multiples of {2, 5, 10, 100}·`q0` over the smooth body's prediction · 4 · A, C · P1 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L255 | **M — marginal shape (28).** M07 tells the encoder exactly where the stored knots are wrong; the rest describes tails and modes the knots cannot hold. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L259 | M01 — L-moments · L-skew `τ3`, L-kurtosis `τ4` from sorted values · 2 · A · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L260 | M02 — quantile shape · Bowley skew; `s`(Moors kurtosis − 1.23); `tanh((q99 − q75)/(4·IQR))`; `tanh((q25 − q01)/(4·IQR))` · 4 · A · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L261 | M03 — tails · Hill index on the top and bottom `⌈√n⌉` excesses, `s(·)`; sign of the mean-excess slope, upper and lower · 4 · A, C · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L262 | M04 — distance from normal · `1 − W′` (Shapiro–Francia on the standardised column); `s`(Anderson–Darling `A*²`) · 2 · A · P2 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L263 | M05 — modes · dip statistic z; Ashman's D of the best 1-D two-way split (exact from prefix sums); valley depth between the two modes · 3 · A · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L264 | M06 — entropy · Vasicek spacing entropy of the standardised column; 16-bin entropy / `ln 16` · 2 · A · P2 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L265 | **M07** — **packet-marginal residual** · PIT of FIT values through the *decoded packet* marginal (P8): 8-bin histogram as `s((share − 1/8)·√(8n))`; KS `(v, z)`; W1 in unit … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L267 | **D — dependence between features (16).** | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L271 | D01 — correlation profile · max, second max and mean of `&#124;ρ_ns&#124;` over partners · 3 · A (B at stored rank) · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L272 | D02 — redundancy · squared multiple correlation `1 − 1/VIF` from the precision of `R_ns + 0.05 I` · 1 · A · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L273 | D03 — communality · `&#124;L_j&#124;²` at rank 4, 8, 12 · 3 · B · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L274 | D04 — factor residual · max and RMS of row `j` of `R_ns − L Lᵀ` at the stored rank · 2 · A · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L275 | D05 — functional dependence · `max_i ξ(x_i → x_j)`, `max_i ξ(x_j → x_i)`, `max_i (ξ − ρ_s²)` · 3 · A · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L276 | D06 — tail co-movement · max over partners of the upper and of the lower 10% co-exceedance rate minus the Gaussian-copula value at the same `ρ` · 2 · A · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L277 | D07 — determinism (discrete columns) · `min_i H(x_j &#124; x_i) / H(x_j)` over discrete partners · 1 · A, C · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L278 | D08 — co-missingness · max `φ` between this column's missing flag and any other · 1 · A · P2 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L280 | **R — relation to the target (48).** R09–R14 describe the conditional the tree auditor will look for. R01–R08 are cheap and mostly redundant with the stored weights (§2.3); they are kept for the `z` channel and for … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L284 | R01–R08 — marginal association, each `(v, z)` · Pearson; Spearman; Chatterjee ξ; mutual information (8 × 8 bins, Miller–Madow corrected, over `H(y)`); correlation ratio `η²` (16 … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L285 | R09 — best split · max between-children share over the 15 bin edges `(v, z)`; threshold position in [0, 1] · 3 · A · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L286 | R10 — conditional linear relevance · `s(t)` of the multivariable coefficient; drop-column loss share `β_j² / (G⁻¹)_jj`, normalised; sign · 3 · A (B: stored weight) · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L287 | R11 — main-effect nonlinearity · `η²` of the linear residual on the binned column `(v, z)`; quadratic and cubic contrasts of the 16-bin residual profile; share of … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L288 | R12 — heteroscedasticity · `η²` of `&#124;R&#124;` on the binned column `(v, z)` · 2 · A, C · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L289 | R13 — interactions · max over the 10 most relevant partners of the pure 4 × 4 interaction share of `R` `(v, z)`; that partner's relevance · 3 · A, C · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L290 | **R14** — **tree witness (P7)** · held-out permutation-importance share; gain share; split-count share; mean threshold position; threshold spread · 5 · A, C · **P0** | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L291 | R15 — split-half stability · between two disjoint halves of FIT: `&#124;v₁ − v₂&#124; / (&#124;v₁&#124; + &#124;v₂&#124; + ε)` for R10 and for R14; sign agreement of the coefficient · 3 · A · … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L292 | R16 — class effects · standardised mean difference of the column in class `c` against the rest, classes in packet order, zero-padded · 8 · A · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L294 | **X — missingness and realism witness (5), reserved (3).** | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L298 | X01 — missing rate · share of missing cells · 1 · A, C · P1 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L299 | X02 — informative missingness · MI(missing flag; `y`) `(v, z)` · 2 · A · P1 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L300 | X03 — MAR strength · max `&#124;corr&#124;` between the missing flag and any other column's value · 1 · A · P2 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L301 | C05 — discriminator witness · this column's gain share in the owner-side C2ST of §4.6 · 1 · A · P0 | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **Conflicting release layout** |  |  |  |
| L303 | Eight ordered-pair coordinates as the release ABI (superseded by frozen `contracts/evidence-layout-v1.json` EDGE=16; not run). | Antigravity | NOT STARTED | Superseded draft ABI — never run. [Evidence bundles](#evidence-bundles) |
|  | **4.3 Pair evidence (8 per ordered pair, dense `p × p`)** |  |  |  |
| L305 | Used as an additive attention bias in encoder and decoder (`bias_ij = MLP₈→heads(e_ij)`), and to pick the sparse pairs that get stored corrections. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L309 | E01 — `ρ_ns` normal-score correlation · A (B: `L_i·L_j` at stored rank) · P0 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L310 | E02 — residual `ρ_ns − L_i·L_j` · A · P0 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L311 | E03, E04 — `ξ(x_i → x_j)`, `ξ(x_j → x_i)` · A · P1 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L312 | E05 — mutual information, 8 × 8 bins, corrected, normalised · A · P1 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L313 | E06 — upper minus lower 10% co-exceedance, relative to Gaussian · A · P1 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L314 | E07 — interaction share of the pair with respect to `y` (the R13 matrix; 0 outside the candidate set) · A, C · P0 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L315 | E08 — same NC1 group · B, C · P0 | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **4.4 Group, class and target tokens** |  |  |  |
| L321 | Group (one per NC1 group) — kind one-hot (5); `log2(width)/4`; has_missing; level-mass entropy · 8 · B, C · P0 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L322 | Class (one per class, K ≤ 8) — prior; `log2(count)/10`; below-floor flag; centroid norm in factor space; within/total variance; `s`(Mahalanobis distance to the nearest class); k-NN purity (P6); one-vs-rest recall of the linear probe (LOO); best single-column one-vs-rest AUC; mean LOO margin; calibration gap (mean LOO probability − prior); reserved · 12 · A (B: prior) · P1 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L323 | Target — the target block of §4.5 routed to its own token so target heads attend to it · 16 · A · P1 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **4.5 Global evidence (160)** |  |  |  |
| L329 | N — regime — `log2 n/12`; `log2 p/6`; `p/n`; `log2(min class count)/10`; unique rows / `n`; Marchenko–Pastur edge `(1 + √(p/n))²/4` · 6 · B (n, p), A · P0 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L330 | T — target — regression: L-skew, L-kurtosis, tail weights (2), dip z, lattice flag, integer flag, mass at zero, bounded-below, bounded-above, distinct ratio. Classification: `K/8`, entropy / `ln K`, imbalance ratio, `log2(min count)/10`, share of classes under the floor, sorted priors (8). Zero-padded · 16 · A, C · P1 | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L331 | Y03 — residual shape — skew, kurtosis and 5 quantiles of the standardised residual of the stored target model · 7 · A, C · P1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L332 | Y04 — heteroscedasticity — Breusch–Pagan z; slope of `ln&#124;R&#124;` on `η`; max R12 · 3 · A, C · P1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L333 | **Y05 — calibration of the stored conditional** — 2-fold cross-fitted on FIT: logistic recalibration slope and intercept; 10-bin ECE; Brier reliability; Brier resolution · 5 · A · **P0** | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L334 | **Y06 — label noise** — k-NN LOO error at k = 1 and 5 (P6); share of rows whose 5-NN majority disagrees with their label; 10% and 50% quantiles of the LOO margin · 5 · A · **P0** | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L335 | Y07 — complexity measures — Fisher ratio F1; F1v; feature efficiency F3; MST borderline share N1 (P6); intra/inter NN distance N2; linear error L2; collective efficiency F4; class-density overlap; `ln(n/p)/8` · 9 · A · P2 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L336 | D10 — spectrum — top-12 eigenvalue shares of `R_ns`; effective rank / `p`; spectral entropy; `log10(cond)/6`; eigenvalues above the Marchenko–Pastur edge / `p`; KMO; mean `&#124;ρ&#124;`; share of pairs with `&#124;ρ&#124; > 0.7`; `s`(Bartlett sphericity z) · 20 · A (B at stored rank) · P1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L337 | D11 — joint non-Gaussianity — Mardia skewness z; Mardia kurtosis z; ratio of Mahalanobis² quantiles to `χ²_p` at 0.90 and 0.99; `1/ν` of the moment-matched t-copula; energy distance between FIT and the step-0 decode in normal-score space; suggested regime count / 4 (k-means gap on the top 8 PCs); between-regime variance share · 8 · A, C · P1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L338 | L — micro-landmarks — 12 leave-one-out or out-of-fold skills as `(gain, se)`: ridge linear; + 4-bin additive; + top pairs; 64 random Fourier features; 1-, 5-, 16-NN; Gaussian naive Bayes; best single binned column; oblivious forest; linear and forest at `n/2`. Then 7 contrasts: additive − linear, pairs − additive, RFF − additive, 5-NN − linear, forest − additive, linear − linear@n/2, forest − forest@n/2 · 31 · A · P1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L339 | G — geometry — TwoNN intrinsic dimension / `p`; mean and CV of the 1-NN distance (dimension-normalised); hubness skew (k = 5); Hopkins statistic; share with Mahalanobis² above `χ²_{p,0.999}`; duplicate-row share; near-duplicate share; k-means variance explained at k = 2, 3, 4; 5% quantile of the FIT 1-NN distance (privacy radius) · 12 · A · P1 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L340 | Xg — missingness — overall rate; `log2(patterns)/10`; pattern entropy; top-pattern mass; monotone-pattern score; `s`(Little's MCAR statistic) · 6 · A, C · P2 | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L341 | **B — stability** — between two disjoint halves of FIT: cosine of the linear weights; Spearman of R14; correlation of the off-diagonal `ρ_ns`; cosine of class centroids; the two learning-curve contrasts of L · 6 · A · **P0** | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L342 | **C — closed loop** (§4.6) — C2ST AUC of the step-0 decode against FIT; self-audit retention of the stored model for the linear and the forest probe; illegal-row share of the step-0 decode; mean PIT-KS z; share of columns with KS z > 3; top-3 discriminator gain shares; energy distance in unit space · 10 · A · **P0** | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L343 | K — packet state — bytes / budget; knot level / 17; rank / 12; has-effects; has-trees; code tier / 3 · 6 · B · P0 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L344 | reserved — zeros · 10 · · | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L346 | The native 23-landmark panel (`landmarks-native-v1`) stays what it is in MLC-v3: an optional, masked 46-vector (`g/(1+&#124;g&#124;)` + validity), cached per stored table and absent for cropped training views. It joins the global … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **4.6 Closed-loop discrepancy (owner side)** |  |  |  |
| L350 | The owner can decode its own packet. After the analytic sections are fitted, P8 produces the packet's marginals in closed form and 256 counter-seeded rows, and the following are computed against FIT: | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L352 | **Per column:** PIT histogram, KS, W1 (M07). | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L353 | **Per selected pair** (top 16 by `&#124;E02&#124; + E07`): the 4 × 4 quartile contingency of FIT minus the packet-implied table; the four corner cells go to the encoder as extra pair evidence and decide which pairs earn a stored … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L354 | **Conditional:** RMS over P7 leaves of (leaf mean of `y`) − (leaf mean of the stored `η`), with per-column attribution into R14. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L355 | **Discriminator witness:** a 16-tree, depth-3, 2-fold forest separating FIT from the decoded rows: out-of-fold AUC and per-column gain share (C05). | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L356 | **Self-audit:** train the linear probe and the P7 forest on decoded rows, score on the held-out FIT fold, both folds; report the two retentions. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L358 | Cost control in training: items 1–3 are vectorised and run every episode; items 4–5 run on a fixed share of episodes (default 25%) and are masked otherwise, so the encoder learns to work without them. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
|  | **4.7 Packet-derived features (channel B, 24 per column, 16 global, 4 per pair)** |  |  |  |
| L362 | Recomputable by the receiver, so they cost no bytes and need no format change. They can be tested on the existing V7 trainer today, as the control arm that shows how far receiver-side features alone go. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L366 | PB01 — knot shape: second differences of the 17 knots in normal score, pooled to 8; tail slopes (2); knot-implied Bowley skew and tail weights (3) · 13 | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L367 | PB02 — loadings: communality; uniqueness; index of the dominant factor / 12; max and mean of `&#124;L_j · L_k&#124;` over partners · 5 | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L368 | PB03 — target: share of `Var(η)` carried by the column, `w_j (Σ w)_j / wᵀΣw` with `Σ = L Lᵀ + Ψ`; weight rank / `p`; `Q_nl` linear, quadratic and cubic contrasts; `Q_nl` energy · 6 | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L369 | PB04 — pair bias: implied `ρ_ij = L_i · L_j`; `&#124;w_i w_j&#124;` normalised; same-group; both-discrete · 4 per pair | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L370 | PB05 — global: eigenvalues of `LᵀL` (12); implied signal `wᵀΣw / (wᵀΣw + σ²)` (regression) or `d′ = √(wᵀΣw)` (classification); prior entropy; bytes / budget; share of discrete columns · 16 | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **4.8 Size and cost** |  |  |  |
| L376 | Column evidence — 128 · 3,072 | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L377 | Pair evidence — 8 · 4,608 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L378 | Group tokens — 8 · ≤ 192 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L379 | Class tokens — 12 · 48 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L380 | Global evidence — 160 (+ 46 optional landmarks) · 160 | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L382 | Measured cost of a numpy reference of P1–P5 with R01–R13 on the public panel (median `n = 1,024`, `p = 21`, one core): 8.2 ms for the vectorised blocks, against 5.6 ms for the repository's own `extract_dse_packet_numpy` … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L386 | T0 — P1, P2: S, M, D05–D07 · ≤ 1 ms · ≤ 0.3 ms | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L387 | T1 — P3–P5: D, E, R01–R12, L01–L04, L08–L09, Y, B · ≤ 3 ms · ≤ 1 ms | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L388 | T2 — P6–P8: R13–R15, L05–L07, L10–L12, G, C · ≤ 15 ms · ≤ 3 ms (items 4–5 of §4.6 on 25% of episodes) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L389 | T3 — native 23 landmarks · cached / optional · never | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L391 | The view builder's measured budget is 1.0–2.0 ms per episode (`view-builder-contract.md` §6), so T0–T1 roughly doubles data time and T2 is the first thing to subsample if data wait exceeds 10% of step time. | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L395–422 | Charge the complete packet to the 2 KiB headline. Section effectiveness still requires paired DEV tests. | Antigravity | KEPT | Run `v3-report-dev-s5-8` (`3fb802fd`, `primary.json` `55974867`): complete-file bytes charged literally (`len(file)`), `dope-v3b-auto-v8` 1,787/2,048 B (0/44 over-budget, 4× TRG −0.121 [−0.189, −0.049]), `dse8-p0-v3b` 868/2,048 B (0/44 over-budget). [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **5. Packet: `DSE8` (typed sections plus a learned code)** |  |  |  |
| L397 | New magic `DSE8`, version 1. `DSE7` v2 readers, the embedded V7.1 checkpoint and every historical decoder stay untouched. Sections are written in nested-prefix order so that truncating to a lower level is a prefix cut … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **5.1 Sections** |  |  |  |
| L403 | H — header · as `DSE7` + section bitmap, code tier, `floor(log2 n_fit)` · 24 · · P0 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L404 | G — group map · one byte per NC1 group: kind (3 bits), has_missing (1), width − 1 (4; escape byte above 16) · `groups` · S01, S02 · P0 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L405 | K — scale card · continuous and lattice columns: centre f32 + scale bf16 as today. Boolean and nominal-level columns: nothing (NC1 fixes their values … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L406 | S — support · per column 1 byte: class {continuous, lattice, atoms, zero-inflated continuous, constant} + flags; lattice columns add 1 byte `q0` … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L407 | A — atoms · per atom column: up to 32 atoms, 6-bit masses by the simplex rule of `single-encoding-v1.md` §7.4; values implied for boolean/level … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L408 | C — categorical groups · per nominal group: one int8 intercept per level, solved owner-side so emitted level masses equal FIT masses. Decode: `u_l = b_l +` … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L409 | M — knots · 5 / 9 / 17 codes for continuous and lattice columns as today, plus 1 byte of tail shape per side pair (exponential, Pareto or … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L410 | Hp — heaping · flagged multiples (1 B bitmap) + 6-bit masses; decode draws then snaps with the stored probability · `1 + ⌈6h/8⌉` per flagged column … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L411 | L — loadings · `r` int8 per latent unit: one per continuous/lattice/atom column, one per nominal level · `r · units` · D03, D04 · P0 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L412 | Rg — regimes · `m − 1` prior bytes + `m × r` int8 factor means; classes are regimes for classification, k-means regimes of the factor scores … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L413 | W — linear target · as today · `o·(8 + units)` · R10 · P0 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L414 | E — main effects (replaces `Q_nl`) · column bitmap + 4 int8 codes per output for gated columns, jointly fitted against the stored linear model, bins at the column's … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L415 | Pr — pairs · up to 15 products + a constant, gated, as stat-packet S10 · `16·(2 + o)` · R13, E07 · P1 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L416 | Tr — oblivious trees · `T` trees of depth 3: per level (column u8, knot index u8; the threshold is that column's stored knot, so it costs no extra bytes), … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L417 | Rs — residual model · regression: 5 knots of the standardised residual quantile function and up to 4 heteroscedastic slope codes. Classification: … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L418 | Ms — missingness · per column with missing cells: rate (1 B); missing-indicator columns join their value column's group · `p_miss` · X01, Xg · P1 | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L419 | Z — learned code · global `Gz` bytes then `Cz` bytes per column, finite-scalar quantised, nested by tier (§6.2) · `Gz + Cz·p` · all of §4 · P0 | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L420 | CRC —  · CRC-32 of everything before it · 4 · · | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **5.2 Byte economics** |  |  |  |
| L424 | A ≤ 12-value column costs `44 + o` bytes in `DSE7` at the full layout (6 scale card, 1 flag, 4 lo/span, 17 knots, 12 loadings, `o` weights, 4 `Q_nl`). Typed as a boolean it costs `1 + 2 + 12 + o` (support byte, two 6-bit … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L425 | Median `DSE7` size on that panel is 985 B. With the typing savings the median table has about 1.2 KiB free under the 2,048 B headline. | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L426 | A default spend at `p = 21`, binary target: effects 3 + 4·12 = 51 B; 40 trees of depth 3 = 564 B; regimes 2 × 12 + 1 = 25 B; code tier 1 = 16 + 2·21 = 58 B. Total 698 B. The measured gap these sections address (GAM and … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L427 | Budget fitting stays deterministic and never drops a column: sections are removed in a fixed order (code tier 3 → 2, pairs, trees halved repeatedly, regimes, heaping, effects, code tier 1, then today's knot/rank … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L428 | Bytes below the headline are reported, not hidden: the packet states its level, and the 0.5 / 1 / 2 / 4 KiB curve comes from one encode per table by prefix truncation. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **6.1 Tokens and attention** |  |  |  |
| L436 | **Column token:** `Linear(Ψ_col ⊕ deq(z_j))`, where `Ψ_col` is the 46 inputs of V7 with `Q_nl` replaced by the effect codes, plus the 24 packet-derived features of §4.7 and S01–S02. The projection of `deq(z_j)` is … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L437 | **Group tokens** (one per NC1 group), **class tokens** (one per class), **target token**, **global token** (`Ψ_glob ⊕ deq(z_g)`). Columns of one group attend to their group token; target heads read class and target … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L438 | **Pair bias:** attention logits get `MLP(e_ij)` per head. In the decoder `e_ij` is the 4 packet-derived pair features; in the encoder it is those plus the 8 owner-side pair features. The MLP's last layer is … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L439 | Width and depth stay in today's range (d 160, 3 layers, < 2M parameters, < 8 MB F32). V5 (the 16M model bought nothing over the 5.6M one) and V7 (d128 to d192 all at step 0) both showed that capacity was not the limit. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L440 | Rank-≤ 2 SafeTensors, split Q/K/V/O projections and `tanh`-bounded heads are kept so `engine/dope-core/src/single_encoding` loads the checkpoint with the existing parser. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **6.2 Owner-side evidence encoder and the code** |  |  |  |
| L444–446 | E_φ : (column evidence 128, pair evidence 8, group/class/target tokens, global evidence 160, Ψ(packet)) → (z_g, z_1..z_p) | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L448 | Same block type as the decoder, 2 layers, d 128. It is owner-side only; its weights are not in the receiver's closure, and its bytes are reported as owner cold bytes. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L449 | **Finite-scalar quantisation.** `z = round(L/2 · tanh(W h))` with `L = 16` levels per dimension (4 bits), straight-through gradient. Per column 4, 8 and 16 dimensions at tiers 1, 2 and 3; global 32, 64 and 128. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L453 | 0 — 0 · 0 · 0 B (pure statistic packet) | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L454 | 1 — 2 B · 16 B · 64 B | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L455 | 2 — 4 B · 32 B · 128 B | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L456 | 3 — 8 B · 64 B · 256 B | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L458 | **Nested dropout:** each training episode draws a tier (0 with probability 0.25) and zeros the dimensions above it, so one checkpoint serves every tier and a prefix of the code is a valid code. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L459 | **Evidence dropout:** each evidence block is zeroed with its mask bit cleared with probability 0.15, and tier-2 blocks with probability 0.5, so the encoder degrades to whatever the owner can afford (the … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L460 | **Embedding.** The 64/128-D embedding for JopeDime is `deq(z_g)` alone (64-D at tier 2, 128-D at tier 3), so it is a function of the packet bytes, stable under column permutation, and trained for split-half agreement … | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
|  | **6.3 Decoder heads (each a zero-initialised residual on an analytic decode)** |  |  |  |
| L466 | Marginal — typed decode: atoms, lattice snap, heaping, monotone rational-quadratic interpolation between knots (replaces piecewise-linear, so the density has no kinks at knots), tail family (S, A, M, Hp) · bounded knot-spacing change (as today) and tail-shape change | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L467 | Categorical group — intercepts + loadings, arg-max (C) · bounded intercept and loading change | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L468 | Dependence — regime mixture of rank-`r` factor copulas (L, Rg) · bounded loading change; per-regime mean change; **a row-local residual flow in latent space**: two coupling steps `u ← u +` … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L469 | Target — `η = linear + effects + pairs + trees` (W, E, Pr, Tr) · hyper-MLP as today, zero-initialised; per-class temperature | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L470 | Noise — residual quantile function, heteroscedastic slopes (Rs) · bounded scale change | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L471 | Missingness — rates and group rule (Ms) · — | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L473 | Rules carried over from the contract and the gap matrix: row `g` depends only on `(packet, weights, seed, g)`; no batch statistic; labels are always drawn, never arg-maxed; a nominal group emits exactly one level; a … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
|  | **6.4 Owner-side code refinement (after the amortised encoder works)** |  |  |  |
| L477 | The owner holds the FIT rows and the decoder is differentiable, so the code can be tuned per table: | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L479 | Split FIT into two folds. Fit the statistic sections on each complement. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L480 | Start from `E_φ`'s pre-quantisation latents `u`. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L481 | For `K ≤ 30` Adam steps, minimise the §7 loss of `decode(packet(A ∖ F_k), q(u))` scored on `F_k`, summed over the two folds. Weights are frozen. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L482 | Quantise, write the packet from all of FIT with the refined code. | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
| L484 | Cross-fitting keeps the code from memorising one half's noise, which TRG would punish and the privacy gate would flag. Report encode time for `fast` (no refinement) and `accurate` (refined) separately (gap-matrix row … | Antigravity | NOT STARTED | [Closed-loop evidence](#closed-loop-evidence) |
|  | **7. Objective** |  |  |  |
| L490 | Per episode the trainer holds FIT rows `A`, real held-out rows `Q` (today's QRY), the packet, the code and `m` decoded rows `S`. Change the episode to carry `&#124;Q&#124; ≥ 256` where the table allows and split `Q` into `Q_B` (a … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L492–495 | L | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L497 | Weights are starting values; §8.3 sets them from measured gradient norms. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **7.1 `ℓ_ret` — reference-normalised retention (replaces `tstr_loss`)** |  |  |  |
| L501 | For each differentiable surrogate learner `f` (ridge / converged IRLS on raw columns; the same on 64 random Fourier features; §7.5's tree surrogate): | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L503–509 | L_S, L_A, L_0 | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L511 | Why: this is the KPI's own ratio, so its gradient points at TRG. `L_A` is a control variate: the irreducible noise of the table is in both terms and cancels, and tables with a tiny real gain are masked by the same … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L513 | Use `m = 4·n_fit` decoded rows for this term (capped at 1,024), the KPI's multiplier, with hard sampled labels and a straight-through estimator (today's `--hard-tstr-v73` path), since the scoreboard never sees soft … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **7.2 `ℓ_cond` — proper scoring of the decoder's conditional on real rows** |  |  |  |
| L517 | Today's `cond_loss`, kept, with two changes: subtract the same quantity for a ridge/logistic model fitted on `A` and divide by the real gain (same normalisation as §7.1), and add a calibration term for classification: … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **7.3 `ℓ_pit` — closed-form marginal fit (replaces `marg_loss`)** |  |  |  |
| L521 | For continuous and lattice columns the decoder's marginal CDF `F_j` is analytic. For each real row in `Q`, `u = F_j(x)` must be uniform: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L523–525 | ℓ_pit = mean_j (1/n_Q) · [ 1/(12 n_Q) + Σ_i ( u_(i) − (2i − 1)/(2 n_Q) )² ] # Cramér–von Mises ω²; 1/(6 n_Q) in expectation when the fit is exact | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L527 | No sampling, no quantile of 192 draws, tails included. Atom and group columns use the multinomial negative log-likelihood of `Q`'s level counts under the decoded masses. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **7.4 `ℓ_cop` — closed-form dependence likelihood (replaces `corr_loss`, reduces reliance on `swd_loss`)** |  |  |  |
| L531 | With `z_j = Φ⁻¹(F_j(x_j))` on the continuous block, the regime factor copula has an exact density: per regime `log c = −½ log&#124;Σ&#124; − ½ zᵀ(Σ⁻¹ − I) z` with `Σ = L Lᵀ + Ψ` by the Woodbury identity (`O(p r²)`), combined by … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **7.5 `ℓ_tree` — random-tree leaf matching (generalises `tree_visible_joint_loss`)** |  |  |  |
| L535 | Each step draws 64 oblivious trees of depth 3. Each level picks a column (half uniformly, half in proportion to R14) and a threshold at a random stored knot. Real rows of `Q` fall into leaves exactly; decoded rows get … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L537–539 | ℓ_tree = mean over trees of Σ_leaves [ (π_S − π_Q)² + π_Q · ‖μ_S − μ_Q‖² ] | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L541 | `π` is leaf mass and `μ` the leaf mean of the target (class-probability vector or standardised value). This is a maximum mean discrepancy under a tree kernel on the joint of features and target. It matches exactly the … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L543 | The same trees give the **tree surrogate** for §7.1: fit leaf values on `S` (closed form: ridge on the soft leaf-membership matrix of all 64 trees), score on `Q_T` with hard membership. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **7.6 `ℓ_drv`, `ℓ_aux`, `ℓ_twin`, rate and guards** |  |  |  |
| L547 | **`ℓ_drv` (drivers).** For the linear surrogate: `1 − cosine` between the coefficient vectors fitted on `S` and on `A`, summed over NC1 groups. For the tree surrogate: total-variation distance between per-group gain … | Claude | NOT STARTED | [T90 and D80](#t90-and-d80) |
| L548 | **`ℓ_aux` (evidence reconstruction).** Small heads on the decoder trunk predict the P0 evidence blocks (S, M07, D11, Y05, Y06, B, C) from packet + code, Huber loss, stop-gradient on targets, masked where evidence is … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L549 | **`ℓ_twin` (split-half agreement).** Encode two disjoint halves of `A`; penalise `1 − cosine` between their global codes and the symmetrised difference of their decoded parameters (knots, loadings, effects). InfoNCE … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L550 | **Rate.** `λ_R · (code bytes + optional-section bytes)`, with learned per-section gates relaxed by straight-through; train a ladder of `λ_R` values or rely on nested dropout and report the measured 0.5 / 1 / 2 / 4 KiB … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L551 | **`ℓ_guard` (privacy).** Hinge on the share of decoded rows whose nearest FIT row is closer than their nearest `Q` row exceeding 0.55, on a 128-row subsample. As sections become more faithful (atoms, lattices, trees) … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L552 | **Anchor.** Keep a small L2 on residual magnitudes for the first 500 steps only. §7.1–7.4 are measured on real held-out rows against real-row references, so a residual is rewarded only when it improves the held-out fit, … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **8.1 Data stream** |  |  |  |
| L562 | Stop falling back to prior-only when the one-pass queue empties. Use the counter sampler's family-balanced strata with its exposure cap; real-table views ≥ 60% of every step for the whole run — §1.2: zero-repeat currently turns most of a run into SCM-prior training · `s3_v7_fast_trainer.py` → `s3_counter_sampler.py` (already built) | Claude | TESTING | [Episode stream](#episode-stream) |
| L563 | **Random-teacher targets on real features**: for a real table, draw a random function of its real columns (random oblivious forest, random 2-layer net, sparse additive spline, with random signal-to-noise and random class count) as the target — Unlimited distinct supervised tasks whose feature joint is real: lattices, groups, heaping and dependence all present. Counts as a new dataset for exposure accounting only under its own stratum · new `s3_view_sources.random_teacher_target` | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L564 | Carry NC1 group metadata in the arena index (`groups`: kind, start, width, has_missing per table) and through column crops — Typed decode and group tokens need it; today it is lost before the arena · `s3_flat_corpus.py`, `formats.md` §1 (new optional `groups` field, new arena version) | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L565 | Prior tables with truth: lattices, heaping, zero-inflation, one-hot groups, three missingness mechanisms, censoring, derived columns — The SCM prior has none of the structures §2.2 found to matter (gap-matrix row `neural:N09`) · `s3_v7_entropy.sample_synthetic_prior_table` → Rust `engine/dope-prior` | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L566 | One decode law in training and serving: integer and lattice snapping, the Rust discrete rule, group arg-max — §2.2: C2ST 0.99 in the training view against 0.87 emitted · `v8_single_encoding.py` | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L567 | `&#124;Q&#124; ≥ 256` and a twin `Q_B` where the table allows — §7.1 needs a stable denominator; 128 rows give a noisy one · sampler `qry_lo` | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **8.2 Variance at the source** |  |  |  |
| L571 | **Reference-normalised, bounded per-episode losses** (§7.1, §7.2). Largest single effect expected; every term is in comparable units before any clipping. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L572 | **Closed-form terms instead of sampled ones** (§7.3, §7.4): no Monte Carlo noise in the marginal and dependence gradients. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L573 | **Common random numbers.** One counter-seeded noise tensor per episode, shared by every loss term and by the step-0 reference decode; antithetic pairs `(ε, −ε)` for factor and column noise; stratified normals … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L574 | **More decoded rows where the KPI uses them** (`m = 4·n_fit` for §7.1). | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L575 | **Per-episode gradient clipping before reduction.** The farm already takes `autograd.grad` per episode and computes its squared norm (`farm-contract.md` §4). Clip each episode's gradient to `c × running median norm` … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L576 | **Batch from the noise scale.** With `ep_sq_norm_sum` and `grad_norm` the farm gives `B_noise ≈ (mean ‖g_i‖² − ‖ḡ‖²) / ‖ḡ‖²`. Run with a global batch of at least `B_noise/2` (expect 128–512 episodes) and scale the … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L577 | **Unit-wise adaptive clipping on the hyper heads** (`hyper_w1`, `hyper_w2`, tree and effect residual heads): clip each output unit's gradient to 2% of its weight norm. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **8.3 Loss balance and optimiser** |  |  |  |
| L581 | Log per-term gradient norms and pairwise cosines on the shared trunk every 50 steps (gap-matrix row `v1-objective:T07`). Set each weight so its trunk gradient norm is within 2× of `ℓ_ret`'s at step 500, then freeze; do … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L582 | AdamW `β₂ = 0.99` at the larger batch; warmup 1,000 steps; warmup-stable-decay as in H2; keep three EMAs (0.99, 0.999, 0.9999) and choose on DEV. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L583 | Stage the training so each stage starts at the previous stage's scoreboard value: (1) statistic sections only, decoder residuals on; (2) encoder and code on, tiers by nested dropout; (3) residual flow on. Promotion … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L584 | Keep DiLoCo for cross-host sync, but sync the per-term loss scales and the clip median with the weights. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **8.4 Instruments to add before any run** |  |  |  |
| L588 | Gradient-noise scale; share of episodes clipped; per-term norms and cosines; a frozen 256-episode probe whose loss is logged every 100 steps with fixed noise (variance target: half of today's); relaxed-vs-hard decode gap … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **9. Measurement plan and gates** |  |  |  |
| L594 | All numbers come from `dope_kpi` on the DEV panel with the contract's rule; nothing is promoted on a proxy loss. Every arm is paired on identical cells, seeds and informative flags. | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **9.1 Additions to the scoreboard** |  |  |  |
| L600 | Oracle and diagnostic control arms — `real_x_packet_y`, `packet_x_teacher_y`, `packet_x_smalltree_y`, `real_x_smalltree_y`, `exact_marginal_rank12`, `exact_marginal_fullrank` (labels always sampled; teacher = the cross-fitted boosted member already in `teacher/`) · `engine/dope-arena/src/kpi/controls.rs` | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L601 | Per-column realism ledger — one row per (table, column): support class chosen, `q0`, atoms, distinct count, 1-D boosted-tree AUC, twin AUC, bytes; aggregated by class (gap-matrix row `marginals:M18`) · new `kpi/col_ledger.rs` | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L602 | Illegal-row and group-violation counts per arm — from the NC1 group map; feeds the realism gate · `kpi/guard_eval.rs` | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L603 | Driver agreement for the tree auditor on DEV — grouped permutation-importance mass agreement against the real-vs-real ceiling (today's monitor reports a ridge-coefficient Spearman only) · `kpi/score.rs`, `dvs/drivers.rs` | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L604 | Component headroom in every receipt — the oracle arms' TRG beside the candidate, so each proposal is read against the ceiling of the component it changes · `kpi/render.rs` | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
|  | **9.2 Ordered experiments** |  |  |  |
| L610 | E0 — Control arms and ledger only · — · reproduces the ordering of §2.2 on DEV; otherwise re-order the rest · hours of CPU; decides everything below | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L611 | E1 — Typed marginals + group decode (G, K, S, A, C, M tails, Ms), no training · step-0 `DSE7` · C2ST median ≤ 0.75 (from 0.955); realism coverage + 0.25 absolute; illegal rows 0; ΔTRG lower bound > −0.01; bytes ≤ … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L612 | E2 — E1 + analytic effects, trees, residual model (E, Tr, Rs), no training · E1 · ΔTRG lower bound > 0 and point ≥ +0.03; binary lower bound > −0.02 · +0.05 to +0.08 on packet X, up to +0.2 as X improves | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L613 | E3 — E2 + regimes (Rg) · E2 · realism coverage + 0.10 or ΔTRG lower bound > 0 · C2ST pass share 0.58 → 0.84 with exact marginals; retention change of a few … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L614 | E4 — Today's V7 + packet-derived features and pair bias (channel B only), today's loss · V7.3 best · ΔTRG lower bound > 0 · ≈ 0; this is the control for "features without information" | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L615 | E5 — V7 + the §7 objective, no new information · V7.3 best · clipped-step share < 5%, probe-loss variance halved, ΔTRG ≥ 0 · training health | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L616 | E6 — `SingleEncV8` on E3 packets: encoder + code, tiers 0–3 · E3 (tier 0, untrained) · ΔTRG lower bound > 0 at equal measured bytes; ≥ 3 training seeds, same sign · the learned part's first … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L617 | E6a–h — Leave-one-family-out: {S}, {M07 + C}, {D + pairs}, {R09–R14}, {R01–R08}, {Y05 + Y06}, {B}, {L + G} · E6 · family kept if removing it costs ≥ 0.01 with the interval excluding 0; otherwise dropped from the layout · prunes the catalogue … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L618 | E7 — E6 + residual latent flow · E6 · C2ST −0.03 or ΔTRG ≥ +0.02 · the non-copula remainder: about +0.12 TRG-like and C2ST 0.63 → 0.51 available on the public panel | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L619 | E8 — Owner-side code refinement, `fast` vs `accurate` · E6/E7 · ΔTRG lower bound > 0, privacy coverage Δ > −0.03, encode time reported · — | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L620 | E9 — Byte curve at 0.5 / 1 / 2 / 4 KiB by prefix; embedding acceptance (half-vs-half cosine, family retrieval, permutation stability) · stat packet, Rust Format 2 · curve dominates at every grid point it claims · — | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L622 | Rules for every row: fixed budget per arm (plan M5.6: 2 h × 16 cores, ≥ 3 seeds, same sampler plan), KEEP / DROP / INCONCLUSIVE as in R7, failures stay in the denominator, DEV-SELECT for rung decisions and one ledgered … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **9.3 Why this order** |  |  |  |
| L626 | E1–E3 need no training and no GPU, close the largest measured realism gap, and move the step-0 baseline that every trained arm must beat. E4 and E5 separate the three things that are otherwise confounded in any new run: … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **10.1 Work packages** |  |  |  |
| L636 | W0 — Control arms, column ledger, group-violation counts, tree-driver agreement · Rust (`engine/dope-arena/src/kpi/`) · — · E0 receipt on DEV | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L637 | W1 — Arena v2 with NC1 group metadata; view builder keeps groups through crops and permutations · Python (training data path) + Rust reader · — · `test_s3_flat_corpus`, `test_s3_view_builder` extended; old arenas still read | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L638 | W2 — Evidence primitives P1–P5 and blocks S, M, D, E, R01–R12, N, T, Y, B · Rust `engine/dope-core/src/evidence/` · W1 · T1–T5, T9, T10 below | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L639 | W3 — P6–P8 and blocks R13–R15, L, G, Xg, C · Rust, same module · W2, W5 (for P8) · same tests; cost budget of §4.8 | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L640 | W4 — Torch mirror of W2–W3 for batched episodes (masks for padding, one sort per column, batched float32 solves) · Python `research/levels/v8_evidence.py` · W2 · parity ≤ 1e-5 on the fixture set at batch 1 and padded batches | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L641 | W5 — `DSE8` codec, typed marginal and group decode, effects, trees, regimes, residual model, budget planner · Rust `engine/dope-core/src/single_encoding/v8/` + numpy/torch mirror of the *decode* for training · W1 · T6–T8; E1, E2, E3 receipts | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L642 | W6 — Losses of §7 and per-episode clipping in the farm · Python `research/levels/s3_v8_losses.py`, `s3_grad_farm.py` · — · unit tests against closed forms; E5 | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L643 | W7 — `SingleEncV8` encoder, code, decoder; trainer; SafeTensors export; Rust inference · Python `v8_single_encoding.py`, `s3_v8_fast_trainer.py`; Rust `single_encoding/v8/model.rs`; `dope_v8_emit` · W4, W5, W6 · zero-init … | Codex | TESTING | [Hard forward](#hard-forward) |
| L644 | W8 — Residual flow; code refinement · Python + Rust · W7 · E7, E8 | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L645 | W9 — Random-teacher targets, typed prior tables · Python sampler now, Rust `engine/dope-prior` later · W1 · exposure ledger shows real-view share ≥ 60% for the full run | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L647 | Ownership follows the split in `docs/plans/v2-gap-closure/README.md` §5: scoreboard, evidence modules, codec and speed core on the proof lane; architecture, Rust engine integration and launches on the architecture lane. … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L649 | Language policy: evidence extraction, codec, decode and scoring are Rust. Python holds the model, the losses, the batched mirrors that sit inside the training step, and nothing else; each new Python file needs its … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **10.2 New files** |  |  |  |
| L653–670 | engine/dope-core/src/evidence/ mod.rs layout.rs primitives.rs support.rs marginal.rs dependence.rs pairs.rs relevance.rs target.rs landmarks_micro.rs geometry.rs missing.rs stability.rs closed_loop.rs … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L672 | Interfaces: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L674–696 | x_unit, x_raw, missing, y, task, classes, schema, packet, tier, seed, column, pair, group, global, layout_sha256, extract | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L698–702 | extract_batch, packet | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **10.3 Tests** |  |  |  |
| L708 | T1 — Rust vs torch evidence on 12 fixture tables (3 tasks × continuous / mixed / lattice / grouped), batch 1 and padded · max abs difference ≤ 1e-5 | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L709 | T2 — Poison: replace every QRY/T row and label · evidence and packet bytes identical | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L710 | T3 — Invariance: row permutation; column permutation with inverse; class relabel; column sign flip; monotone transform of a column · identical, permuted, permuted, signed entries flip and the rest identical, rank-based entries identical | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L711 | T4 — Null calibration: 2,000 label-shuffled tables · every `z` with a closed-form null has mean 0 ± 0.1 and SD 1 ± 0.15; permutation-based ones within ± 0.3 | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L712 | T5 — Planted mechanisms: 0.01 lattice; heaping at 5 and 10; zero inflation; U-shaped effect; one interaction; noise columns; one-hot group; MAR column · `q0` recovered; S07 > 0; S05 mass within 0.02; R11 quadratic contrast ranks first; planted pair ranks first in E07; noise columns `z` … | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L713 | T6 — Zero-init equality: fresh `SingleEncV8` against the analytic `DSE8` decode · bit-identical rows in Rust; ≤ 1e-6 in torch | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L714 | T7 — Codec: `unpack(pack(x)) == x`; byte table; prefix truncation; every single-bit flip rejected; chunk-invariant and one-row-vs-bulk decode · exact | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L715 | T8 — Floors: a level on 9 rows, a class on 9 rows, a unique extreme, a canary value · absent from packet bytes and from 100,000 decoded rows | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L716 | T9 — Determinism across 1 / 2 / 8 threads · identical evidence bytes | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L717 | T10 — Cost bench at n = 1,024, p = 24 · within the §4.8 budgets | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L718 | T11 — Loss units: `ℓ_ret` on a copy of A is 0 ± noise and on label-shuffled rows is ≈ 1; `ℓ_cop` is 0 at step 0; `ℓ_pit` on rows drawn from the decoder itself is `1/(6 n_Q)` ± noise · as stated | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **11. Risks and stop rules** |  |  |  |
| L724 | **A more faithful packet is closer to the data.** Atoms, lattices and trees raise privacy exposure. Every rung reports privacy coverage; R5's floor applies to every section; `ℓ_guard` and the contract's exact-copy guard … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L725 | **Wide tables.** At `p > 32` the statistic sections already fill the budget. Typing returns bytes, the planner drops optional sections in a fixed order, and coverage is reported by `p` bucket (≤ 24, 25–48, 49–128, > 128) … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L726 | **The encoder can learn noise at small `n`.** Evidence triples, evidence dropout, `ℓ_twin` and the stability block are the controls. Check E6 separately on the `n_A < 256` stratum. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L727 | **Public-panel numbers may not transfer.** They come from a Python harness, sklearn auditors and one-hot-encoded OpenML tables. E0 repeats them on DEV first. If E0 disagrees, its ordering wins. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L728 | **Feature count is not the goal.** E4 is expected to show nothing, and E6a–h is expected to delete families. A family that does not pay at equal bytes leaves the layout, and the layout hash changes. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L729 | **Stop rule for the learned lane.** If E6 and E7 do not beat the E3 statistic packet at equal measured bytes after one full run and one ablation round, ship E3 and stop (gap-matrix rows `neural:N06`, `neural:N19`). | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L730 | **MLC-v3.** No conditioning features are added to its router. If the 256-D conditioning vector is extended for JopeDime, that is a separate value question to be decided by a powered version of the H470 probe (gate G0 … | Antigravity | NOT STARTED | [JopeDime and embedding](#jopedime-and-embedding) |
|  | **Appendix C. What the measurements say not to build first** |  |  |  |
| L761 | **A t-copula degrees-of-freedom byte.** Moment-matched t-copula with exact marginals: no change in the TRG-like statistic or C2ST against the Gaussian copula on the public panel (§2.4). | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L762 | **More univariate association tests.** 24 of them together add about 0.06 Spearman to driver recovery over the stored weights; one cross-fitted tree witness adds 0.13 and reaches the twin ceiling (§2.3). | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L763 | **More receiver-side inputs without new packet information** (beyond the free channel-B set), and **more MLC-v3 router conditioning**. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L764 | **Rank beyond 12.** Exact marginals with rank 12 score 0.699 C2ST against 0.679 at full rank. | Owner | NOT STARTED | [Execution scope](#execution-scope) |

</details>

### Runbook

<details>
<summary>Source proposals and DEV test routing</summary>

| Source | Proposal | Owner | State | DEV test |
| --- | --- | --- | --- | --- |
|  | **0. Agent Execution Charter, Boundaries & Non-Negotiable Invariants** |  |  |  |
| L13 | Any autonomous engineering agent executing this specification **must** enforce all invariants below before committing or promoting any milestone. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **0.1 Repository & Language Boundaries** |  |  |  |
| L16 | **Production Serving Must Be Pure Rust (`engine/`):** | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L17 | Zero Python runtime dependencies in `engine/dope-norm`, `engine/dope-core`, and `engine/dope-arena`. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L18 | Zero third-party ML runtimes (no ONNX Runtime, no LibTorch, no Candle). All neural inference kernels in `engine/dope-core/src/single_encoding/` are pure-Rust `f32` matrix/vector ops. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L19 | Respect `agent/boundaries.toml`, `agent/check-python-v1-boundary.sh`, `agent/check-benchmark-boundary.sh`, and `agent/check-external-boundary.sh`. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L20 | **Python Training Boundary (`research/levels/`):** | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L21 | PyTorch/NumPy code is strictly restricted to `research/levels/` for offline meta-training, synthetic prior generation, checkpoint export, and Python-vs-Rust bit-parity conformance tests. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L22 | Every statistical extractor, wire codec, and model forward pass written in Python must have an exact Rust counterpart with golden parity $\max &#124;\Delta&#124; < 10^{-4}$ (and exact byte identity on `DSE8` wire … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L23 | **Frozen Contracts (DO NOT MUTATE):** | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L24 | Do **not** edit `contracts/dope-kpi.json` (`dope-kpi/v2`), `contracts/dope-fidelity-v3-candidate.json`, `contracts/tdl-contract-v1.json`, or frozen `nc1-minmax-onehot-v1` behavior. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L25 | All new normalization behavior is additive under `nc2-bulk-tails-v2` (NC2.1). All new wire packets use magic `DSE8` (version `3`), leaving `DSE7` intact for baseline comparison. | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **0.2 Scientific, Privacy & Runtime Invariants** |  |  |  |
| L28 | **FIT-Only Estimation:** All normalization transforms, quantiles, normal scores, statistical tests, tree probes, precision matrices, and wire packets are computed **strictly on physical `A_fit` rows**. Never read … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L29 | **Regression Target Affineness:** Regression targets remain strictly affine (no nonlinear warping of $y$) to guarantee zero Jensen bias on inverse transform. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L30 | **Privacy by Construction:** | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L31 | Zero raw rows, zero column names, zero category strings, and zero raw sample min/max values may ever enter `Nc2PublicSchema`, `StatTensorV8`, or `DSE8`. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L32–33 | All quantile endpoints, discrete atoms, boundary masses, and decision-tree split leaves must obey the support floor $k_{\min} = 10$ FIT rows: $$u_{\text{floor}} = \min\!\left(0.20,\; \frac{10}{\max(n_{\text{fit}} + 1,\; … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L34 | Empirical privacy gates remain mandatory: $\text{DCR} \le 0.55$, $\text{MIA AUC} \le 0.55$, $\text{MIA TPR@1\%FPR} \le 0.02$, $\text{exact\_copy\_excess} \le 0.005$, $\text{illegal\_rows} = 0$. (Formal DP / HIPAA … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L35 | **Zero Batch Statistics at Decode Time (Strict Row Locality):** | Codex | TESTING | [Recipient behavior](#recipient-behavior) |
| L36 | Every synthetic row $i \in \{0, 1, \dots, N-1\}$ is a pure deterministic function of `(packet_bytes, weights, seed, row_idx)`. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L37 | Never compute batch means, batch standard deviations, batch ranks, or batch whitening across generated rows during decode. Generating $N = 1$ row or $N = 10^6$ rows must yield bit-identical outputs for identical `(seed,` … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L38 | **Rank-$\le 2$ SafeTensors Export:** | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L39 | Every tensor saved to `single_enc_v8.safetensors` must have `ndim &lt;= 2` (`[D]` or `[D_out, D_in]`) and dtype `F32` (no 3D/4D tensors, no fused QKV tensors, no `BF16`/`F16` weights). | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L40 | **Step-0 Zero-Init Safety:** | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L41 | With all neural residual output heads initialized to exact `0.0`, `SingleEncV8` must reproduce the exact analytic `DSE8` step-0 packet generator. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
|  | **1. Forensic Diagnosis of V7.1 / V7.2 Plateaus** |  |  |  |
| L47 | A forensic audit of `models/single-enc-v7/V7_RELEASE_CHECKPOINT.md`, `research/levels/v7_single_encoding.py`, `research/levels/s3_v7_batch_ops.py`, and `engine/dope-arena/src/statpacket/` identifies four root causes that … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L49 | Root Cause — V7.1 / V7.2 Mechanism · Observed Failure Mode · V8 Architectural Fix | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L50 | :--- — :--- · :--- · :--- | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L51 | **1. Feature Starvation** — Only 46 floats/col (`17 knots + 12 L + 8 W + 5 flags + 4 Q_nl`) and 16 global floats. `class_means_dq` is zero-filled and never populated. Multiclass `Q_nl` collapses $K$ classes to unsigned $(1 - p_{\text{true}})$. Zero tail dependence, partial correlation, heteroscedasticity (`T03`), multimodality, or pairwise interaction statistics. · Transformer cannot infer nonlinear or joint manifold geometry from insufficient statistics. · **`StatTensorV8`**: $128$ per-column … | Antigravity | TESTING | [Repaired V7](#repaired-v7) |
| L52 | **2. Single-Regime Gaussian Copula Ceiling** — `z_lat = z_factor @ L^T + sqrt(psi) * z_noise` produces strictly unimodal, symmetric-tailed Gaussian dependence with zero coupling flow and zero class-conditional feature shifts. · Boosted-tree C2ST trivially separates synthetic vs real rows on multimodality, asymmetric tails, nonlinear $X$-couplings, and … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L53 | **3. Unanchored, Weak Target Head** — Mean-pooled `g_pool` drives a tiny `h_target = 24` SiLU MLP with no explicit tree or pairwise structure and no trust gate. · Trained target heads (`full`/`yh`) regress below step-0 retention (`xh` $\approx$ step-0) on linear/low-$n$ tables and miss sharp … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L54 | **4. Objective & Gradient Misalignment** — Single fixed-ridge `rff_dim=16` MSE surrogate (even for classification), zero tree auditor, zero driver/ALE loss, unnormalized multi-task scales, and high-variance STE (`hard_tstr_v73`). · Misaligned with `dope-kpi/v2` (`min(linear, gbdt)` TRG + held-out `mlp`) and `dope-fidelity/v3` (`driver` TV/ALE + `C2ST &lt;= 0.70`). … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **2. Normalization Upgrade (`dope-norm` NC2.1)** |  |  |  |
| L60 | **Files to modify:** | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L61 | `engine/dope-norm/src/public_schema.rs` | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L62 | `engine/dope-norm/src/nc2.rs` | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L63 | `engine/dope-norm/tests/nc2.rs` | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **2.1 Privacy-Safe Structural Side-Channel (`Nc2ColumnMetaV2`)** |  |  |  |
| L66 | Extend `Nc2PublicSchema` with an optional backward-compatible vector `col_meta_v2: Vec&lt;Nc2ColumnMetaV2>` emitted under profile `nc2-bulk-tails-v2`. Each entry contains **zero raw strings and zero raw extrema**: | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L68–90 | group_id, group_kind, group_width, parent_col_idx, skew_guard_active, support_lattice_flag, log_pre_tail_asym, log_unique_bucket | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **2.2 Unified Tie-Aware Normal Scores & Disattenuation ($\kappa_j$)** |  |  |  |
| L93 | Implement a single shared math kernel in `engine/dope-core/src/single_encoding/math.rs` and mirror it bit-for-bit in `research/levels/v8_stat_tensor.py`: | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L95 | Sort the $n_{\text{obs}}$ non-missing FIT values of column $j$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L96 | Partition into contiguous tied value blocks $b = [i_{\text{start}}, i_{\text{end}})$ of length $m_b = i_{\text{end}} - i_{\text{start}}$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L97 | Let $a_b = \max(u_{\text{eps}}, i_{\text{start}} / n_{\text{obs}})$ and $b_b = \min(1 - u_{\text{eps}}, i_{\text{end}} / n_{\text{obs}})$ with $u_{\text{eps}} = 0.5 / n_{\text{obs}}$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L98–99 | If $m_b = 1$ (untied at 0-based rank $r$): $$z_r = \Phi^{-1}\!\left(\frac{r + 0.5}{n_{\text{obs}}}\right)$$ | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L100–102 | If $m_b > 1$ (tied block spanning $(a_b, b_b)$): $$z_{\text{tie}}(a_b, b_b) = \frac{\phi(\Phi^{-1}(a_b)) - \phi(\Phi^{-1}(b_b))}{b_b - a_b}$$ where $\phi(t) = \frac{1}{\sqrt{2\pi}} \exp(-t^2 / 2)$ is the standard normal … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L103 | For missing entries in column $j$, impute normal score $0.0$ strictly for correlation estimation (missingness indicator carries the $M_j$ signal). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L104–105 | Compute exact untied reference standard deviation $\sigma_{\text{ref}} = \sqrt{\frac{1}{n_{\text{obs}}}\sum_{r=0}^{n_{\text{obs}}-1} \Phi^{-1}((r+0.5)/n_{\text{obs}})^2}$ and column attenuation factor: $$\kappa_j = … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L106–107 | Disattenuated latent correlation and class mean: $$r_{jk}^{\text{lat}} = \text{clip}\!\left(\frac{\text{corr}(z_j^{\text{tie}}, z_k^{\text{tie}})}{\kappa_j \kappa_k},\; -0.985,\; 0.985\right), \qquad … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L108 | Project the $P \times P$ matrix $\hat{R}^{\text{lat}}$ onto the positive definite correlation cone by eigendecomposition: clamp eigenvalues $\lambda_m \leftarrow \max(\lambda_m, 10^{-3})$, reconstruct, and renormalize … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **3. Complete `StatTensorV8` Feature Specification** |  |  |  |
| L114 | **Files to create:** | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L115 | Rust: `engine/dope-core/src/single_encoding/stat_tensor.rs` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L116 | Python mirror: `research/levels/v8_stat_tensor.py` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **3.1 Constants, Bounding & Fallback Contract** |  |  |  |
| L119 | `COL_FEAT_DIM = 128` | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
| L120 | `EDGE_FEAT_DIM = 16` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L121 | `GLOB_FEAT_DIM = 64` as the release ABI (superseded by frozen `contracts/evidence-layout-v1.json` GLOB=160; not run). | Antigravity | NOT STARTED | Superseded draft ABI — never run. [Evidence bundles](#evidence-bundles) |
| L122 | `MAX_COLS_HEADLINE = 48` (expandable to `96` on `16 KiB` / `64 KiB` tiers; active mask `col_mask[j] in {0.0, 1.0}`) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L123 | `MAX_CLASSES = 8` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L124 | **Smooth Squashing Helper:** $\psi(v) = \frac{v}{1 + &#124;v&#124;} \in (-1, 1)$ for unbounded positive/signed test statistics; all tensor elements must be finite (`NaN`/`Inf` replaced by `0.0`) and hard-clamped to $[-4.0, 4.0]$. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L125 | **Small-$n$ / Constant-Column Guard:** Any column with $n_{\text{obs}} < 20$ or $\text{sd}(x_j) < 10^{-12}$ sets `is_const = 1.0`, `kappa = 1.0`, and zeroes all correlation/interaction/tail statistics deterministically. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **3.2 Per-Column Feature Tensor $F_{\text{col}} \in \mathbb{R}^{P \times 128}$** |  |  |  |
| L131 | Every slot index is fixed and immutable across Rust and Python. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **Block A1: Enhanced Marginal & L-Moment Geometry (`0..26`, 26 dims)** |  |  |  |
| L134 | **`0..17` (`knots_dq[0..17]`):** Dequantized 17-knot normal-score quantiles $Q_j(\Phi(z_k))$ at uniform grid $z_k = -z_{\max} + \frac{2 z_{\max} k}{16}$, where $z_{\max} = \Phi^{-1}(1 - u_{\text{floor}})$, standardized … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L135–137 | **`17..21` (`l_moments[0..4]`):** L-scale ratio $\lambda_2 / \max(Q_{1-u_{\text{floor}}} - Q_{u_{\text{floor}}}, 10^{-6})$, L-skewness $\tau_3 = \lambda_3 / \lambda_2$, L-kurtosis $\tau_4 = \lambda_4 / \lambda_2$, and … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L138 | **`21` (`bowley_skew`):** $(Q_{0.75} + Q_{0.25} - 2Q_{0.50}) / \max(Q_{0.75} - Q_{0.25}, 10^{-6})$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L139 | **`22` (`moors_kurt`):** $\psi\!\left(\frac{(Q_{0.875} - Q_{0.625}) + (Q_{0.375} - Q_{0.125})}{\max(Q_{0.75} - Q_{0.25}, 10^{-6})} - 1.23\right)$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L140 | **`23` (`pickands_upper`):** $\psi\!\left(\frac{1}{\ln 2} \ln \frac{\max(Q_{1-u} - Q_{1-2u}, \epsilon)}{\max(Q_{1-2u} - Q_{1-4u}, \epsilon)}\right)$ with $u = \max(0.05, u_{\text{floor}})$ (if $4u \ge 0.45$, return $0$). | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L141 | **`24` (`pickands_lower`):** Mirror lower tail Pickands index using $Q_{u}, Q_{2u}, Q_{4u}$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L142 | **`25` (`kappa_attenuation`):** Tie-attenuation factor $\kappa_j \in [0.15, 1.0]$ from Section 2.2. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **Block A2: Multimodality, Point-Mass & Support Geometry (`26..40`, 14 dims)** |  |  |  |
| L145 | **`26` (`hartigan_dip`):** Hartigan's Dip Test statistic $D_n \in [0, 0.5]$ scaled by $2\sqrt{n_{\text{obs}}}$ and passed through $\psi(\cdot)$. Exact $O(n)$ algorithm on sorted unique empirical CDF points $(x_{(i)}, … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L146 | **`27` (`sarle_bimodality`):** $BC_j = \frac{\gamma_1^2 + 1}{\gamma_2 + 3(n-1)^2/((n-2)(n-3))} \in [0, 1]$ (where $\gamma_1$ is sample skew, $\gamma_2$ is excess kurtosis; clipped to $[0, 1]$). | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L147 | **`28` (`kde_mode_count`):** $\min(\text{n\_peaks}, 6) / 6.0$ from a 64-point 1D Gaussian KDE evaluated on $[Q_{u_{\text{floor}}}, Q_{1-u_{\text{floor}}}]$ with Silverman bandwidth $h = 0.9 \min(\text{sd}, … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L148 | **`29` (`kde_peak_valley_ratio`):** $1 - \frac{f_{\text{valley}}}{f_{\text{second\_peak}}}$ if $\ge 2$ peaks else $0.0$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L149 | **`30` (`kde_secondary_mass`):** Fraction of KDE mass outside the primary mode's basin of attraction ($0.0$ if unimodal). | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L150 | **`31` (`zero_point_mass`):** $P(x_j^{\text{raw\_centered}} = 0)$ (or exact atom at normalized zero image from `Nc2PublicSchema`), $0$ if $< u_{\text{floor}}$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L151 | **`32` (`lower_boundary_mass`):** $P(x_j \le Q_{u_{\text{floor}}} + 10^{-6} \cdot \text{span})$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L152 | **`33` (`upper_boundary_mass`):** $P(x_j \ge Q_{1-u_{\text{floor}}} - 10^{-6} \cdot \text{span})$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L153 | **`34` (`shannon_entropy_norm`):** $H_1 / \ln B$ over $B = \min(32, U_j)$ equal-width/unique bins. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L154 | **`35` (`renyi2_entropy_norm`):** $(-\ln \sum_{b=1}^B p_b^2) / \ln B$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L155 | **`36` (`lattice_regularity`):** Fraction of sorted adjacent unique differences $\Delta u_k$ satisfying $&#124;\Delta u_k / \text{med}(\Delta u) - \text{round}(\Delta u_k / \text{med}(\Delta u))&#124; < 0.05$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L156 | **`37` (`unique_log_ratio`):** $\ln(U_j) / \max(\ln(n_{\text{obs}}), 1.0)$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L157 | **`38` (`mode_dominance`):** $\max_v P(x_j = v) \in [0, 1]$. | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L158 | **`39` (`is_disc_or_const`):** `1.0` if discrete support ($U_j \le 32$ and all retained atoms $\ge 10$ rows) else `0.0` (encode `is_const` via `-1.0`). | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **Block A3: Missingness Mechanism Diagnostics (`40..46`, 6 dims)** |  |  |  |
| L161 | **`40` (`missing_rate`):** $m_j = \frac{1}{n_{\text{fit}}} \sum_i \mathbb{I}(x_{ij} \text{ is NaN})$. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L162 | **`41` (`mar_r2`):** If $m_j \ge u_{\text{floor}}$ and $m_j \le 1 - u_{\text{floor}}$: 2-fold ridge $R^2$ predicting binary $M_j$ from $X_{-j}$ (clipped to $[0, 1]$); else $0.0$. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L163 | **`42` (`mnar_target_lift`):** If missingness active: $&#124;\text{corr}(M_j, y)&#124;$ (reg/bin) or $\eta^2(M_j, y)$ (multiclass); else $0.0$. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L164 | **`43` (`max_missing_cooccur`):** $\max_{k \neq j} \max(0, \text{corr}(M_j, M_k))$ if active else $0.0$. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L165 | **`44` (`nc2_skew_guard_active`):** From `Nc2ColumnMetaV2.skew_guard_active` (`0.0` or `1.0`). | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L166 | **`45` (`nc2_log_pre_tail_asym`):** From `Nc2ColumnMetaV2.log_pre_tail_asym` $\in [-1, 1]$. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **Block A4: Non-Parametric & Nonlinear Target Dependence Suite (`46..62`, 16 dims)** |  |  |  |
| L169 | Let $y_{\text{scalar}}$ be standardized $y$ for regression, $y \in \{0, 1\}$ for binary, and for multiclass compute per-class statistics over one-vs-rest indicators $y_c = \mathbb{I}(y = c)$ and take the prior-weighted … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L170 | **`46..54` (`W_dq[0..8]`):** Dequantized analytic linear target weights (Ridge for reg, 6-step IRLS Logistic for bin, 6-step IRLS Softmax for multiclass; padded with `0.0` to `MAX_CLASSES = 8`). | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L171 | **`54` (`spearman_abs`):** $&#124;\rho_{\text{Spearman}}(x_j, y)&#124;$ (multiclass: $\max_c &#124;\rho(x_j, y_c)&#124;$). | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L172 | **`55` (`kendall_tau_b`):** Exact Kendall's $\tau_b(x_j, y)$ computed in $O(n \log n)$ via Fenwick tree (Knight's algorithm) with tie correction (multiclass: $\max_c &#124;\tau_b(x_j, y_c)&#124;$). | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L173–175 | **`56` (`chatterjee_xi`):** Chatterjee's rank correlation $\xi_n(x_j, y)$ computed in $O(n \log n)$: sort rows by $x_j$ (deterministic tie-break using seed-hashed index), let $r_i$ be rank of $y_{(i)}$ and $l_i$ be count … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L176 | **`57` (`dcor_unbiased`):** Exact $O(n \log n)$ Huo-Székely 1D unbiased distance correlation $\text{dCor}(x_j, y)$ (using Fenwick tree prefix sums over sorted $x_j$ and $y$); clamped to $[0, 1]$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L177 | **`58` (`rbf_hsic_norm`):** Nyström-approximated normalized RBF-HSIC using $M = \min(64, n)$ deterministic quantile landmarks on $(z_j, \tilde{y})$ with median-heuristic bandwidth ($\sigma = 1.0$ in standardized … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L178 | **`59` (`mi_norm`):** Miller-Madow bias-corrected mutual information $\widehat{I}(X_j; Y) / \max(H(Y), 10^{-4})$ using 8 equal-frequency quantile bins for $X_j$ (and 8 quantile bins for reg $Y$, exact classes for cls … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L179 | **`60` (`monotonicity_index`):** Fit PAVA (Pool Adjacent Violators Algorithm, $O(n)$) isotonic regression $x_j \to y$ (increasing and decreasing; take higher $R^2_{\text{iso}}$). Return $R^2_{\text{iso}} / … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L180 | **`61` (`anova_eta2_or_kw`):** $\eta^2 = \text{SS}_{\text{between}} / \text{SS}_{\text{total}}$ of $z_j$ grouped by exact classes (classification) or $Y$-quartiles (regression). Strictly class-label permutation … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **Block A5: Residual Curvature, Nonlinear Profile & Heteroscedasticity — Unmasking `T03` (`62..80`, 18 dims)** |  |  |  |
| L183 | Define the standardized linear target residual vector $e \in \mathbb{R}^{n_{\text{fit}}}$: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L184 | Regression: $e_i = (y_i - \hat{y}^{\text{lin}}_i) / \max(\hat{\sigma}_e, 10^{-4})$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L185 | Binary: Pearson residual $e_i = (y_i - \hat{p}^{\text{lin}}_i) / \sqrt{\max(\hat{p}^{\text{lin}}_i(1 - \hat{p}^{\text{lin}}_i), 0.02)}$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L186 | Multiclass (**fixes V7.2 sign-collapse bug!**): For each column $j$, select the **dominant contrast class** $c^*(j) = \arg\max_c \text{Var}(\text{EB-8bin}(y_{i,c} - \hat{p}^{\text{lin}}_{i,c} \mid x_{ij}))$ and define … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L188 | Slots: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L189–191 | **`62..70` (`Q_nl8[0..8]`):** 8-bin Empirical-Bayes shrunken residual means across 8 normal-score octiles of $z_j$: $$q_{j,b} = \frac{\sum_{i \in b} e^{(j)}_i}{n_b + \tau_0}, \qquad \tau_0 = 8.0$$ Passed through … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L192–194 | **`70` (`s8_chi2_gate_score`):** Exact S8 main-effect gate statistic from `statpacket/effects.rs`: $$\chi^2_{\text{S8},j} = \sum_{b=1}^{8} n_b \bar{e}_{j,b}^2, \qquad \text{slot}_{70} = … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L195 | **`71` (`ale_curvature_energy`):** Mean absolute deviation of centered cumulative local residual slope across the 8 octiles: $\frac{1}{8}\sum_{b=1}^{8} &#124;q_{j,b} - \bar{q}_j&#124;$, clipped to $[0, 2]$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L196 | **`72` (`breusch_pagan_r2`):** Unmasks `T03`! Compute $R^2$ of OLS regressing $(e_i^2 - 1)$ on $[1, z_{ij}, z_{ij}^2]$ (reg/bin; max-class for multiclass). Clipped to $[0, 1]$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L197 | **`73` (`glejser_slope`):** $\text{corr}(&#124;e_i&#124;, z_{ij}) \in [-1, 1]$ (signed heteroscedasticity fan direction). | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L198 | **`74` (`quantile_slope_spread`):** Fit 5-step smoothed IRLS quantile regression at $\tau = 0.90$ and $\tau = 0.10$ for $e$ on $[1, z_j]$; store $\psi(\hat{\beta}_{0.90} - \hat{\beta}_{0.10})$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L199 | **`75` (`cond_resid_skew_diff`):** $\psi(\text{skew}(e \mid z_j > Q_{0.75}) - \text{skew}(e \mid z_j < Q_{0.25}))$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L200 | **`76` (`cond_resid_tail_kurt`):** $\psi(\text{kurt}(e \mid &#124;z_j&#124; > 1.0) - \text{kurt}(e \mid &#124;z_j&#124; \le 1.0))$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L201 | **`77..80` (`basis_probe_excess[0..3]`):** Excess $R^2$ over linear fit on $e$ for 3 orthogonalized 1D probes: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L202 | `77`: $\text{asinh}(2 z_j) - c_1 z_j$ (saturating/log) | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L203 | `78`: $z_j^2 - 1$ (quadratic/U-shape) | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L204 | `79`: $\max(\text{corr}(e, \sin(\pi z_j / 2))^2, \text{corr}(e, \cos(\pi z_j / 2))^2)$ (periodic/bump) | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
|  | **Block A6: Tree-Geometry & Axis-Aligned Boundary Probes (`80..90`, 10 dims)** |  |  |  |
| L207 | Run a deterministic, ultra-fast FIT-only axis-aligned probe (`fit_fast_tree_probe` in $O(n P)$ over 32 quantile candidate cuts per column, strictly enforcing $\ge 10$ rows per child leaf): | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L208 | **`80` (`stump_gain_norm`):** Best 1D decision stump variance reduction $\Delta I_j^* / \text{Var}(y)$ (reg) or Gini/information gain ratio (cls). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L209 | **`81` (`stump_threshold_z`):** Normal-score location of optimal 1D split $t_j^* / z_{\max} \in [-1, 1]$. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L210 | **`82` (`stump_vs_linear_excess`):** $(\Delta I_j^* - R^2_{\text{lin},j}) / \max(\Delta I_j^*, R^2_{\text{lin},j}, 0.02) \in [-1, 1]$. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L211 | **`83` (`resid_stump_gain`):** Best 1D split variance reduction on linear residual $e^{(j)}$ along $x_j$. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L212 | **`84` (`two_split_interval_gain`):** Gain of best 2-threshold (3-leaf) partition on $e^{(j)}$ minus `resid_stump_gain` (detects interval/bump effects $a < x_j < b$). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L213 | **`85` (`local_1d_smoothness`):** Sort by $x_j$; compute $\text{corr}(y_{(i)}, \frac{1}{2}(y_{(i-1)} + y_{(i+1)}))$. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L214 | **`86..90` (`gbdt_probe_split_hist[0..4]`):** Fit a deterministic 16-tree depth-3 FIT-only GBDT probe (learning rate $0.15$, min child weight $10$, 32 quantile bins, exact $L_2$ / log-loss gradient). Record normalized … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **Block A7: Multivariate Dependence, Precision & Tail-Copula Centrality (`90..112`, 22 dims)** |  |  |  |
| L217 | Let $\Theta = (\hat{R}^{\text{lat}} + \lambda_{\text{prec}} I)^{-1}$ with adaptive ridge $\lambda_{\text{prec}} = \max(0.05, \min(0.25, P / (2 n_{\text{fit}})))$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L218 | **`90..102` (`L_dq[0..12]`):** Rank-12 principal-axis factor loadings on $\hat{R}^{\text{lat}}$ (sign-canonicalized so largest-magnitude element of each factor column is positive). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L219 | **`102` (`copula_psi`):** $\max(0.02, 1 - &#124;L_{\text{dq}}[j]&#124;_2^2)$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L220 | **`103` (`vif_log`):** $\ln(1 + \max(0, \Theta_{jj} - 1)) / 4.0$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L221 | **`104` (`partial_corr_rms`):** $\sqrt{\frac{1}{\max(P-1, 1)} \sum_{k \neq j} \rho_{jk \cdot \text{rest}}^2}$ where $\rho_{jk \cdot \text{rest}} = -\Theta_{jk} / \sqrt{\Theta_{jj}\Theta_{kk}}$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L222 | **`105` (`max_abs_corr`):** $\max_{k \neq j} &#124;r_{jk}^{\text{lat}}&#124;$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L223 | **`106` (`mean_abs_corr`):** $\frac{1}{\max(P-1, 1)} \sum_{k \neq j} &#124;r_{jk}^{\text{lat}}&#124;$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L224 | **`107` (`mean_upper_tail_dep`):** $\frac{1}{\max(P-1, 1)} \sum_{k \neq j} \hat{\lambda}_U(j, k)$ (from Edge Tensor Channel 5). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L225 | **`108` (`mean_lower_tail_dep`):** $\frac{1}{\max(P-1, 1)} \sum_{k \neq j} \hat{\lambda}_L(j, k)$ (from Edge Tensor Channel 6). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L226 | **`109` (`max_pair_target_int`):** $\max_{k \neq j} \max(&#124;E_{jk,9}&#124;, &#124;E_{jk,10}&#124;)$ (strongest product or XOR interaction with $Y$). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L227 | **`110` (`max_nongauss_copula`):** $\max_{k \neq j} &#124;E_{jk,4}&#124;$ (strongest non-Gaussian bivariate copula discrepancy). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L228 | **`111` (`regime_mean_shift_norm`):** $\sqrt{\sum_m \pi_m (\mu_{m,j}^{\text{lat}})^2}$ across classes (cls) or $Y$-quartiles (reg). | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **Block A8: Downstream Driver & Model-Family Divergence Probes (`112..122`, 10 dims)** |  |  |  |
| L231 | **`112` (`lin_driver_share`):** $&#124;W_{\text{dq}}[:, j]&#124;_2 / (\sum_k &#124;W_{\text{dq}}[:, k]&#124;_2 + 10^{-6})$. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L232 | **`113` (`gbdt_driver_share`):** Normalized total split-gain importance of column $j$ from the 16-tree depth-3 FIT GBDT probe. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L233 | **`114` (`perm_group_imp_share`):** Fast 2-fold out-of-fold grouped permutation importance share of column $j$'s NC group using the linear+GBDT probe blend. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L234 | **`115` (`tree_minus_lin_rank_div`):** $(\text{rank}_{\text{GBDT}}(j) - \text{rank}_{\text{Lin}}(j)) / \max(P, 1) \in [-1, 1]$ (positive means tree relies far more on $j$ than linear!). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L235 | **`116` (`coalition_cluster_imp`):** Sum of `gbdt_driver_share` across all columns $k$ in $j$'s collinear cluster ($&#124;r_{jk}^{\text{lat}}&#124; > 0.70$). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L236 | **`117..122` (`regime_centroid_proj[0..5]`):** Top-5 singular projections of class-conditional (or reg quartile-conditional) latent means $\mu_{m,j}^{\text{lat}}$ (sign-canonicalized). | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **Block A9: Structural Metadata & Confidence Tokens (`122..128`, 6 dims)** |  |  |  |
| L239 | **`122` (`flag_is_numeric`):** `1.0` if `group_kind == 0` else `0.0`. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L240 | **`123` (`flag_is_int_or_ord`):** `1.0` if `group_kind in {1, 2}` else `0.0`. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L241 | **`124` (`flag_is_nominal_oh`):** `1.0` if `group_kind == 4` else `0.0`. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L242 | **`125` (`norm_group_width`):** $\log_2(\max(\text{group\_width}_j, 1)) / 6.0$. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L243 | **`126` (`sample_confidence`):** $n_{\text{obs}, j} / (n_{\text{obs}, j} + 20.0) \in [0, 1]$. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L244 | **`127` (`knot_alloc_ratio`):** Allocated marginal knot budget $K_j / 17.0 \in (0, 1]$. | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **3.3 Pairwise Edge Feature Tensor $F_{\text{edge}} \in \mathbb{R}^{P \times P \times 16}$** |  |  |  |
| L250 | Symmetric on all 16 channels ($F_{\text{edge}}[i, j, :] = F_{\text{edge}}[j, i, :]$), with diagonal $F_{\text{edge}}[i, i, :] = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, \text{slot}_{111}^2, 0, 1, 1]$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L252 | Ch — Name · Formula (FIT-only, $i \neq j$) | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L253 | :--- — :--- · :--- | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L254 | `0` — `r_lat` · Disattenuated latent correlation $r_{ij}^{\text{lat}} \in [-0.985, 0.985]$ | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L255 | `1` — `partial_corr` · $-\Theta_{ij} / \sqrt{\Theta_{ii}\Theta_{jj}} \in [-1, 1]$ | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L256 | `2` — `spearman_minus_pearson` · $\rho_{\text{Spearman}}(x_i, x_j) - r_{\text{Pearson}}(x_i, x_j)$ | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L257 | `3` — `kendall_tau_ij` · Kendall's $\tau_b(x_i, x_j)$ (or exact $\frac{2}{\pi}\arcsin(r_{ij}^{\text{lat}})$ fallback if $P > 48$ and $n > 8192$ for latency … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L258 | `4` — `nongauss_dcor_excess` · $\text{dCor}(z_i, z_j) - \text{dCor}_{\text{Gauss}}(r_{ij}^{\text{lat}})$, where Gaussian analytical approximation is … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L259 | `5` — `lambda_upper` · $\hat{\lambda}_U(i, j) = \frac{1}{n_{\text{fit}} q} \sum_t \mathbb{I}(u_{ti} > 1 - q \land u_{tj} > 1 - q)$ at $q = \max(0.10, … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L260 | `6` — `lambda_lower` · $\hat{\lambda}_L(i, j) = \frac{1}{n_{\text{fit}} q} \sum_t \mathbb{I}(u_{ti} < q \land u_{tj} < q)$ at $q = \max(0.10, … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L261 | `7` — `tail_asym` · $\hat{\lambda}_U(i, j) - \hat{\lambda}_L(i, j) \in [-1, 1]$ | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L262 | `8` — `volatility_coupling` · $\text{clip}(\text{corr}(z_i^2 - 1, z_j^2 - 1) - (r_{ij}^{\text{lat}})^2, -1, 1)$ (exact $0$ for bivariate Gaussian; $> 0$ for … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L263 | `9` — `s10_prod_interaction` · $\psi\!\left(\frac{\text{corr}(z_i z_j - r_{ij}^{\text{lat}}, e)}{\sqrt{1 / \max(n_{\text{fit}}, 20)}} / 3.0\right)$ (standardized … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L264 | `10` — `xor_quadrant_interaction` · Partition rows by $(z_i > 0, z_j > 0)$ into 4 quadrants ($n_{++}, n_{+-}, n_{-+}, n_{--}$; zero if any quadrant $< 10$). Compute … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L265 | `11` — `tree_path_cooccur` · Fraction of root-to-leaf weighted samples in the 16-tree FIT GBDT probe where both $i$ and $j$ appear on the path | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L266 | `12` — `regime_mean_align` · $\psi(\sum_m \pi_m \mu_{m,i}^{\text{lat}} \mu_{m,j}^{\text{lat}})$ (between-regime alignment vs within-regime correlation) | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L267 | `13` — `missing_pmi` · If both $m_i, m_j \ge u_{\text{floor}}$: $\psi(\ln \frac{P(M_i=1, M_j=1) + 10^{-4}}{m_i m_j + 10^{-4}})$, else $0.0$ | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L268 | `14` — `same_nc_group` · `1.0` if `group_id[i] == group_id[j]` else `0.0` | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L269 | `15` — `collinear_cluster` · `1.0` if $ · r_{ij}^{\text{lat}} · > 0.70$ else `0.0` | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **3.4 Global Topology & Landmark Tensor $F_{\text{glob}} \in \mathbb{R}^{64}$** |  |  |  |
| L275 | Slots — Name · Definition | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L276 | :--- — :--- · :--- | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L277 | `0..4` — `task_onehot[4]` · `[is_reg, is_bin, is_mc, reserved_0]` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L278 | `4..12` — `class_priors[8]` · Padded class probabilities (all `0.0` for regression) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L279 | `12` — `log_n_fit` · $\log_2(\text{clip}(n_{\text{fit}}, 16, 16384)) / 14.0$ | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L280 | `13` — `p_norm` · $P / 48.0$ (clipped to $[0, 2.0]$) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L281 | `14` — `aspect_ratio` · $\text{clip}(\log_2(P / \max(n_{\text{fit}}, 1)), -12, 4) / 12.0$ | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L282 | `15` — `target_entropy_or_iqr` · $H(Y) / \max(\ln K, 1)$ (cls) or $\text{IQR}(y) / \max(\text{sd}(y), 10^{-4}) / 2.0$ (reg) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L283 | `16` — `erank_ratio` · $\exp(-\sum_m \tilde{\lambda}_m \ln \tilde{\lambda}_m) / \max(P, 1)$ where $\tilde{\lambda}_m = \lambda_m / \sum \lambda$ | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L284 | `17..21` — `eig_cum_var[4]` · Cumulative variance explained by top `1, 3, 5, 12` eigenvalues of $\hat{R}^{\text{lat}}$ | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L285 | `21` — `mardia_skew_z` · $\psi\!\left(\frac{n_{\text{fit}} b_{1,p} / 6 - \text{df}_s}{\sqrt{2 \text{df}_s}} / 5.0\right)$ where $b_{1,p} = … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L286 | `22` — `mardia_kurt_z` · $\psi\!\left(\frac{b_{2,p} - P(P+2)}{\sqrt{8 P(P+2)/n_{\text{fit}}}} / 5.0\right)$ where $b_{2,p} = \frac{1}{n} \sum_i (z_i^\top … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L287 | `23` — `hz_normality_stat` · Henze-Zirkler BHEP statistic $T_\beta$ on whitened $w_i = \Theta^{1/2} z_i$ (subsampled to $256$), transformed via $\psi(\ln(1 + … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L288 | `24` — `inv_student_t_nu` · Estimated inverse degrees of freedom $1 / \hat{\nu} \in [0, 0.45]$ from Mahalanobis kurtosis excess: $\kappa_M = … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L289 | `25` — `twonn_id_ratio` · TwoNN intrinsic dimensionality $\hat{d}_{\text{TwoNN}} / \max(P, 1)$ computed on $n_{\text{sub}} = \min(n_{\text{fit}}, 512)$ … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L290 | `26` — `global_missing_frac` · Mean missing rate across active columns | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L291 | `27` — `global_mar_share` · Fraction of columns with `mar_r2 > 0.05` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L292 | `28` — `lin_fit_quality` · Linear $R^2$ (reg) or McFadden pseudo-$R^2$ (cls) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L293 | `29` — `resid_sd_scaled` · $\text{clip}(\hat{\sigma}_e, 0.01, 2.0) / 2.0$ (reg) or mean Brier score $\times 2$ (cls) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L294 | `30` — `target_skew` · $\psi(\text{skew}(y))$ (reg) or $\max_c \pi_c - \min_c \pi_c$ (cls) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L295 | `31` — `target_kurt` · $\psi(\text{excess\_kurt}(y) / 3.0)$ (reg) or $0.0$ (cls) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L296 | `32` — `target_dip` · Hartigan dip statistic of $y$ (reg) or $0.0$ (cls) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L297 | `33` — `global_white_hetero` · Mean of top-4 column `breusch_pagan_r2` scores | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L298 | `34` — `s8_pass_frac` · Fraction of active columns passing S8 $\chi^2$ main-effect gate ($\chi^2_{\text{S8},j} > 14.067$) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L299 | `35` — `s10_pass_frac` · Fraction of column pairs passing S10 or XOR interaction gate ($ · \text{score} · > 0.55$) | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L300 | `36..51` — `landmark_gains[15]` · Distilled normalized landmark gains (all `0.0` when `landmark_mode == none`): `[lin, mono_add, sym_add, hinge_step, periodic, pairs,` … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L301 | `51..59` — `landmark_contrasts[8]` · `[tree_vs_lin, pairs_vs_lin, kernel_vs_lin, local_vs_global, lin_lc_slope, gbdt_lc_slope, driver_gini, landmark_entropy]` (all `0.0` … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L302 | `59..64` — `landmark_mask_flags[5]` · `[has_macro, has_larger, has_fine, obs_fraction, signal_above_null_flag]` | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
|  | **Conflicting wire version** |  |  |  |
| L306 | DSE8 version 3 wire format (superseded by frozen little-endian DSE8 v1 wire; not run). | Antigravity | NOT STARTED | Superseded draft wire — never run. [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **4. Binary Wire Format `DSE8` (Version 3)** |  |  |  |
| L308 | **Files to create:** | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L309 | Rust: `engine/dope-core/src/single_encoding/packet_v8.rs` | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L310 | Python mirror: `research/levels/v8_packet.py` | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **4.1 Byte Budget & Section Framing** |  |  |  |
| L313 | Every `DSE8` packet is self-contained, little-endian, CRC32-verified, and strictly obeys `packet.len() &lt;= budget_bytes` (`2048` headline, `16384` medium, `65536` large). | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L315 | Section — Fixed / Variable Layout · Typical Headline Size ($P = 20$) · Contents & Purpose | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L316 | :--- — :--- · :--- · :--- | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L317 | **H0: Header** — Fixed `32 B` · `32 B` · `magic: [u8; 4] = *b"DSE8"`, `version: u8 = 3`, `task: u8`, `n_classes: u8`, `n_cols: u8`, `copula_rank: u8`, `n_regimes:` … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L318 | **S1: Embedded ScaleCard** — `6 B/col + 4 B/group` · `136 B` · Compact `bf16` median/scale, `group_id` (`u8`), `group_kind` (`u4`) + `skew_guard` (`u1`) + `is_disc` (`u1`) + … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L319 | **S2: Adaptive Marginals** — `2 B/col` header + variable `K_j` knots (`K_j in {5, 9, 17}`) or discrete atoms (`k >= 10`) · `260 B` · Each column stores `knot_code` (`2 bits`), `pickands_upper_i4`, `pickands_lower_i4`, `boundary_mass_u8`, then … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L320 | **S3: Analytic Linear Target** — `8 B` priors/bias + `1 B/active_w` (`int8` + per-class `f16` scale) · `120 B` · Sparse bitmask over active columns + quantized $W_{\text{dq}}, b_{\text{dq}}$, class priors, and calibration temperature. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L321 | **S4: Multi-Regime Copula** — `P * r` `int8` loadings + `r * f16` col scales + **`n_regimes * r` `int8` regime means** + `P` `u8` uniqueness $\psi_j$ · `310 B` · **Fixes V7's zero `class_means_dq` bug.** Stores disattenuated loadings $L_{\text{dq}}$ and class/quartile regime shifts … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L322 | **S5: Gated Nonlinear Main Effects** — `n_s5_cols * 11 B` (`col_idx: u8`, `scale_f16: u16`, `q_nl8: [i8; 8]`, `hetero_slope: i8`) · `176 B` (for 16 gated cols) · Emitted **only** for columns passing S8 $\chi^2$ gate (ranked by $\chi^2_{\text{S8},j}$). Supplies … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L323 | **S6: Gated Pairwise Interactions** — `n_s6_pairs * 10 B` (`col_a: u8`, `col_b: u8`, `mode: u8`, `coef_y: i8`, `scale_y_f16: u16`, `coef_flow: i8`, `split_a: i8`, `split_b: i8`, `flags: u8`) · `160 B` (for 16 gated pairs) · `mode` bitmask enables multiplicative product (`z_a * z_b`), tree-XOR quadrant contrast, and/or … | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L324 | **S7: Learned Neural Conditioning Sketch** — `s7_sketch_dim` `i8` global latent `z_cond` (`64 B`) + `top_M` cols $\times$ `8 B` quantized token sketch (`i8`) + scales · `256–512 B` · Emitted by the **V8 Encoder** at `fit` time from the full `StatTensorV8` so the `source_free` decoder receives rich … | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **4.2 Dynamic Rate-Distortion Bit Allocator (`choose_dse8_layout`)** |  |  |  |
| L327 | Given `budget_bytes`: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L328 | Compute mandatory base cost `C_base = H0(32) + S1 + S2(all cols K_j=5) + S3 + S4(r=min(4, P), n_regimes)`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L329 | If `C_base > budget_bytes` (extreme high-$P$ edge case), reduce `r` down to `2` and sparsify `S3` to top-$K$ weights by $&#124;W_j&#124;$ until `C_base &lt;= budget_bytes`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L330 | Compute remaining byte budget `B_rem = budget_bytes - C_base`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L331 | Allocate `B_rem` in strict marginal-gain priority order: | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L332 | **Pass 1 (S5 + S6 Structural Signal, up to 35% of `B_rem`):** Admit columns passing S8 gate (`chi2 > 14.067`) up to `max_s5 = 16`, and pairs passing S10/XOR/copula gate (`score > 0.55`) up to `max_s6 = 16`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L333 | **Pass 2 (S7 Neural Sketch, up to 35% of `B_rem` when neural weights enabled):** Reserve `64 B` for `z_cond` + `8 B/col` for top columns ranked by `gbdt_driver_share + lin_driver_share + hartigan_dip`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L334 | **Pass 3 (Marginal Knot Upgrade `5 -> 9 -> 17`):** Upgrade columns in descending order of `driver_share + 0.5 * hartigan_dip + 0.5 * max(pickands)`. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L335 | **Pass 4 (Copula Rank Upgrade `4 -> 8 -> 12`):** Upgrade rank $r$ if remaining unexplained eigenvalue mass $> 0.08$. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L336 | **Pass 5 (Residual Sweep):** Spend remaining bytes expanding S7 per-column sketch coverage or S5/S6 counts. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **5. `SingleEncV8` Model Architecture & SafeTensors Contract** |  |  |  |
| L342 | **Files to create:** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L343 | Python: `research/levels/v8_single_encoding.py` | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L344 | Rust: `engine/dope-core/src/single_encoding/v8_weights.rs`, `engine/dope-core/src/single_encoding/v8_decode.rs`, `engine/dope-core/src/single_encoding/auto/v8.rs` | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **5.1 Two-Stage Source-Free Split (Encoder at `fit`, Decoder at `sample`)** |  |  |  |
| L347 | **Encoder (`encode_stat_tensor` — runs ONCE at `fit` time on FIT `StatTensorV8`):** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L348 | Consumes $(F_{\text{col}} \in \mathbb{R}^{P \times 128}, F_{\text{edge}} \in \mathbb{R}^{P \times P \times 16}, F_{\text{glob}} \in \mathbb{R}^{64})$. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L349 | Runs a 4-layer `PairSplitAttnBlock` Transformer ($d = 192, n_{\text{heads}} = 6, d_h = 32, d_e = 32$). | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L350 | Compresses output tokens via bottleneck projections into quantized `S7` packet sketch (`z_cond` $\in [-127, 127]^{64}$ + per-col `col_sketch` $\in [-127, 127]^{P \times 8}$), packed into `DSE8`. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L351 | **Decoder (`decode_packet_to_plan` + `sample_rows` — runs strictly from `DSE8` packet + weights, ZERO raw data):** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L352 | Reconstructs a `DecodePlanV8` from `DSE8` (`S1..S7`) via a lightweight 2-layer `PairSplitAttnBlock` Decoder Conditioning Network ($d = 192, n_{\text{heads}} = 6$) over dequantized packet features (`56 floats/col`) + … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L353 | Emits compact, row-independent runtime parameter tables (`DecodePlanV8`) so generating $N$ rows requires **zero Transformer passes per row**—only vectorized linear algebra and Soft-NODE evaluation ($> 500\text{k}$ … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
|  | **5.2 `PairSplitAttnBlock` (Strictly Rank-$\le 2$ Weights)** |  |  |  |
| L356 | For layer $\ell \in \{0, \dots, L-1\}$: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L357 | Pre-LayerNorm on column tokens: $\tilde{H} = \text{LN}(H^{(\ell)}) \in \mathbb{R}^{P \times d}$. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L358 | Separate 2D projections (no fused QKV): $Q = \tilde{H} W_Q^\top + b_Q$, $K = \tilde{H} W_K^\top + b_K$, $V = \tilde{H} W_V^\top + b_V$. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L359 | Edge bias projection: $B_{\text{edge}} = E^{(\ell)} W_{EB}^\top + b_{EB} \in \mathbb{R}^{P \times P \times n_{\text{heads}}}$ (implemented as a rank-2 matrix multiply on reshaped $[P^2, d_e] \times [d_e, … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L360–361 | Multi-head attention with additive pairwise bias and active column mask: $$A_{h, i, j} = \text{softmax}_j\!\left(\frac{Q_{h, i}^\top K_{h, j}}{\sqrt{d_h}} + B_{\text{edge}, i, j, h} + M_{\text{pad}, j}\right)$$ | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L362–363 | Output projection + SwiGLU FFN ($d_{\text{ff}} = 384$): $$H' = H^{(\ell)} + \text{Attn}(H^{(\ell)}, E^{(\ell)}) W_O^\top, \qquad H^{(\ell+1)} = H' + \big(\text{SiLU}(\text{LN}(H') W_{\text{gate}}^\top) \odot … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L364 | Lightweight Edge Update (Encoder layers only): project $U_1 = \text{LN}(H') W_{e1}^\top \in \mathbb{R}^{P \times 16}$, $U_2 = \text{LN}(H') W_{e2}^\top \in \mathbb{R}^{P \times 16}$, form symmetric outer interaction … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L365 | **4-Query Cross-Attention Pooling:** Learned queries $Q_{\text{pool}} \in \mathbb{R}^{4 \times d}$ (`[q_copula, q_flow, q_target, q_gate]`) attend over final tokens $H^{(L)}$ to yield 4 specialized global context … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **5.3 Deterministic Counter-Seeded Row-Local PRNG** |  |  |  |
| L368 | To guarantee bit-identical Python/Rust parity and zero batch dependence: | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L369 | Implement `SplitMix64 -> ChaCha8` (or deterministic 4-round Philox-4x32-10) counter PRNG keyed by `(seed: u64, row_idx: u64, stream_id: u32, col_or_dim: u32)`. | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L370 | Distinct `stream_id` allocations: `0` = regime uniform, `1` = Student-$t$ radial chi-square normals, `2` = factor normals $f_i$, `3` = idiosyncratic noise $\epsilon_i$, `4` = nominal/discrete/hurdle uniforms, `5` = … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
| L371–373 | Use exact Box-Muller transform for standard normals and Marsaglia-Tsang or sum-of-squares/Wilson-Hilferty approximation for $\chi^2_\nu$ (for smooth differentiability in Python and bit-parity in Rust, use exact … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
|  | **5.4 Decoder $X$: Multi-Regime Student-$t$ Copula + 4-Layer Row-Local RealNVP Flow** |  |  |  |
| L376 | Given `DecodePlanV8` parameters (all neural deltas initialized to `0.0`): | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L377 | **Regime Draw:** Draw regime $m_i \sim \text{Cat}(\pi)$ (for classification, $m_i \in \{0, \dots, K-1\}$ anchored at class priors; for regression, $m_i \in \{0, 1, 2, 3\}$ initialized to uniform quartiles). | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L378 | **Heavy-Tailed Radial Scale:** Let $\omega = \text{clip}(\omega_{\text{base}} + 0.25 \tanh(\Delta \omega), 0.0, 0.45)$ where $\omega = 1 / \nu_{\text{eff}}$. Compute $s_i = \big(\max(0.15, (1 - \frac{2}{9}\omega + g_i … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L379–381 | **Correlated Latent Vector:** $$z^{(0)}_i = \mu_{m_i} + s_i \Big((L_{\text{dq}} + \Delta L) f_i + \sqrt{\psi_{\text{eff}}} \odot \epsilon_i\Big)$$ where $\mu_{m_i} = \mu_{m_i}^{\text{base}} + \Delta \mu_{m_i}$ (anchored … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L382 | **4-Layer Zero-Init Row-Local RealNVP Coupling Flow:** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L383 | Deterministic alternating binary masks $M_t \in \{0, 1\}^P$ for $t \in \{0, 1, 2, 3\}$ (interleaved even/odd and block halves, permuted by column copula-discrepancy rank). | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L384–386 | For each layer $t$, `DecodePlanV8` holds hypernetwork-generated tiny 2-layer coupling weights $(A^{(t)}_1 \in \mathbb{R}^{h_c \times P}, b^{(t)}_1 \in \mathbb{R}^{h_c}, A^{(t)}_2 \in \mathbb{R}^{2P \times h_c}, b^{(t)}_2 … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L387 | Plus explicit S6 bivariate quadratic/skew warp on gated `copula_flow` pairs $(a, b)$: $z_b \leftarrow z_b + c_{\text{flow}, ab} \tanh(z_a^2 - 1)$. | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L388 | **Analytic Marginal Uniformity & Quantile-Knot Inversion (Zero Batch Stats):** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L389–390 | Normalize each coordinate analytically using closed-form mixture-moment scale $\sigma^{\text{ana}}_j = \sqrt{\sum_m \pi_m (\mu_{m,j}^2 + \frac{1}{1 - 2\omega} \sigma_{0,j}^2) - (\sum_m \pi_m \mu_{m,j})^2}$ (note: pure … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L391 | Invert via monotone piecewise-linear 17-knot table $Q_j^{\text{eff}} = Q_j^{\text{dq}} + \Delta Q_j$ (enforcing strict monotonicity via softplus delta reparameterization) with Pickands-guided smooth tail extrapolation … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L392 | **Exact Structural Support Enforcement:** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L393 | **Boundary Hurdle:** If `boundary_mass` active, snap $u_{ij} < p_{\text{low}}$ to $Q_{u_{\text{floor}}}$ and $u_{ij} > 1 - p_{\text{high}}$ to $Q_{1-u_{\text{floor}}}$. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L394 | **Nominal One-Hot Simplex Groups (`group_kind == 4`):** For each logical nominal group $G$ of width $W_g \ge 2$, take the group's $W_g$ latent scores $z^{(4)}_{i, G}$ plus calibrated bias $\log \pi_G$, draw hard one-hot … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L395 | **Discrete / Integer / Boolean Columns:** Snap to support-floor $k \ge 10$ atoms via Gumbel-Rao STE in training and exact inverse-CDF atom lookup in Rust inference. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L396 | **Missingness Indicators (`group_kind == 5`):** Decoded as boolean columns inside the copula (preserving MAR/MNAR correlations via Edge Ch 13 and Block A3!). | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
|  | **5.5 Decoder $Y$: Analytic Anchor + Soft-NODE + Heteroscedastic Skew-$t$ Head** |  |  |  |
| L399–401 | **Immutable Analytic Step-0 Predictor $\eta_{\text{base}}(x)$:** $$\eta_{\text{base}}(x) = b_{\text{dq}} + W_{\text{dq}} x + \sum_{j \in \mathcal{S}_5} g_{\text{bin},j}(x_j) + \sum_{(a,b) \in \mathcal{S}_6} \gamma_{ab} … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L402 | **Hypernetwork Soft Oblivious Decision Trees (`Soft-NODE`, $T = 8$ trees, depth $D = 3$, $2^3 = 8$ leaves/tree):** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L403 | From `DecodePlanV8`, predict per tree $t$ and depth $d \in \{0, 1, 2\}$: | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L404 | Sparse feature routing vector $w_{t,d} = \text{entmax}_{1.5}(\text{FeatLogits}_{t,d}) \in \Delta^{P-1}$; | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L405 | Split threshold $\theta_{t,d} \in [-2, 2]$ (initialized around weighted Block A6 stump thresholds $t_j^*$); | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L406 | Split sharpness $\gamma_{t,d} = \text{softplus}(\tilde{\gamma}_{t,d}) + 1.5 \in [1.5, 12.0]$. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L407–411 | Row-local evaluation (purely 24 dot products per row!): $$p_{t,d}(x_i) = \sigma\!\big(\gamma_{t,d} (w_{t,d}^\top x_i - \theta_{t,d})\big)$$ Compute 8 leaf weights $\ell_{t, \text{leaf}}(x_i) = \prod_{d=0}^{2} … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L412 | **Low-Rank Bilinear Pairwise Residual $\eta_{\text{pair}}(x_i)$:** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L413 | Rank-4 bilinear interaction $\sum_{r=1}^{4} (u_r^\top x_i)(v_r^\top x_i) c_r$ with $c_r \in \mathbb{R}^{O}$ **zero-initialized**. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L414–416 | **Learned Per-Table Trust Gate $\alpha_{\text{table}}$:** $$\alpha_{\text{table}} = \sigma\big(\text{GateHead}(q_{\text{gate}}) - 2.0\big) \in (0, 1)$$ $$\eta(x_i) = \Big(\eta_{\text{base}}(x_i) + \alpha_{\text{table}} … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L417 | **Target Sampling:** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L418 | **Binary / Multiclass:** Sample $y_i \sim \text{Cat}(\text{softmax}(\eta(x_i)))$ (forward hard draw, Gumbel-Rao backward). | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L419–421 | **Regression (Heteroscedastic Skew-$t$ Head — Unmasks `T03`):** $$\log \sigma(x_i) = \log \sigma_{\text{base}} + 0.50 \tanh\!\Big(b_\sigma + \alpha_{\text{table}} \big(w_\sigma^\top x_i + … | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
|  | **5.6 SafeTensors Export Naming Contract (Strict Rank-$\le 2$ F32)** |  |  |  |
| L424 | All weight tensors exported by `export_v8_safetensors()` must match this flat prefix scheme so `v8_weights.rs` loads them without dynamic shape hacks: | Codex | TESTING | [Hard forward](#hard-forward) |
| L425 | `enc.col_proj.weight` `[192, 128]`, `enc.col_proj.bias` `[192]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L426 | `enc.glob_proj.weight` `[192, 64]`, `enc.glob_proj.bias` `[192]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L427 | `enc.edge_proj.weight` `[32, 16]`, `enc.edge_proj.bias` `[32]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L428 | `enc.layers.{0..3}.{ln1,ln2}.{weight,bias}` `[192]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L429 | `enc.layers.{0..3}.attn.{wq,wk,wv,wo}.{weight,bias}` `[192, 192]` / `[192]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L430 | `enc.layers.{0..3}.attn.edge_bias.{weight,bias}` `[6, 32]` / `[6]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L431 | `enc.layers.{0..3}.ffn.{w_gate,w_up,w_down}.{weight,bias}` `[384, 192]` / `[192, 384]` | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L432 | `enc.layers.{0..3}.edge_upd.{w1,w2,w_out}.{weight,bias}` (all 2D) | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L433 | `enc.sketch.{z_cond_w,col_sketch_w}` (`[64, 192]`, `[8, 192]`) | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L434 | `dec.*` (identical 2-layer `PairSplitAttnBlock` naming + `dec.pool.queries` `[4, 192]` + `dec.heads.{copula,flow,node,pair,gate,hetero}.*`, every tensor strictly rank $\le 2$ F32). | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L435 | Total parameters: $\approx 2.4\text{M}$ ($\approx 9.6\text{ MB}$ uncompressed F32 SafeTensors). | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **6. Training Objective, Multi-Auditors & Gradient Smoothing** |  |  |  |
| L441 | **Files to create:** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L442 | `research/levels/s3_v8_priors.py` | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L443 | `research/levels/s3_v8_losses.py` | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L444 | `research/levels/s3_v8_trainer.py` | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.1 Differentiable Multi-Auditor TSTR Loss ($\mathcal{L}_{\text{TSTR}}$)** |  |  |  |
| L447 | For each training episode with split `(A_fit, Q_qry)` and generated synthetic batch `(X_syn, Y_syn)` of size $N_{\text{syn}} = \min(\max(n_{\text{fit}}, 128), 512)$: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L449 | **Auditor 1 — `A_lin` (Exact Ridge / 3-Step Differentiable Newton-IRLS):** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L450 | Regression: Solve $(X_{\text{syn}}^\top X_{\text{syn}} + \lambda I) \hat{\beta} = X_{\text{syn}}^\top y_{\text{syn}}$ via Cholesky with $\lambda = \max(10^{-2}, 0.05 P)$. Evaluate MSE on `Q_qry`. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L451 | Binary / Multiclass: Run 3 differentiable damped Newton-IRLS iterations minimizing regularized exact BCE/CE on `(X_syn, Y_syn_soft_gr)`. Evaluate exact BCE/CE on `Q_qry`. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L452 | **Auditor 2 — `A_tree` (Differentiable Soft-Oblivious-Forest Auditor):** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L453 | Sample $K_{\text{aud}} = 16$ episode-fixed depth-3 oblivious trees (`3` feature indices + quantile thresholds chosen from `A_fit` marginal quantiles so `X_syn` and `A_fit` share an identical tree basis!). | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L454 | Build soft leaf design matrix $\Phi_{\text{tree}}(X) \in \mathbb{R}^{N \times 128}$ using temperature $\beta = 6.0$. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L455 | Fit leaf weights $\hat{v}_{\text{syn}}$ in closed-form Ridge (reg) or 2-step IRLS (cls) on $(\Phi_{\text{tree}}(X_{\text{syn}}), Y_{\text{syn}})$, and fit reference $\hat{v}_{\text{fit}}$ identically on … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L456 | Evaluate proper loss on $(\Phi_{\text{tree}}(X_{\text{qry}}), Y_{\text{qry}})$. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L457 | **Auditor 3 — `A_mlp` (Multi-Bandwidth Nyström/RFF + 2-Step Unrolled Meta-MLP):** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L458 | Construct $D = 64$ multi-bandwidth RFF features ($\sigma \in \{0.3, 1.0, 3.0\} \cdot \sqrt{P}$), solve Ridge/2-step IRLS base, plus 2 unrolled preconditioned gradient steps on a 1-hidden-layer (`H = 32`) SiLU MLP … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L459 | **Twin-Referenced Retention Gap & Smooth-Min Reduction:** | Antigravity | TESTING | [Hard-transfer objective](#hard-transfer-objective) |
| L460 | Let $L_0(\text{qry})$ be the trivial null-predictor loss (mean for reg, class prior for cls). | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L461–462 | For each auditor $a \in \{\text{lin}, \text{tree}, \text{mlp}\}$: $$g_a = \frac{L_a(\text{syn} \to \text{qry}) - L_a(\text{fit} \to \text{qry})}{\max\big(L_0(\text{qry}) - L_a(\text{fit} \to \text{qry}),\; 0.05 \cdot … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L463–465 | Smooth-max over harm (equivalent to smooth-min over retention, directly matching `dope-kpi/v2` `cell_reduction: "minimum"`): $$\mathcal{L}_{\text{TSTR}} = \frac{1}{4.0} \log \sum_{a \in \{\text{lin}, \text{tree}, … | Antigravity | TESTING | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.2 Differentiable Driver-Importance & ALE Loss ($\mathcal{L}_{\text{driver}}$)** |  |  |  |
| L468 | Using fitted `A_lin` and `A_tree` models (`syn` vs `fit`) evaluated on held-out `Q_qry`: | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L469 | **Grouped Permutation Importance TV:** For each logical NC group $g \in \{1, \dots, G\}$, deterministically roll `Q_qry[:, cols(g)]` by $\lfloor N_{\text{qry}}/2 \rfloor$ rows and compute loss increase $I_{S,g}, I_{A,g} … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L470 | **Differentiable Soft-Rank Spearman Loss:** Compute smooth ranks $\tilde{r}_g = \sum_{h} \sigma((I_g - I_h) / 0.02)$ and add $\mathcal{L}_{\text{rank}} = 1 - \text{cosine}(\tilde{r}_S - \bar{r}_S,\; \tilde{r}_A - … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L471 | **1D ALE Curve L1 Loss:** On the top-8 columns by $I_{A,j}$, evaluate `A_tree` predictions at 8 fixed quantile grid shifts on `Q_qry` and minimize mean L1 difference between centered synthetic-trained and real-trained 1D … | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
| L472 | Combine: $\mathcal{L}_{\text{driver}} = 0.50 \mathcal{L}_{\text{perm}} + 0.25 \mathcal{L}_{\text{rank}} + 0.25 \mathcal{L}_{\text{ale}}$. | Claude | NOT STARTED | [Tree witnesses and driver fidelity](#tree-witnesses-and-driver-fidelity) |
|  | **6.3 Tree-Visible Joint $(X, Y)$ Realism Loss ($\mathcal{L}_{\text{real}}$)** |  |  |  |
| L475 | Always compare $(X_{\text{syn}}, \tilde{Y}_{\text{syn}})$ against **held-out $(X_{\text{qry}}, \tilde{Y}_{\text{qry}})$** (never `A_fit`!) to enforce generalization and protect `DCR &lt;= 0.55` and `MIA &lt;= 0.55`: | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L476 | **Joint Multi-Bandwidth RBF-MMD + Energy Distance ($\mathcal{L}_{\text{mmd}}$):** Concatenate standardized $Z_{\text{joint}} = [X, \tilde{Y}]$ ($\tilde{Y} = y$ for reg, $\sqrt{2} \cdot p_{\text{soft}}$ for cls). Compute … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L477 | **Tail-Weighted Marginal $W_1$ + Higher-Order Copula Loss ($\mathcal{L}_{\text{cop}}$):** | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L478 | 17-quantile $W_1$ with $2\times$ weights on outer tail knots $\{0, 1, 15, 16\}$; | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L479 | Pairwise correlation Frobenius $&#124;\hat{R}_{\text{syn}} - \hat{R}_{\text{qry}}&#124;_F / P$; | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L480 | **Squared-normal-score correlation loss** $&#124;\text{corr}(z_{\text{syn}}^2) - \text{corr}(z_{\text{qry}}^2)&#124;_F / P$ (enforces tail/Student-$t$ volatility coupling!). | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L481 | **2-Fold Cross-Fit Differentiable Soft-Tree C2ST Hinge ($\mathcal{L}_{\text{c2st}}$):** | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L482 | Split concatenated `[syn, qry]` into 2 balanced halves (`fold_a`, `fold_b`). | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L483–484 | Fit a 16-tree depth-3 Soft-Oblivious-Forest binary classifier (`2-step IRLS`, ridge $\lambda = 2.0$) on `fold_a` to distinguish `syn` (0) vs `qry` (1), evaluate soft pairwise Mann-Whitney AUC on `fold_b` (and vice … | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L485 | Apply hinged penalty: $\mathcal{L}_{\text{c2st}} = \text{ReLU}(\widehat{\text{AUC}}_{\text{soft}} - 0.52)^2 \cdot 10.0$. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L486 | Combine: $\mathcal{L}_{\text{real}} = 0.45 \mathcal{L}_{\text{mmd}} + 0.30 \mathcal{L}_{\text{cop}} + 0.25 \mathcal{L}_{\text{c2st}}$. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **6.4 Conditional NLL + Soft-ECE Calibration Anchor ($\mathcal{L}_{\text{cond}}$)** |  |  |  |
| L489 | Evaluated directly on held-out `(X_qry, Y_qry)`: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L490 | **Regression:** Heteroscedastic Student-$t$ / Gaussian NLL $\frac{1}{2}\big(\frac{(y - \hat{\eta})^2}{\hat{\sigma}(x)^2} + \ln \hat{\sigma}(x)^2\big)$ normalized by null variance. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L491 | **Classification:** Calibrated BCE/CE normalized by null entropy + **15-bin Triangular-Kernel Soft-ECE** (`weight = 0.25`) targeting `dope-fidelity/v3` `calib` gate. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L492–494 | **Adaptive Trust-Region Anchor ($\mathcal{L}_{\text{reg}}$):** $$w_{\text{anchor}} = \frac{10^{-2}}{1 + 4 \cdot (\text{s8\_pass\_frac} + \text{s10\_pass\_frac})}$$ Penalize $w_{\text{anchor}} \big(&#124;\Delta W&#124;_2^2 + … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
|  | **6.5 Gumbel-Rao Variance Reduction & PCGrad-Lite Conflict Projection** |  |  |  |
| L497 | **Gumbel-Rao Straight-Through ($K_{\text{GR}} = 4$):** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L498 | Replace raw STE on binary/multiclass target sampling, nominal one-hot simplex decoding, and discrete atom snapping with exact Gumbel-Rao conditional expectation: draw hard sample $D \sim \text{Cat}(p)$, sample … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L499 | Anneal $\tau$ via cosine schedule $1.0 \to 0.15$ across training steps. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L500 | **GradNorm-Lite + PCGrad-Lite (4 Semantic Gradient Groups):** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L501 | Compute backward gradients separately (or via 4-group Jacobian accumulation) for `g_tstr`, `g_driver`, `g_real`, `g_cond`. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L502 | Normalize each group by its EMA $L_2$ norm (`ema_decay = 0.98`) and multiply by target relative budgets: `w = [1.00, 0.35, 0.70, 0.50]`. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L503 | Project out negative conflicts against the primary `g_tstr` objective: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L504–505 | For each $g_i \in \{g_{\text{driver}}, g_{\text{real}}, g_{\text{cond}}\}$: if $\langle g_i, g_{\text{tstr}} \rangle < 0$, set: $$g_i \leftarrow g_i - \frac{\langle g_i, g_{\text{tstr}} \rangle}{&#124;g_{\text{tstr}}&#124;^2 + … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L506 | Sum projected gradients + $\nabla \mathcal{L}_{\text{reg}}$, clip global grad norm at `1.0`, and step optimizer. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **6.6 Non-Gaussian Synthetic Priors, Curriculum & Optimizer Recipe** |  |  |  |
| L509–510 | **Synthetic Non-Gaussian Prior Generator (`s3_v8_priors.py`):** Generate on-the-fly episodes ($n \in [64, 2048]$, $P \in [4, 48]$, balanced across `reg/bin/mc`) covering all regimes where V7 failed: | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L511 | Heavy-tailed Student-$t$ ($\nu \in [2.5, 6]$) and asymmetric Clayton/Gumbel copulas; | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L512 | Multimodal 2–4 component Gaussian mixtures and zero-inflated hurdles; | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L513 | Sharp depth-3 axis-aligned decision trees and pure 2x2 XOR interactions; | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L514 | Nonlinear additive (sinusoidal, U-shaped, piecewise-step) + heteroscedastic variance $\sigma(x) = \sigma_0 \exp(\beta^\top x)$; | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L515 | Block-MAR and target-MNAR missingness mechanisms + wide one-hot categorical groups ($W_g \in [2, 10]$). | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L516 | **3-Phase Training Curriculum (`total_steps = 12000`, effective batch `= 16` episodes):** | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L517 | **Phase 1 (Steps `0..1800`):** Linear LR warmup `0 -> 3e-4`, 40% synthetic priors + 60% S3 real tables, `A_lin + A_tree` active, `L_c2st` weight ramped `0 -> 1`. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L518 | **Phase 2 (Steps `1800..9600`):** Stable LR `3e-4`, 20% adversarial synthetic priors + 80% max-entropy S3 real tables, **30% landmark dropout** (`50% full / 25% fast / 25% none`), full multi-auditor + driver + C2ST + … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L519 | **Phase 3 (Steps `9600..12000`):** Cosine LR decay `3e-4 -> 1e-5` on 100% real S3 tables (oversampling nonlinear and multiclass families $1.5\times$), full `A_mlp` active. | Claude | NOT STARTED | [Episode stream](#episode-stream) |
| L520 | **Optimizer, EMA & Soup:** Decoupled AdamW (`betas=(0.9, 0.98)`, `weight_decay=1e-2`), EMA (`decay=0.995`), evaluate DEV every `500` steps, and build final candidate via **Top-5 DEV-TRG Valley Checkpoint Soup** (uniform … | Claude | NOT STARTED | [Episode stream](#episode-stream) |
|  | **7. Telemetry, Arena Integration & Parity Harness** |  |  |  |
| L526 | **Files to create / extend:** | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L527 | `research/levels/s3_v8_eval.py` | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L528 | `engine/dope-arena/src/bin/dope_v8_emit.rs` | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L529 | `tests/test_v8_rust_parity.py` | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **7.1 Upgraded DEV Evaluator (`s3_v8_eval.py`)** |  |  |  |
| L532 | Emit a JSON receipt + Markdown table at every DEV checkpoint containing: | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L533 | `trg_min_auditor`: Exact `dope-kpi/v2` `min(linear, gbdt)` cell-reduced TRG across DEV panel; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L534 | `trg_linear`, `trg_gbdt`, `trg_mlp`: Individual auditor TRGs + `tree_minus_lin_gap`; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L535 | `trg_by_task`: Separate TRG for `reg`, `bin`, `mc`; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L536 | `c2st_x_auc` and `c2st_joint_xy_auc`: 2-fold boosted-tree C2ST on $X$ alone and joint $(X, Y)$; | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L537 | `driver_grouped_tv`, `driver_spearman`, `ale_l1_median`: Exact `dope-fidelity/v3` driver diagnostics; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L538 | `calib_ece15_delta`, `lift_decile_l1`: Exact calibration and lift diagnostics; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L539 | `dcr_max`, `mia_auc_max`, `mia_tpr1_max`, `exact_copy_max`, `illegal_rows_total`, `privacy_headroom_min`; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L540 | `alpha_gate_linear_mean` vs `alpha_gate_nonlin_mean`: Verifies learned trust gate activates selectively on nonlinear tables; | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L541 | `packet_bytes_p50`, `packet_bytes_max`: Strict `&lt;= 2048` assertion. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
|  | **7.2 Arena Emitter (`dope_v8_emit.rs`)** |  |  |  |
| L544 | Mirror `engine/dope-arena/src/bin/dope_v7_emit.rs` CLI flags (`--train-csv`, `--out-syn-csv`, `--weights`, `--budget-bytes`, `--seed`, `--landmark-mode`, `--packet-out`, `--receipt-out`) so `dope_kpi primary`, `dope_kpi` … | Codex | NOT STARTED | [Recipient behavior](#recipient-behavior) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L546–551 | Keep the twelve mixed/task training reference cases with the approved exact discrete and 1e-5 numeric thresholds. | Codex | KEPT | Run `codex-hard-parity@8f4b4b92` (12/12 synthetic fixtures passed max abs float diff ≤ 1e-5 & 100% exact discrete label match) + `ag-pytest-parity@7968ea7f` (`test_v8_rust_parity.py` 13/13 passed ≤ 1e-4). [Hard forward](#hard-forward) |
|  | **7.3 Mandatory Python-vs-Rust Bit-Parity Tests (`tests/test_v8_rust_parity.py`)** |  |  |  |
| L547 | Must pass across 12 fixture datasets (regression, binary, multiclass, high-missingness MAR, wide one-hot nominal, zero-inflated, heavy-tailed, tiny-$n$ $n=24$, constant column, extreme $P=48$): | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L548 | `StatTensorV8` parity: looser `< 1e-4` threshold superseded by tightened `<= 1e-5` numeric contract. | Codex | REJECTED | Run `codex-hard-parity@8f4b4b92`: 12/12 mixed/task fixtures passed stricter `<= 1e-5` numeric tolerance (`test_v8_rust_parity.py@7968ea7f` also 13/13 passed). [Hard forward](#hard-forward) |
| L549 | `DSE8` wire byte-for-byte identity: `packet_py == packet_rs` and CRC32 verified, `len &lt;= 2048`. | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L550 | `SingleEncV8` decode parity: looser `< 1e-4` and `> 99.99%` label agreement superseded by tightened `<= 1e-5` & 100% exact label contract. | Codex | REJECTED | Run `codex-hard-parity@8f4b4b92`: 12/12 step-0 and random-weight SafeTensors fixtures passed `<= 1e-5` float diff and 100% exact label equality. [Hard forward](#hard-forward) |
| L551 | Strict row-locality check: row `i` generated inside $N = 1000$ equals row `i` generated in isolation ($N = 1$ at `row_idx = i`) to `0.0` diff. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L552 | Privacy & schema audit: zero raw column names/strings/min/max in packet bytes; `illegal_rows == 0`. | Codex | NOT STARTED | [Hard forward](#hard-forward) |
|  | **8. Phased Agent Execution Plan (Tickets T0–T12 & Gates M0–M5)** |  |  |  |
| L558 | Execute tickets strictly in dependency order. Do **not** proceed past a milestone gate (`M0`–`M4`) until all its automated verification commands pass and receipts are saved. | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L560 | Ticket — Dependencies · Concrete Deliverable & Target Files · Automated Verification Command / Acceptance Check | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L561 | :--- — :--- · :--- · :--- | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L562 | **T0: Baseline Snapshot** — None · Run V7.1 step-0 & best checkpoint on isolated DEV panel; save reference JSON receipt under `tmp/v8-dev/baseline_v7.json`. Verify … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L563 | **T1: NC2.1 Side-Channel** — T0 · Add `Nc2ColumnMetaV2` to `engine/dope-norm/src/public_schema.rs` & `nc2.rs` (`nc2-bulk-tails-v2`). Add unit tests proving zero raw … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L564 | **T2: Unified Tie-Aware Math** — T1 · Implement tie-aware normal scores ($z_{\text{tie}}, \kappa_j$), PD correlation projection, Fenwick Kendall $\tau_b$, $O(n\log n)$ … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L565 | **T3: `StatTensorV8` Engine** — T2 · Implement full `COL=128`, `EDGE=16`, `GLOB=64` extractor (including fast 16-tree depth-3 FIT GBDT probe, S8/S10 gates, precision … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L566 | **T4: `DSE8` Wire Codec & Analytic Step-0** — T3 · Implement `packet_v8.rs` & `v8_packet.py` (CRC32, `choose_dse8_layout`, populated `class_means`, S5 8-bin main effects, S6 … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L567 | **T5: `SingleEncV8` Model (PyTorch)** — T4 · Implement `research/levels/v8_single_encoding.py`: `PairSplitAttnBlock` Encoder (`S7` sketch quantizer w/ STE), 2-layer Decoder … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L568 | **T6: Pure-Rust `SingleEncV8` Inference** — T5 · Implement `v8_weights.rs`, `v8_decode.rs`, `auto/v8.rs`, and wire weights loading into `dope_v8_emit.rs`. Zero third-party ML deps. … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L569 | **T7: Priors, Multi-Auditors & Losses** — T5 · Implement `s3_v8_priors.py`, `s3_v8_losses.py` (`A_lin` IRLS, `A_tree` Soft-Forest, `A_mlp` Meta-MLP, smooth-min TSTR, Driver+ALE, … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L570 | **T8: Trainer & Upgraded DEV Eval** — T7 · Implement resumable `s3_v8_trainer.py` (3-phase curriculum, 30% landmark dropout, WSD LR, EMA `0.995`, top-5 soup) and … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L571 | **T9: Ablation M1 (Target + Encoder)** — T8 · Train `SingleEncV8` with Decoder $X$ copula/flow frozen at M0; optimize Encoder + Soft-NODE + trust gate + heteroscedastic head. · … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L572 | **T10: Ablation M2 (Mixture Copula + Flow)** — T9 · Unfreeze Decoder $X$ Student-$t$ mixture regimes + 4-layer RealNVP flow + Joint $(X,Y)$ MMD + Soft-Tree C2ST hinge. · **Gate M2:** … | Antigravity | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L573 | **T11: Ablation M3 + M4 (Full Multi-Seed & Soup)** — T10 · Run full 3-phase curriculum with all 3 auditors + Driver/ALE + PCGrad-Lite across $\ge 3$ independent seeds; build Top-5 DEV-TRG … | Claude | TESTING | [Power and promotion](#power-and-promotion) |
| L574 | **T12: Sealed CONFIRM Qualification (M5)** — Freeze checkpoint and run one preregistered sealed CONFIRM campaign (blocked by active DEV-only pause; never run). | Owner | NOT STARTED | Policy-blocked by DEV-only pause — never run. [Execution scope](#execution-scope) |
|  | **9. Risk Mitigations & Debug Playbook for the Agent** |  |  |  |
| L580 | **If M1 target head still underperforms M0 on tiny-$n$ or purely linear tables:** | Claude | NOT STARTED | [Training operations](#training-operations) |
| L581 | Check `alpha_gate_linear_mean` in `s3_v8_eval.py`. Increase negative init bias on `GateHead` from `-2.0` to `-3.0` and raise adaptive trust-region anchor numerator from `1e-2` to `3e-2`. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L582 | **If M2 C2ST improves on `Q_qry` during training but fails to transfer to boosted-tree C2ST on DEV:** | Claude | NOT STARTED | [Training operations](#training-operations) |
| L583 | Verify `A_tree` / C2ST discriminator splits use `A_fit` marginal quantiles (not uniform random floats) and increase C2ST soft-tree depth from `3` to `4` (or increase `K_aud` from `16` to `24`). | Claude | NOT STARTED | [Training operations](#training-operations) |
| L584 | Verify nominal one-hot simplex enforcement and boundary hurdle snapping are active in both Python and Rust. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L585 | **If `MIA AUC` or `DCR` creeps above `0.52` on DEV:** | Claude | NOT STARTED | [Training operations](#training-operations) |
| L586 | Confirm $\mathcal{L}_{\text{mmd}}$ and $\mathcal{L}_{\text{c2st}}$ are computed **strictly against held-out `Q_qry`** and never `A_fit`. | Claude | NOT STARTED | [Training operations](#training-operations) |
| L587 | Enforce minimum support floor $k_{\min} = 10$ across all discrete atoms, boundary masses, and S5/S6 bins; add `0.02` Gaussian jitter to `S7` sketch during training (information bottleneck regularization). | Claude | NOT STARTED | [Training operations](#training-operations) |
| L588 | **If Python-vs-Rust parity exceeds $10^{-4}$:** | Claude | NOT STARTED | [Training operations](#training-operations) |
| L589 | Isolate layer-by-layer (`stat_tensor` $\to$ `packet_v8` $\to$ `enc_tokens` $\to$ `dec_plan` $\to$ `z_lat` $\to$ `z_flow` $\to$ `x_syn` $\to$ `y_syn`). Common culprits: sorting tie-break order in Kendall/Chatterjee/ranks … | Claude | NOT STARTED | [Training operations](#training-operations) |

</details>

### Compiler

<details>
<summary>Source proposals and DEV test routing</summary>

| Source | Proposal | Owner | State | DEV test |
| --- | --- | --- | --- | --- |
|  | **1. Root-Cause Audit of the Current `dope` Codebase** |  |  |  |
| L13 | Direct inspection of `models/single-enc-v7/V7_RELEASE_CHECKPOINT.md`, `engine/dope-core/src/single_encoding/`, `engine/dope-arena/src/statpacket/`, `engine/dope-norm/`, and `research/levels/` reveals seven structural … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L15 | \# — Subsystem & Exact File · Current Bottleneck in Codebase · Downstream Failure Mode | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L16 | ----- — ----- · ----- · ----- | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L17 | **1** — **Post-Bottleneck Neural Placement**  `single_encoding/mod.rs`, `v7_single_encoding.py` · `SingleEncV7` runs the Transformer **only after** compressing FIT data into 46 floats/col (`knots`, rank-12 $L$, linear $W$, 5 … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L18 | **2** — **Unimodal Gaussian Copula Ceiling**  `single_encoding/mod.rs` (`decode_row`) · Latent $X$ generation is strictly $z_{\text{lat}} = z_f L^\top + z_n \sqrt{\psi}$ with zero radial mixing, zero skew warp, and … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L19 | **3** — **Ungrounded Hyper-Target MLP**  `v7_single_encoding.py` (`SingleEncV7`) · `hyper_b1` and `hyper_w2` are predicted from mean-pooled `g_pool`, while `hyper_w1` is predicted per-column with zero pairwise … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L20 | **4** — **Linear-Correlation Column Pruning**  `single_encoding/scale_card.rs` (`ScaleCardV7::fit`) · When $p_{\text{raw}} > p_{\text{eff}}$, columns are pruned strictly by univariate \$ · \\operatorname{corr}(x_j, y) | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L21 | **5** — **Miscalibrated Multiclass Base**  `v7_single_encoding.py`, `single_encoding/math.rs` · Multiclass `W_base` uses one-shot ridge onto $(Y_{\text{oh}} - p) \cdot 3 + \log(p)$ rather than regularized multinomial … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L22 | **6** — **Surrogate-to-KPI Misalignment**  `s3_v7_batch_ops.py`, `s3_v7_eval.py` · Training optimizes a 16-dim RFF ridge TSTR proxy, 24 random linear SWD projections, and `tree_visible_joint_loss` over fixed index … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
| L23 | **7** — **Unresolved Multi-Objective Gradient Conflict**  `s3_v7_fast_trainer.py` · Fixed scalar weights (`0.70*cond + 0.30*tstr + 1.0*marg + 1.2*corr + 0.9*swd + anchor`) summed into one backward pass with uniform … | Antigravity | NOT STARTED | [Repaired V7](#repaired-v7) |
|  | **Core Architectural Thesis for DOPE-V8** |  |  |  |
| L27 | **Move the Deep Model Upstream of the Wire Bottleneck (Neural Compiler Pattern):** Compute the full FIT-only statistical test suite on the normalized private table $A$ (zero wire-size restriction on encoder inputs) and … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L29 | **Upgrade Both Sides of the Generative Engine:** Replace the unimodal Gaussian copula with a **Regime-Mixture Skew-**$t$ **Copula + Top-Pair Coupling Warp**, and replace the unstructured hyper-MLP with **Calibrated … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **2. Normalize-Once Preprocessing Upgrades (`dope-norm` NC2.5 & `ScaleCardV8`)** |  |  |  |
| L33 | All feature extraction operates **strictly inside the FIT (**$A$**) owner boundary** on normalized coordinates under the `nc2-bulk-tails-v1` contract (`engine/dope-norm/src/nc2.rs`), with three mandatory upgrades: | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **2.1 Preserve Missingness as a First-Class Correlated Channel** |  |  |  |
| L37 | Currently, `ScaleCardV7::forward_x` silently imputes non-finite values with the column median, erasing Missing-Not-At-Random (MNAR) / Missing-At-Random (MAR) signal that GBDT auditors exploit natively. | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L39 | **Change:** For any column $j$ with missing fraction $m_j \in [k/n, 1 - k/n]$ (respecting `NC2_SUPPORT_K = 10`), record missingness rate $m_j$, target-missingness $\chi^2$/point-biserial association, and missingness … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
|  | **2.2 Nonlinear-Aware Column Screening (`ScaleCardV8`)** |  |  |  |
| L43 | Replace univariate $&#124;\operatorname{corr}_{\text{Pearson}}(x_j, y)&#124;$ pruning in `ScaleCardV7::fit` when $p_{\text{raw}} > p_{\text{cap}}$ with a fast $O(n \log n)$ **Composite Signal Score** $S_j$: | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L45–47 | $$ S_j = \max\!\Big(\, &#124;r_{\text{Pearson}}(x_j, y)&#124;,\; &#124;\rho_{\text{Spearman}}(x_j, y)&#124;,\; \sqrt{\max(0, \hat{\xi}_n(x_j, y))},\; \sqrt{\hat{\eta}^2_{\text{KW/ANOVA}}(x_j, y)},\; \sqrt{\hat{G}_{\text{stump}}(x_j, y)} … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L49 | where $\hat{\xi}_n$ is Chatterjee's rank correlation, $\hat{\eta}^2$ is class-conditional variance explained (exact for multiclass), and $\hat{G}_{\text{stump}}$ is the best single-split variance/Gini reduction across 8 … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **2.3 Canonical Dual-View Normalized Matrix** |  |  |  |
| L53 | Compute two aligned FIT-only matrices once in Rust SIMD ($O(n p \log n)$): | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L55 | $U \in [-6, 6]^{n \times p}$ **(`NC2` Bulk-Tails View):** Preserves affine target geometry, $\operatorname{asinh}$ skew guard, and exact tail ordering. | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
| L57 | $Z \in \mathbb{R}^{n \times p}$ **(Tie-Disattenuated Normal-Score View):** Uses `statpacket/dependence.rs`'s conditional-mean normal scores with tie-attenuation correction $\kappa_j$, ensuring all copula, … | Claude | NOT STARTED | [Extraction identity](#extraction-identity) |
|  | **Conflicting release layout** |  |  |  |
| L59 | `76/18/112` as the release ABI (superseded by frozen `contracts/evidence-layout-v1.json` 128/16/16/4/8/160; not run). | Antigravity | NOT STARTED | Superseded draft ABI — never run. [Evidence bundles](#evidence-bundles) |
|  | **3. Maximal Statistical Feature Bank for Deep Learning Conditioning** |  |  |  |
| L61 | To maximize transfer lift, driver-importance fidelity, and tree-C2ST realism, DOPE-V8 replaces `SingleEncV7`'s 46-float column vector and 16-float global vector with a **Three-Tensor Statistical Representation** computed … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L63 | **Per-Column Feature Tensor:** $C \in \mathbb{R}^{P \times 76}$ | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L65 | **Pairwise Edge Tensor:** $E \in \mathbb{R}^{P \times P \times 18}$ (fed directly as edge-biased attention into the encoder; top-$K$ edges compiled into the `DSE8` wire packet) | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L67 | **Table-Global & Landmark Tensor:** $G \in \mathbb{R}^{112}$ | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L69 | All unbounded statistics are smoothly squashed via $\phi(v) = \frac{v}{1 + &#124;v&#124;}$ or $\operatorname{asinh}(v)$ with explicit binary presence masks when $n$ or support floors prevent safe estimation. | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
|  | **3.1 Tier A: Per-Column Distributional, Structural & Target-Driver Features ($C \in \mathbb{R}^{P \times 76}$)** |  |  |  |
| L73 | Block — Dims · Exact Statistical Tests & Formulas (FIT-only, $k \ge 10$ gated) · Why It Increases Lift / C2ST / Driver Fidelity | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L74 | ----- — ----- · ----- · ----- | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L75 | **A1. Marginal Knots & Base Copula** — 29 · 17-knot normal-score quantiles ($u \in [u_{\text{floor}}, 1 - u_{\text{floor}}]$) + rank-12 principal-axis loadings $L_{j, 1..12}$ … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L76 | **A2. L-Moments & Robust Shape** — 6 · Sample **L-moments**: L-scale $\ell_2$, **L-skewness** $\tau_3 = \ell_3 / \ell_2$, **L-kurtosis** $\tau_4 = \ell_4 / \ell_2$, Bowley … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L77 | **A3. Normality, Modality & Lattice Tests** — 7 · **Anderson-Darling** $A^2$ normality statistic (tail-weighted), **Jarque-Bera** $\log(1 + \text{JB})$, **Hartigan's Dip Statistic** … | Antigravity | TESTING | [Typed rows and marginals](#typed-rows-and-marginals) |
| L78 | **A4. Support, Boundary & Missingness** — 6 · `is_const`, `is_disc`, `is_int`, boundary/zero-inflation pileup mass ($P(x_j = \text{mode})$), missingness fraction $m_j$, and … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L79 | **A5. Calibrated Linear & Class-Conditional Base** — 10 · Calibrated linear target weights $W_{j, 1..8}$ (exact Newton-IRLS / multinomial L-BFGS, not ridge hack), **Partial Correlation** … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L80 | **A6. Non-Parametric Univariate Target Tests** — 8 · **Chatterjee's** $\xi_n(X_j, Y)$, **Reverse** $\xi_n(Y, X_j)$, **Unbiased Distance Correlation** $\operatorname{dCor}(X_j, Y)$ ($O(n … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
| L81 | **A7. Conditional Mean, Heteroscedasticity & ALE Profile** — 10 · **6-Quantile EB-Shrunken Residual Means** $\mu_{j, 1..6}$ (replacing 4-bin $Q_{\text{nl}}$), **Residual Variance Slope & … | Antigravity | NOT STARTED | [Shape and nonlinear statistics](#shape-and-nonlinear-statistics) |
|  | **3.2 Tier B: Pairwise Feature-Feature & Target-Interaction Edge Tensor ($E \in \mathbb{R}^{P \times P \times 18}$)** |  |  |  |
| L85 | This is the single highest-leverage missing signal in `SingleEncV7`. Computing $E$ on $P \le 48$ active columns requires at most $\binom{48}{2} = 1{,}128$ pairs — taking $< 3\text{ ms}$ in vectorized Rust SIMD ($n \le … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L87 | Edge Feature Group — Dims · Exact Pairwise Statistical Tests on $(X_i, X_j)$ and $Y$ · Architectural Lever Controlled | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L88 | ----- — ----- · ----- · ----- | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L89 | **B1. Monotone vs. Linear Association** — 4 · Pearson $r_{ij}$, Spearman $\rho_{ij}$, **Blomqvist's Medial Correlation** $\beta_{ij}$ (robust quadrant dependence), and … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L90 | **B2. Tail Dependence & Copula Asymmetry** — 4 · **Empirical Upper Tail Dependence** $\hat{\lambda}_U(i, j) = P(U_i > 0.9 \mid U_j > 0.9)$, **Lower Tail Dependence** … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L91 | **B3. Non-Elliptical Leaf-Density Discrepancy** — 3 · $4 \times 4$ **Quantile-Grid Cramér's** $V$ (excess over Gaussian-copula expected $4 \times 4$ cell counts), **Hoeffding's** $D$ … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L92 | **B4. Conditional Independence Graph (Markov Blanket)** — 2 · **Ridge-Regularized Precision Partial Correlation** $\pi_{ij} = -\Omega_{ij} / \sqrt{\Omega_{ii}\Omega_{jj}}$ (with Ledoit-Wolf … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L93 | **B5. Pairwise Target Interaction & Synergy Tests** — 5 · 1\. **Bilinear Residual Score** $\operatorname{Corr}(r, Z_i Z_j)$ (signed) & $\chi^2$ significance. 2\. **Depth-2 XOR / Tree … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
|  | **3.3 Tier C: Table-Global Spectral, Geometric, Causal & Landmark Features ($G \in \mathbb{R}^{112}$)** |  |  |  |
| L97 | Unifies `DOPE-MLC-v3`'s proven landmarking & building-block ecosystem (`descriptors.py`, `building_block_landmarks.py`) with deep multivariate statistical tests missing across the entire repository: | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L99 | Block — Dims · Exact Global Statistical Tests · What It Controls | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L100 | ----- — ----- · ----- · ----- | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L101 | **C1. Task, Shape & Privacy-Envelope Metadata** — 16 · Task one-hot (4), class priors (8), $\log_2(n_{\text{fit}})/12$, $p_{\text{eff}}/P_{\max}$, $\log(n_{\text{fit}}/p_{\text{eff}})$, … | Antigravity | NOT STARTED | [Typed rows and marginals](#typed-rows-and-marginals) |
| L102 | **C2. Multivariate Normality & Copula Goodness-of-Fit** — 8 · **Mardia's Multivariate Skewness** $b_{1,p}$, **Mardia's Multivariate Kurtosis** $b_{2,p} - p(p + 2)$, **Henze-Zirkler Multivariate … | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L103 | **C3. Random-Matrix Spectrum & Intrinsic Geometry** — 12 · Top-6 correlation eigenvalue shares, **Marchenko-Pastur Bulk-Edge Ratio** $\lambda_1 / (1 + \sqrt{p/n})^2$ + count of signal … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L104 | **C4. Multi-Modal Regime & Class Geometry** — 12 · **GMM / K-Means Silhouette & BIC Uplift** for $M \in \{2, 3, 4\}$ regimes, **Box's M / Log-Det Covariance Divergence** across … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L105 | **C5. Residual Diagnostics & Driver Concentration** — 12 · Base linear $R^2$ / log-loss gain, residual **White's Heteroscedasticity** $R^2$, residual **Jarque-Bera** & skewness, **Effective … | Antigravity | NOT STARTED | [Evidence bundles](#evidence-bundles) |
| L106 | **C6. Unified FIT-CV Landmark & Building-Block Vector** — 52 · Compact, level-compatible projection of DOPE's **23 FIT-CV landmarks** (`lin`, `log`, `sqrt`, `pow`, `sat`, `abs`, `hinge`, `step`, … | Antigravity | TESTING | [Evidence bundles](#evidence-bundles) |
|  | **4.1 Architectural Paradigm: Pre-Bottleneck Neural Compiler + Row-Local Program Decoder** |  |  |  |
| L112 | In V7, the Transformer sat *after* a lossy $2\text{ KiB}$ bottleneck (`FIT -> analytic 2 KiB packet -> Transformer -> Gaussian + hyper-MLP`). In V8, we invert the information bottleneck: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L114–128 | v | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L130 | **Why this breaks every V7 limitation simultaneously:** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L132 | **Zero Information Starvation:** The Transformer sees all 76 column tests, all $P \times P \times 18$ pairwise interaction/tail/precision tests, and all 112 global/landmark tests. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L134 | **Ultra-Fast Zero-Dependency Sampling:** Because the neural network compiles its refinements directly into the quantized `DSE8` generative program at `encode()` time, `decode_row()` in Rust is pure arithmetic ($O(P \cdot … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **4.2 Encoder Backbone: Edge-Biased Split-QKV Transformer (`DopeV8Compiler`)** |  |  |  |
| L138 | Maintain strict rank-$\le 2$ F32 SafeTensors compatibility (matching `single_encoding/safetensors.rs`) while adding true **2D Pairwise Edge Reasoning**: | Codex | NOT STARTED | [Hard forward](#hard-forward) |
| L140 | **Inputs:** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L142 | Column tokens: $H^{(0)}_i = \operatorname{Linear}_{76 \to d}(C_i) + \operatorname{Linear}_{112 \to d}(G)$ for $i \in \{1, \dots, P\}$ ($d = 192$, $n_{\text{heads}} = 6$, $n_{\text{layers}} = 4$). | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L144 | Edge tokens: $B^{(0)}_{ij} = \operatorname{Linear}_{18 \to d_e}(E_{ij})$ ($d_e = 48$, symmetric + antisymmetric channels preserved). | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L146 | **Edge-Biased Split-QKV Attention (per layer** $\ell$**):** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L149–151 | $$ A_{ij}^{(h)} = \frac{\langle q_i^{(h)}, k_j^{(h)} \rangle}{\sqrt{d_h}} + \operatorname{Linear}_{d_e \to 1}^{(h)}\!\big(\operatorname{LN}(B_{ij}^{(\ell)})\big) + \operatorname{mask}_j $$ | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L153 | Value aggregation injects edge context via gated bias: | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L156–158 | $$ v_{ij}^{(h)} = v_j^{(h)} + \sigma(g_{ij}^{(h)}) \odot \operatorname{Linear}_{d_e \to d_h}(B_{ij}^{(\ell)}) $$ | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L160 | **Lightweight Edge Update (between layers):** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L163–165 | $$ B_{ij}^{(\ell+1)} = B_{ij}^{(\ell)} + \operatorname{FFN}_e\!\big([\operatorname{LN}(B_{ij}^{(\ell)}),\; \operatorname{Proj}(H_i^{(\ell+1)}) \odot \operatorname{Proj}(H_j^{(\ell+1)})]\big) $$ | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L168 | strictly composed of rank-$\le 2$ linear layers. | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L170 | **Learned Tree-Slot Queries (for Soft Oblivious Trees):** | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
| L172 | $T = 8$ learned query tokens $Q_{\text{tree}} \in \mathbb{R}^{T \times d}$ attend over final column embeddings $H^{(L)}$ via 2 cross-attention layers to emit the $T$ oblivious trees' split selectors, thresholds, and leaf … | Antigravity | NOT STARTED | [Owner compiler](#owner-compiler) |
|  | **4.3 Non-Gaussian Generative $X$-Decoder (Solving Boosted-Tree C2ST)** |  |  |  |
| L176 | Every synthetic row $r$ is generated **strictly row-locally** from counter-seeded RNG `(seed, row_idx)` via four composable stages: | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L178 | **Regime / Class Mixture Selection (**$M \in \{1, \dots, 4\}$ **continuous regimes, or** $M = K_{\text{eff}}$ **class-conditional regimes):** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L180 | Draw regime $m \sim \operatorname{Categorical}(\pi_1, \dots, \pi_M)$ (for classification, optionally factorize jointly or draw latent $z_f \sim \mathcal{N}(\mu_m, I_R)$ using **activated class/regime factor means** … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L182 | **Heavy-Tail Elliptical Radial Mixing (Student-**$t$ **Copula):** | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L184 | Draw row-local scale $w_r = \sqrt{\nu_m / \chi^2_{\nu_m}}$ (where $\nu_m = 2.5 + \exp(\eta_{\nu, m}) \in [2.5, 60]$ is predicted from Mardia kurtosis and $\hat{\lambda}_U, \hat{\lambda}_L$). | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L186 | Form $z^{(0)} = w_r \cdot \big(z_f L_{\text{eff}}^\top + z_n \odot \sqrt{\psi}\big) + \mu_{m, \text{col}}$. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L188 | **Result:** Instant symmetric tail dependence | Antigravity | NOT STARTED | [Dependence and regimes](#dependence-and-regimes) |
| L191–193 | $$ \lambda_U = \lambda_L = 2 t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right) > 0 $$ | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L196 | at a cost of **1 byte per regime**. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L198 | **Per-Column Sinh-Arcsinh Skew-Tail Warp (Jones-Pewsey Transform):** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L200 | Apply $z^{(1)}_j = \sinh\!\big(\delta_j \cdot \operatorname{asinh}(z^{(0)}_j) + \epsilon_j\big)$ normalized to unit scale, with $(\epsilon_j, \delta_j)$ predicted per column (2 bytes/col, zero-initialized to $\epsilon_j … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L202 | **Result:** Breaks upper/lower tail symmetry ($\lambda_U \neq \lambda_L$), capturing Clayton-like lower-tail crashes and Gumbel-like upper-tail surges. | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L204 | **Sparse Top-**$K$ **Non-Elliptical Coupling Warp (2-Step Row-Local Residual Coupling):** | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L206 | Select top $K_c \le 12$ pairs $(a_k, b_k)$ with highest Gaussian-excess $4 \times 4$ Cramér's $V$ and skew-copula curvature (`EdgeStats` B3). | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L208 | Apply an ordered (acyclic, $a_k < b_k$) quadratic/tanh coupling shift in latent space: | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L211–213 | $$ z^{(2)}_{b_k} = z^{(1)}_{b_k} + \gamma_{1,k} \tanh(z^{(1)}_{a_k}) + \gamma_{2,k} \left(\frac{(z^{(1)}_{a_k})^2 - 1}{\sqrt{2}}\right) \operatorname{sech}\!\big(0.35\, z^{(1)}_{a_k}\big) $$ | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
| L215 | Followed by **monotonic 17-knot PCHIP/linear interpolation** (with refined `d_knots`) for continuous columns, **exact discrete atom rounding** for discrete columns ($\text{count} \ge 10$), and **copula-thresholded … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
|  | **4.4 Calibrated & Tree-Expressive Target $Y$-Decoder (Eliminating Target-Head Regression)** |  |  |  |
| L219 | Replace V7's ungrounded `hyper_w1/w2` MLP with a **Decomposed Additive + Interaction + Soft-Oblivious-Tree Target Program**, strictly zero-initialized around an **upgraded analytic Step-0**: | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L221–223 | $$ \eta_c(x) = \tau \cdot \Bigg(\underbrace{b_c + w_{\text{eff}, c}^\top x}_{\text{Calibrated Linear (IRLS/L-BFGS)}} + \underbrace{\sum_{j \in \mathcal{S}_8} f_{\text{main}, j, c}(x_j)}_{\text{Gated 6-Bin Spline Main … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L225 | **Upgraded Analytic Step-0 (Before Any Neural Residual):** | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L227 | **Regression:** Cross-validated optimal ridge $\lambda^*$ (from 5-point GCV in $O(p^3)$). | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L229 | **Binary:** 8-step damped Newton-IRLS logistic regression with GCV-tuned $\lambda^*$. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L231 | **Multiclass:** **Exact Regularized Multinomial Newton-IRLS / L-BFGS** (eliminates V7's log-odds ridge bias). | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L233 | **Gated S8/S10:** Gated S8 (6-bin main effects) + Gated S10 (top-16 pairs) fitted via `statpacket/effects.rs`'s Wilson-Hilferty $\chi^2$ gate (`GATE_ALPHA = 0.05`). | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L235 | **Differentiable Soft Oblivious Decision Trees (**$T=8$ **trees, depth** $d=3 \implies 8$ **leaves/tree):** | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L237 | At depth $k \in \{1, 2, 3\}$ of tree $t$, the encoder emits a **sparse top-1/top-2 column index** $j_{t,k}$ (via Gumbel-Softmax / entmax during training, hard `u8` column index on wire), a split threshold $\theta_{t,k}$ … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L239 | Leaf routing probability is $p_{t,k}(x) = \sigma\!\big(\beta_{t,k}(x_{j_{t,k}} - \theta_{t,k})\big)$, yielding exact $2^3 = 8$ leaf weights $P_{t,\ell}(x)$ in 3 comparisons/multiplies. | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L241 | Wire cost per tree: $3 \times (\text{col } 1\text{ B} + \text{thr } 1\text{ B} + \text{sharp } 1\text{ B}) + 8 \text{ leaves} \times 1\text{ B} = 17\text{ bytes/tree}$ (only **136 bytes** for all 8 trees on … | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L243 | **Heteroscedastic Conditional Noise Head (Regression):** | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L246–248 | $$ \log \sigma(x) = \log \sigma_{\text{base}} + 0.5 \tanh\!\big(g_{\text{het}, 0} + w_{\text{het}}^\top x\big) $$ | Antigravity | NOT STARTED | [Conditional sections](#conditional-sections) |
| L250 | Zero-initialized ($w_{\text{het}} = 0$); activated only when Breusch-Pagan / White heteroscedasticity stats indicate non-constant variance, improving proper NLL and decile calibration (`ECE`). | Antigravity | TESTING | [Conditional sections](#conditional-sections) |
| L252 | **Per-Head Trust-Region Gates:** Every neural residual block (`d_knots`, `d_L`, `skew_t`, `coupling`, `d_W`, `bilinear`, `soft_trees`, `het_noise`) has a scalar learnable gate $\alpha_m = \tanh(\gamma_m)$ initialized to … | Antigravity | NOT STARTED | [Advanced decoder](#advanced-decoder) |
|  | **Verified scope of adopted suggestions** |  |  |  |
| L254–256 | Keep the strict complete 2 KiB headline cap. Automatic widening and projections remain separate proposals. | Antigravity | KEPT | Run `public-lite-replay` (`7c1f4598`, 310/310 ≤ 2,048 B) + `v3-report-dev-s5-8` (`3fb802fd`, `primary.json` `55974867`, `compare.json` `5ac5a9bd`): bytes guard PASS (max 2,048 B, share within cap 1.000, 0/44 failed). [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **5. Wire Format & Byte Budget Allocation (`DSE8` Container)** |  |  |  |
| L256 | `DSE8` strictly preserves the $\le 2048\text{ B}$ headline cap (with progressive degradation for high-$P$ tables and automatic expansion at $4\text{ KiB} / 16\text{ KiB} / 64\text{ KiB}$ frontier tiers in `dope-kpi/v2`): | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L258–277 | p_eff, knot_level, rank_use, n_regimes, has_skew_t, n_couplings, n_main_s8, n_pairs_s10, n_trees, has_het_noise | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **Representative Byte Budget Breakdown ($P = 20$ cols, Binary/Regression, $2048\text{ B}$ Headline Cap)** |  |  |  |
| L281 | Section — Contents · Bytes · Notes | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L282 | ----- — ----- · ----- · ----- | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L283 | **Header + CRC32 + ScaleCardV8** — Magic, flags, $z_{\max}$, `resid_sd`, `y_mean/sd`, active col centers/scales (`bf16`), int/disc/miss flags · `136 B` · Self-contained, zero sidecar. | Antigravity | TESTING | [Packet bytes and layout](#packet-bytes-and-layout) |
| L284 | **Marginals (S2/S5/S7)** — 17 knots/col (`lo/span f16` + 15 interior `u8`) + missingness `u8` · `400 B` · $20 \times 20\text{ B}$ | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L285 | **Regime Skew-**$t$ **Copula (S4/S6/S9)** — Rank-10 loadings (`200 B`) + 3 regimes $\mu_m, \pi_m, \nu_m$ (`36 B`) + Skew-Tail $(\epsilon_j, \delta_j)$ (`40 B`) · `276 B` · Non-Gaussian tail dependence + multimodality. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L286 | **Sparse Coupling Flow (S11)** — 10 highest-Cramér-$V$ nonlinear pairs `(a, b, g1, g2)` · `40 B` · Breaks elliptical C2ST contours. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L287 | **Target Base + S8 + S10** — Calibrated `W_base` (`28 B`) + 10 gated S8 cols (`70 B`) + 12 gated S10 pairs (`48 B`) · `146 B` · Upgraded analytic Step-0. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L288 | **Soft Oblivious Trees (S12)** — 8 depth-3 oblivious trees (split indices, thresholds, sharpnesses, leaf weights) · `136 B` · Matches GBDT decision surfaces. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L289 | **Heteroscedastic Head (S13)** — Conditional log-scale weights (`20 B`, regression) · `20 B` · Calibrated prediction intervals. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
| L290 | **Total Used** — **Well inside** $2{,}048\text{ B}$ **cap (**$894\text{ B}$ **headroom for wider** $P$ **/ multiclass)** · **`1,154 B`** · Progressive layout planner scales cleanly up to $P = 48$. | Antigravity | NOT STARTED | [Packet bytes and layout](#packet-bytes-and-layout) |
|  | **6.1 KPI-Aligned Multi-Auditor Differentiable Loss ($\mathcal{L}_{\text{V8}}$)** |  |  |  |
| L296 | Directly mirror `dope-kpi/v2` (`TRG` with $\min(\text{linear}, \text{gbdt})$ reduction, held-out MLP check, C2ST realism, privacy gates) and `dope-fidelity/v3` (`pred`, `lift`, `calib`, `driver` … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L298–300 | $$ \mathcal{L}_{\text{V8}} = w_1 \mathcal{L}_{\text{cond}} + w_2 \mathcal{L}_{\text{tstr-worst}} + w_3 \mathcal{L}_{\text{driver}} + w_4 \mathcal{L}_{\text{dist-copula}} + w_5 \mathcal{L}_{\text{tree-c2st}} + w_6 … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L302 | $\mathcal{L}_{\text{cond}}$ **(Proper-Scoring-Rule Conditional NLL on Held-Out `QRY`):** | Antigravity | TESTING | [Hard-transfer objective](#hard-transfer-objective) |
| L304 | **Regression:** Heteroscedastic Gaussian NLL: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L307–309 | $$ \frac{1}{2}\left(\frac{(y - \hat{\mu})^2}{\hat{\sigma}(x)^2} + 2\log\hat{\sigma}(x)\right) + 0.5\,\operatorname{Huber}(y - \hat{\mu}) $$ | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L311 | **Binary/Multiclass:** Proper cross-entropy + **Differentiable Soft-ECE Calibration Penalty** across 10 confidence bins. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L313 | $\mathcal{L}_{\text{tstr-worst}}$ **(Worst-Auditor Twin-Referenced TSTR Surrogate):** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L315 | Fit **three** differentiable student models on $(X_{\text{syn}}, y_{\text{syn}})$ (using straight-through Gumbel/Bernoulli hard labels for classification so students train on discrete labels just like real auditors): | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L317 | **Student 1 (Linear/Logistic):** Closed-form ridge / 2-step differentiable Newton-IRLS. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L319 | **Student 2 (Multi-Scale RFF / Kernel MLP):** 32-dim RFF across 3 lengthscales ($\sigma \in \{0.5, 1.5, 4.0\}$). | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L321 | **Student 3 (Differentiable Soft-GBDT Surrogate):** 4-stage gradient-boosted depth-2 soft oblivious trees fit via closed-form leaf Newton updates on $(X_{\text{syn}}, y_{\text{syn}})$. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L323 | Compute real-twin reference loss $L_{B, a}$ for each auditor $a \in \{1, 2, 3\}$ on split $B$, and minimize the **Smooth Worst-Auditor Twin-Referenced Gap** (mirroring `dope-kpi/v2`'s `cell_reduction: "minimum"`): | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L326–328 | $$ \mathcal{L}_{\text{tstr-worst}} = \frac{1}{\tau_a} \log \sum_{a=1}^3 \exp\!\left(\tau_a \cdot \frac{L_{S, a} - L_{B, a}}{\max(L_{0, a} - L_{A, a}, 0.02)}\right), \quad \tau_a = 4.0 $$ | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L330 | $\mathcal{L}_{\text{driver}}$ **(Explicit Driver-Importance & ALE Preservation Loss):** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L332 | For both the real-FIT teacher $f_A$ and synthetic-trained students $f_S$, compute per-column attribution vectors: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L334 | Linear magnitude $&#124;w_j&#124;$, | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L336 | Soft-Tree split gain importance $g_j$, | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L338 | **6-Bin Accumulated Local Effects (ALE) vector** $\operatorname{ALE}_{j, 1..6}$. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L340 | Minimize: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L343–345 | $$ \mathcal{L}_{\text{driver}} = \lambda_{\text{ale}} &#124;\operatorname{ALE}(f_S) - \operatorname{ALE}(f_A)&#124;_1 + \lambda_{\text{rank}} \Big(1 - \operatorname{CosSim}\big(\tilde{r}(I(f_S)),\, \tilde{r}(I(f_A))\big)\Big) $$ | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L348 | where $\tilde{r}(\cdot)$ is a differentiable soft-rank operator (temperature $\tau_r = 0.05$) weighted toward top drivers (NDCG-style $\frac{1}{\log_2(1 + \operatorname{rank}_A)}$ weighting so true drivers dominate over … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L350 | $\mathcal{L}_{\text{dist-copula}}$ **(Permutation-Equivariant Edge-Targeted Marginal + Tail + Copula Loss):** | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L352 | Marginal 17-quantile $W_1$ + **Tail-Weighted Anderson-Darling** $W_1$ ($2\times$ weight on $q \in \{0.02, 0.05, 0.95, 0.98\}$). | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L354 | **Edge-Selected** $4 \times 4$ **Soft Copula Contingency** $L_1$**:** Replace `tree_visible_joint_loss`'s fixed `torch.roll(1,2,3,5)` with the **Top-24 highest-dependence & highest-interaction pairs** from `EdgeStats` … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L356 | **Multi-Scale Sliced Wasserstein** (48 random projections + top-8 principal directions of $X_{\text{qry}}$). | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L358 | $\mathcal{L}_{\text{tree-c2st}}$ **(Differentiable Soft-Tree C2ST Critic — Directly Attacks Boosted-Tree Realism):** | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L360 | Maintain a tiny per-episode **Differentiable Soft-Oblivious-Tree Discriminator** $D_\psi(x, y)$ (6 depth-3 trees, fit in 3 inner gradient-ascent steps with spectral-norm/$L_2$ penalty on $[X_{\text{qry}}, … | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L362 | Backpropagate the generator loss $\max(0, \operatorname{AUC}_{\text{soft}}(D_\psi) - 0.52)^2$ through the unrolled critic into the Skew-$t$ copula, regime mixture, and coupling warp. | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L364 | $\mathcal{L}_{\text{priv}}$ **(Differentiable Empirical Privacy Barrier):** | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L366 | Compute soft-minimum $L_2$ distances from $X_{\text{syn}}$ to FIT $X_A$ vs. holdout $X_{\text{qry}}$: | Antigravity | TESTING | [Hard-transfer objective](#hard-transfer-objective) |
| L369–371 | $$ \operatorname{DCR}_{\text{soft}} = \frac{1}{N_{\text{syn}}} \sum_{i} \sigma\!\left(\frac{d_{\min}(s_i, X_{\text{qry}}) - d_{\min}(s_i, X_A)}{0.15}\right) $$ | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
| L373 | Apply smooth barrier: | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L376–378 | $$ \mathcal{L}_{\text{priv}} = 2.0 \cdot \operatorname{ReLU}(\operatorname{DCR}_{\text{soft}} - 0.51)^2 + 5.0 \cdot \operatorname{ReLU}\!\big(0.02 - d_{\min}(s_i, X_A)\big)^2 $$ | Antigravity | NOT STARTED | [Hard-transfer objective](#hard-transfer-objective) |
| L381 | keeping $\text{DCR} \le 0.55$, $\text{MIA} \le 0.55$, and $\text{exact\_copy\_excess} \le 0.005$ by construction. | Claude | NOT STARTED | [Realism and privacy](#realism-and-privacy) |
|  | **6.2 Gradient Smoothing, Conflict Resolution & Optimization Schedule** |  |  |  |
| L385 | To completely eliminate the noisy DEV trajectories and $X$-vs-$Y$ destructive interference seen in V7.1/V7.2: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L387 | **CAGrad / PCGrad Across Loss Groups (Zero Destructive Interference):** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L389 | Partition objectives into 4 semantic groups: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L392–394 | $$ g_{\text{utility}} = \nabla(\mathcal{L}_{\text{cond}} + \mathcal{L}_{\text{tstr-worst}}), \quad g_{\text{driver}} = \nabla \mathcal{L}_{\text{driver}}, \quad g_{\text{realism}} = … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L396 | Whenever $\langle g_u, g_v \rangle < 0$ on shared backbone parameters, project out the conflicting component: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L399–401 | $$ g_u \leftarrow g_u - \frac{\langle g_u, g_v \rangle}{&#124;g_v&#124;^2 + \epsilon} g_v $$ | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L404 | (PCGrad) or apply CAGrad ($c = 0.4$) to maximize minimum worst-group improvement. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L406 | **Task-Normalized Gradient Balancing (GradNorm-Lite):** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L408 | Normalize per-task gradient norms across $\{\text{regression}, \text{binary}, \text{multiclass}\}$ using exponential moving averages of task loss variance so high-variance regression episodes never drown multiclass … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L410 | **Two-Stage Decoupled Curriculum & Head-Specific LR Groups:** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L412 | **Stage 1 (Steps** $0\% \to 25\%$**):** $X$**-Manifold & Copula Warmup.** Freeze target residual heads ($\alpha_{\text{tree}}, \alpha_{\text{bilin}}, \Delta W$ frozen at $0.0$); train Encoder + Regime Skew-$t$ + Coupling … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L414 | **Stage 2 (Steps** $25\% \to 100\%$**): Joint Utility + Driver + Realism Fine-Tuning.** Unfreeze target heads with $0.25\times$ **learning rate** and cosine-annealed trust-region anchor $\mathcal{L}_{\text{trust}}$ … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L416 | **Periodic Sharpness-Aware Minimization (Periodic-SAM) + Adaptive EMA + Valley Soup:** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L418 | Apply **ASAM** (Adaptive SAM, $\rho = 0.03$) every $K = 4$ optimizer steps (only $\sim 25\%$ compute overhead on CPU-DDP/DiLoCo) to find flat minima that transfer across unseen physical table families. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L420 | Combine with **Parameter-Group Gradient Clipping** ($1.0$ backbone, $0.3$ residual heads, $0.2$ soft-tree thresholds), **Adaptive EMA** ($\beta = 0.995$), **Statistical-Feature Dropout** ($p = 0.10$ zeroing random … | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L422 | **Upgraded Infinite Synthetic Prior (`s3_v8_entropy.py`):** | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L424 | Extend `sample_synthetic_prior_table` to sample: | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L426 | Clayton, Gumbel, Frank, and Student-$t$ ($\nu \in [2.5, 10]$) copulas (not just Gaussian/$t$/Laplace linear factors), | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L428 | Multi-modal Gaussian mixtures ($M \in \{2, 3, 4\}$) with regime-rotated covariances, | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L430 | Pure XOR / depth-4 oblivious trees / Friedman #1–#3 / heteroscedastic variance targets $\sigma(x) = \sigma_0 \exp(\beta^\top x)$, | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
| L432 | High-collinearity factor clusters ($\text{VIF} \in [5, 50]$) with sparse causal drivers to stress-test driver preservation. | Antigravity | NOT STARTED | [Gradient methods](#gradient-methods) |
|  | **7. Overall KPI & Qualification Protocol Enhancements (`dope-kpi/v2` & `dope-fidelity/v3`)** |  |  |  |
| L436 | Without altering the governance freeze rules of `dope-kpi/v2` or `dope-fidelity/v3`, add four high-signal diagnostic and promotion guards to `dope_kpi` and `s3_v7_eval.py`: | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L438 | Proposed Metric / Guard — Exact Definition · Why It Closes a Blind Spot | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L439 | ----- — ----- · ----- | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L440 | **1. Analytic vs. Neural Uplift Decomposition** — Report both $\Delta_{\text{packet}} = \operatorname{TRG}(\text{V8}_{\text{step0}}) - \operatorname{TRG}(\text{V7}_{\text{step0}})$ and $\Delta_{\text{neural}} = \operatorname{TRG}(\text{V8}_{\text{trained}}) - \operatorname{TRG}(\text{V8}_{\text{step0}})$. · Prevents claiming a "deep learning win" when gain came only from better analytic preprocessing — forces the neural compiler to prove … | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L441 | **2. Top-**$K$ **Driver NDCG@5 & Cluster-ALE** $L_1$ — Compute $\operatorname{NDCG@5}$ of synthetic-trained GBDT permutation importances against real-twin $B$ top-5 drivers (in addition to full-column `driver_spearman`). · On 40-column tables with 5 true drivers and 35 noise columns, full-column Spearman is dominated by random rank noise among the 35 … | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L442 | **3. Top-Decile Lift Capture Ratio** — $\operatorname{LiftRatio}_{10\%} = \frac{\operatorname{Lift}_{10\%}(f_S \text{ on } T) - 1}{\operatorname{Lift}_{10\%}(f_A \text{ on } T) - 1}$ for binary/regression business-ranking workloads. · Directly measures the enterprise KPI customers care about (top-decile targeting lift on normalized real test data). | Codex | NOT STARTED | [Baseline replay](#baseline-replay) |
| L443 | **4. Factorized C2ST Attribution Audit** — Run 3 parallel boosted-tree C2ST classifiers: (a) **Marginal-only** (shuffled columns), (b) $X$**-Joint only**, (c) **Full** $(X, Y)$ **Joint**. · Pinpoints immediately whether remaining C2ST leakage is in 1D knots, $X$-copula tails/mixtures, or $P(Y \mid X)$ residual … | Claude | TESTING | [Realism and privacy](#realism-and-privacy) |
|  | **8.1 File-by-File Change Manifest** |  |  |  |
| L449 | Crate / Directory — New / Modified Files · Responsibilities | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L450 | ----- — ----- · ----- | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L451 | `engine/dope-norm/` — `src/nc2.rs` (update)  `src/screening.rs` (new) · Add first-class missingness stats + $O(n \log n)$ Composite Nonlinear Screening Score $S_j$ (replacing linear-only pruning). | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L452 | `engine/dope-core/src/prepare/` — `src/prepare/stat_bank.rs` (new)  `src/prepare/edge_stats.rs` (new)  `src/prepare/global_tests.rs` (new) · Pure-Rust SIMD computation of `ColStats[P, 76]`, `EdgeStats[P, P, 18]`, and `GlobStats[112]` (L-moments, Hartigan Dip, Chatterjee … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L453 | `engine/dope-core/src/single_encoding/` — `src/single_encoding/dse8_wire.rs` (new)  `src/single_encoding/dse8_compiler.rs` (new)  `src/single_encoding/dse8_decode.rs` (new)  `src/single_encoding/math.rs` (update) · Exact regularized multinomial Newton-IRLS; `DSE8` CRC32-verified binary codec ($\le 2\text{ KiB}$ headline + $4/16/64\text{ KiB}$ … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L454 | `engine/dope-arena/src/statpacket/` — `src/statpacket/effects.rs` (reuse)  `src/statpacket/dependence.rs` (reuse) · Share exact tie-disattenuated normal scores (`tie_scores`) and Wilson-Hilferty $\chi^2$-gated S8/S10 fitters with `DSE8` analytic … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L455 | `research/levels/` — `v8_stat_bank.py` (new, bit-exact parity with Rust)  `v8_compiler_model.py` (new)  `s3_v8_losses.py` (new)  `s3_v8_trainer.py` (new)  `s3_v8_entropy.py` (new) · Bit-exact NumPy/Rust parity tests; Edge-Biased Transformer + Tree-Query Cross-Attention; Differentiable Soft-GBDT TSTR + Driver … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
| L456 | `engine/dope-arena/src/kpi/` — `src/kpi/auditors.rs` (update)  `src/kpi/score.rs` (update) · Add NDCG@5 driver metric, top-decile lift capture ratio, factorized C2ST attribution (`marg / X / XY`), and Step-0 vs. Neural uplift … | Owner | NOT STARTED | [Execution scope](#execution-scope) |
|  | **8.2 Phased Verification & Acceptance Gates** |  |  |  |
| L460 | Phase — Deliverable & Test Command · Mandatory Pass Criteria | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L461 | ----- — ----- · ----- | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L462 | **P0: Stat-Bank Parity & Latency** — Rust SIMD `stat_bank` + `edge_stats` + `global_tests` vs. NumPy reference (`cargo test -p dope-core dse8_stat_parity`) · Max abs diff $< 10^{-4}$ across 50 S3 tables; p95 wall-time $< 5.0\text{ ms}$ ($n=2048, P=48$); zero non-finite outputs; strict $k … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L463 | **P1: Analytic Step-0 Upgrade** — `DSE8` Step-0 (NC2.5 + Composite Screening + Multinomial IRLS + Gated S8/S10 + Empirical Regime Skew-$t$) vs. `V7` Step-0 on 30-table DEV · $\Delta\operatorname{TRG}(\text{V8}_{\text{step0}} - \text{V7}_{\text{step0}}) \ge +0.04$, Multiclass retention $+0.08$, … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L464 | **P2: Stage-1** $X$**-Manifold Compiler** — Train `DopeV8Compiler` $X$-heads (Regime Skew-$t$ + Coupling Warp + Knots) with PCGrad + Soft-Tree C2ST · Boosted-Tree C2ST median $\le 0.65$ (strict `dope-fidelity/v3` gate); Realism pass coverage $+15\text{ pp}$ over V7; $\text{DCR} \le … | Claude | NOT STARTED | [Power and promotion](#power-and-promotion) |
| L465 | **P3: Stage-2 Full Utility + Driver Lift** — Unfreeze Bilinear + Soft Oblivious Trees + Heteroscedastic head with Periodic-ASAM + Soft-GBDT TSTR + Driver ALE loss (3 training seeds) · $\Delta\operatorname{TRG}(\text{V8}_{\text{neural}} - \text{V8}_{\text{step0}}) \ge +0.025$ (95% stratified family bootstrap CI low … | Claude | TESTING | [Power and promotion](#power-and-promotion) |
| L466 | **P4: Rust-PyTorch Bit-Parity & Sealed Confirm** — Sealed 90+ family CONFIRM panel (parity tested in `codex-hard-parity@8f4b4b92`; sealed CONFIRM blocked by active DEV-only pause and never run). | Owner | NOT STARTED | Policy-blocked by DEV-only pause — never run. [Execution scope](#execution-scope) |

</details>
