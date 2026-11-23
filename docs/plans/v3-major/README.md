# DOPE v3.0.0 master plan: get more of the real table into the packet, and verify it on `dope_kpi`

Status: plan, 2026-11-23. It is built from all six specs in `tips/v3/` (531 KB) and checked against `origin/main` `a297a9b9`. No number below is a new measurement. Owner directive "Stop dope now" (07:23Z) is in force, so every scoring and training step is staged for when the owner lifts it.

## 1. Why v2 stalled, and the single idea behind v3

V7's network (`SingleEncV7.refine_packet`) runs on the **receiver** and reads only the dequantized ≤ 2 KiB `DSE7` packet: 46 floats per column and 16 globals. A statistic of the real table that is not in the packet cannot reach the sampled rows. That explains the training results:
- H1: 5.85M episodes gave +0.0004 TRG over the untrained packet.
- H2's best is +0.0144.
- On the v2.0.0 scoreboard the step-0 packet scores −0.381, against −0.232 for the shipped `dope-2.0.0-small`. The gap is almost all in the gbdt auditor: median retention 0.812 vs 0.976.

Three facts make the packet weaker than it needs to be:
- Its target decoder is **linear only**. `Q_nl` is a network input and is never decoded (`single_encoding/mod.rs:455-467`).
- `class_means` is allocated and never filled (`v7_single_encoding.py:209`).
- Lattice, atom and one-hot structure is lost. In the Evidence spec's run, 45% of lattice columns are detectable on their own, and 31% of rows on one-hot tables are illegal.

**v3 thesis.** Every new piece of information gets a place to live:
- **(A)** an owner-side evidence encoder writes a small quantized code into the packet;
- **(B)** features derived from the packet itself, at zero bytes;
- **(C)** typed statistic sections, each with an analytic row-local decode;
- **(D)** training-only signals.

The decoder is a zero-initialized residual on the analytic decode, so step 0 always equals the analytic packet.

On the Evidence spec's 81-table public panel:
- Fixing the feature distribution alone gave +0.05.
- Fixing the conditional alone gave +0.01 to +0.08.
- Fixing both gave +0.26, which is twin parity within 0.05.

Neither half pays on its own. That run is not reproducible from the repo, so E0 (§4) re-measures it on DEV with Rust oracle arms before anything else is ordered.

## 2. Invariants (unchanged by v3)

- **Bytes:** ≤ 2,048 B whole-artifact headline. The 512 / 1,024 / 4,096 B grid comes from one encode via prefix truncation. 16 and 64 KiB are development frontiers only, never a headline (they are not in `dope-kpi/v2`).
- **Content:** zero raw rows on the wire. No source values, headers or extrema in packets, errors or sidecars. Support floor k ≥ 10 on every level, class, bin, leaf, atom and regime.
- **Decode:** row-local and counter-seeded, with no batch statistics. Labels are always sampled.
- **Runtime:** pure-Rust inference. Python only for deep-learning training. New extractors, metrics and tests are Rust.
- **Data:** target-last contract. Query targets never become features. Target swaps re-declare the target and rebuild every supervised block.
- **Episodes (owner rule, 2026-11-23):**
  - Every dataset gets a fresh random, duplicate-group-respecting train/test split in every training episode.
  - Every feature, metric, evidence block, column meta and packet statistic is computed from that episode's FIT rows only, so nothing derived from rows is cached per table or across splits.
  - Only split-independent declarations are reused: stored cells, task, the NC1 kind/group schema and dup-group ids.
  - The features are produced ahead of the trainer by the parallel CPU episode factory (C6). Training never waits on them (data-wait share < 3%).
- **History:** DSE7 v2, the V7.1 checkpoint, DPK3.2 and v1 receipts all keep decoding. DSE8 is a new identity.
- **Claims:** TRG on `dope_kpi` only, with Holm, guards and ≥ 3 seeds. Privacy results are empirical, never formal DP or HIPAA.

## 3. Ownership (agreed on the board, 09:25Z and 09:45Z)

| Area | Owner | Lane |
|---|---|---|
| FP-0 loop repairs in the V7 trainer; Rust evidence bank (`dope-core/src/evidence/`); DSE8 codec, typed sections, planner, Rust decode, `dope_v8_emit`, `auto/v8.rs`; V8 encoder, decoder, losses, trainer | Antigravity | `v3-dse8` (`feat/v3-dse8-major`) |
| `dope_kpi` oracle arms (E0) and diagnostics; per-column ledger; illegal-row counts; factorized C2ST; tree-driver agreement; DEV panel power and the `dope-kpi/v3` candidate; promotion receipts; arena v2; sampler, farm, clipping and noise helpers; `dope-norm` `Nc2ColumnMetaV2` | Claude | `claude-v3` (`claude/v3-proof-core`) |
| Lifting the compute stop; normalized-only vs raw-portable packet profile; `MAX_COLS` / `MAX_CLASSES`; successor KPI contract; the release | Owner | — |

## 4. The ladder: each rung is a measured gate on `dope_kpi`

Arms are paired on cells, seeds and the informative flag. Rung decisions use DEV-SELECT; promotion uses one ledgered DEV-CONFIRM look. **KEEP** if the lower bound of ΔTRG > 0 at equal measured bytes. **DROP** if the upper bound < +0.01. Otherwise INCONCLUSIVE, and not adopted.

| Rung | What | Needs training | Owner | Gate | Spec estimate |
|---|---|---|---|---|---|
| E0 | Oracle arms on DSE7 step 0 (§5 C1) | no | Claude | Reproduces the public-panel ordering, or the ladder is re-ordered | real X + small tree ≈ −0.05; packet ≈ −0.31 |
| E1 | DSE8 typed marginals + group-exclusive decode + missingness | no | AG (meta from Claude C4) | C2ST median ≤ 0.75; 0 illegal rows; ΔTRG low > −0.01; bytes ≤ DSE7 | C2ST 0.87 → 0.70 |
| E2 | + gated main effects (replacing `Q_nl`), gated pairs, residual oblivious trees, noise and temperature, multinomial IRLS / GCV base | no | AG | ΔTRG low > 0 and point ≥ +0.03; binary low > −0.02 | +0.05 to +0.2 |
| E3 | + regimes and class factor means, κ-disattenuated loadings | no | AG | realism coverage +0.10 or ΔTRG low > 0 | +0.015 to +0.03 |
| **Analytic headline** | E1–E3 vs `dope-2.0.0-small` (−0.232) | no | Claude scores | PROMOTE rule | first v3 KPI claim |
| E4 | V7 + channel-B features + pair bias (features without information) | yes | AG | control: expected ≈ 0 | — |
| E5 | V7 + v3 objective, no new information | yes | AG | clipped steps < 5%; probe variance halved; ΔTRG ≥ 0 | training health |
| E6 | `SingleEncV8` encoder + FSQ code at tiers 0–3, on E3 packets | yes | AG | ΔTRG low > 0 at equal bytes, ≥ 3 seeds same sign | — |
| E6a–h | Leave-one-family-out over evidence blocks | yes | AG + Claude | keep a family only if removing it costs ≥ 0.01 | prunes the layout |
| E7 | + row-local residual coupling flow | yes | AG | C2ST −0.03 or ΔTRG ≥ +0.02 | the remaining ≈ +0.10 |
| E8 | Encode-time code refinement (`fast` / `accurate`) | yes | AG | ΔTRG low > 0; privacy Δ > −0.03 | — |
| E9 | Byte curve by prefix; embedding acceptance | no | Claude | dominates Format 1/2 at every claimed grid point | — |
| CONFIRM | One DEV-CONFIRM look, then SEALED | no | Claude | `dope-kpi` promotion rule | v3.0.0 headline |

**Stop rule.** If E6 and E7 do not beat E3 at equal bytes after one full run and one ablation round, v3.0.0 ships the analytic E3 packet. A trained model then ships only behind the encode-time acceptance check (FP-6), which picks the best of P0, P1, Format 1 and Format 2 on an inner split.

## 5. Claude's work packages

**C1. E0 oracle arms** (`engine/dope-arena/src/kpi/controls.rs` + new `kpi/oracle*.rs`). All arms are on identical cells, with labels always sampled:
- `real_x_packet_y`
- `packet_x_gam_y`: linear + joint 4-bin main effects
- `packet_x_smalltree_y`: 40 depth-3 oblivious trees on A
- `packet_x_teacher_y`: cross-fitted booster
- `real_x_smalltree_y`
- `exact_marginal_rank12`
- `exact_marginal_fullrank`

The packet side calls the existing `dope_core::single_encoding` DSE7 fit and decode, and DSE8 once its decode entry point exists. The receipt shows the oracle headroom next to every candidate (EV-66, EV-70).

**C2. Diagnostics:**
- `kpi/col_ledger.rs`: per (table, column) support class, distinct count, q0, atoms, 1-D GBDT AUC vs twin AUC, bytes (EV-67).
- Illegal-row and one-hot violation counts from the NC1 group map (EV-68).
- Factorized C2ST: marginal-only, X joint, X+y (V8E-55).
- Tree-auditor grouped permutation-importance agreement vs the real-real ceiling (EV-69, DF-129).
- Driver NDCG@5 and top-decile lift capture as diagnostics only (V8E-53/54).

**C3. Data core for V8 training:**
- `dope.flat-corpus/v2`, readable beside v1. It stores only split-independent declarations: per-column declared kind, NC1 group (kind, start, width), NaN-preserving cells, unchanged physical row identity and `dup_groups` (EV-54, FP-1.3, DF-28). Lattice, atom and missingness meta are row-derived, so they are computed per episode from FIT rows by C4 inside the factory, never stored per table.
- Counter-sampler path for the trainer:
  - real views ≥ 60% of every step, with no prior-only fallback (EV-52, DF-22);
  - family-balanced with an exposure cap;
  - ≥ 40% of episodes at n_fit ≥ 256 (FP-0.2, DF-11);
  - |Q| ≥ 256 where the table allows;
  - a disjoint twin B split by duplicate group (FP-0.3, EV-57, DF-8).
- Gradient farm:
  - per-episode clipping at c × the running median (EV-59);
  - the B_noise estimate (EV-60);
  - helpers for common random numbers, antithetic and stratified noise (EV-58, DF-103..105).

**C4. `dope-norm` `Nc2ColumnMetaV2`.** Typed column meta computed on raw values (gap rule M01), so DSE8 can store q0 and atoms in unit space:
- support class: continuous, lattice, atoms, zero-inflated or constant;
- lattice q0 from mantissa {1, 2, 2.5, 5} × 10^d, with origin and heaping multiples;
- atoms with count ≥ 10, at most 32;
- bound flags and group id.

It reuses `dope_core::marginal::{lattice, atoms}`. It is computed per episode from FIT raw values inside the factory (C6). `Nc2PublicSchema` derives `Eq`, so float fields are stored quantized (or the v2 struct drops `Eq`).

**C6. Parallel episode factory** (`research/levels/s3_episode_factory.py` + Rust `engine/dope-ffi`). Status: first slice built and tested.
- Pipeline per work unit `(step, slot)`:
  1. the counter-sampler plan;
  2. `s3_view_builder` fresh group-safe random split and augmented view, with `emit="raw"` so real columns keep their raw values;
  3. a native FIT-only stage over the C ABI: today the DSE7 packet plus FIT and query rows in packet coordinates; next C4 meta, then AG's evidence bank and the DSE8 fit;
  4. a fixed-layout `/dev/shm` ring with backpressure.
- The trainer reads `get(step, slot)` in any order inside a window of ≥ 2 steps. A gradient-farm worker attaches to the ring by name.
- Determinism: a payload is a pure function of `(seed, step, slot)`. The tests check identical bytes for 1 and 3 producers and against the in-process path.
- FIT-only: rewriting the query rows leaves the packet and every FIT block unchanged (Rust and Python tests). Failed units are counted payloads with static codes, never skipped.
- Measured on the real `s3-v250` arena, xbabe2 under load (10 to 12 ms per episode per producer core; 0 failures in 1,440 episodes; packets 842 B median, 2,048 B max):

  | Producers | Episodes/s |
  |---|---|
  | 1 | 97 |
  | 2 | 167 |
  | 4 | 256 |

  One producer core feeds about 6 to 15 trainer cores at the M3 estimates, so a 1:4 producer ratio leaves ample headroom.
- Every new FIT-only stage must stay single-threaded, deterministic, free of global state and query-poison safe, with a per-episode budget: ≤ 25 ms for T0–T2 at n_fit ≤ 4,096 and P ≤ 48.

**C5. Verification readiness:**
- Measure σ_Δ on DEV. 12 informative headline cells are far too few for +0.01–0.03 gates: about 196–784 families for δ = 0.02–0.01 at σ = 0.10. Build a larger DEV-SELECT/CONFIRM panel with a monotone re-cut that never moves a CONFIRM family.
- Draft a `dope-kpi/v3` candidate. Its main change is failed_value = −1 − R_B, so failing a cell can never beat a valid floor (DF-127). `v2` stays untouched; the owner decides.
- V8 arm adapters for `dope_kpi score`.
- `render-readme` rows for E-rungs.

## 6. Antigravity's work packages (as claimed on the board)

**A0 / FP-0. Loop repairs** (DF-1..25):
- type- and lattice-safe jitter (fixes the `is_disc` skew);
- one Rust discreteness rule;
- per-episode masks (the realism terms used about 26 query rows);
- eval-matched sizes, 4n synthetic rows, no prior-only collapse;
- group-safe splits; no class-0 fallback;
- native-packet training records;
- the same integer and lattice snapping in training and serving.

**A1. Evidence bank** (`engine/dope-core/src/evidence/`, `contracts/evidence-layout-v1.json`):
- per-column: 128 features (DFP1 proposed 188);
- per-pair: 8–19;
- per-class: 12–24;
- per-group: 8;
- global: 160–503.

Built from T0–T3 tiers as (value, z, mask) triples, FIT-only, k ≥ 10, computed per episode inside the factory (C6). The T3 per-table cached native-landmark panel is dropped or moved into the per-episode budget, because a per-table cache would leak across splits, with planted-mechanism, poison, equivariance and determinism tests (FP-1, EV-20..30, V8E-4..22, DF-33..56, DF-1001..2038). The catalogue is weighted toward support and type, what the packet gets wrong, tree-visible conditional structure and noise. It is weighted away from long lists of univariate tests: 24 of those lift a driver probe only from 0.51 to 0.58, while one tree witness reaches 0.64.

**A2. DSE8 codec** (`single_encoding/v8/`). Sections in nested-prefix order:
- header and group map;
- scale card for continuous columns only;
- support class, atoms, lattice, heaping;
- categorical intercepts with arg-max decode;
- knots with tail shape;
- loadings with κ;
- regimes / class means;
- linear target (IRLS / GCV);
- gated main effects;
- gated pairs;
- oblivious trees with knot-index thresholds;
- residual noise and temperature;
- missingness;
- learned code.

A deterministic planner never silently drops a column or class. CRC is applied, prefix truncation gives the byte grid, and the decode is row-local Rust.

**A3. V8 model and objective:**
- Owner encoder over the evidence plus Ψ(packet), producing FSQ codes with nested and evidence dropout.
- Receiver decoder: zero-init residuals on the analytic decode, pair-bias attention, group and class tokens, and an optional coupling flow.
- Loss: ℓ_ret, the twin-referenced retention deficit, should be a **soft-min over surrogates** to match the KPI. Plus ℓ_pit and ℓ_cop in closed form, ℓ_tree (random oblivious trees and a tree-region surrogate), ℓ_drv, ℓ_cond with reliability, ℓ_aux, ℓ_twin, rate and ℓ_guard (FP-3, EV-40..51, V8E-40..46, DF-85..102).
- The MLP stays a **held-out** auditor and is never a training student.

**A4. Encode-time acceptance** (FP-6) and embedding export (FP-7, EV-34).

## 7. Defects in the specs (do not build them as written)

- **Sinh-arcsinh warp (V8E-28):** a per-column monotone warp cannot change the copula, so λ_U and λ_L stay the same. Use regimes or a shared-latent skew instead.
- **Gate α = tanh(γ), γ = 0, on a zero-initialized head (V8E-36):** both gradients are zero, so training never leaves step 0. One of the two must start non-zero.
- **FSQ `round(8·tanh)` (EV-18/31):** gives 17 levels, which does not fit 4 bits. Use 15 levels or a half-step offset.
- **ℓ_ret averaging surrogates:** the KPI takes the minimum over auditors. Use a soft-min.
- **t mixing, regimes and coupling:** these change the latent marginals, so each needs a per-column PIT correction (closed form for t and mixtures).
- **Latency targets** of < 5 ms for the whole V8E bank, and < 3 ms for P² × 18 edge statistics: impossible with O(n²p) and EM blocks. Tier them (EV-20 budgets).
- **16 / 64 KiB tiers and P ≤ 128:** not in the contract, and they break `MAX_COLS = 48`. Development frontiers only.
- **E-1 baseline:** "V7.2 as released" is wrong. The release is V7.1 d160l3 `yh` k6000.
- **Byte table:** DFP1's 40-tree example costs 564 B for binary but 2,804 B at K = 8, so multiclass trees need shared leaves or one-vs-rest packing.

## 8. Owner decisions (each has a default so work is not blocked)

1. **Compute stop:** default HOLD. Nothing is scored or trained until the owner lifts it.
2. **Packet profile:** default DSE8 carries both `self_contained_raw` (today's raw-unit scale card) and `normalized_only`. The headline stays raw-portable, so JopeDime and `dope sample` keep working.
3. **Width and class envelope:** default `MAX_COLS = 48`, `MAX_CLASSES = 8`. Over-envelope tables are refused explicitly and counted, never silently mapped to class 0.
4. **KPI successor:** default `dope-kpi/v2` decides. The `v3` candidate is reported beside it until the owner adopts it.
5. **Zero-repeat vs bounded replay:** default zero-repeat across real families, plus the random-teacher stratum on real features (EV-53). Bounded replay is a curriculum arm only.

## 9. Idea inventory

All six specs were read end to end by five readers; the two Dataset-Features versions were read together. Every proposal has an id in the reader inventories, kept in the session record:
- Feature Pack: FP-0.1…FP-8.4, plus the full DFP1 catalogue (188/19/24/503).
- V8 engineering: V8E-1…57.
- V8 evidence: EV-1…72, plus design rules R1–R7.
- Dataset Features: DF-1…156 and catalogue DF-1001…2038.
- V8 runbook: V8R-*, merged on arrival.

Each id maps to a package above:
- FP-0, DF-1..25, EV-52/56/57 → A0 + C3.
- Typed marginals and groups (EV-3..9/17, V8E-1/30, DF-5/6/28/70..72) → A2 + C4.
- Conditional sections (EV-13..16, FP-2.1..2.5, V8E-31..35, DF-65..68/74) → A2.
- Dependence (EV-10/11, FP-2.4, V8E-3/26/27/29, DF-15/69) → A2.
- Codec and planner (EV-1/2/18/19, FP-2.8/4.4, V8E-38, DF-77..84) → A2.
- Evidence (FP-1/5, EV-20..30/33, V8E-4..22, DF-33..56/1001..2038) → A1.
- Model (FP-4, EV-31..39, V8E-23..25/36/37, DF-57..64) → A3.
- Objective (FP-3, EV-40..51, V8E-40..46, DF-85..102) → A3.
- Optimization (FP-0.6/0.7, EV-58..65, V8E-47..50, DF-103..117) → A3 + C3.
- Data and curriculum (EV-52..55, V8E-51, DF-118..124) → C3 + A3.
- Evaluation (EV-66..72, FP-6/8, V8E-52..57, DF-125..143) → C1/C2/C5 + A4.
- Embedding (FP-7, EV-34) → A4, verified by Claude (E9).
