# v2 gap closure — approved plan (2026-11-22)

Approved by the owner on 2026-11-22. Companion files: [`tips-gap-matrix.md`](tips-gap-matrix.md) (327 requirements), [`decisions.md`](decisions.md), [`formats.md`](formats.md). The specification documents referred to as "tips" (A–I) are kept outside the repository.

---


Evidence: all 9 tips specs read end-to-end (1.1 MB); a 10-lens read-only audit of
code + live fleet; a 14-topic conformance sweep (327 tips requirements checked
against the repo); three implementation designs; two adversarial critiques of
this plan. Markers: [run] executed, [read] read in code, [log] run logs,
[est] estimate to be measured.

## 1. Context

You asked for a detailed plan that closes any/all remaining gaps between the
tips and what we have now — with emphasis on the fitness metric, speed,
gradient smoothing, unique datasets per batch, augmentation, proof of
improvement, advanced input features and a small encoding — coordinated with
Antigravity through `~/dope/agent_chat.md` on a fleet shared with ASR.

**Bottom line.** Of 327 tips requirements, **11 are implemented and measured**
(all in the v1/JopeDime track, and the decisive one — gate G0 — measured a
fail); 203 exist as partial or unmeasured code, 72 are missing and 37 are
contradicted. No `dope-fidelity/v3` number has ever been
computed. The dashboard's 0.90 "Main KPI" comes from a sampler that keeps the
raw rows, so it is not the KPI of an encoding; the best source-free encoding is
Rust Format 2 at 0.60–0.64 and the true single-encoding model (V6) scored
about −0.13 before its runs were stopped. The plan: (1) build the scoreboard the tips specify and measure
everything that exists; (2) ship a ≤ 2 KiB encoding that works without
training, then a ladder that improves it; (3) make training fast, resumable
and ASR-safe on CPU; (4) give every remaining tips requirement an owner.

## 2. Decisions

**Made by you (19:17Z):** (1) **CPU-first** — DOPE on idle CPU; a GPU only when
no ASR process is on it. (2) **The encoding** = zero-raw-row packet, headline at
**≤ 2 KiB/table**, 0.5–4 KiB curve, fixed 64/128-D prefix = the embedding.
(3) **Roles** — Claude = proof + speed core; Antigravity = model architecture,
Rust engine, dashboard, launches.

**Acknowledged by Antigravity on the board (19:43Z):** decisions 1–3; it
stopped its three GPU runs at 19:42Z (0 DOPE GPU processes fleet-wide,
verified), will label V5/flow rows `rows_at_sampler`, will not start a long V6
run before the re-cut, and accepted the file split in section 5.

**Assumed here (say so if wrong):**
4. Python stays only where torch is unavoidable (one checkpoint→rows emitter;
   the training core; later a torch mirror of the packet decode). Scoreboard
   and stat packet are **Rust**.
5. My code ships as draft PRs from my own lane; you or Antigravity merge. I
   never signal, kill or edit Antigravity's runs, lane or files.

**Still yours to make (each has a default so work is not blocked):**
- **Split re-cut** after the retro-score shows the measured MDE. Default:
  monotone re-cut (only train→panel moves, ≈ 29 binary + 40 multiclass
  families), new manifest file, old manifest untouched.
- **Byte target for the neural model**: Antigravity's own plan proposed
  1.5–4.0 KiB; your answer was ≤ 2 KiB. Default: ≤ 2 KiB headline stands and
  the contract is amended with the 0.5 / 1 / 2 / 4 KiB grid (this overrides the
  tips' "claim at the Max tier").
- **v1 / JopeDime lane**: its own go/no-go probe (H470, gate G0) ran and
  **failed** (CI [−0.009, +0.021], MDE 0.060 vs the 0.01 target). Default:
  HOLD all decision-supervised pretraining and 10M-corpus work until a powered
  panel exists (tips C/D say stop; B says record ADVANCE/HOLD/NO_ADVANCE).
- **Third-party baselines** for the final claim (tips decision D-1). Default:
  deferred until the sealed run; controls + Rust Format 2 until then.
- **Idle cores on GPU hosts** (xbabe1–9). Default: not used; xbabe0 only.

## 3. Where we are (verified 19:40Z)

- **Repo**: PR #387 squash-merged 19:04Z → `origin/main feafa4ee` (engine
  2.0.0); Full Jankurai audit failing on main; no open PRs; no v1.3.1/v2.0.0
  tag or release. Antigravity is in plan mode with its own Waves 0–6
  (`~/.gemini/antigravity-cli/brain/77d96545-…/complete_tips_gap_closure_plan.md`);
  its first PR (#388) will fix Jankurai and land the hooks I asked for.
- **Not an encoding KPI** [read]: V5 and today's `flow_v6` rebuild copula,
  knots, discrete values and target weights from the raw fit rows at sampling
  time. V5 at step 50 scored 0.79–0.81; flat after ~2k; multiclass fell
  0.62 → 0.55 [log].
- **Encoding numbers that exist**: Rust Format 2 0.60–0.64 median (binary 0.08)
  on 22 tables by an untracked script; V6 single-encoding −0.13 (C2ST 0.997).
- **Fitness function** [read]: 42 entries, 1 split, 1 seed, weak auditors,
  pooled median, twin-B fitted then discarded, argmax-over-checkpoints
  promotion (noise SD 0.012–0.017 → ≈ +0.05 optimism), val steers the
  curriculum, 30–62 of 453 val tables have same-source train siblings, 33
  multiclass val tables. The Rust `evaluate_v3` reducers run only in tests.
- **Speed** [log]: 27.5 tables/s on 4 A100 ranks (today's V6 runs: 6.5–14.6);
  one Python thread per rank; ASR lost 20–30 points of GPU duty.
- **Gradients** [log]: 33% of steps clipped, p99/median 15×; flow_v6 today:
  grad norm 69, 106 of 1,410 steps skipped.
- **Checkpoints**: EMA weights only, single copy, no resume (ddp4 lost at
  18,615/20,000). 767 V5 checkpoints survive only on xbabe5/xbabe6 cloud VMs.
- **xbabe0** [run]: 64 physical cores (128 threads), 119 GB free, torch 2.10,
  no usable GPU. ASR CPU queues own logical CPUs 48–95 and 112–123; logical
  **32–47 + 96–111 (16 physical cores) are not ASR-assigned**.

## 4. Gap matrix: tips → repo (327 requirements, 14 topics)

Status counts per topic: measured-done / partial-or-unmeasured / missing /
contradicted. Full rows (id, tips refs, threshold, evidence file:line, closing
action, effort, priority, owner) are in the sweep output and get committed as
`docs/plans/v2-gap-closure/tips-gap-matrix.{json,md}` in milestone M0.

| # | Topic (tips refs) | Done / Part / Miss / Contra | Decisive gaps | Closes in |
|---|---|---|---|---|
| 1 | Truth repairs, replication (G S0, H W0, I P0, F W00–02) | 0 / 15 / 3 / 5 | No G0 measurement or replication harness; README/renderers still show stand-in baselines and clipped means; determinism receipt still 25/36 | M1, AG-3 |
| 2 | Arena + `dope-fidelity/v3` (H §3 W1, I §9 P1–P2, G S1–S2, E §19) | 0 / 18 / 4 / 2 | No bundle producer, panels, ledger or receipt; κ* hard-coded 1.25; no table-level or simultaneous-bound rule; contract unfrozen | M1, M4 |
| 3 | Emission + sampler (G S3, I P3, H §5.4, E §20) | 0 / 15 / 5 / 4 | Format 2 `distill` emits weights that sum to 2; Format 2 ignores row offset/namespace; neural samplers use batch statistics | M2, AG-2 |
| 4 | Marginals / column realism (G S4, H W3, I P4) | 0 / 15 / 7 / 1 | Lattice codec fitted on scaled, not raw, values; no instrument sees grids; gate never measured | M2, AG-2 |
| 5 | Teacher → student (H W2, I P5, G S5, E §14–15, F §9) | 0 / 18 / 4 / 0 | Gate never measured; ≤ 4,096 probes and ≤ 80 trees vs "fill the budget"; no direct-label control; poison test poisons nothing | M5, AG-2 |
| 6 | Rules + CATF (H W4, I P6, G S6, E §13/16) | 0 / 17 / 4 / 1 | No ablation arms; CATF at 2–8 trees (measured useless); P+ weights discarded | AG-2 |
| 7 | Closed-loop compile, tiers (H W5, I P7, G S7, E §17) | 0 / 13 / 10 / 0 | Loop not in the product path; ships the BUILD-only fit (64% of rows); no selector, no tiers | AG-2 |
| 8 | Packet, scaling, privacy (I P8–9, H W7, G S8, E §20–21, F §13) | 0 / 18 / 2 / 2 | No measured bytes/fidelity at ≤ 2 KiB; packet stores names, levels, lo/hi; privacy backoff fails open; `embed` is not a function of the packet | M2, AG-2 |
| 9 | Drivers, qualification, confirmation (I P10–11, H W6/W8, G S9, E §18–22) | 0 / 9 / 11 / 4 | No panels or freeze; simplified, uncalibrated statistics; no rival evidence | M1, M4, owner |
| 10 | Neural / research lane (H W9, G §13, I §16, E §23, F §8–16) | 0 / 7 / 6 / 10 | Neural campaign launched before any scoreboard gate; no neural number is an admissible encoding result | M1–M5 |
| 11 | v1 identities + value instrument (A WP00–01, B DV-00–03, C WP0–1, D WP0/10) | 7 / 12 / 2 / 1 | G0 failed on an under-powered panel and no decision was recorded; cache identity enforced nowhere | owner, M7 |
| 12 | v1 objective + architecture (A §7–8, B DV-05–08, C §5, D WP2–6) | 2 / 18 / 3 / 0 | No decision-supervised model ever trained; modules are test-only; no outcome labels | M7 (held) |
| 13 | v1 corpus, streaming, scale (A §9–13, B §6/9/12, C WP2/5–7, D WP5/7) | 0 / 15 / 5 / 4 | Family identity; no resume or exposure ledger; counters; 4,145 bases vs 10M; no scaling evidence | M3, M4 |
| 14 | v1 promotion, integration, delivery (A §14–15, B §11/17, C WP8, D WP8) | 2 / 13 / 6 / 3 | No confirmation panel; no identity, loader or Rust inference for a post-v3 encoder; receipts have no writer | M4, AG-2 |

Open rows by priority: P0 86 · P1 84 · P2 106 · P3 39 (deferred by the tips
themselves). By natural owner: Antigravity 134 · Claude 128 · either 33 · you 18.
A coverage pass over all nine documents found every work package and gate
covered; it added cross-cutting items now included in M0/M7: a register of
owner decisions, the contract amended with your byte grid, a per-package
status ledger for the v1 programme, marking stale-token JopeDime experiments
as non-evidence, `dope sample --given/--labels` (needed by the attribution
arms), and a compute plan for third-party baselines under CPU-first. Topic 11's
"done" rows are measured *nulls*: gate G0 failed.

## 5. Ownership (acked by Antigravity 19:43Z; reconciles its Waves 0–6)

| Area | Owner | Notes |
|---|---|---|
| Scoreboard, panels, re-cut, contract freeze, promotion receipts, champion registry, V5 retro-score | **Claude** | Was its Wave 4 "arena panels + v3 freeze" and Wave 2 "hard-gate promotion" |
| Stat packet, codec, rate–fidelity curve, eval-only ladder | **Claude** | New; its `single_encoding.rs` adopts the same byte layout |
| Training speed core: sampler, view builder, gradient farm, full-state resume, batched step | **Claude** builds, AG adopts | Was its Wave 2 "batching / DDP / `--resume`" |
| V6/V7 architecture, losses, label path, InfoNCE, landmarks in the model | **Antigravity** | I supply tested feature modules + evidence |
| Rust engine: Format 2 bugs R10–R18, marginals on raw values, distill emission, CATF, closed loop, tiers, packet privacy, `embed`, Format 2 section for the packet | **Antigravity** | I verify each on the scoreboard |
| Dashboard, launches, CI (Jankurai on main), releases | **Antigravity** | |
| v1/JopeDime programme | **you** decide, then split | Held by default |

**AG-1 (now):** GPU runs stopped (done 19:42Z); edit only its own board
block (agreed); versioned deploy dirs; keep old run directories and unscored
checkpoints (`keep_last`); keep evaluator and curriculum off my panels; PR #388
with the Jankurai fix and my hooks. **AG-2 (verified by me on the scoreboard):** the
Rust and model items in the matrix rows it owns. **AG-3:** truth-record
repairs (stand-in names, clipped means, registry enforcement, determinism
rerun). Integrator patches I need from it are listed per milestone; each has a
no-patch fallback so I am never blocked.

## 6. Work plan (my milestones; each ends in a measured result)

Execution method: each milestone runs as a multi-agent workflow in my lane
(implement → run the tests in section 8 → independent adversarial review →
fix), and every number I report comes from a ledger that can be re-reduced.

### M0 — Coordination, evidence, safety (first hours)

1. Board: read at the start of every work unit and at least every 15 minutes;
   write only my block (compare-and-swap); post UTC + commit sha at every
   milestone, interface change and claim; review Antigravity's new commits
   each cycle (`git log <last-reviewed>..origin/main`) and post findings.
2. Persist evidence now (it sits in `/tmp`): copy the four study outputs to
   `~/dope/.agent/reports/claude-gap-study-20261122/`; commit the gap matrix
   (ids prefixed per topic, duplicates merged) and a register of owner
   decisions under `docs/plans/v2-gap-closure/`; save the coordination facts
   (roles, CPU envelope, board protocol) to project memory.
3. **Evidence freeze first**: `nice ionice -c3 rsync -a --ignore-existing
   --bwlimit=20000` of every checkpoint + `eval.jsonl`, `train.jsonl`,
   `config.json`, ledgers from all run directories on xbabe5/xbabe6 (≈ 30 GB)
   to `xbabe0:~/dope-data/registry/<run>/`; sha256 on source and copy; no
   delete flags. Store `v5_fidelity_net.py@d7ecc2ef` beside them (not in the
   repo) so old checkpoints stay loadable.
4. Lane: `lanes sweep` as a dry run first; `git fetch`; create
   `.agent/worktrees/claude-proof` on `claude/v2-proof-speed-core` from
   `origin/main`; builds run `nice -n 19 ionice -c3 CARGO_BUILD_JOBS=12` with
   their own `target/`, after a `df` check (xbabe2 disk is 97% full).
5. ASR baseline (30 min, read-only): GPU-job steps/s and `fleet_guard` duty per
   host, CPU-queue job durations on xbabe0, load, free memory.
6. Two safety deliverables for the runs that exist today:
   **launch gate** (`research/levels/tests/test_s3_launch_gate.py`: 20-step
   CPU smoke per arch on a 6-table fixture — train rows present, ≥ 1 optimizer
   step, checkpoint differs from init, evaluator round trip) and
   **interim resume** (`research/levels/s3_full_state.py` + ≈ 10 call-site
   lines for Antigravity: raw weights, AdamW, scheduler, EMAs every 600 s and
   on SIGTERM, `--resume`). Runtime rules requested: failed episodes counted
   by type, abort after 3 steps with zero valid episodes, no checkpoint
   without an optimizer step.

### M1 — Scoreboard MVP and the retro-score (Rust; the first trustworthy numbers)

New modules under `engine/dope-arena/src/`, built by a new auto-discovered
binary `engine/dope-arena/src/bin/dope_kpi.rs` (`#[path]` includes; no new
crates; reads tables from the flat float32 arena, so no edits to
Antigravity's `main.rs`, `lib.rs`, `Cargo.toml` or lockfile). Hooks it is
landing in PR #388: `pub fn fit_predict_auditor` (wrapping the private
`eval::run_auditor`) and `reduce --kappa-file --allow-partial`. Fallback if
they slip: a 40-line local wrapper over the public `eval` fitters.

| Module | Does | Reuses |
|---|---|---|
| `panels_build.rs` (`build-panels`) | Lineage families = canonical stem + column-fingerprint union-find; LEGACY panel on the current split (old-val families with no train sibling); manifests with sha | `dope-arena::panel::PanelManifest` |
| `cellplan.rs` (`plan`, `real-ref`) | Frozen A / B (same size) / C / T indices per (table, data seed); real-only results cached once: L0, L_A, L_B, informative mask, twin retention, real-vs-real C2ST, κ* from real C | `dvs/v3/harm.rs`, `equiv.rs::calibrate_from_real_c`, `eval/*` |
| `score.rs` (`score`) | Fits auditors on synthetic rows; writes `FidelityCellRequest` ledger (`PredictionBundle`, realism, privacy inputs) + compact rows | `dvs/v3/{bundle,c2st,privacy,groups}.rs`, `dope-arena::ledger` |
| `controls.rs` | Arms on identical cells: real twin, copy of A, bootstrap of A, independent marginals, label shuffle, **Rust Format 2** (`dope_core` fit/decode/sample, bytes measured) | `dope-arena::report::run_swap_ladder` |
| `compare.rs` (`primary`, `compare`, `null`) | Primary statistic, stratified family bootstrap, guards, receipt; A/A null from ≥ 2,000 label-swap replicates | `dvs/v3/qualify.rs`, `evaluate_v3` |
| `research/levels/s3_emit.py` (Python, the only one) | PyTorch checkpoint → packet bytes → rows with **sampled** labels as `DOPEMAT1` files; failures are rows, never skips | `S3CorpusStore`, V6 encode/decode |

Definitions (written into `contracts/dope-kpi-v2.json`, frozen by hash in M4):
- **Cell**: n_T = clamp(0.30·G, 200, 2000) duplicate-groups; n_A = n_B ≤ 1024;
  headline iff n_A, n_T ≥ 200, else a separately reported small-n stratum.
  Every stream = sha256(panel, table, data seed, sample seed, purpose).
- **Scored object** = `decode(unpack(pack(encode(A))))`, bytes = `len(file)`;
  class `source_free | rows_at_sampler | control`. m = 4n (primary) and m = n.
- **Auditors** (Rust, duplicate-row invariant): `linear-seeded-v1` and `gbdt`
  (400 rounds, depth 6) enter the statistic; `mlp-seeded-v1` is held out of
  every training loss and used only for a direction-agreement guard.
- **Primary statistic TRG** (selection and promotion only): R_x = (L0 − L_x) /
  (L0 − L_A); x = mean over sample seeds of clip(R_S, −1, 1.25) − R_B; per
  table the **minimum** over {linear, gbdt}; mean within family, mean of
  families within task, mean of the three tasks. 0 = as good as a second real
  sample. Failed, over-budget and single-class cells count as −1. The
  uninformative mask comes from real data only.
- **The claim** stays the contract's unclipped coverage shares from
  `evaluate_v3` (parity vs twin, equivalence, realism, privacy, pass), always
  printed beside TRG with the twin bar, learned lift over an untrained model,
  matched-n value, q10, unclipped median retention, and median/max bytes.

Order and exit results: (a) LEGACY panel + real-ref → CPU-seconds per cell;
(b) **control self-test**: twin CI contains 0 and C2ST ≈ 0.5; copy of A fails
privacy; bootstrap within 0.02 of copy; independent marginals and label shuffle
≈ −1 — no model row is read before this passes; (c) **retro-score** V5 (steps
50, 500, 1k, 2k, 3.1k, 5k, 10k, 15k, last, 30 adjacent plateau checkpoints, one
untrained init), Rust Format 2 and the current V6 → noise SD, per-family paired
SD, **measured MDE**, twin bar, and whether training ever helped.

### M2 — Stat packet v0 and codec (Rust): a ≤ 2 KiB encoding with no training

`engine/dope-arena/src/statpacket/{fit,pack,decode}.rs` — the old V5
generator with every row-derived quantity replaced by a stored, quantised
statistic; it doubles as the Gaussian-copula+ridge control arm (one
implementation). Reuses `dope_core::{copula::Dependence::fit_factor,
marginal::atoms::detect_atoms, marginal::lattice::detect_lattice, rng::hash3}`.

- **Codec `dope.single-encoding/v1`**: 24-byte header (magic, kind
  stat/neural/hybrid, task, K, P, level, bit profile, decoder sha8) + sections +
  CRC32; cold bytes (shared decoder) reported separately.
- **Sections in nested-prefix order**: S1 priors/noise · S2 coarse marginals
  (5 knots, atom flags) · S3 linear target · S4 rank-4 dependence + class
  factor means · S5 knots → 9 · S6 rank 8 · S7 knots → 17 · S8 binned main
  effects · S9 rank 12 · S10 top-16 pairs. Knots are smoothed order statistics
  (no min/max stored); discrete columns are atom tables with a k ≥ 10 support
  floor; decode is row-local and analytic (counter RNG, no batch statistics,
  sampled labels).
- **Bytes** (arithmetic, verified by test): regression P=24/48/64 → level E
  853 / 1,624 / 2,138 B; multiclass K=5 → level C 594 / 1,098 / 1,434 B,
  level E 1,337 / 2,468 / 3,222 B.
- **Headline scope**: full-width packets; anything over 2,048 B is over-budget
  in the denominator; the codec degrades deterministically (lower level, then
  top-m target weights) instead of failing; the share of tables encodable at
  ≤ 2 KiB is reported by task × P bucket (≤ 24, 25–48, 49–128, > 128) × K
  (17.7% of val tables have > 48 columns).
- **Embedding acceptance**: 64-D prefix must (a) agree across two disjoint row
  halves of a table (cosine ≥ the twin reference), (b) retrieve same-family
  tables above a shuffled control, (c) be stable under column permutation.
- Privacy/echo tests: planted canary value and any level with count < 10 never
  appear in packet or samples; chunk-invariant decode.

Exit: TRG + coverage + bytes for the stat packet at 0.5 / 1 / 2 / 4 KiB from one
encode per table; the **first promotion receipt**: stat packet (≤ 2 KiB) vs
Rust Format 2 on the LEGACY panel (expected to win [est]; if it does not, that
is itself the finding and the ladder starts from Format 2).

### M3 — Speed core, model-agnostic part (Python, training-coupled; CPU only)

Built against a `loss_fn(batch) → per-episode vector` contract so it serves the
legacy loss today and any later model. Runs standalone
(`python -m research.levels.s3_speed_loop`) or behind `--core batched`
(≈ 6 call-site lines in Antigravity's trainer).

| Module | Does |
|---|---|
| `s3_flat_corpus.py` | 0.58 GiB float32 arena + index in `/dev/shm`, read-only for every worker |
| `s3_counter_sampler.py` | Step plan = pure function of (seed, optimizer step): family-balanced strata, **zero repeated families per global step**, exposure cap (multiclass:regression ≤ 2, was 6–11×), eval-matched size law (≥ 40% of episodes with n_fit ≥ 256, was 4.6% ≥ 384); exact + HyperLogLog counters for tables, families, views, splits, committed exposures |
| `s3_view_builder.py` | No column loops: column subset / target swap → Rule A-1 regroup on the final view → split → prior-preserving class subsample → lattice-aware jitter → FIT-only scaler from one sort → synthetic columns → targets |
| `s3_grad_farm.py` | W single-thread gradient workers, shared-memory parameters, one flat gradient per worker, **one reduce per step**, global valid-count scaling and step/skip decision, parameter checksum every 200 steps; gloo variant for multi-node. `CUDA_VISIBLE_DEVICES=""` always; no GPU path |
| `s3_full_state.py` | Raw weights, AdamW, scheduler, EMAs, clip history, counters, identity (code, corpus, recipe, torch version); atomic + fsync; 600 s cadence + SIGTERM; restore refuses identity mismatch; async replica |
| `s3_speed_loop.py` | Leader loop, phase timings, liveness aborts |

Exit: sampler replay identical across processes and worker counts; 0 repeated
families and 0 cross-half duplicate rows in 2,000 steps (audit measured 21%
and 4.2%); exact resume (6 steps = 3 + restart + 3, sha256-equal); W=2
shared-memory = 2-process gloo; **measured** episodes/s per core on xbabe0,
worker efficiency ≥ 0.6, data wait < 3%, plus wall-clock to a fixed DEV-lite
TRG and arms × seeds per day. Working estimate, to be replaced by the
measurement [est]: 6 episodes/s per core for the 16M model and 14 for the 5.6M
one, i.e. ≈ 100–220 episodes/s on the 16 always-free cores (3.6–8× the old
4-GPU fleet rate, at zero ASR cost).

### M4 — Panels re-cut, contract freeze, promotion machinery

- Monotone re-cut → `s3-v251/manifest.jsonl` (new file, sha-committed) with
  DEV-SELECT / DEV-CONFIRM / SEALED (≥ 90 families); every old-val family stays
  out of train; old-vs-new comparisons only on LEGACY; a model is eligible for
  CONFIRM/SEALED only if trained from scratch on the new train split.
- Freeze `contracts/dope-kpi-v2.json` (statistic, roster, tiers, clip bounds,
  guards, decision rule, the 0.5 / 1 / 2 / 4 KiB byte grid with the 2 KiB
  headline) before any trained-model outcome is read.
- **Promotion rule**: PROMOTE iff the 2.5th percentile of ΔTRG > 0 on
  DEV-CONFIRM (10,000 family draws), Holm-adjusted p < 0.05, point Δ ≥
  max(0.01, measured noise SD), guards hold (each task Δ lower bound > −0.02;
  held-out MLP Δ ≥ 0; realism and privacy coverage Δ > −0.03; exact-copy
  excess ≤ 0.005; bytes ≤ 2,048), and for trained models ≥ 3 training seeds
  with the same sign. Outcomes PROMOTE / HOLD / REJECT. Then one ledgered
  SEALED look (hash-chained look ledger; panel declared burned after 3 looks).
- Champion registry seeded on day 1 with arms scored on identical cells: Rust
  Format 2, stat packet v0, untrained V6, V5 as a non-eligible
  `rows_at_sampler` reference. Rung decisions use DEV-SELECT only.
- Tiers (data × sample seeds): lite 1×1 (< 2 min per checkpoint), full 2×2,
  CONFIRM 3×2, SEALED 3×3. Target DEV-CONFIRM MDE ≤ 0.035; else add families
  or seeds before any promotion.

### M5 — Ladder: attribution, eval-only rungs, then trained models

1. **Attribution first** (tips' swap ladder / 2×2): real X + packet Y, packet X
   + real Y, n vs 4n rows — says whether marginals, dependence or the
   conditional carries the gap, and orders the rungs.
2. **Eval-only rungs** (no training; keep iff ΔTRG ≥ +0.01 with lower bound
   > 0, or C2ST −0.02, at equal bytes on DEV-SELECT): atoms/lattices on raw
   values · class-conditional factor means · dependence rank · main
   effects/pairs · quantisation profile · **teacher-distilled target**
   (3-fold cross-fitted teacher on FIT rows, λ ∈ {0, 0.25, 0.5}; the tips'
   largest retention lever) · capacity-ceiling arm (uncapped bytes).
3. **Model choice**: hybrid (network predicts zero-initialised corrections to
   packet fields + level choice + embedding prefix, so training starts at the
   packet's KPI) vs pure-neural V6/V7 on the same codec with finite-scalar
   quantised codes and nested dropout. Whichever is trained must beat the
   best packet rung at equal measured bytes.
4. **Then** the model-specific batched step (`v6_batched_{landmarks,glm,
   step}.py`: masks, one sort for quantiles/distinct counts, batched float32
   solves, one backward; equal to the per-episode reference to 1e-5 at B = 1)
   after Antigravity acks an interface freeze on the board.
5. **Trained input-feature rungs** (`se_features.py`, FIT half only, each with
   a query-poison leakage test): class tokens · dependence tokens / attention
   edge bias · type, lattice and functional-dependency descriptors ·
   evaluator-matched signed teacher coefficients · relative learner-risk
   landmarkers (3-fold cross-fitted stump, shallow tree, 1-NN, naive Bayes,
   ridge) · tree-region witnesses.
6. **Ablation protocol** for every trained arm: fixed budget (2 h × 16 cores),
   ≥ 3 seeds, same sampler plan for arm and control; KEEP if lower bound > 0,
   DROP if upper bound < +0.01, else INCONCLUSIVE (not adopted).

### M6 — Gradient quality, augmentation (after a model trains on the core)

Instrument first: gradient-noise scale (local vs global norms), per-term norms
and cosines, clipped-step share, per-task loss scales, frozen 256-episode
probe. Then one change per arm, in order: batch size set from the noise scale ·
evaluator-matched bounded surrogate (total λ = 1, converged IRLS, tree-proxy
learner) with per-task normalisation · antithetic/Sobol generator noise and
4× synthetic rows · per-micro-batch and per-episode bounding · EMA horizon
chosen on DEV (raw / fast / slow saved) · LR-batch scaling from a range test.
Targets: clipped-step share < 5% (was 33%), p99/median gradient norm < 4 (was
15), probe-loss variance ÷ 2. Augmentation ops (noise columns at 1–6 per
episode, derived columns, 0.2–2.5% IQR jitter with lattice exemption, warps,
MCAR/MAR missingness, target swap, SCM/tree prior tables) are ablated
leave-one-out under the M5 protocol; train-side targets decide, DEV TRG is a
non-inferiority guard.

### M7 — Remaining tips requirements with another owner

- **Antigravity (AG-2/AG-3)**: every matrix row it owns, each closed by a
  scoreboard measurement I run — Format 2 bugs R10–R18; marginals on raw
  values; `distill` emission; row-addressable Format 2 sampling; teacher
  capacity and a real poison test; CATF capacity and ablation arms; closed
  loop, refit on all rows, tiers with carry-forward; packet privacy (no names,
  levels, lo/hi; fail-closed backoff; canary); `embed` as a function of the
  packet; Rust port of the stat packet + Format 2 section; V6/V7 label path,
  InfoNCE pairs, batch-statistic-free decode; truth-record repairs; Jankurai
  on main; tags/releases.
- **You**: the v1 lane ruling after G0 failed; third-party baselines; idle
  cores on GPU hosts. v1 rows that stay valid regardless (cache identity
  enforcement, a powered DVAL-300/SEALED-300 panel, identity + strict loader +
  Rust parity before JopeDime may consume any new encoder) are listed in the
  matrix as P1/P2; the decision-supervised pretraining, 10M corpus and Candle
  bridge rows stay P3 until you rule.

## 7. ASR non-interference (CPU envelope)

All my processes: `taskset` to xbabe0 logical CPUs **32–47,96–111** (16 cores
no ASR queue uses), one thread each, `nice 19`, `ionice -c3`,
`systemd-run --user --scope -p CPUWeight=1 -p MemoryMax=40G`,
`CUDA_VISIBLE_DEVICES=""`. Logical 0–31 are used only while their SMT siblings
(ASR queues on 64–95) have been idle for 60 s, and released at the next
micro-batch when one wakes. Ramp W = 8 → 16 (→ 32 with the sibling guard).
Abort/back off if any ASR metric from the M0 baseline moves by more than 2%.
Nothing is installed into `/home/ubuntu/jsr*`; code deploys by rsync of a
committed tree to `xbabe0:~/dope-runtime-claude/<sha>/`.

## 8. Verification

| What | Command / test | Pass |
|---|---|---|
| Scoreboard units | `cargo test -p dope-arena` (new `tests/{panels_build,cellplan,score,compare,statpacket}.rs`) | green |
| Control self-test | `dope_kpi score --arm twin,copy_a,bootstrap_a,indep,label_shuffle` on LEGACY-lite | outcomes of M1(b) |
| Determinism | same checkpoint scored twice | byte-identical ledger and TRG |
| A/A null | `dope_kpi null --replicates 2000` | false-promote ≤ 5% |
| Denominators | scheduled = informative + uninformative + failed | exact |
| Packet | `unpack(pack(e)) == e`; lengths = byte table; `truncate` = prefix; chunk-invariant decode; canary absent | exact |
| Launch gate | `pytest research/levels/tests/test_s3_launch_gate.py` | ≥ 1 optimizer step per arch |
| Sampler / view | `pytest …/test_s3_counter_sampler.py test_s3_view_builder.py` | replay equal; 0 repeats; 0 cross-half duplicates |
| Resume | `pytest …/test_s3_full_state.py` | sha256-equal after restart; SIGKILL during publish leaves a valid pointer |
| Farm | `pytest …/test_s3_grad_farm.py` | W=1 vs W=2 rel. diff < 1e-6; shm = gloo |
| Batched step | `pytest …/test_v6_batched_step.py` | ≤ 1e-5 vs per-episode at B=1; padding invariant |
| Repo gates | `just fast` while editing; `just check` before any push (throttled) | green or "main red" stated in the PR |
| End to end | stat packet → `s3_emit`/Rust arm → `score` → `compare` vs Rust Format 2 → receipt | receipt reproduces from ledgers alone |
| ASR | baseline vs during ramp (duty, steps/s, CPU-queue job time) | within 2% |

## 9. Risks and fallbacks

- **Two writers, one board**: Antigravity wiped my block three times before
  acking at 19:43Z; I now write with compare-and-swap and only my block. Every
  milestone still has a no-patch fallback (auto-discovered binary, standalone
  speed loop, codec adapter in the emitter); I escalate to you rather than
  touch its work.
- **Stat packet does not beat Rust Format 2**: the receipt says so; the ladder
  then starts from Format 2 and the attribution step says where the gap is.
- **Panels too small for the MDE**: add seeds, then the re-cut, then target-swap
  multiclass views as a labelled secondary stratum; no promotion before the
  MDE target holds.
- **CPU throughput below estimate**: default to the 5.6M model (V5 showed the
  16M one bought nothing) and spend cores on arms × seeds.
- **Moving base**: every "verified" fact is re-checked at execution (HEAD,
  PR state, live runs) before it is relied on.
- **Scope**: M0–M2 are the cut line for a first proven result; M3 runs beside
  M2; M4–M6 are gated on the measured outputs of M1–M2.
