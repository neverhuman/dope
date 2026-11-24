# DOPE v3 — training, checkpoint selection and KPI report, pre-registered

Written 2026-11-23 ~11:20Z, before any v3 training run on real arena episodes and before any
DSE8 or V8 DEV score was read by me. Code: `claude/v3-wip` at `3b653a66` (= `claude/v3-major`
plus Antigravity `cbd75bb7`, plus my ABI-3 V8 stage, `s3_v8_feed.py` and `fit_unit_meta`).

## Owner constraints in force
- Spare capacity only, zero JSR impact (owner ~10:20Z). Hosts: xbabe6 only (no H53_S2 rank;
  JSR h52 labelers there are guarded at CPU 0.80-0.85). xbabe0-3, 5, 7, 8, 9: none.
- DEV-SELECT only. No CONFIRM, HOLDOUT, SEALED or FMGAP look of any kind (Watcher 10:47Z
  relaying the owner). Every v3 number is therefore a DEV estimate and is labelled so.

## Training (arm `v8-r1`)
- Trainer `research/levels/s3_v8_trainer.py` (Antigravity, defaults unchanged: d_model 128,
  4 encoder / 2 decoder layers, WSD schedule, warmup 200, lr 3e-4) fed by
  `research/levels/s3_v8_feed.py`: every episode a fresh group-respecting random split of a
  TRAIN-split `s3-v250` table (v3 preset `V3_SAMPLER`, max 48 view columns); the DSE8 packet,
  the receiver decoder inputs, the FIT-only column meta and the evidence bank come from that
  episode's FIT rows only, computed in parallel producer processes (`native_v8`, dope-ffi ABI 3).
  No per-table cache.
- Three DiLoCo replicas (seeds 1, 2, 3; `--sync-every 25`), 8 episodes per step, 12,000 steps,
  1 torch thread + 3 producers each, xbabe6 CPUs 18-29, SCHED_IDLE, nice 19, duty-cycled by
  `budget_guard.py` (host-mean CPU <= 0.60; backs off before any JSR guard target).
- EMA (0.999) checkpoints every 1,000 steps (`checkpoints/step_NNNNNN_ema.safetensors`).

## Arms scored (all source_free, <= 2,048 B, `dope_kpi emit-cli` + `score`)
- `dse7-step0`: v2.0.0 product step 0 (reference; same emitter build as the W8 screen).
- `v2.0.0`: the v2.0.0 release weights (V7.1 d160l3-s01 k6000 yh, sha256 d013eb0c...).
- `dse8-p0`: `dope_v8_emit fit --fit-meta` (analytic, no checkpoint).
- `dse8-p0-nometa`: `dope_v8_emit fit` without meta (what the product path does today).
- `v8-r1-kNNNN`: `dope_v8_emit fit --fit-meta --checkpoint C` + `sample --checkpoint C`.

## Selection (DEV only)
1. Screen: TRG on the 12 informative headline cells of `w8-dev-20261123/schedule-lite-dev.jsonl`
   (the W8 DEV-lite schedule) for every arm above; V8 candidates = replica w0's EMA checkpoints
   at multiples of 2,000 steps plus its final EMA.
2. Finalists: the best 2 V8 candidates by screening TRG, plus `dse8-p0` and `dse8-p0-nometa`.
3. Wide DEV: each finalist and both references on `w8-widedev-20261123/schedule-wide-dev.jsonl`
   (108 cells, same 22 DEV tables, data seeds 1-4).
4. v3 default encoder = the finalist with the best wide-DEV TRG, but a V8 checkpoint replaces the
   analytic `dse8-p0` only if its wide-DEV TRG exceeds `dse8-p0` by >= +0.010 with the paired
   family-bootstrap (task-stratified, 10,000 draws) 2.5 % bound > 0. Otherwise the v3 default is
   `dse8-p0` and the checkpoint ships as the opt-in neural tier with its DEV numbers.

## Report (README / white paper)
- A fresh DEV report schedule, planned now and not scored until the selection above is final:
  the same 22 DEV tables, data seeds 5-8 (never used for any selection). Report the chosen v3
  encoder, `v2.0.0` and `dse7-step0` on it: TRG, per task, paired family-bootstrap 95 % CI of
  v3 - v2.0.0, C2ST, privacy, bytes. Label: "DEV estimate (same 22 DEV tables as selection,
  fresh splits); no CONFIRM look taken under the owner's pause".
- A KPI upgrade is claimed only if the report CI of v3 - v2.0.0 excludes 0.

## Addendum A (written 11:39Z, before any wide-DEV or fresh-DEV score of any arm)
Read so far, all on DEV-lite only: the E2 arms, the v2.0.0/v1.3.0 references, the step-1000 smoke
checkpoint, `dope-v3-auto-v8` -0.185 and `dope-v3-v8-analytic` -0.641 (v2.0.0 auto -0.225). So the
DEV-lite numbers below are selection-exposed; the decision is taken on wide DEV and reported on the
fresh DEV schedule, as in the main text.
- Product candidates for the v3 default: `dope-v3-auto-v8` (Antigravity's 3-way guard, analytic,
  no weights) and, if a v8-r1 checkpoint passes the main-text rule, `auto-v8` with those weights.
  No guard parameter is tuned on DEV cells.
- Rule: the v3 default becomes `auto-v8` iff on wide DEV its TRG minus `dope-2.0.0-auto` is >= +0.010
  with the paired family-bootstrap (task-stratified, 10,000 draws) 2.5 % bound > 0. Else the v3
  default stays `--format auto` and `auto-v8` ships opt-in.
- The fresh-DEV report (data seeds 5-8, same 22 DEV tables) then gives the README numbers for the
  chosen default, `dope-2.0.0-auto`, `dope-2.0.0-small`, `dope-1.3.0-small`, `pr390-v7-step0`.

## Addendum B (written 13:25Z, before any score of code at 7968ea7f)
Antigravity pushed `7968ea7f` (DSE8 target/copula/tree calibration, 2-fold auto-v8 guard, V8 trainer
"r2 stability"), merged as `claude/v3-wip` `4b68275e`. Wide DEV has been read for every earlier arm, so it
is now a reused selection set; the fresh DEV report schedule (data seeds 5-8) is still unread.
- New candidates: `dope-v3b-auto-v8` (product `dope --format auto-v8`, analytic) and `dse8-p0-v3b`
  (`dope_v8_emit fit --fit-meta`), both built at `4b68275e`.
- Screen on DEV-lite, then decide on wide DEV with the addendum A rule: the v3 default becomes
  `auto-v8` iff its wide-DEV TRG minus `dope-2.0.0-auto` is >= +0.010 with the paired family-bootstrap
  2.5 % bound > 0 (`dope_kpi compare`, contract bootstrap). No parameter is tuned on DEV cells.
- Any V8 checkpoint trained from this code (v8-r2) is screened and decided by the main-text rule
  against `dse8-p0-v3b`, and enters auto-v8 only through the same addendum A rule.
- If a default passes, the fresh DEV report schedule is scored once for the README; else nothing is
  claimed.

## Addendum C (written 13:39Z, after the wide-DEV decision, before any report-schedule score)
Wide-DEV decision (addendum A/B rule): `dope-v3b-auto-v8` vs `dope-2.0.0-auto` +0.0368 [+0.0084, +0.0623],
Holm p 0.0094, so the rule passes and the v3.0.0 default is `--format auto-v8` (analytic, engine at
`4b68275e`). The contract decision of the same receipt is HOLD: guards task_ci_low (regression
-0.034), held-out MLP delta (-0.211, driven by one binary table: 2 of the 25 MLP cells) and privacy
coverage (-0.066) fail. That is recorded, not re-decided.
- The v3.0.0 selection is final now. v8-r2 cannot change the v3.0.0 default; it is judged by the
  main-text rule against `dse8-p0-v3b` for the opt-in neural tier only.
- Report look (once): the fresh DEV report schedule `v3-dev-report/schedule-report-dev.jsonl` (22 DEV
  tables, data seeds 5-8) for `dope-v3b-auto-v8`, `dope-2.0.0-auto`, `dope-2.0.0-small`,
  `dope-1.3.0-small`, `pr390-v7-step0`, plus `dse8-p0-v3b` (informative). `dope_kpi guards` for all,
  then `dope_kpi compare --candidate dope-v3b-auto-v8 --baseline dope-2.0.0-auto`. README numbers come
  from this receipt with its guard outcomes; an upgrade is claimed only if its CI excludes 0.
- The release binary must reproduce the scored packets: any later engine change is checked by
  re-emitting the report schedule and comparing packet sha256 to this look's `emit.jsonl`; if any
  differs, the change is reverted or the report is re-run and labelled a second look.

## Addendum D (written 15:42Z, before any training step of these arms)
Owner directive (2026-11-23 ~15:2xZ): show additional lift, starting from the previously trained model
by model surgery rather than from zero, on spare fleet capacity that does not touch JSR.
- Init: `research/levels/v8_surgery.py` at `02f39c7c` transplants the shipped V7.1 weights
  (`models/single-enc-v7/model.safetensors`, sha256 d013eb0c…) into SingleEncV8's receiver decoder:
  275,712 of its 367,019 parameters, with every residual head at zero, so the step-0 plan is exactly the
  analytic DSE8 packet. The owner encoder keeps the seeded init.
- Arms. Code `02f39c7c`, deploy `v3-02f39c7c`. Recipe identical to v8-r2 (8 episodes/step, grad
  accumulation 4, max-p curriculum 16 over 1,000 steps, lr 3e-4 wsd, warmup 200, DiLoCo sync every 25
  steps, outer lr 0.25, momentum 0.70) except init, replicas and length:
  - `v8-s7`: xbabe1 CPUs 32-47, init `v8-from-v7.safetensors` (sha256 3f1f4f68…), 4 replicas, seeds 21-24, 8,000 steps.
  - `v8-s7w`: xbabe3 CPUs 32-47, init `v8-from-v7-dw.safetensors` (sha256 e7692dd0…; adds V7's per-class
    weight head), 4 replicas, seeds 31-34, 8,000 steps.
  - From-scratch control: v8-r2 (xbabe6, 3 replicas, random init, 4,000 steps).
- Screens (information only, no decision): the w0 EMA checkpoint every 1,000 steps on DEV-lite with
  `dope_v8_emit-4b68275e` (packets identical to the `02f39c7c` emitter: only evidence row-identity code
  changed, and plain `extract` keeps its bytes). Surgery vs from-scratch is reported descriptively at
  matched steps 1,000-4,000. `s7w-step0` (the delta_W init before training) is screened once.
- Finalists: per arm, the two best w0 EMA checkpoints by DEV-lite TRG4x among k2000, k4000, k6000 and
  k8000. On wide DEV (108 cells, a reused selection set) each finalist is scored as
  `dope_v8_emit --checkpoint` and inside `dope fit --format auto-v8 --v8-weights`.
- Rules, Holm-adjusted over the finalists of both arms, paired family bootstrap 10,000 draws:
  (1) neural tier: checkpoint arm minus `dse8-p0-v3b` >= +0.010 with 2.5 % bound > 0;
  (2) additional lift: auto-v8 with weights minus `dope-v3b-auto-v8` (the v3.0.0 encoder) >= +0.010 with
  2.5 % bound > 0.
- If a finalist passes (2): one report look on fresh DEV splits (the 22 DEV tables, data seeds 9-12; the
  schedule is generated and hashed before any score) for that finalist's auto-v8, `dope-v3b-auto-v8` and
  `dope-2.0.0-auto`, with `dope_kpi guards`. Additional lift is claimed only if the 95 % CI of
  auto-v8+weights minus `dope-v3b-auto-v8` excludes 0. Otherwise nothing is claimed and the result is
  reported as measured. DEV-SELECT only: no CONFIRM, HOLDOUT, SEALED or FMGAP B/C.
- Deviation (written 15:45Z, before any step of `v8-s7w`): JSR started an eval lane and the
  always-on engine benchmark on xbabe3 at ~15:40Z, so `v8-s7w` runs on xbabe2 CPUs 16-31 instead of
  xbabe3. Init, seeds, recipe, steps, screens and rules are unchanged.
- Deviation 2 (written 16:49Z; `v8-s7` reached step ~55 and has no checkpoint or score): the memory
  watchdog ended `v8-s7` at 15:46:16Z when xbabe1 MemAvailable fell to 15 GiB (JSR wave1b and CI KVM
  guests; the watcher measured 9 GiB at 16:13Z). xbabe1 cannot hold a DOPE arm safely, and no other
  host has 16 spare CPUs under the 0.60 host-mean rule, so `v8-s7` is withdrawn. `v8-s7w` is the only
  surgery arm. Finalists, rules (1) and (2), the Holm family (now `v8-s7w` finalists only) and the report
  look are otherwise unchanged.
- Deviation 3 (written 19:23Z; stops a losing arm, selects nothing): `v8-s7w` was stopped at 19:23:04Z at
  step ~1,972 for futility. Its alpha gate fell from 0.10 to 0.03, its bilinear head is dead by construction
  (u = v = w = 0), its total loss rose from 1.42 to 1.77 after step 1,000, and Codex's paired DEV-lite
  screen of the w1 k1500 EMA gave REJECT, ΔTRG −0.182 [−0.398, −0.016]. That screen is information
  only: Addendum D screens w0 at 1,000-step milestones, and k1500 on w1 is off that grid. No `v8-s7w`
  checkpoint is a finalist. Checkpoints are kept. Addendum D now has no live surgery arm; the next
  surgery arms are pre-registered in Addendum E before launch.

## Addendum E: v3c neural arms and the single final look (written 2026-11-23T19:33Z, before any Addendum E step)

Reason: owner instruction (11-23 19:04Z) to finish v3.0.0 cleanly, acting as owner. Addendum D's
surgery arms are closed: v8-s7 was withdrawn, and v8-s7w was stopped for futility. Antigravity's
Track B (`v8-r3-surg-r2d128`, xbabe3) is the only arm with an open gate. Neural weights can ship
only as an opt-in `--v8-weights` asset, never as the default.

- **Code and recipe.** Commit `da813445` (`claude/v3c-surgery`), deploy `v3-da813445`
  (engine binaries identical to `v3-02f39c7c`: libdope_ffi `cde51998…`, dope_v8_emit `c318682b…`).
  - Every arm uses Track B's trainer flags exactly: lr 2.5e-4 wsd, warmup 150, backbone lr x0.5
    (warmup 300), anchor 0.02, alpha floor 0.25 for 1,000 steps (weight 0.05), resid-cond 0.35,
    8 episodes x 4 accumulation, p<=16 curriculum for 1,000 steps, DiLoCo sync every 25.
  - Each arm has 4 replicas and runs 8,000 steps.
- **Arm E1 `v8-a160z`** (launched now, on xbabe0 CPUs 24-39, seeds 501-504):
  - init `v7-to-v8` d160, 3 enc + 3 dec, `--zero-delta-w`, alpha 0.25, sha256 `7a327abf…`, from the
    corrected module (V7 col 38 -> 39; no global slot 3).
- **Arms E2/E3** (xbabe0 CPUs 40-55 and 88-103, seeds 601-604 and 701-704) are set by Antigravity's
  first posted paired DEV-lite receipt for Track B vs `dse8-p0-v3b`:
  - If its ΔTRG 2.5 % bound is > 0: E2 `v8-tb-s2` and E3 `v8-tb-s3`, the Track B init
    (`cd8b8247…`, `repair-v8` of r2 w0 k1500) with new seeds.
  - Otherwise: E2 `v8-a160w` (init `8650fe00…`, with V7's delta_W) and E3 `v8-rep-r2k3000`
    (`repair-v8` of r2 w0 k3000 EMA).
  - If no receipt is posted by 21:00Z, the second branch.
- **Screens** (information only): the w0 EMA every 1,000 steps on the frozen DEV-lite schedule
  `9207d157…` with the same scoring binaries.
  - **Futility:** an arm stops after two consecutive screens that are both below `dse8-p0-v3b` (−0.2776)
    by more than 0.03.
- **Finalists (11-24 10:00Z):** per arm, the best two of {k2000, k4000, k6000, k8000 if reached} by
  DEV-lite TRG4x.
  - Also eligible: Track B under the same rule (if Antigravity agrees) and v8-r2 (Addendum B: k2000, k4000).
- **Wide DEV** (the 108-cell reused selection set), Holm-adjusted over all finalists:
  - (1) checkpoint minus `dse8-p0-v3b` >= +0.010 with 2.5 % bound > 0;
  - (2) `auto-v8 --v8-weights` minus the served auto-v8 >= +0.010 with 2.5 % bound > 0.
  - An opt-in neural tier also needs **3 seeds of one recipe** with the same sign on (2). Only the Track B
    recipe can have 3 seeds (Track B, E2, E3 in the first branch). A single-seed arm can at most be TESTING.
- **The single final look** (supersedes Addendum D's look, and is shared with Antigravity's v3c guard
  fixes):
  - Data and timing: data seeds 9-12, all 22 DEV tables. The schedule is generated, hashed and posted
    at the release-candidate freeze (target 11-24 14:00Z) before any score.
  - Roster, in one Holm family:
    (a) RC auto-v8 vs `dope-2.0.0-auto`;
    (b) RC auto-v8 vs the served `b8a0037c` auto-v8;
    (c) only if a neural recipe passed (1), (2) and the 3-seed rule before the freeze:
        RC auto-v8 `--v8-weights` vs RC auto-v8.
  - Guards via `dope_kpi guards`, plus a 2,000-replicate null.
  - Claude scores it; Codex re-reduces it from the ledgers. The outcome (PROMOTE, HOLD or REJECT) is
    reported as measured.
  - Fallback: if the RC is not ready by 14:00Z, `b8a0037c` ships on its seeds 5-8 receipt (HOLD
    disclosed) and seeds 9-12 stay unspent.
- DEV-SELECT only. No CONFIRM, HOLDOUT, SEALED or FMGAP B/C.

### Deviation 4 (written 2026-11-23T20:41Z, before any E3 screen and before any probe step)

- **E3 `v8-rep-r2k3000` is stopped and its checkpoints are kept.**
  - Reason: the owner (Jepson, 20:30Z) asked whether much larger models would give more lift at the same encoding size.
  - E3 has the lowest prior of the three E arms. It uses Track B's recipe family, whose paired DEV-lite receipts against `dse8-p0-v3b` are k500 HOLD +0.010 [−0.025, +0.049], k1000 REJECT, and k1500/k2000 HOLD negative.
  - E3 itself has no screen, so stopping it uses no E3 outcome.
  - E1 and E2, the finalist rules, the futility rule and the single final look are unchanged.
- **Its CPUs (xbabe0 88-103, behind x0_gate2) run the capacity × channel probe P1-P4.**
  - The probe is **information only for v3.1**. It is not a finalist and is never scored on DEV wide or on seeds 9-12.
  - Design: zero-head fresh inits, so every arm starts exactly at the analytic packet; Track B recipe; no `--init-from`; 1,500 steps; one replica per arm; identical `--seed 801`, so the episode stream is shared and comparisons are paired.

    | Arm | `d_model` | Learned code (`--code-dim`) | CPUs |
    |---|---|---|---|
    | P1 | 128 | 64 (32 B, today) | 1 torch + 3 producers |
    | P2 | 128 | 1,024 (512 B) | 1 torch + 3 producers |
    | P3 | 256 | 64 | 2 torch + 2 producers |
    | P4 | 256 | 1,024 | 2 torch + 2 producers |

  - All arms use 4 encoder and 2 decoder layers.
- **Readout.**
  - Paired per-step training loss (`total`, `l_ret`, `l_direct_cond`) on the shared fresh-split episode stream, at the largest common step, in 250-step bins after the curriculum (step 1000).
  - Capacity effect = P3 − P1 and P4 − P2; channel effect = P2 − P1 and P4 − P3.
  - P1 and P3 (64-value code) may also get information-only DEV-lite screens through the native emitter. P2 and P4 cannot be emitted on the current wire.
- **Decision rule for v3.1, stated now.**
  - If neither effect lowers post-curriculum `total` by ≥ 0.02 at the common step, parameter count and code width are not the lever, and v3.1 work goes to the learning signal and gate.
  - Otherwise the next rung is a native wire code tier and a 20M → 100M ladder, with matched-byte native replay on every cell (Codex 20:38Z).

### Deviation 4b (written 2026-11-23T21:24Z, before any GPU probe step)

- **The owner dedicated xbabe1 (one RTX 4090 24 GB) to DOPE** (Jepson, about 21:12Z, relayed on the board). This adds compute; it changes no rule.
- **P1-P4 also run on that GPU, with every argument of deviation 4 unchanged:**
  - deploy `v3-49fb765e`, Track B recipe, zero-head fresh init, 1,500 steps, `--seed 801`;
  - plus `--device cuda`, in fp32 (no AMP);
  - **one arm at a time**, in the order P1, P2, P3, P4;
  - each with the same 20 producer CPUs (`0-19`) and 4 torch threads (`20-23`), so the four GPU arms share identical settings. On xbabe0 the producer counts differ (3 vs 2).
- **Readout of record: the GPU set.** It uses the readout and the v3.1 decision rule of deviation 4.
  - The xbabe0 CPU set keeps running as an independent replica and is reported when it completes, including any sign disagreement with the GPU set.
  - A GPU arm is never compared with a CPU arm.
- **Still information only for v3.1.** Never a finalist, never scored on wide DEV or on seeds 9-12, and no effect on E1/E2, the finalist rules or the final look.

### Addendum F (written 2026-11-23T22:10Z, before any cand6 build exists or is scored)

- **Why.** Jepson, about 22:05Z: "close any error gaps you are finding... finalize our major release with our MAJOR model update."
  - This reopens D1 (fallback, 21:26Z) for one bounded RC track.
  - The `dope-engine-v3.0.0` tag waits until the owner's 11-24 14:00Z time box.
  - PR #1 (closeout, docs, evidence; no engine change) still merges through the queue.
- **DEV evidence already seen (no new data).**
  - Wide DEV seeds 1-4 ledgers (schedule `97439e8a…`).
  - Codex's attribution, 22:05Z.
  - My exact route-composition screen: 12 task-dispatch composites of v3b / cand4 / v2, `route_compose.py`. The identity composite reproduces Codex's cand4-vs-v2 receipt exactly. Receipts: `screen-vs-v2.json` and `screen-vs-v3b.json`, Holm m = 12 and 11.
  - What it found:
    - v3b's MLP failure is binary table `0a64bb31` (both views, 2 informative MLP rows).
    - v3b's privacy failure is mainly multiclass table `48b63627`: 0/16 files pass vs cand4's 4/16.
    - cand4's regression route equals v2's exactly.
    - Best composite: regression and binary from cand4, multiclass from v3b. It scores +0.051 [+0.005, +0.125] vs v2, with privacy −0.035 (fails) and every other guard passing.
- **RC build cand6 (AG).**
  - Regression and binary take cand4's route.
  - Multiclass takes v3b's DSE8 route, kept only where cand4's FIT-only privacy vetoes pass; otherwise it falls back to cand4's choice.
  - Budget 2,048 B; everything else unchanged.
  - At most **3** builds under this addendum (cand6, cand6b, cand6c), all before **11-24 08:00Z**. Holm m equals the number of builds tried.
- **Eligibility for the single final look, judged on wide DEV seeds 1-4 only:**
  1. **vs `dope-2.0.0-auto`: PROMOTE.** All contract conditions pass: ci_low > 0, Holm p < 0.05, delta ≥ 0.01, task CI lower bound > −0.02, MLP ≥ 0, realism and privacy coverage > −0.03, exact-copy ≤ 0.005, bytes ≤ 2,048.
  2. **vs `dope-v3b-auto-v8` (= `b8a0037c`): point delta ≥ 0.**
  - This replaces D1's "full bar against both baselines". Reason: with 3-4 families per task, a significant lift over b8a0037c is out of reach on DEV.
  - The contract's promotion standard is defined against the previous release, v2.0.0, and b8a0037c itself is HOLD against it.
  - The change is stated here before any cand6 number exists.
- **If a build is eligible:** freeze it by 10:00Z. Run Addendum E's single final look on seeds 9-12: arms RC, `b8a0037c` and `dope-2.0.0-auto` plus controls, in one Holm family.
  - **The RC ships** if, on the look, it PROMOTEs vs `dope-2.0.0-auto` and its delta vs `b8a0037c` is ≥ 0.
  - **Otherwise `b8a0037c` ships**, and the look receipt is disclosed with it.
- **If nothing is eligible by 08:00Z:** `b8a0037c` ships (D1), and seeds 9-12 stay unspent.
- **Neural V8 stays out of v3.0.0.** No checkpoint beats the analytic packet. The conditional and marginal losses do not learn, and the gate settles near 0.05 (see the training study). Stage A (deviation 4b) continues as v3.1 research.
- DEV-SELECT only.

### Addendum F freeze (written 2026-11-23T22:48Z, before any seeds 9-12 score)

- **RC: `v3c-cand6b`**, source `bc92df0b1beae93254718e270e4e7ef7b05b38a3` (`origin/feat/v3c-cand6b`).
  - The look binary is Claude's source build `582ebfc17b3af23ec1f5c558cfcb4111ce9583e0955f2467766fcf41193e547b`.
  - On wide DEV it is packet-identical, 216/216, to builder binary `41f9efe8…` (Codex identity TSV `620bbd1d…`).
  - It is the only eligible Addendum F build, out of 2 tried: cand6 HOLD; cand6b PROMOTE, Holm m=2 p 0.0222, +0.0121 vs b8a0037c. Codex re-reduction `2cb5174f…`.
- **Look inputs** (`xbabe2:~/dope-data/kpi-runs/v3-final-look`):
  - schedule `beaa2a0fa3e2bcc44c37f23bf15adca0c2ff5b4adb5f212a198c99f3d23f71a2` (108 cells, data seeds 9-12, 22 DEV tables);
  - panel `d5bd12b0…`, dev tables `6fe72230…`;
  - contract `200ea0ed…` (embedded);
  - `dope_kpi` `ea855a89…` (3.0.0, `git_sha` `bc92df0b`; scores byte-identically to `c01fd919` on wide DEV).
  - The same recipe with seeds 5-8 reproduces the report schedule `14238f1d…` byte for byte.
- **Arms:**
  - `rc-auto-v8` = RC;
  - `served-auto-v8` = `b8a0037c` served build `dope-v3-3fb802fd` `58314f0a…`;
  - `dope-2.0.0-auto` = `0caec190…`;
  - native controls.
- **Holm family:** {rc vs 2.0.0, rc vs served}, m = 2.
- **Ship rule (unchanged from Addendum F):** the RC ships iff it PROMOTEs vs `dope-2.0.0-auto` on this look **and** its point delta vs `served-auto-v8` is ≥ 0. Otherwise `b8a0037c` ships, and this look's receipts are disclosed with it.
- **Reported with the decision (disclosure only; no decision role):**
  - the per-family decomposition of the delta;
  - the exact family sign-flip randomization p;
  - the contract's own null calibration;
  - the per-task deltas.

### Addendum F result (written 2026-11-23T23:03Z, after the look; no rule changed)

- Timeline on xbabe2 `~/dope-data/kpi-runs/v3-final-look` (from `look.log`): plan step (schedule and panel copied from the xbabe0 plan, hash check) 22:47:16Z; freeze posted on the board 22:48:04Z; realref 22:48:16Z; arms 22:48:17Z; first scored ledger 22:52:28Z; receipts 22:57:21Z. No arm was scored before the freeze. (Corrected 2026-11-23T23:56:43Z after Codex review: the first version said "scored 22:47-22:57Z", which conflated the plan step with scoring.) Receipts: primary `892a8ee0442aec76a442eb7bc28fe6951c5187a54ccc92158ed5d7e440c44c73`, family `2be57ae09637d8c786b565bd7f51c14b98ffd09b26d22e8c8c3ed2758ff18a6e`, compare-a `4ddabe59…`, compare-b `d59dfcfa…`. Codex re-reduced both comparisons independently (`93630510…`, `9b5f86ba…`).
- RC `bc92df0b` vs `dope-2.0.0-auto`: PROMOTE, +0.01648 [+0.00221, +0.03742], Holm (m=2) p 0.0234, all six guards pass.
- RC vs `served-auto-v8` (`b8a0037c`): −0.00195 [−0.04533, +0.04243] (1×: −0.0349), HOLD.
- **Ship rule: second condition fails (point < 0). `b8a0037c` ships as v3.0.0; this look's receipts are disclosed with it.**
- Disclosure-only items (as pre-stated): RC-vs-2.0.0 gain from 3 of 10 families (multiclass `4dac5905` 71 %), exact family sign-flip p 0.125, contract null flip-guard rate 0.137. Descriptive, outside the Holm family: served vs 2.0.0 on this look HOLD +0.0184 [−0.0282, +0.0669], realism −0.056 and task lower bound −0.138 fail (`served-vs-v2.json` `0610fb25…`).
