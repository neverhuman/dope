# Decision register — v2 gap closure

One row per decision that gates work. "Tips" names the specification
recommendation the decision accepts or overrides (documents A–I as listed in
[`tips-gap-matrix.md`](tips-gap-matrix.md)).

| # | Decision | Answer | Decided by | Date (UTC) | Tips |
|---|---|---|---|---|---|
| D1 | How much may DOPE cost the ASR job? | **CPU-first.** DOPE trains and evaluates on idle CPU cores; a GPU is used only when no ASR process is on it. | owner | 2026-11-22 19:17 | Overrides the GPU budgets in H D-7 and I §12; consistent with F §16 (measure before optimising) |
| D2 | What counts as "the encoding" for the headline? | Zero-raw-row packet (global vector + per-column codes, shared decoder allowed); headline at **<= 2 KiB per table**; size-vs-fidelity curve at 0.5 / 1 / 2 / 4 KiB; fixed 64/128-D prefix is the embedding. | owner | 2026-11-22 19:17 | Overrides "claim at the Max tier" (G O-5, I D-4) and the 2/4/16/64 KiB grid; keeps E §20.5 byte accounting |
| D3 | Roles of the two agents | Claude: scoreboard, sealed panels, promotion receipts, stat packet and codec, CPU training core, exact resume, review. Antigravity: model architecture, Rust engine, dashboard, launches. | owner; acknowledged by Antigravity 19:43 | 2026-11-22 19:17 | H rule 1 / G rule 3 (one writer per checkout, one integrator) |
| D4 | Language of new code | Scoreboard and stat packet in Rust; Python only where PyTorch is unavoidable (checkpoint-to-rows emitter, training core, later a torch mirror of the packet decode). | plan default, approved with the plan | 2026-11-22 20:25 | Repository language policy (AGENTS.md) |
| D5 | Primary statistic for selection and promotion | Twin-referenced gap (TRG, see the plan §M1); the publishable claim stays the unclipped `dope-fidelity/v3` coverage shares printed beside it. | plan default, approved with the plan | 2026-11-22 20:25 | H §3, I §9 (no composite, twin reference); deviation: a bounded statistic is used for selection only |

## Open (work proceeds on the default)

| # | Decision | Default in force | Needed by |
|---|---|---|---|
| O1 | Re-cut of the corpus split | Monotone re-cut into a new manifest (only training-to-panel moves; about 29 binary and 40 multiclass families), decided after the retro-score gives the measured detectable effect | Before any trained model is scored on DEV-CONFIRM or SEALED |
| O2 | v1 / JopeDime lane after gate G0 failed (H470: CI [-0.009, +0.021], detectable effect 0.060 against a 0.01 target) | HOLD decision-supervised pretraining, the 10M corpus and the Candle bridge; keep cache-identity enforcement and a powered DVAL/SEALED panel as valid work | Before any v1 programme work is scheduled |
| O3 | Third-party baselines for the final claim (tips D-1: Python workers, licences, compute) | Deferred to the sealed run; until then controls and Rust Format 2 | Before the sealed confirmation |
| O4 | Idle CPU cores on GPU hosts (xbabe1–9) | Not used; xbabe0 logical CPUs 32–47 and 96–111 only | When CPU throughput on xbabe0 limits experiments |
| O5 | Review rule for pull requests from the second writer | Draft pull requests; Antigravity or the owner merges | First pull request |
