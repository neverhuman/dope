# D08 — Build bounded batched training and exact resume around frozen wire

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P1**. Prerequisites: D00, D05, D07, D10.

## Objective and non-goals

Engineer a bounded shape-batched, resumable trainer that D09's model/objective can use. D07 wire/basis/reference sampler must be frozen first; the trainer cannot invent another served graph. This task proves mechanics with small synthetic fixtures, not a real training campaign, 30× throughput promise, GPU qualification or model registration. Existing v3 trainer/checkpoints retain their original semantics.

## Ordered read-first files

1. Current `AGENTS.md`, claims/resource owners, `docs/{architecture,testing}.md`, owner/test maps, D00 contracts and D07 frozen registry/hash.
2. `research/levels/v3_ablation_and_eval.py` table/tier loop and accumulation; `research/levels/s3_multilevel_trainer.py`; `research/levels/train_landmark_dropout.py`.
3. `research/levels/tests/{test_multilevel_model,test_dope_mlc_v3,test_multiclass_and_dropout}.py`, current checkpoint/RNG/sampler resume tests and `research/levels/precalc/` source-role identities.
4. D05 input schema/groups, D07 reference sampler and D09 proposed train-step interface. Any JopeDime donor implementation is read-only until its actual path/license/owner is verified; no guessed kernel port is required.

## Exact scope and conflicts

Create `research/levels/v4_training/{__init__.py,batching.py,prefetch.py,state.py,trainer.py}` and `research/levels/tests/test_v4_training.py`. A dedicated frozen fixture can live in a new public synthetic v4 fixture area. Integrator handles exception/boundary/test-map registration if required. Do not change old checkpoints/trainers, source datasets, S3 stores, live workers, qualification locks, global thread settings or workspace manifests. D09 owns model/loss files; common interface changes require stopped-head handshake.


Python boundary integration: before adding proposed Python source/tests, read `PYTHON_EXCEPTIONS.md`, `docs/internal/PYTHON_EXCEPTIONS.md` and `agent/boundaries.toml`. Integrator updates the authoritative exception inventory and synchronized record as current rules require; change allowed roots only if genuinely necessary. These are shared, currently owner-sensitive paths: module workers submit proposed entries instead of racing edits. No Python enters the Rust product closure.

## Ordered implementation and difficult cases

1. Define proposed `TrainStep` interface and `Batch` with padded features/targets, row/column/class masks, physical-source/FIT-scope IDs, group map, native-report recipe/mask and packet-contract hash. Initially test with a tiny dummy module; no duplicate scientific model.
2. Implement deterministic two-dimensional row/expanded-column buckets. Bound batch padding and estimated allocation before collation. Mask padded rows/columns/classes in loss and solve; never select first64/256 silently. Keep dataset/family weighting explicit so bucketing does not change the objective by oversampling large/frequent tables.
3. Construct tensors once per batch and reuse resident inputs. CPU prefetch is bounded and can stop cleanly; optional pinned-memory ring/event machinery must define producer/consumer ownership and event lifetime. No unbounded queue, hidden source retry, per-view `.item()` synchronization or global Rayon/BLAS pool mutation.
4. Keep FP32 accumulation/Gram/solve where numerical stability requires it. AMP/BF16/fused optimizer paths are optional capabilities with explicit fallback/identity; failed/non-finite solve records a cell failure, never silently drops it. Batch and unbatched fixture gradients must agree within preregistered numeric tolerances.
5. Save `ResumeState`: model, optimizer/scaler/scheduler, Python/NumPy/Torch CPU/device RNG, bucket permutation and cursor, accumulation remainder, update count, input/contract/software identities. Publish atomically without replacing a prior valid checkpoint on partial write. Refuse mismatched recipe/cohort/graph unless explicit migration exists.
6. Test uninterrupted K updates versus stop/reload/continue on the same tiny fixture, including stop mid-accumulation and prefetch shutdown. Exactness expectations distinguish CPU deterministic path from optional nondeterministic device modes; do not broaden tolerance after seeing mismatches.
7. Add counters for tables/valid rows/cells, optimizer updates, padding fraction, wall spans, peak owned memory and failures. Warm/cold and setup/compute boundaries are separate. Hardware throughput measurement belongs to D11's separately admitted plan.

## Success criteria

- **D08-S1:** Batching respects memory/shape/group/role limits and produces correct masks without truncation.
- **D08-S2:** Batched and reference loss/gradients agree on public synthetic fixtures; no padded cell changes updates.
- **D08-S3:** Interrupted/resumed CPU run reaches the same declared state/update sequence as uninterrupted execution.
- **D08-S4:** Identity drift, partial checkpoint, queue interruption and non-finite math fail safely with retained state.
- **D08-S5:** Counters distinguish real optimizer updates, accumulation, discarded/failed work and setup; no unmeasured speedup is claimed.

## Validation and failure interpretation

Proposed focused command: `rtk proxy python3 -B -m unittest research.levels.tests.test_v4_training -v`. Existing owning tests: `rtk proxy python3 -B -m unittest research.levels.tests.test_multilevel_model research.levels.tests.test_multiclass_and_dropout -v`; boundary checker from current test map; required repository checks/review after capacity admission. Retain deterministic fixture seed, initial/final/resumed hashes, mask/gradient differences, counters and checkpoint failure cases. These bounded optimizer steps are conformance fixtures, not campaign evidence. Missing CUDA is NOT_RUN, not CPU equivalence proof.

## Stop, rollback and handoff

Stop on unfrozen wire, identity mismatch, unknown input exposure, memory admission failure, hidden dropped cells or changed old checkpoints. Disable new trainer and retain last valid checkpoint; never restart a scientific attempt from this task. Handoff exact HEAD, interface/contract hashes, resume proof, supported capability matrix and D09 integration requirements.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D08/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D08-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
