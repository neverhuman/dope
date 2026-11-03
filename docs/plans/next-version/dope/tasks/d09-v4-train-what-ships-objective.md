# D09 — Implement v4 objective and model against the served graph

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P1**. Prerequisites: D00, D04, D05, D06, D07, D08.

## Objective and non-goals

Implement one additive v4 model/objective that generates synthetic training data, fits a learner on it and scores disjoint real query rows. Preserve a single basis/gate/sampler contract with D07. Fix the multiclass common-shift defect in the successor, not released v3. Bounded fixtures only: no v4 campaign, candidate adoption, promised useful-update count or serving-weight change.

## Exact read-first order

1. Repository `AGENTS.md`, current claims, `docs/{architecture,testing}.md`, maps and D00/D05/D06/D07 frozen contracts.
2. `research/levels/multilevel_model.py::forward_conditioning`, `_build_differentiable_basis`, `encode_table`, `generate_synthetic`; `research/levels/multiclass_head.py`.
3. `research/levels/{encoder,observed_conditioning,train_landmark_dropout,strata_audit}.py`; current gradient/frozen-weight tests under `research/levels/tests/`.
4. D07 complete gate registry, decoded basis/reference draws and source-free sampler; D08 batch/mask/resume APIs; D06 metric and role/inference contracts.

## Write scope and ownership

Create `research/levels/v4_model.py`, `research/levels/v4_objectives.py`, `research/levels/v4_encode.py` and `research/levels/tests/test_v4_objectives.py`. Trainer owner retains D08 files; codec owner retains D07 wire/sampler. Use their APIs and coordinate interface changes before edits. Do not change v3 weights/architecture/preparation/goldens, frozen reports, datasets, campaign preregistrations, live pins or private catalog. Integrator alone changes CLI/manifest/test-map registration.


Python boundary integration: before adding proposed Python source/tests, read `PYTHON_EXCEPTIONS.md`, `docs/internal/PYTHON_EXCEPTIONS.md` and `agent/boundaries.toml`. Integrator updates the authoritative exception inventory and synchronized record as current rules require; change allowed roots only if genuinely necessary. These are shared, currently owner-sensitive paths: module workers submit proposed entries instead of racing edits. No Python enters the Rust product closure.

## Ordered implementation and hard parts

1. Implement group-aware feature-set encoder with width/expansion limits from D00/D05, masks from D08 and exact task/class vocabulary. Permutation-equivariance tests cover columns/groups with inverse mapping; no first64 positional truncation or unsupported silent widening.
2. Multiclass residual emits class-specific logits. Test that different-class residuals change probabilities and produce nonzero useful gradients to the intended head. A scalar common shift should remain an explicit negative-control fixture with zero gradient, demonstrating why the old formula fails.
3. Consume actual-native report fixtures with recipe/mask identities and requested/not-observed distinction. Dropout masks condition inputs while supervision roles remain explicit; never invent full gains from heuristics or use real query labels to select reports/context. Dynamic native/source execution is outside this task.
4. Implement proposed `synthesize_then_learn_loss(fit, query, conditioning, draws, contract)`: generate synthetic x/y using the served basis/gates, fit a differentiable declared learner on synthetic rows, score held-out real query targets, and combine only predeclared loss terms. For discrete choices/quantization, document the differentiable approximation/estimator and separately test actual serialized inference; do not claim exact gradient through nondifferentiable wire operations.
5. Grouped driver loss follows D06 grouping; task losses, native landmark supervision and distribution terms have named weights frozen before any science. S evaluation is a downstream qualification metric, not a pretext to fit on real TEST. Each intended trainable gate needs finite, nonzero useful-gradient evidence on a designed fixture; inactive-by-task gates are explicitly classified.
6. Encode fixture model states to D07 packets and compare fitted basis against decoded basis with nondefault gates and coupled draws. A Python/Rust shared omission cannot pass. Check actual sampler outputs on tiny analytic tasks, not only proxy training loss.
7. Integrate D08 update/resume mechanics. Prove changed intended parameters and frozen unintended components, finite solves, task/class/width masks and exact CPU resume. Add model/graph/preparation/wire identities to checkpoints; old weights cannot load as v4 by shape alone.

## Success criteria

- **D09-S1:** Multiclass class-specific head changes probabilities and learns on a fixture; common-shift negative control does not.
- **D09-S2:** TSTR loss actually consumes synthetic rows and their learner; perturbing synthesis affects loss/gradients as predicted.
- **D09-S3:** Every intended gate/column module has task-appropriate finite useful-gradient evidence, with inactive cases declared.
- **D09-S4:** FIT/query roles, grouped masks and report identities remain isolated; poison query/context tests cannot alter fitting inputs.
- **D09-S5:** Encoded/decoded basis, actual sampler and training approximation discrepancy are measured and within frozen fixture limits.
- **D09-S6:** Bounded update/resume tests pass; v3 hashes/decoders unchanged; no actual trained-v4 quality claim appears.

## Validation and receipts

Proposed: `rtk proxy python3 -B -m unittest research.levels.tests.test_v4_objectives research.levels.tests.test_v4_training research.levels.tests.test_v4_wire -v`. Existing v3 and optional-landmark test suites remain; D07 native sampler tests run after admitted capacity, followed by current boundary/repository checks. Receipt includes analytic expected values, gradient matrix, changed/frozen parameter hashes, model/wire identities, sampler-vs-proxy differences and exact fixture update counts. Falling loss alone or all-zero gradients is failure, not training proof.

## Stop, rollback and handoff

Stop if loss never touches synthetic rows, common-shift head persists, gates are omitted, approximation masks harm or contract drift invalidates prior evidence. Roll back new opt-in integration; retain failed fixtures/checkpoints for diagnosis. Handoff full HEAD, objective/graph spec, compatibility limits and D11 readiness checklist. No campaign can begin from implementation completion alone.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D09/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D09-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
