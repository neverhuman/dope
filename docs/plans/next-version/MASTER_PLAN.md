# Next version master plan for DOPE and JopeDime

Planning date: November 2, 2026, America/Denver. Source snapshots and coordination receipts use UTC, including November 3. Status: reviewable engineering plan; no implementation, experiment, release, or deployment performed.

Highest-value sequence: repair measurement, make DOPE affect JopeDime search through a reversible retrieval consumer, consolidate identities and landmark ownership, then build a measured normalize-once successor. Train a new model only after honest evaluation and affordable training exist. Artifact search and synthetic curricula follow demonstrated downstream value.

## Objective and boundaries

Objective: increase useful predictive/search value per byte and per unit of compute while reducing duplicated code and preserving reproducible historical results. Produce focused work packets a fresh implementation agent can execute without disturbing current workers.

Non-goals for this planning assignment: product source changes; switching existing checkouts; resolving or merging other owners' PRs; changing serving/model pins; launching experiments, catalog jobs, builds, CI or GPU work; releasing tags or weights; storage/credential changes; deleting lanes or evidence. No recommendation below grants those actions now.

Completion for this assignment: both repositories have isolated branches based on verified remote `main`; source findings distinguish observation from repository-reported results and proposals; ordered work packets include ownership, dependencies, validation, stop rules and handoff requirements; shared overview and repository-specific task files with individual success criteria exist in both planning lanes. “Next version” names this work program; it does not choose a product release number.

Assumptions: canonical product repositories are on xbabe2, where the requested paths resolve to `/home/ubuntu/JopeDime` and lowercase `/home/ubuntu/dope`. Local `/Users/bentaylor/Documents/ChatGPT/JopeDime` stores coordination/evidence and has no Git remote; local Dope checkout is stale. Remote repositories, not those local working copies, supplied the code audit.

## Source pins and isolation

| Item | Verified state at planning snapshot |
| --- | --- |
| DOPE remote `main` | `59b1e0513683fa4c5c347e4db6465eff84f496ef` |
| JopeDime remote `main` | `4548321f0f6c2baabbbec6c33a4a143fa8242800` |
| DOPE planning worktree | `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102` |
| JopeDime planning worktree | `/home/ubuntu/JopeDime/.agent/worktrees/codex-value-plan-20261102` |
| Branch in each repository | `codex/dope-jope-master-plan-20261102` |
| Plan package in each lane | `docs/plans/next-version/` |
| Remote setup and handoff receipts | `/home/ubuntu/jope_plan/evidence/codex-value-plan-20261102/` |
| Local review package | `/Users/bentaylor/Documents/ChatGPT/JopeDime/output/master-plan-20261102/` |

Worktrees were clean after creation. Canonical DOPE remained on `main` at `245d4f184ca5a0351bc28faf850b459038883ae8`, with its existing tracked modifications. Canonical JopeDime remained detached at `9483d5854fd9b2976feebc39924b7b8b6820b39a`, with existing coordination/cleanup modifications. Only required append-only coordination claims were added there under `flock`; no checkout, reset, index change or source edit occurred in either canonical tree.

Open JopeDime PR #2071, `v1/fastlane` → `main`, is an active master merge. DOPE had no open PR at the snapshot. JopeDime already exceeded its documented eight-lane ceiling before this assignment. The explicit request for isolated worktrees was applied narrowly to one planning-only lane per repository; this does not waive implementation queue, compute or concurrency rules. Existing lane registrations and `.claude/worktrees` symlinks were preserved. Reconcile capacity with the current integrator before implementation.

`DOPE:` below means a path relative to the pinned DOPE planning worktree. `JD:` means a path relative to the pinned JopeDime planning worktree. Paths marked **new** are proposed. Source pins are observations, not claims about later `main`.

## What current evidence changes

| Finding | Evidence and consequence |
| --- | --- |
| Evaluation repair is first priority | **Observed:** `DOPE:research/levels/v3_canonical_runner.py::evaluate_dataset_across_tiers` fits a transform on synthetic rows and applies it to real test rows. Fix this common coordinate-map error before using new retention numbers. |
| Per-learner scaling is not automatically leakage | **Observed:** conventional auditor pipelines have learner preprocessing. Preserve valid training-only preprocessing for declared raw baselines; do not delete every `StandardScaler` under a broad cleanup. |
| Tie handling needs a versioned successor | **Observed:** `DOPE:research/transfer/tdl/drivers.py::rank_agreement` returns perfect agreement for an uninformative all-tied case. Historical v1 goldens/readers must remain intact; introduce corrected metric semantics under a new protocol. |
| Engine gate already exists | **Observed:** `DOPE:ops/ci/engine.sh` formats, lints and tests `engine/`; CPU CI calls it. `just fast` covers the legacy root surface. Require the engine gate for engine changes; do not claim product CI is absent or duplicate it without checking routing. |
| DOPE currently does not seed search through the planned path | **Observed:** `JD:crates/jope-pipeline/src/run.rs` logs the skipped propose stage. Retrieval machinery exists in `JD:crates/jope-learning/src/retrieval.rs`. Consumer wiring is a stronger early investment than enabling an unused encoding by default. |
| Dependency direction constrains retrieval | **Observed:** `jope-learning` depends on `jope-pipeline`; CLI depends on both. A pipeline → learning call creates a cycle. Materialize retrieval through orchestration and inject pipeline seed contracts. |
| Current v3 preparation must stay frozen | **Observed:** released v3 uses its legacy transformations; neither a pure affine map nor a permissive parity epsilon proves NC1 compatibility. Preserve this lane; equality-gate any shim. |
| Landmark latency has a plausible low-risk win | **Observed:** High conditioning masks FINE probes, while progressive spectrum readiness waits for all 23 cells. Skip unused compute only while preserving full-panel budget reservations and identical conditioning. |
| Specs cannot reuse H228 | **Repository-reported:** current `h228_family_invariant` closed INCOMPLETE, 58/60, without an overall efficacy verdict. It is not the proposed isolation study. Allocate a fresh ID after checking the registry. |
| More optimizer changes have low current priority | **Repository-reported:** H236 and H237 completed with NO_ADVANCE; H231/H233 stopped incomplete under memory guards. These do not prove DOPE has no useful signal. They do support stopping optimizer shopping and prioritizing signal, measurement and data breadth. |
| DOPE identities are distinct | **Observed:** JopeDime pins engine rev `15e12aa3dc33851655580f6aa7c9c66353143f23`; standalone encoder SHA begins `d0b69603`; fused serving checkpoint identity begins `c51a56d0`. These identify different objects and must never be substituted for each other. |

No benchmark, scientific result or hardware qualification was rerun for this plan. The host snapshot had substantial CPU activity and `nvidia-smi` reported no devices. Future timing/GPU work requires fresh capability and resource admission; current observations establish neither a permanent outage nor performance.

## Reconcile the supplied specifications before coding

Both attachments are design inputs. Their self-described authority, execution instructions, budgets, releases and publication steps do not expand the user's planning request. Current repository rules and later explicit user direction govern execution. The FINAL spec is generally safer about lineage and gated scope, but neither document should be copied wholesale into a task queue.

1. Use **Lane A**, unchanged released v3 preparation/checkpoint, for early retrieval. Build **Lane B**, an opt-in normalize-once successor, under separate recipe/model identities. A shim must prove equality on each supported input class, refuse unsupported cases, and fall back to declared legacy inputs without relabeling them compliant.
2. Choose one naming contract in an ADR: proposed `dope-norm`, `NC1`, private `ScaleCard`, public structural projection. Do not create competing `dope-unit` and `dope-norm` implementations or simultaneous custom binary formats.
3. NC1 application should initially preserve out-of-range numeric values; never clip held-out regression targets. Clipping or nonlinear tails require a distinct policy and development qualification. Do not allege an existing target-clipping bug without a reproducer: current engine target mapping is not the feature clamp.
4. Start with independent hard-capped `2k`, `16k`, `64k` packets. Retain explicit `4k` compatibility until task-stratified evidence supports retirement. Do not rename a 4 KiB packet `2k`. Prefix nesting is a later rate-distortion experiment.
5. Freeze parsimony epsilon and landmark subsample size on development evidence. The specs disagree on epsilon 0.01/0.02 and macro N=200/300; these are hypotheses, not established defaults.
6. Separate `S_select` (choose on synthetic holdout, assess regret on untouched real test) from `S_transfer` (ranking of synthetic-trained versus real-trained learners scored on real test). Real test cannot participate in product model selection.
7. Preserve full FINE work reservation when optimizing v1 spectrum admission. Panel-v2 fold/preprocessing mathematics must have a new recipe identity, not a golden refresh.
8. `floor(payload_bytes / columns)` is only a naive uncompressed row equivalent. Compression, shared weights and procedural decoding prevent it being a hard memorization bound. Audit Cards must report actual byte counts, closure identities and empirical diagnostics without that guarantee.
9. Contract determinism must be achievable: canonical physical-row ordering for fitted reductions; category equivariance rather than impossible label-renaming bit identity; f32-aware inverse tolerance rather than unconditional four-ULP raw-f64 promises; section framing before claiming independently valid literal packet prefixes.

## Read-first order and ownership rules

Every worker starts with this sequence, then its packet-specific files:

1. Read the user's current assignment, `/Users/bentaylor/.codex/RTK.md` when working from this Mac, and current repository `AGENTS.md`. All shell commands use `rtk`; use `rtk proxy` when an unfiltered command is needed.
2. Check current remote `main`, open PR queue, registered worktrees, active claims and current resource/experiment ownership. Read latest relevant canonical `AGENT_CHAT.md` entries before claiming. An old timestamp or absent PID does not release another owner's paths.
3. JopeDime: read `MASTER_PLAN.md`, `docs/adr/ADR-0005-canonical-learning-loop.md`, `docs/learning.md`, `docs/parity/capability-matrix.md`, and current claim/experiment registry. Read `ops/AGENTS.md` before ops changes and any nested contract for actual scope.
4. DOPE: read `docs/architecture.md`, `docs/testing.md`, `docs/data-contract.md`, `docs/release.md`, `agent/owner-map.json`, `agent/test-map.json`. Read `contracts/AGENTS.md` or `ops/AGENTS.md` before touching those trees. Historical DPK3.2/v1 compatibility is a fixed boundary.
5. Read the packet's actual APIs and tests at the new source pin. Reconcile changes introduced by PR #2071 or other intervening work before editing. Use existing mechanisms rather than adding parallel registries or wrappers.

One integrator owns cross-cutting manifests, workspace files, lockfiles, CLI registration and ownership/test maps. Module workers may prepare disjoint patches but cannot race on those files. JopeDime permits one implementation PR at a time; no implementation lane opens while the existing queue requires its current owner's attention. DOPE's dirty canonical workspace/MLC files also require a stopped-head owner handoff before overlapping integration.

Forbidden without a new explicit assignment: canonical source/index mutation; other owners' worktrees or branches; `v1/fastlane`, PR #2071 or merger scripts; credentials and protection settings; live serving/learning pointers; sealed/meta-test data; GPU/job control; bulk S3 publication; cleanup. Generated/read-only zones include historical receipts and goldens, released model files, immutable contract inputs, scientific preregistrations, source datasets, and `target/health/`. Recompute derived artifacts only through their owning proof lane. Never blanket-delete `target/` or use `/tmp` for source, evidence or build output.

## Ordered implementation packets

Packet IDs below are new planning labels, not experimental wave IDs. Effort classes are relative sequencing aids, not deadlines or resource grants. Each packet lands through the repository's current reviewed PR lifecycle before its dependent packet adopts it.

### P0 — Preserve baseline and resolve contract decisions

Owner: one integrator; small. Dependencies: current merger/claim handoffs. Read first: isolation receipts, current identity guards, both supplied specs, current model manifests and experiment registry.

Likely edits: **new** `DOPE:docs/adr/normalize-once-and-dvs-v1.md`; **new** `JD:docs/coordination/dope-integration-contract.md`. Add a brief delivery pointer to `JD:MASTER_PLAN.md` only after claiming that live file. These documents decide the nine reconciliation points above and record an exposure ledger for dev/confirmation data. Select a fresh experimental ID without editing historical H228.

Steps: inventory current engine/model/preparation/conditioning/bank/evaluator hashes; name the existing legacy lane; declare unsupported cases and rollback; freeze FIT/TUNE/TEST roles and physical row/family identity; record which historical panels have been selection-exposed. Do not select epsilon, sample size, model or default from confirmation outcomes.

Gate: no unresolved ambiguity about private/public data, lineage, role separation, metric version or cap names. Stop if current owner directions conflict, a required raw source is unavailable, or an experiment identifier is already claimed. Missing provenance remains missing, not inferred from scaled values.

### P1 — Repair evaluation before producing new headline numbers

Owner: DOPE evaluation worker; small to medium. Depends on P0's protocol naming. Read first: `research/levels/v3_canonical_runner.py`, `research/levels/scaling.py`, `research/transfer/tdl/drivers.py`, `research/transfer/adapters/auditors/conventional.py`, `bench/real-world/auditors.py`, `docs/metric.md`, `engine/dope-core/src/eval/mod.rs`.

Edit/create: the live canonical runner's evaluator, **new** `research/levels/transfer_evaluation.py`, **new** focused evaluation tests under `research/levels/tests/`; a **new versioned** rank reducer/tests in the owning transfer metric area and matching `rust/metrics/drivers.rs`. Preserve existing v1/v2 semantics and use an explicit successor version. Update engine metric/CLI badge code only where a separate reproducer proves an actual semantic error; do not bundle speculative fixes.

Steps:

1. Capture a deterministic failing fixture with out-of-range real test features, mixed categories and regression tails. Record exact prepared-test digest for each arm.
2. Fit the common preparation once on declared real FIT, freeze it, then apply that same object to synthetic and real evaluation views. Synthetic rows already in that prepared space bypass a second application. Reject coordinate-space mismatches rather than guessing.
3. Keep learner-specific training-only pipeline transforms where part of the declared baseline. Keep raw-input and prepared-input comparator recipes distinct.
4. Add successor rank behavior for all-tied, constant-importance, empty, non-finite and partial-tie cases. Mark uninformative comparisons explicitly; do not silently score them perfect or change historical v1 expected bytes.
5. Give corrected results a new protocol identity; retain old receipts as historical diagnostics. Recompute a bounded development baseline with raw-row equal-byte control only after fixture gates pass and compute is admitted.

Code shape: `evaluate(prepared_fit, prepared_synth, prepared_test, protocol_id)` accepts identity-tagged views; it does not call `fit()` on the synthetic or test arm. Assertions bind FIT scope, common test digest and target role.

Validation: focused new Python tests; existing target-last tests; metric/golden compatibility tests; then owning `just check` and `just bench-check` requirements. Engine changes additionally require `just engine`. Retain reproducer-before/fix-after logs, protocol IDs, arm preprocessing identities and complete failed-cell denominators. Gate: known fixture fails before repair, passes after, and old receipt decoding remains unchanged. A new result cannot qualify an old protocol.

### P2 — Make existing DOPE embeddings affect search, behind an opt-in flag

Owner: JopeDime orchestration/retrieval worker; medium. May proceed alongside P1 after current JopeDime queue/handoffs permit. Does not depend on SDK, model re-pin, NC1 or retraining.

Read first: `crates/jope-learning/src/retrieval.rs`, `crates/jope-pipeline/src/{run.rs,seeds.rs,neural_search.rs}`, `crates/jope-cli/Cargo.toml`, CLI run orchestration, `crates/jope-encoders/src/dope_encoder.rs`, existing encoder tests and context-producer contracts.

Likely edits: `JD:crates/jope-learning/src/retrieval.rs`; `JD:crates/jope-cli/src/main.rs` and a **new** upper orchestration module; `JD:crates/jope-pipeline/src/cli.rs` seed-injection seam plus `run.rs`/`seeds.rs`; **new** CLI/pipeline integration test and versioned bank-context manifest. There is no existing `jope-cli/src/run.rs`: current `Command::Pipeline` delegates to `jope_pipeline::cli::dispatch`, whose `SearchArgs::options`/`run` construct the lower-layer request. Expose a data-only hook without importing retrieval into pipeline. Resolve decomposition again after #2071. Keep policy weights and accepted policy-context producer outside scope.

Steps:

1. Trace `retrieval::materialize`, `seeds::admit_v2` and production `resolve_seeding`. Current run path consumes v1. Add explicit version dispatch and carry admitted v2 ancestry/source-mixture/caps through search and receipts, retaining exact v1 replay. Existing modes are `Cold`, `Warm`, `ShuffledContext`; the latter rotates coordinates. Add separately named/versioned dataset-shuffle and random-eligible controls on the same eligible support. Do not relabel coordinate rotation as dataset shuffling.
2. Build bank contexts from permitted TRAIN families using identical released-v3 weights, legacy preparation, landmark mode `none`, vector slice and normalization. Bind bank/program/source/recipe/weight hashes. Exclude current dataset, physical source family and parent variants before nearest-neighbor selection; define tie order and distance behavior for zero/non-finite vectors.
3. In CLI orchestration, supply an implementation of a data-only seed-provider interface invoked by pipeline **after its existing partition**, receiving only that cut's TRAIN view. Encode/materialize there, then inject admitted `SeedSetV2` into the ordinary CPU search. Current global `resolve_seeding` runs before partition and cannot safely receive supervised DOPE seeds for every cut. Initial opt-in feature requires `cuts=1` and refuses larger values before encoding; multi-cut support later requires a per-cut seed/receipt map, never a union shared across cuts. Reuse the existing partition helper rather than reproducing splits in CLI. Do not add `jope-learning` to `jope-pipeline` dependencies. Preserve CPU `search::ELITES` cap (currently 4). Separate neural `MAX_INCUMBENTS=16` is a later explicitly scoped extension.
4. Flag off must preserve output seeds and behavior exactly. Flag on with missing weights, empty eligible bank or unsupported encoding must follow a declared bounded fallback with a receipt; never claim retrieval ran when zero valid contexts were used. Handle empty support before `materialize`, because `SeedSetV2::check` rejects empty draws; no empty success artifact.
5. Add receipts for requested/eligible/retrieved/accepted seeds, fallback reason, vector identity, time and actual generation-0 contribution. Preserve all other seed sources and their budget.

Code shape: `CLI -> encoder + retrieval::materialize -> SeedSetV2 -> pipeline`; never `pipeline -> jope-learning`.

Validation: encoder CPU tests; `jope-learning --test retrieval_exposure`; v1 replay/v2 seed-admission tests; CLI-to-generation-0 fixture proving a bank program actually appears; disabled-path exact replay; held-target poisoning leaves query vector and seeds unchanged; cross-cut feature refusal occurs before any encoding; family leak/empty-bank/invalid-vector/malformed-v2 tests; `cargo metadata` dependency graph inspection; owning repository checks. Acceptance is real search consumption, not a nonempty encoding log. Keep feature opt-in until P3 passes.

### P3 — Decide whether DOPE adds search value before policy investment

Owner: one registered study owner; medium, separately admitted compute. Depends on P1 where its evaluation is used and P2 for end-to-end search. Read all current H228/H231/H233/H236/H237 terminal records and existing isolation studies before choosing the smallest unanswered contrast.

Edits: **new** preregistration and runner/tests under a fresh `JD:research/flywheel_selfplay/<new_id>_dope_retrieval/`, plus owning index rows. No overwrite or retry of H228; no new GPU training in this packet.

Study A: matched baseline versus warm retrieval, shuffled context/dataset and random eligible programs. Report both equal evaluation-budget and equal wall-time comparisons, paired by dataset/family and seed. Charge bank construction separately and query encoding/retrieval inside product cost. Endpoints: final real-CV loss/utility, time-to-quality, harm tail, fallback/use counts and family-level uncertainty.

Study B, only if not already answered/owned: CPU ranker isolation separating program-only, true `Dope512`, shuffled `Dope512`, landmark-derived features and combinations. Preserve available information and model capacity across arms; label post-hoc contrasts exploratory. This study must not quietly replace Study A's end-to-end claim.

Pre-register margins, power/cohort, family bootstrap, missing/failure policy and stopping rules before outcomes. If powered positive effect and harm/cost gates pass, consider default CPU encoding with default `none` in a separate product PR. If null, harmful or incomplete, keep opt-in; do not switch to adapter training as an automatic retry. Study IDs and command syntax are allocated by the owning registry, not copied from either spec.

### P4 — Add a thin SDK and one compatibility source of truth

Owner: DOPE SDK worker, then JopeDime gateway worker; medium. Depends on P0 and stopped-head handoff for DOPE's dirty workspace/lockfile paths. Retrieval can continue on the current pin.

Read first: `DOPE:engine/dope-core/src/lib.rs`, public preparation/landmark/MLC APIs; `JD:crates/jope-encoders/src/dope_encoder.rs`, direct imports and current evaluator identity transition machinery; `JD:python/src/jope/north_star/dope_freeze_guard.py`, `python/src/jope/telemetry/dope_guard.py`, `crates/jope-telemetry/src/hooks/dope.rs`.

Edits: **new** `DOPE:engine/dope-sdk/{Cargo.toml,src/lib.rs,tests/surface.rs}` plus integrator-owned workspace/lockfile/maps; **new** `JD:models/dope-compat.json` and `crates/jope-encoders/src/dope_lock.rs`; gateway imports and `Cargo.lock` after the facade is available at a qualified immutable commit. First lock the current dependency/model identities in JopeDime without waiting for or adopting a new SDK. Use a single manifest filename, not both specs' alternatives.

Steps: facade v0 re-exports/wraps only the currently needed surface; no cache/session/scaling redesign in that PR. Record Rust engine revision, SDK contract, standalone model SHA, preparation recipe, landmark recipe/mask, vector layout and evaluator identity as distinct fields. Fused serving checkpoint identity is separate. Checker rejects drift; historical documentation/receipts are allowlisted as historical, not rewritten to current pins. Only `starforge` directly depends on SDK; transitive `dope-core` is expected.

Preserve `lock_dope_weights_and_eval_mode`, `reject_research_dope_unfreeze`, `snapshot_dope_weights`, `assert_optimizer_excludes_dope`, normalizer-buffer and frozen-anchor checks. Serving/flywheel must still reject `JOPE_RESEARCH_DOPE_UNFREEZE != 0`. An identity transition requires the existing transition receipt, not a new expected hash pasted into a failing test.

Validation: no-feature/CPU graph, SDK API fixture, old/new gateway exact vector parity, existing encoder/native tests, freeze guard tests, lock drift refusal. Stop on any preparation/vector/evaluator change; split behavior change into its own packet. Release/tag and re-pin are separate qualification steps, never implied by facade compilation.

### P5 — Reduce landmark cost without changing its meaning

Owner: DOPE landmark worker; medium. Depends on claims for `prepare`/landmark code. Integrator serializes shared workspace/maps with P4. First PR excludes new panel mathematics, defaults, cache overhaul and deletion.

Read first: `DOPE:engine/dope-core/src/landmark/{panel.rs,bundle_scope.rs,progressive.rs}`, `prepare/conditioning.rs`, existing resident/session and optional-landmark tests, `docs/optional-landmarks-qualification.md`.

Edit: one shared slot/group constant table; conditioning-requirement planner; progressive spectrum readiness/admission; focused regression fixtures. Move FINE computation off High conditioning while retaining full-panel work reservation and deterministic completion/order. Distinguish requested/not-needed/skipped/failed cells; preserve coverage semantics. Do not solve the readiness deadlock by fabricating successful probes.

Gate: exact High conditioning tensors (`vec256` and 64×17 columns), Dope512/629 vectors, admitted spectrum cells and sampling identities equal baseline across none/macro/full, budgets and row/column shapes. If bytes drift, stop identity-preserving claim and investigate. New mathematics receives a new recipe.

After parity: profile admitted CPU with fixed cold/warm boundaries and thread counts; compare N=200/300/full and `none` on corrected development evaluation by task. Choose product defaults only from cost/value results. Add content-addressed cache or resident sessions only if measured reuse justifies them; cache keys include source/FIT-role, preparation, recipe, mask, weights, seed, budget and implementation identity. Concurrent cache writes use atomic publication and bounded admission.

### P6 — Normalize once and rebuild trustworthy dataset views

Owner: one contract integrator, DOPE normalization worker, then JopeDime intake worker; large, split into focused PRs. Depends on P0 and P1. This is Lane B; v3 lane and live catalog remain unchanged.

Read first: `DOPE:docs/data-contract.md`, declared-kind intake/schema, legacy scaling; `JD:crates/jope-datasets/src/`, `JD:python/src/jope/intake/`, raw-source catalog manifests and split bindings. Exact current intake owners choose the existing entry points; no parallel intake pipeline.

Create: **new** `DOPE:engine/dope-norm/{Cargo.toml,src/lib.rs,src/fit.rs,src/apply.rs,src/card.rs,src/public_schema.rs,tests/contract.rs}` and matching contract schemas under the governed schema/contract area; additive JopeDime intake adapters and raw-source manifest. Keep the crate independent of neural/CUDA dependencies.

Steps:

1. Resolve explicit target and declared kinds; preserve source hashes, physical rows and target-last contract. Do not infer task or target from last column, cardinality or numeric values.
2. `fit(FitView, DeclaredSchema) -> (PrivateScaleCard, UnitSchema)` uses FIT rows only. Canonicalize reductions by physical-row ID. Handle finite extremes/overflow, constant/all-missing columns, missing indicators, high-cardinality cap and reserved-token collisions deterministically. Declared categorical groups always include unknown handling; positive-class semantics are explicit.
3. `apply(&PrivateScaleCard, Rows) -> UnitTable` never refits or reads query targets into features. Keep regression test tails unclipped. Private card holds raw names, vocabularies and extrema; public projection exposes only necessary shape/kind/group metadata. Public errors include safe codes/indices, no source headers, values or extrema.
4. Choose an existing binary container unless measured mmap/alignment needs justify a new one. Bind schema, FIT scope, transform, dataset/view and payload identities separately. Include f32 precision/tolerance rules and malformed/oversized-input refusal.
5. JopeDime inventory proves raw source and declared target provenance before creating a new content-addressed view. Suspected union scaling is a provenance investigation, not proof for every catalog entry. Missing raw sources are quarantined from clean confirmation; never attempt to reconstruct them from normalized values.
6. Publish additive catalog identities through the current curator after qualification. Preserve old datasets, split identities, live workers and historical comparability; no in-place catalog overwrite. Change serving/intake defaults only after emitter parity and adoption gates.

Validation: FIT-only poisoning fixture; disjoint TUNE/TEST mutation leaves fitted bytes unchanged; unseen/missing/constant/large-range fixtures; private-to-public canary scan; deterministic canonical-row fit; schema/role refusals; Python/Rust parity with explicit numeric tolerances. New catalog receipt reports measured recovered/excluded coverage and physical-source/view correspondence. Stop if required raw provenance or public/private separation cannot be established.

### P7 — Ship one trustworthy downstream value suite

Owner: DOPE evaluation/engine worker; medium. Depends on P1 and P0's metric definitions; can begin fixtures before P6 completes, using clearly tagged legacy versus NC1 views. Does not wait for final DML4.

Create: **new** `DOPE:engine/dope-core/src/dvs/{mod.rs,drivers.rs,selection.rs,receipt.rs}`, focused tests and a versioned receipt schema; minimal CLI command/report registration through integrator. Read `docs/metric.md` and current TDL reducers before implementing, preserve old readers.

Implement frozen auditor roster and exact FIT/TUNE/TEST procedure; R relative to declared baseline with explicit degenerate-denominator rules; grouped D according to the chosen versioned definition; both S metrics with ties/failures defined; complete payload and shared-decoder closure bytes; actual synthetic row count, CPU/device/time and failure denominators; empirical copy/DCR diagnostics. Models/scalers fitted for a specific arm cannot consume real test outcomes. One-hot groups permute coherently.

Audit Card uses receipt values directly and labels empirical claims. No DP/de-identification, byte-bound memorization or causal claim follows from payload size. Distinguish proposed targets from measured dev and untouched confirmation results. Establish exposure ledger before collecting any release score.

Validation: analytic ranking/tie/group fixtures, zero-lift and failed-arm fixtures, deterministic receipt replay, public canaries and CLI end-to-end. Release gate compares current release and raw-row sampling at equal complete bytes and declared compute; report task/family strata and paired uncertainty. Calibrate epsilon, macro N, synthetic-count policy and any S threshold on development only, freeze, then confirm once.

### P8 — Make v4 training affordable, then train what is shipped

Owner: trainer/performance worker then model worker; large, multiple PRs. Depends on P6/P7 and the P9 codec/wire/basis contract and reference sampler being frozen before objective/trainer implementation. Packet task files order D07 before D08/D09; final product qualification follows training. Read `DOPE:research/levels/{multilevel_model.py,v3_ablation_and_eval.py,strata_audit.py,multiclass_head.py}`, wire/sampler code, and candidate JopeDime HPC donors. Review donor licenses/attribution before copying; DOPE must not depend on private/product JopeDime crates.

First PR: versioned v4 trainer scaffold and CPU analytic tests. Fix class-specific multiclass residual (a scalar added to all logits cancels under softmax); demonstrate nonzero class-specific gradient and changed parameters. Carry every active gate consistently through training, encoding, wire and sampling. Cross-language parity alone cannot detect a shared omission; test fitted-basis versus decoded-basis semantics.

Second PR: profile and then batch by row/column shape, use bounded prefetch and local thread admission, FP32 Gram/solves where needed, explicit memory ownership and CUDA event lifetime, fewer host synchronizations, deterministic resume including optimizer/RNG/data-cursor state. One reused target/cache per lane. No promised 30× speedup without measurement; establish throughput, memory and numerical acceptance against the unbatched reference.

Third PR: actual-native landmark reports plus mask dropout; group-aware expanded-column encoder; synthesize-then-learn loss evaluated on disjoint real query rows; separate TSTR/driver/selection ablations. Train and validate through the same sampler/basis/gate contract. An update-count target such as 5,000 is a preregistered candidate, not success by itself. Actual optimizer updates, changed parameters and resume equivalence must be in receipts.

Gate: v4 beats/qualifies against current release and equal-byte raw rows under corrected DVS, with task-stratified harm bounds, matching sampler/wire semantics and declared compute. No automatic model registration, serving adoption or JopeDime policy retraining. Failure returns to a diagnosed defect/hypothesis, not repeated confirmation peeking.

### P9 — Productize hard caps and owner-side execution

Owner: DOPE codec/SDK worker then JopeDime emitter worker; large. Depends on P6/P7/P8 contracts; codec fixtures can be developed before training completes.

Edit/create: **new** versioned `DOPE:engine/dope-core/src/mlc_v4/` packer/wire/sampler tests, SDK entry points, and `JD:crates/feat-math/src/unit_stage.rs` plus emitter integration/tests. Verify exact `feat-math` source layout after #2071 before edits.

Implement deterministic packing that either meets complete cap or returns a safe typed budget refusal; never emit oversize success. Group-aware sampling respects mutually exclusive one-hot members. Keep old DML3/DPK readers intact. Evaluate independent packets first. Prefix nesting requires its own framing design and equal-total-byte non-inferiority; it is not a prerequisite.

Owner-side ScaleCard stage applies exactly the frozen transform to raw rows and reverses only declared target mapping. Keep private card outside public `.dope` artifact. Validate generated Rust and each supported language on ordinary, missing, unseen and out-of-range rows under language-specific tolerances. Code generated from NC1/synthetic training must run on raw owner rows with identical semantic predictions.

Gate: hard-cap refusal/success, corrupt/truncated/bounds/allocation tests, generator parity, legacy decoder replay and emitted-code parity. Historical byte tiers remain readable. No artifact-search launch before these gates and S assessment.

### P10 — Expand only after measured value

Priority order: JopeDime as a DOPE auditor; artifact search on small packets with owner-side validation/refit; diverse synthetic curriculum with real-CV confirmation; optional zero-init adapter and v4 context retraining if P3 shows useful context; nested packets/resident socket/broader kernel ports only when measurements justify them.

Use separate PRs and preregistered experiments. Synthetic tasks must not cross physical-family/role boundaries or replace real-data qualification. A zero-initialized adapter is a useful identity-at-step-zero property, not proof of value or authorization to change serving. Delete duplicate generic landmark/router paths only after the SDK gateway is the sole qualified consumer and a reference/call-site audit proves retirement safe. Freeze receipt-dependent legacy paths rather than deleting toward a line-count target. Do not collapse workspaces merely to satisfy a diagram.

## Dependencies and parallel execution

Critical path: P0 → P1 → P6/P7 → P9 wire/basis contract and reference sampler → P8 → P9 final product qualification → P10. The individual task indexes give executable order (D07 precedes D08/D09). Fast value path: current JopeDime queue/handoff → P2 → P3, using unchanged Lane A. P4/P5 improve maintenance and latency without blocking the first retrieval study. A null retrieval result stops that feature's default/policy investment; it does not by itself veto independently qualified DOPE compression or artifact-search work.

Safe parallel packets, once owners and compute agree:

| Packet pair | Disjoint write scopes | Shared integration rule |
| --- | --- | --- |
| P1 evaluation / P2 retrieval | Different repositories | Separate integrators; no extra JopeDime implementation PR while #2071 is active |
| P4 SDK / P5 landmark internals | New SDK crate versus existing landmark modules | One integrator serializes `engine/Cargo.toml`, lockfile, `lib.rs`, owner/test maps and shared docs |
| P6 normalization / P7 DVS | New normalization crate versus DVS module | Freeze interface first; one owner integrates schemas, workspace, CLI and maps |

Do not parallelize retrieval and adapter work on the same CLI/context plumbing. Do not assign multiple workers overlapping manifests, generated schemas, live delivery records, experiment indices or model/pin guards. Source review can run concurrently without write ownership. Concurrent builds/training are not implied by disjoint source files.

## Validation commands and failure interpretation

These commands are future worker checks, not tests performed for this plan. Run only after resource admission in the claimed lane. All Cargo runs explicitly set `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; keep `CARGO_TARGET_DIR` unset globally, using the lane's one reused target and a bounded job count chosen by the current owner. Do not run broad checks against an active canonical checkout.

DOPE existing checks from the DOPE lane:

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only just engine
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only just check
rtk just bench-check
rtk proxy python3 -B -m unittest research.transfer.tests.test_target_last research.transfer.tests.test_pilot_worker_inputs -v
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test resident_release_limits
```

JopeDime existing checks from the JopeDime lane, with actual test counts recorded:

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p starforge --features dope --test dope_encoding
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning --test retrieval_exposure
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-telemetry --lib hooks::dope
rtk proxy python3 -m pytest -q python/tests/test_north_star_flywheel_guards.py python/tests/test_telemetry_smoke.py
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo metadata --locked --format-version 1 --no-deps
```

Check current package features and test discovery after rebase. A filter matching zero tests is failure of validation, not a pass. New test files/commands become part of `agent/test-map.json` or JopeDime's existing test inventories through the owning integrator. New CLI examples in the supplied specs are target-state examples; verify `--help` and actual implementation before using them as acceptance commands.

For each source packet: focused reproducer and exact affected tests first; then full required repository checks once, at the frozen final head; refresh exact-head review/CI after any change. JopeDime's current ordinary workflow requires manual `ci.yml` at the branch head, both Rust shards, frontend and aggregate, substantive head-bound review and verified remote/main-guard. Historical emergency exceptions do not transfer to this plan. DOPE follows `docs/testing.md`/`docs/release.md` and actual required workflow routing. Full checks may wait for admitted capacity; do not relabel missing, failed, canceled, path-skipped or unavailable work as passing.

Interpretation: hash drift stops identity-preserving claims; new coordinate semantics require new identities; incomplete studies carry no global efficacy verdict; infrastructure failures retain failed cells and evidence; unavailable raw data blocks clean confirmation; an out-of-scope failing test is reported with its owner and impact, never silently waived. No broad reruns without a new change, failure or unresolved concern.

## Logging, stop conditions and final handoff

Before the first write and after resume, hold `flock` on canonical `AGENT_CHAT.md.lock`, reread claims, append checkout/branch/full base/owner/paths/intended verification, then release the append lock. Recheck branch, HEAD, dirty state and overlap immediately before mutation. Do not treat that append lock as long-lived checkout ownership. Refresh `main` in an isolated lane without rewriting shared branches or touching canonical dirty files.

Each worker stores durable receipts outside bulk source trees, with commands/raw logs linked from a small machine-readable summary: full source and comparison SHAs; branch/worktree; exact diff scope; prereg/exposure identity; data/FIT/family/view/recipe/model/protocol hashes; resource budget and actual use; test collected/executed/skipped/failed counts; requested/eligible/failed outcomes; output paths and SHA-256; observed results versus proposals; rollback; remaining limitations. Unknown cost/outcomes remain null, not zero. Status updates state what is source-implemented, fixture-proven, CI-verified, live-qualified or serving-adopted separately.

Stop dependent work when ownership overlaps, prerequisite identity changes, current main introduces conflicting APIs, raw provenance is missing, public artifacts expose private canaries, a required guard fails, compute exceeds admitted limits, or confirmation data is being reused for selection. Preserve evidence and hand the blocker to its owner. Do not change thresholds, goldens, credentials, protection or budgets to force a pass.

Final handoff includes stopped full HEAD and dirty files; focused diff/review; commands/exits and exact-head CI IDs; receipts/hashes and recovery route; open blockers; whether claims are retained or released; next packet and its prerequisites. A plan file is not implementation completion. A compiled feature is not demonstrated value. A qualified model is not serving adoption.

Planning lanes remain available for this user's review/follow-up. If their eventual PRs close, preserve and independently restore-verify unique source/evidence, then remove lanes through `git worktree remove` under current custody rules; no cleanup is performed under this planning assignment.
