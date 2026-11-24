# DOPE 3.0 release closeout

Reviewed main `b8a0037c7dc635c50d8efeee49dbe036c8c51342` on 2026-11-23.
The [master plan](README.md), [canonical decision ledger](STATUS.md) and
[independent verification](verification/README.md) provide the detailed evidence.
`agent_chat.md` carries live ownership, host capacity and merge handoffs.

## Release decision

The current product candidate is the explicit `dope fit --format auto-v8 --budget 2048` analytic route.
Format 1 is the CLI default. The candidate passed the predeclared wide-DEV selection rule
and reproduced every scored packet in the fresh DEV report with the earlier candidate binary.
The report compares fresh row splits of the same DEV families. The product's
`dope-kpi/v2` result remains **HOLD**. Jepson approved tagging and publishing the
verified package at 19:42 UTC on 2026-11-23 and accepts the analytic result with
HOLD disclosed. The Path 2 deadline is 2026-11-24 14:00 UTC. If an admissible
neural finalist wins before the deadline, bind the release to its frozen
serving receipt and matching version. Otherwise ship the frozen analytic route
as an uncertified research-engine package. The single release PR is
`feat/v3c-promote-and-release`. Claude opens the paper PR after it merges.

| Requirement | Evidence on hand | Remaining decision or proof |
| --- | --- | --- |
| Product lift at the 2 KiB cap | Wide DEV seeds 1–4, ΔTRG +0.03678 [ +0.00839, +0.06234 ]. One fresh DEV report, seeds 5–8, ΔTRG +0.060 [ +0.009, +0.114 ] vs v2.0.0 `auto`, Holm p=0.0091, 44 informative cells and 11 families. 1,787 B median / 2,048 B max, zero failures. [Receipt](../../../bench/real-world/results/scoreboard/v3.0.0/compare.json) | Keep the frozen report and its scope. |
| Promotion guards | Task lower bound −0.053, held-out MLP Δ −0.007, realism coverage Δ −0.111 and empirical privacy coverage Δ −0.031 all failed. Exact-copy and bytes passed. | Antigravity owns v3c remediation. Claude freezes the decision design and Codex independently verifies the exact candidate. Jepson approved the tag and analytic HOLD disclosure. Certification requires its own passing evidence and authorization. |
| Native and supply-chain CI | All four required contexts are green on the exact source. Claude restored xbabe4 and the portable and shared-levels ARM64 jobs passed. [CI](https://github.com/neverhuman/dope/actions/runs/45235796572) · [Shared levels](https://github.com/neverhuman/dope/actions/runs/45235796816) · [Audit](https://github.com/neverhuman/dope/actions/runs/45235796756) | Recheck every required context on the final candidate and binding commit. |
| Signed delivery | Engine version is 3.0.0. #404 preserved the full `engine/` tree from `7810eed1`, whose signed-engine workflow passed its main-branch check. [Engine check](https://github.com/neverhuman/dope/actions/runs/45216307415) | Build and verify the exact approved tag's signed draft. Verify the staged archive as an isolated recipient and match the frozen report's 108 packet digests before publication. |
| Neural increment | r1 was rejected on wide DEV. r2 k1500 was HOLD despite a positive screen. The new s7w w1 k1500 and r2 k3000 receipts are in [STATUS](STATUS.md#dev-select-receipts). | The s7w w1 checkpoint was rejected and r2 k3000 stays under evaluation. A checkpoint enters a release PR only with a passing, replayable DEV decision for its exact serving route. |

## Publication record (v3.0.0)

`dope-engine-v3.0.0` was published on 2026-11-24 at 03:02:47 UTC as the Latest release. It is an uncertified
research-engine package. Its `dope-kpi/v2` outcome is HOLD.

| Item | Value |
| --- | --- |
| Source | `e15c415bad9554f7c419dd240aa670d1a79a27fe` (PR #406 squash), tree `901a9e170c52a7014fdef8894f995df68eaa0b1c`. `cpu`, `paper-clean`, `security` and `jankurai-audit` succeeded on it. |
| Annotated tag | `dope-engine-v3.0.0`, tag object `590386580fecfb7f95ff7d41149c86de4f18551d`, peels to the source above. |
| Signed build | [Signed Rust engine run 45313406130](https://github.com/neverhuman/dope/actions/runs/45313406130), Sigstore OIDC attestation against `engine-release.yml @ refs/tags/dope-engine-v3.0.0`. |
| Binary | `dope-linux-x86_64`, sha256 `8723955b1fcd59cab8996842a39b3a579c596c957ac1f89f291b0a9d1ebfaea0`. `BUILDINFO.json` records version 3.0.0, that source commit, `dirty_source: false` and rustc 1.88.0. Linux x86-64 GNU only, glibc 2.39 or newer. |
| SBOM | CI run 45308315817 artifact `dope-sbom.spdx.json`, SPDX-2.3, 840 packages, sha256 `9d7c35753a3191fa3100d0499c3058b3f970614baad1396b3d6726ad379f831f`. |
| Packet reproduction (G13) | Before publication, the signed binary re-emitted every scored packet of both DEV looks: the seeds 5–8 report arm `dope-v3b-auto-v8` and the seeds 9–12 look arm `served-auto-v8`. On each look, 216 of 216 emission keys matched in status, size and sha256. Receipts `76d43389…` and `b74bb290…` (`dope.v3-release-repro/v2`). Codex re-joined both ledgers independently. |
| Recipient install | `install.sh --from-dir`, the tagged `install.sh` and the README's `curl … main/scripts/install.sh` each verified the attestation and installed the same binary. `dope --version`, `dope doctor`, `fit --format auto-v8` and `sample` ran with only `HOME` set. |
| Rollback | `bash scripts/install.sh --tag dope-engine-v1.3.0 --source-digest cd35d9d0f0a772bea0e1ae2352eb8eb7d34ad09b --require-attestation --no-weights`. |
| Not done | JopeDime is not re-pinned (owner decision). No learned V8 weights ship; `--v8-weights` takes research SafeTensors only, and `active-rank-v1` checkpoints need a later engine. |

The successor is v3.1. It carries the Q4 large-model enabler, the `active-rank-v1` decoder and the 3.0.0 forward
guard. A learned V8 tier qualifies only if Addendum G of the selection pre-registration selects it and a fresh DEV look
confirms it. Public `dope-engine-v3.1.0` is published only when five gates pass: that look PROMOTEs, the signed
binary plus weights reproduce every scored packet, Sigstore verification passes, a clean recipient install passes,
and the release commit has all four required checks on main. Otherwise the enabler stays on main and v3.1.0 is HOLD.

## Tips traced to current main

All six v3 tip hashes matched the [source register](STATUS.md#source-register). The detailed
ledger has 2,662 indexed rows with an owner and a test route. It records 8 KEPT,
2 REJECTED, 111 TESTING and 2,541 NOT STARTED. The test route for an unfinished row
describes the required evidence. The 22 grouped proposals are a practical execution index.

| v3 tip in the source register | Rows | Landed or verified on current main | Remaining proof that can change a decision |
| --- | ---: | --- | --- |
| Catalog, `DOPE_Dataset_Features_and_Transfer_Engineering_Spec.md` | 721 | Declared target-last, stable physical row identity in the evidence ABI, FIT-only grouped splits and frozen evidence layout. | Family ablations for support/type, distribution shape, dependence, missingness and tree-visible target evidence. Admit individual descriptors only after family-level evidence. |
| RichStats, `DOPE_Dataset_Features_and_Transfer_Engineering_Spec (1).md` | 778 | Native value/z/mask evidence, cross-fit primitives and a bounded evidence inventory. | Demonstrate support and uncertainty behavior, missing-versus-zero handling, redundancy and consumed coordinates under the same downstream learner and compute. |
| DFP1, `DOPE_FEATURE_PACK_V1_SPEC.md` | 306 | FP-0 lattice/mask/group repairs have fixtures. FIT-only episodes, column metadata, V8 forward and acceptance-loop code landed. | Repaired-V7 paired control, measured V8 data wait and exposure, evidence-pack ablation, sender refinement and embedding admission. |
| Evidence, `DOPE_V8_EVIDENCE_SPEC.md` | 278 | DSE8 typed sections, evidence extraction, calibrated conditional path, E0 fixtures and native V8 sampling exist. | Native E0 attribution, matched-byte X/conditional/dependence effects, closed-loop evidence increment and full legality/realism/privacy table. |
| Runbook, `dope_v8_engineering_agent_execution_spec.md` | 386 | Native hard-forward parity fixtures, training lanes with checkpoints, held-out MLP guard and CI proof. | Exact-source real-view smoke, interruption/resume, factory throughput, seed agreement and product checkpoint qualification. |
| Compiler, `dope_v8_engineering_specification.md` | 193 | Grouped FIT selection, DPA1 tag 8 and product CLI paths, bounded native runtime and compatibility fixtures. | Equal-byte allocation, independent decoder/product parity, complete recipient package, matched analytic versus learned guards, byte frontier and export qualification. |

The older tips supply prior mechanisms and longer-horizon hypotheses. The release-relevant
pieces are routed below, with their exact filenames and hash prefixes for reproduction.

| Older tip | Release-relevant evidence and gap | Route |
| --- | --- | --- |
| `tips/v2/DOPE-FINAL-review-and-plan.md` `2ab08c3b` | Twin-referenced scoreboard, explicit failure denominator and complete byte accounting exist. Joint T90/D80 bracket, authentic expanded roster and challenge coverage are open. | Codex plus Claude, benchmark track |
| `tips/v2/DOPE_Consolidated_Final_Research_Engineering_Spec.md` `d74366a5` | Canonical normalization and bounded owner compilation exist. Cross-fitted functional signatures, moment correction and statistical workload still need independent value tests. | Claude, feature admission |
| `tips/v2/DOPE_FINAL_SPEC_R3.md` `d2372c89` | DSE8 provides typed row-local generation and a legal-byte rejection path. Tail and joint fidelity, driver shape and information/privacy limits still need matched validation. | Antigravity plus Codex, analytic ablations |
| `tips/v2/DOPE_Fidelity_Research_Engineering_Spec.md` `61775e96` | Native candidate paths and canonical sample contracts exist. Cross-fit functional sketches, exact family-level driver fidelity and the external roster remain open. | Claude plus Codex, guards and benchmark |
| `tips/v2/DOPE_UNIFIED_FINAL_SPEC.md` `205a00ed` | Useful teacher and rate-distortion hypotheses have historical pilot evidence. Repeat at identical budgets and independent family units before assigning a product claim. | Antigravity, bounded E0/E1–E3 |
| `tips/DOPE_10M_Pretraining_Engineering_Spec.md` `7c305815` | Consumer attribution, unique-instance accounting and compute-matched program supervision are separate research gates. | Owner, gated future 100k → 1M → 3M → 10M decision |
| `tips/DOPE_10M_Pretraining_and_JopeDime_Engineering_Spec_2026-11-17.md` `31589010` | Actual consumer utility, fixed reveal cost, streaming resume and program outcome labels need their own receipts. | Owner, separate consumer qualification |
| `tips/DOPE_MAX_VALUE_SPEC.md` `1ae9335f` | DOPE-JX and program-oracle tests require a matched equal-reveal consumer experiment and direct downstream program improvement. | Owner, gated consumer research |
| `tips/DOPE_VALUE_MAXIMIZATION_SPEC.md` `404eaaa5` | Rust delivery and frozen receipts carry forward. JopeDime release/pin changes require independent consumer ownership and evidence. | Owner, separate consumer qualification |

## Feature admission

**Owners** Antigravity builds and prices one candidate family at a time. Claude freezes
the DEV design and verifies the scorer. Codex independently checks evidence and records
the disposition in [STATUS](STATUS.md). Each row should link the hypothesis, input role,
representation, consumed fields, exact ablation, receipt and release consequence.

[F0 feature results](STATUS.md#input-feature-screen-f0) identify one candidate
for the conditional F1 ablation. The screen recomputed FIT-only evidence on
every random split. The F1 test measures matched-cost native transfer before
feature admission.

| Priority | Evidence family or design | Decision for this wave |
| --- | --- | --- |
| Product boundary | Stable FIT identities and grouped cross-fit, typed legality, declared targets, support masks, complete-byte caps, recipient-only sampling | Preserve these existing safety and correctness constraints. Recheck observable refusals and the packed result in every candidate. |
| First ablation wave | Supported lattice/atom/missingness summaries, OOF conditional calibration, shallow trees, rank/regime closure | Test the cheap typed group first, then matched conditional and dependence ablations. Start from the existing DSE8 features and allocate added bytes only where the paired evidence justifies them. |
| Conditional second wave | Sparse held-out residual witnesses, grouped permutation/fANOVA and class-factor evidence | Advance only where the E0 and first-wave receipts identify residual headroom in the same task stratum. |
| Deferred | Large banks of overlapping moments and bivariate coefficients, high-rank and kNN geometry, T3 closed-loop, skew/tail variants, neural flows, per-table refinement and temporal F133–F144 | Restore a proposal to the queue when a maintained cheaper feature fails on a predeclared use case or a native ablation identifies a missed mechanism. |
| Rejected directions in STATUS | Shared row-derived caches across splits, obsolete admission and positional tie-breaks, native runtime Python/ONNX/LibTorch, use of held-out MLP as a training student | Retain the measured rejection or replacement evidence. |

1. **Test the mechanism cheaply.** Review the native value, support mask and wire path.
   Use FIT-only queries with physical row identity and group complements. Require legal
   typed output, deterministic evidence and poison/refusal fixtures. Use planted XOR,
   conditional and missingness cases to check useful interaction signals. Univariate
   correlation is a screening diagnostic. A low marginal correlation can coexist with
   a valuable conditional or joint signal.
2. **Remove structural redundancy before training.** Resolve aliases, deterministic
   transformations, zero-byte packet-derived equivalents and proposals with no runtime
   consumer. Identify a distinct function for each retained coordinate. Choose the lowest
   byte and owner-cost version of equivalent signals. Keep boundary safeguards such as
   support refusals and duplicate grouping for their conformance purpose.
3. **Screen families in order.** Start with supported type/missingness, calibrated main
   effects and shallow conditional trees, then rank/regime or class-factor evidence.
   Use the native E0 feature-by-conditional factorial to measure where residual headroom
   lives. Broader scalar catalogs, high-rank descriptors, T3 closed-loop probes, learned
   coupling and test-time refinement must earn a separate matched-cost experiment.
   Claude's F0 screen ranks already extractable blocks by family-grouped out-of-fold
   incremental R² for predicting analytic step-zero episode loss, extraction cost and
   redundancy. Native transfer evidence governs admission. At most three F0 blocks
   reach a trained ablation after an arm beats step zero.
4. **Run a fair ablation.** Lock the baseline, candidate roster, physical families, task
   strata, FIT rows, seeds, support rules, auditor recipes and one 2,048 B cap before
   scoring. Compare packet-derived features alone against one FIT evidence family with
   the same receiver capacity and resource allowance. Include mask-only, within-task
   shuffle, equal-width noise and the best cheaper substitute. Record actual complete
   bytes, wall time, memory, failures, admissions and unused bytes. For optional packet
   sections use an equal-byte allocation or report the unmatched frontier explicitly.
5. **Down-select and prune.** Use 10,000 task-stratified paired family draws and the
   preregistered Holm family. Apply the existing rung-specific measurable keep/drop
   thresholds and all applicable guards. A candidate with a positive lower bound that
   misses the task, MLP, legality, realism, privacy or byte gate stays testing. Classify
   measured lack of meaningful incremental value as rejected when the rung's rejection
   condition is met. Ambiguous or thin strata stay testing. Record superseded designs
   with the evidence supporting their replacement. After a family passes, run its
   leave-one-family-out test and remove coordinates whose incremental value is redundant
   under the same resources and guards.
6. **Retain one public release account.** Record experiments in the existing ledger and
   keep detailed run receipts with the applicable evidence. Update the current
   receipt-bound scoreboard and white paper only when new evidence changes a release
   decision, an important guard, or a material public claim. The internal ledger carries
   routine negative results and the reasons to retire or defer features.

## Work order and gates

| Order | Owner | Task and evidence required | Exit |
| --- | --- | --- | --- |
| 1. Source and queue | Codex, watcher | Pin current main, capture open PRs, exact workflow attempts and candidate/contract hashes. Work in independent lanes. Record any differing tree and identify a single writer for shared files in `agent_chat.md`. | Verified main identity and zero open PRs at handoff. |
| 2. Restore required CPU | Claude and the watcher, Codex verifies | Claude brought back the xbabe4 ARM64 controller. The current portable job has a runner. After an expired or failed x64 partition, rerun the complete workflow so x64 receipts share attempt and head. Confirm shared-levels ARM64 plus green `cpu`, `paper-clean`, `security`, `jankurai-audit`. | Four green required contexts on exact admitted source, with the portable receipt. [Required run](https://github.com/neverhuman/dope/actions/runs/45235796572/job/137907511729) |
| 3. Measure guard closure | Antigravity, Claude and Codex | Implement the bounded v3c interventions below in one preparation lane. Use the existing wide DEV selection data for every comparison and a predeclared release candidate/roster. Freeze the accepted order and evidence before the one fresh DEV look. | Admit a finalist only with passing paired evidence and native conformance. Otherwise retain the analytic fallback with HOLD disclosed. |
| 4. Bind binary to reported rows | Codex | Prepare `just engine-release` evidence on a clean exact-head lane and record Rust toolchain, source, features, dependencies, lock, notices, command and binary digests. For the retained v3b candidate, use the existing seed 5–8 emission declaration for digest replay. For a selected v3c candidate, bind the package to the new frozen report emission. Verify ordered cell/budget packet hashes and failures from the applicable receipt. | Exact replay of the frozen report packets or a recorded discrepancy with source and repair evidence. |
| 5. Recipient conformance | Antigravity, Codex independent replay | Run the native wire and CLI fixtures against that package. Five of seven independent wire tests are ignored by default pending a digest-pinned emitter. Rebind the draft fixture framing before counting those tests. Cover 9/10 support, 48/49 slots, 8/9 classes, illegal one-hot/missingness, exact 2 KiB and lower-cap refusal, CRC-valid bad TLV, wrong/missing weight hash and target-last. Replay fixed-seed single/batch/chunk/resumed ranges in recipient isolation. Include DSE7, DSE1, DPK3.2 and v1 receipts. | Record actual counts, hashes and refusal codes. [Conformance checklist](verification/README.md#wire-and-compatibility-checks) |
| 6. Paper and claim checks | Codex with white-paper owner | Verify generated macros, README and hero checks, and rebuild the v3 paper separately. Check figure and PDF outputs against their receipt inputs and confirm DEV-only scope, complete bytes, comparator identity and the exact guard verdicts on public surfaces. The hosted `paper-clean` job covers the historical pipeline. The v3 renderer's `--check` verifies macros. | Rebuilt outputs and artifact inventory attached to the exact release source. |
| 7. Scope decision | Jepson | Review guard decision, x64 and ARM64 proof, package and discrepancy record, and the [Rust-only migration policy](../next-version/dope/language-policy/README.md). The new reporting script is nontraining Python and the repository policy remains NOT_COMPLIANT. Route a native port and parity/CI activation through the existing migration owners if it is needed for the approved release scope. | The recorded t404u authorization covers the tag after final evidence, a green main and the applicable selection decision. |
| 8. Stage and verify signed draft | Jepson authorizes, release owner executes, Codex verifies | After the recorded decision and green gates, allow the matching `dope-engine-v3.0.0` tag on admitted main. Check tag/source binding, trusted workflow/OIDC identity, attestation and inventory using `scripts/verify-engine-release.sh`. List every staged asset and recheck the delivered binary digest and packet replay. Verify a last signed passing manifest and restore command. | Codex verifies the draft, release notes and rollback record before publication. [Signed workflow](../../signed-engine-release.md) |
| 9. Publish once | Jepson, release owner | Publish the verified draft and its receipt-bound materials under t404u. Record commit, tag, package, lock, manifest, provenance, test report, release URL and rollback reference in the board and ledger. | Exact release artifact verified and no stale release PRs. |

### v3c guard closure

Treat each proposed change as a hypothesis and price it on the reused wide DEV selection
set. Preserve v3b as a control and include shipped `dope-2.0.0-auto` for the final product
claim. Use one source baseline and record each component's regressions and omissions.

| Guard | Proposed mechanism | Acceptance evidence |
| --- | --- | --- |
| Realism and empirical privacy coverage | Covariance shrinkage and stabilizing small-support regime estimates | Compare fixed typed marginals and conditional at matched bytes. Report regime support, joint C2ST and applicable privacy/duplicate metrics. Require both coverage deltas > −0.03 and exact-copy excess ≤ 0.005. |
| Held-out MLP | Preserve supported sparse and heavy-tail mass within the same packet budget | Inspect matched schedule row coverage, minority/task deltas, illegal values and complete packet cost. Retain the external held-out MLP with Δ ≥ 0. |
| Conditional calibration | Grouped out-of-fold temperature and stable row identities in supervised complements | The existing target fitter assigns folds by row position. Verify no duplicate group intersects both fold roles, reproducible folds under row permutation and stable FIT results when query values or labels change. Cover rare-class and unsupported-fit cases. Compare OOF calibration and task transfer at the same owner input. |
| Task robustness | Regression OOF selection margin in the grouped product route | Compare actual selected arm and routed packet on each matched task. Require every paired task lower bound > −0.02 while maintaining overall lift. |

Only advance a combined candidate after the ablations identify which mechanisms earn
their place. Require ΔTRG ≥ max(0.010, measured noise SD), positive paired 95% lower
bound, Holm p < 0.05, legal rows and 2,048 B complete bytes. Set the declaration and
Holm family before scoring. Report the four guard deltas and the applied failure floor
even if the result remains HOLD or REJECT. The candidate needs one fresh DEV report
after selection. If the product candidate and a learned finalist would use the same
fresh splits, agree a single frozen ordered roster and reporting decision beforehand.
Use any failed report as the recorded outcome and route further changes through a new
predeclared design.

**Serial PR rule.** Agents prepare and test in their own lanes. The integrator checks the
open PR list immediately before opening one release PR, rebases from fresh main and posts
its scope, source, owner, required checks and exact evidence. Run `just fast` during
edits and `just check` before commit or push. Watch named contexts, diagnose red from
logs, and repair on that same head. Merge after all four required contexts pass, with
ordinary permissions. Recheck main and sweep the completed lane before opening the next
PR. Record runner-only failures with evidence and retry the exact run where its proof
permits. A genuine release blocker gets the one available PR slot.

## Training and validation still available

| Track | Current position | Next bounded step | Advance condition |
| --- | --- | --- | --- |
| v8-r2 scratch control, Claude | Three replicas have k3000. k1000 and k1500 were HOLD, k2000 was weaker and k3000 is HOLD. xbabe6 guard has yielded to load. | Preserve all checkpoints and resume/health logs. Finish the declared k4000 only under the existing spare-CPU guard. Use the previously declared finalist rule with a wide paired DEV run against analytic `dse8-p0-v3b`. | Preregistered point lift ≥ +0.010 with positive 2.5% bound for opt-in consideration, plus the full contract guards for promotion. |
| v8-s7w V7 transplant, Claude | w1 k1500 REJECT. The board records a futility stop after gate collapse and higher loss as the curriculum ended. Checkpoints are retained. Original v8-s7 was withdrawn after the memory stop. | Review the stop receipt and the documented deviation. For a replacement arm, bind the main surgery code, open gate, live bilinear gradient and matched scratch control before training. The board targets 2026-11-24 10:00 UTC for any finalist. | On wide DEV, test both `dope_v8_emit --checkpoint` vs analytic and actual `auto-v8 --v8-weights` vs frozen `dope-v3b-auto-v8`. Each needs its paired estimate and declared Holm adjustment. An opt-in checkpoint needs served-path evidence. |
| Track A d160/L3, Antigravity | The board reports the stopped workers cleaned up on xbabe1. The pinned Rust emitter round trip and V7 backbone parity test support this architecture. | Confirm active work and safety with the watcher before any restart. Record exact init hash, source, probe receipts and matched scratch or narrower control before spending more compute. | Watcher's memory rule and runtime parity, then a measured guarded DEV increment at equal bytes. |
| Track B saddle/gate repair, Antigravity | Board at 19:12 UTC reports step 1,917 on xbabe3, k1500 on disk, bilinear norm 6.82 and gate 0.134. | Antigravity owns the paired k500/k1000/k1500/k2000 DEV-lite sweep. Bind checkpoint hash and matched r2 donor. Review regression, multiclass, MLP and failure metrics alongside aggregate TRG. | Positive paired wide-DEV result with matched donor, full guard evidence and declared training budget. |
| v8-a160z and successor seeds, Claude | The 19:34 UTC board records a d160/L3 restart on xbabe0 with seeds 501–504 and the guarded spare-CPU launch. | Review Track B's first paired receipt. The recorded follow-up chooses additional Track B seeds after a positive lower bound, or the d160 delta-W and r2 repair arms otherwise. | Only the frozen Addendum E roster and full served-path decision can qualify an opt-in artifact. |
| Training operations, Claude plus Antigravity | FIT-only identity/poison fixtures and prior-only launch evidence exist. Native V8 resume and exposure claims need direct evidence. | Record a real-view 20-step smoke, at least 60% real views, one-thread extraction at 1,024 rows with 24 and 48 slots, factory wait under 3%, atomic save and 200-step interrupted-versus-uninterrupted replay. Bind EMA, optimizer, RNG, sampler cursor, exposures and distributed state to the run. Log gradient clipping, p99/median norm, hard-forward parity and native TRG. | Match resume on the same backend. Check three independent training seeds, exact hard exports and task/legal/byte gates for a learned release. |
| Encoder capacity, Antigravity and Claude | The r2 EMA stores 942,843 F32 parameters. Its strongest paired DEV-lite checkpoint improved 4× TRG by +0.02694 [ +0.00949, +0.04570 ] at the analytic control’s 834 B median size, with HOLD on the task and held-out MLP guards. | Use native E0 and the complete-byte frontier to locate the bottleneck. If an equal-byte sender oracle has headroom, compare the repaired sender with a medium-capacity sender that fits the same analytic receiver and wire. Match the training exposure, record compute and latency, and rebuild each episode’s FIT features after its random split. | Evaluate a 100M-class sender or offline teacher after smaller scaling shows a repeatable native gain at the same 2,048 B cap. Use three training seeds and the existing task, MLP, realism, privacy, legality and byte gates. |
| Analytic and diagnostic ablations, Antigravity plus Claude | Calibrated DSE8 and grouped selection were measured end to end. Individual section attribution remains open. | Run native E0, then typed X, conditional, dependence and layout rungs at matched bytes. Evaluate no-trees/no-regimes, grouped drivers and T90/D80. Include full guards and explicit failures. | Keep only section or evidence-family changes that earn an incremental benefit with their applicability and costs recorded. |
| Wider benchmark and power, Codex plus Claude | Historical authentic SDV replay exists. Current wide DEV has 46 informative cells but only 3 binary, 4 multiclass and 4 regression family units. | Calculate paired power and sign-flip calibration from DEV, expand prospectively admissible DEV families, and qualify authentic CTGAN, ARF, TabSyn, TabDDPM and TabDiff individually. Pin recipes, closure, timeouts, independent CatBoost and measured bytes. Keep protected evaluation inputs under the custodian. | At least 80% simulated paired power for the claim and guards. Compute resources and coverage must match the claim. [Benchmark roster](verification/published-method-benchmark.md#next-public-dev-roster) |

Claude’s capacity probe starts from matching step-zero heads and a shared episode
stream. Current training sends 64 four-bit FSQ values into the decoder, or
32 B of code. The current native wire defines tiers through 128 B. The first
research comparison pairs `d128` with 32 B against `d256` with a proposed
512 B code. All four width-by-code combinations were staged at 1,500 planned
steps with the same seed on xbabe0. The JSR guard froze the workers at launch.
Matched-step loss contrasts separate width and channel effects in this training
probe. The wider code still requires native framing, parity and whole-file byte
validation. A flat early loss comparison leaves the capacity question open.
FIT-only inputs, per-episode extraction and native paired DEV tests govern
admission for a larger sender. [Native E0 evidence](STATUS.md#e0-attribution)

Claude is preparing Addendum E for one fresh DEV seeds 9–12 look, with one ordered
roster and Holm family shared by the v3c candidate and any admitted learned finalist.
Phasing stays on reused DEV until the exact baseline and declaration are hashed.
Current authorization covers DEV only. A protected test or GPU qualification for
certification is a separate owner-authorized event under the [release procedure](../../release.md).

### Evidence currently staged

- Verified both new paired comparison hashes, checkpoint hashes, the baseline row ledger,
  schedule and real reference. On both arms, 27 cells emitted successfully and 162 score
  entries passed. The one-core sequential runs stayed above 71 GiB sampled available RAM.
  See `agent_chat.md` at 19:08 UTC for complete hashes and receipt paths.
- Downloaded exact-head x64 CPU receipts and Jankurai evidence for `b8a0037c`. Verified
  the dedicated v3 macro check and the v3 scoreboard render check. `just fast` passed.
- Ran the independent native receipt reducer over the two checkpoint comparisons
  and the frozen public DEV primary and compare receipts. The frozen report's hashes
  match its schedule, reference, score rows, guard rows and guard summaries.
  Its reconstructed ΔTRG is +0.05991177 at 4× on 44 cells and +0.02389160 at 1×.
  The paired 4× point agrees under the unit failure floor and with unclipped results.
  This audit used retained inputs and made no additional scoring look.
- Independently replayed the archived static-pie candidate `3fb802fd` against the
  frozen DEV seed 5–8 emit ledger. All 108 packet digests and 216 emission keys
  matched. Packets ranged from 110 to 2,048 B. The binary SHA-256 is
  `58314f0a415544a9a80b36ff4d1368dd7bf828f23fc083d816994b53c2d72b3a`.
  Repeat the replay with the exact signed draft before publication.
- Built the clean `b8a0037c` reference package with Rust 1.88.0. The workspace
  gate reported 655 passing tests. The package contains one CPU Rust executable
  (SHA-256 `94b201406e6abadf8a37892f686bb26fc754c645423f01a7098dde1d625b8eb0`)
  and its checksum, build, dependency and license records. Strict inventory
  verification passed. Both analytic V8 and `auto-v8` passed the expanded smoke
  with an empty helper PATH. This unsigned reference executable independently
  reproduced the frozen DEV report's 108/108 packet digests and 216/216 emission
  keys. Maximum complete size was 2,048 B. The replay receipt SHA-256 is
  `27dc3ff3c6e2d061d4e4a4637e47061fcd20da3f91f5c6902d626ef783df6195`.
- Ran the five optional research-emitter conformance fixtures against a digest-pinned
  `b8a0037c` build of `dope_v8_emit`. Deterministic fit and first-row replay passed.
  CRC-valid malformed packet rejection also passed. The 48-slot, eight-class fixture
  fit into 2,046 B. The 49-slot, eight-class fixture fit at 2,928 B under a 4,096 B
  cap and exceeded the draft width limit. The 4 KiB fixture's 2,119 B legal base
  could not truncate to 2 KiB. The research emitter's unkeyed evidence changed when
  tied fingerprints were reordered. Stable row IDs have separate ABI-4 coverage.
  These three research-emitter limitations remain open. The test log SHA-256 is
  `657a20d7c46ada72e8eb554c1193dcd3e7473d76ef715a2931380a396c1608d8`.
- Ran the two focused model-surgery tests using the merged training source. The optional
  native round trip exercised the pinned `7bdd58f4` emitter from the existing deployment.
  The FP-0 and V8 loss fixtures passed all seven focused cases.
- Rebuilt the v3 PDF to the lane's `target/` directory. Its extracted text matches the
  committed PDF. The independently regenerated frontier figure also matches the
  committed figure pixel for pixel at 144 dpi. Rebind both outputs to the selected
  release receipts when the source or candidate changes.

### Handoff in the board

On every ownership or evidence change, read the active status and the newest entries in
`agent_chat.md`. Append the UTC timestamp, exact source or checkpoint hash, status,
evidence paths, limits, current owner and next dependency. Post a release blocker
immediately. Coordinate new compute against live spare CPU, the 16 GiB available-memory
floor and the existing JSR guard. Preserve one public outcome record after a decision.
