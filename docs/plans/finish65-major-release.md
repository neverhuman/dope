# DOPE completion and major release plan

Prepared for Jepson Taylor, Fleet Admiral super_captain, captain_dope, and the existing DOPE owners. Local date: 18 October 2026 MDT. Source fence: 19 October 2026 02:34 UTC, equivalent to 18 October 8:34 PM MDT. This is an execution proposal with explicit gates; it does not mark any unfinished task accepted.

The objective is to finish the five canonical tasks W01, E03, E04, E05 and E06, preserve the other 60 accepted tasks, merge the intended source and approved evidence into Git, pass every applicable release CI lane on the receiving release commit, and publish a signed major release with measured claims and a tested rollback. The critical path is official model qualification, development pilot, statistical and protocol freeze, sealed confirmation, independent reproduction, then final release qualification. Infrastructure repair, executable preparation, artifact packaging and publication preparation can advance alongside that path within disjoint ownership scopes.

## Completion criteria and scope

Success requires all of the following, each with a verifier, immutable receipt and exact source identity:

1. The canonical 65-task ledger reports 65 done, zero blocked and zero unresolved acceptance exceptions. Each of the five new completions meets its actual acceptance criteria. The original 60 receipts retain their original commit, command and log identities; current-source equivalence or fresh checks are separate evidence.
2. The complete intended source payload is on receiving main. Closed-unmerged unique work is either incorporated or explicitly retained/accounted with rationale; an empty queue does not prove delivery. No feature or safety requirement is silently dropped to increase the count.
3. Latest applicable CPU, paper-clean, security, ARM64, full Jankurai and admitted GPU/release gates pass for the release source and artifacts. Preserve historical failures and distinguish PR-head, test-merge checkout, receiving-main and release-binary identities.
4. xbabe1, xbabe2, xbabe3 and xbabe4 ARM64 guest each have actual independent CI evidence: commit, tree, command, environment, budget, exit status, logs and receipt. xbabe4 does not acquire CUDA or sealed-evaluator qualification from ARM portability.
5. README, active manuscript, first-page figure, supplements, downloadable PDFs and release notes agree with the final approved ledger. Every quantitative comparison is matched and traceable; losses, incomplete cells and negative results remain visible.
6. Release authorization, source/model rights, empirical privacy, formal-DP claims, legal claims, independent review, scientific certification and historical custody incidents are accounted on their separate axes. A passing source build cannot close any of these gates.
7. The allowlisted artifacts, SPDX SBOM, notices, checksums, source provenance, signed manifest/tag, verification instructions, monitoring and rollback drill pass. A clean fresh-user install exercises the actual staged artifacts.

E05 acceptance means 100 percent terminal accounting of the declared experiment, not 100 percent successful scientific outcomes. All 65 tasks may be honestly executed while a scientific release gate fails. In that case retain the negative outcome and keep release NO-GO. Do not rerun exposed confirmation data, weaken thresholds, rename an engineering milestone as scientific success or publish the requested certified major release prematurely. A differently scoped research release requires an explicit new release decision.

Non-goals: unrelated Admiralty/JopeDime/Joil/Jarambit source work, a new worker fleet, replacing independent review, reconstructing lost custody records, new spending or subscriptions, switching provider/model to evade quota, using xbabe0/Jeryu, and changing the agreed denominator of 65.

## Current source and findings

The fresh native baseline is main `001a8af11928e5fc03541682c30022b448488f37`, tree `44f4663e8bc52b2fa89b94083c5fb9aea24f7c45`, with zero open DOPE PRs. PR279 is merged after exact-head neverhumanbot review and the required source checks. Receiving-main CPU job117854704583 is still running on xbabe3; ARM on xbabe4, security on xbabe1, paper and audit on xbabe2 pass. Full audit is88 against85, caps0, two medium soft findings and zero high/critical. These observations are a fence to refresh, not a future completion assertion.

Canonical tasks revision102 has60 done and5 blocked. Final native refresh at02:52UTC retains the same main, zero open PRs and CPU still running within its original budget. SC-DEC126 at02:45:45UTC re-drove existing Codex2 and Claude on W01 adapter/lock and M001 receipt verification. Preserve those owners; Root takes the disjoint execution-bridge preparation after F65-00 and fresh exact path leases. The task meanings below are authoritative:

| Task | Actual deliverable | Completion evidence |
| --- | --- | --- |
| W01 | Qualify actual official Kumo and TabICL runtimes | Native Q0–Q9 receipts for required task/recipe variants, verified weights and environment, independent parity and rights/resource admission |
| E03 | Run the smallest actual observational and adverse-mechanism development pilot | Both task types, DOPE/copula/compressed-row controls, two budgets, qualified required models, full execution trace and cost report |
| E04 | Precommit one confirmatory design after development selection | Complete recursively resolved design and trusted independent signature before any sealed access |
| E05 | Execute the admitted sealed confirmation without tuning | Exactly declared schedule, durable access/debit history, every cell terminal, honest outcomes |
| E06 | Obtain independent reproduction at four distinct levels | Separate actual L1/L2/L3/L4 replay receipts from an eligible non-author executor |

The new readiness packet originally mislabeled E03–E06, mixed system-Python absence with certification-environment absence and contradicted its own TabICL cache boolean. Admiral SC-DEC125 supplied immutable corrective addenda; preserve originals and cite those addenda. The confirmed cache locator is `/home/ubuntu/.cache/huggingface/hub/models--jingang--TabICL`; presence alone proves neither the pinned revision nor complete, verified checkpoint bytes.

The current Kumo adapter and lock require NVIDIA `structured-data-models` / `sdm.models.KumoTabular`, not the relational `kumoai` API. NVIDIA's [official model card](https://huggingface.co/nvidia/Kumo-Tabular) provides local inference and identifies OpenMDW1.1 weights; the [official source](https://github.com/NVIDIA/structured-data-models) separates source and model-asset licensing. Therefore missing API credentials or buying KumoRFM service is not an established W01 dependency. Review the pinned terms and artifacts before acquisition/use; do not infer approval from public availability.

The official [TabICL checkpoint card](https://huggingface.co/jingang/TabICL) declares BSD-3-Clause. Its pinned code LICENSE also includes a separate forecast-code license. The pinned checkpoint repository's standalone `LICENSE` URL returned404 in this inspection; retain that fact, the model-card metadata and code license for the rights reviewer rather than inventing a missing file or silently expanding extras.

Further source gaps matter to execution:

- The actual contract filename is `contracts/tdl-contract-v2-candidate.json`; dotted filenames in older narratives are not locators.
- `research/transfer/precommit.py::validate_blocked_precommit` intentionally accepts only blocked skeletons, and `sealed_access_allowed` always returnsFalse. E04 needs a distinct, authenticated admission implementation, not a boolean toggle.
- `research/transfer/executor.py::ResearchExecutor` currently reports family exposure. `scheduler.py` maintains schedules and budgets. Neither executes a real scientific campaign. E03/E05 need an explicit orchestration bridge to genuine native workers and the protected evaluator.
- The model registry still contains fixture-only/unresolved entries beyond Kumo/TabICL. Audit the actual required panel for the pilot and confirmation. W03 engineering completion does not qualify every model in the registry.
- D19's old prose maps E03/E04 to older calibration/comparison names, while the canonical tasks and candidate contract use different definitions. Reconcile the decision before campaign launch; do not silently choose the convenient interpretation or treat a proposed100-table by5-seed schedule as already admitted.
- `docs/reproducibility.md` already exists. It correctly distinguishes historical L1 success from L2/L3/L4 gaps. Extend it for the new campaign; do not recreate or relabel its historical evidence.

## Authority and ownership

Jepson's latest direct instruction authorizes Root to implement DOPE work while keeping the Admiral aware. It supersedes the earlier root read-only source-lane restriction for this objective. It does not transfer occupied checkouts, admit spending, sign outside agreements, open sealed labels or resolve custody by assertion.

SC-DEC125 reserves the existing `/home/ubuntu/dope/.agent/worktrees/lane-durable-acceptance` for Root's next admitted packet. A fresh read found HEAD and tree matching M001 and no tracked/untracked changes; its existing branch is `paper/q01-figures`. Before writing, verify no active owner/process or new edits, obtain the exact path lease, and create a `codex/finish65-release` branch only if that exact branch is absent and the captain has admitted it. Creating a branch in this existing checkout is not creating a worktree. Do not reset, clean, detach or repurpose a busy branch.

| Role | Work and authority |
| --- | --- |
| Root Codex | Own this plan and the explicitly leased next implementation packet; can implement runtime qualification/orchestration/validators and repair ordinary code. Cannot independently approve its own head, sign for a custodian or reproduce its own compiler as an independent executor. |
| super_captain, screen4060655 | Decide internal compute sequencing, available existing-host reservations, owner handoffs and project governance within its written proxy authority; record decisions and resolve collisions. Never send input to ticker671167. |
| captain_dope, screen2716151 | Sole canonical tasks/progress/decision ledger writer and integrator. Establish one active integration PR, preserve proof history, merge via ordinary protection and harvest actual receiving-main results. |
| Existing Codex2 screen203893 | Retains current checkout ownership until explicit handoff; can take a disjoint ordinary packet after fresh checks. No blind root writes into its rt-L2 checkout. |
| Existing eligible Claude/reviewer | Perform independent scientific/protocol or exact-source review if qualified and not its author. Prior session name alone is not independence or reviewer eligibility. |
| Existing runner owner via Admiral | Diagnose/admit host and runner changes; retain original failures and actual host/configuration receipts. Root needs a separate host lease before changing runner configuration. |
| Separate authorized custodian | Own protected files, evaluator UID, trusted signatures and durable access ledger. Producer/model agents do not inherit this authority. |
| Historical C2 recovery owner | Sole archival recovery/cleanup executor if a concrete supplied archive and exact preservation plan qualify. Current availability and lease must be freshly established; no revival by assumption. |

One writer per checkout. Disjoint path lists in a single checkout do not permit two writers there. Parallel owners require different already-existing admitted checkouts and non-overlapping actual paths. Captain serializes changes to `agent/owner-map.json`, `agent/test-map.json`, registries, shared schemas and canonical ledgers.

Forbidden or separately owned zones: `/home/ubuntu/admiralty`, Admiralty deploys, other project checkouts, foreign MFS/canonical/live worker trees, `.agent/worktrees/integration` and its preserved assets, historical receipt originals, sealed/private source directories, archival supervisors/journals/locks, and any path without a current lease. Preserve ignored assets and staged indexes. No broad filesystem or transcript search for missing evidence.

## Read first in this order

All paths below are relative to the admitted DOPE checkout unless absolute. Read the current checkout's AGENTS and nested instructions before editing. Local copied source in this plan directory is pinned read-only planning input, not a live development checkout.

1. `/Users/bentaylor/.codex/RTK.md`; all shell entries use `rtk` or `rtk proxy`.
2. `/Users/bentaylor/Documents/ChatGPT/Dope/target/DOPE-custody-current.json`, its newest event and immutable incident pointers.
3. `/home/ubuntu/xbabe2_kicker/captains-20261017/`: current Admiral status, latest applicable `gb_sc_fix_*`, `ROOT_*`, `JEPSON_RULING_*`, delegation, SC-DEC125 and subsequent dispositions. Status prose is an unverified report; compare with native source and receipts.
4. Canonical packet `/home/ubuntu/dope/target/unified-spec-workplan-20261017/packet-v2-9e11fd166f5f/`: `tasks.json`, `WORKPLAN.md`, `DECISIONS.json`, `progress.json`, the source-requirement traceability and supplied exact receipt locators. Only captain edits these shared files.
5. `docs/architecture.md`, `docs/testing.md`, `docs/release.md`, `docs/audit.md`, `docs/decisions.md`, `docs/claims-policy.md`, `agent/owner-map.json`, `agent/test-map.json`, `agent/cost-budget.toml`, `rust-toolchain.toml` and current `.github/workflows/*`.
6. `research/transfer/adapters/auditors/{kumo_tabular,tabicl,conventional}.py`, `adapters/base.py`, `adapters/worker_abi.py`, `locks/{kumo-tabular,other-auditors}.json`, `registries/{models,auditors,generators,generator-panel,sources}.json`.
7. `contracts/tdl-contract-v2-candidate.json`, `protocols/{statistics-v1,statistics-v2-development,confirmatory-precommit-v1}.json`, `precommit.py`, `manifests/e03-pilot.json`, `pilot_admission.py`, `lineage.py`, `lineage_manifest.py`, `power.py`, `statistics.py`, `scheduler.py`, `executor.py`, `tdl/access.py` and matching tests.
8. `rust/evaluation.rs`, its included modules and custody integration tests; `rust/release.rs`, `rust/release/profile.rs`, `rust/certification.rs`, source-free bundle/charge modules and `docs/reproducibility.md`.
9. Current README, publication navigation, `paper/scripts/figure_one.py`, `paper/generated/hero-frontier.csv`, `docs/whitepaper/scripts/figure_one.py`, the actual active manuscript/build recipes, and their generated hash/ledger inputs. The two figure paths represent different published surfaces; inspect links before choosing one.

## Decision packet before protected work

The Admiral owns routine decisions and should produce a decision receipt with exact scope rather than bouncing every item to Jepson. The following decisions are dependencies, not automatic grants from this plan.

| Decision | Concrete record and default proposal | Who resolves it |
| --- | --- | --- |
| Model acquisition and use | Exact SDK revision, task checkpoints, license text hashes, dependency closure, existing/public-cache reuse, execution and redistribution scope; no API dependency for current KumoTabular adapter | Admiral/authorized rights owner within existing authority; Jepson only for new paid terms or outside legal commitment |
| Pilot host reservation | Existing host/device, CPU cpuset/thread caps, measured available memory, disk, bounded wall time, planned attempts and overlap with existing jobs | Admiral and runner/GPU owner; JopeDime priority remains |
| Scientific protocol | Canonical W01/E03/E04/E05/E06 mapping, D05 estimand/calibration and D19 multiplicity/freeze, required source/model roster and exact candidate/budget family | Qualified independent scientific reviewer plus Admiral governance decision |
| Confirmation eligibility | Actually unexposed source split, rights/use grants, producer UID, independent evaluator UID, trust anchors and access budget | Authorized separate-principal custodian; no producer self-signature |
| Reproduction | Named eligible non-author executor, permissions per level, environment and tolerances before observing results | Captain and independent reviewer/custodian |
| Major version and release profile | Propose v1.0.0 with an RC sequence; choose the actual compatibility promise, certification and privacy claim scope before final artifact freeze | Admiral/captain with user-reserved decisions escalated only when required |
| Historical custody | Exact supplied existing archive/relocation/restoration locator and preservation evidence, or a specific primary-human incident disposition if recovery is impossible | Historical recovery owner C2 and original incident authority; no fabricated replacement supervisor |

Do not place raw source rows, column names, extrema, private locators, API credentials or private signature keys in public planning/decision output. Presence booleans and custody-held keyed commitments are sufficient outside the private boundary.

## Work packets and ordering

Packet IDs F65-00 through F65-10 are execution groupings inside the existing65-task scope. They do not increase the denominator. Owner estimates are planning allowances, not measured campaign cost or delivery promises. Re-estimate at each measured gate.

### F65-00 Establish the executable source and ownership fence

Owner: Root in its reserved existing checkout, coordinated with captain. Captain alone updates canonical ledgers. Start immediately after the fresh exact path lease.

1. Record branch, clean status, HEAD/tree, parentage, current native PR queue, all applicable required checks, current speedctl value and source/task receipt pointers. Keep the existing CPU job running within its original60-minute quota; do not retry because it is slow. Inspect its actual result when terminal.
2. Correct the Kumo runtime identity in the readiness narrative to `structured-data-models`/NVIDIA KumoTabular, retaining the immutable original and addenda. Inspect the supplied exact TabICL snapshot paths/refs for the pinned revision and complete bytes; no broad cache crawl. Probe the actual admitted environment executable, module path and distribution metadata, not just `/usr/bin/python3`.
3. Repair the D19/task/contract-path mapping in `docs/decisions.md` and current planning references. Preserve the historical decisions and explain the successor interpretation. Keep `protocol_frozen=false` until all science inputs qualify. Reconcile task dependencies explicitly: W01 is an effective E03 prerequisite even where the older DAG lists only W03.
4. Captain aligns `progress.json` main fields, task titles, active leases and PR state with native evidence without rewriting the original60 proofs. Root supplies the exact diff/receipt to the captain.
5. Commit the adopted plan to proposed `docs/plans/finish65-major-release.md` through the single integration PR. Proposed `research/transfer/manifests/release-readiness-v1.json` records actual states and dependencies; a plan manifest is never a scientific admission.

Validation: mapping check against actual `tasks.json`; no five-ID/title mismatches; exact contract locator exists; `rtk git diff --check`; scoped documentation/manifest checks. Record the actual checkout and module paths for every test invocation. Definition of done: one trustworthy execution queue and fresh leases, no automatic scientific task promotion.

Estimated focused engineering allowance:1–2hours. The new model identity removes the unsupported KumoRFM-API assumption immediately; terms/native qualification remain.

### F65-01 Qualify official model runtimes for W01

Owner: existing Codex2 under SC-DEC126, with its exact path/checkout lease refreshed; existing eligible Claude or another already admitted non-author reviewer checks native parity. Root assists by read-only evidence review and takes W01 source only after an explicit owner handoff. Use one model environment at a time; preserve existing environments. Code scope:

- Existing `research/transfer/adapters/auditors/kumo_tabular.py`, `tabicl.py`, `conventional.py` only for demonstrated parity defects.
- Existing `locks/kumo-tabular.json`, `locks/other-auditors.json` and selected `registries/models.json` entries, serialized with captain.
- Proposed `research/transfer/native_qualification.py`, `research/transfer/tests/test_native_qualification.py`, and a bounded native-run entry `research/transfer/run_native_qualification.py`.
- Public qualification summaries under proposed `research/transfer/qualifications/<edition>/`; raw/environment/measurement receipts under `target/<campaign>/w01/` and the existing admitted immutable evidence store.

Pinned initial identities:

| Model | Code pin | Weight pin and task files |
| --- | --- | --- |
| KumoTabular | NVIDIA/structured-data-models `d50589692dbbbad4852eb1d9627f3b40c2673b63` | nvidia/Kumo-Tabular `3520742f7f3b937a2751266cea22d7fb6686882c`; required size/task combination from the admitted recipe |
| TabICLv2 | soda-inria/tabicl `10719e21fd72b8945d4a4f9ae077e350ba8650d9`, package2.0.3 | jingang/TabICL `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`; classifier-v2-20260212 and regressor-v2-20260212 |

For Kumo large, the lock names classifier854,850,034bytes/SHA `cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613` and regressor862,909,259bytes/SHA `745e0e5d97e4256d8a24cb61c2eecbd1128da6c83938c196b7589ea6f62e7421`. TabICL classifier110,368,038bytes/SHA `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` and regressor114,324,594bytes/SHA `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`. These are expected upstream metadata pins until local bytes are actually hashed. Do not fetch every model size or obsolete checkpoint unnecessarily.

Ordered qualification:

1. Rights owner records exact permitted use/distribution and notices. Reuse an existing admitted matching snapshot when available. If acquisition is admitted, download only exact allowlisted files at the pinned revision into an admitted acquisition cache, verify every byte, then execute with network and auto-download denied. No Kumo API calls or new external data transmission.
2. Build an isolated, fully pinned dependency closure for Python3.11+ and the model requirements, including native wheels/libraries. Record Python executable, `module.__file__`, source file hashes, package versions, GPU/CUDA/driver, OS/architecture and lock digest. No global installs/auth switches or silently replacing the fleet's Torch.
3. Run genuine native models with no injected factory. Compare the adapter and an independent official reference configured to the same recipe. Official example defaults may differ from the adapter: Kumo large/16 estimators is not qualified by a small/8-estimator run. A smaller first resource probe stays a separate identity.
4. Fill the Q0–Q9 matrix below for regression and binary classification separately. Missing task/head/checkpoint/terms or OOM is an explicit blocker, never a skipped pass.
5. Fix only measured adapter defects, preserving the old recipe identity where behavior remains unchanged and creating a new version when it changes. Re-run the affected narrow parity before any expensive suite.
6. Independent reviewer validates raw parity and resource receipts, including preprocessing and returned units. Only then register qualified task/recipe variants. Model/role eligibility remains explicit; a teacher used for selection is not later reported as an unseen holdout family.

| Gate | Required observation |
| --- | --- |
| Q0 | Exact source, package, checkpoint, closure, recipe, rights and role identities resolve |
| Q1 | Genuine official architecture and verified checkpoint load, strict tensor assignment; no random fallback, unsafe model deserialization or fake factory |
| Q2 | Independent official-reference parity on the admitted minimal example, with tolerance declared before outputs |
| Q3 | Correct task, class order and output shape; raw regression units; Kumo q001–q999 finite monotone grid and declared median/grid-mean rule |
| Q4 | Fit-only transformations and query-label isolation; no final labels in context, caches or worker requests |
| Q5 | Record native order/batch/transductive behavior; adapter follows that declared behavior rather than falsely promising invariance |
| Q6 | Identical frozen preprocessing/context/recipe/resource policy across real/null/synthetic arms; actual consumed rows and all internal attempts |
| Q7 | Deterministic or predeclared-tolerance replay with complete environment and seed pins |
| Q8 | Actual wall/CPU time, memory/VRAM method, row counts and attempts; finite caps enforced without claiming an unmeasured zero |
| Q9 | Independent admission review and machine-readable qualified/incomplete/refused result bound to all prior gates |

Narrow existing checks:

```bash
rtk proxy python3 -B -m unittest research.transfer.tests.test_kumo_tabular research.transfer.tests.test_other_auditors -v
```

New proposed entry after implementation:

```bash
rtk proxy python3 -B -m research.transfer.run_native_qualification --manifest <admitted-W01-manifest> --out <leased-target>/w01
```

The new CLI must refuse a missing signed/admitted manifest and absent local weights; do not execute that placeholder before implementation and admission. Fixtures test malformed pins, wrong task, missing class, target overflow, reversed quantiles, changed dependency and fake-runtime rejection. Native checkpoint tests run separately from ordinary source CI.

W01 is done only when its required native regression/binary variants have complete independent receipts. An installation, cache directory, CPU unit test or public model card is insufficient. Engineering allowance:1–2focused days; resource and rights wait depend on the actual existing-host reservation and model parity findings.

### F65-02 Build the genuine pilot execution bridge

Owner: Root in its existing leased checkout. This can be prepared while W01 admission is pending; its real launch waits for W01 and the required full panel. Scope: proposed `research/transfer/campaign_runner.py`, `campaign_cli.py`, `tests/test_campaign_runner.py`, `tests/test_campaign_cli.py`; narrow extensions to `scheduler.py`, `executor.py` and `manifests/e03-pilot.json` only when needed. Reuse `DurableSchedule`, `CampaignSchedule`, worker ABI, lineage admission, charge/bundle rules and Rust product commands; do not duplicate metric or custody truth.

Implementation sequence:

1. Define immutable `CampaignRunSpec` binding source commit/tree, protocol, roster, source/model/generator/prepared-map hashes, commands, seed schedule, endpoints, allowed outputs, resource caps and prior-attempt ledger. Hash the effective specification, not an unexpanded template.
2. Add a command-construction layer that uses existing CLI/worker APIs and validates argv. No shell-eval of manifest strings or untrusted executable callbacks. Private worker data stays inside its approved process boundary.
3. Before a stage starts, verify the manifest and rights/lease, write/fsync the scheduled attempt/debit, and check caps. After it ends, bind output file hashes, return status and observed resources. Interrupted work remains charged; stop new launches when caps are exceeded.
4. Execute prepare, compile, reload, source-free sampling, real/null/synthetic learner calls, driver analysis, Rust receipt and aggregate report. Sample with original source access disabled and full recipient closure available. A successful pipeline cannot silently use a producer's original training table at sample time.
5. Redact public summaries through the existing schema/allowlist. Keep per-query losses, source identifiers and private access traces under custodian-controlled storage; never commit them.
6. Implement `campaign_cli plan`, `campaign_cli pilot`, and a resume/inspect mode that reconciles terminal attempts without repeating model fits or protected access. Confirmation remains a separately authenticated custodian entry.

State model:

```text
planned -> admitted -> running -> complete|failed|timed_out|unsupported|over_budget|missing
```

Map these public states explicitly to scheduler.py's existing terminal vocabulary, including `completed`, `timeout`, `oom`, `license_denied`, `invalid`, `cancelled` and `unresolved`. Do not introduce an incompatible enum by copying this conceptual diagram. Count retries as child attempts under the same frozen cell identity.

Tests must cover cold restart after charging but before output, duplicate terminal attempt, wrong cached recipe, missing source/model identity, fake factory rejection, denied label access, material protocol drift, source-free sampling, redaction and total-budget exhaustion. An orchestration fixture proves orchestration only.

Engineering allowance:1–2focused days, with exact API inspection before source edits.

### F65-03 Run and account for E03

Owner: the admitted pilot executor with a fresh existing-host reservation. The smallest actual pilot has one observational task and one adverse-mechanism task, two admitted byte budgets, DOPE, Gaussian-copula and compressed-real-row reference, and each required qualified learner. `N` and any `4N` expansion must follow the admitted schedule; it is not added after seeing results. The existing E07 public Bikeshare run is previously exposed, negative, observed-unqualified evidence and cannot become fresh confirmation or substitute for this pilot.

1. Admit exact public/approved source bytes and lineage groups, disjoint fit/select/query roles, use rights and actual model panel. Prefer non-sensitive public development data when its task and mechanism satisfy the source requirements. Respect real source dependencies and temporal/entity grouping; aliases do not add independent sources.
2. Replace e03-pilot.json's null launch command only with the implemented, tested command and actual caps. Record every control/model/budget/seed cell before launch. Minimum method-budget-task combinations are2×3×2=12 before learner/seed/sample-size expansion; derive the final cell count from the actual manifest, not from this lower bound.
3. Run one bounded development resource probe within admission, preserve it in the attempt ledger, then price the complete pilot from its actual cost. Do not automatically expand to Quick12/100-table campaigns. Admiral decides each expansion within the existing budget.
4. Execute the frozen pilot schedule and all trace stages. Retain negative transfer, unqualified slices, failed required families, driver noncollapse/stability failures, privacy attack calibration/power and all missing cells.
5. Report cost by fit, sample, learner, driver/privacy/calibration and pipeline setup; distinguish CPU, GPU and wall time. Size future compute using observed cell costs and a declared contingency, without calling an estimate a measured SLO.
6. Select candidate recipes only on development data. Produce an immutable development-selection receipt and a list of exact unresolved inference/rights/resource gates for E04.

Existing checks before native launch:

```bash
rtk proxy python3 -B -m unittest research.transfer.tests.test_e03_pilot_manifest research.transfer.tests.test_pilot_admission research.transfer.tests.test_scheduler research.transfer.tests.test_executor -v
```

Proposed actual CLI after F65-02:

```bash
rtk proxy python3 -B -m research.transfer.campaign_cli plan --manifest <admitted-pilot.json> --out <leased-target>/pilot-plan
rtk proxy python3 -B -m research.transfer.campaign_cli pilot --manifest <admitted-pilot.json> --out <leased-target>/e03
```

E03 done: genuine qualified required runtimes, both task types and required controls, complete trace/terminal accounting, private/public separation, actual cost and independent acceptance. A null command, fixture-only panel or preflight cannot pass it. Do not count a pilot with unresolved required models as a scientifically qualified pilot. Runtime/elapsed estimate follows the probe.

### F65-04 Resolve statistical design and implement E04 admission

Owner: Root implements; a qualified non-author reviewer owns design acceptance and signature. Existing v1 blocked protocol and historical contracts remain immutable inputs. Proposed new files: `research/transfer/precommit_admission.py`, `research/transfer/tests/test_precommit_admission.py`, `research/transfer/protocols/confirmatory-precommit-v2.json`, a matching versioned schema if the old strict schema cannot represent the admitted state. Update owner/test maps and route Rust contract/custody changes to their actual owners.

1. Resolve D05/D19 against the canonical five task meanings and source traceability. Declare the estimand, independent unit, source/entity/seed dependencies, required family roster, all byte endpoints, candidate selection policy, original-unit losses, T90 harm orientation, D80 applicability and practical margins. Do not treat aspirational T90/D80 thresholds as ratified results.
2. Calibrate the exact final design on development/simulation data before any final outcome. The candidate requires at least10,000 bootstrap draws and at least2,000 independent Monte Carlo calibration replications; these are different axes. Existing defaults1999/399/120 do not satisfy confirmation calibration. Record type-I/coverage/power intervals, finite-variance/heavy-tail sensitivity, convergence and small-cluster abstention. Quote and admit actual compute first.
3. Enumerate every artifact/budget slot eligible for minimum/existence TDL reporting. The current ten-byte-endpoint candidate grid is512 through262,144bytes. Include all required contrasts in joint error control; `0.05/10` is only an upper bound per endpoint before additional contrasts. A single dev-selected candidate per budget does not remove across-budget selection. Fixed-artifact conjunctive IUT and minimum-TDL selection require distinct, correct handling.
4. Resolve each source/model/generator/recipe/environment/prepared-map/metric/protocol/cache/closure/roster/privacy/budget/power pin. Bind actual payload and recipient-required closure; do not claim a recipe placeholder is a frozen artifact. No nulls or template identities remain in mandatory signed fields.
5. Keep `validate_blocked_precommit` and historical refusal behavior. Add a separate admitted validator with strict schema/version, finite canonical JSON, recursively resolved local-only references, no duplicate keys/IDs, no symlink/path escape, cycle detection, complete roster, known signature algorithm and trusted external reviewer/custodian identity. Verification must use configured trust roots, not a public key supplied only inside the document being verified.
6. Require independent reviewer signature and separate custodian attestation over the canonical binding, source and allowed query/attempt limits. A string `reviewed_by`, self-signed producer key or HMAC held by the producer is not independent authority. Persist signatures before first protected access.
7. Generate `target/<campaign>/PRECOMMIT.json` plus an approved public projection for Git. Record precommit publication/signing times and source pins. Freeze a successor scientific protocol only when the validator and independent review both pass; the original candidate and its historical protocol ID remain identifiable.

Code-shaped admission rule:

```python
# Proposed behavior; use the project's real canonicalization, schemas and trust API.
def admit_precommit(document, trusted_reviewers, trusted_custodian, source_inventory):
    validate_exact_schema_and_finite_values(document)
    verify_all_resolved_content_hashes(document, source_inventory)
    verify_complete_schedule_and_joint_error_family(document)
    verify_calibration_rights_and_resource_receipts(document)
    verify_reviewer_signature(document, trusted_reviewers)
    verify_custodian_attestation(document, trusted_custodian)
    return immutable_admission_capability(document_binding=document['binding_sha256'])
```

This capability alone does not open labels. The real evaluator authenticates the authorized principal and durable budget at run time. Tests: valid synthetic independent-signature fixture; wrong key, same-principal signer, expired/foreign grant, changed source hash, omitted budget/cell, unresolved nested input, NaN, duplicate JSON keys, key-order canonicalization, cycle, altered seed, unsupported algorithm and pre-access ordering. No protected labels in source CI.

E04 done: authenticated complete signed precommit and exact-design calibration precede sealed access; not merely an editable blocked template. Engineering allowance:1–2focused days plus measured calibration and qualified independent review.

### F65-05 Execute E05 once under actual custody

Owner: the existing authorized separate-principal custodian, with an admitted executor. Root may prepare code and audit redacted results, but does not open the custodian's files or sign its own admission.

1. Verify actual source freshness/exposure, approved source/model rights and protocol, signed E04 binding, producer/evaluator UID separation, socket peer credentials and permissions. Protect final features and labels from producer caches, mounts, callbacks, archives and dashboards.
2. Use the existing Rust `evaluation::serve_frozen_batch` and release/certification protections; inspect included modules before bridging the new campaign. Owner root is0700 and owner-only, with one exclusive durable ledger namespace per confirmation ID. Freeze declaration, ordered batch, maximum attempts and query-open budget before starting.
3. Verify no inherited unauthorized descriptors/network route or producer privilege escalation. The endpoint's seccomp and SO_PEERCRED gates remain. Charge/fsync an attempted open before label access; record actual successful opens and first-access time independently.
4. Run exactly the ordered precommitted batch. Account every expected cell. Do not optimize after final access, choose a runner-up, change sample size/seed/model/budget, mask a failed contrast, or auto-retry failed scientific outcomes.
5. For an admitted infrastructure interruption, reconcile the same immutable batch and access ledger first. Resume only within already frozen attempt/query/resource limits; a damaged/partial ledger blocks reuse. Preserve failures and prior debits.
6. Publish only schema-approved aggregate projections. Keep the full private records and access audit in the admitted immutable private store. Campaign completeness and artifact/scientific/privacy/release states are separate fields.

Run synthetic custody tests before the owner launch:

```bash
rtk cargo test --locked --test vnext_custody
```

They require the authorized non-root fixture launcher and synthetic target-only files; they never establish real custody. The actual custodian command must be implemented and pinned in E04/E05's signed manifest. Do not run generic `ops/ci/gpu.sh` as a substitute for the new scientific experiment or expose final data to build CI. The scheduled GPU lane needs its own admission and may validate the same declared artifact only if its evaluator/declaration is genuinely compatible.

E05 done: all declared cells have an authentic terminal disposition, real access/attempt accounting and exact protocol/runtime evidence. Publication success is conditional on the measured scientific and privacy outcomes. No promised elapsed date before the final manifest's measured compute quote.

### F65-06 Reproduce E06 independently

Owner: an existing admitted eligible non-author executor. If Root implements the compiler or its campaign logic, Root cannot claim its own L4 run as independent. Same agent names do not prove separate principals or permissions.

Prepare versioned replay commands in proposed `research/transfer/reproduce.py` and `tests/test_reproduce.py`, and extend existing `docs/reproducibility.md`. These wrappers reuse actual metric/bundle/compiler paths. Pin the commands after E03 and freeze them with E04 where their implementation affects the evidence.

| Level | Exact work | Positive and negative evidence |
| --- | --- | --- |
| L1 | Regenerate approved aggregate tables/figures from immutable campaign public ledger | Exact output/hash comparison; a deliberately changed scratch ledger value must be detected |
| L2 | Recompute private per-query losses, paired metrics, drivers and intervals under the authorized boundary and fixed protocol | Declared numeric tolerance, complete denominator and pins; mismatched query/protocol/seed must fail |
| L3 | Run isolated source-free sampler from full recipient bundle, with its runtime/import closure and no producer dataset | Exact or predeclared-tolerance outputs and actual sample receipts; missing closure/corrupt artifact/forbidden source access must fail |
| L4 | Recompile from admitted development fit/select inputs, exact recipe, toolchain and resources | Compare artifact identity or predeclared valid numerical tolerance plus full metrics/closure; independent executor and a negative artifact/control |

Use one actual admissible positive artifact and one actual negative/control artifact at each required level. A synthetic fixture can test the replay harness but cannot replace an absent positive empirical artifact. If the pilot/confirmation produces no eligible positive case, E06 reports that exact blocker; do not relabel a failing artifact or alter the scientific selection.

Proposed commands after implementation and permissions:

```bash
rtk proxy python3 -B -m research.transfer.reproduce --level L1 --manifest <signed-replay-manifest> --out <leased-target>/e06/L1
rtk proxy python3 -B -m research.transfer.reproduce --level L2 --manifest <private-authorized-manifest> --out <private-target>/e06/L2
rtk proxy python3 -B -m research.transfer.reproduce --level L3 --manifest <signed-replay-manifest> --out <leased-target>/e06/L3
rtk proxy python3 -B -m research.transfer.reproduce --level L4 --manifest <fit-select-authorized-manifest> --out <private-target>/e06/L4
```

Each level records separate input/output hashes, command, executor, source/tree, device/environment, budget and result. An aggregate replay or existing metadata-only custody replay is not L2/L3/L4. E06 done requires all four actual independently accepted levels and the required positive/negative coverage. Preparation allowance:half to1day; actual replay time follows measured budgets.

### F65-07 Qualify runners and every CI lane

This packet can proceed independently of model/campaign work on disjoint admitted scopes. The current CPU job remains untouched. Existing runner owner leads host configuration; Root can implement repository CI fixes under a separate lease. Do not expand caps merely to make a job appear green.

1. Harvest each terminal M001 result and bind its actual logs and checkout. Preserve A1 CPU OOM and the successful different-host A2. The b95 admission receipt's failure time, host-wide/single-process claims remain unsupported; get the existing owner's immutable corrective disposition. Its current free-RAM snapshot and a proposed direct-I/O change do not qualify actual changed configuration or global host recovery.
2. Before future work, measure bounded aggregate resource use, existing leases, current jobs, disk headroom and actual VM/unit limits. The standing cap is85percent mean host CPU counting existing jobs. Memory admission must include guest RAM, QEMU overhead and page-cache/cgroup charge; an8GiB guest under roughly9GiB unit cap can still OOM. Any configuration repair is for future admitted jobs and must not interrupt live services or protected GPU work.
3. Record runner labels/architecture and actual host mapping. Require at least one appropriate exact-release validation receipt from each xbabe1/2/3 and xbabe4 ARM guest. Suggested allocation, subject to reservations: xbabe1 security, xbabe2 paper/audit, xbabe3 CPU after qualified admission, xbabe4 ARM portability. If reassigned, preserve exact actual roles and add an appropriate bounded independent check on the uncovered host; do not infer participation from online status.
4. Reconcile code, testing docs and machine budgets. Current pinned source sets CPU/paper/security/audit60minutes, ARM20minutes; scheduled GPU120minutes; one neural fit600seconds and estimated GPU memory16GiB. Root's local older docs state35/30minutes; the pinned native M001 source is current. Do not copy old quotas into the new plan.
5. Run changed-file narrow checks during implementation, `just fast` while editing and `just check` before committing. Reuse valid exact-source successes and caches without rebranding them as a different source. Required remote gates remain required. Investigate failures before bounded reruns; no unchanged science/expensive matrix replay.
6. At final PR head, require all five protected contexts and exact-head neverhumanbot approval. After merge, qualify all five on actual receiving main. Run/admit GPU backend/certification checks where release claims require them; skipped/public-fork-only jobs are accurately classified rather than called passes.

Existing local commands, only inside an admitted checkout/resource lease:

```bash
rtk just fast
rtk just check
rtk proxy bash scripts/ci-local.sh cpu
rtk proxy bash scripts/ci-local.sh security
rtk proxy bash scripts/ci-local.sh audit
rtk just paper-tests
rtk just paper-check
rtk proxy bash ops/ci/check-budgets.sh
```

Run the equivalent full lane once when it supplies new necessary evidence; do not mechanically execute every overlapping full command above back to back. `just paper-tests` and `just paper-check` are distinct required surfaces. `bash ops/ci/paper-clean.sh` mutates generated paths and runs only in the admitted owner checkout/disposable CI guest.

Native checks, using the required GH binary:

```bash
rtk proxy /home/ubuntu/.local/bin/gh pr checks <integration-PR> --repo neverhuman/dope
rtk proxy /home/ubuntu/.local/bin/gh api repos/neverhuman/dope/pulls/<integration-PR>/reviews
rtk proxy /home/ubuntu/.local/bin/gh pr merge <integration-PR> --repo neverhuman/dope --rebase --match-head-commit <exact-reviewed-head>
```

Captain verifies review eligibility and the latest non-dismissed approval is for exactly that head before merge. No `--admin`, protection weakening, self-approval, main force push or global auth switch. Admiralty's code-first ruling is not a DOPE gate exception. If newer source reaches main, rebind readiness to it with actual receiving checks and explicit scientific file-equivalence analysis.

## F65-08 Refresh publication and major release artifacts

Owner: existing paper owner or Root under a non-overlapping lease. This packet's scaffolding can start before confirmation; final numbers wait for approved E05/E06 outputs.

1. Lock the active publication entry points. Preserve current README and paper improvements; regenerate quantitative content from approved immutable ledgers and exact byte accounting. For the first-page/README hero, clearly state the task, utility/driver criterion, charged bytes and matched baseline. Use the same source/model/split/recipe/budget conditions; exclude historical unmatched/cohort comparisons from a superiority claim.
2. Show complete budget/method rosters, uncertainty with the correct simultaneous family, failed/missing dispositions and negative results. Explain data retained by compressed-row controls and distinguish a packet's full recipient closure from a core payload. No unlabeled log axes, ambiguous percentage denominators or result cherry-picking.
3. Check the hero at README width and the actual paper's first page; perform image/PDF visual QA and fresh-user comprehension checks. Verify every link to main/supplement/anonymous variants/downloads. Both legacy and active paper surfaces must be identified; do not update only one unused figure.
4. Update `README.md`, actual manuscript/captions, `docs/benchmark.md`, `docs/reproducibility.md`, `CHANGELOG.md`, `docs/release.md`, source notices, install/docs navigation and generated ledgers/hashes/PDFs through the normal build recipes. Never hand-patch generated numbers or PDFs.
5. Select the major version and compatibility policy before source/artifact freeze. Proposed public goal is v1.0.0; confirm API/wire guarantees against preserved DPK3.2/v1/DPK3.3 fixtures. Release candidates are staged and labeled honestly; no GitHub release currently exists in the baseline.
6. Build candidate binaries/bundles for the supported target matrix. Keep model weights, restricted data and private logs out of Git/release bundles unless their exact redistribution is independently admitted. Test the actual tarball/installer, notices, checksums, SBOM and source-free sample path in a fresh environment and the supported ARM guest.
7. Produce `release-manifest.json` binding source commit/tree, executable file inventory, contract/protocol ID, campaign/reproduction receipts, rights/privacy state, notices/SBOM, artifact sizes/hashes, supported targets and signing identity. Verify signature using the established trusted key; do not create a signing identity silently if none is provisioned.
8. Keep a distinction between scientific execution commit C, final documentation/version commit R and receiving main M. Before shipping, prove all evidence-relevant implementation/configuration files and staged dataset artifacts match their declared C pins. No copying C's receipt and changing its SHA toM. A material scientific change after E04 requires a successor protocol and fresh eligible confirmation data, not an automatic rerun of exposed data.

Existing commands: `rtk just paper`, `rtk just paper-tests`, `rtk just paper-check`, `rtk just security`, `rtk just score`, plus the actual supported release/install smoke from `ops/ci/native-release-smoke.sh` and its manifest. Use the correct provisioned Python and locked toolchainRust1.88.0. All new paths receive owner/test-map entries through the integrator.

## F65-09 Resolve custody and launch operations

Two historical custody incidents remain HOLD. Read only supplied exact archive/relocation/restoration paths with preservation evidence. Do not probe the absent supervisor paths again, recreate directories/locks/journals, restore from a tracked-history-only bundle or treat new files as originals. C2 executes any qualified recovery under a fresh lease. A new campaign's proper custody is separately verified and does not retroactively close either historical incident.

For release, write an explicit gate matrix for historical incident disposition, current evaluator custody, source/model rights, scientific evidence, security/SBOM, staging/signing, backup, monitoring, rollback and abuse controls. If an original artifact is irrecoverable, obtain a specific authorized incident disposition that records loss and scientific consequences; no fictitious recovered evidence. The existing pending archive question is not repeated without new facts.

Test rollback in a scratch/staging namespace against an actual signed prior passing manifest, if one exists. If no prior signed release exists, declare first-release status and verify that withdrawing the staged candidate leaves no public candidate bundle, plus restoring a verified staging baseline. Do not invent a prior signed production backup. Record every restored hash and command. No deleting non-scratch or production data under this plan.

Set monitoring to check artifact availability/integrity, release checks, campaign/access ledger health, privacy/certification status and rollback readiness. The existing30-minute monitor continues with unchanged cadence/target/notification intent. It follows this queue and asks the Admiral to resolve ordinary decisions; it stays quiet on unchanged states. Transient same-chat/model capacity failures use the already authorized bounded cool-off policy; actual quota exhaustion does not trigger retries or substitute workers.

## F65-10 Final acceptance and publication

Captain and independent reviewer assemble the final evidence; Admiral records GO/NO-GO. Root verifies native GitHub/source/artifact evidence and can repair leased code, but cannot waive its own review or independent scientific gates.

1. Compare all65 canonical task IDs against actual acceptance receipts. Hash-verify the five new task receipts and confirm no preflight-only/test-factory placeholder was counted. Validate the original60 on final source by affected-file dependency equivalence and relevant fresh checks; preserve originals and explicitly record any new proof source.
2. Verify the complete intended PR/payload queue, all live/recent closed-unmerged branches and the off-main scope-guard evidence disposition. Integrate approved unique payload; retain exact reason and locator for evidence-only material. No silent discard and no unrelated stacked PRs.
3. Verify final head review/checks, ordinary merge, receiving-main jobs and actual all-host evidence. All applicable release CI is green; relevant GPU/scientific gates pass. Historical red runs stay recorded. A stale all-green badge from an ancestor does not qualifyM.
4. Verify every staged file against the signed manifest, SBOM and notices. Download/verify via the ordinary artifact access path and run install/demo/evaluate/report/sampler smoke under the supported environment and declared access rules.
5. Independently inspect README/first-page graph, quantitative claims, paper and release navigation against final approved ledgers. Release notes specify measured limitations and the exact privacy/reproducibility/certification scope.
6. Admiral records a signed/internal decision with a complete gate matrix. Publish the signed tag and allowlisted GitHub release only under that actual GO. The source/tag/artifact identities must be auditable. No implicit protected or paid scope is added during publication.
7. After publication, verify live downloads, version/manifest/checksums and monitor state. Mark release complete only when the final handoff contains the actual release URL/tag, source commit/tree,65/65 receipts, all CI/host IDs, signatures and rollback evidence.

## Integration strategy and conflict control

Use one active integration PR for the remaining adopted source work, unless the user expressly changes that scope. Keep small ordered internal commits for F65-00, native qualification, execution bridge, precommit admission, replay, publication and release packaging. Any simultaneous owners use separate existing admitted checkouts and non-overlapping leases; captain integrates their reviewed commits without cherry-picking foreign dirty work or resetting canonical.

Parallel preparation allowed:

- Root: disjoint execution-bridge and admitted-validator preparation in its reserved existing checkout; current Codex2 retains W01 adapter/lock work. Registry updates are queued for captain.
- Existing eligible paper owner: documentation/navigation/release packaging templates, excluding runtime modules and shared ledgers; final figures wait for approved evidence.
- Existing runner owner: prospective host/runner admission repair under a separate host lease, excluding live jobs and repository source unless explicitly handed off.
- Independent reviewer/executor: read-only scientific design/replay preparation, not co-writing the producer's checkout.

Precommit admission and execution bridge cannot independently rewrite scheduler, schemas, registries or maps in parallel. Serialize those changes. Shared docs/release.md and both figure surfaces also need one owner. Source main at the release fence is refreshed before integration; resolve conflicts in the lease owner's branch and re-review any changed head. No floating-HEAD self-approval, merge race or blanket equivalence claim.

## Evidence schemas and accounting

Proposed per-packet receipt fields:

```json
{
  "schema": "dope.finish65.packet-receipt.v1",
  "task_ids": ["W01"],
  "packet_id": "F65-01",
  "status": "blocked",
  "execution": {
    "owner": null,
    "lease_id": null,
    "checkout": null,
    "commit": null,
    "tree": null,
    "command_argv": [],
    "environment_sha256": null,
    "module_files": [],
    "host": null,
    "budget": null,
    "started_at": null,
    "completed_at": null,
    "exit_status": null
  },
  "inputs": [],
  "outputs": [],
  "original_failures": [],
  "attempts_charged": 0,
  "independent_review": null,
  "acceptance_criteria": [],
  "blockers": ["NOT_EXECUTED"],
  "scientific_qualification": "unmeasured",
  "release_authorization": "not_authorized"
}
```

Nulls above describe a not-executed template. A successful receipt must require all applicable pins, finite observed resources, log/output hashes and independent acceptance; never serialize unmeasured cost as zero. Bind any approved store locator to digest, size and version/immutability metadata. Public projections omit sensitive raw/private identities; private custody receipts retain the required full pins.

Persist real logs/results in `target/<campaign>/` and the existing admitted immutable store. Commit source, tests, locks, manifests, approved compact aggregate receipts, reproducibility commands, paper inputs/outputs and release metadata to Git. Generated evidence only in ignored target or an uncommitted scratch folder cannot satisfy the final durable handoff. Large/private/third-party artifacts use admitted storage and approved signed hash pointers instead of Git blobs.

Canonical status updates use compare-and-swap/revision controls and the captain's sole writer. Report three separate measures: engineering readiness, executed65-task acceptance and major-release gate readiness. Keep historical141/126 reports and their original source scope separate from this65-task plan.

## Validation matrix and failure handling

| Area | Existing narrow checks or required proof | Failure meaning |
| --- | --- | --- |
| Kumo/TabICL | `research.transfer.tests.test_kumo_tabular`, `test_other_auditors`, plus actual official-native Q0–Q9 | Unit success alone leaves W01 unqualified |
| Pilot | `test_e03_pilot_manifest`, `test_pilot_admission`, `test_scheduler`, `test_executor`; actual admitted pilot receipt | Fixture/preflight is engineering only |
| Precommit | `test_precommit`, new admitted-validator tests, `test_tdl_result_schemas`; signed exact-design calibration | Null/unresolved/untrusted/cyclic input denies sealed access |
| Source lineage/access | `test_lineage`, `test_tdl_access_trace`, actual source/custodian admission | Synthetic fixtures do not authorize real data |
| Statistics | `test_statistics`, `test_calibration_receipt`, `test_frontier_coverage`; exact-design independent calibration | Smoke/default draws do not satisfy confirmation |
| Bundle/replay | `test_bundle`, `test_bundle_closure`, `test_bundle_isolation`, new replay tests; independent L1–L4 | Metadata or aggregate pass cannot qualify sampler/compiler replay |
| Product custody | `cargo test --locked --test vnext_custody`; actual owner/evaluator proof | A fixture pass does not close historical or production custody |
| Compatibility | Full locked Rust unit/CLI, historical DPK3.2/v1/DPK3.3 and byte-cap fixtures through required CPU lane | Wire/byte/notice regression blocks release |
| Paper | `just paper-tests`, `just paper-check`, clean-generation drift and visual QA | Passing generation does not validate scientific claims |
| Security/release | strict security, full audit>=85/caps0/nohighcritical, SBOM/notices, signed artifact/install/rollback | Missing class blocks launch |
| All-host CI | Four actual host receipts and receiving-main required contexts | Online runners/ancestor passes do not satisfy it |

For packet-aware acceptance, run from the actual admitted checkout with `DOPE_PACKET_DIR` pointing at the complete real canonical packet and bind `research.transfer.tests.test_workplan_acceptance.__file__` and its hash. Record interpreter, cwd, commit/tree and exact command. A clean-checkout fallback, inaccessible packet, skipped test or historical hardcoded PR expectation is reported as that limited scope. Do not rewrite old receipt source IDs to make historical tests appear current; add a new exact-source verifier when needed.

Stop the dependent task on missing lease, unsupported rights, unknown model identity, absent checkpoint, unsigned/unresolved protocol, inaccessible or damaged access ledger, insufficient reference/power, rare-slice failure, cap exhaustion, wrong principal or new material scientific change. Preserve partial logs and emit a typed redacted blocker with the next executable repair. Other independent admitted packets continue. No unbounded retry, selective omission, protected launch from CI or new spending.

## Schedule and checkpoints

The first checkpoint is a trustworthy queue, actual model identity/cache/environment inventory, complete next implementation packet and compute quote; target within the first focused work session. Suggested engineering allowances are F65-00 one to two hours; W01 implementation one to two days; runner/orchestration bridge one to two days; E04 implementation/design one to two days; replay/publication/release preparation one to two days with disjoint work overlapping. These are planning ranges for existing owners, not a promise that the science or rights/custody process completes in that time.

Do not give a calendar release date until W01 native parity, the E03 timing probe, final statistical calibration cost and actual host reservations are known. At E03 completion publish a measured critical-path estimate: remaining qualified cell count times observed cost, plus declared setup, independent review, calibration/reproduction and final CI. Quote GPU time separately and keep the existing one120-minute scheduled weekly GPU cap unless explicitly changed by the authorized budget owner; the final matrix may exceed it and remain blocked pending a concrete budget decision.

Every30-minute monitor checks new commits, actual owner activity, receipts, current main/PRs/checks and the next admitted action. A missed deliverable without an executable next step produces one new actionable packet to the Admiral, not a repeated ACK prompt. New finite work must have a lease, command, expected output, budget, stop condition and owner. Preserve the transient capacity cool-off rule on the same existing session/model and settings; maximum3attempts per check window, about2then5minutes, with no active duplicate.

## Final handoff

Supply the user and Admiral: release URL/tag and version; receiving main commit/tree; complete65-task acceptance table; the five new immutable receipts; retained original60 and explicit current-source equivalence/fresh-check report; all PR/head/review/merge/CI identities; four-host receipts; native model qualification and rights notices; pilot and independent signed precommit; sealed access/campaign terminal accounting; independent L1–L4 results; README/paper/figure/download links and hashes; SBOM, signed allowlisted artifact manifest and verification result; actual custody/incident disposition; monitoring and rollback proof; precise unresolved exceptions if any.

If any required gate remains, hand off the exact unfinished action and keep the release NO-GO. Do not mark this plan, a queued command, a merged code PR or65counters alone as the completed major release.
