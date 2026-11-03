# V4 qualification readiness

**Status: HOLD — qualification-not-run.** This packet defines admission gates and evidence fields. It is not a campaign authorization, scientific result, model release, or serving approval. No training, evaluation campaign, source-cohort import, GPU work, model import, or release action has been performed for this packet.

## Objective and scope

Prepare a finite, auditable qualification campaign for a future DOPE successor. The packet is complete only when every required readiness field is supported by current evidence, every owner decision is recorded, and an authorized owner changes the admission state. A complete manifest means *ready for owner admission*; it does not mean efficacy passed.

Included: protocol skeleton, evidence/status separation, named hard gates, source identity distinctions, and an acceptance receipt. Excluded: campaign execution, cohort discovery or download, outcomes, threshold selection after exposure, CI dispatch, compute reservation, model import/release, serving changes, or JopeDime adoption.

## Fixed engineering contracts

- Normalization: `nc1-minmax-onehot-v1`. Fit statistics and vocabulary on FIT only. Preserve out-of-range numeric values; do not clip. Categorical encoding uses the frozen FIT vocabulary and explicit unknown handling.
- Owner transform: `privateownerScaleCard`, f64, unclipped, with full FIT vocabulary capped at 256 expanded features. Private transform state remains owner-side; public artifacts must not leak fitted extrema, vocabulary, or canaries.
- Wire: DML4 hard-capped packets at 2k, 16k, and 64k. No implicit 4k retirement or relabeling. Existing 4k compatibility remains until separately justified and approved.
- Historical contracts: legacy v3 preparation and v1/v2 metrics remain byte/semantic compatible. Prospective corrected metric is `dope-dvs-v1`; successor rank metric is `driver-rank-informative-v1`, preserving C02 full-family maximum-T behavior.
- Do not use a proposed successor identity to rewrite historical receipts, goldens, or claims.

## Identity and evidence boundaries

The planning snapshot lists DOPE source base `59b1e0513683fa4c5c347e4db6465eff84f496ef`. The manifest records the observed engineering checkpoint and its implementation parent chain. It deliberately leaves the actual campaign source unselected; the campaign must bind its actual full source HEAD after all prerequisite work and review. This source identity is separate from the standalone encoder SHA `d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c` and JopeDime identity `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. These identify different artifacts and must not be substituted.

The current program snapshot has 22 tasks and 104 success criteria. D11 depends on D00–D10; task count and plan status do not prove any prerequisite passed. Exact-current-head source, evaluator, metrics, NC1, wire, model/trainer graph, checkpoint identity, and their receipts are all unverified for this packet. Every missing/stale item blocks admission.

Keep these states separate in all evidence and reporting:

1. source implementation and exact stopped HEAD;
2. fixture evidence;
3. exact-head CI evidence;
4. scientific efficacy qualification;
5. release qualification;
6. serving/retrieval adoption.

Passing one state never implies a later state. In particular, source or fixture success is not efficacy, and D11 readiness is not release or JopeDime adoption.

## Required campaign design before admission

The owner must freeze a finite cohort roster with immutable physical-family IDs and roles for FIT/TRAIN, development/TUNE, independent confirmation, and any sealed/custodian set. Record historical exposure for every family, including whether architecture, defaults, epsilon, macro sample size, or thresholds were influenced by it. Unknown raw provenance blocks any claim that depends on that provenance; transformed values cannot reconstruct it. No outcome-bearing family may be used for development and confirmation.

Before outcomes are exposed, preregister hypotheses, exact matched baseline/control, scheduled cells and seeds, task strata, synthetic row policy, equal-complete-byte raw-row/current-release controls, R/D/S endpoints, harm tails, multiplicity, uncertainty method, failure denominators, and exclusion rules. Freeze parsimony epsilon, macro N, architecture, step count, thresholds, and numerical tolerances using development evidence only. Do not fill these fields with guessed numeric values. Use task/family bootstrap or another predeclared family-clustered uncertainty method; rows within a physical family are not independent replicates. Record every attempted cell, including failures, incomplete runs, actual optimizer updates, retries/repairs, and cost.

The plan must state exact checkpoint selection and termination rules, no-retry/repair policy, resume semantics, and what constitutes a fresh scientific attempt. Keep engineering profiling, model training, development selection, and untouched confirmation as distinct phases with source/checkpoint lineage. A protocol or source identity change invalidates prior exact-head evidence.

Before launch, name the accountable owner, host/device, thread, memory, disk, wall limits, admission lock, containment procedure, output custody, and recovery owner. Attach fresh capability evidence and an explicit budget/admission decision. Budget, host, deadline, expected cost, and actual cost stay null until evidenced; no unused allocation from another study transfers to this campaign. CPU fixture success does not establish CUDA availability or latency.

Recovery must be verified before admission: durable checkpoint and raw logs, per-cell identity/cost, immutable artifact manifest, and restore proof. Any later release requires separate reviewed release qualification; serving or retrieval adoption requires separate JopeDime identity, freeze, and efficacy gates.

## Readiness decision and stop conditions

The machine manifest is the source of truth for each required field, evidence reference, status, and blocker. Admission is blocked unless all engineering prerequisite receipts are current, finite-family roles and exposure are verified, the preregistration is frozen before outcomes, comparator accounting and failure policy are specified, resources and recovery are admitted, and an accountable owner signs the exact manifest hash. Missing evidence is `null` plus a named blocking condition, never `TBD` presented as certification.

Stop on stale or mismatched source identity, reused exposure or experiment ID, missing/unknown provenance where required, unresolved denominator or inference design, missing budget/owner, unavailable recovery, resource ownership conflict, or confirmation exposure during selection. Preserve the blocker and route it to the relevant engineering or study owner. Do not weaken gates to force admission.

## Acceptance and handoff

Acceptance for this readiness deliverable means the manifest is syntactically valid, all required keys and explicit blockers are present, source/artifact identities remain distinct, and the receipt records the exact candidate file hashes. It does **not** mean any D11-S criterion passed or that a campaign is ready. Future readiness requires owner review of evidence and an explicit transition to `READY_FOR_OWNER_ADMISSION`; launch still requires separate authorization and an admitted resource slot.

Current handoff: readiness packet only. Bounded synthetic CPU optimizer fixtures are engineering evidence; actual campaign training, campaign outcomes, exact-head CI, release, and adoption are `NOT_RUN`. All unverified prerequisites and campaign-design/resource fields remain blocking. Do not execute a campaign from this document.
