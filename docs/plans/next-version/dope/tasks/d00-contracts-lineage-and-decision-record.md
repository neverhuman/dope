# D00 — Freeze contracts, lineage and exposure decisions

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P0**. Prerequisites: none; live ownership admission still required.

## Objective and non-goals

Produce the small decision record that removes incompatible assumptions before code changes. Name unchanged released-v3 **Lane A** and additive normalize-once **Lane B**. This is contract engineering, not permission to implement every proposal. Do not choose hyperparameters from confirmation results, register weights, retire tiers, or rewrite historical experiment records.

## Ordered read-first files

1. Current user assignment, `/Users/bentaylor/.codex/RTK.md`, repository `AGENTS.md`, current canonical `AGENT_CHAT.md` claims and open PR/worktree inventory.
2. `docs/architecture.md`, `docs/testing.md`, `docs/data-contract.md`, `docs/release.md`, then `contracts/AGENTS.md` before any contract change.
3. `agent/owner-map.json`, `agent/test-map.json`, `engine/dope-cli/src/mlc_v3.rs`, `engine/dope-core/src/prepare/mod.rs`, and `engine/dope-core/src/mlc_v3/mod.rs`.
4. Both supplied specifications; the current master plan and source audit; JopeDime's consumer pin/model/freeze manifests through read-only coordination.

## Ownership and exact write scope

One contract integrator owns **new** `docs/adr/normalize-once-and-dvs-v1.md` and **new** `docs/plans/dope-jope-contract-decisions.json`. Only that integrator may update owner/test maps when needed. Existing frozen `contracts/tdl-contract-v1.json`, released weight files, historical receipts/goldens, generated paper evidence and private datasets remain read-only. Do not create another registry if current main now supplies one; adapt this task to its actual owner and record the substitution.

## Implementation steps and hard cases

1. Record separate identities for engine source, SDK surface, encoder file, preparation recipe, landmark recipe/mask, feature layout, packet version, evaluator protocol and physical FIT scope. The `d0b69603…` standalone encoder and JopeDime `c51a56d0…` serving checkpoint identify different objects; never substitute one for another.
2. Freeze one successor normalization name: proposed `dope-norm`, NC1 recipe, private ScaleCard and public structural schema. Describe FIT-only feature statistics/vocabulary, target declaration, category groups, unknown/missing semantics, overflow/width refusal and unclipped apply-time numeric tails. Target-last is existing column-role normalization, not evidence numerical NC1 already exists.
3. State Lane A preserves exact v3 preparation and weight pairing. Any optional shim must prove equality per supported input class and refuse/fall back explicitly; an affine-invariance argument or larger epsilon is insufficient.
4. Resolve `2k/16k/64k` cap naming and retain historical `4k` compatibility pending evidence. Prefix nesting is a later hypothesis. Epsilon 0.01/0.02 and macro N=200/300 remain development-calibrated choices, with no invented universal default.
5. Define `S_select` and `S_transfer` separately. Freeze FIT/TUNE/TEST roles and exposure ledger, grouping by physical source family. Test data cannot select algorithms or stopping settings. Assign fresh experiment IDs using the live registry; H228 is occupied.
6. Define all-tied metric ineligibility, full payload/shared-decoder accounting and empirical privacy terminology. Never present byte/column row-equivalents as a memorization or de-identification guarantee. Mark unresolved choices explicitly with owner/blocking task.

Suggested decision entry: `{id, status, chosen_contract, alternatives, evidence_class, source_sha, affected_tasks, owner, unresolved}`. Separate observed facts from repository-reported measurements and proposed gates.

## Individually verifiable success criteria

- **D00-S1:** Every identity above has a distinct field and source path; unavailable provenance remains null with a blocker.
- **D00-S2:** Both lanes' allowed input/preparation/weights combinations are explicit, including unsupported/fallback behavior.
- **D00-S3:** Every supplied-spec conflict has a recorded choice or named blocking decision; no accidental dual normalization/lock implementation.
- **D00-S4:** Exposure roles and fresh-ID lookup prevent current studies or confirmation data being repurposed silently.
- **D00-S5:** Contract consumers and integrator acknowledge the exact frozen document hash before dependent writes.

## Validation, receipts and failure interpretation

Existing checks: `rtk git diff --check`; `rtk proxy python3 -m json.tool docs/plans/dope-jope-contract-decisions.json`. Proposed checker: a small schema/link/unique-decision-ID validation, only if needed to verify this actual document. Verify every referenced local path with `rtk rg --files`; label new paths explicitly. No model imports or experiments validate this task. Retain source pin, decision-file hashes, unresolved entries, owner acknowledgements and diff. Successful parsing does not resolve scientific uncertainty.

## Stop, rollback and handoff

Stop on conflicting current owner direction, unknown source/protocol lineage, occupied study ID or unresolved private/public boundary. Revert only this task's unmerged document edits; preserve claims/history. Handoff stopped full HEAD, dirty scope, decision hashes, exact commands/exits and which downstream tasks are unblocked. Documentation handoff follows current repository review/CI policy, not an inferred exception.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D00/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D00-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
