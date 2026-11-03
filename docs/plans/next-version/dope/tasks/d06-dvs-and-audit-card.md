# D06 — Implement one versioned downstream-value receipt and Audit Card

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P1**. Prerequisites: D00, D01, D02, D05, D10.

## Objective and non-goals

Create a product DVS evaluator for transfer retention R, grouped driver fidelity D, selection fidelity S, complete bytes and empirical privacy diagnostics. It consumes identity-tagged views and emits a replayable receipt/Audit Card. This is correctness engineering on synthetic fixtures; no scientific benchmark, threshold calibration, model release or legal privacy certification is authorized.

## Ordered read-first files

1. `AGENTS.md`, current claims, `docs/{architecture,testing}.md`, `contracts/AGENTS.md`, owner/test maps and D00/D01/D02 decisions.
2. `docs/metric.md`, `engine/dope-core/src/eval/{mod,frontier,audit}.rs`, `engine/dope-core/src/encode.rs`.
3. `research/transfer/tdl/{drivers,common,golden}.py`, `rust/metrics/drivers.rs`, existing v2 schemas/goldens and C02 inference records. Existing v2 requires simultaneous `full_family_max_t` bounds, positive floors and query/group/perturbation/prepared-view identity alignment.
4. D05 public/private types and D01 evaluation views; existing `engine/dope-cli/src/main.rs`, command definitions and report styling before minimal CLI registration.

## Scope and owners

Create `engine/dope-core/src/dvs/{mod.rs,drivers.rs,selection.rs,receipt.rs}`, `engine/dope-core/tests/dvs.rs`, a new governed DVS receipt schema and bounded CLI integration tests. Proposed CLI adapter `engine/dope-cli/src/dvs.rs` renders existing receipt values. Integrator registers modules/commands/maps; D02 worker owns common rank semantics. Do not add a second formula, overwrite old evaluator results, alter v1/v2 inference, change live weights/defaults, read private/sealed panels or edit generated paper claims.

## Implementation sequence and difficult semantics

1. Define explicit FIT/TUNE/TEST inputs, role/physical-family hashes, public schema/preparation identity, learner recipe/seed, complete scheduled cohort and statuses. Refuse common-coordinate/target-role mismatch; caller cannot pass TEST into learner fitting/selection. Reuse D01's receiver map.
2. Implement R relative to declared null/real reference with explicit low/zero real-gain eligibility. Keep per-task losses, actual synthetic row count and actual fit costs. Do not clamp low-gain values into passing ratios or discard failed arms.
3. Implement grouped permutation D on real-test rows using versioned current metric discipline and real-real controls. One-hot members move together. Reuse D02 successor ties while preserving old readers. If promotion claims use simultaneous inference, retain full-family multiplicity/positive-floor and aligned hashes; a naive ratio-to-ceiling is not an equivalent replacement.
4. Implement distinct `S_select`: choose learner using synthetic validation only, then assess its regret/winner on untouched real TEST. Implement `S_transfer`: compare synthetic-trained versus real-trained learner rankings evaluated on identical real TEST. Define ties, absent learner, failures and insufficient informative comparisons. Neither metric may use real TEST to choose product model or tune thresholds.
5. Account complete per-table packet, schema and mandatory sidecars; list shared decoder/weight closure separately and report owner-private card dependency separately. No hidden per-table bytes or invented hard capacity/memorization bound. Record raw-row equal-byte comparator definition, not an unrun result.
6. Add exact-copy/DCR diagnostics with explicit evaluation domain and sample sizes. They are empirical diagnostics, not formal DP or de-identification. Public receipts and cards contain aggregate permitted fields, no private canaries/raw rows.
7. Renderer uses typed receipt directly; no independent calculations or hand-entered success badges. Legacy versus successor results and measured versus NOT_RUN fields remain distinct. No CLI target-state command is documented as existing before implemented help/test coverage.

## Individually verifiable success criteria

- **D06-S1:** Analytic R, grouped D and both S fixtures give hand-derived results, including ties/zero gain/failures.
- **D06-S2:** FIT/TEST role and common-view mismatches refuse; no data selection consumes TEST outcomes.
- **D06-S3:** All scheduled failed/incomplete cells remain represented; unknown costs/results remain null.
- **D06-S4:** Existing v1/v2 metric/inference goldens remain unchanged, and successor receipts bind required identities/multiplicity policy.
- **D06-S5:** Reported bytes equal actual serialized dependency inventory; Audit Card matches receipt exactly and leaks no canaries.
- **D06-S6:** CLI success and refusal journeys work with bounded public fixtures, without Python product dependency.

## Validation and expected receipts

Proposed: `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --manifest-path engine/Cargo.toml --locked -p dope-core --test dvs`; add named DVS CLI test and run it through current `dope-cli --test cli`. Existing `rtk proxy python3 -B -m unittest research.transfer.tests.test_drivers -v`, `rtk proxy just golden-parity`, `rtk proxy just engine`, final checks/review. Retain analytic inputs/expected outputs, schemas/protocol hashes, source pin, test counts, byte inventory and rendered card. Correct math/serialization establishes no actual retention, D80, S threshold or privacy pass.

## Stop, rollback and handoff

Stop on ambiguous S meaning, weakened inference, unknown grouping, private leakage or hidden dependencies. Disable new command/registration; retain old evaluator unchanged. Handoff exact HEAD, API/CLI schema, fixture pack, inference compatibility proof, explicit unmeasured fields and D11 qualification prerequisites.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D06/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D06-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
