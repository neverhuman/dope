# D05 — Implement additive FIT-only normalization with private owner state

Status: **PLANNED — no implementation or qualification executed.** Repository: DOPE on xbabe2. Audited base: `59b1e0513683fa4c5c347e4db6465eff84f496ef`. Planning checkout: `/home/ubuntu/dope/.agent/worktrees/codex-value-plan-20261102`; branch: `codex/dope-jope-master-plan-20261102`. Relative paths below resolve under that checkout. Refresh upstream, claims and API differences before implementation; never change the dirty canonical `/home/ubuntu/dope` checkout. The supplied engineering specifications are proposals. This task grants no campaign, release, deployment, cloud write, credential change or foreign-job control.

Priority: **P1**. Prerequisites: D00, D03, D10.

## Objective and non-goals

Implement Lane B's normalize-once contract as one Rust crate and a bounded reference oracle. FIT statistics/vocabulary remain private owner state; consumers get structural schema and transformed matrices. Released v3 remains paired with its original preparation. No catalog rebuild, source download, default switch, weight retrain, shim assertion or public raw-unit reconstruction occurs here.

## Exact read-first order

1. `AGENTS.md`, current claims, `docs/{architecture,testing,data-contract}.md`, `contracts/AGENTS.md`, owner/test maps and D00 frozen contract.
2. `engine/dope-core/src/{schema,encode,table}.rs` for declared kinds/current transformation; `engine/dope-core/src/prepare/{scaling,scaling_fit,scaling_prepare}.rs` and `research/levels/scaling.py` as frozen compatibility references.
3. `research/transfer/target_last.py` and `research/transfer/tests/test_target_last.py`; existing target-last source/physical-row provenance semantics.
4. D03 SDK surface and JopeDime J05 intake dependency agreement; schemas/contract ownership before creating new identifiers.

## Scope, files and owners

Create `engine/dope-norm/{Cargo.toml,src/lib.rs,src/fit.rs,src/apply.rs,src/card.rs,src/public_schema.rs,tests/contract.rs}`. Proposed new schema files: `schemas/dope-nc1-private-card-v1.schema.json`, `schemas/dope-nc1-public-schema-v1.schema.json`. Integrator owns workspace/lockfiles/maps and SDK exports. Create bounded oracle/tests under `research/levels/nc1_reference.py` and `research/levels/tests/test_nc1.py` only as test/reference boundary, not a second product authority. Do not modify v3 scaling math/goldens, immutable TDL contracts, live catalog, datasets, source files, released weights or cloud objects.


Python boundary integration: before adding proposed Python source/tests, read `PYTHON_EXCEPTIONS.md`, `docs/internal/PYTHON_EXCEPTIONS.md` and `agent/boundaries.toml`. Integrator updates the authoritative exception inventory and synchronized record as current rules require; change allowed roots only if genuinely necessary. These are shared, currently owner-sensitive paths: module workers submit proposed entries instead of racing edits. No Python enters the Rust product closure.

## Ordered implementation and edge cases

1. Introduce explicit `FitView`, declared target/kinds and stable physical-row IDs. Never infer target from last source column, task/cardinality or values. Preserve target-last order and source-to-view correspondence without overwriting source bytes.
2. Implement `fit(FitView, DeclaredSchema) -> (PrivateScaleCard, PublicUnitSchema)`. Fit numeric finite min/max and imputation from FIT only; reduce in canonical physical-row order. Avoid overflow in `max-min` and division. Define constants/all-missing/non-finite refusal or deterministic mapping exactly as D00 decided.
3. Categorical vocabulary depends only on FIT feature strings and declared policy, never y. Reserve unknown handling always; missing handling follows the frozen recipe. Escape/represent literal reserved-token collisions explicitly. Stable group IDs bind one-hot columns, missing indicators and target classes. Enforce expansion/aggregate byte limits before allocations; reject excess width instead of truncating.
4. `apply(&PrivateScaleCard, Rows)` never refits. FIT finite numeric support maps to [0,1]; out-of-range query values may remain outside [0,1]. Regression query targets are never clipped. Classification vocabulary/positive-class identity cannot grow from query. Missing query categories/values follow explicit policy even when absent at FIT.
5. Private serialization carries raw names, vocabularies, extrema and inverse metadata. Public projection uses only required kinds, dimensions and structural group IDs. Public errors carry safe codes/positions, never source text/extrema. Use separate types/serializers so a private `Serialize` object cannot accidentally become a packet sidecar.
6. Bind distinct source/FIT-role/transform/schema/view/payload identities; freeze deterministic JSON/container encoding. Prefer an existing tensor carrier over inventing a second binary format. Set precision/inverse tolerances from stored f32/f64 representation; do not promise raw-f64 exact recovery from f32.
7. Add bounded reference parity and canary tests. Do not feed NC1 to released v3; optional equality shim is a separate gated task and unsupported cases retain Lane A.

## Success criteria

- **D05-S1:** TEST/TUNE extrema, labels, categories and ordering mutations cannot alter FIT card/schema bytes.
- **D05-S2:** Target declaration/roles, source physical-row identity and target-last layout survive every view conversion.
- **D05-S3:** Constants/missing/unseen/reserved tokens/finite extremes/high cardinality behave deterministically or refuse safely; no silent truncation.
- **D05-S4:** Public serialization/errors contain none of the planted private names, vocabulary or extrema.
- **D05-S5:** Rust/reference outputs agree within declared precision; input-order determinism uses canonical rows, while category relabeling tests verify semantic equivariance rather than impossible byte identity.
- **D05-S6:** Existing v3 preparation and decoder fixtures remain unchanged; SDK requires explicit successor identity.

## Validation and evidence

Proposed: `rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --manifest-path engine/Cargo.toml --locked -p dope-norm --test contract`; `rtk proxy python3 -B -m unittest research.levels.tests.test_nc1 -v`. Existing target-last and prepare parity suites, `just engine`, required repository checks and exact-head review remain. Receipt includes source/recipe/schema hashes, poisoning/canary matrix, precision bounds, actual cases and compatibility hashes. Schema parsing alone does not prove privacy or FIT-only behavior.

## Stop, rollback and handoff

Stop on unresolved contract policy, public leakage, overflow, incomplete raw lineage or manifest ownership conflict. Roll back new exports/default registration; never change sources or legacy transforms to force parity. Handoff full HEAD, versioned API/schema inventory, numeric tolerances, refusal cases and conformance pack. J05/J06 and D07 may consume only this qualified contract, not an assumed v3 shim.

## Durable completion evidence

Store this task's receipt under `/home/ubuntu/jope_plan/evidence/next-version/D05/<full-head>/`: `RECEIPT.json`, `commands.log` and `artifacts.sha256`. `RECEIPT.json` maps each `D05-S*` criterion to an actual evidence path/hash and verdict (`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`), records base/stopped HEAD, claim/owner, dirty files, command exits/executed counts, exclusions and remaining blockers. Never mark planned checks passed. Store bulky generated data outside source trees; retain only small source-linked pointers. Required repository exact-head review/CI and claim disposition remain separate fields.
